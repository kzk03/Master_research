#!/usr/bin/env python3
"""
============================================================
レビュー承諾予測 - Maximum Causal Entropy IRL モデルの訓練・評価
============================================================
train_model.py を MCE-IRL 用に複製した版。

逆強化学習 (IRL) として真に名乗れるよう、以下の点を train_model.py から変更
している:

  - 学習対象: 二値 action a_t ∈ {0,1} (= 月 t にレビュー応答するか) の
              trajectory log-likelihood を Boltzmann 方策の下で最大化
              (Ziebart 2010 の Maximum Causal Entropy IRL に従う)
  - 損失関数: Focal Loss → softmax cross entropy on reward logits
  - ネットワーク: 2-unit reward head (R(s,0), R(s,1)) を持つ Bi-LSTM
  - 推論: π(a=1|s) = softmax(R(s,·))[1] を「継続確率」として返す
            (BatchContinuationPredictor 互換)

train_model.py との API 互換性は CLI 引数レベルで維持し、保存ファイル名のみ
mce_irl_model.pt に変えている。直接の差し替え用途を想定している。

【データ構造】
訓練用軌跡 (extract_directory_level_trajectories or extract_mce_trajectories):
{
    'developer': {...},
    'monthly_activity_histories': [[...], ...],
    'step_labels': [0, 1, 1, 0, ...],          # MCE 学習で a_t として再解釈
    'sample_weight': 1.0 or 0.1
}
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

# ランダムシード固定
RANDOM_SEED = 777
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from review_predictor.IRL.model.mce_irl_predictor import MCEIRLSystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_review_requests(csv_path: str) -> pd.DataFrame:
    """
    レビュー依頼データを読み込む
    
    Args:
        csv_path: レビュー依頼CSVファイルのパス
        
    Returns:
        レビュー依頼データフレーム
    """
    logger.info(f"レビュー依頼データを読み込み: {csv_path}")
    df = pd.read_csv(csv_path)

    # nova_raw.csv は email/timestamp、旧データは reviewer_email/request_time
    # どちらでも動くように統一する
    if 'email' in df.columns and 'reviewer_email' not in df.columns:
        df = df.rename(columns={'email': 'reviewer_email'})
    if 'timestamp' in df.columns and 'request_time' not in df.columns:
        df = df.rename(columns={'timestamp': 'request_time'})

    df['request_time'] = pd.to_datetime(df['request_time'])
    logger.info(f"総レビュー依頼数: {len(df)}")
    logger.info(f"承諾数: {(df['label']==1).sum()} ({(df['label']==1).sum()/len(df)*100:.1f}%)")
    logger.info(f"拒否数: {(df['label']==0).sum()} ({(df['label']==0).sum()/len(df)*100:.1f}%)")
    return df


def extract_review_acceptance_trajectories(
    df: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    future_window_start_months: int = 0,
    future_window_end_months: int = 3,
    min_history_requests: int = 0,
    reviewer_col: str = 'reviewer_email',
    date_col: str = 'request_time',
    label_col: str = 'label',
    project: str = None,
    extended_label_window_months: int = 12,
    negative_oversample_factor: int = 1,
    pu_unlabeled_weight: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    レビュー承諾予測用の軌跡を抽出（データリークなし版）

    重要：訓練期間内で完結させるため、訓練期間を以下のように分割：
    - **特徴量計算期間**: train_start ～ (train_end - future_window_end_months)
    - **ラベル計算期間**: 特徴量計算期間終了後 ～ train_end

    これにより、訓練期間（train_end）を超えるデータを参照せずにラベル付けが可能。

    特徴：
    - **訓練**：特徴量計算期間内にレビュー依頼を受けた開発者のみを対象
    - **正例**：ラベル計算期間内に少なくとも1つのレビュー依頼を承諾した
    - **負例**：ラベル計算期間内にレビュー依頼を受けたが、全て拒否した
    - **負例（拡張）**：ラベル計算期間に依頼なし（この期間では活動なし）、拡張期間に依頼あり
    - **除外**：拡張期間内にもレビュー依頼を受けていない開発者（本当に離脱）

    Args:
        df: レビュー依頼データ
        train_start: 学習開始日
        train_end: 学習終了日
        future_window_start_months: 将来窓開始（月数）
        future_window_end_months: 将来窓終了（月数）
        min_history_requests: 最小履歴レビュー依頼数
        reviewer_col: レビュアー列名
        date_col: 日付列名
        label_col: ラベル列名（1=承諾, 0=拒否）
        project: プロジェクト名（指定時は単一プロジェクトのみ）
        extended_label_window_months: 拡張ラベル期間（月数、デフォルト12）

    Returns:
        各レビュアーの軌跡のリスト（1レビュアー=1サンプル）
    """
    logger.info("=" * 80)
    logger.info("レビュー承諾予測用軌跡抽出を開始（データリークなし版）")
    logger.info(f"訓練期間全体: {train_start} ～ {train_end}")
    logger.info(f"将来窓: {future_window_start_months}～{future_window_end_months}ヶ月")
    logger.info(f"拡張ラベル期間: {future_window_start_months}～{extended_label_window_months}ヶ月")
    if project:
        logger.info(f"プロジェクト: {project} (単一プロジェクト)")
    else:
        logger.info("プロジェクト: 全プロジェクト")
    logger.info("ラベル定義: この期間で承諾=1、この期間で拒否=0、依頼なし→拡張期間チェック、訓練時は拡張期間にも依頼なしで除外")
    logger.info("データリーク防止: 訓練期間内でラベル計算")
    logger.info("=" * 80)
    
    # プロジェクトフィルタを適用
    if project and 'project' in df.columns:
        df = df[df['project'] == project].copy()
        logger.info(f"プロジェクトフィルタ適用後: {len(df)}件")
    
    trajectories = []
    
    # 訓練期間全体を特徴量計算に使用（固定）
    history_start = train_start
    history_end = train_end
    
    # ラベル計算は各月末時点から将来窓を見る（月次ラベル用）
    # ここでは全体のポジティブ/ネガティブ判定用にtrain_end時点からのラベルを計算
    label_start = train_end + pd.DateOffset(months=future_window_start_months)
    label_end = train_end + pd.DateOffset(months=future_window_end_months)
    
    logger.info(f"特徴量計算期間（訓練全体）: {history_start} ～ {history_end}")
    logger.info(f"全体ラベル期間（train_end時点から）: {label_start} ～ {label_end}")
    
    # 特徴量計算期間のレビュー依頼データ
    history_df = df[
        (df[date_col] >= history_start) &
        (df[date_col] < history_end)
    ]
    
    # ラベル計算期間のレビュー依頼データ
    label_df = df[
        (df[date_col] >= label_start) &
        (df[date_col] < label_end)
    ]

    # 拡張ラベル計算期間のレビュー依頼データ
    extended_label_start = train_end + pd.DateOffset(months=future_window_start_months)
    extended_label_end = train_end + pd.DateOffset(months=extended_label_window_months)
    extended_label_df = df[
        (df[date_col] >= extended_label_start) &
        (df[date_col] < extended_label_end)
    ]

    # 特徴量計算期間内にレビュー依頼を受けたレビュアーを対象
    active_reviewers = history_df[reviewer_col].unique()
    logger.info(f"特徴量計算期間内のレビュアー数: {len(active_reviewers)}")

    skipped_min_requests = 0
    skipped_no_requests_until_end = 0  # 訓練期間末尾まで依頼がない（除外）
    positive_count = 0
    negative_count = 0
    negative_with_requests = 0  # 依頼あり→拒否
    negative_without_requests = 0  # 依頼なし（拡張期間に依頼あり）
    
    for reviewer in active_reviewers:
        # 特徴量計算期間のレビュー依頼
        reviewer_history = history_df[history_df[reviewer_col] == reviewer]
        
        # 最小レビュー依頼数を満たさない場合はスキップ
        if len(reviewer_history) < min_history_requests:
            skipped_min_requests += 1
            continue
        
        # ラベル計算期間のレビュー依頼（訓練期間内）
        reviewer_label = label_df[label_df[reviewer_col] == reviewer]

        # ラベル計算期間にレビュー依頼を受けていない場合、拡張期間をチェック
        if len(reviewer_label) == 0:
            # 拡張期間のレビュー依頼をチェック
            reviewer_extended_label = extended_label_df[extended_label_df[reviewer_col] == reviewer]

            # 訓練時は拡張期間まで見て除外判定
            if len(reviewer_extended_label) == 0:
                # 拡張期間にも依頼がない → 訓練期間末尾までアサインがない
                if pu_unlabeled_weight <= 0.0:
                    # PU学習無効: 除外（従来動作）
                    skipped_no_requests_until_end += 1
                    continue
                # PU学習有効: Unlabeledとして非常に弱い重みで負例扱い
                future_acceptance = False
                accepted_requests = pd.DataFrame()
                rejected_requests = pd.DataFrame()
                had_requests = False
                sample_weight = pu_unlabeled_weight
                skipped_no_requests_until_end += 1  # カウントは維持（参考用）
                negative_count += 1
                negative_without_requests += 1
            else:
                # 拡張期間に依頼がある → 再びアサインされる可能性 → 重み付き負例
                future_acceptance = False  # この期間では活動なし
                accepted_requests = pd.DataFrame()  # 空
                rejected_requests = pd.DataFrame()  # 空（依頼自体がない）
                had_requests = False  # この期間に依頼がなかった
                sample_weight = 0.1  # 非常に低い重み（依頼なし）

                # 統計カウント
                negative_count += 1
                negative_without_requests += 1
        else:
            # 通常のラベル期間に依頼がある場合
            # 継続判定：ラベル計算期間内に少なくとも1つのレビュー依頼を承諾したか
            accepted_requests = reviewer_label[reviewer_label[label_col] == 1]
            rejected_requests = reviewer_label[reviewer_label[label_col] == 0]
            future_acceptance = len(accepted_requests) > 0
            had_requests = True  # この期間に依頼があった
            sample_weight = 1.0  # 通常の重み（依頼あり）

            if future_acceptance:
                positive_count += 1
            else:
                negative_count += 1
                negative_with_requests += 1
        
        # 特徴量計算期間の月次ラベルを計算（訓練用、データリークなし）
        history_months = pd.date_range(
            start=history_start,
            end=history_end,
            freq='MS'  # 月初
        )
        
        step_labels = []
        monthly_activity_histories = []  # 各月時点での活動履歴
        step_context_dates = []  # 各月ステップの基準日（month_end）
        step_total_project_reviews = []  # 各月ステップ時点のプロジェクト全体レビュー依頼数

        for month_start in history_months[:-1]:  # 最後の月を除く
            month_end = month_start + pd.DateOffset(months=1)

            # この月からfuture_window後のラベル計算期間
            future_start = month_end + pd.DateOffset(months=future_window_start_months)
            future_end = month_end + pd.DateOffset(months=future_window_end_months)

            # 重要：future_endがtrain_endを超えないようにクリップ（データリーク防止）
            if future_end > train_end:
                future_end = train_end

            # train_endを超える場合はこの月のラベルは作成しない
            if future_start >= train_end:
                continue

            # 将来期間のレビュー依頼（訓練期間内のみ）
            month_future_df = df[
                (df[date_col] >= future_start) &
                (df[date_col] < future_end) &
                (df[reviewer_col] == reviewer)
            ]

            # この月のラベル：将来期間にレビュー依頼を受けて承諾したか
            if len(month_future_df) == 0:
                # レビュー依頼なし → ラベル0
                month_label = 0
            else:
                # レビュー依頼あり → 承諾の有無
                month_accepted = month_future_df[month_future_df[label_col] == 1]
                month_label = 1 if len(month_accepted) > 0 else 0

            step_labels.append(month_label)
            step_context_dates.append(month_end)  # 月末を基準日として保存

            # プロジェクト全体のレビュー依頼数（history_start から month_end まで）
            total_proj = len(df[(df[date_col] >= history_start) & (df[date_col] < month_end)])
            step_total_project_reviews.append(total_proj)

            # この月時点（month_end）までの活動履歴を保存（LSTM用）
            month_history = reviewer_history[reviewer_history[date_col] < month_end]
            monthly_activities = []
            for _, row in month_history.iterrows():
                activity = {
                    'timestamp': row[date_col],  # タイムスタンプは常にdate_col
                    'action_type': 'review',
                    'project': row.get('project', 'unknown'),
                    'project_id': row.get('project', 'unknown'),  # Multi-project: project_id
                    'request_time': row.get('request_time', row[date_col]),
                    'response_time': row.get('first_response_time'),  # response_time計算用
                    'accepted': row.get(label_col, 0) == 1,
                    'owner_email': row.get('owner_email', ''),  # 協力スコア計算用
                    'is_cross_project': row.get('is_cross_project', False),  # Multi-project: cross-project flag
                    'files_changed': row.get('change_files_count', 0),
                    'lines_added': row.get('change_insertions', 0),
                    'lines_deleted': row.get('change_deletions', 0),
                }
                monthly_activities.append(activity)
            # 自分が作成したPR（authored）の履歴も追加
            # → total_changes / reciprocity_score の計算に使用
            authored_history = df[
                (df['owner_email'] == reviewer) &
                (df[date_col] >= history_start) &
                (df[date_col] < month_end)
            ]
            for _, row in authored_history.iterrows():
                monthly_activities.append({
                    'action_type': 'authored',
                    'timestamp': row[date_col],
                    'owner_email': reviewer,           # 自分がowner
                    'reviewer_email': row.get('email', row.get('reviewer_email', '')),
                    'files_changed': row.get('change_files_count', 0),
                    'lines_added': row.get('change_insertions', 0),
                    'lines_deleted': row.get('change_deletions', 0),
                })
            monthly_activity_histories.append(monthly_activities)

        # 全期間の活動履歴も保持（評価用）
        activity_history = []
        for _, row in reviewer_history.iterrows():
            activity = {
                'timestamp': row[date_col],  # タイムスタンプは常にdate_col
                'action_type': 'review',
                'project': row.get('project', 'unknown'),
                'project_id': row.get('project', 'unknown'),  # Multi-project: project_id
                'request_time': row.get('request_time', row[date_col]),
                'response_time': row.get('first_response_time'),  # response_time計算用
                'accepted': row.get(label_col, 0) == 1,
                'owner_email': row.get('owner_email', ''),  # 協力スコア計算用
                'is_cross_project': row.get('is_cross_project', False),  # Multi-project: cross-project flag
                'files_changed': row.get('change_files_count', 0),
                'lines_added': row.get('change_insertions', 0),
                'lines_deleted': row.get('change_deletions', 0),
            }
            activity_history.append(activity)
        # 自分が作成したPR（authored）の履歴も追加（total_changes / reciprocity_score 用）
        authored_all = df[
            (df['owner_email'] == reviewer) &
            (df[date_col] >= history_start) &
            (df[date_col] < train_end)
        ]
        for _, row in authored_all.iterrows():
            activity_history.append({
                'action_type': 'authored',
                'timestamp': row[date_col],
                'owner_email': reviewer,
                'reviewer_email': row.get('email', row.get('reviewer_email', '')),
                'files_changed': row.get('change_files_count', 0),
                'lines_added': row.get('change_insertions', 0),
                'lines_deleted': row.get('change_deletions', 0),
            })
        
        # 開発者情報
        developer_info = {
            'developer_id': reviewer,  # Multi-project: use developer_id
            'developer_email': reviewer,
            'first_seen': reviewer_history[date_col].min(),
            'changes_authored': 0,
            'changes_reviewed': len(reviewer_history[reviewer_history[label_col] == 1]),
            'requests_received': len(reviewer_history),
            'requests_accepted': len(reviewer_history[reviewer_history[label_col] == 1]),
            'requests_rejected': len(reviewer_history[reviewer_history[label_col] == 0]),
            'acceptance_rate': len(reviewer_history[reviewer_history[label_col] == 1]) / len(reviewer_history),
            'projects': reviewer_history['project'].unique().tolist() if 'project' in reviewer_history.columns else []
        }
        
        # 軌跡を作成（LSTM用に月次活動履歴を追加）
        trajectory = {
            'developer_info': developer_info,
            'activity_history': activity_history,  # 全期間の活動履歴（評価用）
            'monthly_activity_histories': monthly_activity_histories,  # 各月時点の活動履歴（LSTM訓練用）
            'step_context_dates': step_context_dates,  # 各月ステップの基準日（month_end）
            'step_total_project_reviews': step_total_project_reviews,  # 各月ステップ時点のプロジェクト全体レビュー数
            'context_date': train_end,
            'step_labels': step_labels,
            'seq_len': len(step_labels),
            'reviewer': reviewer,
            'history_request_count': len(reviewer_history),
            'history_accepted_count': len(reviewer_history[reviewer_history[label_col] == 1]),
            'history_rejected_count': len(reviewer_history[reviewer_history[label_col] == 0]),
            'label_request_count': len(reviewer_label),
            'label_accepted_count': len(accepted_requests),
            'label_rejected_count': len(rejected_requests),
            'future_acceptance': future_acceptance,
            'had_requests': had_requests,  # この期間に依頼があったか
            'sample_weight': sample_weight  # サンプル重み（依頼なし=0.3、依頼あり=1.0）
        }
        
        trajectories.append(trajectory)

    # 負例が少ない場合の簡易オーバーサンプリング（訓練専用）
    if negative_oversample_factor > 1:
        negative_samples = [t for t in trajectories if not t['future_acceptance']]
        if negative_samples:
            import copy
            import random

            extra = [copy.deepcopy(random.choice(negative_samples))
                     for _ in range(len(negative_samples) * (negative_oversample_factor - 1))]
            trajectories.extend(extra)
            random.shuffle(trajectories)
            logger.info(f"負例をオーバーサンプリング: 元{len(negative_samples)}件 -> {len(negative_samples) * negative_oversample_factor}件")
    
    logger.info("=" * 80)
    logger.info(f"軌跡抽出完了: {len(trajectories)}サンプル（レビュアー）")
    logger.info(f"  スキップ（最小依頼数未満）: {skipped_min_requests}")
    logger.info(f"  スキップ（訓練期間末尾まで依頼なし）: {skipped_no_requests_until_end}")
    if trajectories:
        logger.info(f"  正例（この期間で承諾あり）: {positive_count} ({positive_count/len(trajectories)*100:.1f}%)")
        logger.info(f"  負例（この期間で承諾なし）: {negative_count} ({negative_count/len(trajectories)*100:.1f}%)")
        if negative_with_requests > 0:
            logger.info(f"    - 依頼あり→拒否（重み=1.0）: {negative_with_requests}")
        if negative_without_requests > 0:
            logger.info(f"    - 依頼なし（拡張期間に依頼あり、重み=0.1）: {negative_without_requests}")
        total_steps = sum(t['seq_len'] for t in trajectories)
        continued_steps = sum(sum(t['step_labels']) for t in trajectories)
        logger.info(f"  総ステップ数: {total_steps}")
        if total_steps > 0:
            logger.info(f"  継続ステップ率: {continued_steps/total_steps*100:.1f}% ({continued_steps}/{total_steps})")
    logger.info("=" * 80)
    
    return trajectories


def _process_one_reviewer(
    reviewer,
    df,
    history_df,
    path_extractor,
    train_start,
    train_end,
    future_window_start_months,
    future_window_end_months,
    reviewer_col,
    date_col,
    label_col,
    dirs_column,
    multitask,
):
    """1レビュアー分の軌跡を抽出する（並列処理用ワーカー関数）。"""
    history_start = train_start
    history_end = train_end

    reviewer_history = history_df[history_df[reviewer_col] == reviewer]

    reviewer_dirs = set()
    for dirs in reviewer_history[dirs_column]:
        if dirs:
            reviewer_dirs.update(d for d in dirs if d != '.')

    results = []
    positive_count = 0
    negative_count = 0
    skipped_count = 0

    MULTITASK_WINDOWS = [(0, 3), (3, 6), (6, 9), (9, 12)]

    for directory in reviewer_dirs:
        label_start = train_end + pd.DateOffset(months=future_window_start_months)
        label_end = train_end + pd.DateOffset(months=future_window_end_months)

        label_df = df[
            (df[date_col] >= label_start) &
            (df[date_col] < label_end) &
            (df[reviewer_col] == reviewer) &
            (df[dirs_column].map(lambda ds: directory in ds if ds else False))
        ]

        if len(label_df) == 0:
            future_acceptance = False
            sample_weight = 0.5
        else:
            accepted = label_df[label_df[label_col] == 1]
            future_acceptance = len(accepted) > 0
            sample_weight = 1.0

        if future_acceptance:
            positive_count += 1
        else:
            negative_count += 1

        history_months = pd.date_range(start=history_start, end=history_end, freq='MS')

        step_labels = []
        step_labels_per_window = {"0-3": [], "3-6": [], "6-9": [], "9-12": []} if multitask else None
        monthly_activity_histories = []
        step_context_dates = []
        step_total_project_reviews = []
        path_features_per_step = []

        for month_start in history_months[:-1]:
            month_end = month_start + pd.DateOffset(months=1)

            future_start = month_end + pd.DateOffset(months=future_window_start_months)
            future_end = month_end + pd.DateOffset(months=future_window_end_months)

            if future_end > train_end:
                future_end = train_end
            if future_start >= train_end:
                continue

            month_future_df = df[
                (df[date_col] >= future_start) &
                (df[date_col] < future_end) &
                (df[reviewer_col] == reviewer) &
                (df[dirs_column].map(lambda ds: directory in ds if ds else False))
            ]

            if len(month_future_df) == 0:
                month_label = 0
            else:
                month_accepted = month_future_df[month_future_df[label_col] == 1]
                month_label = 1 if len(month_accepted) > 0 else 0

            step_labels.append(month_label)

            if multitask:
                for fw_s, fw_e in MULTITASK_WINDOWS:
                    mt_future_start = month_end + pd.DateOffset(months=fw_s)
                    mt_future_end = month_end + pd.DateOffset(months=fw_e)
                    if mt_future_end > train_end:
                        mt_future_end = train_end
                    if mt_future_start >= train_end:
                        step_labels_per_window[f"{fw_s}-{fw_e}"].append(0)
                        continue
                    mt_df = df[
                        (df[date_col] >= mt_future_start) &
                        (df[date_col] < mt_future_end) &
                        (df[reviewer_col] == reviewer) &
                        (df[dirs_column].map(lambda ds: directory in ds if ds else False))
                    ]
                    if len(mt_df) == 0:
                        mt_label = 0
                    else:
                        mt_label = 1 if len(mt_df[mt_df[label_col] == 1]) > 0 else 0
                    step_labels_per_window[f"{fw_s}-{fw_e}"].append(mt_label)
            step_context_dates.append(month_end)

            total_proj = len(df[(df[date_col] >= history_start) & (df[date_col] < month_end)])
            step_total_project_reviews.append(total_proj)

            month_history = reviewer_history[reviewer_history[date_col] < month_end]
            monthly_activities = []
            for _, row in month_history.iterrows():
                monthly_activities.append({
                    'timestamp': row[date_col],
                    'action_type': 'review',
                    'project': row.get('project', 'unknown'),
                    'project_id': row.get('project', 'unknown'),
                    'request_time': row.get('request_time', row[date_col]),
                    'response_time': row.get('first_response_time'),
                    'accepted': row.get(label_col, 0) == 1,
                    'owner_email': row.get('owner_email', ''),
                    'is_cross_project': row.get('is_cross_project', False),
                    'files_changed': row.get('change_files_count', 0),
                    'lines_added': row.get('change_insertions', 0),
                    'lines_deleted': row.get('change_deletions', 0),
                })
            authored_history = df[
                (df['owner_email'] == reviewer) &
                (df[date_col] >= history_start) &
                (df[date_col] < month_end)
            ]
            for _, row in authored_history.iterrows():
                monthly_activities.append({
                    'action_type': 'authored',
                    'timestamp': row[date_col],
                    'owner_email': reviewer,
                    'reviewer_email': row.get('email', row.get('reviewer_email', '')),
                    'files_changed': row.get('change_files_count', 0),
                    'lines_added': row.get('change_insertions', 0),
                    'lines_deleted': row.get('change_deletions', 0),
                })
            monthly_activity_histories.append(monthly_activities)

            pf = path_extractor.compute(
                reviewer, frozenset({directory}), month_end.to_pydatetime()
            )
            path_features_per_step.append(pf)

        if not step_labels:
            skipped_count += 1
            continue

        developer_info = {
            'developer_id': reviewer,
            'email': reviewer,
            'first_seen': reviewer_history[date_col].min(),
            'changes_reviewed': len(reviewer_history[reviewer_history[label_col] == 1]),
            'requests_received': len(reviewer_history),
            'acceptance_rate': (
                len(reviewer_history[reviewer_history[label_col] == 1]) / len(reviewer_history)
                if len(reviewer_history) > 0 else 0.0
            ),
            'projects': (
                reviewer_history['project'].unique().tolist()
                if 'project' in reviewer_history.columns else []
            ),
        }

        trajectory = {
            'developer_info': developer_info,
            'directory': directory,
            'activity_history': [],
            'monthly_activity_histories': monthly_activity_histories,
            'step_context_dates': step_context_dates,
            'step_total_project_reviews': step_total_project_reviews,
            'path_features_per_step': path_features_per_step,
            'context_date': train_end,
            'step_labels': step_labels,
            'seq_len': len(step_labels),
            'reviewer': reviewer,
            'future_acceptance': future_acceptance,
            'sample_weight': sample_weight,
        }
        if step_labels_per_window is not None:
            trajectory['step_labels_per_window'] = step_labels_per_window
        results.append(trajectory)

    return results, positive_count, negative_count, skipped_count


def extract_directory_level_trajectories(
    df: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    path_extractor,
    future_window_start_months: int = 0,
    future_window_end_months: int = 3,
    reviewer_col: str = 'reviewer_email',
    date_col: str = 'request_time',
    label_col: str = 'label',
    dirs_column: str = 'dirs',
    multitask: bool = False,
    n_jobs: int = -1,
) -> List[Dict[str, Any]]:
    """
    ディレクトリ単位の軌跡を抽出する。[サーバ版: joblib 並列化]

    1サンプル = (開発者, ディレクトリ) ペア。
    ラベルは「その開発者がそのディレクトリに将来も貢献するか」。
    特徴量は common_features 25次元 + path_features 3次元 = 28次元。
    """
    from joblib import Parallel, delayed
    import numpy as np

    logger.info("=" * 80)
    logger.info("ディレクトリ単位の軌跡抽出を開始 [サーバ版: 並列化]")
    logger.info(f"訓練期間: {train_start} ～ {train_end}")
    logger.info(f"将来窓: {future_window_start_months}～{future_window_end_months}ヶ月")
    logger.info(f"並列数: n_jobs={n_jobs}")
    logger.info("=" * 80)

    history_start = train_start
    history_end = train_end

    history_df = df[
        (df[date_col] >= history_start) & (df[date_col] < history_end)
    ]
    active_reviewers = history_df[reviewer_col].unique()
    logger.info(f"特徴量計算期間内のレビュアー数: {len(active_reviewers)}")

    # レビュアー単位で並列処理
    batch_results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(_process_one_reviewer)(
            reviewer, df, history_df, path_extractor,
            train_start, train_end,
            future_window_start_months, future_window_end_months,
            reviewer_col, date_col, label_col, dirs_column, multitask,
        )
        for reviewer in active_reviewers
    )

    # 結果を集約
    trajectories = []
    positive_count = 0
    negative_count = 0
    skipped_count = 0
    for trajs, pos, neg, skip in batch_results:
        trajectories.extend(trajs)
        positive_count += pos
        negative_count += neg
        skipped_count += skip

    logger.info("=" * 80)
    logger.info(f"ディレクトリ単位軌跡抽出完了: {len(trajectories)} ペア")
    logger.info(f"  正例: {positive_count}, 負例: {negative_count}")
    logger.info(f"  スキップ: {skipped_count}")
    if trajectories:
        total_steps = sum(t['seq_len'] for t in trajectories)
        continued_steps = sum(sum(t['step_labels']) for t in trajectories)
        logger.info(f"  総ステップ数: {total_steps}")
        if total_steps > 0:
            logger.info(f"  継続ステップ率: {continued_steps/total_steps*100:.1f}%")
    logger.info("=" * 80)

    return trajectories


def extract_evaluation_trajectories(
    df: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    history_window_months: int = 12,
    future_window_start_months: int = 0,
    future_window_end_months: int = 3,
    min_history_requests: int = 0,
    reviewer_col: str = 'reviewer_email',
    date_col: str = 'request_time',
    label_col: str = 'label',
    project: str = None,
    extended_label_window_months: int = 12,
) -> List[Dict[str, Any]]:
    """
    評価用軌跡を抽出（スナップショット特徴量用）

    Args:
        df: レビュー依頼データ
        cutoff_date: Cutoff日（通常は訓練終了日）
        history_window_months: 履歴ウィンドウ（ヶ月）
        future_window_start_months: 将来窓の開始（ヶ月）
        future_window_end_months: 将来窓の終了（ヶ月）
        min_history_requests: 最小履歴レビュー依頼数
        reviewer_col: レビュアー列名
        date_col: 日付列名
        label_col: ラベル列名
        project: プロジェクト名（指定時は単一プロジェクトのみ）
        extended_label_window_months: 拡張ラベル期間（月数、デフォルト12）

    Returns:
        軌跡リスト
    """
    logger.info("=" * 80)
    logger.info("評価用軌跡抽出を開始（スナップショット特徴量用）")
    logger.info(f"Cutoff日: {cutoff_date}")
    logger.info(f"履歴ウィンドウ: {history_window_months}ヶ月")
    logger.info(f"将来窓: {future_window_start_months}～{future_window_end_months}ヶ月")
    logger.info(f"拡張ラベル期間: {future_window_start_months}～{extended_label_window_months}ヶ月")
    if project:
        logger.info(f"プロジェクト: {project} (単一プロジェクト)")
    else:
        logger.info("プロジェクト: 全プロジェクト")
    logger.info("継続判定: この期間で承諾=1、この期間で拒否=0、依頼なし→拡張期間チェック、評価時は拡張期間にも依頼なしで除外")
    logger.info("=" * 80)

    # プロジェクトフィルタを適用
    if project and 'project' in df.columns:
        df = df[df['project'] == project].copy()
        logger.info(f"プロジェクトフィルタ適用後: {len(df)}件")

    trajectories = []

    # 履歴期間
    history_start = cutoff_date - pd.DateOffset(months=history_window_months)
    history_end = cutoff_date

    # 評価期間
    eval_start = cutoff_date + pd.DateOffset(months=future_window_start_months)
    eval_end = cutoff_date + pd.DateOffset(months=future_window_end_months)

    logger.info(f"履歴期間: {history_start} ～ {history_end}")
    logger.info(f"評価期間: {eval_start} ～ {eval_end}")

    # 履歴期間のレビュー依頼データ
    history_df = df[
        (df[date_col] >= history_start) &
        (df[date_col] < history_end)
    ]

    # 評価期間のレビュー依頼データ
    eval_df = df[
        (df[date_col] >= eval_start) &
        (df[date_col] < eval_end)
    ]

    # 拡張評価期間のレビュー依頼データ
    extended_eval_start = cutoff_date + pd.DateOffset(months=future_window_start_months)
    extended_eval_end = cutoff_date + pd.DateOffset(months=extended_label_window_months)
    extended_eval_df = df[
        (df[date_col] >= extended_eval_start) &
        (df[date_col] < extended_eval_end)
    ]

    # 履歴期間内にレビュー依頼を受けたレビュアーを対象
    active_reviewers = history_df[reviewer_col].unique()
    logger.info(f"履歴期間内のレビュアー数: {len(active_reviewers)}")

    skipped_min_requests = 0
    skipped_no_requests_until_end = 0
    positive_count = 0
    negative_count = 0
    negative_with_requests = 0
    negative_without_requests = 0

    for reviewer in active_reviewers:
        # 履歴期間のレビュー依頼
        reviewer_history = history_df[history_df[reviewer_col] == reviewer]

        # 最小レビュー依頼数を満たさない場合はスキップ
        if len(reviewer_history) < min_history_requests:
            skipped_min_requests += 1
            continue

        # 評価期間のレビュー依頼
        reviewer_eval = eval_df[eval_df[reviewer_col] == reviewer]

        # 評価期間にレビュー依頼を受けていない場合、拡張期間をチェック
        if len(reviewer_eval) == 0:
            reviewer_extended_eval = extended_eval_df[extended_eval_df[reviewer_col] == reviewer]

            if len(reviewer_extended_eval) == 0:
                skipped_no_requests_until_end += 1
                continue

            future_acceptance = False
            accepted_requests = pd.DataFrame()
            rejected_requests = pd.DataFrame()
            had_requests = False
            sample_weight = 0.1

            negative_count += 1
            negative_without_requests += 1
        else:
            # 評価期間に依頼がある場合
            # 継続判定：評価期間内に少なくとも1つのレビュー依頼を承諾したか
            accepted_requests = reviewer_eval[reviewer_eval[label_col] == 1]
            rejected_requests = reviewer_eval[reviewer_eval[label_col] == 0]
            future_acceptance = len(accepted_requests) > 0
            had_requests = True
            sample_weight = 1.0

            if future_acceptance:
                positive_count += 1
            else:
                negative_count += 1
                negative_with_requests += 1

        # 活動履歴を構築
        activity_history = []
        for _, row in reviewer_history.iterrows():
            activity = {
                'timestamp': row[date_col],  # タイムスタンプは常にdate_col
                'action_type': 'review',
                'project': row.get('project', 'unknown'),
                'project_id': row.get('project', 'unknown'),  # Multi-project: project_id
                'request_time': row.get('request_time', row[date_col]),
                'response_time': row.get('first_response_time'),  # response_time計算用
                'accepted': row.get(label_col, 0) == 1,
                'owner_email': row.get('owner_email', ''),  # 協力スコア計算用
                'is_cross_project': row.get('is_cross_project', False),  # Multi-project: cross-project flag
                # IRL特徴量計算用のデータを追加
                'files_changed': row.get('change_files_count', 0),  # 強度計算用
                'change_files_count': row.get('change_files_count', 0),  # 強度計算用
                'lines_added': row.get('change_insertions', 0),  # 規模計算用
                'lines_deleted': row.get('change_deletions', 0),  # 規模計算用
                'change_insertions': row.get('change_insertions', 0),  # 規模計算用
                'change_deletions': row.get('change_deletions', 0),  # 規模計算用
            }
            activity_history.append(activity)
        # 自分が作成したPR（authored）の履歴も追加（total_changes / reciprocity_score 用）
        authored_all = df[
            (df['owner_email'] == reviewer) &
            (df[date_col] >= history_start) &
            (df[date_col] < cutoff_date)
        ]
        for _, row in authored_all.iterrows():
            activity_history.append({
                'action_type': 'authored',
                'timestamp': row[date_col],
                'owner_email': reviewer,
                'reviewer_email': row.get('email', row.get('reviewer_email', '')),
                'files_changed': row.get('change_files_count', 0),
                'lines_added': row.get('change_insertions', 0),
                'lines_deleted': row.get('change_deletions', 0),
            })
        
        # 開発者情報
        developer_info = {
            'developer_id': reviewer,  # Multi-project: use developer_id
            'developer_email': reviewer,
            'first_seen': reviewer_history[date_col].min(),
            'changes_authored': 0,
            'changes_reviewed': len(reviewer_history[reviewer_history[label_col] == 1]),
            'requests_received': len(reviewer_history),
            'requests_accepted': len(reviewer_history[reviewer_history[label_col] == 1]),
            'requests_rejected': len(reviewer_history[reviewer_history[label_col] == 0]),
            'acceptance_rate': len(reviewer_history[reviewer_history[label_col] == 1]) / len(reviewer_history),
            'projects': reviewer_history['project'].unique().tolist() if 'project' in reviewer_history.columns else []
        }
        
        # 月次活動履歴を構築（LSTM時系列入力用）
        history_months = pd.date_range(start=history_start, end=cutoff_date, freq='MS')
        monthly_activity_histories = []
        step_context_dates = []
        step_total_project_reviews = []
        for month_start in history_months:
            month_end = month_start + pd.DateOffset(months=1)
            if month_end > cutoff_date:
                month_end = cutoff_date
            month_history_df = reviewer_history[reviewer_history[date_col] < month_end]
            monthly_acts = []
            for _, row in month_history_df.iterrows():
                monthly_acts.append({
                    'timestamp': row[date_col],
                    'action_type': 'review',
                    'project': row.get('project', 'unknown'),
                    'response_time': row.get('first_response_time'),
                    'accepted': row.get(label_col, 0) == 1,
                    'owner_email': row.get('owner_email', ''),
                    'files_changed': row.get('change_files_count', 0),
                    'lines_added': row.get('change_insertions', 0),
                    'lines_deleted': row.get('change_deletions', 0),
                })
            # 自分が作成したPR（authored）の履歴も追加（total_changes / reciprocity_score 用）
            authored_month = df[
                (df['owner_email'] == reviewer) &
                (df[date_col] >= history_start) &
                (df[date_col] < month_end)
            ]
            for _, row in authored_month.iterrows():
                monthly_acts.append({
                    'action_type': 'authored',
                    'timestamp': row[date_col],
                    'owner_email': reviewer,
                    'reviewer_email': row.get('email', row.get('reviewer_email', '')),
                    'files_changed': row.get('change_files_count', 0),
                    'lines_added': row.get('change_insertions', 0),
                    'lines_deleted': row.get('change_deletions', 0),
                })
            monthly_activity_histories.append(monthly_acts)
            step_context_dates.append(month_end)
            total_proj = len(df[(df[date_col] >= history_start) & (df[date_col] < month_end)])
            step_total_project_reviews.append(total_proj)

        # 軌跡を作成
        trajectory = {
            'developer': developer_info,
            'activity_history': activity_history,
            'monthly_activity_histories': monthly_activity_histories,
            'step_context_dates': step_context_dates,
            'step_total_project_reviews': step_total_project_reviews,
            'context_date': cutoff_date,
            'future_acceptance': future_acceptance,
            'reviewer': reviewer,
            'history_request_count': len(reviewer_history),
            'history_accepted_count': len(reviewer_history[reviewer_history[label_col] == 1]),
            'history_rejected_count': len(reviewer_history[reviewer_history[label_col] == 0]),
            'eval_request_count': len(reviewer_eval),
            'eval_accepted_count': len(accepted_requests),
            'eval_rejected_count': len(rejected_requests),
            'had_requests': had_requests,  # この期間に依頼があったか
            'sample_weight': sample_weight  # サンプル重み（依頼なし=0.3、依頼あり=1.0）
        }
        
        trajectories.append(trajectory)
    
    logger.info("=" * 80)
    logger.info(f"評価用軌跡抽出完了: {len(trajectories)}サンプル")
    logger.info(f"  スキップ（最小依頼数未満）: {skipped_min_requests}")
    logger.info(f"  スキップ（拡張期間にも依頼なし）: {skipped_no_requests_until_end}")
    if trajectories:
        logger.info(f"  正例（この期間で承諾あり）: {positive_count} ({positive_count/len(trajectories)*100:.1f}%)")
        logger.info(f"  負例（この期間で承諾なし）: {negative_count} ({negative_count/len(trajectories)*100:.1f}%)")
        if negative_with_requests > 0:
            logger.info(f"    - 依頼あり→拒否（重み=1.0）: {negative_with_requests}")
        if negative_without_requests > 0:
            logger.info(f"    - 依頼なし（拡張期間に依頼あり、重み=0.1）: {negative_without_requests}")
    logger.info("=" * 80)

    return trajectories


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "f1",
    recall_floor: float = 0.8
) -> Dict[str, float]:
    """
    最適な閾値を探索

    Args:
        y_true: 真のラベル
        y_pred: 予測確率
        metric: 閾値選択指標
            - "f1": F1最大化（従来動作）
            - "precision_at_recall_floor": recallがrecall_floor以上の点でPrecision最大
            - "youden": Youden J (TPR - FPR) 最大化
        recall_floor: metricがprecision_at_recall_floorの場合の下限リコール

    Returns:
        閾値と主要指標
    """
    metric = metric.lower()

    # Precision-Recall based candidates
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

    if metric == "f1":
        best_idx = np.argmax(f1_scores)
    elif metric == "precision_at_recall_floor":
        valid_idx = np.where(recall >= recall_floor)[0]
        if len(valid_idx) == 0:
            best_idx = np.argmax(f1_scores)
        else:
            best_idx = valid_idx[np.argmax(precision[valid_idx])]
    elif metric == "youden":
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred)
        j = tpr - fpr
        j_best = np.argmax(j)
        best_threshold = roc_thresholds[j_best] if j_best < len(roc_thresholds) else 0.5
        return {
            'threshold': float(best_threshold),
            'precision': float(precision[np.argmax(f1_scores)]),
            'recall': float(recall[np.argmax(f1_scores)]),
            'f1': float(f1_scores[np.argmax(f1_scores)]),
            'metric': 'youden'
        }
    else:
        # フォールバックは従来のF1最大化
        best_idx = np.argmax(f1_scores)

    best_threshold = pr_thresholds[best_idx] if best_idx < len(pr_thresholds) else 0.5

    return {
        'threshold': float(best_threshold),
        'precision': float(precision[best_idx]),
        'recall': float(recall[best_idx]),
        'f1': float(f1_scores[best_idx]),
        'metric': metric
    }


def main():
    parser = argparse.ArgumentParser(
        description="レビュー承諾予測 - Maximum Causal Entropy IRL 学習"
    )
    parser.add_argument(
        "--reviews",
        type=str,
        default="data/review_requests_openstack_multi_5y_detail.csv",
        help="レビュー依頼CSVファイルのパス"
    )
    parser.add_argument(
        "--train-start",
        type=str,
        default="2021-01-01",
        help="訓練開始日 (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--train-end",
        type=str,
        default="2023-01-01",
        help="訓練終了日 (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--future-window-start",
        type=int,
        default=0,
        help="将来窓開始（月数）"
    )
    parser.add_argument(
        "--future-window-end",
        type=int,
        default=3,
        help="将来窓終了（月数）"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="訓練エポック数（early stoppingあり）"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping の patience（何エポック改善なしで停止）"
    )
    parser.add_argument(
        "--min-history-events",
        type=int,
        default=0,
        help="最小履歴イベント数"
    )
    parser.add_argument(
        "--negative-oversample-factor",
        type=int,
        default=1,
        help="負例オーバーサンプリング係数（>1で訓練時に負例を複製）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/review_acceptance_irl",
        help="出力ディレクトリ"
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="プロジェクト名（単一プロジェクトのみ）"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="既存モデルのパス（評価のみの場合）"
    )
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        help=(
            "warm-start: 既存 checkpoint (例: Focal-supervised の irl_model.pt) から"
            " 重みを初期化して MCE-IRL NLL で fine-tune する。state_dim が一致する場合のみ"
            " load_state_dict を実行 (不一致時は ValueError)。"
        ),
    )
    parser.add_argument(
        "--init-lr-scale",
        type=float,
        default=1.0,
        help="warm-start 時に学習率を scale する倍率 (デフォルト 1.0、推奨 0.1)。"
             " --init-from 指定時のみ使用。",
    )
    parser.add_argument(
        "--directory-level",
        action="store_true",
        help="ディレクトリ単位で学習する（ラベル・特徴量をディレクトリ対応に変更）"
    )
    parser.add_argument(
        "--raw-json",
        type=str,
        nargs="+",
        default=["data/raw_json/openstack__nova.json"],
        help="ディレクトリマッピング用の raw JSON パス（複数指定可）"
    )
    parser.add_argument(
        "--model-type",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4, 5],
        help="モデルバリアント (0:LSTM, 1:LSTM+Attn, 2:Transformer, 3:LSTM+MT, 4:LSTM+Attn+MT, 5:Trans+MT)"
    )
    parser.add_argument(
        "--trajectories-cache",
        type=str,
        default=None,
        help="軌跡キャッシュファイルのパス（.pkl）。存在すればロード、なければ抽出後に保存"
    )
    parser.add_argument(
        "--skip-threshold",
        action="store_true",
        help="訓練データ上の F1 最適閾値計算をスキップ。"
             " 閾値は 46928 軌跡の逐次推論で 1 時間以上かかるため、AUC/Spearman ベース"
             " 評価しか使わない場合は不要。"
    )
    args = parser.parse_args()
    
    # 出力ディレクトリを作成
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # レビュー依頼データを読み込み
    df = load_review_requests(args.reviews)
    
    # 日付をパース
    train_start = pd.Timestamp(args.train_start)
    train_end = pd.Timestamp(args.train_end)
    logger.info(f"将来窓: {args.future_window_start}～{args.future_window_end}ヶ月")
    
    # 訓練用軌跡を抽出（キャッシュ対応）
    import pickle
    cache_path = args.trajectories_cache

    if args.model is None:
        if cache_path and Path(cache_path).exists():
            logger.info(f"軌跡キャッシュを読み込み: {cache_path}")
            with open(cache_path, 'rb') as f:
                train_trajectories = pickle.load(f)
            logger.info(f"キャッシュから {len(train_trajectories)} 軌跡を読み込みました")

            # 月次 MCE-IRL 用キャッシュ schema 検証
            #   - Focal-supervised の outputs/trajectory_cache/traj_*.pkl には
            #     step_actions が無いため、誤って指定されないよう拒否する。
            #   - イベント単位キャッシュ (event_features 付き) も schema 不一致なので拒否する
            #     (本スクリプトは月次集約 state_dim=23 のため)。
            if not train_trajectories:
                raise ValueError(
                    f"軌跡キャッシュが空です: {cache_path}\n"
                    f"extract_mce_trajectories.py で再生成してください。"
                )
            sample = train_trajectories[0]
            if "step_actions" not in sample:
                raise ValueError(
                    f"軌跡キャッシュ {cache_path} は MCE-IRL 用ではありません。"
                    f" step_actions キーが欠損しています。\n"
                    f"  （Focal-supervised の outputs/trajectory_cache/traj_*.pkl"
                    f"  を誤指定していないか確認してください）\n"
                    f"  → extract_mce_trajectories.py で生成された "
                    f"  outputs/mce_irl_trajectory_cache/mce_traj_*.pkl を指定してください。"
                )
            ef = sample.get("event_features")
            if ef:
                raise ValueError(
                    f"軌跡キャッシュ {cache_path} はイベント単位 (event_features 有り) です。"
                    f" 月次 MCE-IRL では受け付けられません。\n"
                    f"  → イベント単位で学習する場合は train_mce_event_irl.py を使用してください。"
                )
            # state_dim をキャッシュ内容から推定して args との不整合を検出する
            # (path_features_per_step が空ならグローバル IRL = state_dim 20、
            #  存在すれば directory-level = state_dim 23)
            has_path_features = bool(sample.get("path_features_per_step"))
            inferred_state_dim = 23 if has_path_features else 20
            expected_state_dim = 23 if args.directory_level else 20
            if inferred_state_dim != expected_state_dim:
                raise ValueError(
                    f"キャッシュ {cache_path} は state_dim={inferred_state_dim} 用ですが、"
                    f" args.directory_level={args.directory_level} の指定だと"
                    f" state_dim={expected_state_dim} を要求します。"
                    f" --directory-level の指定とキャッシュの種類を一致させてください。"
                )
            state_dim = inferred_state_dim
        elif args.directory_level:
            # ディレクトリ単位モード
            from review_predictor.IRL.features.path_features import (
                PathFeatureExtractor,
                attach_dirs_to_df,
                load_change_dir_map,
                load_change_dir_map_multi,
            )
            logger.info("ディレクトリ単位モード: ディレクトリマッピングを構築...")
            if len(args.raw_json) == 1:
                cdm = load_change_dir_map(args.raw_json[0], depth=2)
            else:
                cdm = load_change_dir_map_multi(args.raw_json, depth=2)
            df = attach_dirs_to_df(df, cdm, column='dirs')
            # PathFeatureExtractor は 'email', 'timestamp' 列を期待するので
            # train_model.py のリネーム後の列名を戻したコピーを渡す
            df_for_path = df.rename(columns={
                'reviewer_email': 'email',
                'request_time': 'timestamp',
            })
            path_extractor = PathFeatureExtractor(df_for_path, window_days=180)

            from review_predictor.IRL.model.network_variants import is_multitask
            _multitask = is_multitask(args.model_type)
            logger.info(f"ディレクトリ単位の軌跡を抽出... (multitask={_multitask})")
            train_trajectories = extract_directory_level_trajectories(
                df,
                train_start=train_start,
                train_end=train_end,
                path_extractor=path_extractor,
                future_window_start_months=args.future_window_start,
                future_window_end_months=args.future_window_end,
                multitask=_multitask,
                n_jobs=-1,
            )
            # step_actions を付与: extract_mce_trajectories.py と同じ規約。
            # これがないと再実行時に cache 検証 (step_actions 欠損チェック) で失敗する。
            for traj in train_trajectories:
                traj["step_actions"] = [
                    int(bool(l)) for l in traj.get("step_labels", [])
                ]
            state_dim = 23  # 20 + path(3)

            # キャッシュ保存
            if cache_path:
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, 'wb') as f:
                    pickle.dump(train_trajectories, f)
                logger.info(f"軌跡キャッシュを保存: {cache_path}")
        else:
            logger.info("訓練用軌跡を抽出...")
            train_trajectories = extract_review_acceptance_trajectories(
                df,
                train_start=train_start,
                train_end=train_end,
                future_window_start_months=args.future_window_start,
                future_window_end_months=args.future_window_end,
                min_history_requests=args.min_history_events,
                project=args.project,
                negative_oversample_factor=args.negative_oversample_factor
            )
            state_dim = 20  # v2: STATE_FEATURES(20) + ACTION_FEATURES(5)

        if not train_trajectories:
            logger.error("訓練用軌跡が抽出できませんでした")
            return

        # warm-start を使う場合は LR を scale する
        base_lr = 3e-4
        effective_lr = base_lr * (args.init_lr_scale if args.init_from else 1.0)
        config = {
            'state_dim': state_dim,
            'action_dim': 5,
            'hidden_dim': 128,
            'sequence': True,
            'seq_len': 0,
            'learning_rate': effective_lr,
            'dropout': 0.2,
            'model_type': args.model_type,
        }
        irl_system = MCEIRLSystem(config)

        # warm-start: 既存 checkpoint から重みを load
        if args.init_from:
            init_path = Path(args.init_from)
            if not init_path.exists():
                raise FileNotFoundError(f"--init-from が指定する checkpoint が見つかりません: {init_path}")
            logger.info(f"warm-start: {init_path} から重みを load")
            init_sd = torch.load(init_path, map_location='cpu', weights_only=True)
            init_state_w = init_sd.get('state_encoder.0.weight')
            if init_state_w is not None and init_state_w.shape[1] != state_dim:
                raise ValueError(
                    f"warm-start checkpoint の state_dim={init_state_w.shape[1]} は"
                    f" 学習側 state_dim={state_dim} と一致しません。"
                    f" 互換性のある checkpoint (同じ --directory-level 設定で学習されたもの)"
                    f" を指定してください。"
                )
            # 旧実装 (self.lstm/self.lstm_norm 直接保持) で保存された Focal baseline checkpoint を
            # 現実装 (self.backbone = LSTMBackbone(...)) のキー名に正規化してから転移する。
            renamed_sd = {}
            for k, v in init_sd.items():
                if k.startswith('lstm.'):
                    renamed_sd['backbone.' + k] = v
                elif k == 'lstm_norm.weight':
                    renamed_sd['backbone.norm.weight'] = v
                elif k == 'lstm_norm.bias':
                    renamed_sd['backbone.norm.bias'] = v
                else:
                    renamed_sd[k] = v
            # shape 一致するパラメータだけ抽出 (Focal 出力 [1,64] vs MCE 出力 [2,64] のような
            # ヘッド差異を許容し、本体 (state_encoder / LSTM / attention) のみ転移する)
            cur_sd = irl_system.network.state_dict()
            compat_sd = {}
            skipped: list[tuple[str, tuple, tuple]] = []
            for k, v in renamed_sd.items():
                if k in cur_sd and cur_sd[k].shape == v.shape:
                    compat_sd[k] = v
                else:
                    skipped.append((k, tuple(v.shape), tuple(cur_sd[k].shape) if k in cur_sd else ()))
            missing, unexpected = irl_system.network.load_state_dict(compat_sd, strict=False)
            logger.info(
                f"warm-start: 互換 {len(compat_sd)}/{len(renamed_sd)} 個のパラメータを load"
            )
            if skipped:
                for k, src, dst in skipped[:5]:
                    logger.info(f"  skip {k}: ckpt{src} ↔ model{dst}")
                if len(skipped) > 5:
                    logger.info(f"  ... 他 {len(skipped) - 5} 個")
            if missing:
                logger.info(f"  missing (random init): {len(missing)} 個")
            logger.info(
                f"warm-start 完了: LR scale={args.init_lr_scale} → 実効 LR={effective_lr:.2e}"
            )

        # 訓練データの正例率（参考情報のみ。MCE-IRL では Focal Loss は使わない）
        positive_count = sum(1 for t in train_trajectories if t['future_acceptance'])
        positive_rate = positive_count / len(train_trajectories)
        logger.info(
            f"訓練データ正例率: {positive_rate:.1%} "
            f"({positive_count}/{len(train_trajectories)})"
        )

        # MCE-IRL 訓練 (trajectory log-likelihood を最大化)
        logger.info("MCE-IRL モデルを訓練...")
        irl_system.train_mce_irl_temporal_trajectories(
            train_trajectories,
            epochs=args.epochs,
            patience=args.patience,
        )

        # 学習完了後、まず重みとメタデータを保存する。閾値計算は重く、
        # 途中で kill されてもここまでで Phase 3 が走れるようにするため。
        model_path = output_dir / "mce_irl_model.pt"
        torch.save(irl_system.network.state_dict(), model_path)
        logger.info(f"モデルを保存 (閾値計算前): {model_path}")

        metadata = {
            'model_class': 'mce_irl',
            'model_type': args.model_type,
            'state_dim': state_dim,
            'action_dim': 5,
            'hidden_dim': 128,
            'dropout': 0.2,
            'num_actions': 2,
            'loss': 'softmax_cross_entropy_trajectory_nll',
            'warm_start_from': str(args.init_from) if args.init_from else None,
            'init_lr_scale': args.init_lr_scale if args.init_from else None,
            'effective_lr': effective_lr,
        }
        metadata_path = output_dir / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"メタデータを保存: {metadata_path}")

        if args.skip_threshold:
            logger.info("--skip-threshold 指定のため、訓練データ上の閾値計算をスキップします。")
            return

        # 訓練データ上で最適閾値を決定
        logger.info("訓練データ上で最適閾値を決定...")
        train_y_true = []
        train_y_pred = []

        for traj in train_trajectories:
            developer = traj.get('developer', traj.get('developer_info', {}))
            email = developer.get('email', developer.get('developer_id', ''))

            if args.directory_level and traj.get('monthly_activity_histories'):
                # ディレクトリ単位: monthly prediction を使う（activity_history が空のため）
                result = irl_system.predict_continuation_probability_monthly(
                    developer,
                    traj['monthly_activity_histories'],
                    traj.get('step_context_dates', []),
                    context_date=traj.get('context_date'),
                    step_total_project_reviews=traj.get('step_total_project_reviews'),
                    step_path_features=traj.get('path_features_per_step'),
                )
                prob = result.get('continuation_probability', 0.5)
            else:
                result = irl_system.predict_continuation_probability_snapshot(
                    developer,
                    traj['activity_history'],
                    traj['context_date']
                )
                prob = result['continuation_probability']
            train_y_true.append(1 if traj['future_acceptance'] else 0)
            train_y_pred.append(prob)
        
        train_y_true = np.array(train_y_true)
        train_y_pred = np.array(train_y_pred)

        # F1スコアを最大化する閾値を訓練データで決定
        positive_rate = train_y_true.mean()
        logger.info(f"訓練データ正例率: {positive_rate:.1%}")

        # find_optimal_threshold を使用してF1スコアを最大化する閾値を探索
        train_optimal_threshold_info = find_optimal_threshold(train_y_true, train_y_pred)
        train_optimal_threshold = train_optimal_threshold_info['threshold']
        train_optimal_threshold_info['positive_rate'] = float(positive_rate)
        train_optimal_threshold_info['method'] = 'f1_maximization_on_train_data'

        logger.info(f"F1最大化閾値（訓練データ）: {train_optimal_threshold:.4f}")
        logger.info(f"訓練データ性能: Precision={train_optimal_threshold_info['precision']:.3f}, Recall={train_optimal_threshold_info['recall']:.3f}, F1={train_optimal_threshold_info['f1']:.3f}")
        
        # 訓練データの予測確率分布も保存（デバッグ用）
        train_optimal_threshold_info['train_prediction_stats'] = {
            'min': float(train_y_pred.min()),
            'max': float(train_y_pred.max()),
            'mean': float(train_y_pred.mean()),
            'std': float(train_y_pred.std()),
            'median': float(np.median(train_y_pred))
        }
        logger.info(f"訓練データ予測確率: [{train_optimal_threshold_info['train_prediction_stats']['min']:.4f}, {train_optimal_threshold_info['train_prediction_stats']['max']:.4f}]")

        # 閾値を保存 (モデル本体とメタデータは閾値計算前に保存済み)
        threshold_path = output_dir / "optimal_threshold.json"
        with open(threshold_path, 'w') as f:
            json.dump(train_optimal_threshold_info, f, indent=2)
        logger.info(f"最適閾値を保存: {threshold_path}")
    else:
        # 既存 MCE-IRL モデルを読み込み
        logger.info(f"既存 MCE-IRL モデルを読み込み: {args.model}")
        # state_dim は state_dict から自動判定する
        state_dict = torch.load(args.model, map_location='cpu', weights_only=True)
        state_encoder_w = state_dict.get('state_encoder.0.weight')
        loaded_state_dim = state_encoder_w.shape[1] if state_encoder_w is not None else 23
        config = {
            'state_dim': loaded_state_dim,
            'action_dim': 5,
            'hidden_dim': 128,
            'sequence': True,
            'seq_len': 0,
            'dropout': 0.2,
            'model_type': args.model_type,
        }
        irl_system = MCEIRLSystem(config)
        irl_system.network.load_state_dict(state_dict)
        irl_system.network.eval()
        
        # 保存された閾値を読み込み
        threshold_path = Path(args.model).parent / "optimal_threshold.json"
        if threshold_path.exists():
            with open(threshold_path) as f:
                train_optimal_threshold_info = json.load(f)
                train_optimal_threshold = train_optimal_threshold_info['threshold']
            logger.info(f"保存された閾値を読み込み: {train_optimal_threshold:.4f}")
        else:
            train_optimal_threshold = 0.5
            logger.warning(f"閾値ファイルが見つからないため、デフォルト値 {train_optimal_threshold:.4f} を使用")
    
    # ディレクトリ単位モードでは後続の評価は eval_path_prediction.py で行う
    if args.directory_level:
        logger.info("ディレクトリ単位モード: 学習完了。評価は eval_path_prediction.py で実施してください。")
        return

    # 評価用軌跡を抽出
    logger.info("評価用軌跡を抽出...")
    history_window_months = int((train_end - train_start).days / 30)
    
    # future_window_start_monthsとfuture_window_end_monthsを使用
    # これらは--future-window-startと--future-window-endから来る
    eval_trajectories = extract_evaluation_trajectories(
        df,
        cutoff_date=train_end,
        history_window_months=history_window_months,
        future_window_start_months=args.future_window_start,
        future_window_end_months=args.future_window_end,
        min_history_requests=args.min_history_events,
        project=args.project
    )
    
    if not eval_trajectories:
        logger.error("評価用軌跡が抽出できませんでした")
        return
    
    # 予測
    logger.info("予測を実行...")
    y_true = []
    y_pred = []
    predictions = []
    
    for traj in eval_trajectories:
        # スナップショット特徴量で予測
        result = irl_system.predict_continuation_probability_snapshot(
            traj['developer'],
            traj['activity_history'],
            traj['context_date']
        )
        prob = result['continuation_probability']
        true_label = 1 if traj['future_acceptance'] else 0
        
        y_true.append(true_label)
        y_pred.append(prob)
        
        predictions.append({
            'reviewer_email': traj['reviewer'],
            'predicted_prob': float(prob),
            'true_label': true_label,
            'history_request_count': traj['history_request_count'],
            'history_acceptance_rate': traj['developer']['acceptance_rate'],
            'eval_request_count': traj['eval_request_count'],
            'eval_accepted_count': traj['eval_accepted_count'],
            'eval_rejected_count': traj['eval_rejected_count']
        })
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # メトリクスを計算
    logger.info("メトリクスを計算...")

    # 訓練データで決定した閾値を使用（データリーク防止）
    optimal_threshold = train_optimal_threshold
    logger.info(f"訓練データで決定した閾値を使用: {optimal_threshold:.4f}")

    # 参考：評価データ上での最適閾値も計算（比較用）
    eval_optimal_threshold_info = find_optimal_threshold(y_true, y_pred)
    logger.info(f"参考：評価データ上での最適閾値: {eval_optimal_threshold_info['threshold']:.4f} (F1={eval_optimal_threshold_info['f1']:.3f})")

    y_pred_binary = (y_pred >= optimal_threshold).astype(int)
    
    auc_roc = roc_auc_score(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    auc_pr = auc(recall, precision)
    
    precision_at_threshold = precision_score(y_true, y_pred_binary)
    recall_at_threshold = recall_score(y_true, y_pred_binary)
    f1_at_threshold = f1_score(y_true, y_pred_binary)
    
    metrics = {
        'auc_roc': float(auc_roc),
        'auc_pr': float(auc_pr),
        'optimal_threshold': float(optimal_threshold),
        'threshold_source': 'train_data',  # 訓練データで決定した閾値を使用
        'precision': float(precision_at_threshold),
        'recall': float(recall_at_threshold),
        'f1_score': float(f1_at_threshold),
        'positive_count': int(y_true.sum()),
        'negative_count': int((1 - y_true).sum()),
        'total_count': int(len(y_true)),
        # 参考情報：評価データでの最適閾値
        'eval_optimal_threshold': float(eval_optimal_threshold_info['threshold']),
        'eval_optimal_f1': float(eval_optimal_threshold_info['f1']),
        # 予測確率の分布統計
        'prediction_stats': {
            'min': float(y_pred.min()),
            'max': float(y_pred.max()),
            'mean': float(y_pred.mean()),
            'std': float(y_pred.std()),
            'median': float(np.median(y_pred))
        }
    }
    
    logger.info("=" * 80)
    logger.info("評価結果:")
    logger.info(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    logger.info(f"  AUC-PR: {metrics['auc_pr']:.4f}")
    logger.info(f"  最適閾値（訓練データ決定）: {metrics['optimal_threshold']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")
    logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
    logger.info(f"  正例数: {metrics['positive_count']}")
    logger.info(f"  負例数: {metrics['negative_count']}")
    logger.info("---")
    logger.info("予測確率の分布:")
    logger.info(f"  範囲: [{metrics['prediction_stats']['min']:.4f}, {metrics['prediction_stats']['max']:.4f}]")
    logger.info(f"  平均: {metrics['prediction_stats']['mean']:.4f}")
    logger.info(f"  標準偏差: {metrics['prediction_stats']['std']:.4f}")
    logger.info(f"  中央値: {metrics['prediction_stats']['median']:.4f}")
    logger.info("=" * 80)
    
    # 結果を保存
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    predictions_df = pd.DataFrame(predictions)
    predictions_df['predicted_binary'] = y_pred_binary
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    logger.info(f"結果を保存: {output_dir}")


if __name__ == "__main__":
    main()

