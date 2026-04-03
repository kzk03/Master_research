#!/usr/bin/env python3
"""
複数プロジェクト対応のクロス時間評価スクリプト

卒論と同じ設計:
- IRL訓練: train_history_start ～ train_history_end（固定、デフォルト2021-01-01～2022-01-01）
  - 特徴量はこの全期間から計算
  - ラベルは train_history_end 時点から将来窓（train_X-Xm）で定義
- 評価: eval_cutoff（固定、デフォルト2023-01-01）時点でスナップショット
  - 特徴量は eval_cutoff から eval_history_months 分遡って計算
  - ラベルは eval_cutoff 時点から将来窓（eval_X-Xm）で定義

全16パターン（制約なし、4×4）:
- train_0-3m  → eval_0-3m, 3-6m, 6-9m, 9-12m
- train_3-6m  → eval_0-3m, 3-6m, 6-9m, 9-12m
- train_6-9m  → eval_0-3m, 3-6m, 6-9m, 9-12m
- train_9-12m → eval_0-3m, 3-6m, 6-9m, 9-12m

意味:
  train_X-Xm: IRLの報酬関数学習に使うラベルの将来窓（train_history_end 時点からX～Xヶ月後）
  eval_X-Xm:  予測評価時の将来窓（eval_cutoff 時点からX～Xヶ月後）
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

# パス設定
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))
ANALYSIS = ROOT / "scripts" / "analysis"
if str(ANALYSIS) not in sys.path:
    sys.path.append(str(ANALYSIS))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_date_offset(base_date: pd.Timestamp, months: int) -> pd.Timestamp:
    """日付にN月を加算"""
    return base_date + pd.DateOffset(months=months)


def generate_evaluation_patterns(
    total_months: int = 12
) -> List[Dict]:
    """
    評価パターンを生成（train_fw_start <= eval_fw_start の制約で10パターン）

    train_X-Xm: IRLラベル将来窓のオフセット（train_history_end 基点）
    eval_X-Xm:  評価ラベル将来窓のオフセット（eval_cutoff 基点）

    Args:
        total_months: 総期間（月数、デフォルト12ヶ月）

    Returns:
        評価パターンのリスト（将来窓オフセット月数を含む）
    """
    patterns = []

    # 3ヶ月間隔の将来窓を定義
    windows = []
    for i in range(0, total_months, 3):
        windows.append({
            'name': f'{i}-{i+3}m',
            'fw_start': i,
            'fw_end': i + 3,
        })

    # train_fw_start <= eval_fw_start の制約: 10パターン
    for train_w in windows:
        for eval_w in windows:
            if train_w['fw_start'] <= eval_w['fw_start']:
                patterns.append({
                    'train_name': train_w['name'],
                    'eval_name': eval_w['name'],
                    'train_fw_start': train_w['fw_start'],
                    'train_fw_end': train_w['fw_end'],
                    'eval_fw_start': eval_w['fw_start'],
                    'eval_fw_end': eval_w['fw_end'],
                })

    logger.info(f"生成されたパターン数: {len(patterns)}")
    for p in patterns:
        logger.info(f"  {p['train_name']} → {p['eval_name']}")

    return patterns


def _extract_monthly_rf_train_features(
    df_rf: pd.DataFrame,
    train_history_start: pd.Timestamp,
    train_history_end: pd.Timestamp,
    future_window_start_months: int,
    future_window_end_months: int,
    project_type: str = "nova",
) -> pd.DataFrame:
    """月次スナップショットを積み上げたRF訓練データを生成（卒論設計準拠）。

    IRL訓練と同様に、各月末を基点として特徴量とラベルを計算し、
    全月分を結合してRFの訓練データとする。これによりRFの訓練ラベルが
    train_history_end以前の期間に散在し、評価ラベルと分離される。

    Args:
        df_rf: RF用DataFrame (email/timestamp/label列を持つ)
        train_history_start: 特徴量計算の開始日（固定）
        train_history_end: 訓練期間の終端（この月まで月次で繰り返す）
        future_window_start_months: 将来窓の開始オフセット（ヶ月）
        future_window_end_months: 将来窓の終了オフセット（ヶ月）
        project_type: プロジェクト種別（未使用、API互換のため保持）

    Returns:
        全月分を結合したDataFrame（feature列 + label + email）
    """
    monthly_dfs = []
    month_starts = pd.date_range(start=train_history_start, end=train_history_end, freq='MS')

    for month_start in month_starts[:-1]:  # 最終月を除く（将来窓がtrain_history_endを超えないように）
        month_end = month_start + pd.DateOffset(months=1)
        future_start = month_end + pd.DateOffset(months=future_window_start_months)
        future_end = month_end + pd.DateOffset(months=future_window_end_months)

        # 将来窓がtrain_history_endを超えないようにクリップ（データリーク防止）
        if future_start >= train_history_end:
            continue
        if future_end > train_history_end:
            future_end = train_history_end

        from review_predictor.IRL.model.rf_predictor import extract_features_for_window
        month_df = extract_features_for_window(
            df_rf,
            train_history_start,  # 特徴量は常に先頭から累積
            month_end,
            future_start,
            future_end,
            project_type,
        )
        if len(month_df) > 0:
            monthly_dfs.append(month_df)

    if not monthly_dfs:
        return pd.DataFrame()
    return pd.concat(monthly_dfs, ignore_index=True)


def _prepare_rf_dataframe(df_src: pd.DataFrame, project_name: str = None) -> pd.DataFrame:
    """RF用にtimestamp/email/プロジェクトを整える"""
    df_rf = df_src.copy()

    if project_name and 'project' in df_rf.columns:
        df_rf = df_rf[df_rf['project'] == project_name].copy()

    if 'timestamp' not in df_rf.columns:
        if 'created' in df_rf.columns:
            df_rf['timestamp'] = pd.to_datetime(df_rf['created'])
        elif 'request_time' in df_rf.columns:
            df_rf['timestamp'] = pd.to_datetime(df_rf['request_time'])
        elif 'context_date' in df_rf.columns:
            df_rf['timestamp'] = pd.to_datetime(df_rf['context_date'])
        else:
            raise ValueError("timestamp列が見つかりません (created/request_time/context_date いずれもなし)")

    if 'email' not in df_rf.columns:
        if 'developer_email' in df_rf.columns:
            df_rf['email'] = df_rf['developer_email']
        elif 'reviewer_email' in df_rf.columns:
            df_rf['email'] = df_rf['reviewer_email']
        elif 'owner_email' in df_rf.columns:
            df_rf['email'] = df_rf['owner_email']
        else:
            raise ValueError("email列が見つかりません (developer_email/reviewer_email/owner_email いずれもなし)")

    return df_rf


def train_and_evaluate_pattern(
    pattern: Dict,
    reviews_csv: str,
    output_base: Path,
    train_history_start: pd.Timestamp,
    train_history_end: pd.Timestamp,
    eval_cutoff: pd.Timestamp,
    eval_history_months: int = 24,
    project: str = None,
    epochs: int = 20,
    min_history: int = 0,
    learning_rate: float = 0.0001,
    threshold_metric: str = "f1",
    recall_floor: float = 0.8,
    focal_alpha: float = None,
    focal_gamma: float = None,
    negative_oversample_factor: int = 1,
    run_rf_baseline: bool = False,
    hidden_dim: int = 128,
    pu_unlabeled_weight: float = 0.0,
) -> Dict:
    """
    1つのパターンで訓練・評価を実行

    Args:
        pattern: 評価パターン（train_fw_start/end, eval_fw_start/end を含む）
        reviews_csv: レビュー依頼CSVファイル
        output_base: 出力ベースディレクトリ
        train_history_start: IRL訓練の特徴量計算開始日（固定）
        train_history_end: IRL訓練の特徴量計算終了日（固定）= ラベル計算の基点
        eval_cutoff: 評価スナップショット日（固定）= ラベル計算の基点
        eval_history_months: 評価用履歴ウィンドウ（ヶ月）
        project: プロジェクト名（Noneの場合は全プロジェクト）
        epochs: 訓練エポック数
        min_history: 最小履歴イベント数

    Returns:
        評価結果
    """
    train_name = pattern['train_name']
    eval_name = pattern['eval_name']

    # 出力ディレクトリ
    train_dir = output_base / f"train_{train_name}"
    eval_dir = train_dir / f"eval_{eval_name}"
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info(f"パターン: {train_name} → {eval_name}")
    logger.info(f"IRL訓練ラベル将来窓: {pattern['train_fw_start']}～{pattern['train_fw_end']}ヶ月")
    logger.info(f"評価ラベル将来窓: {pattern['eval_fw_start']}～{pattern['eval_fw_end']}ヶ月")
    logger.info("=" * 80)

    # train_model.pyから関数をインポート
    # パスを追加
    import sys
    from pathlib import Path as PathLib
    script_dir = PathLib(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    import numpy as np
    import torch
    from review_predictor.IRL.model.rf_predictor import (
        extract_features_for_window,
        prepare_rf_features,
        train_and_evaluate_rf,
    )
    from sklearn.metrics import (
        auc,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from train_model import (
        extract_evaluation_trajectories,
        extract_review_acceptance_trajectories,
        find_optimal_threshold,
        load_review_requests,
    )

    from review_predictor.IRL.model.irl_predictor_v2 import RetentionIRLSystem
    from review_predictor.IRL.features.common_features import STATE_FEATURES, ACTION_FEATURES

    # データ読み込み
    df = load_review_requests(reviews_csv)

    logger.info(f"IRL訓練履歴期間: {train_history_start} ～ {train_history_end}")
    logger.info(f"IRL訓練ラベル将来窓: {pattern['train_fw_start']}～{pattern['train_fw_end']}ヶ月 (train_history_end 基点)")
    logger.info(f"評価スナップショット日: {eval_cutoff}")
    logger.info(f"評価履歴ウィンドウ: {eval_history_months}ヶ月")
    logger.info(f"評価ラベル将来窓: {pattern['eval_fw_start']}～{pattern['eval_fw_end']}ヶ月 (eval_cutoff 基点)")

    # 訓練用軌跡を抽出（固定の履歴期間、パターンに応じた将来窓）
    logger.info("訓練用軌跡を抽出...")
    train_trajectories = extract_review_acceptance_trajectories(
        df,
        train_start=train_history_start,
        train_end=train_history_end,
        future_window_start_months=pattern['train_fw_start'],
        future_window_end_months=pattern['train_fw_end'],
        min_history_requests=min_history,
        project=project,
        negative_oversample_factor=negative_oversample_factor,
        pu_unlabeled_weight=pu_unlabeled_weight,
    )

    if not train_trajectories:
        logger.error("訓練用軌跡が抽出できませんでした")
        return None

    # IRLシステムを初期化（マルチプロジェクト対応）
    # data/multiproject_paper_data.csv を使用する場合は14次元
    config = {
        'state_dim': len(STATE_FEATURES),  # common_featuresから動的取得
        'action_dim': len(ACTION_FEATURES),
        'hidden_dim': hidden_dim,
        'sequence': True,
        'seq_len': 0,
        'learning_rate': learning_rate,
        'dropout': 0.2,
    }
    irl_system = RetentionIRLSystem(config)

    # 訓練データの正例率を計算してFocal Lossを調整
    positive_count = sum(1 for t in train_trajectories if t['future_acceptance'])
    positive_rate = positive_count / len(train_trajectories)
    logger.info(f"訓練データ正例率: {positive_rate:.1%} ({positive_count}/{len(train_trajectories)})")
    if focal_alpha is not None and focal_gamma is not None:
        irl_system.set_focal_loss_params(focal_alpha, focal_gamma)
        logger.info(f"Focal Loss 手動設定: alpha={focal_alpha:.3f}, gamma={focal_gamma:.3f}")
    else:
        irl_system.auto_tune_focal_loss(positive_rate)

    # 訓練
    logger.info("IRLモデルを訓練...")
    irl_system.train_irl_temporal_trajectories(
        train_trajectories,
        epochs=epochs
    )

    # 訓練データ上で最適閾値を決定
    logger.info("訓練データ上で最適閾値を決定...")
    train_y_true = []
    train_y_pred = []

    for traj in train_trajectories:
        developer = traj.get('developer', traj.get('developer_info', {}))
        # 月次シーケンスで予測（卒論設計準拠）
        result = irl_system.predict_continuation_probability_monthly(
            developer,
            traj.get('monthly_activity_histories', []),
            traj.get('step_context_dates', []),
            traj.get('context_date'),
            step_total_project_reviews=traj.get('step_total_project_reviews'),
        )
        train_y_true.append(1 if traj['future_acceptance'] else 0)
        train_y_pred.append(result['continuation_probability'])

    train_y_true = np.array(train_y_true)
    train_y_pred = np.array(train_y_pred)

    # F1スコアを最大化する閾値を探索
    train_optimal_threshold_info = find_optimal_threshold(
        train_y_true,
        train_y_pred,
        metric=threshold_metric,
        recall_floor=recall_floor
    )
    train_optimal_threshold = train_optimal_threshold_info['threshold']

    logger.info(f"最適閾値: {train_optimal_threshold:.4f}")
    logger.info(f"訓練データ性能: Precision={train_optimal_threshold_info['precision']:.3f}, "
                f"Recall={train_optimal_threshold_info['recall']:.3f}, "
                f"F1={train_optimal_threshold_info['f1']:.3f}")

    # モデルと閾値を保存
    model_path = train_dir / "irl_model.pt"
    torch.save(irl_system.network.state_dict(), model_path)
    logger.info(f"モデルを保存: {model_path}")

    threshold_path = train_dir / "optimal_threshold.json"
    with open(threshold_path, 'w') as f:
        json.dump(train_optimal_threshold_info, f, indent=2)
    logger.info(f"最適閾値を保存: {threshold_path}")

    # 評価用軌跡を抽出（固定のcutoff日、eval_history_months分の履歴、パターンに応じた将来窓）
    logger.info("評価用軌跡を抽出...")
    eval_trajectories = extract_evaluation_trajectories(
        df,
        cutoff_date=eval_cutoff,
        history_window_months=eval_history_months,
        future_window_start_months=pattern['eval_fw_start'],
        future_window_end_months=pattern['eval_fw_end'],
        min_history_requests=min_history,
        project=project,
    )

    if not eval_trajectories:
        logger.error("評価用軌跡が抽出できませんでした")
        return None

    # 予測
    logger.info("予測を実行...")
    y_true = []
    y_pred = []
    predictions = []

    for traj in eval_trajectories:
        # 月次シーケンスで予測（卒論設計準拠）
        result = irl_system.predict_continuation_probability_monthly(
            traj['developer'],
            traj.get('monthly_activity_histories', []),
            traj.get('step_context_dates', []),
            traj.get('context_date'),
            step_total_project_reviews=traj.get('step_total_project_reviews'),
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

    # メトリクスを計算（訓練データで決定した閾値を使用）
    y_pred_binary = (y_pred >= train_optimal_threshold).astype(int)

    auc_roc = roc_auc_score(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    auc_pr = auc(recall, precision)

    precision_at_threshold = precision_score(y_true, y_pred_binary)
    recall_at_threshold = recall_score(y_true, y_pred_binary)
    f1_at_threshold = f1_score(y_true, y_pred_binary)

    metrics = {
        'auc_roc': float(auc_roc),
        'auc_pr': float(auc_pr),
        'optimal_threshold': float(train_optimal_threshold),
        'threshold_source': 'train_data',
        'precision': float(precision_at_threshold),
        'recall': float(recall_at_threshold),
        'f1_score': float(f1_at_threshold),
        'positive_count': int(y_true.sum()),
        'negative_count': int((1 - y_true).sum()),
        'total_count': int(len(y_true))
    }

    logger.info("=" * 80)
    logger.info(f"評価結果 ({train_name} → {eval_name}):")
    logger.info(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    logger.info(f"  AUC-PR: {metrics['auc_pr']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")
    logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
    logger.info("=" * 80)

    # 結果を保存
    with open(eval_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    predictions_df = pd.DataFrame(predictions)
    predictions_df['predicted_binary'] = y_pred_binary

    # IRL gradient-based feature importance
    try:
        irl_importance = irl_system.compute_gradient_importance(eval_trajectories)
        if irl_importance:
            with open(eval_dir / "irl_feature_importance.json", "w") as f:
                json.dump(irl_importance, f, indent=2)
            logger.info(f"IRL特徴量重要度を保存: {eval_dir / 'irl_feature_importance.json'}")
    except Exception as e:
        logger.warning(f"IRL特徴量重要度の計算で例外: {e}")

    rf_metrics = None
    if run_rf_baseline:
        logger.info("RFベースラインを評価...")
        try:
            df_rf = _prepare_rf_dataframe(df, project)
            project_type = 'nova' if project else 'multi'

            # RF訓練: 月次スナップショットを積み上げ（IRL訓練と同設計）
            train_features_df = _extract_monthly_rf_train_features(
                df_rf,
                train_history_start,
                train_history_end,
                pattern['train_fw_start'],
                pattern['train_fw_end'],
                project_type,
            )
            # RF評価: eval_cutoff を基点、eval_history_months 分遡って特徴量計算
            eval_history_start = eval_cutoff - pd.DateOffset(months=eval_history_months)
            eval_fw_end_date = eval_cutoff + pd.DateOffset(months=pattern['eval_fw_end'])
            eval_features_df = extract_features_for_window(
                df_rf,
                eval_history_start,
                eval_cutoff,
                eval_cutoff,
                eval_fw_end_date,
                project_type
            )

            if len(train_features_df) < 2 or len(eval_features_df) < 2:
                logger.warning("RFベースライン: サンプル数が不足のためスキップ")
            else:
                X_train, y_train = prepare_rf_features(train_features_df, project_type)
                X_eval, y_eval = prepare_rf_features(eval_features_df, project_type)
                rf_metrics = train_and_evaluate_rf(X_train, y_train, X_eval, y_eval)

                if rf_metrics:
                    rf_predictions = rf_metrics.pop('predictions', None)
                    if rf_predictions is not None:
                        # RF予測を email→prob のマッピングとして構築
                        rf_threshold = rf_metrics['optimal_threshold']
                        rf_pred_map = dict(zip(
                            eval_features_df['email'].values,
                            rf_predictions,
                        ))
                        # IRL の predictions_df に left join で RF 列を追加
                        predictions_df['rf_predicted_prob'] = predictions_df['reviewer_email'].map(rf_pred_map)
                        predictions_df['rf_predicted_binary'] = (
                            predictions_df['rf_predicted_prob'] >= rf_threshold
                        ).astype('Int64')  # nullable int for NaN rows
                        matched = predictions_df['rf_predicted_prob'].notna().sum()
                        total = len(predictions_df)
                        logger.info(f"RF予測を predictions_df に結合: {matched}/{total} レビュアー一致")

                    rf_metrics['pattern'] = f"{train_name} → {eval_name}"
                    rf_metrics_path = eval_dir / "rf_metrics.json"
                    with open(rf_metrics_path, "w") as f:
                        json.dump(rf_metrics, f, indent=2)
                    logger.info(f"RFベースライン結果を保存: {rf_metrics_path}")
        except Exception as e:
            logger.exception(f"RFベースライン評価で例外: {e}")

    # predictions.csv を保存（RF列を含む場合あり）
    predictions_df.to_csv(eval_dir / "predictions.csv", index=False)
    logger.info(f"結果を保存: {eval_dir}")

    return {'irl': metrics, 'rf': rf_metrics}


def _build_matrices_from_json(
    output_base: Path,
    patterns: List[Dict],
    json_filename: str,
    csv_prefix: str,
) -> None:
    """metrics JSON からマトリクス CSV を生成する共通ロジック。

    Args:
        output_base: 出力ベースディレクトリ
        patterns: 評価パターンリスト
        json_filename: 読み取る JSON ファイル名 (e.g. "metrics.json", "rf_metrics.json")
        csv_prefix: 出力 CSV プレフィックス (e.g. "matrix", "rf_matrix")
    """
    period_names = sorted(list(set([p['train_name'] for p in patterns])))

    metrics_names = ['AUC_ROC', 'AUC_PR', 'F1', 'PRECISION', 'RECALL']
    key_map = {
        'AUC_ROC': 'auc_roc',
        'AUC_PR': 'auc_pr',
        'F1': 'f1_score',
        'PRECISION': 'precision',
        'RECALL': 'recall',
    }
    matrices = {
        metric: pd.DataFrame(index=period_names, columns=period_names, dtype=float)
        for metric in metrics_names
    }

    found = 0
    for pattern in patterns:
        train_name = pattern['train_name']
        eval_name = pattern['eval_name']

        metrics_file = output_base / f"train_{train_name}" / f"eval_{eval_name}" / json_filename

        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
            for matrix_key, json_key in key_map.items():
                matrices[matrix_key].loc[train_name, eval_name] = metrics.get(json_key, None)
            found += 1

    if found == 0:
        logger.warning(f"{json_filename} が見つかりませんでした。マトリクスをスキップします。")
        return

    for metric_name, matrix in matrices.items():
        output_file = output_base / f"{csv_prefix}_{metric_name}.csv"
        matrix.to_csv(output_file)
        logger.info(f"保存: {output_file}")
        logger.info(f"\n{matrix}")


def create_matrices(output_base: Path, patterns: List[Dict]):
    """IRL メトリクスマトリクスを作成"""
    logger.info("=" * 80)
    logger.info("IRL メトリクスマトリクスを作成中...")
    logger.info("=" * 80)

    _build_matrices_from_json(output_base, patterns, "metrics.json", "matrix")

    logger.info("=" * 80)
    logger.info("IRL マトリクス作成完了")
    logger.info("=" * 80)


def create_rf_matrices(output_base: Path, patterns: List[Dict]):
    """RF メトリクスマトリクスを作成"""
    logger.info("=" * 80)
    logger.info("RF メトリクスマトリクスを作成中...")
    logger.info("=" * 80)

    _build_matrices_from_json(output_base, patterns, "rf_metrics.json", "rf_matrix")

    logger.info("=" * 80)
    logger.info("RF マトリクス作成完了")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="複数プロジェクト対応のクロス時間評価"
    )
    parser.add_argument(
        "--reviews",
        type=str,
        required=True,
        help="レビュー依頼CSVファイルのパス"
    )
    parser.add_argument(
        "--train-history-start",
        type=str,
        default="2021-01-01",
        help="IRL訓練の特徴量計算開始日 (YYYY-MM-DD、デフォルト: 2021-01-01)"
    )
    parser.add_argument(
        "--train-history-end",
        type=str,
        default="2023-01-01",
        help="IRL訓練の特徴量計算終了日＝ラベル基点 (YYYY-MM-DD、デフォルト: 2023-01-01)"
    )
    parser.add_argument(
        "--eval-cutoff",
        type=str,
        default="2023-01-01",
        help="評価スナップショット日＝ラベル基点 (YYYY-MM-DD、デフォルト: 2023-01-01)"
    )
    parser.add_argument(
        "--eval-history-months",
        type=int,
        default=24,
        help="評価用履歴ウィンドウ（ヶ月、デフォルト: 24）"
    )
    parser.add_argument(
        "--total-months",
        type=int,
        default=12,
        help="将来窓の総期間（月数、デフォルト12ヶ月）"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="出力ディレクトリ"
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="プロジェクト名（指定しない場合は全プロジェクト）"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="訓練エポック数"
    )
    parser.add_argument(
        "--threshold-metric",
        type=str,
        default="f1",
        choices=["f1", "precision_at_recall_floor", "youden"],
        help="閾値探索指標 (f1/precision_at_recall_floor/youden)"
    )
    parser.add_argument(
        "--recall-floor",
        type=float,
        default=0.8,
        help="precision_at_recall_floorで要求する最低Recall"
    )
    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=0.65,
        help="Focal Loss alpha (デフォルト: 0.65)"
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=1.5,
        help="Focal Loss gamma (デフォルト: 1.5)"
    )
    parser.add_argument(
        "--min-history-events",
        type=int,
        default=0,
        help="最小履歴イベント数"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="学習率 (デフォルト: 1e-3)"
    )
    parser.add_argument(
        "--negative-oversample-factor",
        type=int,
        default=2,
        help="負例オーバーサンプリング係数（>1で訓練時に負例を複製）"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="LSTMの隠れ層次元数（デフォルト: 128）"
    )
    parser.add_argument(
        "--run-rf",
        action="store_true",
        help="RFベースラインも同じパターンで評価する"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="乱数シード（安定化用、未指定時はランダム）"
    )
    parser.add_argument(
        "--pu-weight",
        type=float,
        default=0.0,
        help="PU学習: 除外Unlabeledサンプルに付与するソフト重み（0.0=無効、推奨: 0.02）"
    )

    args = parser.parse_args()

    # 乱数シード固定（安定化）
    if args.seed is not None:
        import random
        import numpy as np
        import torch as _torch
        random.seed(args.seed)
        np.random.seed(args.seed)
        _torch.manual_seed(args.seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(args.seed)
        logger.info(f"乱数シードを固定: {args.seed}")

    # 出力ディレクトリを作成
    output_base = Path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)

    # 日付をパース
    train_history_start = pd.Timestamp(args.train_history_start)
    train_history_end = pd.Timestamp(args.train_history_end)
    eval_cutoff = pd.Timestamp(args.eval_cutoff)

    logger.info(f"IRL訓練履歴期間: {train_history_start} ～ {train_history_end}")
    logger.info(f"評価スナップショット日: {eval_cutoff}")
    logger.info(f"評価履歴ウィンドウ: {args.eval_history_months}ヶ月")

    # 評価パターンを生成（10パターン: train<=eval の上三角）
    patterns = generate_evaluation_patterns(args.total_months)

    # 各パターンで訓練・評価
    logger.info("=" * 80)
    logger.info(f"全{len(patterns)}パターンの訓練・評価を開始")
    logger.info("=" * 80)

    for i, pattern in enumerate(patterns, 1):
        logger.info(f"\n【{i}/{len(patterns)}】パターン実行中...")

        metrics = train_and_evaluate_pattern(
            pattern,
            args.reviews,
            output_base,
            train_history_start=train_history_start,
            train_history_end=train_history_end,
            eval_cutoff=eval_cutoff,
            eval_history_months=args.eval_history_months,
            project=args.project,
            epochs=args.epochs,
            min_history=args.min_history_events,
            learning_rate=args.learning_rate,
            threshold_metric=args.threshold_metric,
            recall_floor=args.recall_floor,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            negative_oversample_factor=args.negative_oversample_factor,
            run_rf_baseline=args.run_rf,
            hidden_dim=args.hidden_dim,
            pu_unlabeled_weight=args.pu_weight,
        )

        if metrics is None:
            logger.warning(f"パターン {pattern['train_name']} → {pattern['eval_name']} をスキップ")

    # メトリクスマトリクスを作成
    create_matrices(output_base, patterns)

    if args.run_rf:
        create_rf_matrices(output_base, patterns)

    logger.info("=" * 80)
    logger.info("全パターンの訓練・評価が完了しました")
    logger.info(f"結果: {output_base}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
