"""
MCE-IRL モデルによるバッチ continuation_prob 推論

batch_predictor.py の MCE-IRL 版。学習済みの MCEIRLSystem (mce_irl_model.pt)
をロードして、月次シーケンスから π(a=1|s) を継続確率として返す。
評価スクリプト eval_mce_irl_path_prediction.py から利用する。

月次シーケンス構築ロジックは train_model.py / batch_predictor.py と同一。
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


class MCEBatchContinuationPredictor:
    """
    学習済み MCE-IRL モデルを使い、開発者の continuation_prob (= π(a=1|s)) を
    バッチ推論するクラス。

    使い方:
        predictor = MCEBatchContinuationPredictor(
            model_path="outputs/.../mce_irl_model.pt",
            df=df,
            history_start=datetime(2019, 1, 1),
        )
        probs = predictor.predict_batch(
            emails=["alice@example.com", "bob@example.com"],
            prediction_time=datetime(2022, 1, 1),
        )
        # -> {"alice@example.com": 0.82, "bob@example.com": 0.34}
    """

    def __init__(
        self,
        model_path: str | Path,
        df: pd.DataFrame,
        history_start: datetime,
        device: str = "cpu",
        reviewer_col: str = "email",
        date_col: str = "timestamp",
        label_col: str = "label",
    ) -> None:
        self.model_path = Path(model_path)
        self.df = df.copy()
        self.history_start = history_start
        self.device = torch.device(device)
        self.reviewer_col = reviewer_col
        self.date_col = date_col
        self.label_col = label_col

        # timestamp を datetime 型に揃える
        if not pd.api.types.is_datetime64_any_dtype(self.df[self.date_col]):
            self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])

        # IRL システムを遅延ロード
        self._irl_system: Optional[Any] = None

    def _load_model(self) -> None:
        """MCE-IRL モデルをロード（初回のみ）。"""
        if self._irl_system is not None:
            return

        import json
        from review_predictor.IRL.model.mce_irl_predictor import MCEIRLSystem
        from review_predictor.IRL.features.common_features import (
            ACTION_FEATURES,
        )

        # モデルメタデータ (model_class, model_type など) を読み込む
        metadata_path = self.model_path.parent / "model_metadata.json"
        model_type = 0
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            model_type = metadata.get("model_type", 0)
            model_class = metadata.get("model_class", "")
            if model_class and model_class != "mce_irl":
                logger.warning(
                    "model_metadata.json の model_class=%s は MCE-IRL ではありません。"
                    " このローダは mce_irl 用です。", model_class
                )

        # state_dim を保存された重みから自動判定
        state_dict = torch.load(
            self.model_path, map_location=self.device, weights_only=True
        )
        state_encoder_weight = state_dict.get("state_encoder.0.weight")
        if state_encoder_weight is not None:
            state_dim = state_encoder_weight.shape[1]
        else:
            state_dim = 20  # デフォルト

        config = {
            "state_dim": state_dim,
            "action_dim": len(ACTION_FEATURES),
            "hidden_dim": 128,
            "dropout": 0.1,
            "model_type": model_type,
        }
        irl_system = MCEIRLSystem(config)
        irl_system.device = self.device
        irl_system.network = irl_system.network.to(self.device)

        irl_system.network.load_state_dict(state_dict)
        irl_system.network.eval()

        self._irl_system = irl_system
        logger.info(
            "MCE-IRL モデルをロード: %s (model_type=%d, state_dim=%d)",
            self.model_path,
            model_type,
            state_dim,
        )

    def _build_monthly_data(
        self,
        email: str,
        prediction_time: datetime,
    ) -> tuple:
        """
        train_model.py:312-398 と同一のロジックで月次データを構築する。

        Returns:
            (monthly_activity_histories, step_context_dates, step_total_project_reviews,
             developer_info)
        """
        df = self.df
        history_start = pd.Timestamp(self.history_start)
        pred_time = pd.Timestamp(prediction_time)

        # 特徴量計算期間のレビュー依頼データ（このレビュアーの履歴）
        reviewer_history = df[
            (df[self.reviewer_col] == email)
            & (df[self.date_col] >= history_start)
            & (df[self.date_col] < pred_time)
        ]

        if reviewer_history.empty:
            return [], [], [], None

        # 月次ステップを列挙
        history_months = pd.date_range(
            start=history_start, end=pred_time, freq="MS"
        )

        monthly_activity_histories: List[List[Dict[str, Any]]] = []
        step_context_dates: List[datetime] = []
        step_total_project_reviews: List[int] = []

        for month_start in history_months[:-1]:
            month_end = month_start + pd.DateOffset(months=1)
            if month_end > pred_time:
                break

            step_context_dates.append(month_end.to_pydatetime())

            # プロジェクト全体のレビュー依頼数
            total_proj = len(
                df[
                    (df[self.date_col] >= history_start)
                    & (df[self.date_col] < month_end)
                ]
            )
            step_total_project_reviews.append(total_proj)

            # この月時点までの活動履歴（レビュー）
            month_history = reviewer_history[
                reviewer_history[self.date_col] < month_end
            ]
            monthly_activities: List[Dict[str, Any]] = []
            for _, row in month_history.iterrows():
                activity = {
                    "timestamp": row[self.date_col],
                    "action_type": "review",
                    "project": row.get("project", "unknown"),
                    "project_id": row.get("project", "unknown"),
                    "request_time": row.get("request_time", row[self.date_col]),
                    "response_time": row.get("first_response_time"),
                    "accepted": row.get(self.label_col, 0) == 1,
                    "owner_email": row.get("owner_email", ""),
                    "is_cross_project": row.get("is_cross_project", False),
                    "files_changed": row.get("change_files_count", 0),
                    "lines_added": row.get("change_insertions", 0),
                    "lines_deleted": row.get("change_deletions", 0),
                }
                monthly_activities.append(activity)

            # 自分が作成した PR（authored）の履歴も追加
            authored_history = df[
                (df["owner_email"] == email)
                & (df[self.date_col] >= history_start)
                & (df[self.date_col] < month_end)
            ]
            for _, row in authored_history.iterrows():
                monthly_activities.append(
                    {
                        "action_type": "authored",
                        "timestamp": row[self.date_col],
                        "owner_email": email,
                        "reviewer_email": row.get(
                            "email", row.get("reviewer_email", "")
                        ),
                        "files_changed": row.get("change_files_count", 0),
                        "lines_added": row.get("change_insertions", 0),
                        "lines_deleted": row.get("change_deletions", 0),
                    }
                )
            monthly_activity_histories.append(monthly_activities)

        # 開発者情報
        accepted = reviewer_history[reviewer_history[self.label_col] == 1]
        developer_info = {
            "developer_id": email,
            "email": email,
            "first_seen": reviewer_history[self.date_col].min(),
            "changes_reviewed": len(accepted),
            "requests_received": len(reviewer_history),
            "acceptance_rate": (
                len(accepted) / len(reviewer_history)
                if len(reviewer_history) > 0
                else 0.0
            ),
            "projects": (
                reviewer_history["project"].unique().tolist()
                if "project" in reviewer_history.columns
                else []
            ),
        }

        return (
            monthly_activity_histories,
            step_context_dates,
            step_total_project_reviews,
            developer_info,
        )

    def predict_developer_directory(
        self,
        email: str,
        directory: str,
        prediction_time: datetime,
        path_extractor=None,
        head_index: int = 0,
    ) -> float:
        """
        (開発者, ディレクトリ) ペアの continuation_prob を推論する。

        Args:
            email: 開発者のメールアドレス
            directory: 対象ディレクトリ（例: "nova/compute"）
            prediction_time: 予測時点 T
            path_extractor: PathFeatureExtractor インスタンス

        Returns:
            continuation_prob (0.0 ~ 1.0)
        """
        self._load_model()

        (
            monthly_activity_histories,
            step_context_dates,
            step_total_project_reviews,
            developer_info,
        ) = self._build_monthly_data(email, prediction_time)

        if developer_info is None or not monthly_activity_histories:
            return 0.5

        # 各月ステップのパス特徴量を計算
        step_path_features = None
        if path_extractor is not None:
            step_path_features = []
            task_dirs = frozenset({directory})
            for ctx_date in step_context_dates:
                pf = path_extractor.compute(email, task_dirs, ctx_date)
                step_path_features.append(pf)

        result = self._irl_system.predict_continuation_probability_monthly(
            developer=developer_info,
            monthly_activity_histories=monthly_activity_histories,
            step_context_dates=step_context_dates,
            context_date=prediction_time,
            step_total_project_reviews=step_total_project_reviews,
            step_path_features=step_path_features,
            head_index=head_index,
        )

        return float(result.get("continuation_probability", 0.5))

    def predict_developer_directories(
        self,
        email: str,
        directories: List[str],
        prediction_time: datetime,
        path_extractor=None,
        head_index: int = 0,
    ) -> Dict[str, float]:
        """
        1人の開発者について複数ディレクトリの continuation_prob を一括推論する。

        predict_developer_directory を dir ごとに呼ぶと _build_monthly_data
        (pandas iterrows × 月数) が directory に依存しないのに毎回再計算され
        非効率なため、月次データは 1 回だけ構築して dir ごとに path_features
        と forward だけループする。

        Returns:
            {directory: continuation_prob} の辞書。履歴がない場合は全 dir に 0.5。
        """
        self._load_model()

        (
            monthly_activity_histories,
            step_context_dates,
            step_total_project_reviews,
            developer_info,
        ) = self._build_monthly_data(email, prediction_time)

        if developer_info is None or not monthly_activity_histories:
            return {d: 0.5 for d in directories}

        results: Dict[str, float] = {}
        for directory in directories:
            step_path_features = None
            if path_extractor is not None:
                step_path_features = []
                task_dirs = frozenset({directory})
                for ctx_date in step_context_dates:
                    pf = path_extractor.compute(email, task_dirs, ctx_date)
                    step_path_features.append(pf)

            result = self._irl_system.predict_continuation_probability_monthly(
                developer=developer_info,
                monthly_activity_histories=monthly_activity_histories,
                step_context_dates=step_context_dates,
                context_date=prediction_time,
                step_total_project_reviews=step_total_project_reviews,
                step_path_features=step_path_features,
                head_index=head_index,
            )
            results[directory] = float(result.get("continuation_probability", 0.5))

        return results

    def predict_developer(
        self,
        email: str,
        prediction_time: datetime,
    ) -> float:
        """
        1人の開発者の continuation_prob を推論する。

        Args:
            email: 開発者のメールアドレス
            prediction_time: 予測時点 T

        Returns:
            continuation_prob (0.0 ~ 1.0)。履歴が無い場合は 0.5。
        """
        self._load_model()

        (
            monthly_activity_histories,
            step_context_dates,
            step_total_project_reviews,
            developer_info,
        ) = self._build_monthly_data(email, prediction_time)

        if developer_info is None or not monthly_activity_histories:
            return 0.5

        result = self._irl_system.predict_continuation_probability_monthly(
            developer=developer_info,
            monthly_activity_histories=monthly_activity_histories,
            step_context_dates=step_context_dates,
            context_date=prediction_time,
            step_total_project_reviews=step_total_project_reviews,
        )

        return float(result.get("continuation_probability", 0.5))

    def predict_batch(
        self,
        emails: List[str],
        prediction_time: datetime,
    ) -> Dict[str, float]:
        """
        複数開発者の continuation_prob を一括推論する。

        Returns:
            {email: continuation_prob}
        """
        self._load_model()
        results: Dict[str, float] = {}
        for i, email in enumerate(emails):
            prob = self.predict_developer(email, prediction_time)
            results[email] = prob
            if (i + 1) % 50 == 0:
                logger.info(
                    f"  predict_batch: {i + 1}/{len(emails)} 完了"
                )
        logger.info(
            f"predict_batch 完了: {len(emails)} 名, "
            f"avg_prob={np.mean(list(results.values())):.3f}"
        )
        return results
