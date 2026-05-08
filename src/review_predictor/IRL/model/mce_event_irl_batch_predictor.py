"""MCE-IRL イベント単位モデル用のバッチ推論クラス。

mce_irl_batch_predictor.py が月次集約用なのに対し、こちらはイベント単位の軌跡
で学習されたモデル (state_dim = 27 = state 20 + path 3 + event 4) のための
推論クラス。

入力構築のロジックは scripts/train/train_model_event.py の
``_process_one_reviewer_event`` と同じく、対象 (developer, directory) ペアに
紐付くレビュー依頼イベントを時系列に並べ、各イベント時点の累積特徴量・
パス特徴量・イベント特徴量を抜き出して 27 次元 state を作る。

最終ステップでの π(a=1|s) を「継続確率」として返す。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


class MCEEventBatchContinuationPredictor:
    """学習済み MCE-IRL イベント単位モデルから π(a=1|s) を取り出すバッチ推論クラス。

    使い方:
        predictor = MCEEventBatchContinuationPredictor(
            model_path="outputs/.../mce_event_irl_model.pt",
            df=df,
            history_start=datetime(2019, 1, 1),
            sliding_window_days=180,
            max_events=256,
        )
        prob = predictor.predict_developer_directory(
            email="alice@example.com",
            directory="nova/compute",
            prediction_time=datetime(2022, 1, 1),
            path_extractor=path_extractor,
        )
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
        dirs_column: str = "dirs",
        sliding_window_days: int = 180,
        max_events: int = 256,
        per_dev: Optional[bool] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.df = df.copy()
        self.history_start = history_start
        self.device = torch.device(device)
        self.reviewer_col = reviewer_col
        self.date_col = date_col
        self.label_col = label_col
        self.dirs_column = dirs_column
        self.sliding_window_days = sliding_window_days
        self.max_events = max_events
        # per_dev: None なら model_metadata.json から自動検出。
        # True: per-dev 軌跡で学習されたモデル。推論時は β 戦略で最終 step の path_features のみを target_dir で上書き。
        self._per_dev_override = per_dev
        self.per_dev: bool = False

        if not pd.api.types.is_datetime64_any_dtype(self.df[self.date_col]):
            self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])

        self._irl_system: Optional[Any] = None

    # ------------------------------------------------------------------
    # モデルロード
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._irl_system is not None:
            return

        from review_predictor.IRL.features.common_features import ACTION_FEATURES
        from review_predictor.IRL.model.mce_irl_predictor import MCEIRLSystem

        metadata_path = self.model_path.parent / "model_metadata.json"
        model_type = 0
        meta_per_dev = False
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            model_type = metadata.get("model_type", 0)
            meta_per_dev = bool(metadata.get("per_dev", False))
            mcls = metadata.get("model_class", "")
            if mcls and mcls != "mce_event_irl":
                raise ValueError(
                    f"{self.model_path} の model_metadata.json は model_class={mcls!r} で"
                    f" あり、イベント単位 MCE-IRL ローダで読み込めるのは"
                    f" model_class='mce_event_irl' のみです。\n"
                    f"  → mce_event_irl_model.pt を指定してください。"
                )
        else:
            logger.warning(
                "model_metadata.json が見つかりません: %s。model_class の検証をスキップします"
                "（state_dim==27 チェックは下で実施されます）。",
                metadata_path,
            )

        # per_dev フラグ確定: コンストラクタ引数 (override) を優先、なければメタから
        if self._per_dev_override is None:
            self.per_dev = meta_per_dev
        else:
            self.per_dev = bool(self._per_dev_override)

        # state_dim は重みから自動判定 (期待値: 27 = state 20 + path 3 + event 4)
        state_dict = torch.load(
            self.model_path, map_location=self.device, weights_only=True
        )
        state_encoder_w = state_dict.get("state_encoder.0.weight")
        state_dim = state_encoder_w.shape[1] if state_encoder_w is not None else 27
        if state_dim != 27:
            raise ValueError(
                f"{self.model_path} の state_dim={state_dim} は 27 ではありません。"
                f" イベント単位 MCE-IRL モデルでない可能性があります"
                f" (月次 MCE-IRL は state_dim=23、グローバル IRL は state_dim=20)。"
            )

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
            "MCE-IRL [event] モデルをロード: %s (model_type=%d, state_dim=%d, per_dev=%s)",
            self.model_path,
            model_type,
            state_dim,
            self.per_dev,
        )

    # ------------------------------------------------------------------
    # イベント時系列の構築 (train_model_event.py と整合)
    # ------------------------------------------------------------------

    def _build_event_sequence(
        self,
        email: str,
        directory: str,
        prediction_time: datetime,
        path_extractor=None,
    ) -> tuple:
        """対象 (email, directory) のイベント単位月次相当データを構築する。

        Returns:
            (event_activity_histories, step_context_dates, step_total_project_reviews,
             path_features_per_step, event_features, developer_info)
            活動履歴がなければすべて空 + developer_info=None。
        """
        df = self.df
        history_start = pd.Timestamp(self.history_start)
        pred_time = pd.Timestamp(prediction_time)

        reviewer_history = df[
            (df[self.reviewer_col] == email)
            & (df[self.date_col] >= history_start)
            & (df[self.date_col] < pred_time)
        ]
        if reviewer_history.empty:
            return [], [], [], [], [], None

        if self.dirs_column not in df.columns:
            return [], [], [], [], [], None

        # 対象ディレクトリに関するイベント
        dir_events = reviewer_history[
            reviewer_history[self.dirs_column].map(
                lambda ds: directory in ds if ds else False
            )
        ].sort_values(self.date_col)
        if dir_events.empty:
            return [], [], [], [], [], None
        if len(dir_events) > self.max_events:
            dir_events = dir_events.tail(self.max_events)

        sliding_delta = pd.Timedelta(days=self.sliding_window_days)

        event_activity_histories: List[List[Dict[str, Any]]] = []
        step_context_dates: List[datetime] = []
        step_total_project_reviews: List[int] = []
        path_features_per_step: List[np.ndarray] = []
        event_features_list: List[Dict[str, float]] = []

        prev_event_time = None
        for _, event_row in dir_events.iterrows():
            event_time = event_row[self.date_col]
            window_start = event_time - sliding_delta

            window_history = reviewer_history[
                (reviewer_history[self.date_col] >= window_start)
                & (reviewer_history[self.date_col] < event_time)
            ]
            activities: List[Dict[str, Any]] = []
            for _, row in window_history.iterrows():
                activities.append(
                    {
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
                )
            authored_window = df[
                (df["owner_email"] == email)
                & (df[self.date_col] >= window_start)
                & (df[self.date_col] < event_time)
            ]
            for _, row in authored_window.iterrows():
                activities.append(
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
            event_activity_histories.append(activities)
            step_context_dates.append(event_time)

            total_proj = len(
                df[(df[self.date_col] >= history_start) & (df[self.date_col] < event_time)]
            )
            step_total_project_reviews.append(total_proj)

            if path_extractor is not None:
                pf = path_extractor.compute(
                    email,
                    frozenset({directory}),
                    event_time.to_pydatetime() if hasattr(event_time, "to_pydatetime")
                    else event_time,
                )
            else:
                pf = np.zeros(3, dtype=np.float32)
            path_features_per_step.append(pf)

            ins = event_row.get("change_insertions", 0)
            dels = event_row.get("change_deletions", 0)
            lines_changed = (ins if pd.notna(ins) else 0) + (
                dels if pd.notna(dels) else 0
            )
            rt_raw = event_row.get("response_latency_days", 0.0)
            response_time = float(rt_raw) if pd.notna(rt_raw) else 0.0
            accepted = 1 if event_row.get(self.label_col, 0) == 1 else 0
            time_since_prev = (
                (event_time - prev_event_time).total_seconds() / 86400.0
                if prev_event_time is not None
                else 30.0
            )
            event_features_list.append(
                {
                    "event_lines_changed": max(0.0, min(lines_changed / 2000.0, 1.0)),
                    "event_response_time": max(0.0, min(response_time / 14.0, 1.0)),
                    "event_accepted": float(accepted),
                    "time_since_prev_event": max(
                        0.0, min(time_since_prev / 180.0, 1.0)
                    ),
                }
            )
            prev_event_time = event_time

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
            event_activity_histories,
            step_context_dates,
            step_total_project_reviews,
            path_features_per_step,
            event_features_list,
            developer_info,
        )

    # ------------------------------------------------------------------
    # per-dev (全 dir 横断) のイベント時系列構築
    # ------------------------------------------------------------------

    def _build_dev_event_sequence(
        self,
        email: str,
        directory: Optional[str],
        prediction_time: datetime,
        path_extractor=None,
    ) -> tuple:
        """対象 reviewer の全 dir 横断イベント時系列を構築する（per-dev 推論用）。

        β 戦略:
            - 各 step の path_features は その event 自身の dirs で計算
            - **最終 step の path_features のみ target_dir で上書き**
            - event_features は本物のまま（target_dir 信号は path_features 末尾のみで伝える）

        directory=None を渡すと β 戦略をスキップ（dev レベル予測用）。
        最終 step の path_features は素のままその event の dirs で計算した値を保持する。

        Returns:
            (event_activity_histories, step_context_dates, step_total_project_reviews,
             path_features_per_step, event_features, developer_info)
        """
        df = self.df
        history_start = pd.Timestamp(self.history_start)
        pred_time = pd.Timestamp(prediction_time)

        reviewer_history = df[
            (df[self.reviewer_col] == email)
            & (df[self.date_col] >= history_start)
            & (df[self.date_col] < pred_time)
        ]
        if reviewer_history.empty:
            return [], [], [], [], [], None
        if self.dirs_column not in df.columns:
            return [], [], [], [], [], None

        def _row_dirs(ds):
            if not ds:
                return []
            return [d for d in ds if d != "."]

        # 全 event を時系列ソート（dirs が空の row は除外）
        reviewer_history_sorted = reviewer_history.sort_values(self.date_col)
        dir_events = reviewer_history_sorted[
            reviewer_history_sorted[self.dirs_column].map(
                lambda ds: len(_row_dirs(ds)) > 0
            )
        ]
        if dir_events.empty:
            return [], [], [], [], [], None
        if len(dir_events) > self.max_events:
            dir_events = dir_events.tail(self.max_events)

        sliding_delta = pd.Timedelta(days=self.sliding_window_days)

        event_activity_histories: List[List[Dict[str, Any]]] = []
        step_context_dates: List[datetime] = []
        step_total_project_reviews: List[int] = []
        path_features_per_step: List[np.ndarray] = []
        event_features_list: List[Dict[str, float]] = []

        prev_event_time = None
        for _, event_row in dir_events.iterrows():
            event_time = event_row[self.date_col]
            ev_dirs = _row_dirs(event_row[self.dirs_column])
            if not ev_dirs:
                continue
            ev_dirs_frozen = frozenset(ev_dirs)
            window_start = event_time - sliding_delta

            window_history = reviewer_history[
                (reviewer_history[self.date_col] >= window_start)
                & (reviewer_history[self.date_col] < event_time)
            ]
            activities: List[Dict[str, Any]] = []
            for _, row in window_history.iterrows():
                activities.append(
                    {
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
                )
            authored_window = df[
                (df["owner_email"] == email)
                & (df[self.date_col] >= window_start)
                & (df[self.date_col] < event_time)
            ]
            for _, row in authored_window.iterrows():
                activities.append(
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
            event_activity_histories.append(activities)
            step_context_dates.append(event_time)

            total_proj = len(
                df[(df[self.date_col] >= history_start) & (df[self.date_col] < event_time)]
            )
            step_total_project_reviews.append(total_proj)

            # 通常 step: その event 自身の dirs で path_features を計算
            if path_extractor is not None:
                pf = path_extractor.compute(
                    email,
                    ev_dirs_frozen,
                    event_time.to_pydatetime() if hasattr(event_time, "to_pydatetime")
                    else event_time,
                )
            else:
                pf = np.zeros(3, dtype=np.float32)
            path_features_per_step.append(pf)

            ins = event_row.get("change_insertions", 0)
            dels = event_row.get("change_deletions", 0)
            lines_changed = (ins if pd.notna(ins) else 0) + (
                dels if pd.notna(dels) else 0
            )
            rt_raw = event_row.get("response_latency_days", 0.0)
            response_time = float(rt_raw) if pd.notna(rt_raw) else 0.0
            accepted = 1 if event_row.get(self.label_col, 0) == 1 else 0
            time_since_prev = (
                (event_time - prev_event_time).total_seconds() / 86400.0
                if prev_event_time is not None
                else 30.0
            )
            event_features_list.append(
                {
                    "event_lines_changed": max(0.0, min(lines_changed / 2000.0, 1.0)),
                    "event_response_time": max(0.0, min(response_time / 14.0, 1.0)),
                    "event_accepted": float(accepted),
                    "time_since_prev_event": max(
                        0.0, min(time_since_prev / 180.0, 1.0)
                    ),
                }
            )
            prev_event_time = event_time

        if not step_context_dates:
            return [], [], [], [], [], None

        # ── β 戦略: 最終 step の path_features を target_dir で上書き ──
        # directory=None なら上書きをスキップ（dev レベル予測用）
        if (
            directory is not None
            and path_extractor is not None
            and path_features_per_step
        ):
            last_event_time = step_context_dates[-1]
            last_pf = path_extractor.compute(
                email,
                frozenset({directory}),
                last_event_time.to_pydatetime()
                if hasattr(last_event_time, "to_pydatetime")
                else last_event_time,
            )
            path_features_per_step[-1] = last_pf

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
            event_activity_histories,
            step_context_dates,
            step_total_project_reviews,
            path_features_per_step,
            event_features_list,
            developer_info,
        )

    # ------------------------------------------------------------------
    # 推論 API
    # ------------------------------------------------------------------

    def predict_developer_directory(
        self,
        email: str,
        directory: str,
        prediction_time: datetime,
        path_extractor=None,
        head_index: int = 0,
    ) -> float:
        """イベント時系列から最終時点の π(a=1|s) を返す。

        per_dev=True なら全 dir 横断で軌跡を構築し、最終 step の path_features のみ
        target_dir で上書きする (β 戦略)。
        """
        self._load_model()

        if self.per_dev:
            (
                event_histories,
                step_context_dates,
                step_total_proj,
                path_feats,
                event_feats,
                developer_info,
            ) = self._build_dev_event_sequence(
                email, directory, prediction_time, path_extractor
            )
        else:
            (
                event_histories,
                step_context_dates,
                step_total_proj,
                path_feats,
                event_feats,
                developer_info,
            ) = self._build_event_sequence(
                email, directory, prediction_time, path_extractor
            )

        if developer_info is None or not event_histories:
            return 0.5

        result = self._irl_system.predict_continuation_probability_monthly(
            developer=developer_info,
            monthly_activity_histories=event_histories,
            step_context_dates=step_context_dates,
            context_date=prediction_time,
            step_total_project_reviews=step_total_proj,
            step_path_features=path_feats,
            step_event_features=event_feats,
            head_index=head_index,
        )
        return float(result.get("continuation_probability", 0.5))

    def predict_developer(
        self,
        email: str,
        prediction_time: datetime,
        path_extractor=None,
        head_index: int = 0,
    ) -> float:
        """dev レベル継続確率を返す（per-dev モデル専用、target_dir 不要）。

        β 戦略の path_features 上書きをスキップし、各 step の path_features は
        その event 自身の dirs で計算した素の値を使う。学習時の軌跡構築と推論時の
        軌跡構築が完全に一致するため、per-dev 学習目標 (= dev が継続するか) と
        整合的な評価になる。

        per_dev=False のモデルでは ValueError を出す。
        """
        self._load_model()
        if not self.per_dev:
            raise ValueError(
                "predict_developer は per_dev=True で学習されたモデルでのみ使えます。"
                " (dev, dir) ペア軌跡で学習されたモデルでは "
                "predict_developer_directory を使ってください。"
            )

        (
            event_histories,
            step_context_dates,
            step_total_proj,
            path_feats,
            event_feats,
            developer_info,
        ) = self._build_dev_event_sequence(
            email, None, prediction_time, path_extractor
        )

        if developer_info is None or not event_histories:
            return 0.5

        result = self._irl_system.predict_continuation_probability_monthly(
            developer=developer_info,
            monthly_activity_histories=event_histories,
            step_context_dates=step_context_dates,
            context_date=prediction_time,
            step_total_project_reviews=step_total_proj,
            step_path_features=path_feats,
            step_event_features=event_feats,
            head_index=head_index,
        )
        return float(result.get("continuation_probability", 0.5))

    def predict_batch_pairs(
        self,
        pairs: List[tuple],
        prediction_time: datetime,
        path_extractor=None,
    ) -> Dict[tuple, float]:
        """[(email, directory), ...] のリストに対して一括推論。"""
        self._load_model()
        results: Dict[tuple, float] = {}
        for i, (email, directory) in enumerate(pairs):
            results[(email, directory)] = self.predict_developer_directory(
                email, directory, prediction_time, path_extractor
            )
            if (i + 1) % 50 == 0:
                logger.info(
                    "predict_batch_pairs: %d/%d 完了", i + 1, len(pairs)
                )
        return results
