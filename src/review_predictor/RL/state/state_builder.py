"""
状態構築モジュール（RL/state/state_builder.py）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ このファイルの役割
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
強化学習では「今の状況」を数値ベクトルで表現する必要がある。
このファイルはその「状態ベクトル」を生データ（DataFrame）から作る。

  開発者のレビュー履歴 (DataFrame)
        ↓ StateBuilder.build()
  状態ベクトル: [0.5, 0.3, 0.8, ...] （14次元の numpy 配列）
        ↓ ReviewEnv に渡す
  エージェントが行動を判断する材料になる

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ 状態ベクトルの次元
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
現在（Phase 1 継承）: 14次元（ミクロ特徴量）
  インデックス 0〜9  : 状態特徴量（開発者の経験・活動・協力度など）
  インデックス 10〜13: 行動特徴量（レビューの強度・速度・サイズなど）

将来（M3 実装後）: 14 + マクロ特徴量の次元数
  追加分: プロジェクト全体のタスク数・リリース位相など

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ 設計のポイント：StateBuilder を変えても他は変更不要
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
特徴量を追加・削除しても ReviewEnv や RewardFunction は変更不要。
（ReviewEnv は obs_dim プロパティで次元数を自動取得するため）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ 使用例
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    from RL.state.state_builder import StateBuilder
    from datetime import datetime

    builder = StateBuilder(window_days=90, normalize=True)

    # 1人分の状態ベクトル（shape: (14,)）
    vec = builder.build(df=df, developer_id="alice@example.com",
                        current_time=datetime(2013, 6, 1))

    # 全員分をまとめて取得（dict）
    state_dict = builder.build_all(df=df, developer_ids=[...],
                                   current_time=datetime(2013, 6, 1))
    # → {"alice@...": array([...]), "bob@...": array([...])}
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, FrozenSet, List, Optional

import numpy as np   # 数値計算ライブラリ。状態ベクトルは np.ndarray で表現
import pandas as pd  # データフレーム操作

# 卒論で作った特徴量抽出関数を流用
# FEATURE_NAMES: 14個の特徴量名のリスト（文字列）
# extract_common_features: DataFrame から特徴量を計算して辞書で返す関数
# normalize_features: 特徴量を 0〜1 に揃える関数
from IRL.features.common_features import (
    FEATURE_NAMES,
    extract_common_features,
    normalize_features,
)
from IRL.features.path_features import (
    PATH_FEATURE_DIM,
    PATH_FEATURE_NAMES,
    PathFeatureExtractor,
)

logger = logging.getLogger(__name__)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  ブロック A: 特徴量リスト定義（MICRO_FEATURES / MACRO_FEATURES）        ║
# ╠══════════════════════════════════════════════════════════════════════╣
# ║  【このブロックの役割】                                                 ║
# ║  状態ベクトルに「どの特徴量を・どの順番で」含めるかを定義する。             ║
# ║  ここを変えると状態ベクトルの次元数と中身が変わる。                        ║
# ║                                                                      ║
# ║  【変更できること】                                                     ║
# ║  ・MACRO_FEATURES: コメントを外して特徴量を追加する（M3 で実施）          ║
# ║      → 追加したら必ず _build_macro() にも計算ロジックを書く              ║
# ║  ・MICRO_FEATURES にある特徴量の順番を変えたい場合:                      ║
# ║      common_features.py の FEATURE_NAMES の並び順が変わる              ║
# ║      ただし reward.py の REVIEW_LOAD_IDX=9 など                       ║
# ║      インデックス参照箇所も合わせて変更が必要                             ║
# ║                                                                      ║
# ║  【変更してはいけないこと】                                              ║
# ║  ・MICRO_FEATURES の削除・変更                                         ║
# ║      → IRL との特徴量の整合性が崩れる                                   ║
# ║  ・ALL_FEATURES の定義方法（MICRO + MACRO の順）                       ║
# ║      → StateBuilder.feature_names が依存している                       ║
# ╚══════════════════════════════════════════════════════════════════════╝

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 特徴量の次元定義（ここを変えると状態ベクトルの中身が変わる）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Phase 1（卒論）からそのまま引き継ぐミクロ特徴量（14次元）
# FEATURE_NAMES の中身:
#   ['experience_days', 'total_changes', 'total_reviews',
#    'recent_activity_frequency', 'avg_activity_gap', 'activity_trend',
#    'collaboration_score', 'code_quality_score', 'recent_acceptance_rate',
#    'review_load',                                    ← ここまで状態特徴量（10次元）
#    'avg_action_intensity', 'avg_collaboration',
#    'avg_response_time', 'avg_review_size']           ← 行動特徴量（4次元）
MICRO_FEATURES: List[str] = FEATURE_NAMES  # IRL/features/common_features.py 参照

# Phase 2 で追加するマクロ特徴量（プロジェクト全体の情報）
# M3 で実装する際は、ここのコメントを外して _build_macro() も実装する
MACRO_FEATURES: List[str] = [
    # "task_queue_length",        # オープンタスク数（正規化済み）
    # "release_cycle_phase",      # リリースサイクルの位相 0.0（リリース直後）〜1.0（直前）
    # "project_acceptance_rate",  # プロジェクト全体の直近30日承諾率
]

# 全特徴量名（ミクロ + マクロ）
# MACRO_FEATURES が空なら MICRO_FEATURES と同じ
ALL_FEATURES: List[str] = MICRO_FEATURES + MACRO_FEATURES


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  ブロック B: StateBuilder クラス                                       ║
# ╠══════════════════════════════════════════════════════════════════════╣
# ║  【このブロックの役割】                                                 ║
# ║  DataFrame（生データ）→ numpy 配列（状態ベクトル）の変換を担う。          ║
# ║  ReviewEnv と RewardFunction はこのクラスの出力を受け取るだけで、        ║
# ║  中身の計算方法を知らなくてよい設計になっている。                          ║
# ║                                                                      ║
# ║  【サブブロック構成】                                                   ║
# ║  B-1: __init__()     設定の受け取り（window_days, normalize, ...）      ║
# ║  B-2: feature_names  現在有効な特徴量名リストを返す @property            ║
# ║  B-3: obs_dim        状態ベクトルの次元数を返す @property               ║
# ║  B-4: build()        1人分の状態ベクトルを構築するメインメソッド           ║
# ║  B-5: build_all()    全員分を一括構築して辞書で返す                      ║
# ║  B-6: _build_macro() マクロ特徴量の計算（M3 で実装）                    ║
# ║                                                                      ║
# ║  【変更できること】                                                     ║
# ║  ・window_days のデフォルト値（現在 90 日）                              ║
# ║      短くすると直近の変化を敏感に拾う。長くすると長期傾向を重視する。        ║
# ║  ・normalize のデフォルト値（現在 True）                                 ║
# ║      False にするとスケールがバラバラになり学習が不安定になりやすい         ║
# ║  ・build_all() に並列化を導入してパフォーマンスを改善できる               ║
# ║      例: concurrent.futures.ThreadPoolExecutor を使った並列計算          ║
# ║                                                                      ║
# ║  【変更してはいけないこと】                                              ║
# ║  ・obs_dim プロパティの計算方法（ReviewEnv がこれに依存）                 ║
# ║  ・build() の戻り値の型（np.ndarray float32）                           ║
# ║  ・build_all() の戻り値の型（Dict[str, np.ndarray]）                    ║
# ╚══════════════════════════════════════════════════════════════════════╝

class StateBuilder:
    """
    RL 環境用の状態ベクトルを構築するクラス。

    インスタンス化するとき（StateBuilder(...)）に設定を渡し、
    build() / build_all() で実際の状態ベクトルを生成する。

    Args:
        window_days: 特徴量を計算するのに使う過去データの期間（日数）
                     例: 90 なら「現在から90日前まで」のデータで計算
        normalize:   True にすると全特徴量を 0〜1 に正規化する
                     （ニューラルネットは 0〜1 のスケールで学習が安定する）
        use_macro:   True にするとマクロ特徴量も含める（M3 実装後に切り替える）
    """

    def __init__(
        self,
        window_days: int = 90,
        normalize: bool = True,
        use_macro: bool = False,  # M3 実装前は False のまま
        path_extractor: Optional[PathFeatureExtractor] = None,
    ) -> None:
        self.window_days = window_days
        self.normalize = normalize
        self.use_macro = use_macro
        # Phase 2 (Step 3): ディレクトリ親和度特徴量
        # None のときは path features を付けない (ablation 用)
        self.path_extractor = path_extractor

    # ── B-1: __init__ ────────────────────────────────────────────────────
    # 【変更できること】
    # ・window_days: 30（直近1ヶ月）〜 365（1年）で調整できる
    #   短い → 直近の活動状況を重視する
    #   長い → 長期的な傾向・経験値を重視する
    # ・use_macro: M3 実装後に True に変えると自動でマクロ特徴量が追加される

    # ── B-2,3: @property ─────────────────────────────────────────────────
    # 【@property とは】
    # メソッドだが () なしで属性のように呼べる。
    # builder.obs_dim() ではなく builder.obs_dim で呼ぶ。
    # 読み取り専用の値を提供するのに使う。
    # 【変更してはいけないこと】
    # obs_dim の計算式は変えない（ReviewEnv の observation_space がこれに依存）

    # ── プロパティ（値を返す getter）──────────────────────────────────────

    @property  # ← メソッドを属性のように使えるデコレータ（() なしで呼べる）
    def feature_names(self) -> List[str]:
        """
        現在の設定での特徴量名リストを返す。

        use_macro=False → MICRO_FEATURES（14次元）
        use_macro=True  → MICRO_FEATURES + MACRO_FEATURES（14 + α次元）

        使い方:
            builder = StateBuilder()
            print(builder.feature_names)  # ['experience_days', ...]
        """
        names = list(MICRO_FEATURES)   # コピーして元のリストを変えないようにする
        if self.use_macro:
            names += MACRO_FEATURES
        if self.path_extractor is not None:
            names += PATH_FEATURE_NAMES
        return names

    @property
    def obs_dim(self) -> int:
        """
        状態ベクトルの次元数を返す。

        ReviewEnv はこの値を使って observation_space の shape を決める。
        特徴量を追加したときにここを変える必要はない（自動で変わる）。

        使い方:
            builder = StateBuilder()
            print(builder.obs_dim)  # 14
        """
        return len(self.feature_names)

    # ── B-4,5: メイン API ────────────────────────────────────────────────
    # 【変更できること】
    # ・build() に前処理を追加できる
    #   例: 異常値のクリッピング（review_load が 10 以上なら 1.0 に丸めるなど）
    #       vector = np.clip(vector, 0.0, 1.0)  ← normalize 後に追加
    # ・build_all() を並列化できる
    #   例: joblib の Parallel を使って複数開発者を同時計算
    # 【変更してはいけないこと】
    # ・戻り値の型（float32 の ndarray）
    # ・MICRO_FEATURES の順番で配列を構築する部分（報酬関数がインデックス参照するため）

    # ── メイン API ────────────────────────────────────────────────────────

    def build(
        self,
        df: pd.DataFrame,
        developer_id: str,
        current_time: datetime,
        task_dirs: Optional[FrozenSet[str]] = None,
    ) -> np.ndarray:
        """
        1人の開発者の状態ベクトルを構築して返す。

        処理の流れ:
            1. current_time から window_days 日前を計算（特徴量計算の開始点）
            2. IRL/features/common_features.py の extract_common_features() で計算
            3. 辞書形式 → numpy 配列に変換
            4. use_macro=True ならマクロ特徴量を末尾に追加

        Args:
            df:           全期間のレビューデータ（全開発者分）
            developer_id: 対象開発者のメールアドレス（例: "alice@example.com"）
            current_time: 「今」の時刻。これより過去 window_days 日分を使う

        Returns:
            shape (obs_dim,) の numpy float32 配列
            例（14次元）: array([0.5, 0.3, 0.8, 0.1, ...], dtype=float32)
        """
        # 特徴量計算に使うデータの期間を決める
        # 例: current_time=2013-06-01, window_days=90 なら
        #     feature_start=2013-03-03, feature_end=2013-06-01
        feature_start = current_time - timedelta(days=self.window_days)
        feature_end = current_time

        # 卒論の特徴量計算関数を呼ぶ（IRL/features/common_features.py）
        # 戻り値は辞書: {'experience_days': 180.0, 'total_changes': 45.0, ...}
        features = extract_common_features(
            df=df,
            email=developer_id,
            feature_start=feature_start,
            feature_end=feature_end,
            normalize=self.normalize,  # True なら 0〜1 に揃えてくれる
        )

        # 辞書 → numpy 配列に変換
        # MICRO_FEATURES の順番通りに値を並べる（順番が重要！）
        # 例: features["experience_days"] → vector[0]
        #     features["total_changes"]   → vector[1]  など
        vector = np.array([features[name] for name in MICRO_FEATURES], dtype=np.float32)

        # マクロ特徴量が有効なら末尾に追加
        if self.use_macro:
            macro_vector = self._build_macro(df, current_time)
            # np.concatenate: 配列を横に連結する
            # [0.5, 0.3, ...] + [0.7, 0.2, 0.9] → [0.5, 0.3, ..., 0.7, 0.2, 0.9]
            vector = np.concatenate([vector, macro_vector])

        # Path features (Step 3): ディレクトリ親和度を末尾に追加
        if self.path_extractor is not None:
            path_vector = self.path_extractor.compute(
                developer_id=developer_id,
                task_dirs=task_dirs,
                current_time=current_time,
            )
            vector = np.concatenate([vector, path_vector])

        return vector

    def build_all(
        self,
        df: pd.DataFrame,
        developer_ids: List[str],
        current_time: datetime,
        task_dirs: Optional[FrozenSet[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        複数の開発者の状態ベクトルを一括で構築して辞書で返す。

        ReviewEnv から毎ステップ呼ばれるメソッド。

        Args:
            df:            全期間のレビューデータ
            developer_ids: 対象開発者のメールアドレスリスト
            current_time:  「今」の時刻

        Returns:
            {メールアドレス: 状態ベクトル} の辞書
            例: {
                "alice@example.com": array([0.5, 0.3, ...]),
                "bob@example.com":   array([0.2, 0.7, ...]),
            }
        """
        # 辞書内包表記: {キー: 値 for 変数 in イテラブル}
        # 各開発者について build() を呼んで辞書を作る
        return {
            dev_id: self.build(df, dev_id, current_time, task_dirs=task_dirs)
            for dev_id in developer_ids
        }

    # ── B-6: マクロ特徴量（M3 で実装予定）───────────────────────────────
    # 【このブロックを実装するときの手順】
    # Step 1: MACRO_FEATURES（このファイルの上部）のコメントを外す
    # Step 2: _build_macro() に各特徴量の計算ロジックを書く
    # Step 3: StateBuilder(use_macro=True) でインスタンス化して動作確認
    # 【変更できること】
    # ・追加するマクロ特徴量の種類と計算方法
    # ・正規化の方法（現在の MICRO_FEATURES と同様に 0〜1 に揃えることを推奨）
    # 【変更してはいけないこと】
    # ・MACRO_FEATURES のリスト長と return する配列の長さは必ず一致させること
    #   一致しないと obs_dim がズレて ReviewEnv の observation_space が壊れる

    # ── マクロ特徴量（M3 で実装予定）─────────────────────────────────────

    def _build_macro(
        self,
        df: pd.DataFrame,
        current_time: datetime,
    ) -> np.ndarray:
        """
        プロジェクトレベルのマクロ特徴量ベクトルを構築する。

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        TODO (M3): 以下の手順で実装する
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Step 1: MACRO_FEATURES リストのコメントを外して特徴量名を有効化する
        Step 2: 各特徴量の計算ロジックをここに実装する
        Step 3: StateBuilder(use_macro=True) で動作確認

        実装例:
            recent = df[df["timestamp"] >= current_time - timedelta(days=30)]

            task_queue_length = min(len(recent) / 100.0, 1.0)

            # リリースサイクル位相: 別途リリース日程データから計算
            release_cycle_phase = 0.5  # 仮実装

            if "label" in recent.columns and len(recent) > 0:
                project_acceptance_rate = float(recent["label"].mean())
            else:
                project_acceptance_rate = 0.5

            return np.array([
                task_queue_length,
                release_cycle_phase,
                project_acceptance_rate,
            ], dtype=np.float32)

        注意: MACRO_FEATURES の要素数と return する配列の長さを必ず一致させること。
        """
        # MACRO_FEATURES が空リストの場合は長さ0の配列を返す（エラーにならない）
        if len(MACRO_FEATURES) == 0:
            return np.array([], dtype=np.float32)

        # MACRO_FEATURES に名前が入っているのに実装がない場合はエラーを出す
        raise NotImplementedError(
            "TODO (M3): マクロ特徴量の実装が必要です。\n"
            "MACRO_FEATURES リストと _build_macro を同時に更新してください。"
        )
