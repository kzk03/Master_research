"""
ファイルパス特徴量（ディレクトリ単位）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ このモジュールの役割
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
「人 × ファイルパス」2軸の推薦を実現するために、
開発者ごとに「どのディレクトリをどれだけ触ってきたか」という特徴量を
抽出する。従来のミクロ特徴量（経験日数・承諾率など）は
開発者の「全体的な状態」しか表現できないが、path features は
現在タスクとの「親和度」を表現できる。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ 特徴量 (6 次元)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  path_review_count       : 該当ディレクトリでの過去レビュー件数 (log正規化)
  path_recency            : 該当ディレクトリを最後に触ってからの新しさ (0-1)
  path_acceptance_rate    : 該当ディレクトリでの過去承諾率
  path_owner_overlap      : task dir の過去 owner 集合 ∩ reviewer の collaborator 集合 (Jaccard)
                            (Casalnuovo FSE 2015 の tie strength 系)
  path_hub_score          : task_dirs の co-change graph 上での hub score (グローバル max-norm)
                            (Zanetti ICSE 2013 の HITS hub-score 系 / 本研究 B-3, B-11)
  path_neighbor_coverage  : |task_dir の co-change 近傍 ∩ reviewer 担当 dirs| / |近傍|
                            (Zimmermann TSE 2005 の co-change rules / 本研究 B-12 最強単一予測子)

■ 2026-05-14 改訂 (Phase 1)
  path_owner_overlap (Jaccard tie strength) を追加 (+1)。
  累積カウントの差分系（直近 90d レビュー数等）は LSTM のゲーティングで暗黙学習
  できるため意図的に追加しない。

■ 2026-05-15 改訂 (Phase 2)
  co-change graph 由来の 2 特徴量 (hub_score, neighbor_coverage) を追加 (+2)。
  どちらも reviewer の月次累積活動から LSTM が抽出するのは困難な、
  外部グラフ構造に基づく情報源。

■ 2026-05-15 改訂 (Phase 1 振り返り)
  path_lcp_similarity (REVFINDER 式) は v2_phase1 評価で RF importance = 0.000
  と完全 dead だったため削除。サンプルバイアス (依頼を受けた reviewer は対象
  dir に過去触れている率が高い → 自分自身との LCP=1 で飽和) が原因と判断。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ データソース
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
nova_raw.csv にはファイルパス列が無いので、data/raw_json/openstack__nova.json
から change_id → touched file set を抽出し、ディレクトリに集約する。
CSV 側の change_id は `openstack%2Fnova~969380` 形式で、JSON 側は
`project='openstack/nova'` と `_number=969380` から再構築する。

hub_score / neighbor_coverage は事前計算済み CSV を `PathFeatureExtractor` の
コンストラクタに渡す:
  experiments/dependency_analysis/results/hub_scores_main32.csv
  experiments/dependency_analysis/results/cochange_neighbors_main32.csv
未指定なら両特徴量とも 0.0 を返す（後方互換）。
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Union
from urllib.parse import quote

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 定数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#: path features の名前（state vector への追加順）
#  注: path_lcp_similarity は v2_phase1 評価で RF importance=0.000 と完全 dead
#  だったため 2026-05-15 に削除。リスト末尾にダミー出力していたコードは
#  PATH_FEATURE_DIM = len(PATH_FEATURE_NAMES) で自動追従するので影響なし。
PATH_FEATURE_NAMES: List[str] = [
    "path_review_count",
    "path_recency",
    "path_acceptance_rate",
    "path_owner_overlap",        # 2026-05-14 追加 (Jaccard, Casalnuovo 系)
    "path_hub_score",            # 2026-05-15 追加 (co-change HITS hub, Zanetti 系)
    "path_neighbor_coverage",    # 2026-05-15 追加 (co-change neighbor coverage, B-12)
]

#: path features の次元数
PATH_FEATURE_DIM: int = len(PATH_FEATURE_NAMES)

#: log 正規化の上限（review count 100 件で 1.0）
_REVIEW_COUNT_LOG_CAP: float = math.log1p(100.0)

#: hub_score の正規化キャップ。32 main repos での実測 max は ~64 (openstack/nova の
#  68 dir 中、最も connected な dir で 64 近傍)。 64 で 1.0 に正規化。
_HUB_SCORE_NORM_CAP: float = 64.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ヘルパー: ファイルパス → ディレクトリ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def file_to_dir(file_path: str, depth: int = 2) -> str:
    """
    ファイルパスから先頭 `depth` 階層のディレクトリを抽出する。

    例 (depth=2):
        "nova/compute/manager.py"                    -> "nova/compute"
        "nova/tests/functional/regressions/test.py"  -> "nova/tests"
        "README.rst"                                 -> "."

    Args:
        file_path: ファイルパス (例: "nova/compute/manager.py")
        depth:     何階層目までをディレクトリとして使うか (既定 2)

    Returns:
        ディレクトリ文字列。階層が足りないファイル (ルート直下) は "."
    """
    parts = [p for p in file_path.split("/") if p]
    if len(parts) <= 1:
        return "."
    return "/".join(parts[: min(depth, len(parts) - 1)])


def extract_dirs(file_paths: Iterable[str], depth: int = 2) -> FrozenSet[str]:
    """ファイルパスの集合 → ディレクトリ集合 (frozenset)."""
    return frozenset(file_to_dir(fp, depth=depth) for fp in file_paths)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# raw_json → change_id → dirs マッピング構築
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_change_dir_map(
    json_path: str | Path,
    depth: int = 2,
) -> Dict[str, FrozenSet[str]]:
    """
    raw_json (data/raw_json/openstack__<proj>.json) を読み込み、
    CSV 側と整合する change_id キー → ディレクトリ集合の辞書を返す。

    CSV 側のキー形式: `openstack%2Fnova~969380`
    JSON 側: project='openstack/nova', _number=969380

    Args:
        json_path: raw json ファイルのパス
        depth:     ディレクトリ階層数 (既定 2)

    Returns:
        { "openstack%2Fnova~969380": frozenset({"nova/compute", ...}), ... }
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"raw json が見つかりません: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        changes = json.load(f)

    result: Dict[str, FrozenSet[str]] = {}
    skipped = 0
    for change in changes:
        project = change.get("project")
        number = change.get("_number")
        if project is None or number is None:
            skipped += 1
            continue

        # revisions[*].files から全ファイルパスを集める
        files: Set[str] = set()
        for rev in (change.get("revisions") or {}).values():
            for fp in (rev.get("files") or {}).keys():
                # gerrit の特殊エントリ "/COMMIT_MSG" 等を除外
                if fp.startswith("/"):
                    continue
                files.add(fp)

        if not files:
            skipped += 1
            continue

        key = f"{quote(project, safe='')}~{number}"
        result[key] = extract_dirs(files, depth=depth)

    logger.info(
        f"change_dir_map 構築完了: {len(result)} changes "
        f"(スキップ {skipped} 件), depth={depth}"
    )
    return result


def load_change_dir_map_multi(
    json_paths: list[str | Path],
    depth: int = 2,
) -> Dict[str, FrozenSet[str]]:
    """複数の raw JSON ファイルから change_dir_map を統合構築する。"""
    merged: Dict[str, FrozenSet[str]] = {}
    for jp in json_paths:
        partial = load_change_dir_map(jp, depth=depth)
        merged.update(partial)
        logger.info(f"  {Path(jp).name}: {len(partial)} changes")
    logger.info(
        f"load_change_dir_map_multi: 合計 {len(merged)} changes "
        f"({len(json_paths)} files)"
    )
    return merged


def attach_dirs_to_df(
    df: pd.DataFrame,
    change_dir_map: Dict[str, FrozenSet[str]],
    column: str = "dirs",
) -> pd.DataFrame:
    """
    df に 'dirs' 列 (frozenset) を追加した新しい DataFrame を返す。
    マップに存在しない change_id は空 frozenset になる。
    """
    empty: FrozenSet[str] = frozenset()
    df = df.copy()
    df[column] = df["change_id"].map(lambda cid: change_dir_map.get(cid, empty))
    n_matched = int((df[column].map(len) > 0).sum())
    logger.info(
        f"dirs 列を付与: {n_matched}/{len(df)} 行にディレクトリ情報あり"
    )
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PathFeatureExtractor
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PathFeatureExtractor:
    """
    ディレクトリ親和度特徴量を計算するクラス。

    【前提】
    入力 df は事前に attach_dirs_to_df() で 'dirs' 列が付いていること。

    【使い方】
        df = pd.read_csv(...)
        cdm = load_change_dir_map("data/raw_json/openstack__nova.json")
        df = attach_dirs_to_df(df, cdm)

        extractor = PathFeatureExtractor(df, window_days=180)
        vec = extractor.compute(
            developer_id="alice@example.com",
            task_dirs=frozenset({"nova/compute"}),
            current_time=datetime(2020, 6, 15),
        )
        # -> np.ndarray shape (3,)

    Args:
        df:                       'dirs' 列付き DataFrame
        window_days:              この日数内の活動のみ集計 (既定 180)
        dirs_column:              ディレクトリ列の名前
        hub_scores_path:          experiments/dependency_analysis/results/hub_scores_main32.csv
                                  のような CSV ファイル。未指定なら path_hub_score = 0.0
        cochange_neighbors_path:  experiments/dependency_analysis/results/cochange_neighbors_main32.csv
                                  のような CSV ファイル。未指定なら path_neighbor_coverage = 0.0
    """

    def __init__(
        self,
        df: pd.DataFrame,
        window_days: int = 180,
        dirs_column: str = "dirs",
        hub_scores_path: Optional[str | Path] = None,
        cochange_neighbors_path: Optional[str | Path] = None,
    ) -> None:
        if dirs_column not in df.columns:
            raise ValueError(
                f"df に '{dirs_column}' 列がありません。"
                f"attach_dirs_to_df() を先に呼んでください。"
            )
        self.df = df
        self.window_days = window_days
        self.dirs_column = dirs_column

        # timestamp を datetime 型に揃える
        if not pd.api.types.is_datetime64_any_dtype(self.df["timestamp"]):
            self.df = self.df.copy()
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])

        # ── co-change graph 由来の特徴量用キャッシュ ──
        # キーは directory 文字列 (例: "nova/compute")。
        # 32 main repos では directory 名が project namespace 先頭にあるため衝突なし。
        self.hub_score_by_dir: Dict[str, float] = {}
        self.neighbors_by_dir: Dict[str, Set[str]] = {}
        if hub_scores_path is not None:
            self._load_hub_scores(Path(hub_scores_path))
        if cochange_neighbors_path is not None:
            self._load_cochange_neighbors(Path(cochange_neighbors_path))

    def _load_hub_scores(self, path: Path) -> None:
        """hub_scores CSV を読み込み self.hub_score_by_dir に格納."""
        if not path.exists():
            logger.warning(f"hub_scores CSV not found: {path} (path_hub_score=0.0)")
            return
        hs_df = pd.read_csv(path)
        for d, hs in zip(hs_df["directory"], hs_df["hub_score"]):
            d = str(d)
            hs = float(hs)
            # 同 directory 名が複数 project に出現したら max を取る (実際は起きないはず)
            if d in self.hub_score_by_dir:
                self.hub_score_by_dir[d] = max(self.hub_score_by_dir[d], hs)
            else:
                self.hub_score_by_dir[d] = hs
        logger.info(f"hub_scores loaded: {len(self.hub_score_by_dir):,} dirs from {path.name}")

    def _load_cochange_neighbors(self, path: Path) -> None:
        """cochange_neighbors CSV を読み込み self.neighbors_by_dir に格納."""
        if not path.exists():
            logger.warning(
                f"cochange_neighbors CSV not found: {path} (path_neighbor_coverage=0.0)"
            )
            return
        cn_df = pd.read_csv(path)
        for d, group in cn_df.groupby("directory"):
            self.neighbors_by_dir[str(d)] = set(group["neighbor"].astype(str).tolist())
        logger.info(
            f"cochange_neighbors loaded: {len(self.neighbors_by_dir):,} dirs "
            f"({len(cn_df):,} edges) from {path.name}"
        )

    # ── メイン API ────────────────────────────────────────────────────────

    def compute(
        self,
        developer_id: str,
        task_dirs: Optional[FrozenSet[str]],
        current_time: datetime,
    ) -> np.ndarray:
        """
        1 人の開発者について path features (6 次元) を計算する。

        task_dirs が空/None のときは全次元 0.0 を返す
        (= そのタスクにディレクトリ情報が無い場合のフォールバック)。

        返り値の順序は `PATH_FEATURE_NAMES`:
            [review_count, recency, acceptance, owner_overlap,
             hub_score, neighbor_coverage]
        """
        if not task_dirs:
            return np.zeros(PATH_FEATURE_DIM, dtype=np.float32)

        start = current_time - timedelta(days=self.window_days)

        # 開発者 × ウィンドウ内の全行 (LCP / owner_overlap で使う、task_dirs に絞らない)
        mask = (
            (self.df["email"] == developer_id)
            & (self.df["timestamp"] >= start)
            & (self.df["timestamp"] < current_time)
        )
        sub = self.df.loc[mask]
        if sub.empty:
            return np.zeros(PATH_FEATURE_DIM, dtype=np.float32)

        # task_dirs と交差するレビューだけを残す（既存 3 特徴量用）
        dirs_series = sub[self.dirs_column]
        overlap_mask = dirs_series.map(
            lambda ds: bool(ds) and not ds.isdisjoint(task_dirs)
        )
        hits = sub.loc[overlap_mask]

        # ── 既存 3 特徴量 (hits 基準) ────────────────────────────────────
        if hits.empty:
            review_count_norm = 0.0
            recency = 0.0
            acceptance = 0.5  # 中立値
        else:
            # 1) review count (log 正規化)
            review_count = float(len(hits))
            review_count_norm = min(
                math.log1p(review_count) / _REVIEW_COUNT_LOG_CAP, 1.0
            )

            # 2) recency: 最終タッチ日からの新しさ
            last_ts = hits["timestamp"].max()
            days_since = (
                current_time - last_ts.to_pydatetime()
            ).total_seconds() / 86400.0
            recency = max(0.0, 1.0 - min(days_since / self.window_days, 1.0))

            # 3) acceptance rate (label 列が 0/1 前提)
            if "label" in hits.columns:
                acceptance = float(hits["label"].mean())
            else:
                acceptance = 0.5

        # reviewer の窓内 dir 集合 (owner_overlap / neighbor_coverage で使用)
        reviewer_dirs: Set[str] = set()
        for ds in sub[self.dirs_column]:
            if ds:
                reviewer_dirs.update(ds)

        # ── 新規 4) owner overlap (Jaccard) ──────────────────────────────
        # task_dirs に過去 PR を出した owner 集合 ∩ reviewer の collaborator 集合 / 和集合
        # 「該当 dir のオーナーコミュニティと reviewer の交流範囲がどれだけ重なるか」
        if "owner_email" in sub.columns:
            rev_owners: Set[str] = set(sub["owner_email"].dropna().unique())
        else:
            rev_owners = set()

        if "owner_email" in self.df.columns and rev_owners:
            window_mask = (
                (self.df["timestamp"] >= start)
                & (self.df["timestamp"] < current_time)
            )
            window_df = self.df.loc[window_mask]
            td_overlap = window_df[self.dirs_column].map(
                lambda ds: bool(ds) and not ds.isdisjoint(task_dirs)
            )
            td_owners: Set[str] = set(
                window_df.loc[td_overlap, "owner_email"].dropna().unique()
            )
            union = rev_owners | td_owners
            if union:
                path_owner_overlap = len(rev_owners & td_owners) / len(union)
            else:
                path_owner_overlap = 0.0
        else:
            path_owner_overlap = 0.0

        # ── 新規 5) hub_score (Phase 2; Zanetti ICSE 2013 系 / B-3) ────────
        # task_dirs に含まれる dir の co-change graph 上 hub score (= neighbor 数) の最大値。
        # max を取る根拠: マルチ dir change なら「最も中心的な component」が
        # reviewer の継続判断を支配する仮説。グローバル max-norm で 0-1 に。
        if self.hub_score_by_dir:
            scores = [self.hub_score_by_dir.get(d, 0.0) for d in task_dirs]
            path_hub_score = min(max(scores) / _HUB_SCORE_NORM_CAP, 1.0) if scores else 0.0
        else:
            path_hub_score = 0.0

        # ── 新規 6) neighbor_coverage (Phase 2; B-12 本研究最強単一予測子) ─
        # |task_dirs の co-change 近傍 ∩ reviewer が窓内で担当した dirs| / |近傍|
        # reviewer_dirs は LCP 計算で既に構築済み (sub から union)。
        if self.neighbors_by_dir and reviewer_dirs:
            target_neighbors: Set[str] = set()
            for d in task_dirs:
                target_neighbors.update(self.neighbors_by_dir.get(d, set()))
            # 自分自身は coverage の "近傍" に含めない
            target_neighbors -= set(task_dirs)
            if target_neighbors:
                path_neighbor_coverage = (
                    len(target_neighbors & reviewer_dirs) / len(target_neighbors)
                )
            else:
                path_neighbor_coverage = 0.0
        else:
            path_neighbor_coverage = 0.0

        return np.array(
            [
                review_count_norm,
                recency,
                acceptance,
                path_owner_overlap,
                path_hub_score,
                path_neighbor_coverage,
            ],
            dtype=np.float32,
        )

    def compute_all(
        self,
        developer_ids: List[str],
        task_dirs: Optional[FrozenSet[str]],
        current_time: datetime,
    ) -> Dict[str, np.ndarray]:
        """複数開発者分をまとめて計算して辞書で返す。"""
        return {
            dev: self.compute(dev, task_dirs, current_time)
            for dev in developer_ids
        }
