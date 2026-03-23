#!/usr/bin/env python3
"""
データセット構築パイプライン（build_dataset.py）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ このスクリプトの役割
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Gerrit（コードレビューシステム）の REST API からデータを取得し、
機械学習に使える特徴量付き CSV を生成するパイプライン。

    Gerrit API
        ↓ GerritDataFetcher: 変更データを取得
    生データ（JSON）
        ↓ FeatureBuilder._extract_review_requests(): レビュー依頼を1行1件に展開
        ↓ FeatureBuilder._compute_history_features(): 時系列特徴量を計算
    特徴量付き DataFrame
        ↓ カラム名リネーム（developer_email→email, request_time→timestamp）
        ↓ collection_config.json に収集パラメータを記録
    nova_raw.csv（出力）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ 出力 CSV のカラム
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    email        : レビュアーのメール（downstream コードが参照するキー）
    timestamp    : レビュー依頼時刻（downstream コードが参照するキー）
    label        : 0（不承諾）/ 1（承諾）← 予測対象
    change_id    : PR の ID
    project      : プロジェクト名
    owner_email  : PR 作成者のメール
    ...（特徴量 66列）

"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from tqdm import tqdm

# ロギング設定
# format: "時刻 - レベル - メッセージ" の形式でターミナルに出力される
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  クラス 1: GerritDataFetcher                                         ║
# ╠══════════════════════════════════════════════════════════════════════╣
# ║  Gerrit の REST API を叩いて変更データを取得するクラス。               ║
# ║  HTTP 通信・ページネーション・XSSI除去 を担当する。                    ║
# ╚══════════════════════════════════════════════════════════════════════╝

class GerritDataFetcher:
    """
    Gerrit REST API からコードレビューデータを取得するクラス。

    ■ REST API の特徴
    - エンドポイント: /changes/ で Change 一覧を取得できる
    - ページネーション: 1回最大500件。_more_changes フラグで次ページ有無を判定
    - XSSI 保護: レスポンスの先頭に ")]}'" が付く（XSS 攻撃防止のため）
                 → JSONパース前にこのプレフィックスを取り除く必要がある
    """

    def __init__(self, gerrit_url: str, timeout: int = 300):
        """
        Args:
            gerrit_url: Gerrit サーバーのベース URL
                        例: https://review.opendev.org
            timeout:    HTTP リクエストのタイムアウト（秒）
                        5年分のデータ取得など長時間かかる場合は大きめに設定
        """
        self.gerrit_url = gerrit_url.rstrip('/')  # 末尾スラッシュを除去（URLの二重スラッシュ防止）
        self.timeout = timeout
        # requests.Session: 同じサーバーへの接続を使い回す（毎回接続確立するよりも高速）
        self.session = requests.Session()

    def _make_request(self, endpoint: str, params: Dict = None) -> Any:
        """
        Gerrit API に HTTP GET リクエストを送り、JSON をパースして返す内部メソッド。

        ■ XSSI 保護プレフィックスの除去
        Gerrit は XSS 攻撃（スクリプト注入）を防ぐために、
        すべての JSON レスポンスの先頭に ")]}'" という文字列を付加している。
        ブラウザはこのプレフィックスを含む文字列を JSON として実行できないため
        攻撃が無効化される。
        ただし API 利用側はこのプレフィックスを除去してからパースする必要がある。

        Args:
            endpoint: API のパス（例: "changes/"）
            params:   クエリパラメータ（例: {"q": "project:nova", "n": 500}）

        Returns:
            パース済みの JSON（辞書またはリスト）

        Raises:
            Exception: HTTP エラーやタイムアウト時
        """
        url = f"{self.gerrit_url}/{endpoint}"

        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()  # 4xx / 5xx なら例外を投げる

            content = response.text

            # XSSI 保護プレフィックス ")]}'" を除去
            if content.startswith(")]}'"):
                content = content[4:]

            return json.loads(content)

        except Exception as e:
            logger.error(f"API request failed: {url} - {e}")
            raise

    def fetch_changes(self,
                      project: str,
                      start_date: datetime,
                      end_date: datetime,
                      limit: int = 500) -> List[Dict[str, Any]]:
        """
        指定プロジェクト・期間の全 Change データを取得する。

        ■ ページネーションの仕組み
        Gerrit API は 1 回のリクエストで最大 limit 件しか返さない。
        レスポンスの最後の要素に "_more_changes: True" があれば次ページが存在する。
        → start パラメータをずらしながら全件取得するまでループする。

        ■ 取得オプション（"o" パラメータ）
        - DETAILED_ACCOUNTS : レビュアー・オーナーの email 情報を含める
        - DETAILED_LABELS   : +2/+1/0/-1/-2 などの投票情報を含める
        - MESSAGES          : コメント履歴（誰がいつ何を書いたか）を含める
        - CURRENT_REVISION  : 最新リビジョンの ID を含める
        - CURRENT_FILES     : 変更されたファイルパス一覧を含める

        Args:
            project:    プロジェクト名（例: "openstack/nova"）
            start_date: 取得開始日
            end_date:   取得終了日
            limit:      1リクエストあたりの最大取得件数（Gerrit の上限は 500）

        Returns:
            Change 辞書のリスト（1要素 = 1つの PR/Change）
        """
        all_changes = []
        start = 0  # ページネーションのオフセット（何件目から取得するか）

        start_str = start_date.strftime("%Y-%m-%d")
        end_str   = end_date.strftime("%Y-%m-%d")

        # Gerrit のクエリ構文: project: + after: + before: で絞り込む
        query = f"project:{project} after:{start_str} before:{end_str}"

        logger.info(f"Fetching changes for {project} from {start_str} to {end_str}")

        while True:
            params = {
                "q": query,
                # 取得する追加情報を指定（上記の説明参照）
                "o": ["DETAILED_ACCOUNTS", "DETAILED_LABELS", "MESSAGES",
                      "CURRENT_REVISION", "CURRENT_FILES",
                      "REVIEWER_UPDATES", "ALL_REVISIONS",
                      "CURRENT_COMMIT", "TRACKING_IDS",
                      "ALL_FILES", "SUBMITTABLE", "SUBMIT_REQUIREMENTS"],
                "n": limit,    # 1回の取得上限
                "start": start # 取得開始位置（ページネーション用）
            }

            try:
                changes = self._make_request("changes/", params)

                if not changes:
                    break  # 0件なら終了

                all_changes.extend(changes)
                logger.info(f"  Fetched {len(all_changes)} changes so far...")

                # 最後の要素に _more_changes がなければ全件取得完了
                if not changes[-1].get("_more_changes", False):
                    break

                # 次のページへ（offset をずらす）
                start += limit

            except Exception as e:
                logger.error(f"Error fetching changes: {e}")
                break

        logger.info(f"Total changes fetched: {len(all_changes)}")
        return all_changes


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  クラス 2: FeatureBuilder                                            ║
# ╠══════════════════════════════════════════════════════════════════════╣
# ║  生データ（Change のリスト）から特徴量付き DataFrame を構築するクラス。  ║
# ║                                                                      ║
# ║  処理の流れ:                                                          ║
# ║    build()                                                           ║
# ║      ├─ _extract_review_requests()  Change → レビュー依頼に展開       ║
# ║      └─ _compute_history_features() 時系列特徴量を計算               ║
# ╚══════════════════════════════════════════════════════════════════════╝

class FeatureBuilder:
    """特徴量付きデータセットを構築するクラス。"""

    def __init__(self,
                 response_window_days: int = 14,
                 bot_patterns: List[str] = None):
        """
        Args:
            response_window_days: レビュー承諾の判定ウィンドウ（日）
                                  依頼から N 日以内に応答があれば label=1（承諾）
                                  デフォルト 14 日（2週間）
            bot_patterns:         ボットと判定するメールアドレスのパターンリスト
                                  None のときは下記のデフォルトリストを使用
        """
        self.response_window_days = response_window_days

        # ボット判定パターン: これらの文字列がメールに含まれていれば自動化アカウントとして除外
        # CI/CD システム（Zuul, Jenkins）や自動化ボットはレビュアーとして意味がないため除外
        self.bot_patterns = bot_patterns or [
            'zuul', 'jenkins', 'ci@', 'bot@', 'gerrit@',
            'noreply', 'openstack-infra', 'review@',
            'gserviceaccount.com',      # Google サービスアカウント
            'appspot.gserviceaccount.com',
            'luci-project-accounts',    # LUCI（Chrome の CI）自動化
            'luci-bisection',
            'autoroll', 'auto-roller', 'automerger', 'autosubmit', 'autorerun',
            'infra-', 'test-infra-', '-infra@', 'system.gserviceaccount',
            'rubber-stamper', 'findit-for-me', 'tricium', 'chromeperf',
            '-bot@', '-robot', 'sheriffs-robots',
            'android-build-', 'android-test-infra-', 'boq-android-',
            'culprit-assistant', 'stale-change-watcher', 'presubmit-', 'workplan-finisher'
        ]

    def _is_bot(self, email: str) -> bool:
        """
        メールアドレスがボット（自動化アカウント）かどうかを判定する。

        email が None や空文字の場合も True（ボット扱い）を返す。
        → ボット・不明アカウントをすべて除外することで学習データの品質を保つ。

        Args:
            email: 判定するメールアドレス

        Returns:
            True ならボット（除外対象）、False なら人間
        """
        if not email:
            return True
        email_lower = email.lower()
        # any(): リストのどれか1つでも True なら True を返す
        return any(pattern in email_lower for pattern in self.bot_patterns)

    def _parse_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """
        Gerrit の日時文字列を Python の datetime オブジェクトに変換する。

        ■ Gerrit の日時フォーマット
        "2023-05-15 10:30:00.000000000" のように小数点以下の秒がある形式。
        Python の fromisoformat() はナノ秒（.000000000）に対応していないため、
        小数点以下を切り捨ててからパースする。

        Args:
            timestamp_str: Gerrit の日時文字列

        Returns:
            datetime オブジェクト。パース失敗時は None。
        """
        if not timestamp_str:
            return None
        try:
            # 小数点以下（ナノ秒部分）を切り捨て
            if '.' in timestamp_str:
                timestamp_str = timestamp_str.split('.')[0]
            # 末尾の 'Z'（UTC を意味する）を除去して fromisoformat でパース
            return datetime.fromisoformat(timestamp_str.replace('Z', ''))
        except:
            return None

    def _extract_review_requests(self, changes: List[Dict]) -> List[Dict]:
        """
        Change データから「レビュー依頼」レコードを抽出・展開する。

        ■ なぜ展開が必要か
        1つの Change（PR）に対して複数のレビュアーがいる場合、
        「誰がそのレビューを承諾したか」を1行1レビュアーで表現する必要がある。

        例:
            Change #12345（owner: alice）
              ├─ reviewer: bob   → 1行（bob が承諾したか？ label=1/0）
              ├─ reviewer: carol → 1行（carol が承諾したか？ label=1/0）
              └─ reviewer: dave  → 1行（dave が承諾したか？ label=1/0）

        ■ ラベルの定義
        request_time（依頼時刻）から response_window_days 日以内に
        レビュアーがコメントまたは投票をしたら label=1（承諾）、
        無応答なら label=0（拒否/無視）。

        ■ レビュアーの特定方法
        2種類のレビュアーを統合する:
          1. 明示的レビュアー: Gerrit の "reviewers" フィールドに登録された人
          2. 応答したレビュアー: "messages"（コメント履歴）に登場した人
        → どちらにも該当する人は1件にまとめる（set の union を使用）

        Args:
            changes: GerritDataFetcher.fetch_changes() の出力

        Returns:
            レビュー依頼の辞書リスト（1要素 = 1レビュアーへの1依頼）
        """
        review_requests = []

        for change in tqdm(changes, desc="Extracting review requests"):
            change_id   = change.get('id', '')
            project     = change.get('project', '')
            owner_email = change.get('owner', {}).get('email', '')
            created     = self._parse_timestamp(change.get('created', ''))

            # オーナーが不明・ボット・日時不明な Change はスキップ
            if not created or not owner_email or self._is_bot(owner_email):
                continue

            # ── ファイルパス情報の取得 ──────────────────────────────────
            # Gerrit では revisions（リビジョン履歴）の中に files（変更ファイル）が入っている
            # current_revision: 最新リビジョンのハッシュ（fetch 時に CURRENT_REVISION を指定した場合に存在）
            files = []
            current_rev = change.get('current_revision')
            if current_rev and 'revisions' in change:
                rev_info = change.get('revisions', {}).get(current_rev, {})
                files = list(rev_info.get('files', {}).keys())  # ファイルパスのリスト

            insertions = change.get('insertions', 0)  # 追加行数
            deletions  = change.get('deletions', 0)   # 削除行数

            # ── メッセージからレビュアーの応答を抽出 ────────────────────
            messages = change.get('messages', [])
            # {reviewer_email: 最初の応答時刻} の辞書
            reviewers_responded = {}

            for msg in messages:
                author_email = msg.get('author', {}).get('email', '')
                msg_date     = self._parse_timestamp(msg.get('date', ''))

                if not msg_date or not author_email:
                    continue

                # オーナー以外の人間からのメッセージ = レビュー応答
                if author_email != owner_email and not self._is_bot(author_email):
                    # 最初の応答時刻のみ記録（すでに記録済みならスキップ）
                    if author_email not in reviewers_responded:
                        reviewers_responded[author_email] = msg_date

            # ── 明示的レビュアーの取得 ────────────────────────────────
            explicit_reviewers = set()
            for reviewer_info in change.get('reviewers', {}).get('REVIEWER', []):
                reviewer_email = reviewer_info.get('email', '')
                if reviewer_email and not self._is_bot(reviewer_email):
                    explicit_reviewers.add(reviewer_email)

            # 明示的レビュアー + 応答したレビュアーを統合（重複なし）
            all_reviewers = explicit_reviewers | set(reviewers_responded.keys())

            # ── 各レビュアーのレコードを生成 ─────────────────────────
            for reviewer_email in all_reviewers:
                if reviewer_email == owner_email:
                    continue  # 自分自身をレビューするケースは除外

                first_response = reviewers_responded.get(reviewer_email)
                responded      = first_response is not None

                if responded:
                    response_days       = (first_response - created).days
                    responded_in_window = response_days <= self.response_window_days
                else:
                    response_days       = None
                    responded_in_window = False

                review_request = {
                    'change_id':            change_id,
                    'project':              project,
                    'owner_email':          owner_email,
                    'reviewer_email':       reviewer_email,
                    'request_time':         created.isoformat(),
                    'label':                1 if responded_in_window else 0,  # 予測対象
                    'first_response_time':  first_response.isoformat() if first_response else None,
                    'response_latency_days': response_days,
                    'change_insertions':    insertions,
                    'change_deletions':     deletions,
                    'change_files_count':   len(files),
                }

                review_requests.append(review_request)

        return review_requests

    def build(self, changes: List[Dict]) -> pd.DataFrame:
        """
        全処理を統括して特徴量付き DataFrame を返すメインメソッド。

        処理の流れ:
            1. _extract_review_requests(): Change → レビュー依頼に1行展開
            2. is_cross_project を付与（reviewer が複数 project に登場するか）
            3. カラム順を整理して返す

        時系列特徴量（past_reviews, load, tenure 等）は common_features.py で計算するため
        このスクリプトでは生成しない。

        Args:
            changes: GerritDataFetcher.fetch_changes() の出力

        Returns:
            特徴量付き DataFrame（main() で request_time→timestamp にリネーム後 CSV 保存）
        """
        logger.info("Extracting review requests...")
        requests = self._extract_review_requests(changes)
        logger.info(f"Extracted {len(requests)} review requests")

        df = pd.DataFrame(requests)

        # is_cross_project: reviewer が複数のプロジェクトに登場するか
        project_counts = df.groupby('reviewer_email')['project'].nunique()
        df['is_cross_project'] = df['reviewer_email'].map(project_counts) > 1

        # 抽出日を追加（いつ取得したデータかを記録）
        df['extraction_date'] = datetime.now().isoformat()

        # カラム順を明示
        columns_order = [
            'change_id', 'project', 'owner_email', 'reviewer_email',
            'request_time', 'label',
            'first_response_time', 'response_latency_days',
            'change_insertions', 'change_deletions', 'change_files_count',
            'is_cross_project',
            'extraction_date',
        ]
        df = df[columns_order]

        return df


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  main(): コマンドライン引数を受け取り全体を実行するエントリーポイント    ║
# ╚══════════════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(
        description='Gerritからデータを取得し、特徴量付きCSVを生成',
    )

    parser.add_argument('--gerrit-url', required=True,
                        help='GerritサーバーのURL (例: https://review.opendev.org)')
    parser.add_argument('--project', nargs='+', required=True,
                        help='プロジェクト名（複数指定可）')
    parser.add_argument('--start-date', required=True,
                        help='開始日 (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True,
                        help='終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', required=True,
                        help='出力CSVファイルパス')
    parser.add_argument('--response-window', type=int, default=14,
                        help='レビュー応答ウィンドウ（日）（デフォルト: 14）')
    parser.add_argument('--raw-output', required=False, default=None,
                        help='整形前の生データ(JSON)を保存するパス。未指定なら保存しない')
    parser.add_argument('--raw-output-dir', required=False, default=None,
                        help='プロジェクトごとに生データ(JSON)を保存するディレクトリ。未指定なら保存しない')

    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date   = datetime.strptime(args.end_date,   '%Y-%m-%d')

    logger.info("=" * 60)
    logger.info("データセット構築パイプライン")
    logger.info("=" * 60)
    logger.info(f"Gerrit URL: {args.gerrit_url}")
    logger.info(f"Projects:   {args.project}")
    logger.info(f"Period:     {args.start_date} to {args.end_date}")
    logger.info(f"Response window: {args.response_window} days")
    logger.info(f"Output:     {args.output}")
    logger.info("=" * 60)

    # ── Step 1: Gerrit API からデータ取得 ───────────────────────────
    fetcher = GerritDataFetcher(args.gerrit_url)

    all_changes = []
    project_changes: Dict[str, List[Dict[str, Any]]] = {}
    for project in args.project:
        logger.info(f"\nFetching data for {project}...")
        changes = fetcher.fetch_changes(project, start_date, end_date)
        all_changes.extend(changes)
        project_changes[project] = changes

    logger.info(f"\nTotal changes: {len(all_changes)}")

    if not all_changes:
        logger.error("No changes found. Please check the project name and date range.")
        sys.exit(1)

    # ── Step 2（オプション）: 生データを JSON で保存 ─────────────────
    # 生データを残しておくと、特徴量の再計算やデバッグが容易になる
    if args.raw_output:
        raw_path = Path(args.raw_output)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Raw changes をJSONで保存します: {raw_path}")
        with raw_path.open('w', encoding='utf-8') as f:
            json.dump(all_changes, f, ensure_ascii=False)

    if args.raw_output_dir:
        raw_dir = Path(args.raw_output_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"プロジェクト別のRaw changesを保存します: {raw_dir}")
        for proj, changes in project_changes.items():
            if not changes:
                logger.info(f"  {proj}: 0件のためスキップ")
                continue
            safe_name = proj.replace('/', '__')
            out_path  = raw_dir / f"{safe_name}.json"
            logger.info(f"  {proj}: {out_path} に保存 ({len(changes)}件)")
            with out_path.open('w', encoding='utf-8') as f:
                json.dump(changes, f, ensure_ascii=False)

    # ── Step 3: 特徴量を構築 ────────────────────────────────────────
    builder = FeatureBuilder(response_window_days=args.response_window)
    df = builder.build(all_changes)

    # ── Step 4: カラム名を統一 ───────────────────────────────────────
    # downstream のコード（common_features.py 等）は 'email' と 'timestamp' を前提とする
    df = df.rename(columns={
        'reviewer_email': 'email',      # レビュアーのメール
        'request_time':   'timestamp',  # レビュー依頼時刻
    })

    logger.info(f"\nDataset shape: {df.shape}")
    logger.info(f"Positive labels: {(df['label'] == 1).sum()} ({(df['label'] == 1).mean()*100:.1f}%)")
    logger.info(f"Negative labels: {(df['label'] == 0).sum()} ({(df['label'] == 0).mean()*100:.1f}%)")

    # ── Step 5: CSV を保存 ───────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"\nDataset saved to: {output_path}")

    # ── Step 6: 収集パラメータを JSON に記録（再現性のため） ────────────
    # このファイルを見れば同じデータを再取得できる
    config_path = output_path.with_name('collection_config.json')
    collection_config = {
        "gerrit_url":      args.gerrit_url,
        "projects":        args.project,
        "start_date":      args.start_date,
        "end_date":        args.end_date,
        "response_window": args.response_window,
        "output":          str(output_path),
        "n_rows":          len(df),
        "n_positive":      int((df['label'] == 1).sum()),
        "n_negative":      int((df['label'] == 0).sum()),
        "collected_at":    datetime.now().isoformat(),
    }
    with config_path.open('w', encoding='utf-8') as f:
        json.dump(collection_config, f, ensure_ascii=False, indent=2)
    logger.info(f"Collection config saved to: {config_path}")

    logger.info("=" * 60)
    logger.info("完了！")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
