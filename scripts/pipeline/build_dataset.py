#!/usr/bin/env python3
"""
データセット構築パイプライン

Gerrit REST APIからコードレビューデータを取得し、レビュー承諾予測のための
特徴量付きCSVデータセットを生成します。

主要な処理フロー:
1. Gerrit APIから変更（Change）データを取得
   - プロジェクト、日付範囲で絞り込み
   - ページネーションで全件取得（500件ずつ）
   - 詳細情報を含む（アカウント、ラベル、メッセージ、ファイル）

2. レビュー依頼の抽出
   - 各変更に対するレビュアーを特定
   - 明示的レビュアー（reviewersフィールド）
   - 実際に応答したレビュアー（messagesから抽出）
   - レビュー応答の有無を判定（14日以内）

3. 特徴量の計算（約65種類）
   a) 基本情報: change_id, project, owner_email, reviewer_email, request_time
   b) ラベル: label (1=承諾, 0=拒否)
   c) 履歴ベース特徴量:
      - 過去30/90/180日のレビュー数
      - レビュー負荷（7/30/180日）
      - 過去の応答率（180日）
      - オーナーの活動（30/90/180日）
      - オーナーとレビュアーの過去のやりとり（180日）
   d) パス類似度特徴量:
      - Jaccard係数（グローバル/プロジェクト）
      - Dice係数（グローバル/プロジェクト）
      - Overlap係数（グローバル）
      - Cosine類似度（グローバル）

4. CSV出力
   - 時系列順にソート
   - 特徴量とラベルを含む完全なデータセット

実装の特徴:
- ボットアカウントの自動除外（zuul, jenkins, ci@など）
- データリーク防止: 各レビュー依頼時点での履歴のみを使用
- 時系列整合性: 過去のデータのみで特徴量を計算

使用例:
    # 基本的な使い方（単一プロジェクト）
    uv run python scripts/pipeline/build_dataset.py \
        --gerrit-url https://review.opendev.org \
        --project openstack/nova \
        --start-date 2020-01-01 \
        --end-date 2024-01-01 \
        --output data/nova_dataset.csv

    # 複数プロジェクト
    uv run python scripts/pipeline/build_dataset.py \
        --gerrit-url https://review.opendev.org \
        --project openstack/nova openstack/neutron \
        --start-date 2020-01-01 \
        --end-date 2024-01-01 \
        --output data/openstack_dataset.csv

出力CSVのフォーマット:
    change_id, project, owner_email, reviewer_email, request_time,
    label, response_latency_days, insertions, deletions, files_changed,
    reviewer_past_reviews_30d, reviewer_past_reviews_90d, ...,
    path_jaccard_files_global, path_dice_files_global, ...
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import requests
from tqdm import tqdm

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GerritDataFetcher:
    """
    Gerrit REST APIからコードレビューデータを取得するクラス

    Gerrit APIの仕様:
    - REST API: /changes/ エンドポイントを使用
    - 認証: 公開プロジェクトは認証不要でアクセス可能
    - ページネーション: _more_changes フラグで次ページの有無を判定
    - XSSI保護: レスポンスの先頭に ")]}'" が付加される

    実装の詳細:
    - requests.Sessionを使用して接続を再利用（パフォーマンス向上）
    - タイムアウトを設定してハングアップを防止
    - エラーハンドリングでAPI障害に対応
    """

    def __init__(self, gerrit_url: str, timeout: int = 300):
        """
        GerritDataFetcherの初期化

        Args:
            gerrit_url: GerritサーバーのベースURL
                        例: https://review.opendev.org
            timeout: HTTPリクエストのタイムアウト（秒、デフォルト: 300）
                     大規模プロジェクトでは長めに設定
        """
        self.gerrit_url = gerrit_url.rstrip('/')  # 末尾のスラッシュを削除
        self.timeout = timeout
        self.session = requests.Session()  # 接続を再利用してパフォーマンス向上

    def _make_request(self, endpoint: str, params: Dict = None) -> Any:
        """
        Gerrit API にHTTPリクエストを送信し、JSONレスポンスを取得

        実装の詳細:
        1. エンドポイントURLを構築
        2. HTTP GETリクエストを送信
        3. Gerrit特有のXSSI保護プレフィックス ")]}'" を除去
        4. JSONをパースして返す

        Gerrit APIの特徴:
        - XSSI (Cross-Site Script Inclusion) 攻撃を防ぐため、
          すべてのJSONレスポンスの先頭に ")]}'" が付加される
        - このプレフィックスを除去しないとJSONパースに失敗する

        Args:
            endpoint: APIエンドポイント（例: "changes/"）
            params: クエリパラメータ（辞書形式）
                    例: {"q": "project:nova", "n": 500}

        Returns:
            パースされたJSONレスポンス（辞書またはリスト）

        Raises:
            Exception: APIリクエストが失敗した場合（タイムアウト、HTTPエラーなど）
        """
        # エンドポイントURLを構築（認証なしでアクセス）
        url = f"{self.gerrit_url}/{endpoint}"

        try:
            # HTTP GETリクエストを送信
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()  # HTTPエラーをチェック（4xx, 5xx）

            # レスポンスボディを取得
            content = response.text

            # Gerrit特有のXSSI保護プレフィックスを除去
            # レスポンスが ")]}'" で始まる場合、最初の4文字をスキップ
            if content.startswith(")]}'"):
                content = content[4:]

            # JSONをパースして返す
            import json
            return json.loads(content)

        except Exception as e:
            # エラーログを出力して例外を再スロー
            logger.error(f"API request failed: {url} - {e}")
            raise
    
    def fetch_changes(self, 
                      project: str, 
                      start_date: datetime, 
                      end_date: datetime,
                      limit: int = 500) -> List[Dict[str, Any]]:
        """
        変更（Change）データを取得
        
        Args:
            project: プロジェクト名
            start_date: 開始日
            end_date: 終了日
            limit: 一度に取得する最大件数
            
        Returns:
            変更データのリスト
        """
        all_changes = []
        start = 0
        
        # 日付フォーマット
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        query = f"project:{project} after:{start_str} before:{end_str}"
        
        logger.info(f"Fetching changes for {project} from {start_str} to {end_str}")
        
        while True:
            params = {
                "q": query,
                "o": ["DETAILED_ACCOUNTS", "DETAILED_LABELS", "MESSAGES", "CURRENT_REVISION", "CURRENT_FILES"],
                "n": limit,
                "start": start
            }
            
            try:
                changes = self._make_request("changes/", params)
                
                if not changes:
                    break
                
                all_changes.extend(changes)
                logger.info(f"  Fetched {len(all_changes)} changes so far...")
                
                # _more_changesがFalseまたは存在しなければ終了
                if not changes[-1].get("_more_changes", False):
                    break
                
                start += limit
                
            except Exception as e:
                logger.error(f"Error fetching changes: {e}")
                break
        
        logger.info(f"Total changes fetched: {len(all_changes)}")
        return all_changes


class FeatureBuilder:
    """特徴量を構築するクラス"""
    
    def __init__(self, 
                 response_window_days: int = 14,
                 bot_patterns: List[str] = None):
        """
        初期化
        
        Args:
            response_window_days: レビュー応答ウィンドウ（日）
            bot_patterns: ボットと判定するメールパターン
        """
        self.response_window_days = response_window_days
        self.bot_patterns = bot_patterns or [
            # 従来のパターン
            'zuul', 'jenkins', 'ci@', 'bot@', 'gerrit@',
            'noreply', 'openstack-infra', 'review@',
            # Google service accounts
            'gserviceaccount.com',
            'appspot.gserviceaccount.com',
            # LUCI automation
            'luci-project-accounts',
            'luci-bisection',
            # Auto-rollers and automation
            'autoroll',
            'auto-roller',
            'automerger',
            'autosubmit',
            'autorerun',
            # Infrastructure and system accounts
            'infra-',
            'test-infra-',
            '-infra@',
            'system.gserviceaccount',
            # Service accounts
            'rubber-stamper',
            'findit-for-me',
            'tricium',
            'chromeperf',
            # Bot-related keywords
            '-bot@',
            '-robot',
            'sheriffs-robots',
            # Build and CI services
            'android-build-',
            'android-test-infra-',
            'boq-android-',
            # Other automation
            'culprit-assistant',
            'stale-change-watcher',
            'presubmit-',
            'workplan-finisher'
        ]
        
        # 各種履歴を保持
        self.reviewer_history: Dict[str, List[Dict]] = defaultdict(list)
        self.owner_history: Dict[str, List[Dict]] = defaultdict(list)
        self.interaction_history: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
        self.path_history: Dict[str, Set[str]] = defaultdict(set)
        
    def _is_bot(self, email: str) -> bool:
        """ボットかどうか判定"""
        if not email:
            return True
        email_lower = email.lower()
        return any(pattern in email_lower for pattern in self.bot_patterns)
    
    def _parse_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """タイムスタンプをパース"""
        if not timestamp_str:
            return None
        try:
            # Gerritのタイムスタンプ形式
            if '.' in timestamp_str:
                timestamp_str = timestamp_str.split('.')[0]
            return datetime.fromisoformat(timestamp_str.replace('Z', ''))
        except:
            return None
    
    def _extract_review_requests(self, changes: List[Dict]) -> List[Dict]:
        """
        変更データからレビュー依頼を抽出
        
        Returns:
            レビュー依頼のリスト（各レビュアーへの依頼）
        """
        review_requests = []
        
        for change in tqdm(changes, desc="Extracting review requests"):
            change_id = change.get('id', '')
            project = change.get('project', '')
            owner_email = change.get('owner', {}).get('email', '')
            created = self._parse_timestamp(change.get('created', ''))
            
            if not created or not owner_email or self._is_bot(owner_email):
                continue
            
            # ファイルパス情報
            files = []
            current_rev = change.get('current_revision')
            if current_rev and 'revisions' in change:
                rev_info = change.get('revisions', {}).get(current_rev, {})
                files = list(rev_info.get('files', {}).keys())
            
            # 変更統計
            insertions = change.get('insertions', 0)
            deletions = change.get('deletions', 0)
            
            # メッセージからレビュアーと応答を抽出
            messages = change.get('messages', [])
            reviewers_responded = {}  # reviewer_email -> first_response_time
            
            for msg in messages:
                author = msg.get('author', {})
                author_email = author.get('email', '')
                msg_date = self._parse_timestamp(msg.get('date', ''))
                
                if not msg_date or not author_email:
                    continue
                
                # オーナー以外のメッセージをレビュー応答として扱う
                if author_email != owner_email and not self._is_bot(author_email):
                    if author_email not in reviewers_responded:
                        reviewers_responded[author_email] = msg_date
            
            # 明示的なレビュアー（reviewersフィールド）
            explicit_reviewers = set()
            for reviewer_info in change.get('reviewers', {}).get('REVIEWER', []):
                reviewer_email = reviewer_info.get('email', '')
                if reviewer_email and not self._is_bot(reviewer_email):
                    explicit_reviewers.add(reviewer_email)
            
            # 全レビュアー（明示的 + 応答した人）
            all_reviewers = explicit_reviewers | set(reviewers_responded.keys())
            
            for reviewer_email in all_reviewers:
                if reviewer_email == owner_email:
                    continue
                
                first_response = reviewers_responded.get(reviewer_email)
                responded = first_response is not None
                
                # 応答期限内かどうか
                if responded:
                    response_days = (first_response - created).days
                    responded_in_window = response_days <= self.response_window_days
                else:
                    response_days = None
                    responded_in_window = False
                
                review_request = {
                    'change_id': change_id,
                    'project': project,
                    'project_id': project,  # マルチプロジェクト対応: project_idを追加
                    'owner_email': owner_email,
                    'reviewer_email': reviewer_email,
                    'request_time': created.isoformat(),
                    'developer_email': reviewer_email,
                    'context_date': created.isoformat(),
                    'responded_within_days': self.response_window_days,
                    'label': 1 if responded_in_window else 0,
                    'first_response_time': first_response.isoformat() if first_response else None,
                    'response_latency_days': response_days,
                    'change_insertions': insertions,
                    'change_deletions': deletions,
                    'change_files_count': len(files),
                    'files': files,
                    'subject_len': len(change.get('subject', '')),
                    'work_in_progress': change.get('work_in_progress', False),
                }
                
                review_requests.append(review_request)
        
        return review_requests
    
    def _compute_history_features(self,
                                   requests: List[Dict],
                                   changes: List[Dict]) -> pd.DataFrame:
        """
        履歴ベースの特徴量を計算

        時間順に処理し、各依頼時点での履歴情報を使用（データリーク防止）
        """
        # 時間順にソート
        requests_sorted = sorted(requests, key=lambda x: x['request_time'])

        # 各開発者の初出現日を記録
        first_seen: Dict[str, datetime] = {}

        # マルチプロジェクト対応: 各開発者が活動したプロジェクトを追跡
        developer_projects: Dict[str, Set[str]] = defaultdict(set)
        
        # 変更データを時間順にインデックス化
        for change in changes:
            created = self._parse_timestamp(change.get('created', ''))
            owner = change.get('owner', {}).get('email', '')
            project = change.get('project', '')

            if created and owner and not self._is_bot(owner):
                if owner not in first_seen or created < first_seen[owner]:
                    first_seen[owner] = created

                # マルチプロジェクト対応: オーナーのプロジェクト履歴を記録
                developer_projects[owner].add(project)

                # メッセージからレビュアーの初出現も記録
                for msg in change.get('messages', []):
                    author = msg.get('author', {}).get('email', '')
                    msg_date = self._parse_timestamp(msg.get('date', ''))

                    if author and msg_date and not self._is_bot(author):
                        if author not in first_seen or msg_date < first_seen[author]:
                            first_seen[author] = msg_date

                        # マルチプロジェクト対応: レビュアーのプロジェクト履歴を記録
                        developer_projects[author].add(project)
        
        # 履歴を時間順に構築
        reviewer_reviews: Dict[str, List[Tuple[datetime, Dict]]] = defaultdict(list)
        owner_messages: Dict[str, List[Tuple[datetime, Dict]]] = defaultdict(list)
        interactions: Dict[Tuple[str, str], List[Tuple[datetime, Dict]]] = defaultdict(list)
        reviewer_responses: Dict[str, List[Tuple[datetime, bool]]] = defaultdict(list)
        reviewer_paths: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))  # reviewer -> project -> paths
        
        # 変更データから履歴を構築
        for change in changes:
            created = self._parse_timestamp(change.get('created', ''))
            owner = change.get('owner', {}).get('email', '')
            project = change.get('project', '')
            
            if not created or not owner:
                continue
            
            # ファイルパス
            files = []
            current_rev = change.get('current_revision')
            if current_rev and 'revisions' in change:
                rev_info = change.get('revisions', {}).get(current_rev, {})
                files = list(rev_info.get('files', {}).keys())
            
            for msg in change.get('messages', []):
                author = msg.get('author', {}).get('email', '')
                msg_date = self._parse_timestamp(msg.get('date', ''))
                
                if not author or not msg_date or self._is_bot(author):
                    continue
                
                if author != owner:
                    reviewer_reviews[author].append((msg_date, {'project': project}))
                    interactions[(owner, author)].append((msg_date, {'project': project}))
                    
                    # パス履歴を更新
                    for f in files:
                        reviewer_paths[author]['global'].add(f)
                        reviewer_paths[author][project].add(f)
                else:
                    owner_messages[author].append((msg_date, {'project': project}))
        
        # 特徴量を計算
        features_list = []
        
        for req in tqdm(requests_sorted, desc="Computing features"):
            context_date = self._parse_timestamp(req['request_time'])
            reviewer = req['reviewer_email']
            owner = req['owner_email']
            project = req['project']
            files = req.get('files', [])
            
            # 過去のレビュー数
            past_reviews_30d = sum(1 for t, _ in reviewer_reviews[reviewer] 
                                   if context_date - timedelta(days=30) <= t < context_date)
            past_reviews_90d = sum(1 for t, _ in reviewer_reviews[reviewer] 
                                   if context_date - timedelta(days=90) <= t < context_date)
            past_reviews_180d = sum(1 for t, _ in reviewer_reviews[reviewer] 
                                    if context_date - timedelta(days=180) <= t < context_date)
            
            # オーナーのメッセージ数
            owner_msgs_30d = sum(1 for t, _ in owner_messages[owner] 
                                 if context_date - timedelta(days=30) <= t < context_date)
            owner_msgs_90d = sum(1 for t, _ in owner_messages[owner] 
                                 if context_date - timedelta(days=90) <= t < context_date)
            owner_msgs_180d = sum(1 for t, _ in owner_messages[owner] 
                                  if context_date - timedelta(days=180) <= t < context_date)
            
            # インタラクション
            pair_interactions = interactions[(owner, reviewer)]
            interactions_180d = sum(1 for t, _ in pair_interactions 
                                    if context_date - timedelta(days=180) <= t < context_date)
            project_interactions_180d = sum(1 for t, d in pair_interactions 
                                            if context_date - timedelta(days=180) <= t < context_date 
                                            and d.get('project') == project)
            
            # アサインメント負荷（過去の依頼受け数を近似）
            load_7d = sum(1 for t, _ in reviewer_reviews[reviewer] 
                         if context_date - timedelta(days=7) <= t < context_date)
            load_30d = sum(1 for t, _ in reviewer_reviews[reviewer] 
                          if context_date - timedelta(days=30) <= t < context_date)
            load_180d = past_reviews_180d
            
            # 応答率（過去180日）
            responses_180d = [r for t, r in reviewer_responses[reviewer] 
                              if context_date - timedelta(days=180) <= t < context_date]
            response_rate = sum(responses_180d) / len(responses_180d) if responses_180d else 1.0
            
            # 在籍日数
            reviewer_tenure = (context_date - first_seen.get(reviewer, context_date)).days if reviewer in first_seen else 0
            owner_tenure = (context_date - first_seen.get(owner, context_date)).days if owner in first_seen else 0
            
            # 最終活動からの日数
            reviewer_activities = [t for t, _ in reviewer_reviews[reviewer] if t < context_date]
            if reviewer_activities:
                days_since_last = (context_date - max(reviewer_activities)).days
            else:
                days_since_last = 10000  # 活動履歴なし
            
            # パス類似度計算
            path_features = self._compute_path_similarity(
                reviewer, project, files, reviewer_paths, context_date
            )
            
            # マルチプロジェクト対応: is_cross_project フラグを計算
            # この時点でのレビュアーのプロジェクト数が2以上ならクロスプロジェクト
            reviewer_project_count = len(developer_projects.get(reviewer, set()))
            is_cross_project = reviewer_project_count > 1

            # 応答履歴を更新（この依頼の結果）
            responded = req['label'] == 1
            reviewer_responses[reviewer].append((context_date, responded))

            features = {
                **req,
                'days_since_last_activity': days_since_last,
                'reviewer_past_reviews_30d': past_reviews_30d,
                'reviewer_past_reviews_90d': past_reviews_90d,
                'reviewer_past_reviews_180d': past_reviews_180d,
                'owner_past_messages_30d': owner_msgs_30d,
                'owner_past_messages_90d': owner_msgs_90d,
                'owner_past_messages_180d': owner_msgs_180d,
                'owner_reviewer_past_interactions_180d': interactions_180d,
                'owner_reviewer_project_interactions_180d': project_interactions_180d,
                'owner_reviewer_past_assignments_180d': interactions_180d,  # 近似
                'reviewer_assignment_load_7d': load_7d,
                'reviewer_assignment_load_30d': load_30d,
                'reviewer_assignment_load_180d': load_180d,
                'reviewer_past_response_rate_180d': response_rate,
                'reviewer_tenure_days': reviewer_tenure,
                'owner_tenure_days': owner_tenure,
                # マルチプロジェクト対応: 新しい特徴量
                'is_cross_project': is_cross_project,
                'reviewer_project_count': reviewer_project_count,
                **path_features
            }
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def _compute_path_similarity(self,
                                  reviewer: str,
                                  project: str,
                                  files: List[str],
                                  reviewer_paths: Dict,
                                  context_date: datetime) -> Dict[str, float]:
        """パス類似度特徴量を計算"""
        
        def jaccard(set1: Set[str], set2: Set[str]) -> float:
            if not set1 or not set2:
                return 0.0
            return len(set1 & set2) / len(set1 | set2)
        
        def dice(set1: Set[str], set2: Set[str]) -> float:
            if not set1 or not set2:
                return 0.0
            return 2 * len(set1 & set2) / (len(set1) + len(set2))
        
        def overlap_coeff(set1: Set[str], set2: Set[str]) -> float:
            if not set1 or not set2:
                return 0.0
            return len(set1 & set2) / min(len(set1), len(set2))
        
        def cosine(set1: Set[str], set2: Set[str]) -> float:
            if not set1 or not set2:
                return 0.0
            import math
            return len(set1 & set2) / math.sqrt(len(set1) * len(set2))
        
        current_files = set(files)
        current_dir1 = set(f.split('/')[0] if '/' in f else f for f in files)
        current_dir2 = set('/'.join(f.split('/')[:2]) if f.count('/') >= 1 else f for f in files)
        
        # グローバルパス
        global_files = reviewer_paths[reviewer].get('global', set())
        global_dir1 = set(f.split('/')[0] if '/' in f else f for f in global_files)
        global_dir2 = set('/'.join(f.split('/')[:2]) if f.count('/') >= 1 else f for f in global_files)
        
        # プロジェクトパス
        proj_files = reviewer_paths[reviewer].get(project, set())
        proj_dir1 = set(f.split('/')[0] if '/' in f else f for f in proj_files)
        proj_dir2 = set('/'.join(f.split('/')[:2]) if f.count('/') >= 1 else f for f in proj_files)
        
        return {
            'path_overlap_files_global': len(current_files & global_files),
            'path_overlap_dir1_global': len(current_dir1 & global_dir1),
            'path_overlap_dir2_global': len(current_dir2 & global_dir2),
            'path_jaccard_files_global': jaccard(current_files, global_files),
            'path_jaccard_dir1_global': jaccard(current_dir1, global_dir1),
            'path_jaccard_dir2_global': jaccard(current_dir2, global_dir2),
            'path_overlap_files_project': len(current_files & proj_files),
            'path_overlap_dir1_project': len(current_dir1 & proj_dir1),
            'path_overlap_dir2_project': len(current_dir2 & proj_dir2),
            'path_jaccard_files_project': jaccard(current_files, proj_files),
            'path_jaccard_dir1_project': jaccard(current_dir1, proj_dir1),
            'path_jaccard_dir2_project': jaccard(current_dir2, proj_dir2),
            'path_dice_files_global': dice(current_files, global_files),
            'path_dice_dir1_global': dice(current_dir1, global_dir1),
            'path_dice_dir2_global': dice(current_dir2, global_dir2),
            'path_overlap_coeff_files_global': overlap_coeff(current_files, global_files),
            'path_overlap_coeff_dir1_global': overlap_coeff(current_dir1, global_dir1),
            'path_overlap_coeff_dir2_global': overlap_coeff(current_dir2, global_dir2),
            'path_cosine_files_global': cosine(current_files, global_files),
            'path_cosine_dir1_global': cosine(current_dir1, global_dir1),
            'path_cosine_dir2_global': cosine(current_dir2, global_dir2),
            'path_dice_files_project': dice(current_files, proj_files),
            'path_dice_dir1_project': dice(current_dir1, proj_dir1),
            'path_dice_dir2_project': dice(current_dir2, proj_dir2),
            'path_overlap_coeff_files_project': overlap_coeff(current_files, proj_files),
            'path_overlap_coeff_dir1_project': overlap_coeff(current_dir1, proj_dir1),
            'path_overlap_coeff_dir2_project': overlap_coeff(current_dir2, proj_dir2),
            'path_cosine_files_project': cosine(current_files, proj_files),
            'path_cosine_dir1_project': cosine(current_dir1, proj_dir1),
            'path_cosine_dir2_project': cosine(current_dir2, proj_dir2),
        }
    
    def build(self, changes: List[Dict]) -> pd.DataFrame:
        """
        特徴量付きデータセットを構築
        
        Args:
            changes: 変更データのリスト
            
        Returns:
            特徴量付きDataFrame
        """
        logger.info("Extracting review requests...")
        requests = self._extract_review_requests(changes)
        logger.info(f"Extracted {len(requests)} review requests")
        
        logger.info("Computing history-based features...")
        df = self._compute_history_features(requests, changes)
        
        # 不要なカラムを削除
        if 'files' in df.columns:
            df = df.drop(columns=['files'])
        
        # 抽出日を追加
        df['extraction_date'] = datetime.now().isoformat()
        
        # カラム順を整理
        columns_order = [
            'change_id', 'project', 'owner_email', 'reviewer_email',
            'request_time', 'developer_email', 'context_date',
            'responded_within_days', 'label',
            'first_response_time', 'response_latency_days',
            'first_vote_time', 'vote_latency_days',
            'days_since_last_activity',
            'reviewer_past_reviews_30d', 'reviewer_past_reviews_90d', 'reviewer_past_reviews_180d',
            'owner_past_messages_30d', 'owner_past_messages_90d', 'owner_past_messages_180d',
            'owner_reviewer_past_interactions_180d', 'owner_reviewer_project_interactions_180d',
            'owner_reviewer_past_assignments_180d',
            'reviewer_assignment_load_7d', 'reviewer_assignment_load_30d', 'reviewer_assignment_load_180d',
            'reviewer_past_response_rate_180d',
            'reviewer_tenure_days', 'owner_tenure_days',
            'change_insertions', 'change_deletions', 'change_files_count',
            'work_in_progress', 'subject_len',
            # マルチプロジェクト特徴量
            'is_cross_project', 'reviewer_project_count',
        ]
        
        # パス特徴量カラム
        path_columns = [c for c in df.columns if c.startswith('path_')]
        columns_order.extend(sorted(path_columns))
        columns_order.append('extraction_date')
        
        # 存在するカラムのみ選択
        existing_columns = [c for c in columns_order if c in df.columns]
        df = df[existing_columns]
        
        # 欠損カラムを追加（0で埋める）
        for col in columns_order:
            if col not in df.columns:
                df[col] = 0 if col not in ['first_response_time', 'first_vote_time', 'vote_latency_days'] else None
        
        return df


def main():
    parser = argparse.ArgumentParser(
        description='Gerritからデータを取得し、特徴量付きCSVを生成',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
    # OpenStack Novaプロジェクト
    uv run python scripts/pipeline/build_dataset.py \\
        --gerrit-url https://review.opendev.org \\
        --project openstack/nova \\
        --start-date 2020-01-01 \\
        --end-date 2024-01-01 \\
        --output data/nova_dataset.csv
        
    # 複数プロジェクト
    uv run python scripts/pipeline/build_dataset.py \\
        --gerrit-url https://review.opendev.org \\
        --project openstack/nova openstack/neutron \\
        --start-date 2020-01-01 \\
        --end-date 2024-01-01 \\
        --output data/openstack_multi.csv
        """
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
    
    # 日付パース
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    logger.info("=" * 60)
    logger.info("データセット構築パイプライン")
    logger.info("=" * 60)
    logger.info(f"Gerrit URL: {args.gerrit_url}")
    logger.info(f"Projects: {args.project}")
    logger.info(f"Period: {args.start_date} to {args.end_date}")
    logger.info(f"Response window: {args.response_window} days")
    logger.info(f"Output: {args.output}")
    logger.info("=" * 60)
    
    # データ取得
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

    # 生データの保存（オプション）
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
            out_path = raw_dir / f"{safe_name}.json"
            logger.info(f"  {proj}: {out_path} に保存 ({len(changes)}件)")
            with out_path.open('w', encoding='utf-8') as f:
                json.dump(changes, f, ensure_ascii=False)
    
    # 特徴量構築
    builder = FeatureBuilder(response_window_days=args.response_window)
    df = builder.build(all_changes)
    
    logger.info(f"\nDataset shape: {df.shape}")
    logger.info(f"Positive labels: {(df['label'] == 1).sum()} ({(df['label'] == 1).mean()*100:.1f}%)")
    logger.info(f"Negative labels: {(df['label'] == 0).sum()} ({(df['label'] == 0).mean()*100:.1f}%)")
    
    # 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"\nDataset saved to: {output_path}")
    logger.info("=" * 60)
    logger.info("完了！")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
