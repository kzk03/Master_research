"""
IRL と RF で共通の特徴量抽出モジュール（common_features.py）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ このモジュールの役割
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
nova_raw.csv（build_dataset.py の出力）を読み込み、
ある開発者・ある時点における「状態特徴量」と「行動特徴量」を計算する。

IRL（逆強化学習）と RF（ランダムフォレスト）の両モデルで
同じ特徴量を使うことで、モデル間の比較を公平にする。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ 特徴量の構成（合計 23 次元: state 18 + action 5）
■ path 込みなら 26 次元 (state 18 + path 3 + action 5)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
状態特徴量（18次元）: 開発者の「現在の状態」を表す
    window_tenure_days, total_changes, total_reviews,
    recent_activity_frequency, avg_activity_gap, activity_trend,
    unique_collaborator_count, overall_acceptance_rate, recent_acceptance_rate,
    recent_load_ratio_30d_all, days_since_last_activity, reciprocity_score,
    recent_load_ratio_7d_30d, core_reviewer_ratio, acceptance_rate_last10,
    active_months_ratio, response_time_trend, reviewer_owner_ratio

行動特徴量（5次元）: 開発者の「行動パターン」を表す
    avg_action_intensity, avg_change_lines, avg_response_time,
    avg_review_size, repeat_collaboration_rate

■ 2026-05 改訂
  - 削除: acceptance_trend, recent_rejection_streak, complex_pr_bias
  - 追加: reviewer_owner_ratio (total_reviews / (total_reviews + total_changes))
  - 修正: activity_trend を時間ベース二分割に（旧件数二分割は実質常時 0 だった）
          response_time_trend を窓ベース（直近30日 vs 全期間）に
  - log1p 拡張: avg_activity_gap, avg_change_lines, days_since_last_activity

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ データリーク防止
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
extract_common_features() は feature_start〜feature_end の範囲のデータのみを
参照する。予測対象の依頼時刻より未来のデータは使わない。
"""

from datetime import datetime, timedelta
from typing import Dict
import math
import pandas as pd

from review_predictor.IRL.features.path_features import PATH_FEATURE_NAMES


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 特徴量名の定義
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 状態特徴量（18次元）: 開発者の現在の状態を表す特徴量
# 2026-05 改訂:
#   - 削除: acceptance_trend（overall/recent から LSTM が学習可能）,
#           recent_rejection_streak（acceptance_rate_last10 で代替）,
#           complex_pr_bias（窓内中央値ベースで不安定 & 重要度最下位）
#   - 追加: reviewer_owner_ratio（レビュアー寄り vs 投稿者寄りの役割バランス）
#   - 修正: activity_trend を時間ベース二分割に（旧: 件数二分割で常に≈0 のバグ）
#           response_time_trend を窓ベース（直近30日 vs 全期間）に
STATE_FEATURES = [
    'window_tenure_days',
    'total_changes',
    'total_reviews',
    'recent_activity_frequency',
    'avg_activity_gap',
    'activity_trend',
    'unique_collaborator_count',
    'overall_acceptance_rate',
    'recent_acceptance_rate',
    'recent_load_ratio_30d_all',
    'days_since_last_activity',
    'reciprocity_score',
    'recent_load_ratio_7d_30d',
    'core_reviewer_ratio',
    'acceptance_rate_last10',
    'active_months_ratio',
    'response_time_trend',
    'reviewer_owner_ratio',             # NEW: total_reviews / (total_reviews + total_changes)
]

# 行動特徴量（5次元）: 開発者の行動パターンを表す特徴量
ACTION_FEATURES = [
    'avg_action_intensity',
    'avg_change_lines',
    'avg_response_time',
    'avg_review_size',
    'repeat_collaboration_rate',
]

# 入力ベクトルのリスト生成
FEATURE_NAMES = STATE_FEATURES + ACTION_FEATURES
STATE_FEATURES_WITH_PATH = STATE_FEATURES + PATH_FEATURE_NAMES  # 23次元
FEATURE_NAMES_WITH_PATH = STATE_FEATURES_WITH_PATH + ACTION_FEATURES  # 28次元

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 正規化キャップ値
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_NORM_CAPS: Dict[str, float] = {
    'window_tenure_days':  180.0,  # 学習窓サイズ（180日）に合わせる
    'total_changes':     500.0,   # log1p 変換するため、このキャップは _LOG1P_FEATURES 経由で上書きされる
    'total_reviews':     500.0,   # 同上
    'total_activity':   1000.0,   # total_changes + total_reviews のキャップ
    'recent_load_ratio_30d_all':         3.0,  # 3倍でキャップ
    'recent_load_ratio_7d_30d':          3.0,  # 3倍でキャップ
    'core_reviewer_ratio': 0.2,  # 20%でキャップ
}

# 対数変換する特徴量（右裾が重いカウント／時間系）
# log1p(x) / log1p(cap) で 0〜1 に正規化する
_LOG1P_FEATURES: Dict[str, float] = {
    'total_changes':            500.0,   # log1p(500) ≈ 6.22 で正規化
    'total_reviews':            500.0,   # 同上
    'avg_activity_gap':         120.0,   # 旧: 線形 cap 120 → log1p 化（短い間隔の解像度向上）
    'avg_change_lines':        2000.0,   # 旧: 線形 cap 2000 → log1p 化（行数は右裾が重い）
    'days_since_last_activity': 730.0,   # 旧: 線形 cap 730 → log1p 化（短期休止と長期不在を区別）
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# メイン関数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def extract_common_features(
    df: pd.DataFrame,
    email: str,
    feature_start: datetime,
    feature_end: datetime,
    normalize: bool = False,
    total_project_reviews: int = 0,
) -> Dict[str, float]:
    """
    ある開発者・ある時点における共通特徴量を計算する。

    ■ 使い方のイメージ
    レビュー依頼 X（依頼時刻: T）の特徴量を計算したい場合:
        feature_end   = T           （予測時点）
        feature_start = T - 180日   （参照する過去の期間）

    → T より未来のデータは使わないのでデータリークが発生しない。

    Args:
        df:             nova_raw.csv を読み込んだ DataFrame
                        必須カラム: email, timestamp, label, owner_email
        email:          特徴量を計算したい開発者のメールアドレス
        feature_start:  参照する過去データの開始日時
        feature_end:    参照する過去データの終了日時（= 予測時点）
        normalize:      True にすると 0〜1 に正規化して返す

    Returns:
        特徴量名 → 値 の辞書（20次元）
    """
    # ── reviewer としての行（レビュー依頼を受けた側）を絞り込む ──────
    # email カラムは reviewer のメール（build_dataset.py で reviewer_email → email にリネーム済み）
    mask = (
        (df['email'] == email) &
        (df['timestamp'] >= feature_start) &
        (df['timestamp'] < feature_end)
    )
    dev_data = df[mask].copy()

    # ── owner としての行（自分が PR を作った側）を絞り込む ───────────
    # owner_email カラムは PR 作成者のメール
    if 'owner_email' in df.columns:
        owner_mask = (
            (df['owner_email'] == email) &
            (df['timestamp'] >= feature_start) &
            (df['timestamp'] < feature_end)
        )
        owner_data = df[owner_mask]
    else:
        owner_data = pd.DataFrame()

    # reviewer・owner 両方のデータが0件ならデフォルト値を返す
    if len(dev_data) == 0 and len(owner_data) == 0:
        return _get_default_features()

    # ========================================
    # 状態特徴量（10次元）
    # ========================================

    # 1. window_tenure_days: feature窓内での活動日数（窓内の最初の活動日〜feature_end）
    #    df全体を参照すると窓外の古い活動が経験値として計上されてしまうため、
    #    feature_start〜feature_end 内の初出現日を基準にする。
    #    例: 10年前に1回だけレビューした人が現役のように見えてしまうのを防ぐ。
    window_reviewer = df[
        (df['email'] == email) &
        (df['timestamp'] >= feature_start) &
        (df['timestamp'] < feature_end)
    ]['timestamp']
    window_owner = df[
        (df['owner_email'] == email) &
        (df['timestamp'] >= feature_start) &
        (df['timestamp'] < feature_end)
    ]['timestamp'] if 'owner_email' in df.columns else pd.Series(dtype='datetime64[ns]')
    window_dates = pd.concat([window_reviewer, window_owner]).dropna()
    if len(window_dates) > 0:
        first_seen_in_window = window_dates.min()
        window_tenure_days = max((feature_end - first_seen_in_window).days, 0)
    else:
        window_tenure_days = 0

    # 2. total_changes: 自分が作った PR 数（owner として）
    #    owner_data = 自分が PR 作成者である行。PR の多さ = 開発の活発さ。
    total_changes = len(owner_data)

    # 3. total_reviews: レビュー依頼を受けた数（reviewer として）
    #    dev_data = 自分がレビュアーとして依頼を受けた行。
    total_reviews = len(dev_data)

    # 3b. total_activity: プロジェクトへの全体関与度（total_changes + total_reviews）
    #     PRを出す側とレビューする側の両方を合算したプロジェクト全体への貢献量。
    total_activity = len(owner_data) + len(dev_data)

    # dev_data が空の場合は dates が使えないため、以降の計算を一部スキップする
    if len(dev_data) == 0:
        total_project_reviews_early = len(df[
            (df['timestamp'] >= feature_start) &
            (df['timestamp'] < feature_end)
        ])
        # dev_data 空（reviewer 経験ゼロ）→ owner 経験のみで reviewer_owner_ratio を決める
        # total_reviews=0 なので reviewer_owner_ratio は (0 / (0 + total_changes)) = 0.0 寄り
        reviewer_owner_ratio = 0.0 / (0.0 + total_changes + 1e-10) if total_changes > 0 else 0.5
        features = {
            'window_tenure_days':            window_tenure_days,
            'total_changes':              total_changes,
            'total_reviews':              0,
            'recent_activity_frequency':  0.0,
            'avg_activity_gap':           120.0,  # レビュー履歴なし = 非アクティブ（cap値）
            'activity_trend':             0.0,
            'unique_collaborator_count':        0.0,
            'overall_acceptance_rate':         0.5,
            'recent_acceptance_rate':     0.5,
            'recent_load_ratio_30d_all':                0.0,
            'days_since_last_activity':   730.0,  # cap値に合わせて更新
            'reciprocity_score':          0.0,
            'recent_load_ratio_7d_30d':                 0.0,
            'core_reviewer_ratio':        0.0,
            'acceptance_rate_last10':     0.5,
            'active_months_ratio':        0.0,
            'response_time_trend':        0.0,
            'reviewer_owner_ratio':       reviewer_owner_ratio,
            'avg_action_intensity':       0.0,  # レビュー履歴なし = 実績なし
            'avg_change_lines':           0.0,
            'avg_response_time':          0.5,
            'avg_review_size':            0.0,  # レビュー履歴なし = 実績なし
            'repeat_collaboration_rate':  0.0,
        }
        if normalize:
            features = normalize_features(features)
        return features

    # dev_data は空でないことが保証されている（上の早期リターン後）
    dates = dev_data['timestamp'].sort_values()

    # 4. recent_activity_frequency: 直近30日の活動頻度（件/日）
    #    直近の活発さを測る指標。レビュー負荷の計算にも使う。
    recent_cutoff = feature_end - timedelta(days=30)
    recent_data = dev_data[dev_data['timestamp'] >= recent_cutoff]
    recent_activity_frequency = len(recent_data) / 30.0

    # 5. avg_activity_gap: 活動間隔の中央値（日数）
    #    活動と活動の間隔の中央値。大きいほど「まばらにしか活動しない」。
    #    平均ではなく中央値を使うことで、長期不在期間の外れ値に引っ張られにくくする。
    if len(dates) > 1:
        gaps = dates.diff().dt.total_seconds() / 86400.0  # 秒 → 日
        avg_activity_gap = gaps.median()
    else:
        avg_activity_gap = 60.0  # 1件しかなければ計算不能 = 非アクティブとみなす（キャップ値）

    # 6. activity_trend: 活動トレンド（-1.0 / 0.0 / 1.0）
    #    期間の前半と後半の活動量を比較して増減を判定。
    #    詳細は _calculate_activity_trend() を参照。
    activity_trend = _calculate_activity_trend(dates)

    # 7. unique_collaborator_count: 協力スコア（0.0〜1.0）
    #    ユニークな協力者（owner_email）の数をスコア化。
    #    多くの人と関わるほど高スコア。詳細は _calculate_collaboration_score() を参照。
    unique_collaborator_count = _calculate_collaboration_score(dev_data)

    # ※ repeat_collaboration_rate は ACTION 特徴量だが、dev_data が必要なためここで計算する
    #    同じ owner から複数回レビュー依頼を受けた割合。
    #    unique_collaborator_count（幅）との違い: こちらは「深さ」= 信頼されて繰り返し頼まれる度合い。
    #    例: owner が5人いて3人が2回以上依頼してきた → 3/5 = 0.6
    if 'owner_email' in dev_data.columns and len(dev_data) > 0:
        owner_counts = dev_data['owner_email'].value_counts()
        repeat_collaboration_rate = float((owner_counts > 1).mean())
    else:
        repeat_collaboration_rate = 0.0

    # 9. overall_acceptance_rate: 全期間承諾率（reviewer 視点）
    #    自分がレビュアーとして依頼を受けた行のうち label=1（承諾）の割合。
    #    高いほど「依頼を受けたとき承諾しやすい」傾向のレビュアー（甘め / 受容的）。
    #    label カラムがない場合は 0.5（中立値）を返す。
    if 'label' in dev_data.columns:
        accepted_count = (dev_data['label'] == 1).sum()
        overall_acceptance_rate = accepted_count / total_reviews if total_reviews > 0 else 0.5
    else:
        overall_acceptance_rate = 0.5

    # 10. recent_acceptance_rate: 直近30日の承諾率
    #    overall_acceptance_rate の短期版。最近のパフォーマンスを反映する。
    if 'label' in recent_data.columns and len(recent_data) > 0:
        recent_accepted = (recent_data['label'] == 1).sum()
        recent_acceptance_rate = recent_accepted / len(recent_data)
    else:
        recent_acceptance_rate = 0.5  # データなしは中立値

    # 11. recent_load_ratio_30d_all: レビュー負荷（直近の活動量 / 平均的な活動量）
    #     1.0 より大きければ「平均より忙しい」、小さければ「余裕がある」。
    #     経験日数が 0 の場合（初日）は 0.0 を返す。
    if total_reviews > 0 and window_tenure_days > 0:
        avg_reviews_per_30days = (total_reviews / window_tenure_days) * 30.0
        if avg_reviews_per_30days > 0:
            recent_load_ratio_30d_all = len(recent_data) / avg_reviews_per_30days
        else:
            recent_load_ratio_30d_all = 0.0
    else:
        recent_load_ratio_30d_all = 0.0

    # 12. days_since_last_activity: 最終活動からの経過日数
    #     feature_end 時点で最後にレビューしてから何日経っているか。
    #     値が大きいほど「今は非アクティブ」。180日でキャップ。
    days_since_last_activity = (feature_end - dates.max()).days

    # 14. reciprocity_score: 相互レビュー率（0.0〜1.0）
    #     自分がレビューする owner のうち、逆に自分の PR もレビューしてくれている人の割合。
    #     「持ちつ持たれつ」の関係にある人ほど承諾しやすいという仮説。
    #     例: 自分が A・B・C をレビュー、そのうち A・B が自分の PR もレビュー → 2/3 ≒ 0.67
    #     ※ PRを出していない純粋なレビュアー（owner_data=0）は分母が0になるため
    #       reviewer側の視点で計算: 自分がレビューしたownerの中で自分もレビューを依頼したことある人の割合
    if 'owner_email' in dev_data.columns and 'owner_email' in df.columns:
        owners_i_reviewed = set(dev_data['owner_email'].dropna().unique())
        if len(owners_i_reviewed) > 0 and len(owner_data) > 0:
            # 自分がPRを出したことがある場合: 通常の相互レビュー率
            reviewers_of_my_prs = set(owner_data['email'].dropna().unique())
            mutual = owners_i_reviewed & reviewers_of_my_prs
            reciprocity_score = len(mutual) / len(owners_i_reviewed)
        elif len(owners_i_reviewed) > 0 and len(owner_data) == 0:
            # PRを出していない純粋レビュアー: 0.0ではなく、レビュー先の多様性で代替
            # （PRを出さない=相互関係がない、ではなく情報不足として中立値）
            reciprocity_score = 0.5
        else:
            reciprocity_score = 0.0
    else:
        reciprocity_score = 0.0

    # 15. recent_load_ratio_7d_30d: 負荷トレンド（直近7日 / 直近30日の平均）
    #     1.0 より大きければ「最近急に忙しくなっている」、小さければ「最近落ち着いている」。
    #     recent_load_ratio_30d_all（全期間平均との比較）より短期の変化を捉える。
    recent_7d_data = dev_data[dev_data['timestamp'] >= feature_end - timedelta(days=7)]
    load_7d_per_day  = len(recent_7d_data) / 7.0
    load_30d_per_day = len(recent_data) / 30.0
    if load_30d_per_day > 0:
        recent_load_ratio_7d_30d = load_7d_per_day / load_30d_per_day
    else:
        recent_load_ratio_7d_30d = 0.0

    # 16. core_reviewer_ratio: コアレビュアー度（0.0〜1.0）
    if total_project_reviews > 0:
        proj_total = total_project_reviews
    else:
        proj_total = len(df[
            (df['timestamp'] >= feature_start) &
            (df['timestamp'] < feature_end)
        ])
    if proj_total > 0:
        core_reviewer_ratio = total_reviews / proj_total
    else:
        core_reviewer_ratio = 0.0

    # 17. active_months_ratio: 活動した月の割合（0.0〜1.0）
    #     feature窓の全月数のうち、実際にレビュー活動があった月の割合。
    #     recent_activity_frequency（件数/日）とは異なり「定期的に活動しているか」を捉える。
    #     例: 24ヶ月窓で12ヶ月活動 → 0.5
    total_months = max(int((feature_end - feature_start).days / 30), 1)
    active_months = dev_data['timestamp'].dt.to_period('M').nunique() if len(dev_data) > 0 else 0
    active_months_ratio = min(active_months / total_months, 1.0)

    # 18. response_time_trend: 応答速度のトレンド（-1.0〜+1.0）
    #     直近30日の平均応答速度 - 全期間の平均応答速度。
    #     正 → 最近速くなっている（積極的）、負 → 最近遅くなっている（離脱傾向）
    #     旧実装は件数二分割（len(dev_data) >= 4 必須）で 96% が 0.0 集中だったため、
    #     窓ベース（全期間 vs 直近30日）に変更。データ1件でも全期間 speed は計算可能。
    response_time_trend = 0.0
    if 'first_response_time' in dev_data.columns:
        def _mean_speed(part: pd.DataFrame) -> float:
            with_resp = part.dropna(subset=['first_response_time'])
            if len(with_resp) == 0:
                return float('nan')
            rt = (pd.to_datetime(with_resp['first_response_time']) -
                  pd.to_datetime(with_resp['timestamp'])).dt.total_seconds() / 86400.0
            return 1.0 / (1.0 + rt.mean() / 3.0)
        all_speed = _mean_speed(dev_data)
        recent_speed = _mean_speed(recent_data)
        if not math.isnan(all_speed) and not math.isnan(recent_speed):
            response_time_trend = recent_speed - all_speed  # -1〜+1

    # 19. reviewer_owner_ratio: レビュアー寄り（→1.0）vs 投稿者寄り（→0.0）の役割バランス
    #     total_reviews と total_changes だけから決まる追加コストゼロ特徴量。
    #     Bird et al. (FSE 2011) の ownership 分析の発想に近い。
    if total_reviews + total_changes > 0:
        reviewer_owner_ratio = total_reviews / (total_reviews + total_changes)
    else:
        reviewer_owner_ratio = 0.5  # 両方ゼロ = 不明 → 中立値

    # acceptance_rate_last10: 直近10件の承諾率（件数ベース）
    #     日数ではなく件数で直近を定義することで、活動頻度に依存しない最新の行動を反映。
    if 'label' in dev_data.columns and len(dev_data) > 0:
        last10 = dev_data.sort_values('timestamp').tail(10)
        acceptance_rate_last10 = (last10['label'] == 1).mean()
    else:
        acceptance_rate_last10 = 0.5

    # ========================================
    # 行動特徴量（5次元）
    # ========================================

    # 12. avg_action_intensity: 平均行動強度（files × lines の合成スコア）
    #     ファイル数と変更行数を組み合わせたPR複雑度の指標。
    #     両方をキャップ正規化したうえで幾何平均を取る。
    #     avg_review_size（ファイル数のみ）と avg_change_lines（行数のみ）の補完的な合成。
    if 'change_files_count' in dev_data.columns and \
       'change_insertions' in dev_data.columns and 'change_deletions' in dev_data.columns:
        files_score = min(dev_data['change_files_count'].mean() / 20.0, 1.0)
        lines_score = min((dev_data['change_insertions'].fillna(0) +
                           dev_data['change_deletions'].fillna(0)).mean() / 500.0, 1.0)
        avg_action_intensity = (files_score * lines_score) ** 0.5  # 幾何平均
    elif 'change_files_count' in dev_data.columns:
        avg_action_intensity = min(dev_data['change_files_count'].mean() / 20.0, 1.0)
    else:
        avg_action_intensity = 0.0  # カラムなし = 実績なし

    # 13. avg_change_lines: 平均変更行数（insertions + deletions の合計）
    #     1回の Change での変更規模を行数で表す。
    #     ファイル数（avg_action_intensity）とは別軸の規模指標。
    #     500行でキャップ（_NORM_CAPS 参照）。
    if 'change_insertions' in dev_data.columns and 'change_deletions' in dev_data.columns:
        avg_change_lines = (
            dev_data['change_insertions'].fillna(0) +
            dev_data['change_deletions'].fillna(0)
        ).mean()
    else:
        avg_change_lines = 0.0

    # 14. avg_response_time: 平均応答速度（0.0〜1.0、高いほど速い）
    #     応答日数を「速さ」に変換: 1 / (1 + 日数/3)
    #     例: 0日 → 1.0、3日 → 0.5、6日 → 0.25
    #     timestamp（依頼時刻）と first_response_time（応答時刻）の差分を使う。
    if 'first_response_time' in dev_data.columns and 'timestamp' in dev_data.columns:
        dev_data_with_response = dev_data.dropna(subset=['first_response_time'])
        if len(dev_data_with_response) > 0:
            response_times = (
                pd.to_datetime(dev_data_with_response['first_response_time']) -
                pd.to_datetime(dev_data_with_response['timestamp'])
            ).dt.total_seconds() / 86400.0  # 秒 → 日
            avg_response_days = response_times.mean()
            avg_response_time = 1.0 / (1.0 + avg_response_days / 3.0)
        else:
            avg_response_time = 0.5  # 応答データなしは中立値
    else:
        avg_response_time = 0.5

    # 15. avg_review_size: 平均レビューサイズ（0.0〜1.0）
    #     1回の Change の平均ファイル数を 10 で割って 1.0 にキャップ。
    #     avg_action_intensity との違い: こちらは min(..., 1.0) で上限を設ける。
    if 'change_files_count' in dev_data.columns:
        avg_files = dev_data['change_files_count'].mean()
        avg_review_size = min(avg_files / 10.0, 1.0)
    else:
        avg_review_size = 0.0  # カラムなし = 実績なし

    # ── 特徴量辞書を組み立てる ───────────────────────────────────────
    features = {
        # 状態特徴量（18次元）
        'window_tenure_days':            window_tenure_days,
        'total_changes':              total_changes,
        'total_reviews':              total_reviews,
        'recent_activity_frequency':  recent_activity_frequency,
        'avg_activity_gap':           avg_activity_gap,
        'activity_trend':             activity_trend,
        'unique_collaborator_count':        unique_collaborator_count,
        'overall_acceptance_rate':         overall_acceptance_rate,
        'recent_acceptance_rate':     recent_acceptance_rate,
        'recent_load_ratio_30d_all':                recent_load_ratio_30d_all,
        'days_since_last_activity':   days_since_last_activity,
        'reciprocity_score':          reciprocity_score,
        'recent_load_ratio_7d_30d':                 recent_load_ratio_7d_30d,
        'core_reviewer_ratio':        core_reviewer_ratio,
        'acceptance_rate_last10':     acceptance_rate_last10,
        'active_months_ratio':        active_months_ratio,
        'response_time_trend':        response_time_trend,
        'reviewer_owner_ratio':       reviewer_owner_ratio,
        # 行動特徴量（5次元）
        'avg_action_intensity':       avg_action_intensity,
        'avg_change_lines':           avg_change_lines,
        'avg_response_time':          avg_response_time,
        'avg_review_size':            avg_review_size,
        'repeat_collaboration_rate':  repeat_collaboration_rate,
    }

    if normalize:
        features = normalize_features(features)

    return features


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 正規化
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def normalize_features(features: Dict[str, float]) -> Dict[str, float]:
    """
    特徴量を 0〜1 の範囲に正規化する。

    ■ 正規化の方法
    - 通常: value / cap でキャップ割りし、1.0 を超えた分は 1.0 にクリップ。
    - カウント系 (_LOG1P_FEATURES): log1p(value) / log1p(cap) で対数変換後に正規化。
      右裾が重い分布（少数の高活動者が支配する）を圧縮し、低〜中活動者の識別性を向上。
    キャップ値は _NORM_CAPS / _LOG1P_FEATURES に定義。未定義の特徴量は cap=1.0（そのまま）。

    ■ 例外
    activity_trend 等は -1.0〜1.0 の値をとるため正規化対象外。
    そのまま返す。

    Args:
        features: 生の特徴量辞書（extract_common_features の出力）

    Returns:
        正規化された特徴量辞書（値は 0〜1、ただし activity_trend のみ -1〜1）
    """
    # activity_trend / response_time_trend は -1.0〜1.0 の値なのでキャップ不要
    _skip_norm = {'activity_trend', 'response_time_trend'}
    result = {}
    for k, v in features.items():
        if k in _skip_norm:
            result[k] = v
        elif k in _LOG1P_FEATURES:
            # 対数変換: log1p(x) / log1p(cap) → 0〜1
            cap = _LOG1P_FEATURES[k]
            result[k] = min(math.log1p(v) / math.log1p(cap), 1.0)
        else:
            result[k] = min(v / _NORM_CAPS.get(k, 1.0), 1.0)
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ヘルパー関数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _calculate_activity_trend(dates: pd.Series) -> float:
    """
    活動トレンドを計算する（-1.0〜+1.0 の連続値）。

    ■ 計算方法（時間ベース二分割）
    観測期間 [t_min, t_max] の中点 t_mid で時間軸を二分し、件数差を合計で正規化:
        (後半件数 - 前半件数) / (後半件数 + 前半件数)
    → +1.0: 全て後半（時間後半に集中、増加傾向）
    → -1.0: 全て前半（減少傾向）
    →  0.0: 均等（安定）

    旧実装は件数二分割（前半 = dates[:n//2]）だったが、ソート済み列に対して
    そのまま len で数えると常に half ≈ second_half になり、結果が偶奇に依存する
    バグ的な振る舞いをしていた（実分布で 88% が 0.0 に集中）。時間ベースに変えると
    件数が少なくても観測期間内の偏りが連続値として現れる。

    Args:
        dates: タイムスタンプの Series（ソート済みであること）

    Returns:
        -1.0〜+1.0 の連続値
    """
    if len(dates) < 2:
        return 0.0  # 1件では比較不能

    t_min = dates.min()
    t_max = dates.max()
    if t_max == t_min:
        return 0.0  # 全件同時刻 = 比較不能
    mid_time = t_min + (t_max - t_min) / 2
    first_half = int((dates < mid_time).sum())
    second_half = int((dates >= mid_time).sum())
    total = first_half + second_half
    if total == 0:
        return 0.0
    return (second_half - first_half) / total


def _calculate_collaboration_score(dev_data: pd.DataFrame) -> float:
    """
    協力スコアを計算する（0.0〜1.0）。

    ■ 計算方法
    期間内にレビュー依頼を受けた PR のオーナー（owner_email）の
    ユニーク数を数え、10人以上で 1.0 にキャップ。

    「多様な人から依頼を受けている = コミュニティの広さ」を表す。

    Args:
        dev_data: 対象開発者・対象期間の DataFrame

    Returns:
        協力スコア（0.0〜1.0）。owner_email カラムがなければ 0.5。
    """
    if 'owner_email' in dev_data.columns:
        unique_collaborators = dev_data['owner_email'].nunique()
        return min(unique_collaborators / 10.0, 1.0)  # 10人で 1.0
    else:
        return 0.5  # カラムなしは中立値


def _get_default_features() -> Dict[str, float]:
    """
    データが0件の開発者に対して返すデフォルト特徴量。

    ■ デフォルト値の方針
    - 活動量系（経験日数・件数・頻度）: 0.0（実績なし）
    - 率・スコア系（承諾率・品質・協力）: 0.5（不明 = 中立）
    これは「実績がない = 中程度の期待値」という仮定に基づく。
    """
    return {
        # 状態特徴量
        'window_tenure_days':            0.0,
        'total_changes':              0.0,
        'total_reviews':              0.0,
        'recent_activity_frequency':  0.0,
        'avg_activity_gap':           120.0,  # 活動なし = 非アクティブ（cap値）
        'activity_trend':             0.0,
        'unique_collaborator_count':        0.0,   # 活動なし = 協力者なし
        'overall_acceptance_rate':         0.5,
        'recent_acceptance_rate':     0.5,
        'recent_load_ratio_30d_all':                0.0,
        'days_since_last_activity':   730.0,  # cap値に合わせて更新
        'reciprocity_score':          0.0,
        'recent_load_ratio_7d_30d':                 0.0,
        'core_reviewer_ratio':        0.0,
        'acceptance_rate_last10':     0.5,
        'active_months_ratio':        0.0,
        'response_time_trend':        0.0,
        'reviewer_owner_ratio':       0.5,   # 両方ゼロ = 不明 → 中立値
        # 行動特徴量
        'avg_action_intensity':       0.0,   # 活動なし = 実績なし
        'avg_change_lines':           0.0,
        'avg_response_time':          0.5,
        'avg_review_size':            0.0,   # 活動なし = 実績なし
        'repeat_collaboration_rate':  0.0,
    }
