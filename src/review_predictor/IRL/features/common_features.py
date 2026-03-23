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
■ 特徴量の構成（合計 20 次元）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
状態特徴量（15次元）: 開発者の「現在の状態」を表す
    experience_days              経験日数
    total_changes                自分が作った PR 数（owner として）
    total_reviews                レビュー依頼を受けた数（reviewer として）
    recent_activity_frequency    直近30日の活動頻度（件/日）
    avg_activity_gap             平均活動間隔（日）
    activity_trend               活動トレンド（-1.0/0.0/1.0）
    collaboration_score          協力スコア（ユニーク協力者数ベース = 幅）
    code_quality_score           コード品質スコア（承諾率ベース）
    recent_acceptance_rate       直近30日の承諾率
    review_load                  レビュー負荷（直近 / 平均）
    days_since_last_activity     最終活動からの経過日数
    acceptance_trend             承諾率のトレンド（直近 - 全期間）
    reciprocity_score            相互レビュー率（自分がレビューする人が逆に自分をレビューしている割合）
    load_trend                   負荷トレンド（直近7日 / 直近30日の平均）
    core_reviewer_ratio          コアレビュアー度（自分のレビュー数 / プロジェクト全体のレビュー数）

行動特徴量（5次元）: 開発者の「行動パターン」を表す
    avg_action_intensity         平均行動強度（変更ファイル数ベース）
    avg_change_lines             平均変更行数（insertions + deletions）
    avg_response_time            平均応答速度（短いほど大きい値）
    avg_review_size              平均レビューサイズ（ファイル数ベース）
    repeat_collaboration_rate    リピートコラボレーション率（同じ owner と複数回やりとり = 深さ）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ データリーク防止
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
extract_common_features() は feature_start〜feature_end の範囲のデータのみを
参照する。予測対象の依頼時刻より未来のデータは使わない。
"""

from datetime import datetime, timedelta
from typing import Dict
import pandas as pd


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 特徴量名の定義
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 状態特徴量（15次元）: 開発者の現在の状態を表す特徴量
STATE_FEATURES = [
    'experience_days',
    'total_changes',
    'total_reviews',
    'recent_activity_frequency',
    'avg_activity_gap',
    'activity_trend',
    'collaboration_score',
    'code_quality_score',
    'recent_acceptance_rate',
    'review_load',
    'days_since_last_activity',
    'acceptance_trend',
    'reciprocity_score',
    'load_trend',
    'core_reviewer_ratio',
]

# 行動特徴量（5次元）: 開発者の行動パターンを表す特徴量
ACTION_FEATURES = [
    'avg_action_intensity',
    'avg_change_lines',
    'avg_response_time',
    'avg_review_size',
    'repeat_collaboration_rate',
]

# モデルへの入力ベクトル（STATE + ACTION の順）
FEATURE_NAMES = STATE_FEATURES + ACTION_FEATURES

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 正規化キャップ値
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# normalize_features() で各特徴量を 0〜1 に収めるときの上限値。
# 指定がない特徴量のキャップは 1.0（= そのまま使う）。
# 例: experience_days=365 のとき、730.0 でキャップすると 0.5 になる。
_NORM_CAPS: Dict[str, float] = {
    'experience_days':  730.0,   # 2年でキャップ（それ以上は十分な経験者とみなす）
    'total_changes':    500.0,   # 500件でキャップ
    'total_reviews':    500.0,   # 500件でキャップ
    'avg_activity_gap':          60.0,   # 60日でキャップ（それ以上は非アクティブとみなす）
    'avg_change_lines':         500.0,   # 500行でキャップ（insertions + deletions の合計）
    'days_since_last_activity': 180.0,   # 180日でキャップ（半年以上は非アクティブとみなす）
    'load_trend':                 3.0,   # 3倍でキャップ（直近7日が30日平均の3倍以上は最大負荷）
    'core_reviewer_ratio':        0.2,   # 20%でキャップ（全レビューの20%以上を担うと最大のコアメンバー）
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

    # 1. experience_days: 経験日数
    #    DataFrame 全体（feature_start より前も含む）での初出現日から feature_end までの日数。
    #    feature_start 内の最初の活動日を使うと実際の経験より短く見積もられるため、
    #    df 全体から email の初出現日を取得する。
    all_reviewer = df[df['email'] == email]['timestamp']
    all_owner = df[df.get('owner_email', pd.Series(dtype=str)) == email]['timestamp'] if 'owner_email' in df.columns else pd.Series(dtype='datetime64[ns]')
    all_dates = pd.concat([all_reviewer, all_owner]).dropna()
    if len(all_dates) > 0:
        first_seen = all_dates.min()
        experience_days = max((feature_end - first_seen).days, 0)
    else:
        experience_days = 0

    # 2. total_changes: 自分が作った PR 数（owner として）
    #    owner_data = 自分が PR 作成者である行。PR の多さ = 開発の活発さ。
    total_changes = len(owner_data)

    # 3. total_reviews: レビュー依頼を受けた数（reviewer として）
    #    dev_data = 自分がレビュアーとして依頼を受けた行。
    total_reviews = len(dev_data)

    # dev_data が空の場合は dates が使えないため、以降の計算を一部スキップする
    if len(dev_data) == 0:
        total_project_reviews_early = len(df[
            (df['timestamp'] >= feature_start) &
            (df['timestamp'] < feature_end)
        ])
        features = {
            'experience_days':            experience_days,
            'total_changes':              total_changes,
            'total_reviews':              0,
            'recent_activity_frequency':  0.0,
            'avg_activity_gap':           0.0,
            'activity_trend':             0.0,
            'collaboration_score':        0.0,
            'code_quality_score':         0.5,
            'recent_acceptance_rate':     0.5,
            'review_load':                0.0,
            'days_since_last_activity':   180.0,
            'acceptance_trend':           0.0,
            'reciprocity_score':          0.0,
            'load_trend':                 0.0,
            'core_reviewer_ratio':        0.0,
            'avg_action_intensity':       0.1,
            'avg_change_lines':           0.0,
            'avg_response_time':          0.5,
            'avg_review_size':            0.5,
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

    # 5. avg_activity_gap: 平均活動間隔（日数）
    #    活動と活動の間隔の平均。大きいほど「まばらにしか活動しない」。
    #    diff() で隣り合う活動日の差分を取り、秒→日に変換。
    if len(dates) > 1:
        gaps = dates.diff().dt.total_seconds() / 86400.0  # 秒 → 日
        avg_activity_gap = gaps.mean()
    else:
        avg_activity_gap = 0.0  # 1件しかなければ間隔は計算不能

    # 6. activity_trend: 活動トレンド（-1.0 / 0.0 / 1.0）
    #    期間の前半と後半の活動量を比較して増減を判定。
    #    詳細は _calculate_activity_trend() を参照。
    activity_trend = _calculate_activity_trend(dates)

    # 7. collaboration_score: 協力スコア（0.0〜1.0）
    #    ユニークな協力者（owner_email）の数をスコア化。
    #    多くの人と関わるほど高スコア。詳細は _calculate_collaboration_score() を参照。
    collaboration_score = _calculate_collaboration_score(dev_data)

    # ※ repeat_collaboration_rate は ACTION 特徴量だが、dev_data が必要なためここで計算する
    #    同じ owner から複数回レビュー依頼を受けた割合。
    #    collaboration_score（幅）との違い: こちらは「深さ」= 信頼されて繰り返し頼まれる度合い。
    #    例: owner が5人いて3人が2回以上依頼してきた → 3/5 = 0.6
    if 'owner_email' in dev_data.columns and len(dev_data) > 0:
        owner_counts = dev_data['owner_email'].value_counts()
        repeat_collaboration_rate = float((owner_counts > 1).mean())
    else:
        repeat_collaboration_rate = 0.0

    # 9. code_quality_score: コード品質スコア（承諾率ベース）
    #    label=1（承諾）の割合。高いほど「受け入れられやすいコードを書く」開発者。
    #    label カラムがない場合は 0.5（中立値）を返す。
    if 'label' in dev_data.columns:
        accepted_count = (dev_data['label'] == 1).sum()
        code_quality_score = accepted_count / total_reviews if total_reviews > 0 else 0.5
    else:
        code_quality_score = 0.5

    # 10. recent_acceptance_rate: 直近30日の承諾率
    #    code_quality_score の短期版。最近のパフォーマンスを反映する。
    if 'label' in recent_data.columns and len(recent_data) > 0:
        recent_accepted = (recent_data['label'] == 1).sum()
        recent_acceptance_rate = recent_accepted / len(recent_data)
    else:
        recent_acceptance_rate = 0.5  # データなしは中立値

    # 11. review_load: レビュー負荷（直近の活動量 / 平均的な活動量）
    #     1.0 より大きければ「平均より忙しい」、小さければ「余裕がある」。
    #     経験日数が 0 の場合（初日）は 0.0 を返す。
    if total_reviews > 0 and experience_days > 0:
        avg_reviews_per_30days = (total_reviews / experience_days) * 30.0
        if avg_reviews_per_30days > 0:
            review_load = len(recent_data) / avg_reviews_per_30days
        else:
            review_load = 0.0
    else:
        review_load = 0.0

    # 12. days_since_last_activity: 最終活動からの経過日数
    #     feature_end 時点で最後にレビューしてから何日経っているか。
    #     値が大きいほど「今は非アクティブ」。180日でキャップ。
    days_since_last_activity = (feature_end - dates.max()).days

    # 13. acceptance_trend: 承諾率のトレンド（-1.0〜1.0）
    #     直近30日の承諾率 - 全期間の承諾率。
    #     正 → 最近の方が承諾率が上昇（勢いあり）
    #     負 → 最近の方が承諾率が低下（疲弊・忙しい）
    #     既存の2特徴量から追加コストゼロで計算できる。
    acceptance_trend = recent_acceptance_rate - code_quality_score

    # 14. reciprocity_score: 相互レビュー率（0.0〜1.0）
    #     自分がレビューする owner のうち、逆に自分の PR もレビューしてくれている人の割合。
    #     「持ちつ持たれつ」の関係にある人ほど承諾しやすいという仮説。
    #     例: 自分が A・B・C をレビュー、そのうち A・B が自分の PR もレビュー → 2/3 ≒ 0.67
    if 'owner_email' in dev_data.columns and 'owner_email' in df.columns:
        owners_i_reviewed = set(dev_data['owner_email'].dropna().unique())
        reviewers_of_my_prs = set(
            df[
                (df['owner_email'] == email) &
                (df['timestamp'] >= feature_start) &
                (df['timestamp'] < feature_end)
            ]['email'].dropna().unique()
        )
        if len(owners_i_reviewed) > 0:
            mutual = owners_i_reviewed & reviewers_of_my_prs
            reciprocity_score = len(mutual) / len(owners_i_reviewed)
        else:
            reciprocity_score = 0.0
    else:
        reciprocity_score = 0.0

    # 15. load_trend: 負荷トレンド（直近7日 / 直近30日の平均）
    #     1.0 より大きければ「最近急に忙しくなっている」、小さければ「最近落ち着いている」。
    #     review_load（全期間平均との比較）より短期の変化を捉える。
    recent_7d_data = dev_data[dev_data['timestamp'] >= feature_end - timedelta(days=7)]
    load_7d_per_day  = len(recent_7d_data) / 7.0
    load_30d_per_day = len(recent_data) / 30.0
    if load_30d_per_day > 0:
        load_trend = load_7d_per_day / load_30d_per_day
    else:
        load_trend = 0.0

    # 16. core_reviewer_ratio: コアレビュアー度（0.0〜1.0）
    #     プロジェクト全体のレビュー依頼数に占める、この開発者へのレビュー依頼数の割合。
    #     値が大きいほど「プロジェクトの中核を担うレビュアー」。
    #     コアメンバーはプロジェクトへのコミットが強く、離脱しにくいという仮説。
    #     例: 全体1000件中自分に80件来ている → 0.08（0.2でキャップすると0.4に正規化される）
    #     total_project_reviews が指定されている場合はそれを使用（IRL個人履歴df用）、
    #     そうでなければ df 全体のレビュー数から計算（RF用 full_df が渡される場合）。
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

    # ========================================
    # 行動特徴量（5次元）
    # ========================================

    # 12. avg_action_intensity: 平均行動強度
    #     1回の Change で変更したファイル数の平均。20ファイルで 1.0 に正規化。
    #     ファイル数が多いほど「大きな変更」を意味する。
    #     min(..., 1.0) で上限を設けて avg_review_size と統一。
    if 'change_files_count' in dev_data.columns:
        avg_action_intensity = min(dev_data['change_files_count'].mean() / 20.0, 1.0)
    else:
        avg_action_intensity = 0.1

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
        avg_review_size = 0.5

    # ── 特徴量辞書を組み立てる ───────────────────────────────────────
    features = {
        # 状態特徴量（10次元）
        'experience_days':            experience_days,
        'total_changes':              total_changes,
        'total_reviews':              total_reviews,
        'recent_activity_frequency':  recent_activity_frequency,
        'avg_activity_gap':           avg_activity_gap,
        'activity_trend':             activity_trend,
        'collaboration_score':        collaboration_score,
        'code_quality_score':         code_quality_score,
        'recent_acceptance_rate':     recent_acceptance_rate,
        'review_load':                review_load,
        'days_since_last_activity':   days_since_last_activity,
        'acceptance_trend':           acceptance_trend,
        'reciprocity_score':          reciprocity_score,
        'load_trend':                 load_trend,
        'core_reviewer_ratio':        core_reviewer_ratio,
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
    value / cap でキャップ割りし、1.0 を超えた分は 1.0 にクリップ。
    キャップ値は _NORM_CAPS に定義。未定義の特徴量は cap=1.0（そのまま）。

    ■ 例外
    activity_trend は -1.0〜1.0 の値をとるため正規化対象外。
    そのまま返す。

    Args:
        features: 生の特徴量辞書（extract_common_features の出力）

    Returns:
        正規化された特徴量辞書（値は 0〜1、ただし activity_trend のみ -1〜1）
    """
    # activity_trend と acceptance_trend は -1.0〜1.0 の値なのでキャップ不要
    _skip_norm = {'activity_trend', 'acceptance_trend'}
    return {
        k: v if k in _skip_norm else min(v / _NORM_CAPS.get(k, 1.0), 1.0)
        for k, v in features.items()
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ヘルパー関数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _calculate_activity_trend(dates: pd.Series) -> float:
    """
    活動トレンドを計算する（-1.0 / 0.0 / 1.0 の3値）。

    ■ 計算方法
    期間を前半・後半に2分割し、件数の比（後半 / 前半）で判定:
        ratio > 1.2 → 1.0（増加傾向）
        ratio < 0.8 → -1.0（減少傾向）
        それ以外   → 0.0（安定）

    Args:
        dates: タイムスタンプの Series（ソート済みであること）

    Returns:
        -1.0（減少） / 0.0（安定） / 1.0（増加）
    """
    if len(dates) < 2:
        return 0.0  # 1件では比較不能

    half_point = len(dates) // 2
    # 前半の件数に対する後半の件数の比を計算
    ratio = len(dates[half_point:]) / len(dates[:half_point])

    if ratio > 1.2:
        return 1.0   # 後半の方が2割以上多い → 増加
    elif ratio < 0.8:
        return -1.0  # 後半の方が2割以上少ない → 減少
    else:
        return 0.0   # ±20% 以内 → 安定


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
        'experience_days':            0.0,
        'total_changes':              0.0,
        'total_reviews':              0.0,
        'recent_activity_frequency':  0.0,
        'avg_activity_gap':           0.0,
        'activity_trend':             0.0,
        'collaboration_score':        0.5,
        'code_quality_score':         0.5,
        'recent_acceptance_rate':     0.5,
        'review_load':                0.0,
        'days_since_last_activity':   180.0,
        'acceptance_trend':           0.0,
        'reciprocity_score':          0.0,
        'load_trend':                 0.0,
        'core_reviewer_ratio':        0.0,
        # 行動特徴量
        'avg_action_intensity':       0.5,
        'avg_change_lines':           0.0,
        'avg_response_time':          0.5,
        'avg_review_size':            0.5,
        'repeat_collaboration_rate':  0.0,
    }
