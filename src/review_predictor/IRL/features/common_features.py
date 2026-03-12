"""
IRLとRFで共通の特徴量抽出モジュール

このモジュールは、IRLとRFの両方で使用する特徴量を統一的に計算します。
- 計算方法の統一
- 正規化方法の統一
- データソースの統一
"""

from datetime import datetime, timedelta
from typing import Dict
import pandas as pd


# 特徴量名の定義（14次元）
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
]

ACTION_FEATURES = [
    'avg_action_intensity',
    'avg_collaboration',
    'avg_response_time',
    'avg_review_size',
]

FEATURE_NAMES = STATE_FEATURES + ACTION_FEATURES

# normalize_features で使うキャップ値（1.0 以外のもの）
_NORM_CAPS: Dict[str, float] = {
    'experience_days': 730.0,   # 2年でキャップ
    'total_changes':   500.0,
    'total_reviews':   500.0,
    'avg_activity_gap': 60.0,   # 60日でキャップ
}


def extract_common_features(
    df: pd.DataFrame,
    email: str,
    feature_start: datetime,
    feature_end: datetime,
    normalize: bool = False
) -> Dict[str, float]:
    """
    共通の特徴量を抽出（IRLとRF両方で使用）

    Args:
        df: 全データフレーム
        email: 開発者のメールアドレス
        feature_start: 特徴量計算の開始日
        feature_end: 特徴量計算の終了日
        normalize: 正規化するかどうか（デフォルト: False）

    Returns:
        特徴量の辞書
    """
    # 特徴量期間のデータを抽出
    mask = (
        (df['email'] == email) &
        (df['timestamp'] >= feature_start) &
        (df['timestamp'] < feature_end)
    )
    dev_data = df[mask].copy()

    if len(dev_data) == 0:
        # データがない場合はデフォルト値
        return _get_default_features()

    # ========================================
    # 状態特徴量（10次元）
    # ========================================

    # 1. experience_days: 経験日数
    dates = dev_data['timestamp'].sort_values()
    experience_days = (feature_end - dates.iloc[0]).days

    # 2/3. total_changes / total_reviews: 同一値（レビュー依頼数）
    total_changes = total_reviews = len(dev_data)

    # 4. recent_activity_frequency: 直近30日の活動頻度
    recent_cutoff = feature_end - timedelta(days=30)
    recent_data = dev_data[dev_data['timestamp'] >= recent_cutoff]
    recent_activity_frequency = len(recent_data) / 30.0

    # 5. avg_activity_gap: 平均活動間隔（日数）
    if len(dates) > 1:
        gaps = dates.diff().dt.total_seconds() / 86400.0  # 日数に変換
        avg_activity_gap = gaps.mean()
    else:
        avg_activity_gap = 0.0

    # 6. activity_trend: 活動トレンド（-1.0, 0.0, 1.0）
    activity_trend = _calculate_activity_trend(dates)

    # 7. collaboration_score: 協力スコア
    collaboration_score = _calculate_collaboration_score(dev_data)

    # 8. code_quality_score: コード品質スコア（承諾率ベース）
    if 'label' in dev_data.columns:
        accepted_count = (dev_data['label'] == 1).sum()
        code_quality_score = accepted_count / total_reviews if total_reviews > 0 else 0.5
    else:
        code_quality_score = 0.5

    # 9. recent_acceptance_rate: 直近30日の承諾率
    if 'label' in recent_data.columns and len(recent_data) > 0:
        recent_accepted = (recent_data['label'] == 1).sum()
        recent_acceptance_rate = recent_accepted / len(recent_data)
    else:
        recent_acceptance_rate = 0.5

    # 10. review_load: レビュー負荷（直近30日 / 平均）
    if total_reviews > 0 and experience_days > 0:
        avg_reviews_per_30days = (total_reviews / experience_days) * 30.0
        if avg_reviews_per_30days > 0:
            review_load = len(recent_data) / avg_reviews_per_30days
        else:
            review_load = 0.0
    else:
        review_load = 0.0

    # ========================================
    # 行動特徴量（4次元）
    # ========================================

    # 11. avg_action_intensity: 平均行動強度（ファイル数ベース）
    if 'change_files_count' in dev_data.columns:
        avg_action_intensity = dev_data['change_files_count'].mean() / 20.0  # 20ファイルで1.0
    else:
        avg_action_intensity = 0.1

    # 13. avg_response_time: 平均応答時間（素早さに変換）
    if 'first_response_time' in dev_data.columns and 'request_time' in dev_data.columns:
        # request_timeとfirst_response_timeから応答時間を計算
        dev_data_with_response = dev_data.dropna(subset=['first_response_time', 'request_time'])
        if len(dev_data_with_response) > 0:
            response_times = (
                pd.to_datetime(dev_data_with_response['first_response_time']) -
                pd.to_datetime(dev_data_with_response['request_time'])
            ).dt.total_seconds() / 86400.0  # 日数に変換
            avg_response_days = response_times.mean()
            # 素早さに変換（短いほど大きい値）
            avg_response_time = 1.0 / (1.0 + avg_response_days / 3.0)
        else:
            avg_response_time = 0.5
    else:
        # なければデフォルト値
        avg_response_time = 0.5

    # 14. avg_review_size: 平均レビューサイズ（ファイル数ベース）
    if 'change_files_count' in dev_data.columns:
        avg_files = dev_data['change_files_count'].mean()
        avg_review_size = min(avg_files / 10.0, 1.0)
    else:
        avg_review_size = 0.5

    # 特徴量辞書を作成
    features = {
        # 状態特徴量（10次元）
        'experience_days': experience_days,
        'total_changes': total_changes,
        'total_reviews': total_reviews,
        'recent_activity_frequency': recent_activity_frequency,
        'avg_activity_gap': avg_activity_gap,
        'activity_trend': activity_trend,
        'collaboration_score': collaboration_score,
        'code_quality_score': code_quality_score,
        'recent_acceptance_rate': recent_acceptance_rate,
        'review_load': review_load,
        # 行動特徴量（4次元）
        'avg_action_intensity': avg_action_intensity,
        'avg_collaboration': collaboration_score,
        'avg_response_time': avg_response_time,
        'avg_review_size': avg_review_size,
    }

    # 正規化が必要な場合
    if normalize:
        features = normalize_features(features)

    return features


def normalize_features(features: Dict[str, float]) -> Dict[str, float]:
    """
    特徴量を0-1の範囲に正規化（IRLの正規化ロジックと統一）

    Args:
        features: 生の特徴量辞書

    Returns:
        正規化された特徴量辞書
    """
    # activity_trend は既に -1.0～1.0 なのでキャップ不要
    return {
        k: v if k == 'activity_trend' else min(v / _NORM_CAPS.get(k, 1.0), 1.0)
        for k, v in features.items()
    }


def _calculate_activity_trend(dates: pd.Series) -> float:
    """
    活動トレンドを計算

    Args:
        dates: タイムスタンプのSeries（ソート済み）

    Returns:
        トレンド値（-1.0: 減少, 0.0: 安定, 1.0: 増加）
    """
    if len(dates) < 2:
        return 0.0

    # 前半と後半で比較
    half_point = len(dates) // 2
    ratio = len(dates[half_point:]) / len(dates[:half_point])

    if ratio > 1.2:
        return 1.0  # 増加
    elif ratio < 0.8:
        return -1.0  # 減少
    else:
        return 0.0  # 安定


def _calculate_collaboration_score(dev_data: pd.DataFrame) -> float:
    """
    協力スコアを計算

    Args:
        dev_data: 開発者のデータ

    Returns:
        協力スコア（0.0～1.0）
    """
    if 'owner_email' in dev_data.columns:
        # ユニークな協力者の数をスコア化
        unique_collaborators = dev_data['owner_email'].nunique()
        # 10人以上で1.0
        return min(unique_collaborators / 10.0, 1.0)
    else:
        return 0.5  # デフォルト値


def _get_default_features() -> Dict[str, float]:
    """デフォルトの特徴量を返す（データがない場合）"""
    features = {
        # 状態特徴量
        'experience_days': 0.0,
        'total_changes': 0.0,
        'total_reviews': 0.0,
        'recent_activity_frequency': 0.0,
        'avg_activity_gap': 0.0,
        'activity_trend': 0.0,
        'collaboration_score': 0.5,
        'code_quality_score': 0.5,
        'recent_acceptance_rate': 0.5,
        'review_load': 0.0,
        # 行動特徴量
        'avg_action_intensity': 0.5,
        'avg_collaboration': 0.5,
        'avg_response_time': 0.5,
        'avg_review_size': 0.5,
    }
    # デフォルト値は既に [0,1] 内なので normalize しても変わらない
    return features


