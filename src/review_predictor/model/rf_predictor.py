"""
Random Forest baseline for review continuation prediction.

Uses the same 14-dimensional features as IRL (via common_features.py)
to train a scikit-learn RandomForestClassifier.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from review_predictor.features.common_features import (
    FEATURE_NAMES,
    extract_common_features,
)

logger = logging.getLogger(__name__)

RANDOM_SEED = 42


def extract_features_for_window(
    df: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    project_type: str = "multi",
) -> pd.DataFrame:
    """Extract 14-dim features per developer.

    Feature period: train_start ~ train_end
    Label period: eval_start ~ eval_end
      - At least 1 acceptance -> label=1
      - All rejections -> label=0
      - No requests -> excluded

    Args:
        df: DataFrame with 'email', 'timestamp', 'label' columns.
        train_start: Feature computation start.
        train_end: Feature computation end.
        eval_start: Label window start.
        eval_end: Label window end.
        project_type: Project type identifier (unused, kept for API compat).

    Returns:
        DataFrame with feature columns + 'label' + 'email'.
    """
    # Ensure required columns
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must have a 'timestamp' column")
    if "email" not in df.columns:
        raise ValueError("DataFrame must have an 'email' column")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Developers active in feature period
    feature_mask = (df["timestamp"] >= train_start) & (df["timestamp"] < train_end)
    feature_df = df[feature_mask]
    active_emails = feature_df["email"].unique()

    # Label period data
    label_mask = (df["timestamp"] >= eval_start) & (df["timestamp"] < eval_end)
    label_df = df[label_mask]

    rows: List[Dict] = []
    for email in active_emails:
        dev_label = label_df[label_df["email"] == email]
        if len(dev_label) == 0:
            continue  # No requests in eval period -> exclude

        has_acceptance = (dev_label["label"] == 1).any()
        label = 1 if has_acceptance else 0

        features = extract_common_features(
            df, email, train_start, train_end, normalize=False
        )
        features["label"] = label
        features["email"] = email
        rows.append(features)

    if not rows:
        return pd.DataFrame(columns=FEATURE_NAMES + ["label", "email"])

    return pd.DataFrame(rows)


def prepare_rf_features(
    features_df: pd.DataFrame,
    project_type: str = "multi",
) -> tuple[np.ndarray, np.ndarray]:
    """Convert features DataFrame to (X, y) arrays.

    Args:
        features_df: Output of extract_features_for_window.
        project_type: Unused, kept for API compat.

    Returns:
        (X, y) tuple of numpy arrays.
    """
    X = features_df[FEATURE_NAMES].values.astype(float)
    y = features_df["label"].values.astype(int)
    return X, y


def train_and_evaluate_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    random_state: int = RANDOM_SEED,
    **kwargs,
) -> Dict:
    """Train RF and evaluate. Returns metrics dict with same keys as IRL metrics.json.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_eval: Evaluation features.
        y_eval: Evaluation labels.
        n_estimators: Number of trees.
        max_depth: Max tree depth (None = unlimited).
        random_state: Random seed.

    Returns:
        Dict with keys: auc_roc, auc_pr, precision, recall, f1_score,
        positive_count, negative_count, total_count.
        Returns None if evaluation fails.
    """
    if len(np.unique(y_train)) < 2:
        logger.warning("Training set has only one class; skipping RF.")
        return None
    if len(np.unique(y_eval)) < 2:
        logger.warning("Eval set has only one class; skipping RF.")
        return None

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_eval)[:, 1]

    # Find optimal threshold via F1 on eval (same approach as IRL)
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_eval, y_prob)
    f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    y_pred_binary = (y_prob >= best_threshold).astype(int)

    try:
        auc_roc_val = float(roc_auc_score(y_eval, y_prob))
    except ValueError:
        auc_roc_val = 0.0

    auc_pr_val = float(auc(recall_curve, precision_curve))

    # Feature importance from trained RF
    importance = {
        name: float(val)
        for name, val in zip(FEATURE_NAMES, clf.feature_importances_)
    }

    return {
        "auc_roc": auc_roc_val,
        "auc_pr": auc_pr_val,
        "precision": float(precision_score(y_eval, y_pred_binary, zero_division=0)),
        "recall": float(recall_score(y_eval, y_pred_binary, zero_division=0)),
        "f1_score": float(f1_score(y_eval, y_pred_binary, zero_division=0)),
        "optimal_threshold": float(best_threshold),
        "positive_count": int(y_eval.sum()),
        "negative_count": int((1 - y_eval).sum()),
        "total_count": int(len(y_eval)),
        "feature_importance": importance,
        "predictions": y_prob.tolist(),
    }


