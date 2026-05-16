"""Metric and cross-validation helpers used across the notebooks."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

RANDOM_STATE = 42

CV_SCORING = {
    "accuracy": "accuracy",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "f1_macro": "f1_macro",
    "roc_auc": "roc_auc",
}


def stratified_cv(n_splits: int = 5, random_state: int = RANDOM_STATE) -> StratifiedKFold:
    """Return a StratifiedKFold that all models share for fair comparison."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def cv_summary(
    estimator: BaseEstimator,
    X,
    y,
    *,
    cv: StratifiedKFold | None = None,
    scoring: Mapping[str, str] = CV_SCORING,
) -> pd.DataFrame:
    """Run cross-validation and return mean and std for each metric.

    The returned DataFrame has one row per metric with columns ``mean`` and
    ``std``, which makes it easy to compare models side by side.
    """
    cv = cv or stratified_cv()
    results = cross_validate(
        estimator,
        X,
        y,
        cv=cv,
        scoring=dict(scoring),
        n_jobs=-1,
        return_train_score=False,
    )
    rows = []
    for metric in scoring:
        scores = results[f"test_{metric}"]
        rows.append({"metric": metric, "mean": np.mean(scores), "std": np.std(scores)})
    return pd.DataFrame(rows).set_index("metric")


def test_set_report(estimator: BaseEstimator, X_test, y_test) -> dict:
    """Compute the final-test-set metrics for a fitted estimator."""
    y_pred = estimator.predict(X_test)

    report: dict = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_ai": precision_score(y_test, y_pred, pos_label=1),
        "recall_ai": recall_score(y_test, y_pred, pos_label=1),
        "f1_ai": f1_score(y_test, y_pred, pos_label=1),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["human", "ai"]
        ),
    }

    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X_test)[:, 1]
        report["roc_auc"] = roc_auc_score(y_test, proba)
    elif hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(X_test)
        report["roc_auc"] = roc_auc_score(y_test, scores)

    return report
