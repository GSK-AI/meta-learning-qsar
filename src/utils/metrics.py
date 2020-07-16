import logging
import sys
from typing import List

import numpy as np
import torch
from sklearn import metrics as skmetrics

METRICS = [
    "accuracy_score",
    "average_precision_score",
    "precision_score",
    "roc_auc_score",
    "recall_score",
    "mean_squared_error",
]


def binarize_predictions(y: np.ndarray, threshold: float):
    """Binarize array based on provided threshold"""
    preds = np.zeros_like(y)
    preds[y >= threshold] = 1.0
    return preds


def calculate_metrics(
    metrics: List[str], y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5
) -> dict:
    """Helper function for calculating metrics from y_true and y_pred

    Parameters
    ----------
    metrics : List[str]
        Metrics name in the namespace of sklearn.metrics.
        Currently supports :
            "accuracy_score"
            "average_precision_score"
            "precision_score"
            "roc_auc_score"
            "recall_score"
            "mean_squared_error"
    y_true : np.ndarray
        True values with shape (N x T) where N is number of instances
        and T is number of tasks.
    y_pred : np.ndarray
        Predicted values with the same shape as y_true.
    threshold : float, optional
        Threshold value for binary metrics such as accuracy_score, by default 0.5

    Returns
    -------
    dict
        Dictionary mapping metrics name to metrics values.
    """
    unsupported_metrics = set(metrics) - set(METRICS)
    if unsupported_metrics:
        logging.error(
            f"Metrics {unsupported_metrics} are not supported. Choose from {METRICS}, or '' for no metrics."
        )
        sys.exit(1)
    try:
        metric_values = {
            m: getattr(skmetrics, m)(y_true, binarize_predictions(y_pred, threshold))
            if m in ["accuracy_score", "precision_score", "recall_score"]
            else getattr(skmetrics, m)(y_true, y_pred)
            for m in metrics
        }
    except KeyError as e:
        logging.error(e)
        raise (e)

    return metric_values
