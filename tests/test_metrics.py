"""Unit tests for KT metrics, calibration calculation, and formatting."""

import pytest
import numpy as np
from evaluation.metrics import compute_kt_metrics, compute_ece, format_mean_std_results


def test_compute_kt_metrics_perfect():
    y_true = [1, 0, 1, 0, 1, 0]
    y_probs = [0.95, 0.05, 0.90, 0.10, 0.85, 0.15]

    res = compute_kt_metrics(y_true, y_probs)
    assert res["auc"] == 1.0
    assert res["accuracy"] == 1.0
    assert res["f1"] == 1.0
    assert res["log_loss"] < 0.3
    assert res["ece"] < 0.2


def test_compute_kt_metrics_inverted():
    y_true = [1, 0, 1, 0]
    y_probs = [0.1, 0.9, 0.2, 0.8]

    res = compute_kt_metrics(y_true, y_probs)
    assert res["auc"] == 0.0
    assert res["accuracy"] == 0.0
    assert res["f1"] == 0.0


def test_ece_perfect_calibration():
    # Model predicting 0.8 accuracy on samples where true proportion is 0.8
    y_true = np.array([1, 1, 1, 1, 0] * 20)
    y_probs = np.full(100, 0.8)

    ece = compute_ece(y_true, y_probs, n_bins=10)
    assert ece < 1e-4


def test_format_mean_std_results():
    runs = [
        {"auc": 0.82, "f1": 0.75},
        {"auc": 0.84, "f1": 0.77},
        {"auc": 0.80, "f1": 0.73},
    ]
    formatted = format_mean_std_results(runs)
    assert "auc" in formatted
    assert "82.00" in formatted["auc"]
    assert "f1" in formatted
    assert "75.00" in formatted["f1"]
