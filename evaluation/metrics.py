"""Evaluation metrics for Knowledge Tracing.

Implements standard binary KT metrics:
- ROC-AUC (Area Under the Receiver Operating Characteristic)
- Accuracy
- F1 Score
- Expected Calibration Error (ECE)
- Binary Cross Entropy (Log-Loss)
- Summary statistics (Mean ± SD across runs)
"""

import numpy as np
from typing import Dict, Any, List, Union
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, log_loss


def compute_kt_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred_probs: Union[List[float], np.ndarray],
    threshold: float = 0.5
) -> Dict[str, float]:
    """Compute comprehensive KT metrics.
    
    Args:
        y_true: Array of ground-truth binary labels (0 or 1).
        y_pred_probs: Array of predicted probabilities P(Correct) in [0, 1].
        threshold: Decision boundary for binary classification (default 0.5).
        
    Returns:
        Dict mapping metric names to float scores.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred_probs = np.asarray(y_pred_probs, dtype=float)
    
    # Clip probabilities to avoid log(0)
    eps = 1e-7
    y_pred_probs = np.clip(y_pred_probs, eps, 1.0 - eps)
    y_pred = (y_pred_probs >= threshold).astype(int)

    # ROC-AUC
    unique_classes = np.unique(y_true)
    if len(unique_classes) > 1:
        auc = float(roc_auc_score(y_true, y_pred_probs))
    else:
        auc = 0.5  # Fallback if only 1 class is present

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    try:
        bce = float(log_loss(y_true, y_pred_probs))
    except Exception:
        bce = float(-np.mean(y_true * np.log(y_pred_probs) + (1 - y_true) * np.log(1 - y_pred_probs)))

    # Expected Calibration Error (ECE)
    ece = compute_ece(y_true, y_pred_probs, n_bins=10)

    return {
        "auc": round(auc, 5),
        "accuracy": round(acc, 5),
        "f1": round(f1, 5),
        "log_loss": round(bce, 5),
        "ece": round(ece, 5),
    }


def compute_ece(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    n_bins: int = 10
) -> float:
    """Calculate Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (y_probs >= bin_lower) & (y_probs < bin_upper if i < n_bins - 1 else y_probs <= bin_upper)
        bin_size = np.sum(in_bin)
        
        if bin_size > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_probs[in_bin])
            ece += (bin_size / total_samples) * np.abs(bin_acc - bin_conf)

    return float(ece)


def format_mean_std_results(results_list: List[Dict[str, float]]) -> Dict[str, str]:
    """Format list of experiment runs into Mean ± SD strings."""
    aggregated = {}
    keys = results_list[0].keys()
    
    for k in keys:
        values = [r[k] for r in results_list if k in r]
        mean_val = np.mean(values)
        std_val = np.std(values)
        aggregated[k] = f"{mean_val * 100:.2f} ± {std_val * 100:.4f}"
        
    return aggregated
