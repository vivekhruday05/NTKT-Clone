"""Evaluation package for NTKT metrics, benchmarking, and plots."""

from evaluation.metrics import compute_kt_metrics, compute_ece, format_mean_std_results
from evaluation.evaluator import (
    evaluate_ntkt,
    evaluate_baseline,
    compute_user_cold_start_trajectory,
    compute_question_cold_start_comparison,
)
from evaluation.plots import (
    plot_user_cold_start_trajectories,
    plot_question_cold_start_barchart,
)

__all__ = [
    "compute_kt_metrics",
    "compute_ece",
    "format_mean_std_results",
    "evaluate_ntkt",
    "evaluate_baseline",
    "compute_user_cold_start_trajectory",
    "compute_question_cold_start_comparison",
    "plot_user_cold_start_trajectories",
    "plot_question_cold_start_barchart",
]
