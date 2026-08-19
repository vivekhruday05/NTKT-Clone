"""Plotting utilities for Cold-Start results (Figures 2 & 3 from the paper)."""

import os
from typing import Dict, Any, List, Optional
import numpy as np


def plot_user_cold_start_trajectories(
    model_trajectories: Dict[str, Dict[int, float]],
    save_path: str = "artifacts/figure_2_user_cold_start.png",
    metric_name: str = "f1"
) -> None:
    """Plot average F1 score across timesteps for NTKT and baselines (Figure 2).
    
    Args:
        model_trajectories: Dict mapping model_name -> {timestep: score}.
        save_path: Path to save the output plot.
        metric_name: Metric to plot (default 'f1').
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not installed. Skipping plot generation.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    plt.figure(figsize=(8, 5), dpi=300)
    
    markers = ['o', 's', '^', 'v', 'D', 'x', '*']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i, (model_name, traj) in enumerate(model_trajectories.items()):
        steps = sorted(traj.keys())
        scores = [traj[s] for s in steps]
        plt.plot(
            steps,
            scores,
            label=model_name,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            linewidth=2,
            markersize=6
        )

    plt.title("Average F1 Score Across Timesteps in User Cold-Start Scenario", fontsize=12, pad=10)
    plt.xlabel("Interaction Timestep (t)", fontsize=11)
    plt.ylabel(f"Average {metric_name.upper()} Score", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved User Cold Start plot to {save_path}.")


def plot_question_cold_start_barchart(
    model_comparison: Dict[str, Dict[str, float]],
    save_path: str = "artifacts/figure_3_question_cold_start.png",
    metric_name: str = "f1"
) -> None:
    """Plot seen vs unseen question performance comparison (Figure 3).
    
    Args:
        model_comparison: Dict mapping model_name -> {'seen': val, 'cold_start': val}.
        save_path: Output file path.
        metric_name: Metric to plot.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not installed. Skipping plot generation.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    models = list(model_comparison.keys())
    seen_vals = [model_comparison[m]["seen"] for m in models]
    cold_vals = [model_comparison[m]["cold_start"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    rects1 = ax.bar(x - width/2, seen_vals, width, label="Seen Questions", color="#4C72B0")
    rects2 = ax.bar(x + width/2, cold_vals, width, label="Cold-Start Questions", color="#DD8452")

    ax.set_ylabel(f"{metric_name.upper()} Score", fontsize=11)
    ax.set_title("Performance Comparison: Seen vs Unseen Cold-Start Questions", fontsize=12, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=10)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved Question Cold Start plot to {save_path}.")
