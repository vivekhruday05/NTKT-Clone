"""Evaluation runner for Knowledge Tracing experiments.

Evaluates models on:
1. Overall benchmark (Table 1)
2. Feature ablations (Table 2)
3. User Cold Start timestep progression (Figure 2)
4. Question Cold Start on unseen items (Figure 3)
"""

from typing import Dict, Any, List, Optional, Union
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation.metrics import compute_kt_metrics


def evaluate_ntkt(
    model: Any,
    dataloader: DataLoader,
    device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    """Evaluate NTKT LLM model over a DataLoader."""
    model.eval()
    all_targets = []
    all_probs = []
    all_timesteps = []
    all_qids = []
    all_uids = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating NTKT", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_labels = batch["target_labels"].cpu().numpy()

            probs = model.predict_probabilities(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).cpu().numpy()

            all_targets.extend(target_labels.tolist())
            all_probs.extend(probs.tolist())
            
            if "timestep" in batch:
                all_timesteps.extend(batch["timestep"].tolist() if isinstance(batch["timestep"], torch.Tensor) else batch["timestep"])
            if "question_id" in batch:
                all_qids.extend(batch["question_id"].tolist() if isinstance(batch["question_id"], torch.Tensor) else batch["question_id"])
            if "user_id" in batch:
                all_uids.extend(batch["user_id"].tolist() if isinstance(batch["user_id"], torch.Tensor) else batch["user_id"])

    overall_metrics = compute_kt_metrics(all_targets, all_probs)
    
    return {
        "metrics": overall_metrics,
        "targets": np.array(all_targets),
        "probs": np.array(all_probs),
        "timesteps": np.array(all_timesteps) if all_timesteps else None,
        "question_ids": np.array(all_qids) if all_qids else None,
        "user_ids": np.array(all_uids) if all_uids else None,
    }


def evaluate_baseline(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    """Evaluate traditional numeric KT baseline model (DKT, AKT, DTransformer)."""
    model.eval()
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Baseline", leave=False):
            q_ids = batch["question_ids"].to(device)
            concept_ids = batch.get("concept_ids", None)
            if concept_ids is not None:
                concept_ids = concept_ids.to(device)
            correctness = batch["correctness"].to(device)
            mask = batch["mask"].to(device)
            
            text_emb = batch.get("text_embeddings", None)
            if text_emb is not None:
                text_emb = text_emb.to(device)

            outputs = model(
                question_ids=q_ids,
                concept_ids=concept_ids,
                correctness=correctness,
                mask=mask,
                text_embeddings=text_emb
            )

            probs = outputs["probs"].cpu().numpy()
            target_corr = correctness[:, 1:].cpu().numpy()
            target_mask = outputs["mask"].cpu().numpy()

            valid_idx = np.where(target_mask > 0)
            valid_targets = target_corr[valid_idx]
            valid_probs = probs[valid_idx]

            all_targets.extend(valid_targets.tolist())
            all_probs.extend(valid_probs.tolist())

    overall_metrics = compute_kt_metrics(all_targets, all_probs)
    return {
        "metrics": overall_metrics,
        "targets": np.array(all_targets),
        "probs": np.array(all_probs),
    }


def compute_user_cold_start_trajectory(
    targets: np.ndarray,
    probs: np.ndarray,
    timesteps: np.ndarray,
    max_timestep: int = 20
) -> Dict[int, Dict[str, float]]:
    """Compute F1 and accuracy metrics across learner interaction timesteps (Figure 2)."""
    trajectory = {}
    for t in range(1, max_timestep + 1):
        idx = np.where(timesteps == t)[0]
        if len(idx) > 0:
            step_metrics = compute_kt_metrics(targets[idx], probs[idx])
            trajectory[t] = step_metrics
    return trajectory


def compute_question_cold_start_comparison(
    seen_targets: np.ndarray,
    seen_probs: np.ndarray,
    cold_targets: np.ndarray,
    cold_probs: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """Compare performance on Seen Questions vs Cold-Start (unseen) Questions (Figure 3)."""
    seen_metrics = compute_kt_metrics(seen_targets, seen_probs)
    cold_metrics = compute_kt_metrics(cold_targets, cold_probs)
    return {
        "seen_questions": seen_metrics,
        "cold_start_questions": cold_metrics,
    }
