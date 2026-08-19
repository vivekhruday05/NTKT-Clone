"""Experiment 3 (RQ3): Cold-Start Generalisation Runner.

Reproduces Experiment 3 of the paper:
1. User Cold Start: Tracks average F1 score across interaction timesteps t=1..T on held-out learners (Figure 2).
2. Question Cold Start: Evaluates generalization on 10 withheld unseen questions vs seen questions (Figure 3).
"""

import os
import argparse
import json
import torch
import numpy as np
from torch.utils.data import DataLoader

from data.eedi_dataset import (
    load_eedi_raw_data,
    prepare_student_histories,
    build_stepwise_samples,
    split_eedi_data,
    EediSequenceDataset,
)
from data.collator import NTKTDataCollator
from data.synthetic_generator import generate_synthetic_dataset
from models.ntkt_model import NTKTModel
from training.trainer import NTKTTrainer
from evaluation.evaluator import (
    evaluate_ntkt,
    compute_user_cold_start_trajectory,
    compute_question_cold_start_comparison,
)
from evaluation.plots import (
    plot_user_cold_start_trajectories,
    plot_question_cold_start_barchart,
)


def parse_args():
    parser = argparse.ArgumentParser(description="RQ3 Cold Start Evaluation")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--dataset_path", type=str, default="data/processed/processed_eedi.csv")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to pre-trained NTKT checkpoint (optional)")
    parser.add_argument("--output_dir", type=str, default="artifacts/cold_start")
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--max_timesteps", type=int, default=20)
    parser.add_argument("--use_synthetic", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.use_synthetic or not os.path.exists(args.dataset_path):
        train_csv, _ = generate_synthetic_dataset(output_dir="data/synthetic", num_students=60, num_questions=40)
        args.dataset_path = train_csv

    df = load_eedi_raw_data(args.dataset_path)

    # Split: Withhold 10 questions for Question Cold Start and 10% users for User Cold Start
    splits = split_eedi_data(df, test_ratio=0.1, cold_start_question_count=10, seed=42)
    cold_qids = splits["cold_question_ids"]
    print(f"Cold Start Setup: Withheld {len(cold_qids)} questions for Question Cold Start: {cold_qids}")

    train_records = prepare_student_histories(splits["train"])
    user_test_records = prepare_student_histories(splits["user_cold_start_test"])
    cold_q_records = prepare_student_histories(splits["question_cold_start_test"])

    train_samples = build_stepwise_samples(train_records, min_history=1)
    user_test_samples = build_stepwise_samples(user_test_records, min_history=1)
    cold_q_samples = build_stepwise_samples(cold_q_records, min_history=1)

    print(f"Samples: Train={len(train_samples)}, User Cold-Start Test={len(user_test_samples)}, Question Cold-Start Test={len(cold_q_samples)}")

    train_dataset = EediSequenceDataset(train_samples, ablation_mode="full_text")
    user_test_dataset = EediSequenceDataset(user_test_samples, ablation_mode="full_text")
    cold_q_dataset = EediSequenceDataset(cold_q_samples, ablation_mode="full_text")

    is_cuda = torch.cuda.is_available()
    model = NTKTModel(
        model_name_or_path=args.model_name,
        lora_rank=16,
        lora_alpha=16.0,
        load_in_4bit=is_cuda,
        torch_dtype="bfloat16" if is_cuda else "float32",
    )

    if args.checkpoint_dir and os.path.exists(args.checkpoint_dir):
        model.load_adapters(args.checkpoint_dir)
    else:
        # Quick fine-tune if no checkpoint provided
        collator = NTKTDataCollator(tokenizer=model.tokenizer, max_length=2048)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collator)
        val_loader = DataLoader(user_test_dataset, batch_size=4, shuffle=False, collate_fn=collator)

        trainer = NTKTTrainer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            output_dir=os.path.join(args.output_dir, "model_ckpt"),
            learning_rate=2e-4,
            max_steps=args.max_steps,
        )
        trainer.train()

    # 1. Evaluate User Cold Start (Timestep Progression - Figure 2)
    collator = NTKTDataCollator(tokenizer=model.tokenizer, max_length=2048)
    user_test_loader = DataLoader(user_test_dataset, batch_size=4, shuffle=False, collate_fn=collator)

    user_eval = evaluate_ntkt(model, user_test_loader, device=torch.device("cuda" if is_cuda else "cpu"))
    timesteps = user_eval["timesteps"]
    targets = user_eval["targets"]
    probs = user_eval["probs"]

    user_trajectory = {}
    if timesteps is not None:
        user_trajectory = compute_user_cold_start_trajectory(targets, probs, timesteps, max_timestep=args.max_timesteps)
        print("\n" + "="*60)
        print("USER COLD-START: F1 Progression across Timesteps (Figure 2)")
        print("="*60)
        for t, m in user_trajectory.items():
            print(f"  Step t={t:2d} | F1: {m['f1']*100:.2f}% | AUC: {m['auc']*100:.2f}% | ACC: {m['accuracy']*100:.2f}%")

        plot_data = {"NTKT (Ours)": {t: m["f1"] for t, m in user_trajectory.items()}}
        plot_user_cold_start_trajectories(plot_data, save_path=os.path.join(args.output_dir, "figure_2_user_cold_start.png"))

    # 2. Evaluate Question Cold Start (Seen vs Unseen Questions - Figure 3)
    cold_q_loader = DataLoader(cold_q_dataset, batch_size=4, shuffle=False, collate_fn=collator)
    cold_q_eval = evaluate_ntkt(model, cold_q_loader, device=torch.device("cuda" if is_cuda else "cpu"))

    seen_metrics = user_eval["metrics"]
    cold_q_metrics = cold_q_eval["metrics"]

    print("\n" + "="*60)
    print("QUESTION COLD-START: Seen vs Unseen Items (Figure 3)")
    print("="*60)
    print(f"  Seen Questions:        F1 = {seen_metrics['f1']*100:.2f}%, AUC = {seen_metrics['auc']*100:.2f}%, ACC = {seen_metrics['accuracy']*100:.2f}%")
    print(f"  Cold-Start Questions:  F1 = {cold_q_metrics['f1']*100:.2f}%, AUC = {cold_q_metrics['auc']*100:.2f}%, ACC = {cold_q_metrics['accuracy']*100:.2f}%")

    comp_plot_data = {
        "NTKT (Ours)": {
            "seen": seen_metrics["f1"],
            "cold_start": cold_q_metrics["f1"]
        }
    }
    plot_question_cold_start_barchart(comp_plot_data, save_path=os.path.join(args.output_dir, "figure_3_question_cold_start.png"))

    # Save summary json
    summary_path = os.path.join(args.output_dir, "cold_start_results.json")
    with open(summary_path, "w") as f:
        json.dump({
            "user_cold_start_trajectory": {str(k): v for k, v in user_trajectory.items()},
            "seen_vs_cold_questions": {
                "seen": seen_metrics,
                "cold_start": cold_q_metrics
            }
        }, f, indent=2)
    print(f"\nCold start results and plots successfully saved to {args.output_dir}.")


if __name__ == "__main__":
    main()
