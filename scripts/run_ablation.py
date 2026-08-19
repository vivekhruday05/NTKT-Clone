"""Experiment 2 (RQ2): Input Representation Feature Ablation Runner.

Reproduces Table 2 of the paper:
Evaluates NTKT under three input feature configurations:
1. Full Text: Question text, Options A-D, Question ID, and Concept tags
2. Concept-only: Question ID and Concept tags (omits text and options)
3. ID-only: Question ID only (omits text, options, and concept tags)
"""

import os
import argparse
import json
import pandas as pd
import torch
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
from evaluation.evaluator import evaluate_ntkt


def run_single_ablation(
    ablation_mode: str,
    base_model_name: str,
    train_samples: list,
    val_samples: list,
    output_dir: str,
    max_steps: int = 500,
    batch_size: int = 4
):
    print(f"\n{'='*60}")
    print(f"RUNNING ABLATION: {ablation_mode.upper()}")
    print(f"{'='*60}")

    train_dataset = EediSequenceDataset(train_samples, ablation_mode=ablation_mode)
    val_dataset = EediSequenceDataset(val_samples, ablation_mode=ablation_mode)

    is_cuda = torch.cuda.is_available()
    model = NTKTModel(
        model_name_or_path=base_model_name,
        lora_rank=16,
        lora_alpha=16.0,
        load_in_4bit=is_cuda,
        torch_dtype="bfloat16" if is_cuda else "float32",
    )

    collator = NTKTDataCollator(tokenizer=model.tokenizer, max_length=2048)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    ablation_out = os.path.join(output_dir, f"ablation_{ablation_mode}")
    trainer = NTKTTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        output_dir=ablation_out,
        learning_rate=2e-4,
        max_steps=max_steps,
        gradient_accumulation_steps=2,
        eval_steps=max(10, max_steps // 5),
    )

    trainer.train()
    eval_results = evaluate_ntkt(model, val_loader, device=trainer.device)
    return eval_results["metrics"]


def main():
    parser = argparse.ArgumentParser(description="RQ2 Feature Representation Ablation")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--dataset_path", type=str, default="data/processed/processed_eedi.csv")
    parser.add_argument("--output_dir", type=str, default="checkpoints/ablations")
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--use_synthetic", action="store_true")
    args = parser.parse_args()

    if args.use_synthetic or not os.path.exists(args.dataset_path):
        train_csv, _ = generate_synthetic_dataset(output_dir="data/synthetic", num_students=40, num_questions=30)
        args.dataset_path = train_csv

    df = load_eedi_raw_data(args.dataset_path)
    splits = split_eedi_data(df, test_ratio=0.1, seed=42)

    train_records = prepare_student_histories(splits["train"])
    val_records = prepare_student_histories(splits["user_cold_start_test"])

    train_samples = build_stepwise_samples(train_records, min_history=1)
    val_samples = build_stepwise_samples(val_records, min_history=1)

    ablation_modes = ["id_only", "concept_only", "full_text"]
    all_results = {}

    for mode in ablation_modes:
        metrics = run_single_ablation(
            ablation_mode=mode,
            base_model_name=args.model_name,
            train_samples=train_samples,
            val_samples=val_samples,
            output_dir=args.output_dir,
            max_steps=args.max_steps,
            batch_size=args.batch_size
        )
        all_results[mode] = metrics

    # Print Table 2 Reproduction Summary
    print("\n" + "="*70)
    print("TABLE 2: EEDI NTKT PERFORMANCE USING DIFFERENT FEATURE REPRESENTATIONS")
    print("="*70)
    print(f"{'Features':<20} | {'F1 Score (%)':<15} | {'Accuracy (%)':<15} | {'AUC (%)':<15}")
    print("-" * 70)
    for mode in ["id_only", "concept_only", "full_text"]:
        m = all_results[mode]
        name_map = {"id_only": "ID-only", "concept_only": "Concept-only", "full_text": "Full Text"}
        print(f"{name_map[mode]:<20} | {m['f1']*100:<15.2f} | {m['accuracy']*100:<15.2f} | {m['auc']*100:<15.2f}")
    print("="*70)

    summary_file = os.path.join(args.output_dir, "ablation_results_table2.json")
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {summary_file}")


if __name__ == "__main__":
    main()
