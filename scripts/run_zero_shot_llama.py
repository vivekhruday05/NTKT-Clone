"""Zero-Shot / No-FT LLM Baseline Evaluator.

Evaluates an instruction-tuned LLM (e.g. LLaMA-3.2-3B-Instruct) on the KT task
without fine-tuning on student interaction histories, serving as the No-FT control.
"""

import os
import argparse
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
from evaluation.evaluator import evaluate_ntkt


def main():
    parser = argparse.ArgumentParser(description="Evaluate Zero-Shot / No-FT LLM Baseline")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--dataset_path", type=str, default="data/processed/processed_eedi.csv")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--use_synthetic", action="store_true")
    args = parser.parse_args()

    if args.use_synthetic or not os.path.exists(args.dataset_path):
        train_csv, test_csv = generate_synthetic_dataset(output_dir="data/synthetic", num_students=30, num_questions=25)
        args.dataset_path = test_csv

    df = load_eedi_raw_data(args.dataset_path)
    splits = split_eedi_data(df, test_ratio=0.2, seed=42)

    val_records = prepare_student_histories(splits["user_cold_start_test"])
    val_samples = build_stepwise_samples(val_records, min_history=1)
    if args.max_samples and len(val_samples) > args.max_samples:
        val_samples = val_samples[:args.max_samples]

    print(f"Evaluating Zero-Shot LLM on {len(val_samples)} test samples...")
    val_dataset = EediSequenceDataset(val_samples, ablation_mode="full_text")

    is_cuda = torch.cuda.is_available()
    model = NTKTModel(
        model_name_or_path=args.model_name,
        lora_rank=0,  # No LoRA, completely frozen
        is_trainable=False,
        load_in_4bit=is_cuda,
        torch_dtype="bfloat16" if is_cuda else "float32",
    )

    collator = NTKTDataCollator(tokenizer=model.tokenizer, max_length=2048)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    results = evaluate_ntkt(model, val_loader, device=torch.device("cuda" if is_cuda else "cpu"))
    m = results["metrics"]

    print("\n" + "="*60)
    print(f"ZERO-SHOT / NO-FT BASELINE RESULTS ({args.model_name}):")
    print(f"  ROC-AUC:   {m['auc'] * 100:.2f}%")
    print(f"  Accuracy:  {m['accuracy'] * 100:.2f}%")
    print(f"  F1 Score:  {m['f1'] * 100:.2f}%")
    print(f"  Log-Loss:  {m['log_loss']:.4f}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
