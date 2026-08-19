"""Evaluation runner for trained NTKT models and baselines.

Usage:
    python scripts/run_evaluate.py --model_type ntkt --checkpoint_dir checkpoints/ntkt_llama3b/best_checkpoint
    python scripts/run_evaluate.py --model_type dkt --checkpoint_dir checkpoints/baselines/dkt
"""

import os
import argparse
import json
import torch
from torch.utils.data import DataLoader

from data.eedi_dataset import (
    load_eedi_raw_data,
    prepare_student_histories,
    build_stepwise_samples,
    split_eedi_data,
    EediSequenceDataset,
    BaselineSequenceDataset,
)
from data.collator import NTKTDataCollator
from data.synthetic_generator import generate_synthetic_dataset
from models.ntkt_model import NTKTModel
from models.baselines.dkt import DKT
from evaluation.evaluator import evaluate_ntkt, evaluate_baseline


def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained Model")
    parser.add_argument("--model_type", type=str, default="ntkt", choices=["ntkt", "dkt", "akt", "akt_text", "dtransformer"])
    parser.add_argument("--base_model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to checkpoint directory or weights")
    parser.add_argument("--dataset_path", type=str, default="data/processed/processed_eedi.csv")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--use_synthetic", action="store_true")
    args = parser.parse_args()

    if args.use_synthetic or not os.path.exists(args.dataset_path):
        train_csv, test_csv = generate_synthetic_dataset(output_dir="data/synthetic", num_students=30, num_questions=25)
        args.dataset_path = test_csv

    df = load_eedi_raw_data(args.dataset_path)
    splits = split_eedi_data(df, test_ratio=0.1, seed=42)
    val_records = prepare_student_histories(splits["user_cold_start_test"])

    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")

    if args.model_type == "ntkt":
        val_samples = build_stepwise_samples(val_records, min_history=1)
        val_dataset = EediSequenceDataset(val_samples, ablation_mode="full_text")

        model = NTKTModel(
            model_name_or_path=args.base_model_name,
            lora_rank=16,
            lora_alpha=16.0,
            load_in_4bit=is_cuda,
            torch_dtype="bfloat16" if is_cuda else "float32",
        )
        model.load_adapters(args.checkpoint_dir)
        collator = NTKTDataCollator(tokenizer=model.tokenizer, max_length=2048)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

        results = evaluate_ntkt(model, val_loader, device=device)

    else:
        all_qids = sorted(df["QuestionId"].unique())
        all_cids = sorted(df["ConstructName"].unique())
        qid_to_idx = {qid: i + 1 for i, qid in enumerate(all_qids)}
        cid_to_idx = {cid: i + 1 for i, cid in enumerate(all_cids)}

        from scripts.run_train_baseline import prepare_baseline_sequences
        val_seqs = prepare_baseline_sequences(val_records, qid_to_idx, cid_to_idx)
        val_dataset = BaselineSequenceDataset(val_seqs, max_seq_len=100, num_questions=len(qid_to_idx), num_concepts=len(cid_to_idx))
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        model = DKT(num_questions=len(qid_to_idx))
        weights_path = os.path.join(args.checkpoint_dir, "best_baseline_model.pt") if os.path.isdir(args.checkpoint_dir) else args.checkpoint_dir
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location=device))
        model.to(device)

        results = evaluate_baseline(model, val_loader, device=device)

    m = results["metrics"]
    print("\n" + "="*50)
    print(f"EVALUATION RESULTS ({args.model_type.upper()}):")
    print(f"  ROC-AUC:   {m['auc'] * 100:.2f}%")
    print(f"  Accuracy:  {m['accuracy'] * 100:.2f}%")
    print(f"  F1 Score:  {m['f1'] * 100:.2f}%")
    print(f"  Log-Loss:  {m['log_loss']:.4f}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
