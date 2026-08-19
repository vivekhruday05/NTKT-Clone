"""Training script for Next Token Knowledge Tracing (NTKT).

Usage:
    python scripts/run_train_ntkt.py --config configs/default_ntkt_llama3b.yaml
    python scripts/run_train_ntkt.py --config configs/cpu_smoke_test.yaml
"""

import os
import sys
import argparse
import yaml
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train NTKT Model")
    parser.add_argument("--config", type=str, default="configs/default_ntkt_llama3b.yaml", help="Path to YAML config")
    parser.add_argument("--model_name", type=str, default=None, help="Override foundation model name")
    parser.add_argument("--dataset_path", type=str, default=None, help="Override dataset path")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max training steps")
    parser.add_argument("--learning_rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch_size", type=int, default=None, help="Override per-device batch size")
    parser.add_argument("--ablation_mode", type=str, default=None, choices=["full_text", "concept_only", "id_only"])
    parser.add_argument("--use_synthetic", action="store_true", help="Generate and use synthetic data for fast run")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # CLI Overrides
    if args.model_name:
        cfg["model"]["name_or_path"] = args.model_name
    if args.dataset_path:
        cfg["data"]["dataset_path"] = args.dataset_path
    if args.output_dir:
        cfg["training"]["output_dir"] = args.output_dir
    if args.max_steps:
        cfg["training"]["max_steps"] = args.max_steps
    if args.learning_rate:
        cfg["training"]["learning_rate"] = args.learning_rate
    if args.batch_size:
        cfg["training"]["per_device_batch_size"] = args.batch_size
    if args.ablation_mode:
        cfg["data"]["ablation_mode"] = args.ablation_mode

    # 1. Prepare Dataset
    dataset_path = cfg["data"]["dataset_path"]
    if args.use_synthetic or not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}. Generating synthetic dataset for testing...")
        train_csv, _ = generate_synthetic_dataset(output_dir="data/synthetic", num_students=40, num_questions=30)
        dataset_path = train_csv

    print(f"Loading dataset from {dataset_path}...")
    df = load_eedi_raw_data(dataset_path, question_meta_path=cfg["data"].get("question_meta_path"))

    splits = split_eedi_data(
        df,
        test_ratio=cfg["data"].get("test_ratio", 0.1),
        cold_start_question_count=cfg["data"].get("cold_start_question_count", 0),
        seed=cfg["data"].get("seed", 42)
    )

    train_records = prepare_student_histories(splits["train"])
    val_records = prepare_student_histories(splits["user_cold_start_test"])

    train_samples = build_stepwise_samples(train_records, min_history=1)
    val_samples = build_stepwise_samples(val_records, min_history=1)

    print(f"Constructed samples: Train={len(train_samples)}, Val={len(val_samples)}")

    ablation_mode = cfg["data"].get("ablation_mode", "full_text")
    max_history_len = cfg["data"].get("max_history_len", 50)

    train_dataset = EediSequenceDataset(train_samples, ablation_mode=ablation_mode, max_history_len=max_history_len)
    val_dataset = EediSequenceDataset(val_samples, ablation_mode=ablation_mode, max_history_len=max_history_len)

    # 2. Load Model
    print(f"Initializing NTKT Model with base: {cfg['model']['name_or_path']}...")
    is_cuda = torch.cuda.is_available()
    
    model = NTKTModel(
        model_name_or_path=cfg["model"]["name_or_path"],
        lora_rank=cfg["lora"]["rank"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"].get("target_modules"),
        load_in_4bit=cfg["model"].get("load_in_4bit", False) and is_cuda,
        load_in_8bit=cfg["model"].get("load_in_8bit", False) and is_cuda,
        torch_dtype=cfg["model"].get("torch_dtype", "bfloat16" if is_cuda else "float32"),
        use_peft=cfg["model"].get("use_peft", True),
    )

    # 3. Create Collator & DataLoaders
    collator = NTKTDataCollator(
        tokenizer=model.tokenizer,
        max_length=cfg["data"].get("max_length", 4096),
    )

    batch_size = cfg["training"]["per_device_batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    # 4. Train Model
    trainer = NTKTTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        output_dir=cfg["training"]["output_dir"],
        learning_rate=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        max_steps=int(cfg["training"]["max_steps"]),
        warmup_steps=int(cfg["training"]["warmup_steps"]),
        gradient_accumulation_steps=int(cfg["training"]["gradient_accumulation_steps"]),
        eval_steps=int(cfg["training"]["eval_steps"]),
        early_stopping_patience=int(cfg["training"]["early_stopping_patience"]),
        early_stopping_delta=float(cfg["training"]["early_stopping_delta"]),
        max_grad_norm=float(cfg["training"]["max_grad_norm"]),
        use_8bit_adam=cfg["training"].get("use_8bit_adam", False) and is_cuda,
        fp16=cfg["training"].get("fp16", False) and is_cuda,
        bf16=cfg["training"].get("bf16", True) and is_cuda,
    )

    results = trainer.train()

    # 5. Final Evaluation on Validation Set
    print("\nRunning final benchmark evaluation on validation set...")
    eval_results = evaluate_ntkt(model, val_loader, device=trainer.device)
    metrics = eval_results["metrics"]

    print("\n" + "="*50)
    print("FINAL EVALUATION METRICS (NTKT):")
    print(f"  ROC-AUC:   {metrics['auc'] * 100:.2f}%")
    print(f"  Accuracy:  {metrics['accuracy'] * 100:.2f}%")
    print(f"  F1 Score:  {metrics['f1'] * 100:.2f}%")
    print(f"  Log-Loss:  {metrics['log_loss']:.4f}")
    print(f"  ECE:       {metrics['ece']:.4f}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
