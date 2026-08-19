"""Training script for traditional Knowledge Tracing baseline models (DKT, AKT, AKT-text, DTransformer).

Usage:
    python scripts/run_train_baseline.py --config configs/baselines/dkt.yaml
    python scripts/run_train_baseline.py --config configs/baselines/akt.yaml
    python scripts/run_train_baseline.py --config configs/baselines/akt_text.yaml
    python scripts/run_train_baseline.py --config configs/baselines/dtransformer.yaml
"""

import os
import argparse
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader

from data.eedi_dataset import (
    load_eedi_raw_data,
    prepare_student_histories,
    split_eedi_data,
    BaselineSequenceDataset,
)
from data.synthetic_generator import generate_synthetic_dataset
from models.baselines.dkt import DKT
from models.baselines.akt import AKT
from models.baselines.akt_text import AKTText
from models.baselines.dtransformer import DTransformer
from training.baseline_trainer import BaselineTrainer
from evaluation.evaluator import evaluate_baseline


def parse_args():
    parser = argparse.ArgumentParser(description="Train Baseline KT Model")
    parser.add_argument("--config", type=str, default="configs/baselines/dkt.yaml", help="Path to baseline YAML config")
    parser.add_argument("--model_type", type=str, default=None, choices=["dkt", "akt", "akt_text", "dtransformer"])
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--use_synthetic", action="store_true")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def prepare_baseline_sequences(student_records, qid_to_idx, cid_to_idx):
    sequences = []
    for uid, records in student_records.items():
        q_ids = [qid_to_idx.get(r["question_id"], 1) for r in records]
        c_ids = [cid_to_idx.get(r["concept"], 1) for r in records]
        corrects = [r["is_correct"] for r in records]
        sequences.append({
            "user_id": uid,
            "question_ids": q_ids,
            "concept_ids": c_ids,
            "correctness": corrects,
        })
    return sequences


def main():
    args = parse_args()
    cfg = load_config(args.config)

    model_type = args.model_type or cfg["model"].get("type", "dkt").lower()
    dataset_path = args.dataset_path or cfg["data"]["dataset_path"]
    output_dir = args.output_dir or cfg["training"]["output_dir"]
    num_epochs = args.epochs or cfg["training"]["num_epochs"]
    batch_size = args.batch_size or cfg["training"]["batch_size"]

    if args.use_synthetic or not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}. Generating synthetic dataset for baseline...")
        train_csv, _ = generate_synthetic_dataset(output_dir="data/synthetic", num_students=50, num_questions=40)
        dataset_path = train_csv

    df = load_eedi_raw_data(dataset_path)
    splits = split_eedi_data(df, test_ratio=cfg["data"].get("test_ratio", 0.1), seed=cfg["data"].get("seed", 42))

    train_records = prepare_student_histories(splits["train"])
    val_records = prepare_student_histories(splits["user_cold_start_test"])

    # Create integer index mappings for questions and concepts
    all_qids = sorted(df["QuestionId"].unique())
    all_cids = sorted(df["ConstructName"].unique())
    qid_to_idx = {qid: i + 1 for i, qid in enumerate(all_qids)}
    cid_to_idx = {cid: i + 1 for i, cid in enumerate(all_cids)}

    num_questions = len(qid_to_idx)
    num_concepts = len(cid_to_idx)

    train_seqs = prepare_baseline_sequences(train_records, qid_to_idx, cid_to_idx)
    val_seqs = prepare_baseline_sequences(val_records, qid_to_idx, cid_to_idx)

    # Optional text embeddings for AKTText
    text_embeddings = None
    if model_type == "akt_text":
        print("Preparing question text embeddings for AKT-text...")
        try:
            from sentence_transformers import SentenceTransformer
            st_model = SentenceTransformer("all-MiniLM-L6-v2")
            unique_q_df = df[["QuestionId", "QuestionText"]].drop_duplicates(subset=["QuestionId"])
            text_embeddings = {}
            for _, r in unique_q_df.iterrows():
                qid_idx = qid_to_idx.get(r["QuestionId"], 1)
                emb = st_model.encode(str(r["QuestionText"]), convert_to_numpy=True)
                text_embeddings[qid_idx] = emb
            print(f"Extracted {len(text_embeddings)} MiniLM embeddings.")
        except Exception as e:
            print(f"SentenceTransformer not available ({e}). Using random 384-d embeddings.")
            text_embeddings = {qid_idx: np.random.randn(384).astype(np.float32) for qid_idx in qid_to_idx.values()}

    max_seq_len = cfg["data"].get("max_seq_len", 100)
    train_dataset = BaselineSequenceDataset(train_seqs, max_seq_len=max_seq_len, num_questions=num_questions, num_concepts=num_concepts, text_embeddings=text_embeddings)
    val_dataset = BaselineSequenceDataset(val_seqs, max_seq_len=max_seq_len, num_questions=num_questions, num_concepts=num_concepts, text_embeddings=text_embeddings)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Instantiate Baseline Model
    if model_type == "dkt":
        model = DKT(
            num_questions=num_questions,
            embed_dim=cfg["model"].get("embed_dim", 128),
            hidden_dim=cfg["model"].get("hidden_dim", 128),
            dropout=cfg["model"].get("dropout", 0.2)
        )
    elif model_type == "akt":
        model = AKT(
            num_questions=num_questions,
            num_concepts=num_concepts,
            d_model=cfg["model"].get("d_model", 128),
            n_heads=cfg["model"].get("n_heads", 4),
            dropout=cfg["model"].get("dropout", 0.1)
        )
    elif model_type == "akt_text":
        model = AKTText(
            num_questions=num_questions,
            num_concepts=num_concepts,
            text_dim=cfg["model"].get("text_dim", 384),
            d_model=cfg["model"].get("d_model", 128),
            n_heads=cfg["model"].get("n_heads", 4),
            dropout=cfg["model"].get("dropout", 0.1)
        )
    elif model_type == "dtransformer":
        model = DTransformer(
            num_questions=num_questions,
            num_concepts=num_concepts,
            d_model=cfg["model"].get("d_model", 128),
            n_heads=cfg["model"].get("n_heads", 4),
            dropout=cfg["model"].get("dropout", 0.1)
        )
    else:
        raise ValueError(f"Unknown baseline model type: {model_type}")

    trainer = BaselineTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        output_dir=output_dir,
        learning_rate=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        num_epochs=num_epochs,
        early_stopping_patience=cfg["training"].get("early_stopping_patience", 8),
    )

    trainer.train()

    # Final evaluation
    print(f"\nFinal Evaluation on Validation Set ({model_type.upper()}):")
    final_res = evaluate_baseline(model, val_loader, device=trainer.device)
    m = final_res["metrics"]
    print("="*50)
    print(f"  Model:     {model_type.upper()}")
    print(f"  ROC-AUC:   {m['auc'] * 100:.2f}%")
    print(f"  Accuracy:  {m['accuracy'] * 100:.2f}%")
    print(f"  F1 Score:  {m['f1'] * 100:.2f}%")
    print(f"  Log-Loss:  {m['log_loss']:.4f}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
