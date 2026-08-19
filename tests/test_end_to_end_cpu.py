"""End-to-End CPU Integration Test.

Runs complete workflow:
Synthetic Data Generation -> Preprocessing -> NTKT Tokenization & Loss Masking ->
3 Steps of Fine-Tuning with Native LoRA on CPU -> Evaluation & Metrics Output.
"""

import os
import shutil
import tempfile
import pytest
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Config

from data.synthetic_generator import generate_synthetic_dataset
from data.eedi_dataset import (
    load_eedi_raw_data,
    prepare_student_histories,
    build_stepwise_samples,
    split_eedi_data,
    EediSequenceDataset,
)
from data.collator import NTKTDataCollator
from models.ntkt_model import NTKTModel
from training.trainer import NTKTTrainer
from evaluation.evaluator import evaluate_ntkt


def test_end_to_end_pipeline_cpu():
    temp_dir = tempfile.mkdtemp()
    try:
        # 1. Generate Synthetic Dataset
        data_dir = os.path.join(temp_dir, "data")
        train_csv, test_csv = generate_synthetic_dataset(
            output_dir=data_dir,
            num_students=8,
            num_questions=6,
            min_interactions_per_student=3,
            max_interactions_per_student=6,
            seed=42
        )
        assert os.path.exists(train_csv)
        assert os.path.exists(test_csv)

        # 2. Preprocess & Construct Sequence Histories
        df = load_eedi_raw_data(train_csv)
        splits = split_eedi_data(df, test_ratio=0.2, seed=42)

        train_records = prepare_student_histories(splits["train"])
        val_records = prepare_student_histories(splits["user_cold_start_test"])

        train_samples = build_stepwise_samples(train_records, min_history=1)
        val_samples = build_stepwise_samples(val_records, min_history=1)
        assert len(train_samples) > 0
        assert len(val_samples) > 0

        train_dataset = EediSequenceDataset(train_samples, ablation_mode="full_text")
        val_dataset = EediSequenceDataset(val_samples, ablation_mode="full_text")

        # 3. Initialize Model with tiny config for CPU testing
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")
        gpt2_cfg = GPT2Config(
            vocab_size=len(tok),
            n_positions=256,
            n_embd=64,
            n_layer=2,
            n_head=2,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id
        )
        model = NTKTModel(
            model_name_or_path="gpt2",
            lora_rank=4,
            lora_alpha=4.0,
            torch_dtype="float32",
            use_peft=False,
            config=gpt2_cfg
        )

        collator = NTKTDataCollator(tokenizer=model.tokenizer, max_length=256)
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collator)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collator)

        # 4. Train for 3 steps
        ckpt_dir = os.path.join(temp_dir, "checkpoints")
        trainer = NTKTTrainer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            output_dir=ckpt_dir,
            learning_rate=1e-3,
            max_steps=3,
            gradient_accumulation_steps=1,
            eval_steps=3,
            use_8bit_adam=False,
            fp16=False,
            bf16=False,
            device=torch.device("cpu")
        )

        train_res = trainer.train()
        assert train_res["total_steps"] == 3

        # 5. Evaluate Validation Set
        eval_res = evaluate_ntkt(model, val_loader, device=torch.device("cpu"))
        assert "metrics" in eval_res
        metrics = eval_res["metrics"]
        assert "auc" in metrics
        assert "accuracy" in metrics
        assert "f1" in metrics

    finally:
        shutil.rmtree(temp_dir)
