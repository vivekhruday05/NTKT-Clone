"""Unit tests for NTKT Model, LoRA wrapper, and KT Baseline Architectures."""

import pytest
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

from models.lora_wrapper import apply_native_lora, NativeLoraLinear
from models.ntkt_model import NTKTModel
from models.baselines.dkt import DKT
from models.baselines.akt import AKT
from models.baselines.akt_text import AKTText
from models.baselines.dtransformer import DTransformer


def test_native_lora_linear():
    base_linear = nn.Linear(32, 64)
    lora_linear = NativeLoraLinear(base_linear, rank=4, alpha=4.0)

    # Base layer parameters must be frozen
    for p in lora_linear.base_layer.parameters():
        assert not p.requires_grad

    # LoRA parameters must require grad
    assert lora_linear.lora_A.requires_grad
    assert lora_linear.lora_B.requires_grad

    x = torch.randn(2, 8, 32)
    out = lora_linear(x)
    assert out.shape == (2, 8, 64)


def test_ntkt_model_cpu_forward():
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

    input_ids = torch.randint(2, 1000, (2, 16))
    attention_mask = torch.ones((2, 16), dtype=torch.long)
    labels = torch.full((2, 16), -100, dtype=torch.long)
    labels[:, -1] = 10  # unmask final token

    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    assert "loss" in out
    assert out["loss"] is not None
    assert not torch.isnan(out["loss"])

    probs = model.predict_probabilities(input_ids=input_ids, attention_mask=attention_mask)
    assert probs.shape == (2,)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_dkt_baseline():
    model = DKT(num_questions=50, embed_dim=32, hidden_dim=32)
    q_ids = torch.randint(1, 50, (4, 10))
    correctness = torch.randint(0, 2, (4, 10)).float()
    mask = torch.ones((4, 10)).float()

    out = model(question_ids=q_ids, correctness=correctness, mask=mask)
    assert "loss" in out
    assert "probs" in out
    assert out["probs"].shape == (4, 9)


def test_akt_baseline():
    model = AKT(num_questions=50, num_concepts=20, d_model=32, n_heads=2)
    q_ids = torch.randint(1, 50, (4, 10))
    c_ids = torch.randint(1, 20, (4, 10))
    correctness = torch.randint(0, 2, (4, 10)).float()
    mask = torch.ones((4, 10)).float()

    out = model(question_ids=q_ids, concept_ids=c_ids, correctness=correctness, mask=mask)
    assert "loss" in out
    assert "probs" in out
    assert out["probs"].shape == (4, 9)


def test_dtransformer_baseline():
    model = DTransformer(num_questions=50, num_concepts=20, d_model=32, n_heads=2)
    q_ids = torch.randint(1, 50, (4, 10))
    c_ids = torch.randint(1, 20, (4, 10))
    correctness = torch.randint(0, 2, (4, 10)).float()
    mask = torch.ones((4, 10)).float()

    out = model(question_ids=q_ids, concept_ids=c_ids, correctness=correctness, mask=mask)
    assert "loss" in out
    assert "probs" in out
    assert out["probs"].shape == (4, 9)
