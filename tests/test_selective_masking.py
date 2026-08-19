"""Unit tests for NTKT Data Collator and Selective Loss Masking (Eq. 3)."""

import pytest
import torch
from transformers import AutoTokenizer

from data.collator import NTKTDataCollator


class MockTokenizer:
    """Lightweight mock tokenizer for testing collation and masking without downloading weights."""
    def __init__(self):
        self.pad_token = "<|pad|>"
        self.pad_token_id = 0
        self.eos_token = "<|eos|>"
        self.eos_token_id = 1
        self.vocab = {"<|pad|>": 0, "<|eos|>": 1, "Correct": 10, "Incorrect": 11, "</cr>": 12, "<cr>": 13}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def __call__(self, text, add_special_tokens=True, truncation=False, return_tensors=None):
        tokens = text.split()
        ids = [self.vocab.get(t, hash(t) % 1000 + 20) for t in tokens]
        return {"input_ids": ids}

    def encode(self, text, add_special_tokens=False):
        tokens = text.split()
        return [self.vocab.get(t, hash(t) % 1000 + 20) for t in tokens]

    def decode(self, token_ids):
        return " ".join([self.inv_vocab.get(tid, f"tok_{tid}") for tid in token_ids])


def test_collator_selective_masking():
    tokenizer = MockTokenizer()
    collator = NTKTDataCollator(tokenizer=tokenizer, max_length=128)

    sample = {
        "prompt": "Instruction and student history ... target question <cr> ",
        "completion": "Correct </cr>",
        "is_correct": 1
    }

    batch = collator([sample])

    assert "input_ids" in batch
    assert "labels" in batch
    assert "attention_mask" in batch
    assert "target_labels" in batch

    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]

    # Prompt tokens should all be masked to -100
    prompt_token_count = len(sample["prompt"].split())
    for i in range(prompt_token_count):
        assert labels[i].item() == -100, f"Token at position {i} should be masked with -100"

    # Completion tokens (Correct, </cr>) should be unmasked (equal to their token IDs)
    completion_token_count = len(sample["completion"].split())
    for i in range(prompt_token_count, prompt_token_count + completion_token_count):
        assert labels[i].item() != -100, f"Token at position {i} should NOT be masked"
        assert labels[i].item() == input_ids[i].item()

    assert batch["target_labels"][0].item() == 1


def test_collator_batch_padding():
    tokenizer = MockTokenizer()
    collator = NTKTDataCollator(tokenizer=tokenizer, max_length=128, pad_to_multiple_of=8)

    samples = [
        {"prompt": "Short prompt <cr>", "completion": "Correct </cr>", "is_correct": 1},
        {"prompt": "A much longer prompt with more student interaction history details <cr>", "completion": "Incorrect </cr>", "is_correct": 0}
    ]

    batch = collator(samples)
    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape[1] % 8 == 0
    assert batch["labels"].shape == batch["input_ids"].shape
