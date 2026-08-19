"""Unit tests for prompt formatting and XML structure (Listing 1 from paper)."""

import pytest
from data.prompt_formatter import (
    format_options,
    format_exercise_item,
    format_interaction,
    build_ntkt_prompt,
    INSTRUCTION_TEMPLATE,
)


def test_format_options_dict():
    opts = {"A": "10", "B": "5", "C": "9", "D": "7"}
    formatted = format_options(opts)
    assert formatted == "A) 10 B) 5 C) 9 D) 7"


def test_format_options_list():
    opts = ["10", "5", "9", "7"]
    formatted = format_options(opts)
    assert formatted == "A) 10 B) 5 C) 9 D) 7"


def test_exercise_item_full_text():
    item_str = format_exercise_item(
        question_text="What is 2 + 2?",
        options={"A": "3", "B": "4", "C": "5", "D": "6"},
        question_id=42,
        concept="Addition",
        tag_type="Q",
        ablation_mode="full_text"
    )
    assert "<Q>" in item_str
    assert "<text>What is 2 + 2?</text>" in item_str
    assert "<options>A) 3 B) 4 C) 5 D) 6</options>" in item_str
    assert "<QID>42</QID>" in item_str
    assert "<C>Addition</C>" in item_str
    assert "</Q>" in item_str


def test_exercise_item_concept_only():
    item_str = format_exercise_item(
        question_text="What is 2 + 2?",
        options={"A": "3", "B": "4"},
        question_id=42,
        concept="Addition",
        tag_type="Q",
        ablation_mode="concept_only"
    )
    assert "<text>" not in item_str
    assert "<options>" not in item_str
    assert "<QID>42</QID>" in item_str
    assert "<C>Addition</C>" in item_str


def test_exercise_item_id_only():
    item_str = format_exercise_item(
        question_text="What is 2 + 2?",
        options={"A": "3", "B": "4"},
        question_id=42,
        concept="Addition",
        tag_type="Q",
        ablation_mode="id_only"
    )
    assert "<text>" not in item_str
    assert "<options>" not in item_str
    assert "<C>" not in item_str
    assert "<QID>42</QID>" in item_str


def test_build_ntkt_prompt_structure():
    history = [
        {
            "question_id": 1,
            "question_text": "Sample Q1",
            "options": {"A": "1", "B": "2"},
            "concept": "Algebra",
            "is_correct": 1
        }
    ]
    target = {
        "question_id": 2,
        "question_text": "Target Q2",
        "options": {"A": "10", "B": "20"},
        "concept": "Geometry",
        "is_correct": 0
    }

    prompt, completion = build_ntkt_prompt(history, target, ablation_mode="full_text")

    assert INSTRUCTION_TEMPLATE in prompt
    assert "<history>:" in prompt
    assert "<Q>" in prompt
    assert "<cr>Correct</cr>" in prompt
    assert "</history>" in prompt
    assert "What do you predict they will answer for the target question:" in prompt
    assert "<target>" in prompt
    assert "<text>Target Q2</text>" in prompt
    assert "</target>: <cr>" in prompt
    assert completion == "Incorrect</cr>"
