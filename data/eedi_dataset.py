"""Eedi Dataset Loader and Sequence Processor for Knowledge Tracing.

Supports:
- Eedi NeurIPS 2020 Education Challenge format
- Standard CSV datasets with Question Text, Concept, and Student Sequences
- Full Text, Concept-Only, and ID-Only feature ablations (RQ2)
- Seen split, User Cold-Start, and Question Cold-Start splits (RQ3)
- Conversion to NTKT prompt pairs and numerical tensors for baseline models (DKT, AKT, DTransformer).
"""

import os
import json
import ast
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional, Tuple, Set

from data.prompt_formatter import build_ntkt_prompt


class EediSequenceDataset(Dataset):
    """PyTorch Dataset for NTKT language modeling with causal next-token prediction."""

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        ablation_mode: str = "full_text",
        max_history_len: Optional[int] = 50,
    ):
        """
        Args:
            samples: List of dicts, each having 'history' (list of past items),
                     'target_exercise' (dict of target item), 'user_id', 'timestep'.
            ablation_mode: 'full_text', 'concept_only', or 'id_only'.
            max_history_len: Maximum number of previous exercises to include in prompt.
        """
        self.samples = samples
        self.ablation_mode = ablation_mode
        self.max_history_len = max_history_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        history = item["history"]
        target = item["target_exercise"]

        prompt_text, completion_text = build_ntkt_prompt(
            history=history,
            target_exercise=target,
            ablation_mode=self.ablation_mode,
            max_history_len=self.max_history_len,
            include_target_label=True
        )

        is_correct = target.get("is_correct", 1)
        if isinstance(is_correct, str):
            is_correct = 1 if is_correct.lower().startswith("c") or is_correct == "1" else 0

        return {
            "prompt": prompt_text,
            "completion": completion_text,
            "is_correct": int(is_correct),
            "user_id": item.get("user_id", 0),
            "question_id": target.get("question_id", 0),
            "timestep": item.get("timestep", 0),
        }


class BaselineSequenceDataset(Dataset):
    """PyTorch Dataset for traditional numeric KT models (DKT, AKT, DTransformer)."""

    def __init__(
        self,
        student_sequences: List[Dict[str, Any]],
        max_seq_len: int = 100,
        num_questions: int = 2000,
        num_concepts: int = 600,
        text_embeddings: Optional[Dict[Any, np.ndarray]] = None
    ):
        self.sequences = student_sequences
        self.max_seq_len = max_seq_len
        self.num_questions = num_questions
        self.num_concepts = num_concepts
        self.text_embeddings = text_embeddings

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        q_ids = seq["question_ids"][:self.max_seq_len]
        c_ids = seq.get("concept_ids", [0] * len(q_ids))[:self.max_seq_len]
        corrects = seq["correctness"][:self.max_seq_len]

        seq_len = len(q_ids)
        pad_len = self.max_seq_len - seq_len

        # Pad with 0
        padded_q = q_ids + [0] * pad_len
        padded_c = c_ids + [0] * pad_len
        padded_a = corrects + [0] * pad_len
        mask = [1] * seq_len + [0] * pad_len

        data = {
            "question_ids": torch.tensor(padded_q, dtype=torch.long),
            "concept_ids": torch.tensor(padded_c, dtype=torch.long),
            "correctness": torch.tensor(padded_a, dtype=torch.float),
            "mask": torch.tensor(mask, dtype=torch.float),
            "seq_len": torch.tensor(seq_len, dtype=torch.long),
        }

        if self.text_embeddings is not None:
            embed_dim = next(iter(self.text_embeddings.values())).shape[0]
            emb_matrix = np.zeros((self.max_seq_len, embed_dim), dtype=np.float32)
            for i, qid in enumerate(q_ids):
                if qid in self.text_embeddings:
                    emb_matrix[i] = self.text_embeddings[qid]
            data["text_embeddings"] = torch.tensor(emb_matrix, dtype=torch.float)

        return data


def parse_options_field(opt_raw: Any) -> Dict[str, str]:
    """Safely parse options from string/JSON/dict."""
    if isinstance(opt_raw, dict):
        return {str(k): str(v) for k, v in opt_raw.items()}
    if pd.isna(opt_raw) or opt_raw == "":
        return {"A": "", "B": "", "C": "", "D": ""}
    if isinstance(opt_raw, str):
        try:
            parsed = json.loads(opt_raw)
            if isinstance(parsed, dict):
                return {str(k): str(v) for k, v in parsed.items()}
        except Exception:
            pass
        try:
            parsed = ast.literal_eval(opt_raw)
            if isinstance(parsed, dict):
                return {str(k): str(v) for k, v in parsed.items()}
        except Exception:
            pass
        return {"A": opt_raw}
    return {"A": "", "B": "", "C": "", "D": ""}


def load_eedi_raw_data(
    data_path: str,
    question_meta_path: Optional[str] = None
) -> pd.DataFrame:
    """Load and standardize interaction data from CSV or directory."""
    if os.path.isdir(data_path):
        train_cand = os.path.join(data_path, "train_interactions.csv")
        if not os.path.exists(train_cand):
            train_cand = os.path.join(data_path, "train_task_3_4.csv")
        data_path = train_cand

    df = pd.read_csv(data_path)

    # Normalize column names
    col_map = {
        "user_id": "UserId",
        "student_id": "UserId",
        "UserId": "UserId",
        "question_id": "QuestionId",
        "QuestionId": "QuestionId",
        "is_correct": "IsCorrect",
        "IsCorrect": "IsCorrect",
        "CorrectAnswer": "CorrectAnswer",
        "AnswerValue": "AnswerValue",
        "timestamp": "Timestamp",
        "DateAnswered": "Timestamp",
        "Timestamp": "Timestamp",
        "question_text": "QuestionText",
        "QuestionText": "QuestionText",
        "construct_name": "ConstructName",
        "ConstructName": "ConstructName",
        "options": "Options",
        "Options": "Options",
    }
    df = df.rename(columns={c: col_map[c] for c in df.columns if c in col_map})

    if "IsCorrect" not in df.columns:
        if "CorrectAnswer" in df.columns and "AnswerValue" in df.columns:
            df["IsCorrect"] = (df["CorrectAnswer"] == df["AnswerValue"]).astype(int)
        else:
            df["IsCorrect"] = 1

    # Merge question metadata if provided and columns are missing
    if question_meta_path and os.path.exists(question_meta_path):
        q_df = pd.read_csv(question_meta_path)
        q_col_map = {
            "question_id": "QuestionId",
            "QuestionId": "QuestionId",
            "question_text": "QuestionText",
            "QuestionText": "QuestionText",
            "construct_name": "ConstructName",
            "ConstructName": "ConstructName",
            "options": "Options",
            "Options": "Options",
        }
        q_df = q_df.rename(columns={c: q_col_map[c] for c in q_df.columns if c in q_col_map})
        
        merge_cols = [c for c in ["QuestionText", "ConstructName", "Options"] if c in q_df.columns and c not in df.columns]
        if merge_cols:
            df = df.merge(q_df[["QuestionId"] + merge_cols], on="QuestionId", how="left")

    if "QuestionText" not in df.columns:
        df["QuestionText"] = df["QuestionId"].apply(lambda qid: f"Question {qid}")
    if "ConstructName" not in df.columns:
        df["ConstructName"] = "Mathematics"
    if "Options" not in df.columns:
        df["Options"] = "{}"

    if "Timestamp" not in df.columns:
        df["Timestamp"] = range(len(df))

    return df


def prepare_student_histories(
    df: pd.DataFrame
) -> Dict[Any, List[Dict[str, Any]]]:
    """Group interactions by student and order chronologically."""
    df_sorted = df.sort_values(by=["UserId", "Timestamp"]).reset_index(drop=True)
    
    student_records = {}
    for _, row in df_sorted.iterrows():
        uid = row["UserId"]
        if uid not in student_records:
            student_records[uid] = []
            
        options_dict = parse_options_field(row.get("Options", "{}"))
        
        student_records[uid].append({
            "question_id": row["QuestionId"],
            "question_text": str(row.get("QuestionText", f"Question {row['QuestionId']}")),
            "options": options_dict,
            "concept": str(row.get("ConstructName", "General")),
            "is_correct": int(row["IsCorrect"]),
            "timestamp": row.get("Timestamp", 0),
        })
        
    return student_records


def build_stepwise_samples(
    student_records: Dict[Any, List[Dict[str, Any]]],
    min_history: int = 1,
    max_steps_per_user: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Convert student histories into step-by-step training/evaluation instances.
    
    For each user with sequence [e_1, ..., e_T]:
      at step t: history is [e_1, ..., e_{t-1}], target is e_t.
    """
    samples = []
    for uid, records in student_records.items():
        if len(records) <= min_history:
            continue
        
        steps = range(min_history, len(records))
        if max_steps_per_user is not None and len(steps) > max_steps_per_user:
            # Sample evenly across history
            indices = np.linspace(min_history, len(records) - 1, max_steps_per_user, dtype=int)
            steps = indices
            
        for t in steps:
            history = records[:t]
            target = records[t]
            samples.append({
                "user_id": uid,
                "timestep": t,
                "history": history,
                "target_exercise": target,
            })
            
    return samples


def split_eedi_data(
    df: pd.DataFrame,
    test_ratio: float = 0.1,
    cold_start_question_count: int = 0,
    seed: int = 42
) -> Dict[str, pd.DataFrame]:
    """Split data into train, test (seen / user cold-start), and question cold-start sets.
    
    Args:
        df: Processed interaction DataFrame.
        test_ratio: Ratio of learners for user cold-start / hold-out validation.
        cold_start_question_count: Number of questions to completely withhold for RQ3.
        seed: Random seed.
    """
    np.random.seed(seed)
    
    # 1. Withhold cold-start questions if requested (RQ3)
    if cold_start_question_count > 0:
        all_questions = df["QuestionId"].unique()
        cold_qids = set(np.random.choice(all_questions, size=min(cold_start_question_count, len(all_questions)), replace=False))
        question_cold_start_df = df[df["QuestionId"].isin(cold_qids)].copy()
        remaining_df = df[~df["QuestionId"].isin(cold_qids)].copy()
    else:
        cold_qids = set()
        question_cold_start_df = pd.DataFrame()
        remaining_df = df.copy()

    # 2. Hold-out 90% train students, 10% test students (user cold-start)
    all_users = remaining_df["UserId"].unique()
    np.random.shuffle(all_users)
    
    split_point = int((1.0 - test_ratio) * len(all_users))
    train_users = set(all_users[:split_point])
    test_users = set(all_users[split_point:])
    
    train_df = remaining_df[remaining_df["UserId"].isin(train_users)].copy()
    user_cold_start_df = remaining_df[remaining_df["UserId"].isin(test_users)].copy()
    
    return {
        "train": train_df,
        "user_cold_start_test": user_cold_start_df,
        "question_cold_start_test": question_cold_start_df,
        "cold_question_ids": list(cold_qids),
    }
