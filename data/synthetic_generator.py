"""Synthetic Eedi-format dataset generator for fast local CPU testing and validation.

Generates realistic math word problems, options, concepts, and student interaction sequences
without needing external downloads or GPU resources.
"""

import random
import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple


SAMPLE_MATH_TOPICS = [
    ("Negative Numbers", "Asha's office is on level {val1}. Her car is parked in the basement on level -{val2}. How many floors does Asha need to go down to get from the office to her car?",
     lambda v1, v2: v1 + v2),
    ("Fractions", "What is {val1}/{val2} + {val3}/{val2} simplified to its lowest terms?",
     lambda v1, v2, v3: f"{v1+v3}/{v2}"),
    ("Linear Equations", "Solve for x: {val1}x + {val2} = {val3}.",
     lambda v1, v2, v3: (v3 - v2) // v1 if (v3 - v2) % v1 == 0 else f"{(v3-v2)}/{v1}"),
    ("Percentages", "What is {val1}% of {val2}?",
     lambda v1, v2: (v1 * v2) // 100),
    ("Geometry & Angles", "In a triangle, angle A is {val1} degrees and angle B is {val2} degrees. What is angle C?",
     lambda v1, v2: 180 - (v1 + v2)),
    ("Ratios", "A recipe uses flour and sugar in the ratio {val1}:{val2}. If you use {val3} cups of flour, how many cups of sugar are needed?",
     lambda v1, v2, v3: (v2 * v3) // v1),
    ("Probability", "A bag contains {val1} red marbles and {val2} blue marbles. What is the probability of picking a red marble?",
     lambda v1, v2: f"{val1}/{v1+v2}"),
    ("Exponents & Powers", "What is the value of {val1}^{val2}?",
     lambda v1, v2: v1 ** v2),
]


def generate_synthetic_question(question_id: int) -> Dict[str, Any]:
    """Generate a single realistic math question with options and correct answer."""
    topic, template, answer_fn = random.choice(SAMPLE_MATH_TOPICS)
    
    v1 = random.randint(2, 9)
    v2 = random.randint(2, 8)
    v3 = random.randint(10, 30)
    
    if "val3" in template:
        q_text = template.format(val1=v1, val2=v2, val3=v3)
        try:
            correct_ans = str(answer_fn(v1, v2, v3))
        except Exception:
            correct_ans = "12"
    else:
        q_text = template.format(val1=v1, val2=v2)
        try:
            correct_ans = str(answer_fn(v1, v2))
        except Exception:
            correct_ans = "10"
            
    # Generate distractor options
    distractors = set()
    num_tries = 0
    while len(distractors) < 3 and num_tries < 20:
        num_tries += 1
        try:
            val = int(correct_ans)
            d = str(val + random.choice([-3, -2, -1, 1, 2, 3, 5]))
        except ValueError:
            d = f"{random.randint(1, 10)}/{random.randint(2, 12)}"
        if d != correct_ans:
            distractors.add(d)
            
    while len(distractors) < 3:
        distractors.add(f"distractor_{len(distractors)+1}")
        
    all_options_list = [correct_ans] + list(distractors)
    random.shuffle(all_options_list)
    
    option_keys = ["A", "B", "C", "D"]
    options_dict = {k: opt for k, opt in zip(option_keys, all_options_list)}
    correct_key = [k for k, v in options_dict.items() if v == correct_ans][0]
    
    return {
        "question_id": question_id,
        "question_text": q_text,
        "options": options_dict,
        "correct_answer": correct_key,
        "concept": topic,
    }


def generate_synthetic_dataset(
    output_dir: str,
    num_students: int = 50,
    num_questions: int = 40,
    min_interactions_per_student: int = 5,
    max_interactions_per_student: int = 25,
    seed: int = 42
) -> Tuple[str, str]:
    """Generate a synthetic Eedi-style dataset saved as train and test CSV files.
    
    Args:
        output_dir: Directory where CSV files will be saved.
        num_students: Number of distinct learners.
        num_questions: Number of distinct exercises.
        min_interactions_per_student: Minimum sequence length.
        max_interactions_per_student: Maximum sequence length.
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple of (train_csv_path, test_csv_path).
    """
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Create Question Bank
    questions = [generate_synthetic_question(qid) for qid in range(1, num_questions + 1)]
    questions_df = pd.DataFrame([
        {
            "QuestionId": q["question_id"],
            "QuestionText": q["question_text"],
            "Options": json.dumps(q["options"]),
            "CorrectAnswer": q["correct_answer"],
            "ConstructName": q["concept"],
        }
        for q in questions
    ])
    q_meta_path = os.path.join(output_dir, "question_metadata.csv")
    questions_df.to_csv(q_meta_path, index=False)
    
    # 2. Simulate Student Interactions with latent student abilities
    all_interactions = []
    
    for uid in range(1, num_students + 1):
        # Latent student ability in range [0.2, 0.85]
        student_ability = random.uniform(0.2, 0.85)
        seq_len = random.randint(min_interactions_per_student, max_interactions_per_student)
        
        # Sample questions without immediate repetition
        sampled_qids = random.sample(range(1, num_questions + 1), min(seq_len, num_questions))
        
        for step, qid in enumerate(sampled_qids):
            q_info = questions[qid - 1]
            
            # Probability of answering correctly increases with student ability
            p_correct = min(0.95, max(0.05, student_ability + random.gauss(0.0, 0.15)))
            is_correct = 1 if random.random() < p_correct else 0
            
            all_interactions.append({
                "UserId": uid,
                "QuestionId": qid,
                "Timestamp": 1600000000 + step * 60,
                "IsCorrect": is_correct,
                "QuestionText": q_info["question_text"],
                "Options": json.dumps(q_info["options"]),
                "ConstructName": q_info["concept"],
            })
            
    df_all = pd.DataFrame(all_interactions)
    
    # 3. Split 90% students for train, 10% for test (as per paper standard)
    student_ids = list(range(1, num_students + 1))
    random.shuffle(student_ids)
    
    split_idx = int(0.9 * len(student_ids))
    train_uids = set(student_ids[:split_idx])
    test_uids = set(student_ids[split_idx:])
    
    train_df = df_all[df_all["UserId"].isin(train_uids)].sort_values(by=["UserId", "Timestamp"]).reset_index(drop=True)
    test_df = df_all[df_all["UserId"].isin(test_uids)].sort_values(by=["UserId", "Timestamp"]).reset_index(drop=True)
    
    train_path = os.path.join(output_dir, "train_interactions.csv")
    test_path = os.path.join(output_dir, "test_interactions.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Synthetic dataset generated successfully at {output_dir}:")
    print(f"  - Train interactions: {len(train_df)} ({len(train_uids)} students)")
    print(f"  - Test interactions: {len(test_df)} ({len(test_uids)} students)")
    print(f"  - Unique questions: {len(questions)}")
    
    return train_path, test_path
