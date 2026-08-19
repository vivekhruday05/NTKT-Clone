"""Prompt formatting module for Next Token Knowledge Tracing (NTKT).

Implements the exact XML-style prompt structure from Listing 1 of the paper:
'Next Token Knowledge Tracing: Exploiting Pretrained LLM Representations to Decode Student Behaviour'
Supports full-text, concept-only, and ID-only feature ablations.
"""

from typing import List, Dict, Any, Optional, Tuple


INSTRUCTION_TEMPLATE = (
    "Given the following student question and answer history, predict whether the student will "
    "answer the target question correctly or incorrectly. The target question is enclosed in "
    "<target> tags and the options are enclosed in <options> tags. Respond with ''Correct'' if "
    "you think they will answer correctly, or ''Incorrect'' if you think they will answer incorrectly.\n"
)


def format_options(options: Any) -> str:
    """Format multiple choice options into a clean string representation.
    
    Args:
        options: Dict mapping option keys (e.g. 'A', 'B', 'C', 'D') to option text,
                 or a list of option strings, or a pre-formatted string.
                 
    Returns:
        Formatted options string (e.g., 'A) 10 B) 5 C) 9 D) 7').
    """
    if isinstance(options, dict):
        return " ".join([f"{k}) {v}".strip() for k, v in options.items()])
    elif isinstance(options, list):
        keys = ["A", "B", "C", "D", "E", "F"]
        return " ".join([f"{keys[i] if i < len(keys) else i+1}) {opt}".strip() for i, opt in enumerate(options)])
    elif isinstance(options, str):
        return options.strip()
    return ""


def format_exercise_item(
    question_text: Optional[str],
    options: Any,
    question_id: Any,
    concept: Optional[str],
    tag_type: str = "Q",
    ablation_mode: str = "full_text"
) -> str:
    """Format an individual exercise item inside XML tags.
    
    Args:
        question_text: Natural language question text.
        options: Dict/list/str of multiple choice options.
        question_id: Identifier of the question (e.g., integer or string).
        concept: Knowledge component or concept tag.
        tag_type: XML tag name ('Q' for history items, 'target' for target item).
        ablation_mode: One of 'full_text', 'concept_only', 'id_only'.
        
    Returns:
        XML formatted string block.
    """
    lines = [f"<{tag_type}>"]
    
    if ablation_mode == "full_text":
        q_text = (question_text or "").strip()
        lines.append(f"<text>{q_text}</text>")
        
        opt_text = format_options(options)
        if opt_text:
            lines.append(f"<options>{opt_text}</options>")
            
        lines.append(f"<QID>{question_id}</QID>")
        
        c_text = (concept or "General").strip()
        lines.append(f"<C>{c_text}</C>")
        
    elif ablation_mode == "concept_only":
        lines.append(f"<QID>{question_id}</QID>")
        c_text = (concept or "General").strip()
        lines.append(f"<C>{c_text}</C>")
        
    elif ablation_mode == "id_only":
        lines.append(f"<QID>{question_id}</QID>")
        
    else:
        raise ValueError(f"Unknown ablation_mode: {ablation_mode}. Must be one of ['full_text', 'concept_only', 'id_only'].")
        
    lines.append(f"</{tag_type}>")
    return "\n".join(lines)


def format_interaction(
    interaction: Dict[str, Any],
    ablation_mode: str = "full_text"
) -> str:
    """Format a historical interaction containing exercise and correctness label.
    
    Args:
        interaction: Dict containing 'question_text', 'options', 'question_id', 'concept', 'is_correct'.
        ablation_mode: One of 'full_text', 'concept_only', 'id_only'.
        
    Returns:
        Formatted historical interaction string ending with <cr>Correct/Incorrect</cr>.
    """
    q_str = format_exercise_item(
        question_text=interaction.get("question_text", ""),
        options=interaction.get("options", {}),
        question_id=interaction.get("question_id", "0"),
        concept=interaction.get("concept", "General"),
        tag_type="Q",
        ablation_mode=ablation_mode
    )
    
    is_correct = interaction.get("is_correct")
    if is_correct is None:
        cr_label = "Correct"
    elif isinstance(is_correct, (int, bool)):
        cr_label = "Correct" if bool(is_correct) else "Incorrect"
    elif isinstance(is_correct, str):
        cr_label = "Correct" if is_correct.lower().startswith("c") or is_correct == "1" else "Incorrect"
    else:
        cr_label = "Correct" if bool(is_correct) else "Incorrect"
        
    return f"{q_str}<cr>{cr_label}</cr>"


def build_ntkt_prompt(
    history: List[Dict[str, Any]],
    target_exercise: Dict[str, Any],
    ablation_mode: str = "full_text",
    max_history_len: Optional[int] = None,
    include_target_label: bool = True
) -> Tuple[str, str]:
    """Build the complete NTKT input prompt and target completion string.
    
    Args:
        history: Chronological list of historical interaction dicts.
        target_exercise: Dict of target exercise to predict.
        ablation_mode: 'full_text', 'concept_only', or 'id_only'.
        max_history_len: Optional max number of recent interactions to keep.
        include_target_label: If True, returns target outcome label string.
        
    Returns:
        Tuple of (full_prompt_text, target_label_str)
        where full_prompt_text ends with `<target>...</target>: <cr>` (or ` `)
        and target_label_str is `Correct</cr>` / `Incorrect</cr>` or `Correct` / `Incorrect`.
    """
    if max_history_len is not None and len(history) > max_history_len:
        history = history[-max_history_len:]
        
    prompt_parts = [INSTRUCTION_TEMPLATE, "<history>:"]
    for item in history:
        prompt_parts.append(format_interaction(item, ablation_mode=ablation_mode))
    prompt_parts.append("</history>\n")
    
    prompt_parts.append("What do you predict they will answer for the target question:")
    target_str = format_exercise_item(
        question_text=target_exercise.get("question_text", ""),
        options=target_exercise.get("options", {}),
        question_id=target_exercise.get("question_id", "0"),
        concept=target_exercise.get("concept", "General"),
        tag_type="target",
        ablation_mode=ablation_mode
    )
    prompt_parts.append(f"{target_str}: <cr>")
    
    prompt_text = "\n".join(prompt_parts)
    
    target_is_correct = target_exercise.get("is_correct")
    if target_is_correct is None:
        target_label = "Correct"
    elif isinstance(target_is_correct, (int, bool)):
        target_label = "Correct" if bool(target_is_correct) else "Incorrect"
    elif isinstance(target_is_correct, str):
        target_label = "Correct" if target_is_correct.lower().startswith("c") or target_is_correct == "1" else "Incorrect"
    else:
        target_label = "Correct" if bool(target_is_correct) else "Incorrect"
        
    completion_text = f"{target_label}</cr>"
    
    return prompt_text, completion_text
