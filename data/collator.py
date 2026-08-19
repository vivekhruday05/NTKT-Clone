"""Data Collator for Next Token Knowledge Tracing (NTKT) with Selective Loss Masking.

Implements the selective loss masking described in Section 'NTKT Models' of the paper:
Equation (3): Only tokens corresponding to the outcome ('Correct'/'Incorrect') in <cr>...</cr>
are unmasked in the loss computation, while all other tokens receive the sentinel label -100.
"""

from typing import List, Dict, Any, Optional
import torch


class NTKTDataCollator:
    """Data collator that dynamically tokenizes, pads, and creates selective loss masks."""
    
    def __init__(
        self,
        tokenizer: Any,
        max_length: int = 4096,
        mask_history_cr: bool = True,
        pad_to_multiple_of: Optional[int] = 8
    ):
        """
        Args:
            tokenizer: Pretrained HuggingFace tokenizer.
            max_length: Maximum token length before truncation.
            mask_history_cr: If True, only computes loss on the final target outcome <cr>...</cr>.
                             If False, computes loss on all <cr>...</cr> spans in the sequence.
            pad_to_multiple_of: Pad sequence length to a multiple of this value (e.g. 8 for Tensor Cores).
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_history_cr = mask_history_cr
        self.pad_to_multiple_of = pad_to_multiple_of
        
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate a batch of prompt and completion pairs or pre-built strings.
        
        Features can contain:
          - 'prompt': Full prompt string preceding the target outcome.
          - 'completion': Target outcome string (e.g., 'Correct</cr>' or 'Incorrect</cr>').
          - 'is_correct': Binary label 0 or 1.
          OR
          - 'full_text': Combined string of prompt + completion.
        """
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        batch_target_labels = []

        for item in features:
            if "prompt" in item and "completion" in item:
                prompt_text = item["prompt"]
                completion_text = item["completion"]
                full_text = prompt_text + completion_text
                
                # Tokenize prompt and full text to identify the completion token boundaries
                prompt_enc = self.tokenizer(
                    prompt_text,
                    add_special_tokens=True,
                    truncation=False,
                    return_tensors=None
                )
                prompt_ids = prompt_enc["input_ids"]
                
                full_enc = self.tokenizer(
                    full_text,
                    add_special_tokens=True,
                    truncation=False,
                    return_tensors=None
                )
                full_ids = full_enc["input_ids"]
                
                # If full_ids exceeds max_length, truncate from the left (keep prompt suffix & target)
                if len(full_ids) > self.max_length:
                    excess = len(full_ids) - self.max_length
                    full_ids = full_ids[excess:]
                    prompt_len = max(0, len(prompt_ids) - excess)
                else:
                    prompt_len = len(prompt_ids)

                labels = [-100] * len(full_ids)
                
                # Unmask only the tokens corresponding to the completion (the outcome)
                for idx in range(prompt_len, len(full_ids)):
                    token_id = full_ids[idx]
                    # Token ID of outcome or cr tag
                    labels[idx] = token_id

                # Extract target binary correctness (1 for Correct, 0 for Incorrect)
                is_correct = item.get("is_correct")
                if is_correct is None:
                    is_correct = 1 if "correct" in completion_text.lower() and "incorrect" not in completion_text.lower() else 0

            elif "full_text" in item:
                full_text = item["full_text"]
                full_enc = self.tokenizer(
                    full_text,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors=None
                )
                full_ids = full_enc["input_ids"]
                labels = [-100] * len(full_ids)
                
                # Selective masking by locating target <cr> tag
                cr_open_ids = self.tokenizer.encode("<cr>", add_special_tokens=False)
                cr_close_ids = self.tokenizer.encode("</cr>", add_special_tokens=False)
                
                # Unmask tokens between <cr> and </cr>
                # Find all occurrences of <cr> ... </cr>
                in_target_cr = False
                for idx, tid in enumerate(full_ids):
                    # Simple heuristic: if token represents 'Correct' or 'Incorrect'
                    tok_str = self.tokenizer.decode([tid]).strip().lower()
                    if tok_str in ["correct", "incorrect", "Ġcorrect", "Ġincorrect", " correct", " incorrect"]:
                        labels[idx] = tid
                        
                is_correct = item.get("is_correct", 1)
            else:
                raise ValueError("Item in batch must contain either ('prompt', 'completion') or 'full_text'.")

            batch_input_ids.append(torch.tensor(full_ids, dtype=torch.long))
            batch_attention_mask.append(torch.ones(len(full_ids), dtype=torch.long))
            batch_labels.append(torch.tensor(labels, dtype=torch.long))
            batch_target_labels.append(int(is_correct))

        # Pad sequences to max length in this batch
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        
        max_batch_len = max(len(seq) for seq in batch_input_ids)
        if self.pad_to_multiple_of is not None:
            max_batch_len = ((max_batch_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
            
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for seq_ids, seq_mask, seq_lab in zip(batch_input_ids, batch_attention_mask, batch_labels):
            pad_len = max_batch_len - len(seq_ids)
            if pad_len > 0:
                # Right padding
                padded_input_ids.append(torch.cat([seq_ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)]))
                padded_attention_mask.append(torch.cat([seq_mask, torch.zeros(pad_len, dtype=torch.long)]))
                padded_labels.append(torch.cat([seq_lab, torch.full((pad_len,), -100, dtype=torch.long)]))
            else:
                padded_input_ids.append(seq_ids)
                padded_attention_mask.append(seq_mask)
                padded_labels.append(seq_lab)

        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "labels": torch.stack(padded_labels),
            "target_labels": torch.tensor(batch_target_labels, dtype=torch.long),
        }
