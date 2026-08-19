"""Deep Knowledge Tracing (DKT) Baseline Model (Piech et al. 2015).

Processes student interaction sequences using an LSTM network.
Input at each step is the combination of question ID and binary correctness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional


class DKT(nn.Module):
    """Deep Knowledge Tracing neural network."""

    def __init__(
        self,
        num_questions: int = 2000,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()
        self.num_questions = num_questions
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Input is interaction pair (question_id, correctness) -> 2 * num_questions combinations
        self.interaction_embedding = nn.Embedding(2 * num_questions + 1, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.out_linear = nn.Linear(hidden_dim, num_questions + 1)

    def forward(
        self,
        question_ids: torch.Tensor,
        correctness: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            question_ids: LongTensor of shape [batch_size, seq_len]
            correctness: FloatTensor of shape [batch_size, seq_len] with values in {0, 1}
            mask: FloatTensor of shape [batch_size, seq_len] indicating non-padded positions
            
        Returns:
            Dict containing 'logits', 'probs', 'loss'
        """
        batch_size, seq_len = question_ids.shape
        
        # Form interaction indices: index = question_id + correctness * num_questions
        # 0 is reserved for padding
        interaction_ids = torch.where(
            question_ids > 0,
            question_ids + (correctness.long() * self.num_questions),
            torch.zeros_like(question_ids)
        )

        embeds = self.interaction_embedding(interaction_ids)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)
        
        # Predictions for all questions at each timestep: [batch_size, seq_len, num_questions + 1]
        all_logits = self.out_linear(lstm_out)
        all_probs = torch.sigmoid(all_logits)

        # For step t, we predict correctness on question_ids[t+1] using hidden state from step t
        # Shift inputs
        target_qids = question_ids[:, 1:]
        target_correctness = correctness[:, 1:]
        
        if mask is not None:
            target_mask = mask[:, 1:]
        else:
            target_mask = (target_qids > 0).float()

        # Gather the predicted probability for the specific target question ID
        pred_logits_at_step = all_logits[:, :-1, :]  # [batch_size, seq_len - 1, num_questions + 1]
        gathered_logits = torch.gather(pred_logits_at_step, 2, target_qids.unsqueeze(2)).squeeze(2)
        gathered_probs = torch.sigmoid(gathered_logits)

        loss = None
        if target_correctness is not None and target_mask.sum() > 0:
            loss = F.binary_cross_entropy_with_logits(
                gathered_logits,
                target_correctness,
                weight=target_mask,
                reduction="sum"
            ) / (target_mask.sum() + 1e-8)

        return {
            "loss": loss,
            "logits": gathered_logits,
            "probs": gathered_probs,
            "mask": target_mask,
        }
