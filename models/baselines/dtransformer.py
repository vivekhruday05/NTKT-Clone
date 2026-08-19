"""DTransformer (Diagnostic Transformer) Baseline Model (Yin et al. 2023).

Decouples question difficulty representation and student latent proficiency states
with diagnostic attention and contrastive knowledge tracing.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional


class DTransformer(nn.Module):
    """Diagnostic Transformer for Stable Knowledge Tracing."""

    def __init__(
        self,
        num_questions: int = 2000,
        num_concepts: int = 600,
        d_model: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_questions = num_questions
        self.num_concepts = num_concepts
        self.d_model = d_model

        # Embeddings
        self.q_embed = nn.Embedding(num_questions + 1, d_model, padding_idx=0)
        self.c_embed = nn.Embedding(num_concepts + 1, d_model, padding_idx=0)
        self.ans_embed = nn.Embedding(3, d_model, padding_idx=0)

        # Transformer encoder layers for knowledge state
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True
        )
        self.knowledge_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Decoupled projection heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

    def forward(
        self,
        question_ids: torch.Tensor,
        concept_ids: Optional[torch.Tensor] = None,
        correctness: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = question_ids.shape

        if concept_ids is None:
            concept_ids = torch.zeros_like(question_ids)

        q_rep = self.q_embed(question_ids) + self.c_embed(concept_ids)
        ans_labels = torch.where(question_ids > 0, (correctness.long() + 1), torch.zeros_like(question_ids))
        inter_rep = q_rep + self.ans_embed(ans_labels)

        # Causal mask for transformer
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(question_ids.device)
        padding_mask = (mask == 0) if mask is not None else (question_ids == 0)

        # Diagnostic state extraction
        h = self.knowledge_encoder(inter_rep, mask=causal_mask, is_causal=True)

        k_state = self.k_proj(h)
        q_feat = self.q_proj(q_rep)

        combined = torch.cat([k_state, q_feat], dim=-1)
        logits = self.classifier(combined).squeeze(-1)
        probs = torch.sigmoid(logits)

        pred_logits = logits[:, :-1]
        target_correctness = correctness[:, 1:] if correctness is not None else None
        
        if mask is not None:
            target_mask = mask[:, 1:]
        else:
            target_mask = (question_ids[:, 1:] > 0).float()

        loss = None
        if target_correctness is not None and target_mask.sum() > 0:
            loss = F.binary_cross_entropy_with_logits(
                pred_logits,
                target_correctness,
                weight=target_mask,
                reduction="sum"
            ) / (target_mask.sum() + 1e-8)

        return {
            "loss": loss,
            "logits": pred_logits,
            "probs": probs[:, :-1],
            "mask": target_mask,
        }
