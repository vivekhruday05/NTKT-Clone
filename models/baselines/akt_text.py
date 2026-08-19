"""AKT-text Baseline Model.

Augments Attentive Knowledge Tracing with dense semantic question embeddings
extracted using a pretrained sentence transformer (all-MiniLM-L6-v2, 384-dim).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
from models.baselines.akt import AKTAttentionLayer


class AKTText(nn.Module):
    """Text-Augmented Attentive Knowledge Tracing."""

    def __init__(
        self,
        num_questions: int = 2000,
        num_concepts: int = 600,
        text_dim: int = 384,
        d_model: int = 128,
        n_heads: int = 4,
        num_blocks: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_questions = num_questions
        self.num_concepts = num_concepts
        self.d_model = d_model
        self.text_dim = text_dim

        # Embeddings & Projections
        self.q_embed = nn.Embedding(num_questions + 1, d_model, padding_idx=0)
        self.c_embed = nn.Embedding(num_concepts + 1, d_model, padding_idx=0)
        self.ans_embed = nn.Embedding(3, d_model, padding_idx=0)
        self.text_proj = nn.Linear(text_dim, d_model)

        self.attn_layers = nn.ModuleList([
            AKTAttentionLayer(d_model, n_heads, dropout) for _ in range(num_blocks)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_blocks)])

        self.out_linear = nn.Sequential(
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
        text_embeddings: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = question_ids.shape

        if concept_ids is None:
            concept_ids = torch.zeros_like(question_ids)

        # Question representation combines ID, Concept, and Projected Sentence Transformer Text embedding
        q_emb = self.q_embed(question_ids) + self.c_embed(concept_ids)
        if text_embeddings is not None:
            proj_text = self.text_proj(text_embeddings)
            q_emb = q_emb + proj_text

        # Interaction representation
        ans_labels = torch.where(question_ids > 0, (correctness.long() + 1), torch.zeros_like(question_ids))
        inter_emb = q_emb + self.ans_embed(ans_labels)

        # Causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=question_ids.device)).unsqueeze(0).unsqueeze(0)
        if mask is not None:
            causal_mask = causal_mask * mask.unsqueeze(1).unsqueeze(2)

        # Self-attention
        h = inter_emb
        for attn, norm in zip(self.attn_layers, self.layer_norms):
            attn_out = attn(q=q_emb, k=h, v=h, causal_mask=causal_mask)
            h = norm(h + attn_out)

        concat_rep = torch.cat([h, q_emb], dim=-1)
        logits = self.out_linear(concat_rep).squeeze(-1)
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
