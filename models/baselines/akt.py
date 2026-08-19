"""Attentive Knowledge Tracing (AKT) Baseline Model (Ghosh et al. 2020).

Implements Transformer-style attention over past interactions with monotonic decay
and Rasch concept/question representations.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional


class AKTAttentionLayer(nn.Module):
    """Multi-head Attention with learnable temporal decay."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        # Learnable decay parameter gamma
        self.gamma = nn.Parameter(torch.ones(n_heads, 1, 1) * 0.1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = q.shape

        q_proj = self.w_q(q).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k_proj = self.w_k(k).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v_proj = self.w_v(v).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply causal mask (cannot attend to future interactions)
        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask == 0, -1e9)

        # Distance matrix for decay
        positions = torch.arange(seq_len, device=q.device)
        dist = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1)).float()
        dist = dist.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        decay = torch.clamp(self.gamma, min=0.0) * dist
        scores = scores - decay

        attn_probs = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(attn_probs, v_proj)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.w_o(context)


class AKT(nn.Module):
    """Context-Aware Attentive Knowledge Tracing."""

    def __init__(
        self,
        num_questions: int = 2000,
        num_concepts: int = 600,
        d_model: int = 128,
        n_heads: int = 4,
        num_blocks: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_questions = num_questions
        self.num_concepts = num_concepts
        self.d_model = d_model

        # Embeddings
        self.q_embed = nn.Embedding(num_questions + 1, d_model, padding_idx=0)
        self.c_embed = nn.Embedding(num_concepts + 1, d_model, padding_idx=0)
        self.ans_embed = nn.Embedding(3, d_model, padding_idx=0)  # 0: pad, 1: incorrect, 2: correct

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
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = question_ids.shape

        if concept_ids is None:
            concept_ids = torch.zeros_like(question_ids)

        # Question representation
        q_emb = self.q_embed(question_ids) + self.c_embed(concept_ids)

        # Interaction representation (question + response)
        ans_labels = torch.where(question_ids > 0, (correctness.long() + 1), torch.zeros_like(question_ids))
        inter_emb = q_emb + self.ans_embed(ans_labels)

        # Causal mask: strictly lower triangular (cannot attend to future or current step's label)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=question_ids.device)).unsqueeze(0).unsqueeze(0)
        if mask is not None:
            seq_mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
            causal_mask = causal_mask * seq_mask

        # Self-attention over interactions
        h = inter_emb
        for attn, norm in zip(self.attn_layers, self.layer_norms):
            attn_out = attn(q=q_emb, k=h, v=h, causal_mask=causal_mask)
            h = norm(h + attn_out)

        # Concatenate knowledge state with target question embedding
        concat_rep = torch.cat([h, q_emb], dim=-1)
        logits = self.out_linear(concat_rep).squeeze(-1)
        probs = torch.sigmoid(logits)

        # Predict target at step t from history up to t-1
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
