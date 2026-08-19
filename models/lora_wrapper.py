"""LoRA (Low-Rank Adaptation) wrapper for NTKT fine-tuning.

Supports:
1. HuggingFace PEFT (peft.LoraConfig, peft.get_peft_model)
2. Standalone Native PyTorch LoRA layer fallback (zero-dependency, robust across all PyTorch/Transformers versions).
"""

import math
import os
import torch
import torch.nn as nn
from typing import List, Optional, Set, Dict, Any


class NativeLoraLinear(nn.Module):
    """Native PyTorch implementation of Low-Rank Adaptation for Linear/Conv1D layers."""

    def __init__(
        self,
        original_linear: nn.Module,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.05
    ):
        super().__init__()
        # Determine in_features and out_features
        if hasattr(original_linear, "in_features") and hasattr(original_linear, "out_features"):
            self.in_features = original_linear.in_features
            self.out_features = original_linear.out_features
        elif hasattr(original_linear, "weight"):
            shape = original_linear.weight.shape
            if len(shape) == 2:
                # If Conv1D: weight is (nx, nf)
                if hasattr(original_linear, "nf"):
                    self.in_features = shape[0]
                    self.out_features = shape[1]
                else:
                    self.in_features = shape[1]
                    self.out_features = shape[0]
            else:
                self.in_features = shape[-1]
                self.out_features = shape[0]
        else:
            raise ValueError(f"Cannot determine dimensions of {type(original_linear)}")

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 1.0

        # Freeze base linear layer
        self.base_layer = original_linear
        for param in self.base_layer.parameters():
            param.requires_grad = False

        if rank > 0:
            self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
            self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
            self.reset_parameters()
        else:
            self.lora_A = None
            self.lora_B = None

    def reset_parameters(self):
        if self.rank > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        if self.rank > 0:
            lora_out = (self.lora_dropout(x) @ self.lora_A.T) @ self.lora_B.T * self.scaling
            return base_out + lora_out
        return base_out


def apply_native_lora(
    model: nn.Module,
    target_modules: Optional[List[str]] = None,
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.05
) -> nn.Module:
    """Inject NativeLoraLinear into target linear layers of a PyTorch module."""
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "c_attn", "c_proj", "c_fc"]

    # First freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    targets_to_replace = []
    for name, module in model.named_modules():
        for child_name, child_module in module.named_children():
            if isinstance(child_module, NativeLoraLinear):
                continue
            is_linear = isinstance(child_module, nn.Linear) or child_module.__class__.__name__ in ["Conv1D", "Linear"]
            if is_linear:
                full_name = f"{name}.{child_name}" if name else child_name
                if any(t in child_name for t in target_modules) or any(t in full_name for t in target_modules):
                    targets_to_replace.append((module, child_name, child_module))

    replaced_count = 0
    for parent, child_name, child_module in targets_to_replace:
        lora_layer = NativeLoraLinear(
            original_linear=child_module,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
        setattr(parent, child_name, lora_layer)
        replaced_count += 1

    print(f"Applied Native LoRA to {replaced_count} linear layers (rank={rank}, alpha={alpha}).")
    return model


def setup_lora_model(
    model: nn.Module,
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    use_peft_if_available: bool = True
) -> nn.Module:
    """Setup LoRA using PEFT if available, otherwise fallback to NativeLoraLinear."""
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "c_attn", "c_proj", "c_fc"]

    if use_peft_if_available:
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=rank,
                lora_alpha=alpha,
                lora_dropout=dropout,
                target_modules=target_modules,
                bias="none",
            )
            peft_model = get_peft_model(model, peft_config)
            peft_model.print_trainable_parameters()
            return peft_model
        except Exception as e:
            print(f"PEFT initialization skipped ({e}). Falling back to Native LoRA.")

    return apply_native_lora(model, target_modules=target_modules, rank=rank, alpha=alpha, dropout=dropout)


def save_lora_weights(model: nn.Module, save_dir: str) -> None:
    """Save only trainable LoRA parameters to directory."""
    os.makedirs(save_dir, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        try:
            model.save_pretrained(save_dir)
            return
        except Exception:
            pass

    # Native save
    trainable_state = {k: v.cpu() for k, v in model.state_dict().items() if "lora_" in k}
    save_path = os.path.join(save_dir, "native_lora_weights.pt")
    torch.save(trainable_state, save_path)
    print(f"Saved native LoRA weights to {save_path} ({len(trainable_state)} tensors).")


def load_lora_weights(model: nn.Module, load_dir: str) -> nn.Module:
    """Load trainable LoRA parameters from directory."""
    native_path = os.path.join(load_dir, "native_lora_weights.pt")
    if os.path.exists(native_path):
        state_dict = torch.load(native_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded native LoRA weights from {native_path}.")
    return model
