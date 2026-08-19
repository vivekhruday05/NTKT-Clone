"""Trainer module for Next Token Knowledge Tracing (NTKT).

Implements the exact training hyperparameters from Section 'Implementation Details' of the paper:
- Learning rate = 2e-4 with cosine schedule and 50 warmup steps
- Weight decay = 0.01
- Per-device batch size = 4, Gradient accumulation = 4 (effective batch size = 16)
- Early stopping: triggers when eval loss fails to improve by >= 0.001 over 10 eval intervals (evaluated every 250 steps)
- AdamW (8-bit if available, otherwise fp32/bf16 AdamW)
"""

import os
import time
import json
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.scheduler import get_cosine_schedule_with_warmup
from evaluation.evaluator import evaluate_ntkt


class NTKTTrainer:
    """Trainer for fine-tuning NTKT LLM models with selective loss masking."""

    def __init__(
        self,
        model: Any,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        output_dir: str = "checkpoints/ntkt",
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        max_steps: int = 20000,
        warmup_steps: int = 50,
        gradient_accumulation_steps: int = 4,
        eval_steps: int = 250,
        early_stopping_patience: int = 10,
        early_stopping_delta: float = 0.001,
        max_grad_norm: float = 1.0,
        use_8bit_adam: bool = False,
        fp16: bool = False,
        bf16: bool = True,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.eval_steps = eval_steps
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delta = early_stopping_delta
        self.max_grad_norm = max_grad_norm
        self.fp16 = fp16
        self.bf16 = bf16

        os.makedirs(output_dir, exist_ok=True)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if hasattr(self.model, "to") and not hasattr(self.model, "hf_device_map"):
            try:
                self.model.to(self.device)
            except Exception:
                pass

        # Setup Optimizer
        decay_params = []
        no_decay_params = []
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                if any(nd in n for nd in ["bias", "LayerNorm", "layer_norm"]):
                    no_decay_params.append(p)
                else:
                    decay_params.append(p)

        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        # Use 8-bit AdamW if available and requested
        if use_8bit_adam and torch.cuda.is_available():
            try:
                import bitsandbytes as bnb
                self.optimizer = bnb.optim.AdamW8bit(optimizer_grouped_parameters, lr=self.learning_rate)
                print("Using 8-bit AdamW optimizer.")
            except Exception as e:
                print(f"8-bit AdamW unavailable ({e}). Using standard AdamW.")
                self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        else:
            self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate)

        # Setup Scheduler
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.max_steps
        )

        # Mixed Precision Scaler
        self.use_amp = (self.fp16 or self.bf16) and torch.cuda.is_available()
        self.amp_dtype = torch.bfloat16 if (self.bf16 and torch.cuda.is_bf16_supported()) else torch.float16
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16 and torch.cuda.is_available())

    def train(self) -> Dict[str, Any]:
        """Execute the NTKT fine-tuning loop."""
        print(f"Starting NTKT Training for up to {self.max_steps} steps on {self.device}...")
        print(f"  - Learning Rate: {self.learning_rate}")
        print(f"  - Grad Accumulation: {self.gradient_accumulation_steps}")
        print(f"  - Eval Steps: every {self.eval_steps} steps")
        print(f"  - Early Stopping: Patience={self.early_stopping_patience}, Min Delta={self.early_stopping_delta}")

        global_step = 0
        running_loss = 0.0
        best_eval_loss = float("inf")
        patience_counter = 0
        training_history = []

        self.model.train()
        epoch = 0
        train_iter = iter(self.train_dataloader)

        progress_bar = tqdm(total=self.max_steps, desc="Training Steps")

        while global_step < self.max_steps:
            epoch += 1
            for step_in_epoch, batch in enumerate(self.train_dataloader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                if self.use_amp:
                    with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        loss = outputs["loss"] / self.gradient_accumulation_steps
                    self.scaler.scale(loss).backward()
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs["loss"] / self.gradient_accumulation_steps
                    loss.backward()

                running_loss += loss.item() * self.gradient_accumulation_steps

                if (step_in_epoch + 1) % self.gradient_accumulation_steps == 0:
                    if self.use_amp and self.fp16:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.optimizer.step()

                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                    progress_bar.update(1)

                    # Periodic Evaluation
                    if global_step % self.eval_steps == 0 or global_step == self.max_steps:
                        avg_train_loss = running_loss / self.eval_steps
                        running_loss = 0.0

                        eval_results = self.evaluate()
                        eval_loss = eval_results["metrics"]["log_loss"]
                        eval_auc = eval_results["metrics"]["auc"]
                        eval_f1 = eval_results["metrics"]["f1"]
                        eval_acc = eval_results["metrics"]["accuracy"]

                        current_lr = self.optimizer.param_groups[0]["lr"]
                        log_entry = {
                            "step": global_step,
                            "train_loss": round(avg_train_loss, 4),
                            "eval_loss": eval_loss,
                            "eval_auc": eval_auc,
                            "eval_f1": eval_f1,
                            "eval_acc": eval_acc,
                            "lr": current_lr,
                        }
                        training_history.append(log_entry)

                        tqdm.write(
                            f"Step {global_step:5d} | Train Loss: {avg_train_loss:.4f} | "
                            f"Eval Loss: {eval_loss:.4f} | AUC: {eval_auc:.4f} | F1: {eval_f1:.4f} | ACC: {eval_acc:.4f}"
                        )

                        # Check early stopping condition: eval loss improved by at least delta
                        if (best_eval_loss - eval_loss) >= self.early_stopping_delta:
                            best_eval_loss = eval_loss
                            patience_counter = 0
                            # Save best checkpoint
                            best_save_path = os.path.join(self.output_dir, "best_checkpoint")
                            self.model.save_adapters(best_save_path)
                            tqdm.write(f"  --> Saved new best checkpoint to {best_save_path} (eval_loss: {eval_loss:.4f})")
                        else:
                            patience_counter += 1
                            tqdm.write(f"  --> Early stopping patience: {patience_counter}/{self.early_stopping_patience}")
                            if patience_counter >= self.early_stopping_patience:
                                tqdm.write(f"Early stopping triggered at step {global_step}!")
                                progress_bar.close()
                                return {
                                    "best_eval_loss": best_eval_loss,
                                    "total_steps": global_step,
                                    "history": training_history,
                                }

                        self.model.train()

                    if global_step >= self.max_steps:
                        break

        progress_bar.close()
        
        # Save final checkpoint
        final_save_path = os.path.join(self.output_dir, "final_checkpoint")
        self.model.save_adapters(final_save_path)
        with open(os.path.join(self.output_dir, "training_history.json"), "w") as f:
            json.dump(training_history, f, indent=2)

        return {
            "best_eval_loss": best_eval_loss,
            "total_steps": global_step,
            "history": training_history,
        }

    def evaluate(self) -> Dict[str, Any]:
        """Run validation on the validation set."""
        return evaluate_ntkt(self.model, self.val_dataloader, device=self.device)
