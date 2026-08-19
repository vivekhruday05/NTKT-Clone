"""Trainer for traditional Knowledge Tracing baseline models (DKT, AKT, AKT-text, DTransformer)."""

import os
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation.evaluator import evaluate_baseline


class BaselineTrainer:
    """Trainer for numeric / attention KT baselines."""

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        output_dir: str = "checkpoints/baselines",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        num_epochs: int = 50,
        early_stopping_patience: int = 8,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience

        os.makedirs(output_dir, exist_ok=True)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

    def train(self) -> Dict[str, Any]:
        """Execute baseline model training across epochs."""
        best_val_auc = 0.0
        patience = 0
        history = []

        print(f"Training baseline model on {self.device} for {self.num_epochs} epochs...")

        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            running_loss = 0.0
            num_batches = 0

            for batch in self.train_dataloader:
                q_ids = batch["question_ids"].to(self.device)
                concept_ids = batch.get("concept_ids", None)
                if concept_ids is not None:
                    concept_ids = concept_ids.to(self.device)
                correctness = batch["correctness"].to(self.device)
                mask = batch["mask"].to(self.device)
                text_emb = batch.get("text_embeddings", None)
                if text_emb is not None:
                    text_emb = text_emb.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(
                    question_ids=q_ids,
                    concept_ids=concept_ids,
                    correctness=correctness,
                    mask=mask,
                    text_embeddings=text_emb
                )
                loss = outputs["loss"]
                if loss is not None:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    running_loss += loss.item()
                    num_batches += 1

            avg_loss = running_loss / max(1, num_batches)
            val_results = evaluate_baseline(self.model, self.val_dataloader, device=self.device)
            val_auc = val_results["metrics"]["auc"]
            val_f1 = val_results["metrics"]["f1"]
            val_acc = val_results["metrics"]["accuracy"]

            history.append({
                "epoch": epoch,
                "train_loss": avg_loss,
                "val_auc": val_auc,
                "val_f1": val_f1,
                "val_acc": val_acc
            })

            print(f"Epoch {epoch:2d}/{self.num_epochs} | Train Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f} | Val ACC: {val_acc:.4f}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience = 0
                save_path = os.path.join(self.output_dir, "best_baseline_model.pt")
                torch.save(self.model.state_dict(), save_path)
            else:
                patience += 1
                if patience >= self.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch} (Best Val AUC: {best_val_auc:.4f})")
                    break

        return {"best_val_auc": best_val_auc, "history": history}
