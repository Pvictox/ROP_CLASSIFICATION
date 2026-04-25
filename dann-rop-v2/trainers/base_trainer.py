"""
trainers/base_trainer.py
-------------------------
Classe base abstrata para os trainers (OCP: aberto para extensão).

Define a interface comum e utilitários compartilhados entre
Phase1Trainer e Phase2Trainer.
"""

import os
import json
from abc import ABC, abstractmethod
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
)
import numpy as np


class BaseTrainer(ABC):
    """
    Interface base para os trainers do pipeline DANN-ROP.

    Subclasses devem implementar `train_kfold`.
    """

    def __init__(self, config, device: str = "cuda"):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def train_kfold(self, *args, **kwargs) -> Dict[str, Any]:
        """Executa o treinamento com cross-validation e retorna métricas."""
        ...

    # ─── Utilitários ─────────────────────────────────────────────────────────

    @staticmethod
    def compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray | None = None,
    ) -> Dict[str, float]:
        """
        Calcula métricas de classificação binária.

        Parâmetros
        ----------
        y_true : array de rótulos verdadeiros
        y_pred : array de predições (classes)
        y_prob : array de probabilidades da classe positiva (para AUC)

        Retorna
        -------
        Dict com acc, balanced_acc, f1, auc (se y_prob fornecido)
        """
        metrics = {
            "acc": float(accuracy_score(y_true, y_pred)),
            "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        }
        if y_prob is not None:
            try:
                metrics["auc"] = float(roc_auc_score(y_true, y_prob))
            except ValueError:
                metrics["auc"] = float("nan")

        return metrics

    @staticmethod
    def save_checkpoint(
        model: nn.Module,
        optimizer,
        epoch: int,
        metrics: Dict[str, float],
        save_path: str,
    ):
        """Salva checkpoint com estado do modelo e métricas."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
            },
            save_path,
        )

    @staticmethod
    def save_results(results: Dict[str, Any], path: str):
        """Salva resultados (métricas) em JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)

    @staticmethod
    def schedule_alpha(epoch: int, num_epochs: int, alpha_max: float = 1.0) -> float:
        """
        Agenda progressivo do parâmetro alpha do GRL.

        Cresce de 0 a alpha_max ao longo do treinamento.
        Fórmula: alpha = 2 * alpha_max / (1 + exp(-10 * progress)) - alpha_max
        """
        import math
        progress = epoch / num_epochs
        alpha = 2.0 * alpha_max / (1.0 + math.exp(-10 * progress)) - alpha_max
        return max(0.0, alpha)

    def _run_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer=None,
        criterion_class: nn.Module = None,
        criterion_domain: nn.Module = None,
        alpha: float = 0.0,
        is_train: bool = True,
    ) -> Dict[str, float]:
        """
        Executa uma época de treino ou avaliação genérica.

        Subclasses podem sobrescrever para lógicas específicas.
        """
        model.train() if is_train else model.eval()
        total_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []

        ctx = torch.enable_grad() if is_train else torch.no_grad()

        with ctx:
            for batch in loader:
                images, labels, domains = batch[0], batch[1], batch[2]
                images = images.to(self.device)
                labels = labels.to(self.device)

                class_out, _ = model(images, alpha=alpha)

                if criterion_class is not None:
                    loss = criterion_class(class_out, labels)

                if is_train and optimizer is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item() if criterion_class else 0.0
                probs = torch.softmax(class_out, dim=1)[:, 1].detach().cpu().numpy()
                preds = class_out.argmax(dim=1).detach().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs)

        metrics = self.compute_metrics(
            np.array(all_labels), np.array(all_preds), np.array(all_probs)
        )
        metrics["loss"] = total_loss / max(len(loader), 1)
        return metrics
