"""
model.py
--------
Definição do modelo DANN (Domain-Adversarial Neural Network).

Responsabilidade única (SRP): este módulo só contém a arquitetura.

Componentes:
  - GradReverse: camada de inversão de gradiente (Gradient Reversal Layer - GRL)
  - BinaryFocalLoss: perda focal para classes desbalanceadas
  - EfficientNetDANN: backbone EfficientNet + classificador de classes + classificador de domínio
"""

import torch
import torch.nn as nn
from torch.autograd import Function
import timm


# ─────────────────────────────────────────────────────────────────────────────
# Gradient Reversal Layer
# ─────────────────────────────────────────────────────────────────────────────

class GradReverseFunction(Function):
    """
    Implementação do gradiente reverso (GRL).

    Na passagem forward: identidade (não altera o tensor).
    Na passagem backward: multiplica o gradiente por -alpha,
    forçando o backbone a extrair features invariantes ao domínio.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(torch.tensor(alpha))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (alpha,) = ctx.saved_tensors
        return -alpha * grad_output, None


class GradReverse(nn.Module):
    """Wrapper em Module para o GRL."""

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradReverseFunction.apply(x, self.alpha)

    def set_alpha(self, alpha: float):
        self.alpha = alpha


# ─────────────────────────────────────────────────────────────────────────────
# Focal Loss
# ─────────────────────────────────────────────────────────────────────────────

class BinaryFocalLoss(nn.Module):
    """
    Focal Loss binária.

    Reduz o peso de exemplos fáceis, focando o treinamento nos difíceis.
    Útil quando as classes estão desbalanceadas.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets.float())
        pt = torch.exp(-bce_loss)
        focal = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


# ─────────────────────────────────────────────────────────────────────────────
# Cabeças de classificação (helper)
# ─────────────────────────────────────────────────────────────────────────────

def _build_head(in_features: int, num_classes: int, dropout: float = 0.3) -> nn.Sequential:
    """Constrói uma cabeça FC com BatchNorm, Dropout e ativação."""
    return nn.Sequential(
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(256, num_classes),
    )


# ─────────────────────────────────────────────────────────────────────────────
# EfficientNetDANN
# ─────────────────────────────────────────────────────────────────────────────

class EfficientNetDANN(nn.Module):
    """
    Modelo DANN baseado em EfficientNet.

    Arquitetura:
      backbone (EfficientNet sem cabeça) ──→ features
          ├─→ class_classifier  (classificação da tarefa principal)
          └─→ GRL ──→ domain_classifier (classificação de domínio)

    Parâmetros
    ----------
    backbone_name : str
        Nome do modelo no timm (ex.: 'efficientnet_b0').
    num_classes_task : int
        Número de classes da tarefa principal (ex.: 2 para ROP).
    num_classes_domain : int
        Número de domínios (2: source=ORIGA, target=ROP).
    dropout : float
        Taxa de dropout nas cabeças.
    alpha : float
        Valor inicial de alpha do GRL.
    pretrained : bool
        Carregar pesos pré-treinados do timm.
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        num_classes_task: int = 2,
        num_classes_domain: int = 2,
        dropout: float = 0.3,
        alpha: float = 0.0,
        pretrained: bool = True,
    ):
        super().__init__()

        # ─── Backbone ──────────────────────────────────────────────────────
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,     # remove a cabeça original
            global_pool="avg", # global average pooling
        )
        self.in_features: int = self.backbone.num_features

        # ─── Classificador de tarefa ──────────────────────────────────────
        self.class_classifier = _build_head(self.in_features, num_classes_task, dropout)

        # ─── Classificador de domínio (com GRL) ───────────────────────────
        self.grl = GradReverse(alpha=alpha)
        self.domain_classifier = _build_head(self.in_features, num_classes_domain, dropout)

    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        alpha: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parâmetros
        ----------
        x : Tensor [B, C, H, W]
        alpha : float, opcional
            Se fornecido, sobrescreve o alpha atual do GRL.

        Retorna
        -------
        (class_logits, domain_logits) — ambos [B, num_classes]
        """
        if alpha is not None:
            self.grl.set_alpha(alpha)

        features = self.backbone(x)            # [B, in_features]

        class_out = self.class_classifier(features)
        domain_out = self.domain_classifier(self.grl(features))

        return class_out, domain_out

    # ─────────────────────────────────────────────────────────────────────────

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna apenas as features do backbone (sem classificadores)."""
        return self.backbone(x)

    def replace_class_head(self, num_classes: int, dropout: float = 0.3):
        """
        Substitui a cabeça de classificação de tarefa (útil no fine-tuning Fase 2).
        O backbone e o domínio permanecem intactos.
        """
        self.class_classifier = _build_head(self.in_features, num_classes, dropout)
