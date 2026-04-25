"""
trainers/phase2_trainer.py
---------------------------
Fine-tuning supervisionado (Fase 2) com GroupKFold por patient_id.

Responsabilidade única (SRP): apenas o loop de fine-tuning supervisionado.

Correção do bug experimental:
  GroupKFold usa patient_id como grupo. Nenhum paciente do holdout de
  teste aparece aqui (o split já foi feito em rop_loader.py).

Fluxo:
  - Carrega pesos da Fase 1 no backbone
  - Substitui a cabeça de classificação para num_classes_target classes
  - Congela o backbone nas primeiras épocas (warm-up opcional)
  - GroupKFold sobre os dados de treino ROP por patient_id
  - Salva checkpoint do melhor modelo por fold (maior AUC na validação)
"""

import os
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold

from model import EfficientNetDANN
from datasets import RetinaDataset, get_transforms
from trainers.base_trainer import BaseTrainer
from config import Config


class Phase2Trainer(BaseTrainer):
    """
    Trainer da Fase 2 (fine-tuning supervisionado).

    Parâmetros
    ----------
    config : Config
        Objeto de configuração global.
    """

    def __init__(self, config: Config):
        super().__init__(config, device=config.device)

    def train_kfold(
        self,
        train_df,
        phase1_weights_path: str | None = None,
        warmup_epochs: int = 5,
    ) -> Dict[str, Any]:
        """
        Executa o fine-tuning com GroupKFold por patient_id.

        Parâmetros
        ----------
        train_df : pd.DataFrame
            Dados de treino ROP (SEM os pacientes de teste).
            Deve conter: image_path, binary_label, patient_id.
        phase1_weights_path : str | None
            Caminho para o checkpoint da Fase 1.
            Se None, inicializa o backbone com pesos pré-treinados do timm.
        warmup_epochs : int
            Número de épocas com backbone congelado (warm-up).

        Retorna
        -------
        results : dict com métricas por fold e caminhos dos checkpoints.
        """
        cfg_p2 = self.config.phase2
        cfg_m = self.config.model
        cfg_d = self.config.data

        groups = train_df[cfg_d.patient_id_col].values
        gkf = GroupKFold(n_splits=cfg_p2.k_folds)

        fold_results: List[Dict[str, float]] = []
        checkpoint_paths: List[str] = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(train_df, groups=groups)):
            print(f"\n{'='*60}")
            print(f"  FASE 2 — Fold {fold + 1}/{cfg_p2.k_folds}")

            fold_train_df = train_df.iloc[train_idx]
            fold_val_df = train_df.iloc[val_idx]

            # Verificação de sanidade: sem overlap de pacientes
            train_pats = set(fold_train_df[cfg_d.patient_id_col])
            val_pats = set(fold_val_df[cfg_d.patient_id_col])
            overlap = train_pats & val_pats
            assert len(overlap) == 0, f"Overlap de pacientes no fold {fold+1}: {overlap}"
            print(f"  Pacientes treino={len(train_pats)} | Pacientes val={len(val_pats)}")
            print(f"{'='*60}")

            # ─── Datasets / DataLoaders ───────────────────────────────────
            train_ds = RetinaDataset(
                fold_train_df, root_dir="",
                transform=get_transforms(is_train=True),
                domain_label=1,
            )
            val_ds = RetinaDataset(
                fold_val_df, root_dir="",
                transform=get_transforms(is_train=False),
                domain_label=1,
            )

            train_loader = DataLoader(
                train_ds, batch_size=cfg_p2.batch_size,
                shuffle=True, num_workers=4, pin_memory=True,
            )
            val_loader = DataLoader(
                val_ds, batch_size=cfg_p2.batch_size,
                shuffle=False, num_workers=4, pin_memory=True,
            )

            # ─── Modelo ───────────────────────────────────────────────────
            model = EfficientNetDANN(
                backbone_name=cfg_m.backbone,
                num_classes_task=cfg_m.num_classes_target,
                num_classes_domain=2,
                dropout=cfg_m.dropout,
                alpha=0.0,
                pretrained=(phase1_weights_path is None),
            ).to(self.device)

            if phase1_weights_path and os.path.isfile(phase1_weights_path):
                ckpt = torch.load(phase1_weights_path, map_location=self.device)
                # Carrega apenas os pesos do backbone (ignora a cabeça antiga)
                state = ckpt["model_state_dict"]
                backbone_state = {k: v for k, v in state.items()
                                  if k.startswith("backbone.")}
                model.load_state_dict(backbone_state, strict=False)
                print(f"  [Fold {fold+1}] Backbone carregado de {phase1_weights_path}")

            # Substitui a cabeça de classificação para target
            model.replace_class_head(cfg_m.num_classes_target, cfg_m.dropout)
            model = model.to(self.device)

            # ─── Otimizador ───────────────────────────────────────────────
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg_p2.lr,
                weight_decay=cfg_p2.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg_p2.num_epochs
            )
            criterion = nn.CrossEntropyLoss()

            best_auc = 0.0
            best_epoch = 0
            save_path = os.path.join(cfg_p2.save_dir, f"fold{fold+1}_best.pth")
            os.makedirs(cfg_p2.save_dir, exist_ok=True)

            for epoch in range(cfg_p2.num_epochs):
                # Warm-up: congela backbone nas primeiras épocas
                if epoch < warmup_epochs:
                    self._set_backbone_grad(model, requires_grad=False)
                else:
                    self._set_backbone_grad(model, requires_grad=True)

                train_metrics = self._run_epoch(
                    model, train_loader, optimizer=optimizer,
                    criterion_class=criterion, is_train=True,
                )
                val_metrics = self._run_epoch(
                    model, val_loader,
                    criterion_class=criterion, is_train=False,
                )
                scheduler.step()

                print(
                    f"  Época {epoch+1:3d}/{cfg_p2.num_epochs} "
                    f"| Loss(train)={train_metrics['loss']:.4f} "
                    f"| AUC(val)={val_metrics.get('auc', float('nan')):.4f} "
                    f"| F1(val)={val_metrics['f1']:.4f} "
                    f"| Acc(val)={val_metrics['acc']:.4f}"
                )

                val_auc = val_metrics.get("auc", 0.0)
                if val_auc > best_auc:
                    best_auc = val_auc
                    best_epoch = epoch + 1
                    self.save_checkpoint(model, optimizer, epoch, val_metrics, save_path)

            print(f"  [Fold {fold+1}] Melhor AUC={best_auc:.4f} na época {best_epoch}")
            fold_results.append({
                "fold": fold + 1,
                "auc": best_auc,
                "checkpoint": save_path,
            })
            checkpoint_paths.append(save_path)

        # ─── Resumo ───────────────────────────────────────────────────────
        mean_auc = float(np.mean([r["auc"] for r in fold_results]))
        std_auc = float(np.std([r["auc"] for r in fold_results]))
        print(f"\n[Fase 2] AUC média={mean_auc:.4f} ± {std_auc:.4f}")

        results = {
            "fold_results": fold_results,
            "mean_auc": mean_auc,
            "std_auc": std_auc,
            "checkpoint_paths": checkpoint_paths,
        }
        self.save_results(results, os.path.join(cfg_p2.save_dir, "phase2_results.json"))
        return results

    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _set_backbone_grad(model: EfficientNetDANN, requires_grad: bool):
        """Congela ou descongela o backbone."""
        for param in model.backbone.parameters():
            param.requires_grad = requires_grad
