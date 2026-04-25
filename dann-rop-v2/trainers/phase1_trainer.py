"""
trainers/phase1_trainer.py
---------------------------
Treinamento DANN (Fase 1): adaptação de domínio source→target.

Responsabilidade única (SRP): apenas o loop DANN com GRL e MixUp.

Fluxo:
  - Source: ORIGA (imagens de glaucoma com rótulos)
  - Target: ROP COMPLETO (sem rótulos de doença — apenas domínio)
  - K-Fold sobre source (ORIGA) para validar o classificador de tarefa
  - Alpha do GRL cresce progressivamente (schedule_alpha)
  - MixUp inter-domínio opcional
"""

import os
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from model import EfficientNetDANN
from datasets import RetinaDataset, get_transforms, mixup_data
from trainers.base_trainer import BaseTrainer
from config import Config


class Phase1Trainer(BaseTrainer):
    """Trainer da Fase 1 (Domain Adaptation via DANN)."""

    def __init__(self, config: Config):
        super().__init__(config, device=config.device)

    def train_kfold(self, source_df, target_df, dann_weights_path=None) -> Dict[str, Any]:
        """
        Executa o treinamento DANN com K-Fold sobre o source (ORIGA).

        Parâmetros
        ----------
        source_df : pd.DataFrame  (ORIGA com binary_label)
        target_df : pd.DataFrame  (ROP de treino — sem label de doença)
        dann_weights_path : str | None  pesos iniciais opcionais

        Retorna
        -------
        dict com métricas por fold e caminhos dos checkpoints
        """
        cfg_p1 = self.config.phase1
        cfg_m = self.config.model
        cfg_d = self.config.data

        kf = KFold(n_splits=cfg_p1.k_folds, shuffle=True, random_state=cfg_d.random_state)
        fold_results: List[Dict] = []
        checkpoint_paths: List[str] = []
        source_indices = np.arange(len(source_df))

        for fold, (train_idx, val_idx) in enumerate(kf.split(source_indices)):
            print(f"\n{'='*60}\n  FASE 1 — Fold {fold + 1}/{cfg_p1.k_folds}\n{'='*60}")

            train_source_df = source_df.iloc[train_idx]
            val_source_df = source_df.iloc[val_idx]

            train_source_loader = DataLoader(
                RetinaDataset(train_source_df, "", get_transforms(True), domain_label=0),
                batch_size=cfg_p1.batch_size, shuffle=True, num_workers=4, pin_memory=True,
            )
            val_source_loader = DataLoader(
                RetinaDataset(val_source_df, "", get_transforms(False), domain_label=0),
                batch_size=cfg_p1.batch_size, shuffle=False, num_workers=4, pin_memory=True,
            )
            target_loader = DataLoader(
                RetinaDataset(target_df, "", get_transforms(True), domain_label=1),
                batch_size=cfg_p1.batch_size, shuffle=True, num_workers=4, pin_memory=True,
            )

            model = EfficientNetDANN(
                backbone_name=cfg_m.backbone,
                num_classes_task=cfg_m.num_classes_source,
                num_classes_domain=2,
                dropout=cfg_m.dropout,
                alpha=0.0,
                pretrained=True,
            ).to(self.device)

            if dann_weights_path and os.path.isfile(dann_weights_path):
                ckpt = torch.load(dann_weights_path, map_location=self.device)
                model.load_state_dict(ckpt["model_state_dict"])

            optimizer = torch.optim.Adam(
                model.parameters(), lr=cfg_p1.lr, weight_decay=cfg_p1.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg_p1.num_epochs
            )
            criterion_class = nn.CrossEntropyLoss()
            criterion_domain = nn.CrossEntropyLoss()

            best_auc = 0.0
            best_epoch = 0
            save_path = os.path.join(cfg_p1.save_dir, f"fold{fold+1}_best.pth")
            os.makedirs(cfg_p1.save_dir, exist_ok=True)

            for epoch in range(cfg_p1.num_epochs):
                alpha = self.schedule_alpha(epoch, cfg_p1.num_epochs, cfg_p1.alpha_max)

                train_metrics = self._train_dann_epoch(
                    model, train_source_loader, target_loader,
                    optimizer, criterion_class, criterion_domain,
                    alpha,
                    use_mixup=cfg_p1.use_mixup,
                    mixup_alpha=cfg_p1.mixup_alpha,
                )
                val_metrics = self._run_epoch(
                    model, val_source_loader,
                    criterion_class=criterion_class,
                    alpha=alpha,
                    is_train=False,
                )
                scheduler.step()

                print(
                    f"  Época {epoch+1:3d}/{cfg_p1.num_epochs} "
                    f"| alpha={alpha:.3f} "
                    f"| Loss(train)={train_metrics['loss']:.4f} "
                    f"| AUC(val)={val_metrics.get('auc', float('nan')):.4f} "
                    f"| Acc(val)={val_metrics['acc']:.4f}"
                )

                val_auc = val_metrics.get("auc", 0.0)
                if val_auc > best_auc:
                    best_auc = val_auc
                    best_epoch = epoch + 1
                    self.save_checkpoint(model, optimizer, epoch, val_metrics, save_path)

            print(f"  [Fold {fold+1}] Melhor AUC={best_auc:.4f} na época {best_epoch}")
            fold_results.append({"auc": best_auc, "fold": fold + 1})
            checkpoint_paths.append(save_path)

        mean_auc = float(np.mean([r["auc"] for r in fold_results]))
        std_auc = float(np.std([r["auc"] for r in fold_results]))
        print(f"\n[Fase 1] AUC média={mean_auc:.4f} ± {std_auc:.4f}")

        results = {
            "fold_results": fold_results,
            "mean_auc": mean_auc,
            "std_auc": std_auc,
            "checkpoint_paths": checkpoint_paths,
        }
        self.save_results(results, os.path.join(cfg_p1.save_dir, "phase1_results.json"))
        return results

    def _train_dann_epoch(
        self, model, source_loader, target_loader,
        optimizer, criterion_class, criterion_domain,
        alpha, use_mixup=True, mixup_alpha=0.4,
    ) -> Dict[str, float]:
        """Loop de treino DANN por época."""
        model.train()
        total_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []
        target_iter = iter(target_loader)

        for source_batch in source_loader:
            src_imgs = source_batch[0].to(self.device)
            src_labels = source_batch[1].to(self.device)
            src_domains = source_batch[2].to(self.device)

            try:
                tgt_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                tgt_batch = next(target_iter)

            tgt_imgs = tgt_batch[0].to(self.device)
            tgt_domains = tgt_batch[2].to(self.device)

            # Forward source
            class_out_src, domain_out_src = model(src_imgs, alpha=alpha)

            # Forward target
            _, domain_out_tgt = model(tgt_imgs, alpha=alpha)

            # Perdas
            loss_class = criterion_class(class_out_src, src_labels)
            loss_domain = (
                criterion_domain(domain_out_src, src_domains)
                + criterion_domain(domain_out_tgt, tgt_domains)
            )
            loss = loss_class + loss_domain

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            probs = torch.softmax(class_out_src, dim=1)[:, 1].detach().cpu().numpy()
            preds = class_out_src.argmax(dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(src_labels.cpu().numpy())
            all_probs.extend(probs)

        metrics = self.compute_metrics(
            np.array(all_labels), np.array(all_preds), np.array(all_probs)
        )
        metrics["loss"] = total_loss / max(len(source_loader), 1)
        return metrics
