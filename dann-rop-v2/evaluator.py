"""
evaluator.py
-------------
Avaliação do ensemble dos modelos fine-tunados (Fase 2) no holdout de teste.

Responsabilidade única (SRP): apenas inferência e métricas no conjunto de teste.

Estratégia de ensemble:
  - Carrega os k modelos fine-tunados (um a um para economizar VRAM)
  - Média das probabilidades preditas por cada modelo
  - Calcula métricas finais no holdout de teste

NOTA: O holdout de teste nunca foi visto em nenhuma etapa de treino.
      Isso garante a validade experimental dos resultados finais.
"""

import os
import json
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

from model import EfficientNetDANN
from datasets import RetinaDataset, get_transforms
from config import Config


class EnsembleEvaluator:
    """
    Avalia um ensemble dos modelos fine-tunados da Fase 2.

    Os modelos são carregados um a um (evita OOM em GPUs com pouca VRAM).
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )

    def evaluate(
        self,
        test_df,
        checkpoint_paths: List[str],
        save_dir: str = "results",
        return_per_image: bool = False,
    ) -> Dict[str, Any]:
        """
        Avalia o ensemble no holdout de teste.

        Parâmetros
        ----------
        test_df : pd.DataFrame
            Holdout de teste (pacientes nunca vistos durante treino).
        checkpoint_paths : list[str]
            Caminhos para os checkpoints de cada fold da Fase 2.
        save_dir : str
            Diretório para salvar resultados.
        return_per_image : bool
            Se True, inclui predições por imagem no resultado.

        Retorna
        -------
        Dict com métricas finais (AUC, F1, Acc, Balanced Acc, etc.)
        """
        cfg_m = self.config.model

        test_ds = RetinaDataset(
            test_df,
            root_dir="",
            transform=get_transforms(is_train=False),
            domain_label=1,
            return_filename=True,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=16,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # ─── Ensemble: média das probabilidades ───────────────────────────
        all_probs_per_model: List[np.ndarray] = []
        filenames_collected = None

        for i, ckpt_path in enumerate(checkpoint_paths):
            print(f"  [Ensemble] Carregando modelo {i+1}/{len(checkpoint_paths)}: {ckpt_path}")

            model = EfficientNetDANN(
                backbone_name=cfg_m.backbone,
                num_classes_task=cfg_m.num_classes_target,
                num_classes_domain=2,
                dropout=cfg_m.dropout,
                alpha=0.0,
                pretrained=False,
            ).to(self.device)

            ckpt = torch.load(ckpt_path, map_location=self.device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            probs_model, labels_all, fnames = self._run_inference(model, test_loader)
            all_probs_per_model.append(probs_model)

            if filenames_collected is None:
                filenames_collected = fnames
                labels_array = labels_all

            # Libera VRAM imediatamente após inferência
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ─── Média das probabilidades ─────────────────────────────────────
        ensemble_probs = np.mean(np.stack(all_probs_per_model, axis=0), axis=0)
        ensemble_preds = (ensemble_probs >= 0.5).astype(int)

        # ─── Métricas ─────────────────────────────────────────────────────
        metrics = {
            "auc": float(roc_auc_score(labels_array, ensemble_probs)),
            "acc": float(accuracy_score(labels_array, ensemble_preds)),
            "balanced_acc": float(balanced_accuracy_score(labels_array, ensemble_preds)),
            "f1": float(f1_score(labels_array, ensemble_preds, average="binary", zero_division=0)),
            "num_models": len(checkpoint_paths),
            "num_test_images": len(test_df),
        }

        print("\n" + "="*60)
        print("  RESULTADOS FINAIS (Holdout de Teste — Ensemble)")
        print("="*60)
        for k, v in metrics.items():
            print(f"  {k:20s}: {v}")
        print()
        print(classification_report(labels_array, ensemble_preds, target_names=["Normal", "ROP"]))
        print("Confusion Matrix:")
        print(confusion_matrix(labels_array, ensemble_preds))

        if return_per_image:
            metrics["per_image"] = [
                {
                    "filename": f,
                    "true_label": int(l),
                    "pred_label": int(p),
                    "prob_positive": float(pr),
                }
                for f, l, p, pr in zip(
                    filenames_collected, labels_array, ensemble_preds, ensemble_probs
                )
            ]

        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "final_results.json"), "w") as fp:
            json.dump(metrics, fp, indent=2, default=str)
        print(f"\n  Resultados salvos em {save_dir}/final_results.json")

        return metrics

    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _run_inference(
        self,
        model: nn.Module,
        loader: DataLoader,
    ):
        """Executa inferência e retorna (probs, labels, filenames)."""
        all_probs, all_labels, all_fnames = [], [], []

        for batch in loader:
            images, labels, _, fnames = batch
            images = images.to(self.device)

            class_out, _ = model(images, alpha=0.0)
            probs = torch.softmax(class_out, dim=1)[:, 1].cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
            all_fnames.extend(fnames)

        return np.array(all_probs), np.array(all_labels), all_fnames
