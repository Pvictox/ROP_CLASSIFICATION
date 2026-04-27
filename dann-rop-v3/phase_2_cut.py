"""
Recorte da fase 2 do origa_df.ipynb ate:
results_phase2 = train_finetuning_kfold(...)

Mantem a logica experimental do notebook para comparacao:
- carrega ROP via DataFactory
- faz split df_dev/df_test, mas treina com ROP_df inteiro
- usa StratifiedKFold em nivel de imagem no target_dataset inteiro
- inicializa com checkpoint da fase 1 por fold (ou um unico checkpoint opcional)
"""

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import optim
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms


# garante import de modulos do workspace quando o script e chamado por caminho relativo
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from data_factory.data_factory import DataFactory


# ---------------------------------------------------------------------
# Importa RetinaDataset ja existente no projeto (dann-rop-v2/datasets.py)
# ---------------------------------------------------------------------
V2_DIR = ROOT / "dann-rop-v2"
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from datasets import RetinaDataset  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Fase 2 (cut) - fine-tuning em ROP")
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--num-classes-target", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--phase1-checkpoint",
        type=str,
        default=None,
        help=(
            "Se informado, usa o mesmo checkpoint da fase 1 para todos os folds da fase 2. "
            "Se omitido, usa o padrao dann_model_fold_{fold}.pth"
        ),
    )
    parser.add_argument(
        "--phase1-ckpt-dir",
        type=str,
        default=str(ROOT / "dann-rop-v3" / "checkpoints" / "phase1_cut"),
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=str(ROOT / "dann-rop-v3" / "checkpoints" / "phase2_cut"),
    )
    return parser.parse_args()


def resolve_device(device_str):
    if device_str == "cuda":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def get_transforms(img_size=256):
    train_transforms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=360),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transforms, val_transforms


class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


def grad_reverse(x, alpha=1.0):
    return GradientReversalFn.apply(x, alpha)


class EfficientNetDANN(nn.Module):
    def __init__(self, num_classes=2, pre_trained=True):
        super().__init__()

        weights = models.EfficientNet_B0_Weights.DEFAULT if pre_trained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        n_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.class_classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(n_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x, alpha=1.0):
        features = self.backbone.features(x)
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, 1)

        class_output = self.class_classifier(features)
        reverse_features = grad_reverse(features, alpha)
        domain_output = self.domain_classifier(reverse_features)
        return class_output, domain_output


def load_rop_dataframe():
    rop_factory = DataFactory(
        img_path="/backup/pedro_fonseca/PATIENT_ROP/DATASET/images_stack/images_stack",
        metadata_path="DATASET/infant_retinal_database_info.csv",
    )

    rop_df = rop_factory.load_data(verbose=True)

    # Mantido propositalmente (split existe, mas nao e usado no treino)
    df_dev, df_test = train_test_split(
        rop_df,
        test_size=0.2,
        stratify=rop_df["binary_label"],
        random_state=42,
    )

    print("ROP_df total:", len(rop_df))
    print("df_dev (nao usado na fase 2):", len(df_dev))
    print("df_test (nao usado na fase 2):", len(df_test))
    return rop_df


def get_phase1_checkpoint_for_fold(fold, phase1_ckpt_dir, single_checkpoint):
    if single_checkpoint is not None:
        single_checkpoint = Path(single_checkpoint)
        if single_checkpoint.exists():
            return single_checkpoint

        candidate = phase1_ckpt_dir / single_checkpoint.name
        if candidate.exists():
            return candidate

        return single_checkpoint
    return phase1_ckpt_dir / f"dann_model_fold_{fold}.pth"


def replace_class_head(model, num_classes_target):
    n_features = model.class_classifier[1].in_features
    model.class_classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(n_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(512, num_classes_target),
    )


def train_finetuning_kfold(
    target_dataset,
    phase1_ckpt_dir,
    num_classes_target=2,
    k_folds=5,
    num_epochs=30,
    batch_size=32,
    num_workers=4,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    phase1_checkpoint=None,
    save_dir=None,
):
    print("DEVICE:", device)

    # Ajuste para RetinaDataset do v2 (usa .df)
    target_labels = target_dataset.df.iloc[:, 2].values
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    indices = np.arange(len(target_dataset))

    results_phase2 = []
    save_dir.mkdir(parents=True, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, target_labels), 1):
        print(f"\n{'#' * 50}")
        print(f"FASE 2 - FOLD {fold}/{k_folds}")
        print(f"{'#' * 50}")

        model = EfficientNetDANN(num_classes=2, pre_trained=False)
        weights_path = get_phase1_checkpoint_for_fold(
            fold=fold,
            phase1_ckpt_dir=phase1_ckpt_dir,
            single_checkpoint=phase1_checkpoint,
        )

        try:
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict, strict=True)
            print(f"Pesos carregados: {weights_path}")
        except FileNotFoundError:
            print(f"AVISO: Pesos {weights_path} nao encontrados. Iniciando aleatorio.")

        replace_class_head(model, num_classes_target=num_classes_target)
        model = model.to(device)

        train_subset = Subset(target_dataset, train_idx)
        val_subset = Subset(target_dataset, val_idx)

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        optimizer = optim.Adam(
            [
                {"params": model.backbone.parameters(), "lr": 1e-5},
                {"params": model.class_classifier.parameters(), "lr": 1e-3},
            ],
            weight_decay=1e-5,
        )

        criterion = nn.CrossEntropyLoss()

        history = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
            "val_auc": [],
        }

        best_metric = 0.0
        best_model_path = save_dir / f"finetuned_best_fold_{fold}.pth"

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            for imgs, labels, _ in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)

                optimizer.zero_grad()
                class_preds, _ = model(imgs, alpha=0.0)
                loss = criterion(class_preds, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            epoch_train_loss = running_loss / max(len(train_loader), 1)

            model.eval()
            val_loss = 0.0
            all_preds = []
            all_labels = []
            all_probs = []

            with torch.no_grad():
                for imgs, labels, _ in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)

                    class_preds, _ = model(imgs, alpha=0.0)
                    loss = criterion(class_preds, labels)
                    val_loss += loss.item()

                    probs = F.softmax(class_preds, dim=1)
                    _, preds = torch.max(class_preds, 1)

                    all_probs.extend(probs.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            epoch_val_loss = val_loss / max(len(val_loader), 1)
            epoch_val_acc = accuracy_score(all_labels, all_preds)
            epoch_val_f1 = f1_score(all_labels, all_preds, average="weighted")

            try:
                if num_classes_target == 2:
                    all_probs_np = np.array(all_probs)
                    epoch_val_auc = roc_auc_score(all_labels, all_probs_np[:, 1])
                else:
                    epoch_val_auc = roc_auc_score(
                        all_labels,
                        all_probs,
                        multi_class="ovr",
                        average="weighted",
                    )
            except ValueError:
                epoch_val_auc = 0.0

            history["train_loss"].append(epoch_train_loss)
            history["val_loss"].append(epoch_val_loss)
            history["val_acc"].append(epoch_val_acc)
            history["val_f1"].append(epoch_val_f1)
            history["val_auc"].append(epoch_val_auc)

            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Loss: {epoch_train_loss:.4f} | "
                f"Val Loss: {epoch_val_loss:.4f} | "
                f"Acc: {epoch_val_acc:.4f} | "
                f"F1: {epoch_val_f1:.4f} | "
                f"AUC: {epoch_val_auc:.4f}"
            )

            # Mantido como notebook: melhor modelo por acc de validacao
            if epoch_val_acc > best_metric:
                best_metric = epoch_val_acc
                torch.save(model.state_dict(), best_model_path)

        results_phase2.append(
            {
                "fold": fold,
                "best_metric": best_metric,
                "best_model_path": str(best_model_path),
                "history": history,
                "phase1_init": str(weights_path),
            }
        )

    return results_phase2


def main():
    args = parse_args()
    device = resolve_device(args.device)

    train_transforms, _ = get_transforms(img_size=256)
    rop_df = load_rop_dataframe()

    target_dataset = RetinaDataset(
        dataframe=rop_df,
        root_dir="",
        transform=train_transforms,
        domain_label=1,
        image_col="filepath",
    )

    print(f"Tamanho Dataset Target (ROP): {len(target_dataset)}")

    phase1_ckpt_dir = Path(args.phase1_ckpt_dir)
    phase1_checkpoint = Path(args.phase1_checkpoint) if args.phase1_checkpoint else None
    save_dir = Path(args.save_dir)

    results_phase2 = train_finetuning_kfold(
        target_dataset=target_dataset,
        phase1_ckpt_dir=phase1_ckpt_dir,
        num_classes_target=args.num_classes_target,
        k_folds=args.k_folds,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        phase1_checkpoint=phase1_checkpoint,
        save_dir=save_dir,
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    out_json = save_dir / "phase2_results.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results_phase2, f, ensure_ascii=False, indent=2)

    print("\nFase 2 encerrada.")
    print("Folds treinados:", len(results_phase2))
    print("Resultados salvos em:", out_json)


if __name__ == "__main__":
    main()
