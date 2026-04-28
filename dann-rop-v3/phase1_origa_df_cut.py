"""
Recorte direto do origa_df.ipynb até:
results = train_dann_kfold(source_dataset, target_dataset, k_folds=5, num_epochs=30)

Mantém a lógica "estranha" do notebook original:
- usa DataFactory para ORIGA e ROP
- faz split df_dev/df_test no ROP, mas treina com ROP_df inteiro
- usa ORIGA inteiro como source_dataset
"""

from pathlib import Path
import sys
import os
import json

# garante import de módulos do workspace quando o script é chamado por caminho relativo
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import cv2
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import optim
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from PIL import Image

from data_factory.data_factory import DataFactory


# ---------------------------------------------------------------------
# Importa RetinaDataset já existente no projeto (dann-rop-v2/datasets.py)
# ---------------------------------------------------------------------
V2_DIR = ROOT / "dann-rop-v2"
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from datasets import RetinaDataset  # noqa: E402


# ---------------------------------------------------------------------
# Dados (mantendo a estrutura do notebook)
# ---------------------------------------------------------------------
factory = DataFactory(
    img_path="/backup/lucas/datasets/origa/ORIGA/Images",
    metadata_path="/backup/lucas/datasets/origa/ORIGA/OrigaList.csv",
)

df = factory.load_data(verbose=True)

# DataFactory atual é orientado ao padrão de nome do ROP (DG*),
# então ORIGA pode retornar vazio. Fallback para CSV já processado.
if len(df) == 0:
    fallback_csv = ROOT / "origa_processed.csv"
    if not fallback_csv.exists():
        raise RuntimeError(
            "ORIGA vazio via DataFactory e fallback não encontrado em "
            f"{fallback_csv}"
        )

    origa_df = pd.read_csv(fallback_csv)
    if not {"ImagePath", "Glaucoma"}.issubset(set(origa_df.columns)):
        raise RuntimeError(
            "Fallback origa_processed.csv sem colunas esperadas: {'ImagePath', 'Glaucoma'}"
        )

    df = pd.DataFrame(
        {
            "filepath": origa_df["ImagePath"].astype(str),
            "binary_label": origa_df["Glaucoma"].astype(int),
            "patient_id": origa_df.get("Filename", origa_df["ImagePath"]).astype(str),
        }
    )
    df = df[df["filepath"].apply(os.path.exists)].reset_index(drop=True)
    print(f"[Fallback ORIGA] usando origa_processed.csv | amostras={len(df)}")

ROP_factory = DataFactory(
    img_path="/backup/pedro_fonseca/PATIENT_ROP/DATASET/images_stack/images_stack",
    metadata_path="DATASET/infant_retinal_database_info.csv",
)

ROP_df = ROP_factory.load_data(verbose=True)

# Mantido propositalmente (split existe, mas não é usado no treino)
df_dev, df_test = train_test_split(
    ROP_df,
    test_size=0.2,
    stratify=ROP_df["binary_label"],
    random_state=42,
)


# ---------------------------------------------------------------------
# Transforms e datasets
# ---------------------------------------------------------------------
class CLAHELab:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        lab = cv2.merge((l, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(img)


def get_transforms(img_size=256):
    train_transforms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            CLAHELab(),
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
            CLAHELab(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transforms, val_transforms


train_transforms, val_transforms = get_transforms(img_size=256)

# DataFactory retorna coluna filepath, então image_col='filepath'
source_dataset = RetinaDataset(
    dataframe=df,
    root_dir="",
    transform=train_transforms,
    domain_label=0,
    image_col="filepath",
)

# Mantido igual ideia original: target_dataset usa ROP_df inteiro
target_dataset = RetinaDataset(
    dataframe=ROP_df,
    root_dir="",
    transform=train_transforms,
    domain_label=1,
    image_col="filepath",
)

print(f"Tamanho Dataset Source (ORIGA): {len(source_dataset)}")
print(f"Tamanho Dataset Target (ROP): {len(target_dataset)}")


# ---------------------------------------------------------------------
# DANN
# ---------------------------------------------------------------------
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


def get_alpha(current_step, total_steps):
    p = float(current_step) / total_steps
    return 2.0 / (1.0 + np.exp(-10 * p)) - 1


def mixup_data(x1, x2, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    mixed_x = lam * x1 + (1 - lam) * x2
    return mixed_x, lam


def infinite_iterable(loader):
    while True:
        for batch in loader:
            yield batch


def get_new_model(device):
    model = EfficientNetDANN(num_classes=2, pre_trained=True)
    return model.to(device)


def train_dann_kfold(
    source_dataset,
    target_dataset,
    k_folds=5,
    num_epochs=10,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
):
    print("DEVICE:", device)

    # Ajuste para RetinaDataset do v2 (usa .df)
    target_labels = target_dataset.df.iloc[:, 2].values

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    indices = np.arange(len(target_dataset))

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, target_labels)):
        print(f"\n{'='*40}")
        print(f"INICIANDO FOLD {fold+1}/{k_folds}")
        print(f"{'='*40}")

        model = get_new_model(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

        source_loader = DataLoader(source_dataset, batch_size=32, shuffle=True, drop_last=True)
        target_train_subset = Subset(target_dataset, train_idx)
        target_loader = DataLoader(target_train_subset, batch_size=32, shuffle=True, drop_last=True)

        iter_source = infinite_iterable(source_loader)
        iter_target = infinite_iterable(target_loader)

        steps_per_epoch = min(len(source_loader), len(target_loader))
        total_steps = num_epochs * steps_per_epoch

        criterion_class = nn.CrossEntropyLoss()
        criterion_domain = nn.BCEWithLogitsLoss()

        history = {"class_loss": [], "domain_loss": [], "total_loss": []}

        global_step = 0
        model.train()

        for epoch in range(num_epochs):
            running_class_loss = 0.0
            running_domain_loss = 0.0
            running_total_loss = 0.0

            for _ in range(steps_per_epoch):
                global_step += 1
                alpha = get_alpha(global_step, total_steps)

                img_s, label_s, _ = next(iter_source)
                img_t, _, _ = next(iter_target)

                img_s, img_t = img_s.to(device), img_t.to(device)
                label_s = label_s.to(device)

                domain_y_s = torch.zeros(img_s.size(0), 1).to(device)
                domain_y_t = torch.ones(img_t.size(0), 1).to(device)

                optimizer.zero_grad()

                class_preds_s, _ = model(img_s, alpha=alpha)
                loss_class = criterion_class(class_preds_s, label_s)

                _, domain_preds_s = model(img_s, alpha=alpha)
                loss_dom_s = criterion_domain(domain_preds_s, domain_y_s)

                _, domain_preds_t = model(img_t, alpha=alpha)
                loss_dom_t = criterion_domain(domain_preds_t, domain_y_t)

                img_mix, lam = mixup_data(img_s, img_t, alpha=1.0)
                img_mix = img_mix.to(device)
                mixed_domain_label = (1 - lam) * torch.ones(img_s.size(0), 1).to(device)

                _, domain_preds_mix = model(img_mix, alpha=alpha)
                loss_dom_mix = criterion_domain(domain_preds_mix, mixed_domain_label)

                loss_domain_combined = loss_dom_s + loss_dom_t + loss_dom_mix
                loss_total = loss_class + loss_domain_combined * 0.5

                loss_total.backward()
                optimizer.step()

                running_class_loss += loss_class.item()
                running_domain_loss += loss_domain_combined.item()
                running_total_loss += loss_total.item()

            avg_class = running_class_loss / steps_per_epoch
            avg_domain = running_domain_loss / steps_per_epoch
            avg_total = running_total_loss / steps_per_epoch

            history["class_loss"].append(avg_class)
            history["domain_loss"].append(avg_domain)
            history["total_loss"].append(avg_total)

            print(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Class Loss: {avg_class:.4f} | "
                f"Dom Loss: {avg_domain:.4f} | "
                f"Total: {avg_total:.4f}"
            )

        save_dir = ROOT / "dann-rop-v3" / "checkpoints" / "phase1_cut"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"dann_model_fold_{fold+1}.pth"
        torch.save(model.state_dict(), save_path)

        best_total_loss = min(history["total_loss"]) if history["total_loss"] else None

        fold_results.append(
            {
                "fold": fold + 1,
                "model_path": str(save_path),
                "history": history,
                "best_total_loss": best_total_loss,
            }
        )

    return fold_results


if __name__ == "__main__":
    # exatamente o ponto solicitado
    results = train_dann_kfold(source_dataset, target_dataset, k_folds=5, num_epochs=30)

    save_dir = ROOT / "dann-rop-v3" / "checkpoints" / "phase1_cut"
    save_dir.mkdir(parents=True, exist_ok=True)

    best_fold = None
    if results:
        best_result = min(
            results,
            key=lambda item: (
                item.get("best_total_loss")
                if item.get("best_total_loss") is not None
                else float("inf")
            ),
        )
        best_fold = best_result.get("fold")

    results_path = save_dir / "phase1_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"best_fold": best_fold, "results": results},
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("Treino encerrado. Folds:", len(results))
    print("Resultados da fase 1 salvos em:", results_path)
