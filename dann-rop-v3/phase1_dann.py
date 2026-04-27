from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# ============================================================
# Config
# ============================================================

@dataclass
class Phase1Config:
    # ORIGA (source)
    origa_csv: str = "/backup/lucas/datasets/origa/ORIGA/OrigaList.csv"
    origa_images_dir: str = "/backup/lucas/datasets/origa/ORIGA/Images"

    # ROP (target)
    rop_images_dir: str = "/backup/pedro_fonseca/PATIENT_ROP/DATASET/images_stack/images_stack"
    rop_metadata_csv: Optional[str] = None

    # treino
    image_size: int = 224
    batch_size: int = 32
    num_epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-5
    k_folds: int = 5
    alpha_max: float = 1.0
    mixup_alpha: float = 1.0
    seed: int = 42

    # split por paciente no alvo
    rop_test_size: float = 0.2

    # saída
    output_dir: str = "dann-rop-v3/artifacts"


# ============================================================
# Data loading and alignment
# ============================================================

def _infer_patient_id(path: str) -> str:
    return os.path.basename(path)[:3]


def _infer_diagnosis_code(path: str) -> Optional[int]:
    parts = os.path.basename(path).split("_")
    for part in parts:
        if part.upper().startswith("DG"):
            try:
                return int(part[2:])
            except ValueError:
                return None
    return None


def load_origa_dataframe(origa_csv: str, origa_images_dir: str) -> pd.DataFrame:
    df = pd.read_csv(origa_csv)

    filename_col = "Filename" if "Filename" in df.columns else "filename"
    if filename_col not in df.columns:
        raise ValueError(f"ORIGA sem coluna de arquivo. Colunas: {list(df.columns)}")

    label_col = "Glaucoma" if "Glaucoma" in df.columns else "label"
    if label_col not in df.columns:
        raise ValueError(f"ORIGA sem coluna de rótulo. Colunas: {list(df.columns)}")

    out = df[[filename_col, label_col]].copy()
    out.rename(columns={filename_col: "filepath", label_col: "binary_label"}, inplace=True)
    out["filepath"] = out["filepath"].apply(
        lambda p: p if os.path.isabs(str(p)) else os.path.join(origa_images_dir, str(p))
    )
    out["binary_label"] = out["binary_label"].astype(int)
    out["patient_id"] = out["filepath"].apply(lambda p: Path(p).stem)
    out["domain"] = 0

    out["exists"] = out["filepath"].apply(os.path.exists)
    out = out[out["exists"]].drop(columns=["exists"]).reset_index(drop=True)
    print(
        f"[ORIGA] imagens={len(out)} | pos={int(out['binary_label'].sum())} | "
        f"neg={int((out['binary_label']==0).sum())}"
    )
    return out


def _load_rop_from_csv(rop_metadata_csv: str, rop_images_dir: str) -> pd.DataFrame:
    df = pd.read_csv(rop_metadata_csv)

    if "filepath" in df.columns:
        path_col = "filepath"
    elif "image_path" in df.columns:
        path_col = "image_path"
    elif "Filename" in df.columns:
        path_col = "Filename"
    else:
        raise ValueError(f"CSV ROP sem coluna de caminho. Colunas: {list(df.columns)}")

    out = pd.DataFrame({"filepath": df[path_col].astype(str)})
    out["filepath"] = out["filepath"].apply(
        lambda p: p if os.path.isabs(p) else os.path.join(rop_images_dir, p)
    )

    if "binary_label" in df.columns:
        out["binary_label"] = df["binary_label"].astype(int)
    elif "diagnosis_code" in df.columns:
        out["binary_label"] = (df["diagnosis_code"].astype(int) != 0).astype(int)
    else:
        out["binary_label"] = out["filepath"].apply(
            lambda p: 0 if (_infer_diagnosis_code(p) or 0) == 0 else 1
        )

    if "patient_id" in df.columns:
        out["patient_id"] = df["patient_id"].astype(str)
    else:
        out["patient_id"] = out["filepath"].apply(_infer_patient_id)

    return out


def _load_rop_from_directory(rop_images_dir: str) -> pd.DataFrame:
    records = []
    for root, _, files in os.walk(rop_images_dir):
        for fname in files:
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                continue
            full = os.path.join(root, fname)
            dg = _infer_diagnosis_code(full)
            if dg is None:
                continue
            records.append(
                {
                    "filepath": full,
                    "binary_label": 0 if dg == 0 else 1,
                    "patient_id": _infer_patient_id(full),
                }
            )

    if len(records) == 0:
        raise FileNotFoundError(f"Nenhuma imagem válida encontrada em {rop_images_dir}")

    return pd.DataFrame.from_records(records)


def load_rop_dataframe(rop_images_dir: str, rop_metadata_csv: Optional[str] = None) -> pd.DataFrame:
    if rop_metadata_csv and os.path.isfile(rop_metadata_csv):
        out = _load_rop_from_csv(rop_metadata_csv, rop_images_dir)
        print(f"[ROP] carregado de CSV: {rop_metadata_csv}")
    else:
        out = _load_rop_from_directory(rop_images_dir)
        print("[ROP] carregado por varredura de diretório")

    out["domain"] = 1
    out["exists"] = out["filepath"].apply(os.path.exists)
    out = out[out["exists"]].drop(columns=["exists"]).reset_index(drop=True)
    print(
        f"[ROP] imagens={len(out)} | pos={int(out['binary_label'].sum())} | "
        f"neg={int((out['binary_label']==0).sum())} | pacientes={out['patient_id'].nunique()}"
    )
    return out


def split_rop_by_patient(
    rop_df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    groups = rop_df["patient_id"].values
    train_idx, test_idx = next(splitter.split(rop_df, rop_df["binary_label"], groups=groups))

    rop_train = rop_df.iloc[train_idx].copy().reset_index(drop=True)
    rop_test = rop_df.iloc[test_idx].copy().reset_index(drop=True)

    overlap = set(rop_train["patient_id"]) & set(rop_test["patient_id"])
    if overlap:
        raise RuntimeError(f"Overlap de pacientes encontrado: {len(overlap)}")

    print(
        f"[SPLIT ROP] train={len(rop_train)} ({rop_train['patient_id'].nunique()} pacientes) | "
        f"test={len(rop_test)} ({rop_test['patient_id'].nunique()} pacientes)"
    )
    return rop_train, rop_test


# ============================================================
# Dataset and transforms
# ============================================================


def get_transforms(is_train: bool, img_size: int = 224) -> transforms.Compose:
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if is_train:
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=30),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.ToTensor(),
                norm,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            norm,
        ]
    )


class RetinaDomainDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None, domain_label: int = 0, return_path: bool = False):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.domain_label = domain_label
        self.return_path = return_path

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = row["filepath"]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(int(row["binary_label"]), dtype=torch.long)
        domain = torch.tensor(float(self.domain_label), dtype=torch.float32)

        if self.return_path:
            return image, label, domain, path
        return image, label, domain


# ============================================================
# DANN model
# ============================================================


class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x, alpha=1.0):
    return GradientReversalFn.apply(x, alpha)


class EfficientNetDANN(nn.Module):
    def __init__(self, num_classes: int = 2, pre_trained: bool = True):
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
        domain_output = self.domain_classifier(grad_reverse(features, alpha))
        return class_output, domain_output

    def extract_features(self, x):
        features = self.backbone.features(x)
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, 1)
        return features


# ============================================================
# Train phase 1
# ============================================================


def get_alpha(current_step: int, total_steps: int) -> float:
    p = float(current_step) / max(total_steps, 1)
    return 2.0 / (1.0 + np.exp(-10 * p)) - 1.0


def mixup_data(x1: torch.Tensor, x2: torch.Tensor, alpha: float = 1.0) -> tuple[torch.Tensor, float]:
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    mixed_x = lam * x1 + (1 - lam) * x2
    return mixed_x, float(lam)


def _infinite_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")


def evaluate_classification(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    running_loss = 0.0
    all_y, all_pred, all_prob = [], [], []

    with torch.no_grad():
        for imgs, labels, _ in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits, _ = model(imgs, alpha=0.0)
            loss = criterion(logits, labels)
            running_loss += loss.item()

            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            pred = logits.argmax(dim=1).cpu().numpy()
            all_prob.extend(probs.tolist())
            all_pred.extend(pred.tolist())
            all_y.extend(labels.cpu().numpy().tolist())

    y_true = np.array(all_y)
    y_pred = np.array(all_pred)
    y_prob = np.array(all_prob)

    return {
        "val_loss": running_loss / max(len(val_loader), 1),
        "val_acc": float(accuracy_score(y_true, y_pred)),
        "val_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "val_auc": _safe_auc(y_true, y_prob),
    }


def train_phase1_kfold(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    cfg: Phase1Config,
    device: Optional[torch.device] = None,
) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(cfg.output_dir)
    ckpt_dir = out_dir / "checkpoints" / "phase1"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    y_source = source_df["binary_label"].values
    skf = StratifiedKFold(n_splits=cfg.k_folds, shuffle=True, random_state=cfg.seed)

    fold_results = []
    checkpoint_paths = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(source_df)), y_source), start=1):
        print("\n" + "=" * 70)
        print(f"FASE 1 - Fold {fold}/{cfg.k_folds}")
        print("=" * 70)

        src_train_df = source_df.iloc[train_idx].reset_index(drop=True)
        src_val_df = source_df.iloc[val_idx].reset_index(drop=True)

        source_train_ds = RetinaDomainDataset(
            src_train_df,
            transform=get_transforms(True, cfg.image_size),
            domain_label=0,
        )
        source_val_ds = RetinaDomainDataset(
            src_val_df,
            transform=get_transforms(False, cfg.image_size),
            domain_label=0,
        )
        target_ds = RetinaDomainDataset(
            target_df,
            transform=get_transforms(True, cfg.image_size),
            domain_label=1,
        )

        source_train_loader = DataLoader(source_train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
        source_val_loader = DataLoader(source_val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
        target_loader = DataLoader(target_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

        if len(source_train_loader) == 0 or len(target_loader) == 0:
            raise RuntimeError("Loader vazio. Reduza o batch_size ou valide os DataFrames.")

        model = EfficientNetDANN(num_classes=2, pre_trained=True).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        criterion_class = nn.CrossEntropyLoss()
        criterion_domain = nn.BCEWithLogitsLoss()

        best_auc = -1.0
        best_ckpt = ckpt_dir / f"fold{fold}_best.pth"

        source_iter = _infinite_loader(source_train_loader)
        target_iter = _infinite_loader(target_loader)

        steps_per_epoch = min(len(source_train_loader), len(target_loader))
        total_steps = cfg.num_epochs * steps_per_epoch
        global_step = 0

        history = {
            "train_class_loss": [],
            "train_domain_loss": [],
            "train_total_loss": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
            "val_auc": [],
        }

        for epoch in range(cfg.num_epochs):
            model.train()
            running_class, running_domain, running_total = 0.0, 0.0, 0.0

            for _ in range(steps_per_epoch):
                global_step += 1
                alpha = cfg.alpha_max * get_alpha(global_step, total_steps)

                img_s, label_s, _ = next(source_iter)
                img_t, _, _ = next(target_iter)

                img_s = img_s.to(device)
                img_t = img_t.to(device)
                label_s = label_s.to(device)

                domain_y_s = torch.zeros(img_s.size(0), 1, device=device)
                domain_y_t = torch.ones(img_t.size(0), 1, device=device)

                optimizer.zero_grad()

                class_preds_s, domain_preds_s = model(img_s, alpha=alpha)
                _, domain_preds_t = model(img_t, alpha=alpha)

                loss_class = criterion_class(class_preds_s, label_s)
                loss_dom_s = criterion_domain(domain_preds_s, domain_y_s)
                loss_dom_t = criterion_domain(domain_preds_t, domain_y_t)

                # mixup inter-domínio
                img_mix, lam = mixup_data(img_s, img_t, alpha=cfg.mixup_alpha)
                mixed_domain_label = (1 - lam) * torch.ones(img_s.size(0), 1, device=device)
                _, domain_preds_mix = model(img_mix, alpha=alpha)
                loss_dom_mix = criterion_domain(domain_preds_mix, mixed_domain_label)

                loss_domain = loss_dom_s + loss_dom_t + loss_dom_mix
                loss_total = loss_class + 0.5 * loss_domain

                loss_total.backward()
                optimizer.step()

                running_class += loss_class.item()
                running_domain += loss_domain.item()
                running_total += loss_total.item()

            val_metrics = evaluate_classification(model, source_val_loader, criterion_class, device)

            history["train_class_loss"].append(running_class / steps_per_epoch)
            history["train_domain_loss"].append(running_domain / steps_per_epoch)
            history["train_total_loss"].append(running_total / steps_per_epoch)
            history["val_loss"].append(val_metrics["val_loss"])
            history["val_acc"].append(val_metrics["val_acc"])
            history["val_f1"].append(val_metrics["val_f1"])
            history["val_auc"].append(val_metrics["val_auc"])

            print(
                f"Epoch {epoch+1:02d}/{cfg.num_epochs} | "
                f"Cls {history['train_class_loss'][-1]:.4f} | "
                f"Dom {history['train_domain_loss'][-1]:.4f} | "
                f"Tot {history['train_total_loss'][-1]:.4f} | "
                f"Val AUC {val_metrics['val_auc']:.4f} | "
                f"Val Acc {val_metrics['val_acc']:.4f}"
            )

            val_auc_for_best = -1.0 if np.isnan(val_metrics["val_auc"]) else val_metrics["val_auc"]
            if val_auc_for_best > best_auc:
                best_auc = val_auc_for_best
                torch.save(
                    {
                        "fold": fold,
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_metrics": val_metrics,
                        "config": asdict(cfg),
                    },
                    best_ckpt,
                )

        checkpoint_paths.append(str(best_ckpt))
        fold_results.append(
            {
                "fold": fold,
                "best_val_auc": best_auc,
                "checkpoint": str(best_ckpt),
                "history": history,
            }
        )

    mean_auc = float(np.nanmean([f["best_val_auc"] for f in fold_results]))
    std_auc = float(np.nanstd([f["best_val_auc"] for f in fold_results]))

    results = {
        "fold_results": fold_results,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "checkpoint_paths": checkpoint_paths,
    }

    results_path = ckpt_dir / "phase1_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n[FASE 1] Finalizado")
    print(f"AUC média: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"Resultados: {results_path}")
    return results


# ============================================================
# Feature extraction + t-SNE
# ============================================================


def extract_features(
    model: nn.Module,
    df: pd.DataFrame,
    domain_name: str,
    image_size: int = 224,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = RetinaDomainDataset(
        df=df,
        transform=get_transforms(False, image_size),
        domain_label=0,
        return_path=True,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_feats, all_labels, all_paths = [], [], []
    model.eval()

    with torch.no_grad():
        for imgs, labels, _, paths in loader:
            imgs = imgs.to(device)
            feats = model.extract_features(imgs).detach().cpu().numpy()
            all_feats.append(feats)
            all_labels.extend(labels.numpy().tolist())
            all_paths.extend([str(p) for p in paths])

    feat = np.concatenate(all_feats, axis=0)
    meta = pd.DataFrame({"domain": domain_name, "label": all_labels, "filepath": all_paths})
    return feat, meta


def run_tsne(features: np.ndarray, seed: int = 42, perplexity: float = 30.0) -> np.ndarray:
    tsne = TSNE(
        n_components=2,
        init="pca",
        perplexity=perplexity,
        learning_rate="auto",
        random_state=seed,
    )
    return tsne.fit_transform(features)


# ============================================================
# CLI
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="DANN Phase-1 (ORIGA -> ROP)")
    parser.add_argument("--train", action="store_true", help="Executa treino da Fase 1")
    parser.add_argument("--origa-csv", type=str, default=None)
    parser.add_argument("--origa-images", type=str, default=None)
    parser.add_argument("--rop-images", type=str, default=None)
    parser.add_argument("--rop-csv", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = Phase1Config()
    if args.origa_csv:
        cfg.origa_csv = args.origa_csv
    if args.origa_images:
        cfg.origa_images_dir = args.origa_images
    if args.rop_images:
        cfg.rop_images_dir = args.rop_images
    if args.rop_csv:
        cfg.rop_metadata_csv = args.rop_csv
    if args.epochs:
        cfg.num_epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.output_dir:
        cfg.output_dir = args.output_dir

    source_df = load_origa_dataframe(cfg.origa_csv, cfg.origa_images_dir)
    rop_df = load_rop_dataframe(cfg.rop_images_dir, cfg.rop_metadata_csv)
    rop_train_df, rop_test_df = split_rop_by_patient(rop_df, cfg.rop_test_size, cfg.seed)

    split_dir = Path(cfg.output_dir) / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    rop_train_df.to_csv(split_dir / "rop_train_split.csv", index=False)
    rop_test_df.to_csv(split_dir / "rop_test_split.csv", index=False)

    if args.train:
        train_phase1_kfold(source_df=source_df, target_df=rop_train_df, cfg=cfg)
    else:
        print("Dados alinhados e splits salvos. Use --train para treinar a Fase 1.")


if __name__ == "__main__":
    main()
