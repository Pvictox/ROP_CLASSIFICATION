"""
Fase 2 (corrigido por paciente) baseada no phase_2_cut.py.

Objetivo:
- manter o estilo de treino da fase 2 para comparacao
- corrigir o split para evitar vazamento entre pacientes

Fluxo:
1) carrega ROP via DataFactory
2) cria split train/test em nivel de paciente (estratificado por classe)
3) executa GroupKFold no train_df usando patient_id como grupo
4) faz fine-tuning por fold inicializando da fase 1
"""

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, train_test_split
from torch import optim
from torch.utils.data import DataLoader
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from data_factory.data_factory import DataFactory


V2_DIR = ROOT / "dann-rop-v2"
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from datasets import RetinaDataset  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fase 2 corrigida (split por paciente) com scheduler cosseno"
    )
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--num-classes-target", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--max-images-per-patient",
        type=int,
        default=100,
        help="Limite de imagens por paciente antes do split por paciente",
    )
    parser.add_argument(
        "--keep-mixed-patients",
        action="store_true",
        help="Se ligado, nao remove pacientes com labels mistos",
    )
    parser.add_argument(
        "--phase1-checkpoint",
        type=str,
        default=None,
        help=(
            "Se informado, usa o mesmo checkpoint da fase 1 para todos os folds. "
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
        default=str(ROOT / "dann-rop-v3" / "checkpoints" / "phase2_corrigo_cosine"),
    )
    parser.add_argument("--optimizer", type=str, choices=["adam", "adamw"], default="adam")
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["none", "cosine", "cosine-restarts"],
        default="cosine",
    )
    parser.add_argument("--eta-min", type=float, default=1e-6)
    parser.add_argument("--t0", type=int, default=5, help="T_0 para cosine-restarts")
    parser.add_argument("--t-mult", type=int, default=2, help="T_mult para cosine-restarts")
    parser.add_argument(
        "--monitor",
        type=str,
        choices=["val_acc", "val_f1", "val_auc"],
        default="val_f1",
    )
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=2,
        help="Congela backbone nas primeiras epocas",
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
    print("ROP_df total:", len(rop_df))
    return rop_df


def build_patient_splits(
    rop_df,
    test_size=0.2,
    random_state=42,
    max_images_per_patient=100,
    keep_mixed_patients=False,
):
    df = rop_df.copy()

    if not keep_mixed_patients:
        patient_stats = df.groupby("patient_id")["binary_label"].agg(["min", "max"])
        pure_patients = patient_stats[(patient_stats["min"] == patient_stats["max"])].index
        df = df[df["patient_id"].isin(pure_patients)].copy()

    capped_parts = []
    for _, group in df.groupby("patient_id"):
        if len(group) > max_images_per_patient:
            capped_parts.append(group.sample(max_images_per_patient, random_state=random_state))
        else:
            capped_parts.append(group)

    df = pd.concat(capped_parts, axis=0).reset_index(drop=True)

    # Label de paciente para estratificacao train/test em nivel de paciente
    patient_class = df.groupby("patient_id")["binary_label"].mean().reset_index()
    patient_class["label_class"] = (patient_class["binary_label"] > 0.5).astype(int)

    train_patients, test_patients = train_test_split(
        patient_class["patient_id"],
        test_size=test_size,
        stratify=patient_class["label_class"],
        random_state=random_state,
    )

    train_df = df[df["patient_id"].isin(train_patients)].copy().reset_index(drop=True)
    test_df = df[df["patient_id"].isin(test_patients)].copy().reset_index(drop=True)

    train_pat_set = set(train_df["patient_id"])
    test_pat_set = set(test_df["patient_id"])
    overlap = train_pat_set & test_pat_set
    if overlap:
        raise RuntimeError(f"Overlap de pacientes entre train/test: {len(overlap)}")

    print("\n[Split por paciente]")
    print("Pacientes treino:", len(train_pat_set), "| Pacientes teste:", len(test_pat_set))
    print("Imagens treino:", len(train_df), "| Imagens teste:", len(test_df))

    return train_df, test_df


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


def compute_epoch_auc(all_labels, all_probs, num_classes_target):
    try:
        if num_classes_target == 2:
            all_probs_np = np.array(all_probs)
            return roc_auc_score(all_labels, all_probs_np[:, 1])
        return roc_auc_score(
            all_labels,
            all_probs,
            multi_class="ovr",
            average="weighted",
        )
    except ValueError:
        return 0.0


def build_optimizer_and_scheduler(args, model):
    param_groups = [
        {"params": model.backbone.parameters(), "lr": args.lr_backbone},
        {"params": model.class_classifier.parameters(), "lr": args.lr_head},
    ]

    if args.optimizer == "adamw":
        optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(param_groups, weight_decay=args.weight_decay)

    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(args.num_epochs, 1),
            eta_min=args.eta_min,
        )
    elif args.scheduler == "cosine-restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(args.t0, 1),
            T_mult=max(args.t_mult, 1),
            eta_min=args.eta_min,
        )

    return optimizer, scheduler


def step_scheduler(scheduler, scheduler_name, epoch):
    if scheduler is None:
        return
    if scheduler_name == "cosine-restarts":
        scheduler.step(epoch + 1)
    else:
        scheduler.step()


def get_lrs(optimizer):
    if len(optimizer.param_groups) >= 2:
        return optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"]
    lr = optimizer.param_groups[0]["lr"]
    return lr, lr


def get_monitor_value(metric_name, val_acc, val_f1, val_auc):
    if metric_name == "val_acc":
        return val_acc
    if metric_name == "val_auc":
        return val_auc
    return val_f1


def set_backbone_grad(model, requires_grad):
    for param in model.backbone.parameters():
        param.requires_grad = requires_grad


def train_finetuning_groupkfold(
    train_df,
    phase1_ckpt_dir,
    train_transforms,
    val_transforms,
    num_classes_target=2,
    k_folds=5,
    num_epochs=30,
    batch_size=32,
    num_workers=4,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    phase1_checkpoint=None,
    save_dir=None,
    args=None,
):
    print("DEVICE:", device)

    x_train = train_df["filepath"].values
    y_train = train_df["binary_label"].values
    groups = train_df["patient_id"].values

    gkf = GroupKFold(n_splits=k_folds)
    results_phase2 = []
    save_dir.mkdir(parents=True, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(x_train, y_train, groups=groups), 1):
        print(f"\n{'#' * 50}")
        print(f"FASE 2 CORRIGIDA - FOLD {fold}/{k_folds}")
        print(f"{'#' * 50}")

        fold_train_df = train_df.iloc[train_idx].copy().reset_index(drop=True)
        fold_val_df = train_df.iloc[val_idx].copy().reset_index(drop=True)

        train_patients = set(fold_train_df["patient_id"])
        val_patients = set(fold_val_df["patient_id"])
        overlap = train_patients & val_patients
        if overlap:
            raise RuntimeError(f"Overlap de pacientes no fold {fold}: {len(overlap)}")

        print("Pacientes treino:", len(train_patients), "| Pacientes val:", len(val_patients))
        print("Imagens treino:", len(fold_train_df), "| Imagens val:", len(fold_val_df))

        train_dataset = RetinaDataset(
            dataframe=fold_train_df,
            root_dir="",
            transform=train_transforms,
            domain_label=1,
            image_col="filepath",
        )
        val_dataset = RetinaDataset(
            dataframe=fold_val_df,
            root_dir="",
            transform=val_transforms,
            domain_label=1,
            image_col="filepath",
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

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

        optimizer, scheduler = build_optimizer_and_scheduler(args, model)

        criterion = nn.CrossEntropyLoss()

        history = {
            "train_loss": [],
            "train_acc": [],
            "train_f1": [],
            "train_auc": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
            "val_auc": [],
            "lr_backbone": [],
            "lr_head": [],
        }

        best_metric = -float("inf")
        best_epoch = 0
        patience_counter = 0
        best_model_path = save_dir / f"finetuned_best_fold_{fold}.pth"

        for epoch in range(num_epochs):
            if epoch < args.freeze_backbone_epochs:
                set_backbone_grad(model, requires_grad=False)
            else:
                set_backbone_grad(model, requires_grad=True)

            model.train()
            running_loss = 0.0
            train_preds = []
            train_labels = []
            train_probs = []

            for imgs, labels, _ in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)

                optimizer.zero_grad()
                class_preds, _ = model(imgs, alpha=0.0)
                loss = criterion(class_preds, labels)

                probs = F.softmax(class_preds, dim=1)
                _, preds = torch.max(class_preds, 1)
                train_probs.extend(probs.detach().cpu().numpy())
                train_preds.extend(preds.detach().cpu().numpy())
                train_labels.extend(labels.detach().cpu().numpy())

                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            epoch_train_loss = running_loss / max(len(train_loader), 1)
            epoch_train_acc = accuracy_score(train_labels, train_preds)
            epoch_train_f1 = f1_score(train_labels, train_preds, average="weighted")
            epoch_train_auc = compute_epoch_auc(train_labels, train_probs, num_classes_target)

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
            epoch_val_auc = compute_epoch_auc(all_labels, all_probs, num_classes_target)

            history["train_loss"].append(epoch_train_loss)
            history["train_acc"].append(epoch_train_acc)
            history["train_f1"].append(epoch_train_f1)
            history["train_auc"].append(epoch_train_auc)
            history["val_loss"].append(epoch_val_loss)
            history["val_acc"].append(epoch_val_acc)
            history["val_f1"].append(epoch_val_f1)
            history["val_auc"].append(epoch_val_auc)
            lr_backbone, lr_head = get_lrs(optimizer)
            history["lr_backbone"].append(lr_backbone)
            history["lr_head"].append(lr_head)

            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {epoch_train_loss:.4f} | "
                f"Train Acc/F1/AUC: {epoch_train_acc:.4f}/{epoch_train_f1:.4f}/{epoch_train_auc:.4f} | "
                f"Val Loss: {epoch_val_loss:.4f} | "
                f"Val Acc/F1/AUC: {epoch_val_acc:.4f}/{epoch_val_f1:.4f}/{epoch_val_auc:.4f} | "
                f"LR(bb/head): {lr_backbone:.2e}/{lr_head:.2e}"
            )

            monitor_value = get_monitor_value(
                metric_name=args.monitor,
                val_acc=epoch_val_acc,
                val_f1=epoch_val_f1,
                val_auc=epoch_val_auc,
            )

            if monitor_value > (best_metric + args.min_delta):
                best_metric = monitor_value
                best_epoch = epoch + 1
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1

            step_scheduler(scheduler=scheduler, scheduler_name=args.scheduler, epoch=epoch)

            if args.early_stop_patience > 0 and patience_counter >= args.early_stop_patience:
                print(
                    f"Early stopping no fold {fold} na epoca {epoch+1} "
                    f"(monitor={args.monitor}, patience={args.early_stop_patience})"
                )
                break

        results_phase2.append(
            {
                "fold": fold,
                "best_metric": best_metric,
                "best_metric_name": args.monitor,
                "best_epoch": best_epoch,
                "best_model_path": str(best_model_path),
                "history": history,
                "phase1_init": str(weights_path),
                "train_patients": len(train_patients),
                "val_patients": len(val_patients),
                "optimizer": args.optimizer,
                "scheduler": args.scheduler,
                "lr_backbone": args.lr_backbone,
                "lr_head": args.lr_head,
                "weight_decay": args.weight_decay,
            }
        )

    return results_phase2


def save_split_artifacts(save_dir, train_df, test_df):
    save_dir.mkdir(parents=True, exist_ok=True)

    train_csv = save_dir / "patient_train_split.csv"
    test_csv = save_dir / "patient_test_split.csv"
    meta_json = save_dir / "split_meta.json"

    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    meta = {
        "train_images": int(len(train_df)),
        "test_images": int(len(test_df)),
        "train_patients": int(train_df["patient_id"].nunique()),
        "test_patients": int(test_df["patient_id"].nunique()),
    }

    with meta_json.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    device = resolve_device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_transforms, val_transforms = get_transforms(img_size=256)
    rop_df = load_rop_dataframe()

    train_df, test_df = build_patient_splits(
        rop_df=rop_df,
        test_size=args.test_size,
        random_state=args.seed,
        max_images_per_patient=args.max_images_per_patient,
        keep_mixed_patients=args.keep_mixed_patients,
    )

    phase1_ckpt_dir = Path(args.phase1_ckpt_dir)
    phase1_checkpoint = Path(args.phase1_checkpoint) if args.phase1_checkpoint else None
    save_dir = Path(args.save_dir)

    save_split_artifacts(save_dir=save_dir, train_df=train_df, test_df=test_df)

    results_phase2 = train_finetuning_groupkfold(
        train_df=train_df,
        phase1_ckpt_dir=phase1_ckpt_dir,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        num_classes_target=args.num_classes_target,
        k_folds=args.k_folds,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        phase1_checkpoint=phase1_checkpoint,
        save_dir=save_dir,
        args=args,
    )

    out_json = save_dir / "phase2_results.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results_phase2, f, ensure_ascii=False, indent=2)

    print("\nFase 2 corrigida encerrada.")
    print("Folds treinados:", len(results_phase2))
    print("Resultados salvos em:", out_json)


if __name__ == "__main__":
    main()
