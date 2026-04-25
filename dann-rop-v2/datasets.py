"""
datasets.py
-----------
Dataset genérico e utilitários de split correto por paciente.

Responsabilidade única (SRP): este módulo só lida com dados/transformações.

Correção do bug experimental:
  O split de teste é feito ANTES de qualquer treino usando GroupShuffleSplit
  por patient_id. Nenhum paciente do holdout de teste aparece em qualquer
  fold de treino ou validação.

Uso típico:
    from datasets import RetinaDataset, build_rop_splits, get_transforms

    train_df, test_df = build_rop_splits(rop_df, test_size=0.2)
    dataset = RetinaDataset(train_df, root_dir=..., transform=get_transforms(True))
"""

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset
from torchvision import transforms


# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────

def get_transforms(is_train: bool, image_size: int = 224) -> transforms.Compose:
    """
    Retorna as transformações de dados.

    Treino: augmentações leves + normalização ImageNet.
    Val/Teste: apenas resize/crop + normalização ImageNet.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class RetinaDataset(Dataset):
    """
    Dataset genérico para imagens de retina (ORIGA e ROP).

    O DataFrame deve conter ao menos as colunas:
      - image_path : caminho relativo ao root_dir (ou absoluto se root_dir="")
      - binary_label : rótulo numérico (int)
      - patient_id : identificador do paciente (opcional, mas necessário p/ split)

    Parâmetros
    ----------
    dataframe : pd.DataFrame
    root_dir : str
        Diretório raiz das imagens. Se "" usa os caminhos como absolutos.
    transform : transforms.Compose
    domain_label : int
        0 = source (ORIGA), 1 = target (ROP).
    return_filename : bool
        Se True, retorna também o nome do arquivo (útil na inferência).
    label_col : str
        Nome da coluna de rótulo no DataFrame.
    image_col : str
        Nome da coluna de caminho de imagem no DataFrame.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        domain_label: int = 0,
        return_filename: bool = False,
        label_col: str = "binary_label",
        image_col: str = "image_path",
    ):
        self.df = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.domain_label = domain_label
        self.return_filename = return_filename
        self.label_col = label_col
        self.image_col = image_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row[self.image_col]

        if self.root_dir:
            img_path = os.path.join(self.root_dir, img_path)

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(int(row[self.label_col]), dtype=torch.long)
        domain = torch.tensor(self.domain_label, dtype=torch.long)

        if self.return_filename:
            return image, label, domain, str(img_path)
        return image, label, domain


# ─────────────────────────────────────────────────────────────────────────────
# Split correto por paciente  ←  CORREÇÃO DO BUG EXPERIMENTAL
# ─────────────────────────────────────────────────────────────────────────────

def build_rop_splits(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
    patient_col: str = "patient_id",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide o DataFrame ROP em treino e teste garantindo que nenhum
    paciente apareça em ambos os conjuntos.

    Usa GroupShuffleSplit com grupos = patient_id.

    Parâmetros
    ----------
    df : pd.DataFrame com coluna `patient_col`
    test_size : float, proporção de PACIENTES reservada para teste
    random_state : int, semente
    patient_col : str, nome da coluna com o ID do paciente

    Retorna
    -------
    (train_df, test_df) — DataFrames sem overlap de pacientes

    Uso posterior:
        Para os folds de cross-validation da Fase 2, use train_df com
        GroupKFold(grupos=train_df[patient_col]).
        O test_df nunca deve ser usado durante qualquer etapa de treino.
    """
    groups = df[patient_col].values

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=groups))

    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)

    # Verificação de sanidade
    train_patients = set(train_df[patient_col])
    test_patients = set(test_df[patient_col])
    overlap = train_patients & test_patients
    assert len(overlap) == 0, (
        f"BUG: {len(overlap)} paciente(s) aparecem em treino E teste! "
        f"Pacientes com overlap: {overlap}"
    )

    print(
        f"[build_rop_splits] Pacientes treino={len(train_patients)} "
        f"| Pacientes teste={len(test_patients)} "
        f"| Imagens treino={len(train_df)} | Imagens teste={len(test_df)}"
    )

    return train_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# Mixup inter-domínio (auxiliar para Fase 1)
# ─────────────────────────────────────────────────────────────────────────────

def mixup_data(
    x_s: torch.Tensor,
    x_t: torch.Tensor,
    alpha: float = 0.4,
) -> Tuple[torch.Tensor, float]:
    """
    Mixup entre imagens source e target.

    Retorna a imagem misturada e o lambda usado.
    """
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    mixed = lam * x_s + (1 - lam) * x_t
    return mixed, lam
