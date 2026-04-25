"""
data_prep/origa_loader.py
--------------------------
Carrega e pré-processa o dataset ORIGA (source domain).

Responsabilidade única (SRP): apenas leitura e formatação do ORIGA.

O CSV do ORIGA deve conter ao menos:
  - filename (ou image_path): nome/caminho do arquivo de imagem
  - label / Glaucoma: 1 = Glaucoma, 0 = Normal

O DataFrame resultante tem as colunas padronizadas:
  - image_path : caminho relativo ao images_dir (ou absoluto)
  - binary_label : int (0 ou 1)
  - patient_id : str (derivado do nome de arquivo ou coluna própria)
  - domain : 0  (source)
"""

import os
import pandas as pd


def load_origa(
    csv_path: str,
    images_dir: str,
    filename_col: str = "filename",
    label_col: str = "Glaucoma",
) -> pd.DataFrame:
    """
    Lê o CSV do ORIGA e retorna um DataFrame padronizado.

    Parâmetros
    ----------
    csv_path : str
        Caminho para o CSV de metadados do ORIGA.
    images_dir : str
        Diretório raiz das imagens (usado para montar o image_path completo
        ou simplesmente registrado como referência).
    filename_col : str
        Nome da coluna que contém o nome/caminho do arquivo de imagem.
    label_col : str
        Nome da coluna com o rótulo (1 = Glaucoma, 0 = Normal).

    Retorna
    -------
    pd.DataFrame com colunas:
        image_path, binary_label, patient_id, domain
    """
    df = pd.read_csv(csv_path)

    # ─── Validação mínima ────────────────────────────────────────────────────
    for col in [filename_col, label_col]:
        if col not in df.columns:
            raise ValueError(
                f"Coluna '{col}' não encontrada no CSV do ORIGA. "
                f"Colunas disponíveis: {list(df.columns)}"
            )

    # ─── Padronização ────────────────────────────────────────────────────────
    df = df[[filename_col, label_col]].copy()
    df.rename(columns={filename_col: "image_path", label_col: "binary_label"}, inplace=True)

    # Monta o caminho completo se não estiver embutido
    df["image_path"] = df["image_path"].apply(
        lambda p: p if os.path.isabs(p) else os.path.join(images_dir, p)
    )

    # Patient ID: derivado do nome de arquivo (sem extensão)
    df["patient_id"] = df["image_path"].apply(
        lambda p: os.path.splitext(os.path.basename(p))[0]
    )

    df["binary_label"] = df["binary_label"].astype(int)
    df["domain"] = 0  # source

    df = df[["image_path", "binary_label", "patient_id", "domain"]].reset_index(drop=True)

    print(
        f"[load_origa] {len(df)} imagens carregadas. "
        f"Glaucoma={df['binary_label'].sum()} | Normal={(df['binary_label']==0).sum()}"
    )

    return df
