"""
data_prep/rop_loader.py
------------------------
Carrega e pré-processa o dataset ROP (target domain).

Responsabilidade única (SRP): apenas leitura, formatação e split correto do ROP.

CORREÇÃO DO BUG EXPERIMENTAL:
  A separação do holdout de teste é feita AQUI, antes de qualquer treinamento,
  garantindo que nenhum paciente do conjunto de teste apareça nos folds de
  treino/validação da Fase 2.

O CSV de metadados do ROP deve conter ao menos:
  - image_path (ou filename): caminho/nome da imagem
  - binary_label: 0 = Normal, 1 = ROP
  - patient_id: identificador único do paciente

Se os patient_ids estiverem embutidos nos nomes dos arquivos,
ajuste a função `_extract_patient_id` abaixo.

O DataFrame resultante tem as colunas padronizadas:
  - image_path : caminho absoluto ou relativo ao images_dir
  - binary_label : int (0 ou 1)
  - patient_id : str
  - domain : 1  (target)
"""

import os
import pandas as pd

from datasets import build_rop_splits


# ─────────────────────────────────────────────────────────────────────────────
# Helper: extração de patient_id do nome de arquivo
# ─────────────────────────────────────────────────────────────────────────────

def _extract_patient_id_from_filename(filepath: str) -> str:
    """
    Extrai o patient_id do nome de arquivo ROP.

    Convenção real do dataset: os primeiros 3 caracteres do nome do arquivo.
    Ex: "001_OD_DG2_frame001.jpg" -> "001"
    """
    return os.path.basename(filepath)[:3]


def _extract_diagnosis_from_filename(filepath: str) -> int | None:
    """
    Extrai o código de diagnóstico do nome de arquivo ROP.

    Convenção: campo que começa com 'DG' seguido de número.
    Ex: "001_OD_DG2_frame001.jpg" -> 2
    Diagnóstico 0 = sem ROP (Normal), qualquer outro = ROP.
    """
    parts = os.path.basename(filepath).split("_")
    for part in parts:
        if part.upper().startswith("DG"):
            try:
                return int(part[2:])
            except ValueError:
                pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Loader principal
# ─────────────────────────────────────────────────────────────────────────────

def load_rop(
    images_dir: str,
    metadata_csv: str | None = None,
    filename_col: str = "image_path",
    label_col: str = "binary_label",
    patient_col: str = "patient_id",
    test_size: float = 0.20,
    random_state: int = 42,
    split: bool = True,
) -> "tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame":
    """
    Carrega o dataset ROP e opcionalmente realiza o split treino/teste
    seguro por paciente.

    Parâmetros
    ----------
    images_dir : str
        Diretório raiz das imagens.
    metadata_csv : str | None
        CSV com metadados. Se None, constrói o DataFrame varrendo images_dir.
    filename_col : str
        Coluna com o caminho/nome da imagem.
    label_col : str
        Coluna com o rótulo binário.
    patient_col : str
        Coluna com o patient_id.
    test_size : float
        Proporção de PACIENTES para o holdout de teste.
    random_state : int
    split : bool
        Se True, retorna (train_df, test_df).
        Se False, retorna o DataFrame completo (sem split).

    Retorna
    -------
    (train_df, test_df) se split=True, ou df_completo se split=False.
    """

    if metadata_csv is not None and os.path.isfile(metadata_csv):
        df = _load_from_csv(metadata_csv, filename_col, label_col, patient_col, images_dir)
    else:
        print("[load_rop] Nenhum CSV fornecido — varrendo images_dir para construir o DataFrame.")
        df = _build_from_directory(images_dir, label_col, patient_col)

    df["domain"] = 1  # target
    df = df[["image_path", "binary_label", "patient_id", "domain"]].reset_index(drop=True)

    print(
        f"[load_rop] {len(df)} imagens carregadas. "
        f"ROP={df['binary_label'].sum()} | Normal={(df['binary_label']==0).sum()} "
        f"| Pacientes únicos={df['patient_id'].nunique()}"
    )

    if split:
        return build_rop_splits(df, test_size=test_size, random_state=random_state)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Helpers internos
# ─────────────────────────────────────────────────────────────────────────────

def _load_from_csv(
    csv_path: str,
    filename_col: str,
    label_col: str,
    patient_col: str,
    images_dir: str,
) -> pd.DataFrame:
    """Carrega e padroniza o DataFrame a partir de um CSV de metadados."""
    df = pd.read_csv(csv_path)

    missing = [c for c in [filename_col, label_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no CSV ROP: {missing}. Disponíveis: {list(df.columns)}")

    df = df.copy()

    # Padroniza nome da coluna de imagem
    if filename_col != "image_path":
        df.rename(columns={filename_col: "image_path"}, inplace=True)

    # Padroniza nome da coluna de rótulo
    if label_col != "binary_label":
        df.rename(columns={label_col: "binary_label"}, inplace=True)

    # Monta caminho absoluto se necessário
    df["image_path"] = df["image_path"].apply(
        lambda p: p if os.path.isabs(p) else os.path.join(images_dir, p)
    )

    # Patient ID
    if patient_col in df.columns:
        df["patient_id"] = df[patient_col].astype(str)
    else:
        print(
            f"[load_rop] Coluna '{patient_col}' não encontrada — "
            "extraindo patient_id do nome de arquivo."
        )
        df["patient_id"] = df["image_path"].apply(_extract_patient_id_from_filename)

    df["binary_label"] = df["binary_label"].astype(int)

    return df


def _build_from_directory(
    images_dir: str,
    label_col: str,
    patient_col: str,
) -> pd.DataFrame:
    """
    Constrói o DataFrame varrendo images_dir.

    Usa a convenção real do dataset ROP:
      - patient_id = primeiros 3 caracteres do nome do arquivo
      - label = extraído do campo DGN no nome do arquivo (0=Normal, resto=ROP)
    """
    records = []
    for root, _, files in os.walk(images_dir):
        for fname in files:
            if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                continue
            fpath = os.path.join(root, fname)
            patient_id = _extract_patient_id_from_filename(fname)
            diagnosis = _extract_diagnosis_from_filename(fname)

            if diagnosis is None:
                continue  # ignora arquivos sem código de diagnóstico

            # 0 = sem ROP (Normal), qualquer outro código = ROP
            label = 0 if diagnosis == 0 else 1
            records.append({"image_path": fpath, "binary_label": label, "patient_id": patient_id})

    df = pd.DataFrame(records)
    if df.empty:
        print("[load_rop] AVISO: nenhuma imagem encontrada em", images_dir)
    return df
