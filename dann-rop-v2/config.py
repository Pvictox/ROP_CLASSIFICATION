"""
config.py
---------
Configurações centralizadas do pipeline DANN-ROP v2.

Responsabilidade única (SRP): este módulo só gerencia configurações.
Altere os caminhos e hiperparâmetros aqui conforme necessário.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    # ─── ORIGA ────────────────────────────────────────────────────────────────
    # CSV deve ter colunas 'Filename' e 'Glaucoma' (convenção da data_factory)
    origa_csv: str = "/backup/ORIGA/metadata.csv"      # ← AJUSTE AQUI
    origa_images_dir: str = "/backup/ORIGA/images"     # ← AJUSTE AQUI

    # ─── ROP ──────────────────────────────────────────────────────────────────
    # Diretório com as imagens .jpg do ROP (varre recursivamente)
    rop_images_dir: str = "/backup/pedro_fonseca/PATIENT_ROP/DATASET/images_stack/images_stack"
    # Sem CSV de metadados: usa varredura do diretório pelo nome do arquivo
    rop_metadata_csv: Optional[str] = None

    # ─── Colunas do DataFrame interno ─────────────────────────────────────────
    # Nomes usados INTERNAMENTE após a carga (não nos CSVs originais)
    patient_id_col: str = "patient_id"
    label_col: str = "binary_label"   # 0 = Normal, 1 = ROP
    image_col: str = "filepath"       # ← coluna real no data_factory / ROPDataset

    # Fração de PACIENTES reservada exclusivamente para teste final.
    # Esses pacientes nunca entram em nenhum fold de treino/validação.
    test_size: float = 0.20
    random_state: int = 42


@dataclass
class ModelConfig:
    backbone: str = "efficientnet_b0"   # nome compatível com timm
    num_classes_source: int = 2         # ORIGA: Glaucoma vs Normal
    num_classes_target: int = 2         # ROP: Doença vs Normal
    dropout: float = 0.3


@dataclass
class Phase1Config:
    """Configurações do treinamento DANN (Fase 1)."""
    num_epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-4
    k_folds: int = 5                    # folds sobre source (ORIGA)
    alpha_max: float = 1.0              # valor máximo de alpha do GRL
    save_dir: str = "checkpoints/phase1"
    use_mixup: bool = True
    mixup_alpha: float = 0.4


@dataclass
class Phase2Config:
    """Configurações do fine-tuning supervisionado (Fase 2)."""
    num_epochs: int = 30
    batch_size: int = 16
    lr: float = 5e-5
    weight_decay: float = 1e-4
    k_folds: int = 5                    # GroupKFold por patient_id (pacientes de treino)
    save_dir: str = "checkpoints/phase2"
    # Caminho(s) para os pesos da Fase 1 (lista com 1 caminho ou None para aleatorio)
    phase1_weights: Optional[str] = None


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    phase1: Phase1Config = field(default_factory=Phase1Config)
    phase2: Phase2Config = field(default_factory=Phase2Config)
    device: str = "cuda"               # "cuda" ou "cpu"


# Instância padrão — importe onde precisar
default_config = Config()
