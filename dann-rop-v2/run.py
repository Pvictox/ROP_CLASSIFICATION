"""
run.py
-------
Entry point do pipeline DANN-ROP v2.

Orquestra as fases:
  1. Carregamento e split correto dos dados (por paciente)
  2. Fase 1: Adaptação de domínio DANN (ORIGA→ROP)
  3. Fase 2: Fine-tuning supervisionado (GroupKFold por patient_id)
  4. Avaliação final: Ensemble no holdout de teste (nunca visto)

Uso:
    python run.py [--phase {1,2,all}] [--phase1-weights CAMINHO]

    python run.py                     # roda tudo (fase 1 + 2 + avaliação)
    python run.py --phase 1           # só fase 1
    python run.py --phase 2 --phase1-weights checkpoints/phase1/fold1_best.pth
    python run.py --phase eval --p2-dir checkpoints/phase2
"""

import argparse
import os
import sys

# ─── Garante que o diretório do projeto está no PYTHONPATH ───────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import default_config as cfg
from data_prep import load_origa, load_rop
from trainers import Phase1Trainer, Phase2Trainer
from evaluator import EnsembleEvaluator
from datasets import RetinaDataset, get_transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline DANN-ROP v2")
    parser.add_argument(
        "--phase",
        choices=["1", "2", "eval", "all"],
        default="all",
        help="Qual fase executar. 'all' executa tudo em sequência. (default: all)",
    )
    parser.add_argument(
        "--phase1-weights",
        type=str,
        default=None,
        help="Caminho para checkpoint da Fase 1 (usado na Fase 2).",
    )
    parser.add_argument(
        "--p2-dir",
        type=str,
        default=None,
        help="Diretório com checkpoints da Fase 2 (usado em --phase eval).",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results",
        help="Diretório para salvar os resultados finais. (default: results)",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Carregamento de dados
# ─────────────────────────────────────────────────────────────────────────────

def load_all_data():
    """
    Carrega ORIGA e ROP, realizando o split correto de teste por paciente.

    Retorna
    -------
    source_df  : DataFrame ORIGA completo
    rop_train_df : DataFrame ROP de treino (pacientes de treino)
    rop_test_df  : DataFrame ROP de teste (holdout — nunca visto)
    """
    print("\n[Dados] Carregando ORIGA (source)...")
    source_df = load_origa(
        csv_path=cfg.data.origa_csv,
        images_dir=cfg.data.origa_images_dir,
    )

    print("\n[Dados] Carregando ROP (target) com split por paciente...")
    rop_train_df, rop_test_df = load_rop(
        images_dir=cfg.data.rop_images_dir,
        metadata_csv=cfg.data.rop_metadata_csv,
        label_col=cfg.data.label_col,
        patient_col=cfg.data.patient_id_col,
        test_size=cfg.data.test_size,
        random_state=cfg.data.random_state,
        split=True,
    )

    return source_df, rop_train_df, rop_test_df


# ─────────────────────────────────────────────────────────────────────────────
# Fase 1
# ─────────────────────────────────────────────────────────────────────────────

def run_phase1(source_df, rop_train_df) -> dict:
    """Executa a Fase 1: Domain Adaptation DANN."""
    print("\n" + "="*70)
    print("  INICIANDO FASE 1 — Domain Adaptation (DANN)")
    print("="*70)

    trainer = Phase1Trainer(cfg)
    results = trainer.train_kfold(source_df, rop_train_df)

    print(f"\n[Fase 1] Concluída. Checkpoints em: {cfg.phase1.save_dir}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Fase 2
# ─────────────────────────────────────────────────────────────────────────────

def run_phase2(rop_train_df, phase1_weights_path: str | None = None) -> dict:
    """Executa a Fase 2: Fine-tuning supervisionado com GroupKFold por paciente."""
    print("\n" + "="*70)
    print("  INICIANDO FASE 2 — Fine-tuning (GroupKFold por patient_id)")
    print("="*70)

    if phase1_weights_path:
        print(f"  Pesos Fase 1: {phase1_weights_path}")
    else:
        print("  Pesos Fase 1: não fornecidos — inicializando com pré-treinado ImageNet")

    trainer = Phase2Trainer(cfg)
    results = trainer.train_kfold(rop_train_df, phase1_weights_path=phase1_weights_path)

    print(f"\n[Fase 2] Concluída. Checkpoints em: {cfg.phase2.save_dir}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Avaliação final
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(rop_test_df, checkpoint_paths: list, save_dir: str = "results") -> dict:
    """Avalia o ensemble no holdout de teste (nunca visto durante treino)."""
    print("\n" + "="*70)
    print("  AVALIAÇÃO FINAL — Ensemble no Holdout de Teste")
    print(f"  Modelos: {len(checkpoint_paths)} | Imagens de teste: {len(rop_test_df)}")
    print("="*70)

    evaluator = EnsembleEvaluator(cfg)
    results = evaluator.evaluate(
        rop_test_df,
        checkpoint_paths=checkpoint_paths,
        save_dir=save_dir,
        return_per_image=True,
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ─── Carrega dados (necessário para todas as fases) ───────────────────
    source_df, rop_train_df, rop_test_df = load_all_data()

    # ─── Fase 1 ───────────────────────────────────────────────────────────
    if args.phase in ("1", "all"):
        p1_results = run_phase1(source_df, rop_train_df)
        # Usa o checkpoint do fold com melhor AUC como pesos para Fase 2
        best_fold_idx = max(
            range(len(p1_results["fold_results"])),
            key=lambda i: p1_results["fold_results"][i]["auc"],
        )
        args.phase1_weights = p1_results["checkpoint_paths"][best_fold_idx]
        print(f"\n  Melhor checkpoint Fase 1 (fold {best_fold_idx+1}): {args.phase1_weights}")

    # ─── Fase 2 ───────────────────────────────────────────────────────────
    if args.phase in ("2", "all"):
        p2_results = run_phase2(rop_train_df, phase1_weights_path=args.phase1_weights)
        p2_checkpoints = p2_results["checkpoint_paths"]
    else:
        # Carrega checkpoints existentes da Fase 2 se --p2-dir foi fornecido
        if args.p2_dir:
            p2_checkpoints = sorted([
                os.path.join(args.p2_dir, f)
                for f in os.listdir(args.p2_dir)
                if f.endswith("_best.pth")
            ])
        else:
            p2_checkpoints = []

    # ─── Avaliação ────────────────────────────────────────────────────────
    if args.phase in ("eval", "all") and p2_checkpoints:
        run_evaluation(rop_test_df, p2_checkpoints, save_dir=args.save_dir)
    elif args.phase == "eval" and not p2_checkpoints:
        print("[ERRO] --phase eval requer checkpoints da Fase 2. Use --p2-dir.")


if __name__ == "__main__":
    main()
