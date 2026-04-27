# dann-rop-v3

Organização da **Fase 1 do DANN** em um único arquivo Python:

- phase1_dann.py: carga/alinhamento de datasets, split por paciente, modelo DANN, treino K-Fold da Fase 1 e extração de features.
- tsne_before_after_phase1.ipynb: visualização de features com t-SNE antes/depois da Fase 1.

## Execução rápida

1) Só alinhar datasets e salvar splits:

```bash
python dann-rop-v3/phase1_dann.py
```

2) Alinhar + treinar Fase 1:

```bash
python dann-rop-v3/phase1_dann.py --train
```

3) Abrir notebook:

- dann-rop-v3/tsne_before_after_phase1.ipynb

No notebook, rode as células em ordem. Se já houver checkpoints da Fase 1, deixe `RUN_PHASE1 = False`.
