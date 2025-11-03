import wandb
import optuna
import os
#carregar dot env
from dotenv import load_dotenv
load_dotenv()

DB_URL = os.getenv("DB_URL")

print(DB_URL)
study = optuna.load_study(
    study_name='dynamic_efficientnet_optimization_v1',
    storage=DB_URL
)

for trial in study.trials:  # testa só 3 trials
    if trial.value is None or trial.value < 0:
        continue

    run = wandb.init(
        project="dynamic_efficientnet_optimization_v1",
        name=f"trial_{trial.number}",
        config=trial.params,
        reinit=True
    )
    wandb.log({"avg_auc": trial.value})
    run.finish()
