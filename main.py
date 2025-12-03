

import test
from data_factory.data_factory import DataFactory
from utils import Utils
from data_factory.ROP_dataset import ROPDataset
from train_and_val_worker import TrainAndEvalWorker
import os


from optuna_trials import OptunaTrials




# db_url = os.getenv('DB_URL')



IMG_FILE_PATH = '/backup/pedro_fonseca/PATIENT_ROP/DATASET'
CSV_FILE_PATH = '/backup/pedro_fonseca/PATIENT_ROP/DATASET/infant_retinal_database_info.csv'

# IMG_FILE_PATH = '/backup/lucas/rop_dataset/images_stack_without_captions/images_stack_without_captions'
# CSV_FILE_PATH = '/backup/lucas/rop_dataset/infant_retinal_database_info.csv'


IS_INTERACTIVE = True #Isto é uma flag para indicar se o código está sendo executado em um ambiente interativo (como Jupyter Notebook) ou não.

os.makedirs('saved_models', exist_ok=True)
def main():
    data_factory = DataFactory(IMG_FILE_PATH, CSV_FILE_PATH)
    df = data_factory.load_data(allowed_diagnoses=[0, 1, 2, 3, 4,5,6,7,8,9, 10,11,12,13])  #1–4, 8, 9 (ROP) contra 0 (physiological)
    # df = data_factory.load_data()
    if df is not None:
        print(df.head())
        print(f"Número total de imagens processadas: {len(df)}")        
        rop_dataset = ROPDataset(df, is_train=False, apply_clahe=True)
        # X_train, y_train, train_indx, patient_ids_train, gkf, test_dataset = data_factory.prepare_data_for_cross_validation(rop_dataset)
        X_train, y_train, train_indx, patient_ids_train, gkf, test_dataset = data_factory.prepare_data_for_cross_validation_3(rop_dataset, num_splits=3)
        X_train, y_train, train_indx, patient_ids_train, gkf, test_dataset = data_factory.prepare_data_for_cross_validation_3(rop_dataset, num_splits=3)

        train_full_subset, val_full_subset, test_subset = data_factory.prepare_train_val_and_test_datasets(rop_dataset)

        # pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0, interval_steps=1)
        # study = optuna.create_study(direction='maximize', pruner=pruner, study_name='dynamic_efficientnet_optimization_trials_final_MEU_DEUS', storage=db_url, load_if_exists=True
        #             )
        # optuna_trials = OptunaTrials()
        # N_TRIALS = 50
        # try:
        #     study.optimize(lambda trial: optuna_trials.objective(trial, X_train, y_train, patient_ids_train, train_indx, gkf, rop_dataset, num_trials=N_TRIALS, full_train_subset=train_full_subset, full_val_subset=val_full_subset, test_subset=test_subset), n_trials=N_TRIALS)
        # except KeyboardInterrupt:
        #     print("Otimização interrompida pelo usuário.")

        #worker = TrainAndEvalWorker(config=None, model=None)
        # results = worker.evaluate(test_subset, model_path='saved_models/best_dynamic_efficientnet.pth')
        #optuna_trials.save_best_model()
        # print("\n--- Otimização Concluída ---")
        # print(f"Melhor trial: {study.best_trial.number}")
        # print(f"Melhor Acurácia de Validação: {study.best_value:.4f}")
        
        # print("\nMelhor Arquitetura Encontrada:")
        # for key, value in study.best_params.items():
        #     print(f"  {key}: {value}")

        # #Salvando melhor arquitetura em um arquivo de texto
        # with open('best_architecture.txt', 'w') as f:
        #     f.write(f"Melhor trial: {study.best_trial.number}\n")
        #     f.write(f"Melhor Acurácia de Validação: {study.best_value:.4f}\n")
        #     f.write("\nMelhor Arquitetura Encontrada:\n")
        #     for key, value in study.best_params.items():
        #         f.write(f"{key}: {value}\n")
        train_and_val_worker = TrainAndEvalWorker(config=None)
        print("Iniciando a avaliação no conjunto de teste...")
        results = train_and_val_worker.evaluate(test_dataset, model_path='/backup/pedro_fonseca/PATIENT_ROP/saved_models/best_dynamic_efficientnet.pth')

        #     #Dividir em treino e teste
            #Treino e evalidação por paciente
            #Utils.plot_sample_images(rop_dataset, num_samples=5)
            
    




if __name__ == '__main__':
    main()