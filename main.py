

import test
from data_factory.data_factory import DataFactory
from utils import Utils
from data_factory.ROP_dataset import ROPDataset
from train_and_val_worker import TrainAndEvalWorker
import os

IMG_FILE_PATH = '/backup/lucas/rop_dataset/images_stack_without_captions/images_stack_without_captions'
CSV_FILE_PATH = '/backup/lucas/rop_dataset/infant_retinal_database_info.csv'

IS_INTERACTIVE = True #Isto é uma flag para indicar se o código está sendo executado em um ambiente interativo (como Jupyter Notebook) ou não.

os.makedirs('saved_models', exist_ok=True)
def main():
    data_factory = DataFactory(IMG_FILE_PATH, CSV_FILE_PATH)
    df = data_factory.load_data(allowed_diagnoses=[0, 1, 2, 3, 4,8,9])  #1–4, 8, 9 (ROP) contra 0 (physiological)
    # df = data_factory.load_data()
    if df is not None:
        print(df.head())
        print(f"Número total de imagens processadas: {len(df)}")        
        rop_dataset = ROPDataset(df, is_train=False, apply_clahe=True)
        # X_train, y_train, train_indx, patient_ids_train, gkf, test_dataset = data_factory.prepare_data_for_cross_validation(rop_dataset)
        X_train, y_train, train_indx, patient_ids_train, gkf, test_dataset = data_factory.prepare_data_for_cross_validation_2(rop_dataset,num_splits=5)

        train_and_val_worker = TrainAndEvalWorker(config=None)
        print("Iniciando o treinamento e validação com GroupKFold...")
        train_and_val_worker.train(X_train, y_train, patient_ids_train, train_indx, gkf, rop_dataset)
        # print("Iniciando a avaliação no conjunto de teste...")
        results = train_and_val_worker.evaluate(test_dataset, model_path='saved_models/best_model_efficientNET.pth')

        #     #Dividir em treino e teste
            #Treino e evalidação por paciente
            #Utils.plot_sample_images(rop_dataset, num_samples=5)
            
    




if __name__ == '__main__':
    main()