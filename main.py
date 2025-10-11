
from data_factory.data_factory import DataFactory
from utils import Utils
from data_factory.ROP_dataset import ROPDataset

IMG_FILE_PATH = '/home/pedro_fonseca/PATIENT_ROP/DATASET/images_stack_without_captions/images_stack_without_captions/'
CSV_FILE_PATH = '/home/pedro_fonseca/PATIENT_ROP/DATASET/infant_retinal_database_info.csv'

IS_INTERACTIVE = True #Isto é uma flag para indicar se o código está sendo executado em um ambiente interativo (como Jupyter Notebook) ou não.

def main():
    data_factory = DataFactory(IMG_FILE_PATH, CSV_FILE_PATH)
    df = data_factory.load_data()
    if df is not None:
        print(df.head())
        print(f"Número total de imagens processadas: {len(df)}")
        if IS_INTERACTIVE:
            rop_dataset = ROPDataset(df, is_train=False, apply_clahe=True)
            #Dividir em treino e teste
            #Treino e evalidação por paciente
            #Utils.plot_sample_images(rop_dataset, num_samples=5)
            
    




if __name__ == '__main__':
    main()