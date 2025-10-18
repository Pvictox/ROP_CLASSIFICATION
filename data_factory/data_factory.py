from re import X
import pandas as pd
import os 
from glob import glob
from typing import List
from sympy import Subs
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

from data_factory.ROP_dataset import ROPDataset

class DataFactory:
    def __init__(self, img_path, metadata_path):
        self.img_path = img_path
        self.metadata_path = metadata_path


    #Todo: Realizar uma filtragem de acordo com o código do diagnóstico. 
    def load_data(self, verbose=True, allowed_diagnoses=None) -> pd.DataFrame | None:
        image_files = glob(os.path.join(self.img_path, "**", "*.jpg"), recursive=True)
        if not image_files:
            raise FileNotFoundError(f"Nenhum arquivo de imagem encontrado em {self.img_path}")
        if verbose:
            print(f"Número de arquivos de imagem encontrados: {len(image_files)}")

        patient_ids, valid_files, new_binary_labels, diagnosis_codes = [], [], [], []
        for file_path in image_files:
            diagnosis = self.get_diagnosis_from_filename(file_path)
            if allowed_diagnoses is not None and diagnosis not in allowed_diagnoses: # Se a lista de diagnósticos permitidos for fornecida, filtrar
                continue
            if diagnosis is not None:
                diagnosis_codes.append(diagnosis)
                patient_id = self.get_patient_id_from_filename(file_path)
                if patient_id is None:
                    continue
                patient_ids.append(patient_id)
                binary_label = self.convert_to_binary_label(diagnosis, not_rop_list=[0, 10, 11, 12, 13]) #COnsiderando apenas 0 como sem ROP
                valid_files.append(file_path)
                new_binary_labels.append(binary_label)
        if verbose:
                print(f"Número de arquivos válidos após filtragem: {len(valid_files)}")
        
        binarized_dataframe = pd.DataFrame({
            'patient_id': patient_ids,
            'filepath': valid_files,
            'binary_label': new_binary_labels,
            'diagnosis_code': diagnosis_codes
        })

        return binarized_dataframe


    def prepare_metadata(self, metadata_path, verbose=True) -> pd.DataFrame | None:
        try:
            metadata_df:pd.DataFrame = pd.read_csv(metadata_path)
            if verbose:
                print(f"Shape dos metadados: {metadata_df.shape}")
                print(f"Colunas dos metadados: {metadata_df.columns.tolist()}")
                return metadata_df
        except Exception as e:
            raise RuntimeError(f"Erro ao ler o arquivo CSV de metadados: {e}")
        return None

    def get_diagnosis_from_filename(self, filename:str) -> int | None:
        '''
            Dado o nome do arquivo de imagem (ex: IMG_001_DG2_....jpg), extrai o diagnóstico, no caso o valor numérico após 'DG'.
            Obtido de: https://www.kaggle.com/code/mugwewaithaka/rop-screening-binary-deeplearning-model
        '''
        try:
            parts = os.path.basename(filename).split('_')
            for part in parts:
                if part.startswith('DG'):
                    return int(part[2:])  
            return None
        except Exception as e:
            raise RuntimeError(f"Erro ao extrair o diagnóstico do nome do arquivo {filename}: {e}")

    def get_patient_id_from_filename(self, filename:str) -> str | None:
        '''
            Dado o nome do arquivo de imagem, extrai o ID do paciente, no caso os primeiros três caracteres.
        '''
        try:
            return os.path.basename(filename)[:3]
        except Exception as e:
            raise RuntimeError(f"Erro ao extrair o ID do paciente do nome do arquivo {filename}: {e}")

    def convert_to_binary_label(self, code_diagnosis:int, not_rop_list:List[int]) -> int:
        '''
            Converte o código de diagnóstico em um rótulo binário.
            0: Sem ROP (Retinopatia da Prematuridade)
            1: Com ROP

            not_rop_list: lista de códigos que indicam ausência de ROP. Ex: [0, 1, 2] indica que 0, 1 e 2 são códigos sem ROP.
        '''
        if code_diagnosis in not_rop_list:
            return 0
        else:
            return 1

    def prepare_data_for_cross_validation(self, rop_dataset:ROPDataset, test_size=0.2, random_state=42, num_splits=5, verbose=True):
        import numpy as np
        
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        all_images_paths, all_labels, all_patient_ids = rop_dataset.get_data_from_dataframe()
        
        # Converter para arrays NumPy se necessário
        all_images_paths = np.array(all_images_paths)
        all_labels = np.array(all_labels)
        all_patient_ids = np.array(all_patient_ids)
        
        train_indx, test_indx = next(gss.split(all_images_paths, all_labels, groups=all_patient_ids))

        train_dataset = Subset(rop_dataset, train_indx.tolist())
        test_dataset = Subset(rop_dataset, test_indx.tolist())

        patient_ids_train = all_patient_ids[train_indx]

        if verbose:
            print(f"Numero de amostras no conjunto de treino: {len(train_dataset)}")
            print(f"Numero de amostras no conjunto de teste: {len(test_dataset)}")
        
        gkf = GroupKFold(n_splits=num_splits)
        X_train = all_images_paths[train_indx]
        y_train = all_labels[train_indx]

        # train_df = pd.DataFrame({
        #     'patient_id': all_patient_ids[train_indx],
        #     'filepath': all_images_paths[train_indx],
        #     'binary_label': all_labels[train_indx]
        # })

        # #salvando como parquet
        # train_df.to_parquet('/home/pedro_fonseca/PATIENT_ROP/saved_subsets/train_data.parquet', index=False)
        # test_df = pd.DataFrame({
        #     'patient_id': all_patient_ids[test_indx],
        #     'filepath': all_images_paths[test_indx],
        #     'binary_label': all_labels[test_indx]
        # })
        # test_df.to_parquet('/home/pedro_fonseca/PATIENT_ROP/saved_subsets/test_data.parquet', index=False)



        return X_train, y_train, train_indx, patient_ids_train, gkf, test_dataset

