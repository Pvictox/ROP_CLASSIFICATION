import optuna
import torch 
from models.dynamic_efficient_net import DynamicEfficientNet
from train_and_val_worker import TrainAndEvalWorker

class OptunaTrials:
    def __init__(self):
        self.base_state_config = [
            (32, 16, 1, 1),  # Estágio 1
            (16, 24, 2, 2),  # Estágio 2
            (24, 40, 2, 2),  # Estágio 3
            (40, 80, 3, 2),  # Estágio 4
            (80, 112, 3, 1),  # Estágio 5
            (112, 192, 4, 2), # Estágio 6
            (192, 320, 1, 1), # Estágio 7
        ]

        self.best_model = None
        self.best_auc = 0.0
    
    def save_best_model(self, path='saved_models/best_dynamic_efficientnet.pth'):
        if self.best_model is not None:
            torch.save(self.best_model.state_dict(), path)
            print(f"Melhor modelo salvo em {path}")
        else:
            print("Nenhum modelo para salvar.")
    
    def objective(self, trial, X_train, y_train, patient_ids_train, train_indx, gkf, rop_dataset):
        device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
        dynamic_config = [] #Para salvar as config

        MAX_STAGE_FOR_ATT = 3
        
        for i, (in_c, out_c, n_layers, stride) in enumerate(self.base_state_config):
            cfg = {}
            
            # Partes Estáticas
            cfg['in_channels'] = in_c
            cfg['out_channels'] = out_c
            cfg['num_layers'] = n_layers
            cfg['stride'] = stride

            cfg['block_type'] = trial.suggest_categorical(f"stage_{i}_block", ['MBConv', 'FusedMBConv'])
            cfg['kernel_size'] = trial.suggest_categorical(f"stage_{i}_kernel", [3, 5])
            cfg['expand_ratio'] = trial.suggest_categorical(f"stage_{i}_expand", [4, 6])
            cfg['attention_type'] = trial.suggest_categorical(f"stage_{i}_attention", ['se', 'cbam', 'none'])
            
            if i <= MAX_STAGE_FOR_ATT:
                cfg['attention_type'] = trial.suggest_categorical(f"stage_{i}_attention", ['se', 'cbam', 'none'])
                
                if cfg['attention_type'] != 'none':
                    cfg['att_reduction'] = trial.suggest_categorical(f"stage_{i}_att_reduction", [4, 8, 16])
                else:
                    cfg['att_reduction'] = 4 
            else:
                cfg['attention_type'] = 'none'
                cfg['att_reduction'] = 4 
            
            dynamic_config.append(cfg)

        model = DynamicEfficientNet(dynamic_config).to(device)
        try:
            folds_results, avg_auc = TrainAndEvalWorker(config=None, model=model).train(X_train, y_train, patient_ids_train, train_indx, gkf, rop_dataset, trial, dynamic_config=dynamic_config)
        except Exception as e:
            print(f"Trial {trial.number} falhou com erro: {e}")
            return -1.0 # Retorna uma acurácia muito ruim

        if avg_auc > self.best_auc:
            self.best_auc = avg_auc
            self.best_model = model
        return avg_auc
        
