import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
import test
from torch.utils.data import Subset, DataLoader
import numpy as np
from data_factory.ROP_SUBSET_dataset import ROPSubset
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import torch.optim as optim
import timm
from torchvision import transforms
from torch.utils.data import WeightedRandomSampler
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
import optuna
from models.dynamic_efficient_net import DynamicEfficientNet

class TrainAndEvalWorker:
    def __init__(self, config:dict, model=None):
        if not config:
            print("Configuração vazia fornecida. Utilizando valores padrão.")
            self.config = {
                'learning_rate': 1e-3,
                'weight_decay': 1e-4,
                'batch_size': 32,
                'num_epochs_cross': 10,
                'num_epochs': 30,
                'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
                'patience': 3
            }
        else:
            self.config = config

        self.model = model if model else timm.create_model('efficientnet_b0', pretrained=True, num_classes=1).to(self.config['device'])
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 1e-4),
            weight_decay=self.config.get('weight_decay', 1e-5)
        )
        # self.criterion = nn.BCEWithLogitsLoss()
        self.criterion = BinaryFocalLoss(alpha=0.75, gamma=1.0)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5)

    def train_epoch(self, train_loader, val_loader):
        self.model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.config['device']), targets.float().to(self.config['device'])
            self.optimizer.zero_grad()
            outputs = self.model(data).squeeze(-1)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            predicted = torch.sigmoid(outputs) > 0.5
            train_total += targets.size(0)
            train_correct += (predicted == targets.bool()).sum().item()
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        train_accuracy = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        val_loss, val_auc, val_threshold, val_f1 = self.validate(val_loader)

        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val AUC: {val_auc:.4f}, Val F1: {val_f1:.4f}')

        self.scheduler.step(val_loss)
        
        return train_accuracy, avg_train_loss, val_loss, val_auc, val_threshold, val_f1



    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        all_targets = []
        all_outputs = []
        all_predictions = []

        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.config['device']), targets.float().to(self.config['device'])
                outputs = self.model(data).squeeze(-1)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()

                probs = torch.sigmoid(outputs)
                all_targets.extend(targets.cpu().numpy())
                all_outputs.extend(probs.cpu().numpy())

        all_targets = np.array(all_targets)
        all_outputs = np.array(all_outputs)

        # Calcular AUC
        try:
            auc = roc_auc_score(all_targets, all_outputs, average='weighted')
        except:
            auc = 0.5

        # Calcular o threshold ótimo (Youden index)
        try:
            fpr, tpr, thresholds = roc_curve(all_targets, all_outputs)
            youden_index = tpr - fpr
            best_threshold = thresholds[np.argmax(youden_index)]
        except:
            best_threshold = 0.5  # fallback caso falte classe

        # Calcular F1-Score com o threshold ótimo
        predictions = (all_outputs > best_threshold).astype(int)
        f1 = f1_score(all_targets, predictions, zero_division=0)

        avg_val_loss = val_loss / len(val_loader)

        return avg_val_loss, auc, best_threshold, f1



    def setup_finetuning(self):
        for param in self.model.parameters():
            param.requires_grad = False
        
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        

    def custom_collate_fn(self, batch):
        images = []
        labels = []
        
        for item in batch:
            if len(item) == 3:  # (image, label, patient_id)
                image, label, patient_id = item
                
                # Verificar se já é um tensor
                if isinstance(image, torch.Tensor):
                    images.append(image)
                elif hasattr(image, 'mode'):  # É uma PIL Image
                    transform = transforms.ToTensor()
                    image = transform(image)
                    images.append(image)
                else:
                    raise TypeError(f"Tipo de imagem não suportado: {type(image)}")
                
                labels.append(label)
        
        return torch.stack(images), torch.tensor(labels)


    def train_cross_validation(self, X_train, y_train, patient_ids_train, train_index, gkf:GroupKFold, rop_dataset, trial, dynamic_config=None):
        fold_results = []
        best_global_auc = 0.0
        best_model_state = None
        best_fold = 0
        best_thresholds_per_fold = []
        
        fold_aucs = []
        fold_f1_scores = []
        
        for fold, (train_fold_idx, val_fold_idx) in enumerate(gkf.split(X_train, y_train, groups=patient_ids_train)):
            fold_patience = 0
            print(f"{'='*20}")
            print(f"Fold {fold+1}/{gkf.n_splits}")
            print(f"{'='*20}")

            train_fold_absolute_index = train_index[train_fold_idx]
            val_fold_absolute_index = train_index[val_fold_idx]

            train_fold_dataset = Subset(rop_dataset, train_fold_absolute_index)
            val_fold_dataset = Subset(rop_dataset, val_fold_absolute_index)

            train_fold_dataset = ROPSubset(train_fold_dataset, transform=rop_dataset.train_transformations, apply_clahe=True)
            val_fold_dataset = ROPSubset(val_fold_dataset, transform=rop_dataset.val_and_test_transformations, apply_clahe=True)

            labels = [sample[1] for sample in train_fold_dataset]
            class_sample_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])
            weight = 1. / class_sample_count
            samples_weight = np.array([weight[int(t)] for t in labels])
            samples_weight = torch.from_numpy(samples_weight).double()
            print(f"Samples weight distribution: {samples_weight}")

            sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
            train_loader = DataLoader(
                train_fold_dataset, 
                batch_size=self.config.get('batch_size', 32),
                shuffle=True,
                num_workers=0,
                collate_fn=self.custom_collate_fn
            )
            val_loader = DataLoader(
                val_fold_dataset,
                batch_size=self.config.get('batch_size', 32),
                shuffle=False,
                num_workers=0,
                collate_fn=self.custom_collate_fn
            )

            # REINICIAR O MODELO E OTIMIZADOR PARA CADA FOLD
            self.model = DynamicEfficientNet(dynamic_config).to(self.config['device'])
            
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.get('learning_rate', 1e-4),
                weight_decay=self.config.get('weight_decay', 1e-5)
            )

            best_val_auc = 0.0
            best_threshold_fold = 0.5
            best_fold_f1 = 0.0
            epochs = self.config.get('num_epochs_cross', 20)
            
            train_losses = []
            val_losses = []

            for epoch in range(epochs):
                print(f'\nEpoch {epoch+1}/{epochs}')
                train_acc, train_loss, val_loss, val_auc, val_threshold, val_f1 = self.train_epoch(train_loader, val_loader)

                train_losses.append(train_loss)
                val_losses.append(val_loss)

                # RESETAR PATIENCE QUANDO MELHORA
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_threshold_fold = val_threshold
                    best_fold_f1 = val_f1
                    fold_patience = 0
                else:
                    fold_patience += 1
                
                # REMOVER REPORT E PRUNING DAQUI - DEIXAR APENAS NO TREINO FINAL
                
                if fold_patience >= self.config.get('patience', 5):
                    print(f"Early stopping no fold {fold+1} na epoch {epoch+1}")
                    break

                print(f'Epoch finished. Current fold best Val AUC: {best_val_auc:.4f}, F1: {val_f1:.4f}, Global best AUC: {best_global_auc:.4f}')
            
            # Armazenar métricas do melhor resultado deste fold
            fold_aucs.append(best_val_auc)
            fold_f1_scores.append(best_fold_f1)

            print(f"Fold {fold+1} best threshold: {best_threshold_fold:.4f} (best val AUC: {best_val_auc:.4f}, best F1: {best_fold_f1:.4f})")
            
            best_thresholds_per_fold.append(best_threshold_fold)
            
            fold_results.append({
                'fold': fold + 1,
                'best_val_auc': best_val_auc,
                'best_f1_score': best_fold_f1,
                'best_threshold': best_threshold_fold
            })
            
            # Atualizar melhor fold globalmente
            if best_val_auc > best_global_auc:
                best_global_auc = best_val_auc
                best_fold = fold + 1
    
        mean_auc = np.mean(fold_aucs)
        std_auc = np.std(fold_aucs)
        mean_f1 = np.mean(fold_f1_scores)
        std_f1 = np.std(fold_f1_scores)
        avg_threshold = np.mean(best_thresholds_per_fold)
        
        print(f"\n{'='*50}")
        print(f"Resultados da Validação Cruzada")
        print(f"{'='*50}")
        print(f"AUC - Média: {mean_auc:.4f} ± {std_auc:.4f}")
        print(f"F1-Score - Média: {mean_f1:.4f} ± {std_f1:.4f}")
        print(f"Melhor Fold: {best_fold} (AUC: {best_global_auc:.4f})")
        print(f"Threshold médio: {avg_threshold:.4f}")
        print(f"{'='*50}\n")
        
        self.avg_threshold = avg_threshold
        
        # Salvar resultados detalhados
        results_df = pd.DataFrame(fold_results)
        #results_df.to_csv('cross_validation_results.csv', index=False)
        
        return {
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'mean_f1': mean_f1,
            'std_f1': std_f1,
            'best_global_auc': best_global_auc,
            'best_fold': best_fold,
            'avg_threshold': avg_threshold,
            'fold_results': fold_results
        }

    def train(self, X_train, y_train, patient_ids_train, train_index, gkf:GroupKFold, rop_dataset, trial, full_train_subset:ROPSubset, full_val_subset:ROPSubset, test_subset:ROPSubset, dynamic_config=None):
        '''
            Primeiro se realiza o treino da validação cruzada. Depois realiza um treino completo com os dados
            de treino e validação, salvando o melhor modelo baseado no AUC de validação.
        '''
        
        # Etapa 1: Validação Cruzada
        cv_results = self.train_cross_validation(
            X_train, y_train, patient_ids_train, train_index, 
            gkf, rop_dataset, trial, dynamic_config=dynamic_config
        )
        
        print("\n" + "="*50)
        print("Iniciando treino completo com todos os dados de treino...")
        print("="*50 + "\n")
        
        # Etapa 2: Preparar DataLoaders completos
        full_train_loader = DataLoader(
            full_train_subset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=True,
            num_workers=0,
            collate_fn=self.custom_collate_fn
        )
        full_val_loader = DataLoader(
            full_val_subset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=False,
            num_workers=0,
            collate_fn=self.custom_collate_fn
        )
        test_loader = DataLoader(
            test_subset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=False,
            num_workers=0,
            collate_fn=self.custom_collate_fn
        )
        
        # Etapa 3: Reinicializar o modelo para o treino final
        self.model = DynamicEfficientNet(dynamic_config).to(self.config['device'])
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 1e-4),
            weight_decay=self.config.get('weight_decay', 1e-5)
        )
        
        # Etapa 4: Treinar no conjunto completo
        best_val_auc = 0.0
        best_model_state = None
        best_epoch = 0
        best_threshold = 0.5
        best_val_f1 = 0.0
        
        epochs = self.config.get('num_epochs', 20)
        train_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_auc': [],
            'val_f1': []
        }
        
        train_patience = 0

        for epoch in range(epochs):
            print(f'\n{"="*30}')
            print(f'Epoch {epoch+1}/{epochs}')
            print(f'{"="*30}')
            
            train_acc, train_loss, val_loss, val_auc, val_threshold, val_f1 = self.train_epoch(
                full_train_loader, full_val_loader
            )
            
            # Armazenar histórico
            train_history['train_loss'].append(train_loss)
            train_history['train_acc'].append(train_acc)
            train_history['val_loss'].append(val_loss)
            train_history['val_auc'].append(val_auc)
            train_history['val_f1'].append(val_f1)
            
            # Salvar melhor modelo baseado no AUC de validação
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = self.model.state_dict().copy()
                best_epoch = epoch + 1
                best_threshold = val_threshold
                best_val_f1 = val_f1
                train_patience = 0 
                print(f'Novo melhor modelo! Val AUC: {best_val_auc:.4f}, F1: {best_val_f1:.4f}')
            else:
                train_patience += 1
            
            if trial is not None:
                trial.report(val_auc, epoch)
                
                # Verificar pruning
                if trial.should_prune():
                    print("Trial pruned by Optuna.")
                    raise optuna.TrialPruned()
        
            if train_patience >= self.config.get('patience', 5):
                print(f"Early stopping na epoch {epoch+1}")
                break
            
            print(f'Best Val AUC so far: {best_val_auc:.4f} (Epoch {best_epoch})')
    
        self._plot_training_curves(train_history, trial_number = trial.number)

        # Retornar resultados completos
        results_training =  {
            'cv_results': cv_results,
            'best_val_auc': best_val_auc,
            'best_val_f1': best_val_f1,
            'best_epoch': best_epoch,
            'best_threshold': best_threshold,
            'train_history': train_history,
            'model': self.model,
            
        }

        #Salvando localmente
        os.makedirs(f"saved_models/complete_training/trial_{trial.number}", exist_ok=True)
        pd.DataFrame(results_training).to_csv(f"saved_models/complete_training/trial_{trial.number}/training_results.csv")
        return results_training
        

    def evaluate(self, test_dataset, model, trial_number):
        self.model = model 
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=False,
            num_workers=0,
            collate_fn=self.custom_collate_fn
        )
        
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_probs = []
        test_loss = 0.0
        
        print("Avaliando modelo no conjunto de teste...")
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.config['device']), targets.float().to(self.config['device'])
                outputs = self.model(data).squeeze(-1)
                
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()
                
                probs = torch.sigmoid(outputs)
                # usa threshold médio calculado durante o treino, se disponível
                # threshold = getattr(self, 'avg_threshold', 0.5)
                threshold = 0.5
                predicted = (probs > threshold).float() 
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        all_predictions = np.array(all_predictions).astype(int) 
        all_targets = np.array(all_targets).astype(int)
        all_probs = np.array(all_probs)
        print(f"Using threshold = {getattr(self, 'avg_threshold', 0.5):.4f} for final predictions")
        
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, zero_division=0)
        recall = recall_score(all_targets, all_predictions, zero_division=0)
        f1 = f1_score(all_targets, all_predictions, zero_division=0)
        conf_matrix = confusion_matrix(all_targets, all_predictions)
        
        # Calcula AUC-ROC se houver ambas as classes
        try:
            auc_roc = roc_auc_score(all_targets, all_probs, average='weighted')
        except:
            auc_roc = None
            print("Aviso: Não foi possível calcular AUC-ROC (pode haver apenas uma classe no conjunto de teste)")
        
        avg_test_loss = test_loss / len(test_loader)
        
        # Exibe resultados
        print(f"\n{'='*50}")
        print(f"Resultados da Avaliação no Conjunto de Teste")
        print(f"{'='*50}")
        print(f"Test Loss: {avg_test_loss:.4f}")
        print(f"Accuracy: {accuracy*100:.2f}%")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        if auc_roc is not None:
            print(f"AUC-ROC: {auc_roc:.4f}")
        print(f"\nConfusion Matrix:")
        print(conf_matrix)
        print(f"{'='*50}")

        # salvar matriz de confusão como imagem
        os.makedirs(f"saved_models/plot/trial_{trial_number}", exist_ok=True)
        disp = ConfusionMatrixDisplay(conf_matrix)
        disp.plot(cmap='Blues')
        plt.title("Confusion Matrix - Test Set")
        plt.savefig(f"saved_models/plot/trial_{trial_number}/test_confusion_matrix.png")
        plt.close()
        ####
        results = {
            'test_loss': avg_test_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            # 'confusion_matrix': conf_matrix,
            
        }
        # mostrando resultados com threshold padrão 0.5 também
        # threshold

        #salvando results
        os.makedirs(f"saved_models/test_results/trial_{trial_number}", exist_ok=True)
        pd.DataFrame(results, index=[0]).to_csv(f"saved_models/test_results/trial_{trial_number}/test_results.csv")

        return results

    def _plot_training_curves(self, history, trial_number):
        """
        Plota as curvas de Loss, AUC e F1-Score durante o treinamento.
        """
        os.makedirs(f"saved_models/plot/train_curves/trial_{trial_number}", exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(history['train_loss'], label='Train Loss', marker='o')
        axes[0, 0].plot(history['val_loss'], label='Val Loss', marker='s')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(history['train_acc'], label='Train Accuracy', marker='o', color='green')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Training Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # AUC
        axes[1, 0].plot(history['val_auc'], label='Val AUC', marker='d', color='purple')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].set_title('Validation AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # F1-Score
        axes[1, 1].plot(history['val_f1'], label='Val F1-Score', marker='^', color='orange')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1-Score')
        axes[1, 1].set_title('Validation F1-Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'saved_models/plot/training_curves/trial_{trial_number}/training_curves.png', dpi=300)
        plt.close()

        print(f"✓ Curvas de treinamento salvas em: saved_models/plot/training_curves/trial_{trial_number}/training_curves.png")

class BinaryFocalLoss(nn.Module):
    """
    Implementação da Binary Focal Loss com suporte a logits.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: logits (saída bruta da rede antes do sigmoid)
            targets: rótulos binários (0 ou 1)
        """
        # Sigmoid e cálculo da BCE
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Fator focal
        focal_factor = (1 - p_t) ** self.gamma
        loss = self.alpha * focal_factor * bce_loss

        # Redução final
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
