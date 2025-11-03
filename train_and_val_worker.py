import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
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
import json, shutil

class TrainAndEvalWorker:
    def __init__(self, config:dict, model=None):
        if not config:
            print("Configuração vazia fornecida. Utilizando valores padrão.")
            self.config = {
                'learning_rate': 1e-3,
                'weight_decay': 1e-4,
                'batch_size': 32,
                'num_epochs': 15,
                'device': 'cuda:1' if torch.cuda.is_available() else 'cpu',
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

        val_loss, val_auc, val_threshold = self.validate(val_loader)

        print(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}')

        self.scheduler.step(val_loss)
        
        return train_accuracy, avg_train_loss, val_loss, val_auc, val_threshold



    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        all_targets = []
        all_outputs = []

        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.config['device']), targets.float().to(self.config['device'])
                outputs = self.model(data).squeeze(-1)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()

                all_targets.extend(targets.cpu().numpy())
                all_outputs.extend(torch.sigmoid(outputs).cpu().numpy())

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

        avg_val_loss = val_loss / len(val_loader)

        return avg_val_loss, auc, best_threshold



    def setup_finetuning(self):
        for param in self.model.parameters():
            param.requires_grad = False
        
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        

    def custom_collate_fn(self, batch):
        transform = transforms.ToTensor()
        
        images = []
        labels = []
        
        for item in batch:
            if len(item) == 3:  # (image, label, patient_id)
                image, label, patient_id = item
                if hasattr(image, 'mode'):  # É uma PIL Image
                    image = transform(image)
                images.append(image)
                labels.append(label)
        
        return torch.stack(images), torch.tensor(labels)



    def train(self, X_train, y_train, patient_ids_train, train_index, gkf:GroupKFold, rop_dataset, trial, dynamic_config=None):
        #self.model = model.to(self.config['device'])

        # if self.model:
        #     self.setup_finetuning()
        fold_results = []
        # usar AUC como métrica principal global
        best_global_auc = 0.0
        best_model_state = None
        best_fold = 0
        # coletor de thresholds ótimos (um por fold)
        best_thresholds_per_fold = []
        for fold, (train_fold_idx, val_fold_idx) in enumerate(gkf.split(X_train, y_train, groups=patient_ids_train)):
            print(f"{'='*20}")
            print(f"Fold {fold+1}/{gkf.n_splits}") #type:ignore
            print(f"{'='*20}")

            train_fold_absolute_index = train_index[train_fold_idx]
            val_fold_absolute_index = train_index[val_fold_idx]

            train_fold_dataset = Subset(rop_dataset, train_fold_absolute_index)
            val_fold_dataset = Subset(rop_dataset, val_fold_absolute_index)

            train_fold_dataset = ROPSubset(train_fold_dataset, transform=rop_dataset.train_transformations, apply_clahe=True)
            val_fold_dataset = ROPSubset(val_fold_dataset, transform=rop_dataset.val_and_test_transformations, apply_clahe=True)

            # labels = [sample[1] for sample in train_fold_dataset]  # supondo (img, label, id)
            # class_sample_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])
            # weight = 1. / class_sample_count
            # samples_weight = np.array([weight[int(t)] for t in labels])
            # samples_weight = torch.from_numpy(samples_weight).double()
            # print(f"Samples weight distribution: {samples_weight}")

            # sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
            train_loader = DataLoader(
                train_fold_dataset, 
                batch_size=self.config.get('batch_size', 32),
                shuffle=True,
                # sampler=sampler,  # <- substitui shuffle=True
                num_workers=4,
                collate_fn=self.custom_collate_fn
            )
            val_loader = DataLoader(
                val_fold_dataset,
                batch_size=self.config.get('batch_size', 32),
                shuffle=False,
                num_workers=4,
                collate_fn=self.custom_collate_fn
            )

            #PARA CADA FOLD TEM QUE REINICIAR O MODELO E OTIMIZADOR
            self.model = DynamicEfficientNet(dynamic_config).to(self.config['device'])
            
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.get('learning_rate', 1e-4),
                weight_decay=self.config.get('weight_decay', 1e-5)
            )

            # track best threshold for this fold (associated to best val AUC in the fold)
            best_val_auc = 0.0
            best_threshold_fold = 0.5
            epochs = self.config.get('num_epochs', 20)
            best_fold_model_state = None
            # guardar o loss
            train_losses = []
            val_losses = []

            for epoch in range(epochs):
                print(f'\nEpoch {epoch+1}/{epochs}')
                train_acc, train_loss, val_loss, val_auc, val_threshold = self.train_epoch(train_loader, val_loader)

                # guardar o loss
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                #  guarda o melhor modelo deste fold (entre épocas)
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_fold_model_state = self.model.state_dict().copy()
                    best_threshold_fold = val_threshold
                    best_fold = fold + 1
                if best_val_auc > best_global_auc:
                    best_global_auc = best_val_auc
        
                # (não registramos o fold_results aqui — registramos somente ao final do fold)
                print(f'Epoch finished. Current fold best Val AUC: {best_val_auc:.4f}, Global best AUC: {best_global_auc:.4f}')
            
            trial.report(best_val_auc, fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            # ao final do fold, armazenar o threshold ótimo encontrado para este fold
            best_thresholds_per_fold.append(best_threshold_fold)
            print(f"Fold {fold+1} best threshold: {best_threshold_fold:.4f} (best val AUC in fold: {best_val_auc:.4f})")
           
            
            # registrar resultado resumido do fold
            fold_results.append({
                'fold': fold+1,
                'best_val_auc': float(best_val_auc),
                'best_threshold': float(best_threshold_fold)
            })

            # Salvar modelo do fold em /temp
            fold_model_path = f"saved_models/temp/fold_{fold+1}.pth"
            os.makedirs(os.path.dirname(fold_model_path), exist_ok=True)
            # Salvar state_dict + configuração
            torch.save({
                'state_dict': best_fold_model_state,  # pesos do fold
                'config': dynamic_config              # arquitetura usada neste fold
            }, fold_model_path)
            print(f"Modelo do fold {fold+1} salvo em {fold_model_path}")

            # salvar curva de loss por fold
            os.makedirs(f"saved_models/temp/plot/trial{trial.number}", exist_ok=True)
            plt.figure()
            plt.plot(train_losses, label="Train Loss")
            plt.plot(val_losses, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"Fold {fold+1} - Loss Curves")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"saved_models/temp/plot/trial{trial.number}/fold_{fold+1}_loss_curve.png")
            plt.close()


        # =========================================
        # Fim do trial → calcular média e retornar
        # =========================================
        
        print(f"Trial {trial.number}: Achieved best AUC: {best_global_auc:.4f} at fold {best_fold}")
        # calcula média dos thresholds ótimos por fold e guarda em self (usado depois em evaluate)
        if len(best_thresholds_per_fold) > 0:
            self.avg_threshold = float(np.mean(best_thresholds_per_fold))
            print(f"\nAverage threshold across folds: {self.avg_threshold:.4f}")
            # opcional: salvar em arquivo (texto simples)
            try:
                with open('saved_models/avg_threshold.txt', 'w') as f:
                    f.write(str(self.avg_threshold))
            except Exception:
                pass

        # if best_model_state is not None:
        #     torch.save(best_model_state, f'saved_models/trial{trial.number}_best_model_efficientNET.pth')
        #     print(f'\nBest model saved! Fold {best_fold}, Best AUC: {best_global_auc:.4f}')
        avg_auc = sum([r['best_val_auc'] for r in fold_results]) / len(fold_results)
        print(f'\nCross-Validation Results:')
        print(f'Average AUC: {avg_auc:.4f}')

        # =========================================
        # Checar se o modelo deste trial é o melhor geral
        # =========================================

        temp_dir = f"saved_models/temp/"
        best_trial_dir = "saved_models/best_trial"
        best_trial_info_path = os.path.join(best_trial_dir, "best_trial_info.json")
        os.makedirs(best_trial_dir, exist_ok=True)

        # lê JSON anterior
        previous_best_auc = 0.0
        if os.path.exists(best_trial_info_path):
            with open(best_trial_info_path, "r") as f:
                try:
                    data = json.load(f)
                    previous_best_auc = data.get("best_avg_auc", 0.0)
                except json.JSONDecodeError:
                    print("JSON corrompido. Reiniciando.")

        if avg_auc > previous_best_auc:
            print(f"Novo melhor trial! (AUC {avg_auc:.4f} > {previous_best_auc:.4f})")
            
            # remove anterior e copia modelos
            if os.path.exists(best_trial_dir):
                shutil.rmtree(best_trial_dir)
            os.makedirs(best_trial_dir, exist_ok=True)

            # copia os folds do trial atual
            for fold in range(len(fold_results)):
                src = os.path.join(temp_dir, f"fold_{fold+1}.pth")
                dst = os.path.join(best_trial_dir, f"fold_{fold+1}.pth")
                shutil.copy2(src, dst)
            # copiar figuras do trial vencedor para best_trial
            best_plot_dir = os.path.join(best_trial_dir, "plot")
            os.makedirs(best_plot_dir, exist_ok=True)

            temp_plot_dir = f"saved_models/temp/plot/trial{trial.number}"
            if os.path.exists(temp_plot_dir):
                for file_name in os.listdir(temp_plot_dir):
                    src = os.path.join(temp_plot_dir, file_name)
                    dst = os.path.join(best_plot_dir, file_name)
                    shutil.copy2(src, dst)

                
            # salva info do melhor trial
            best_info = {
                "trial_number": int(trial.number),
                "best_avg_auc": float(avg_auc),
                "thresholds": [float(t) for t in best_thresholds_per_fold]
            }
            with open(best_trial_info_path, "w") as f:
                json.dump(best_info, f, indent=4)

        # limpa pasta temporária do trial
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        

        return fold_results, avg_auc

    def evaluate(self, test_dataset, model_path=None):
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.config['device']))
            print(f"Modelo carregado de: {model_path}")
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=False,
            num_workers=4,
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
        os.makedirs("saved_models/plot", exist_ok=True)
        disp = ConfusionMatrixDisplay(conf_matrix)
        disp.plot(cmap='Blues')
        plt.title("Confusion Matrix - Test Set")
        plt.savefig("saved_models/plot/test_confusion_matrix.png")
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
        pd.DataFrame(results, index=[0]).to_csv('test_results.csv')
        
        return results

    def evaluate_ensemble(self, test_dataset):
        best_trial_dir = "saved_models/best_trial"
        ensemble_dir = "saved_models/ensemble"
        os.makedirs(ensemble_dir, exist_ok=True)

        # lista os modelos salvos por fold
        model_files = [f for f in os.listdir(best_trial_dir) if f.startswith("fold_") and f.endswith(".pth")]
        model_files = sorted(model_files)

        # DataLoader do teste
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=False,
            num_workers=4,
            collate_fn=self.custom_collate_fn
        )

        all_fold_metrics = []
        all_fold_conf_matrices = []

        for fold_file in model_files:
            model_path = os.path.join(best_trial_dir, fold_file)
            # self.model.load_state_dict(torch.load(model_path, map_location=self.config['device']))
            checkpoint = torch.load(model_path, map_location=self.config['device'])
            # carregar configuração dinâmica
            fold_model = DynamicEfficientNet(checkpoint['config']).to(self.config['device'])
            fold_model.load_state_dict(checkpoint['state_dict'])
            fold_model.eval()

            all_predictions = []
            all_targets = []
            all_probs = []

            with torch.no_grad():
                for data, targets in test_loader:
                    data, targets = data.to(self.config['device']), targets.float().to(self.config['device'])
                    outputs = fold_model(data).squeeze(-1)
                    probs = torch.sigmoid(outputs)
                    predicted = (probs > 0.5).float()

                    all_predictions.extend(predicted.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())

            all_predictions = np.array(all_predictions).astype(int)
            all_targets = np.array(all_targets).astype(int)
            all_probs = np.array(all_probs)

            # métricas por fold
            f1 = f1_score(all_targets, all_predictions, zero_division=0)
            precision = precision_score(all_targets, all_predictions, zero_division=0)
            recall = recall_score(all_targets, all_predictions, zero_division=0)
            try:
                auc = roc_auc_score(all_targets, all_probs, average='weighted')
            except:
                auc = None
            conf_matrix = confusion_matrix(all_targets, all_predictions)

            all_fold_metrics.append({
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'auc': auc if auc is not None else 0.0
            })
            all_fold_conf_matrices.append(conf_matrix)

            # salva matriz de confusão de cada fold
            cm_path = os.path.join(ensemble_dir, f"{fold_file}_confusion_matrix.npy")
            np.save(cm_path, conf_matrix)

        # calcular média e std das métricas
        metrics_np = {k: np.array([m[k] for m in all_fold_metrics]) for k in all_fold_metrics[0].keys()}
        ensemble_stats = {k: {"mean": float(v.mean()), "std": float(v.std())} for k, v in metrics_np.items()}

        # salvar json com estatísticas do ensemble
        with open(os.path.join(ensemble_dir, "ensemble_metrics.json"), "w") as f:
            json.dump(ensemble_stats, f, indent=4)

        print("Ensemble evaluation completed!")
        print(json.dumps(ensemble_stats, indent=4))

        return ensemble_stats, all_fold_conf_matrices

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
