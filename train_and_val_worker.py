from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from torch.utils.data import Subset, DataLoader

from data_factory.ROP_SUBSET_dataset import ROPSubset
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torchvision import transforms

class TrainAndEvalWorker:
    def __init__(self, config:dict):
        if not config:
            print("Configuração vazia fornecida. Utilizando valores padrão.")
            self.config = {
                'learning_rate': 1e-4,
                'weight_decay': 1e-5,
                'batch_size': 16,
                'num_epochs': 30,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            }
        else:
            self.config = config
        
        self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=1).to(self.config['device'])
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 1e-4),
            weight_decay=self.config.get('weight_decay', 1e-5)
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5)

    def train_epoch(self, train_loader, val_loader):
        self.model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.config['device']), targets.float().to(self.config['device'])
            
            self.optimizer.zero_grad()
            outputs = self.model(data).squeeze(-1)  # Remove apenas a última dimensão
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
        
    
        val_accuracy, val_loss = self.validate(val_loader)
        
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        
        self.scheduler.step(val_loss)
        
        return train_accuracy, val_accuracy, avg_train_loss, val_loss    

    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.config['device']), targets.float().to(self.config['device'])
                outputs = self.model(data).squeeze(-1)  # Remove apenas a última dimensão
                loss = self.criterion(outputs, targets)
                
                predicted = torch.sigmoid(outputs) > 0.5
                val_total += targets.size(0)
                val_correct += (predicted == targets.bool()).sum().item()
                val_loss += loss.item()
        
        val_accuracy = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        return val_accuracy, avg_val_loss

    import torch


    def custom_collate_fn(self, batch):
        # Se as imagens ainda são PIL, converta para tensor
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



    def train(self, X_train, y_train, patient_ids_train, train_index, gkf:GroupKFold, rop_dataset):
        fold_results = []
        best_global_acc = 0.0
        best_model_state = None
        best_fold = 0
        for fold, (train_fold_idx, val_fold_idx) in enumerate(gkf.split(X_train, y_train, groups=patient_ids_train)):
            print(f"{'='*20}")
            print(f"Fold {fold+1}/{gkf.n_splits}")
            print(f"{'='*20}")

            train_fold_absolute_index = train_index[train_fold_idx]
            val_fold_absolute_index = train_index[val_fold_idx]

            train_fold_dataset = Subset(rop_dataset, train_fold_absolute_index)
            val_fold_dataset = Subset(rop_dataset, val_fold_absolute_index)

            train_fold_dataset = ROPSubset(train_fold_dataset, transform=rop_dataset.train_transformations, apply_clahe=True)
            val_fold_dataset = ROPSubset(val_fold_dataset, transform=rop_dataset.val_and_test_transformations, apply_clahe=True)

            train_loader = DataLoader(
                train_fold_dataset, 
                batch_size=self.config.get('batch_size', 32),
                shuffle=True,
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
            self.model = timm.create_model(
                'efficientnet_b0',
                pretrained=True,
                num_classes=1
            ).to(self.config['device'])
            
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.get('learning_rate', 1e-4),
                weight_decay=self.config.get('weight_decay', 1e-5)
            )

            best_val_acc = 0.0  
            epochs = self.config.get('num_epochs', 50)
            
            for epoch in range(epochs):
                print(f'\nEpoch {epoch+1}/{epochs}')
                train_acc, val_acc, train_loss, val_loss = self.train_epoch(train_loader, val_loader)
                
                if val_acc > best_global_acc:
                    best_global_acc = val_acc
                    best_model_state = self.model.state_dict().copy()
                    best_fold = fold + 1
            
                fold_results.append({
                    'fold': fold+1,
                    'best_val_accuracy': best_val_acc
                })
                
                print(f'Fold {fold+1} completed. Best Val Accuracy: {best_val_acc:.2f}%')
        
        if best_model_state is not None:
            torch.save(best_model_state, '/home/pedro_fonseca/PATIENT_ROP/saved_models/best_model_efficientNET.pth')
            print(f'\nBest model saved! Fold {best_fold}, Accuracy: {best_global_acc:.2f}%')
        avg_accuracy = sum([r['best_val_accuracy'] for r in fold_results]) / len(fold_results)
        print(f'\nCross-Validation Results:')
        print(f'Average Accuracy: {avg_accuracy:.2f}%')
        
        return fold_results

    def evaluate(self):
        pass

