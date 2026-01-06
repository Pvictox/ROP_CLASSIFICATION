from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
import torch
import os

class RetinaDataset(Dataset):
    def __init__(self, dataframe, root_dir='', transform=None, domain_label=0, return_filename=False):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame contendo os caminhos e labels.
            root_dir (string): Diretório raiz das imagens.
            transform (callable, optional): Transformações OBRIGATÓRIAS que terminem em ToTensor.
            domain_label (int): 0 para ORIGA (Source), 1 para Retinopatia (Target).
            return_filename (bool): Retorna o nome do arquivo (útil para debug).
        """
        # CORREÇÃO 1: Padronizar o nome da variável (self.dataframe)
        self.dataframe = dataframe 
        self.root_dir = root_dir
        
        # CORREÇÃO 2: Garantir que transform não seja None ou avisar
        self.transform = transform
        
        self.domain_label = domain_label
        self.return_filename = return_filename

        # Ajuste conforme suas colunas (Verifique se 1 e 2 estão corretos para seu CSV)
        self.img_col_idx = 1
        self.label_col_idx = 2

    def __len__(self):
        # CORREÇÃO 1: Agora o nome bate com o __init__
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 1. Montar o caminho da imagem
        img_name = self.dataframe.iloc[idx, self.img_col_idx]
        
        if not str(img_name).lower().endswith(('.png', '.jpg', '.jpeg')):
            img_name = str(img_name) + ".jpg"

        if not os.path.isabs(str(img_name)) and self.root_dir:
            img_path = os.path.join(self.root_dir, str(img_name))
        else:
            img_path = str(img_name)

        # 2. Carregar Imagem
        try:
            image = Image.open(img_path).convert('RGB')
        except (IOError, FileNotFoundError):
            print(f"Erro ao abrir imagem: {img_path}")
            image = Image.new('RGB', (224, 224)) # Use o tamanho padrão da sua rede

        # 3. Label da Doença
        try:
            class_label = int(self.dataframe.iloc[idx, self.label_col_idx])
        except:
            class_label = -1 

        # 4. Aplica CLAHE (Processamento estático)
        # O CLAHE devolve uma PIL Image
        #image = self.apply_CLAHE_RGB(image)
        

        # 5. Aplicar Transformações (Data Augmentation + ToTensor)
        # CRUCIAL: O transform deve conter transforms.ToTensor()
        if self.transform:
            image = self.transform(image)
        else:
            # Fallback de segurança: Se não tiver transform, converte pra tensor
            # para não quebrar o DataLoader
            import torchvision.transforms as T
            image = T.ToTensor()(image)

        # 6. Domain Label
        domain_label_tensor = torch.tensor(self.domain_label, dtype=torch.float32)

        if self.return_filename:
            return image, class_label, domain_label_tensor, img_name
        
        return image, class_label, domain_label_tensor
    
    def apply_CLAHE_RGB(self, image_from_PIL, clip_limit=2.0, tile_grid_size=(8, 8)):
        img_array = np.array(image_from_PIL)
        
        # OTIMIZAÇÃO: Converter direto de RGB para LAB (pula o passo BGR desnecessário)
        img_LAB = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        
        l, a, b = cv2.split(img_LAB)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)
        
        img_lab_clahe = cv2.merge((cl, a, b))
        
        # Converter direto de LAB para RGB
        img_rgb_clahe = cv2.cvtColor(img_lab_clahe, cv2.COLOR_LAB2RGB)
        
        return Image.fromarray(img_rgb_clahe)