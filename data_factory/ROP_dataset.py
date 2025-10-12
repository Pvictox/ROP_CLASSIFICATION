from torch.utils.data import Dataset, Subset
from PIL import Image
import numpy as np
import cv2
from torchvision.transforms import v2
import torch

class ROPDataset(Dataset):

    train_transformations = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((224, 224)),
        v2.RandomHorizontalFlip(),
        v2.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), ratio=(0.75, 1.33)),
        v2.RandomRotation(degrees=15),
    ])

    #para validação e teste, apenas redimensionamento e normalização
    val_and_test_transformations = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((224, 224)),
    ])

    def __init__(self, dataframe, transform=None, is_train=True, apply_clahe=True):
        self.dataframe = dataframe
        if transform is None:
            self.transform = self.train_transformations if is_train else self.val_and_test_transformations
        else:
            self.transform = transform
            
        self.is_train = is_train
        self.apply_clahe = apply_clahe

    def __len__(self):
        return len(self.dataframe)

    def get_data_from_dataframe(self):
        all_images_paths = self.dataframe['filepath'].tolist()
        all_labels = np.array(self.dataframe['binary_label'].tolist())
        all_patient_ids = np.array(self.dataframe['patient_id'].tolist())
        return all_images_paths, all_labels, all_patient_ids

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['filepath']
        label = self.dataframe.iloc[idx]['binary_label']
        patient_id = self.dataframe.iloc[idx]['patient_id']

        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        if self.apply_clahe:
            image = self.apply_CLAHE_RGB(image)

        return image, label, patient_id
    
    def apply_CLAHE_RGB(self, image_from_PIL, clip_limit=2.0, tile_grid_size=(8, 8)):
        '''
        Aplica o CLAHE (Contrast Limited Adaptive Histogram Equalization) na imagem RGB. 
        O clahe é aplicado no canal L (luminosidade) do espaço de cor LAB.
        '''
        img_array = np.array(image_from_PIL)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        img_LAB = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(img_LAB)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)
        img_lab_clahe = cv2.merge((cl, a, b))
        img_bgr_clahe = cv2.cvtColor(img_lab_clahe, cv2.COLOR_LAB2BGR)
        img_rgb_clahe = cv2.cvtColor(img_bgr_clahe, cv2.COLOR_BGR2RGB)
        img_pil_clahe = Image.fromarray(img_rgb_clahe)
        return img_pil_clahe
        