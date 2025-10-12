from cProfile import label
from torch.utils.data import Dataset, Subset
from PIL import Image
import numpy as np
import cv2
from torchvision.transforms import v2
import torch

class ROPSubset(Dataset):


    def __init__(self, subset, transform=None, apply_clahe=True):
        self.subset = subset
        self.transform = transform
        self.apply_clahe = apply_clahe

    def __len__(self):
        return len(self.subset)


    def __getitem__(self, idx):
        image, label, patient_id = self.subset[idx]

        # image = Image.open(image).convert('RGB')

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
        