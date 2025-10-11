import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from data_factory.ROP_dataset import ROPDataset

class Utils():
    @staticmethod
    def plot_sample_images(rop_dataset, num_samples=3):
        '''
        Plota amostras de imagens do ROPDataset fornecido.
        '''
        total_images = len(rop_dataset)
        num_samples = min(num_samples, total_images)
        sample_indices = np.random.choice(total_images, num_samples, replace=False)

        plt.figure(figsize=(10, 5 * num_samples))

        for i, idx in enumerate(sample_indices):
            image, label = rop_dataset[idx]
            # image_np = image.permute(1, 2, 0).numpy()  # Convertendo tensor para numpy array

            plt.subplot(num_samples, 1, i + 1)
            plt.imshow(image)
            plt.title(f'Label: {label}')
            plt.axis('off')

        plt.tight_layout()
        plt.show()
