from torch.utils.data import Dataset
import numpy as np
import random
from typing import List, Tuple

class InMemoryImageDataset(Dataset):
    def __init__(self, input_images, target_images, transform=None):
        assert len(input_images) == len(target_images), "Input and target image sets must have the same length"
        self.input_images = input_images
        self.target_images = target_images
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_image = self.input_images[idx]
        target_image = self.target_images[idx]
        
        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)
        
        return input_image, target_image

def generate_multiple_images(
    n: int,
    partitioned_gi0_image
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Genera múltiples conjuntos de imágenes utilizando la función partitioned_gi0_image.

    Parameters:
    -----------
    n: int
        Número de conjuntos de imágenes a generar.
    partitioned_gi0_image: function
        Función que genera un conjunto de imágenes (g, gi, gI0).

    Returns:
    --------
    conjunto_g : np.ndarray
        Array de forma (n, 100, 100) con n repeticiones de imagen_g.
    conjunto_gi : np.ndarray
        Array de forma (n, 100, 100) con n repeticiones de imagen_gi.
    conjunto_gI0 : np.ndarray
        Array de forma (n, 100, 100) con n repeticiones de imagen_gI0.
    """
    conjunto_g = np.zeros((n, 100, 100))
    conjunto_gi = np.zeros((n, 100, 100))
    conjunto_gI0 = np.zeros((n, 100, 100))
    
    p_gammas: List[float] = [1.0, 1.0, 1.0, 1.0]
    p_looks: List[int] = [1, 1, 1, 1]
    
    alpha_values = [-1.5, -2, -3, -5, -6, -8, -10, -20]
    
    for i in range(n):
        p_alphas: List[float] = random.choices(alpha_values, k=4)
        
        imagen_g, imagen_gi, imagen_gI0 = partitioned_gi0_image(p_alphas, p_gammas, p_looks)
        
        conjunto_g[i] = imagen_g
        conjunto_gi[i] = imagen_gi
        conjunto_gI0[i] = imagen_gI0
    
    return conjunto_g, conjunto_gi, conjunto_gI0
