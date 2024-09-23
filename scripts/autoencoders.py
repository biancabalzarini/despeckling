from torch.utils.data import Dataset
import numpy as np
import random
from typing import List, Tuple

class InMemoryImageDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image

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
