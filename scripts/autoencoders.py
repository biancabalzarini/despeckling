from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
import random
from typing import List, Tuple, Callable, Optional

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
    partitioned_gi0_image: Callable,
    n_cuad_lado: int,
    pixeles_cuad: int,
    alpha_values: Optional[List[float]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Genera múltiples conjuntos de imágenes utilizando la función partitioned_gi0_image.

    Parameters:
    -----------
    n: int
        Número de conjuntos de imágenes a generar.
    partitioned_gi0_image: function
        Función que genera un conjunto de imágenes (g, gi, gI0).
    n_cuad_lado: int
        Número de cuadrados por lado de cada imagen (ver documentación de partitioned_gi0_image).
    pixeles_cuad: int
        Número de píxeles por lado de cada cuadrado que forma cada imagen (ver documentación de
        partitioned_gi0_image).
    alpha_values: Optional[List[float]], opcional
        Lista de valores posibles para alpha. Si no se proporciona, se usarán los valores por defecto.

    Returns:
    --------
    conjunto_g : np.ndarray
        Array de forma (n, n_cuad_lado*pixeles_cuad, n_cuad_lado*pixeles_cuad) con n repeticiones de imagen_g.
    conjunto_gi : np.ndarray
        Array de forma (n, n_cuad_lado*pixeles_cuad, n_cuad_lado*pixeles_cuad) con n repeticiones de imagen_gi.
    conjunto_gI0 : np.ndarray
        Array de forma (n, n_cuad_lado*pixeles_cuad, n_cuad_lado*pixeles_cuad) con n repeticiones de imagen_gI0.
    """
    tam_imagen = n_cuad_lado * pixeles_cuad
    conjunto_g = np.zeros((n, tam_imagen, tam_imagen))
    conjunto_gi = np.zeros((n, tam_imagen, tam_imagen))
    conjunto_gI0 = np.zeros((n, tam_imagen, tam_imagen))
    
    total_cuadrados = n_cuad_lado ** 2
    p_gammas: List[float] = [1.0] * total_cuadrados
    p_looks: List[int] = [1] * total_cuadrados
    
    if alpha_values is None:
        alpha_values = [-1.5, -2, -3, -5, -6, -8, -10, -20]
    
    for i in range(n):
        p_alphas: List[float] = random.choices(alpha_values, k=total_cuadrados)
        
        imagen_g, imagen_gi, imagen_gI0 = partitioned_gi0_image(p_alphas, p_gammas, p_looks, n_cuad_lado, pixeles_cuad)
        
        conjunto_g[i] = imagen_g
        conjunto_gi[i] = imagen_gi
        conjunto_gI0[i] = imagen_gI0
    
    return conjunto_g, conjunto_gi, conjunto_gI0

class Autoencoder(nn.Module): # La clase Autoencoder hereda de la clase nn.Module, que es una clase base para todos los modelos en PyTorch.
                              # Esto permite que nuestra clase Autoencoder tenga todas las funcionalidades necesarias para ser un modelo de aprendizaje profundo en PyTorch.

    # Dentro del método __init__, definimos las capas del autoencoder.
    def __init__(self, encoding_dim):

        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential( # Encoder
                                      # Toma una imagen de entrada y la comprime en una representación de dimensionalidad más baja llamada encoding_dim

            # Secuencia de capas del codificador:
            nn.Linear(100 * 100, 128), # Capa lineal inicial que toma una imagen de 28x28 píxeles (784 dimensiones después de aplanarla) y la reduce a 128 dimensiones utilizando una función lineal.
            nn.ReLU(), # Luego se aplica una función de activación ReLU para introducir no linealidad en la representación.
            nn.Linear(128, encoding_dim), # Finalmente, otra capa lineal reduce la dimensionalidad a encoding_dim.
        )

        self.decoder = nn.Sequential( # Decoder
                                      # Devuelve la imágen a su tamaño original.

            # Secuencia de capas del decodificador:
            nn.Linear(encoding_dim, 128), # Capa lineal que toma la representación de encoding_dim y la expande a 128 dimensiones.
            nn.ReLU(), # Luego, se aplica una función de activación ReLU.
            nn.Linear(128, 100 * 100), # A continuación, otra capa lineal expande la dimensionalidad a 28x28 píxeles (784 dimensiones).
            nn.Sigmoid(), # Finalmente se aplica una función de activación sigmoide para limitar los valores de salida entre 0 y 1.
        )

    def forward(self, x): # El método forward define cómo se propagan los datos a través del autoencoder.

        encoded = self.encoder(x) # Toma una imagen de entrada x, la pasa por el codificador para obtener la representación comprimida encoded.
        decoded = self.decoder(encoded) # Luego pasa esta representación por el decodificador para obtener la reconstrucción decoded.
        return decoded # La reconstrucción se devuelve como salida.
