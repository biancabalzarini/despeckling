from torch.utils.data import Dataset
import torch.nn as nn
import torch
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

class ConfigurableAutoencoder(nn.Module): # La clase Autoencoder hereda de la clase nn.Module, que es una clase base para todos los modelos en PyTorch.
                                          # Esto permite que nuestra clase Autoencoder tenga todas las funcionalidades necesarias para ser un modelo de aprendizaje profundo en PyTorch.
    
    def __init__(self, config: dict):
        super(ConfigurableAutoencoder, self).__init__()
        
        self.config = config
        self.image_size = self.config['training']['n_cuad_lado'] * self.config['training']['pixeles_cuad']
        self.flat_size = self.image_size * self.image_size
        self.encoding_dim = self.config['model']['encoding_dim']

        # Agregamos una capa Flatten para convertir imágenes en vectores
        self.flatten = nn.Flatten()
        self.unflatten = nn.Unflatten(1, (1, self.image_size, self.image_size))
        
        self.encoder = self._build('encoder')
        self.decoder = self._build('decoder')
        
    def _build(self, component: str) -> nn.Sequential:
        layers = []
        if component == 'encoder':
            input_dim = self.flat_size
            component_layers = self.config['encoder']['layers']
        elif component == 'decoder':
            input_dim = self.encoding_dim
            component_layers = self.config['decoder']['layers']
        else:
            raise ValueError(f"El parámetro component solo puede ser 'encoder' o 'decoder', se recibió: {component}")
        
        for layer in component_layers:
            layers.append(nn.Linear(input_dim, layer['dim']))
            if layer['activation'].lower() == 'relu':
                layers.append(nn.ReLU())
            elif layer['activation'].lower() == 'sigmoid':
                layers.append(nn.Sigmoid())
            input_dim = layer['dim']
            
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten solo en la primera pasada
        x = self.flatten(x)  
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        # Reconstruir la imagen a su shape original
        decoded = self.unflatten(decoded)
        return decoded