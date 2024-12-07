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
        self.is_conv = any(
            layer.get('type', 'dense') == 'conv2d' for layer in self.config['encoder']['layers']
        )
        
        if not self.is_conv:
            first_layer_size = self.config['encoder']['layers'][0]['dim']
            assert self.flat_size > first_layer_size, \
                f"El tamaño flat de la imagen ({self.flat_size}) debe ser mayor que el tamaño de la primera capa del encoder ({first_layer_size})"

        self.encoder = self._build('encoder')
        self.decoder = self._build('decoder')
        
    def _get_last_output_dim(self, layers):
        """Obtiene la dimensión de salida de la última capa que tiene dimensiones (Lineal o Conv2d)"""
        for layer in reversed(layers):
            if isinstance(layer, nn.Linear):
                return layer.out_features
            elif isinstance(layer, nn.Conv2d):
                return layer.out_channels
            elif isinstance(layer, nn.ConvTranspose2d):
                return layer.out_channels
        return None
        
    def _build(self, component: str) -> nn.Sequential:
        if component not in ['encoder', 'decoder']:
            raise ValueError(f"El parámetro component solo puede ser 'encoder' o 'decoder', se recibió: {component}")
        
        component_layers = self.config[component]['layers']
        
        layers = []
        for layer in component_layers:
            # Si no se especifica tipo, asumimos que es una capa densa
            layer_type = layer.get('type', 'dense').lower()
            
            if layer_type == 'dense':
                # Determinar input_dim para capas densas
                if len(layers) == 0:
                    if component == 'encoder':
                        input_dim = self.flat_size
                    else:  # decoder
                        input_dim = self.encoding_dim
                else:
                    last_dim = self._get_last_output_dim(layers)
                    if last_dim is None:
                        raise ValueError("No se pudo determinar la dimensión de salida de la capa anterior")
                    input_dim = last_dim
                layers.append(nn.Linear(input_dim, layer['dim']))
            
            elif layer_type == 'conv2d':
                in_channels = layer.get('in_channels', 
                    1 if len(layers) == 0 else self._get_last_output_dim(layers))
                layers.append(nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=layer['filters'],
                    kernel_size=layer['kernel_size'],
                    stride=layer.get('stride', 1),
                    padding=layer.get('padding', 0)
                ))
                
            elif layer_type == 'conv2d_transpose':
                in_channels = layer.get('in_channels', self._get_last_output_dim(layers))
                layers.append(nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=layer['filters'],
                    kernel_size=layer['kernel_size'],
                    stride=layer.get('stride', 1),
                    padding=layer.get('padding', 0)
                ))
            
            elif layer_type == 'maxpool2d':
                layers.append(nn.MaxPool2d(
                    kernel_size=layer['pool_size'],
                    stride=layer.get('stride', None),
                    padding=layer.get('padding', 0)
                ))
            
            elif layer_type == 'flatten':
                layers.append(nn.Flatten())
                
            elif layer_type == 'reshape':
                shape = layer['shape']
                layers.append(lambda x: x.view(x.size(0), *shape))
                
            elif layer_type == 'batchnorm':
                if isinstance(layers[-1], nn.Conv2d) or isinstance(layers[-1], nn.ConvTranspose2d):
                    layers.append(nn.BatchNorm2d(layers[-1].out_channels))
                else:
                    layers.append(nn.BatchNorm1d(layers[-1].out_features))
            
            activation = layer.get('activation', '').lower()
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'elu':
                layers.append(nn.ELU())
            
            if 'dropout' in layer:
                layers.append(nn.Dropout(layer['dropout']))
                
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Si es convolucional y la entrada está aplanada, la reshapeamos
        if self.is_conv and x.dim() == 2:
            x = x.view(x.size(0), 1, self.image_size, self.image_size)
        # Si es lineal y la entrada está en 2D, la aplanamos
        elif not self.is_conv and x.dim() == 4:
            x = x.view(x.size(0), -1)
        
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        # Asegurarnos de que la salida tenga la forma correcta
        if self.is_conv and decoded.dim() == 2:
            decoded = decoded.view(decoded.size(0), 1, self.image_size, self.image_size)
        elif not self.is_conv and decoded.dim() == 4:
            decoded = decoded.view(decoded.size(0), -1)
        
        return decoded