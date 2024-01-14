#!/usr/bin/env python
# coding: utf-8

# ## Jugando con autoenconders
# Esta notebook tiene el único objetivo de hacer primeras pruebas con autoencoders.

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt


# In[ ]:


# Definir la arquitectura del autoencoder
class Autoencoder(nn.Module): # La clase Autoencoder hereda de la clase nn.Module, que es una clase base para todos los modelos en PyTorch.
                              # Esto permite que nuestra clase Autoencoder tenga todas las funcionalidades necesarias para ser un modelo de aprendizaje profundo en PyTorch.

    # Dentro del método __init__, definimos las capas del autoencoder.
    def __init__(self, encoding_dim):

        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential( # Encoder
                                      # Toma una imagen de entrada y la comprime en una representación de dimensionalidad más baja llamada encoding_dim

            # Secuencia de capas del codificador:
            nn.Linear(28 * 28, 128), # Capa lineal inicial que toma una imagen de 28x28 píxeles (784 dimensiones después de aplanarla) y la reduce a 128 dimensiones utilizando una función lineal.
            nn.ReLU(), # Luego se aplica una función de activación ReLU para introducir no linealidad en la representación.
            nn.Linear(128, encoding_dim), # Finalmente, otra capa lineal reduce la dimensionalidad a encoding_dim.
        )

        self.decoder = nn.Sequential( # Decoder
                                      # Devuelve la imágen a su tamaño original.

            # Secuencia de capas del decodificador:
            nn.Linear(encoding_dim, 128), # Capa lineal que toma la representación de encoding_dim y la expande a 128 dimensiones.
            nn.ReLU(), # Luego, se aplica una función de activación ReLU.
            nn.Linear(128, 28 * 28), # A continuación, otra capa lineal expande la dimensionalidad a 28x28 píxeles (784 dimensiones).
            nn.Sigmoid(), # Finalmente se aplica una función de activación sigmoide para limitar los valores de salida entre 0 y 1.
        )

    def forward(self, x): # El método forward define cómo se propagan los datos a través del autoencoder.

        encoded = self.encoder(x) # Toma una imagen de entrada x, la pasa por el codificador para obtener la representación comprimida encoded.
        decoded = self.decoder(encoded) # Luego pasa esta representación por el decodificador para obtener la reconstrucción decoded.
        return decoded # La reconstrucción se devuelve como salida.

