#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
sys.path.append('..')

from scripts.GenrationGI0 import rGI0, partitioned_gi0_image
from scripts.autoencoders import InMemoryImageDataset, generate_multiple_images, Autoencoder

import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


# ---
# # Creación del dataset para entrenar

# In[ ]:


## PARÁMETROS

# Cantidad de imágenes a generar
n = 50000
# Tamaños de los batches
batch_size = 50


# In[ ]:


train_g, train_gi, train_gI0 = generate_multiple_images(n, partitioned_gi0_image)


# In[ ]:


normalize_to_01 = transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min()))

transform = transforms.Compose([
    transforms.ToTensor(),
    normalize_to_01
])

dataset_train = InMemoryImageDataset(train_gI0, train_gi, transform=transform)
train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)


# ---
# # Entrenamiento

# In[ ]:


# PARÁMETROS

# Dimensión de la capa más interna del autoencoder
encoding_dim = 32
# Learning rate
learning_rate = 1e-3
# Cantidad de épocas
num_epochs = 100

