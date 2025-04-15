#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('..')

from scripts.GenrationGI0 import generate_multiple_images, mixed_dataset
from scripts.autoencoders import InMemoryImageDataset, ConfigurableAutoencoder
from scripts.measuring_quality import selecting_cuadrants

import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
import cv2
from omegaconf import OmegaConf
import warnings
import copy


# In[2]:


try:
    OmegaConf.register_new_resolver("eval", eval)
except ValueError:
    pass


# Elegir el archivo de configuración correspondiente:

# In[3]:


config_name = 'config_base_simetrico_mix_imagenes' # Elegir

config_path = f'configs/{config_name}.yaml'
config = OmegaConf.load(config_path)
config


# In[4]:


if min(config.training.pixeles_cuad) < 8:
    warnings.warn(
        "¡El método de primer orden va a fallar!\n"
        "Existen imágenes con cuadrantes demasiado pequeños, "
        "y este método necesita zonas homogéneas más grandes.",
        category=UserWarning
    )


# Cargo el autoencoder ya entrenado:

# In[5]:


# 1. Crear una instancia del modelo (debe tener la misma arquitectura)
autoencoder_cargado = ConfigurableAutoencoder(config=config)
# 2. Carga los parámetros
autoencoder_cargado.load_state_dict(torch.load(f'data/trained_models/{config_name}.pth'))
# 3. Modo evaluación (cuando lo use para inferencia)
autoencoder_cargado.eval()


# Genero dataset de testeo:

# In[6]:


n = config['testing']['n']
batch_size = config['testing']['batch_size']
n_cuad_lado = config['training']['n_cuad_lado']
pixeles_cuad = config['training']['pixeles_cuad']

test_g, test_gi, test_gI0, alphas = mixed_dataset(
    n_total = n,
    generate_multiple_images = generate_multiple_images,
    conjunto_n_cuad_lado = n_cuad_lado,
    conjunto_pixeles_cuad = pixeles_cuad,
    ratios = config.training.get('ratio',[1]),
    save_alpha_values=True
)


# In[7]:


normalize_to_01 = transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min()))

transform = transforms.Compose([
    transforms.ToTensor(),
    normalize_to_01
])

dataset_test = InMemoryImageDataset(test_gI0, test_gi, transform=transform)
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)


# Genero las imágenes procesadas por el autoencoder, y genero las imágenes de ratio (imagen original / imagen filtrada):

# In[8]:


all_inputs = []
all_targets = []
all_outputs = []
all_ratios = []

with torch.no_grad():
    for entrada, salida in test_loader:
        entrada = entrada.float()
        salida = salida.float()
        outputs = autoencoder_cargado(entrada)
        ratios = entrada / outputs
        
        all_inputs.append(entrada.cpu().numpy())
        all_targets.append(salida.cpu().numpy())
        all_outputs.append(outputs.cpu().numpy())
        all_ratios.append(ratios.cpu().numpy())

inputs = np.squeeze(np.concatenate(all_inputs, axis=0))
targets = np.squeeze(np.concatenate(all_targets, axis=0))
outputs = np.squeeze(np.concatenate(all_outputs, axis=0))
ratios = np.squeeze(np.concatenate(all_ratios, axis=0))


# Grafico un set de imágenes a modo de ejemplo:

# In[9]:


ecualizar_hist = True  # Si se quiere o no ecualizar el histograma de la imagen

###

def graph_random_image_with_ratios(inputs, targets, outputs, ratios, ecualizar_hist, show_plot=True):

    index = int(n*np.random.random()) # Índice del ejemplo puntual que se desea seleccionar
    entrada_red, target_red, salida_red, ratios = inputs[index, :, :], targets[index, :, :], outputs[index, :, :], ratios[index, :, :]

    imagenes = [entrada_red, target_red, salida_red, ratios]
    titulos = ['Entrada', 'Salida esperada', 'Salida de la red', 'Ratio original/filtrada']

    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    for ax, imagen, titulo in zip(axes, imagenes, titulos):
        if ecualizar_hist:
            imagen = ((imagen - imagen.min()) * 255) / (imagen.max() - imagen.min())
            imagen = cv2.equalizeHist(imagen.astype(np.uint8))
            titulo += '\n(ecualizada)'
        
        ax.imshow(imagen, cmap='gray')
        ax.set_title(titulo)

    plt.tight_layout()

graph_random_image_with_ratios(inputs, targets, outputs, ratios, ecualizar_hist)


# ---

# In[10]:


### BORRAR
# AHORA TENGO QUE ELEGIR n (5 o 4) AREAS QUE ENTREN EN LA IMAGEN, Y QUE SEAN HOMOGENEAS (QUE NO SE CRUCEN DE CUADRANTE
# Y QUE TENGAN ALPHA MENOR O IGUAL A -6). LAS ZONAS DE AREA 10X10 O 8X8. YO TOMARIA DE 10X10, SI NO ENTRA EN UN
# CUADRANTE, DE 9X9. Y SI NO ENTRA, DE 8X8. y SI NO ENTRA, QUE FALLE Y DECIR QUE LA IMAGEN ES MUY POCO HOMOGENEA.


# In[11]:


cuadrantes = selecting_cuadrants(alphas, M=4)


# In[12]:


pixeles = config.training.pixeles_cuad
partitions = [len(sublista) for sublista in alphas]
p = [pixel for pixel, count in zip(pixeles, partitions) for _ in range(count)]

cuadrantes_i = copy.deepcopy(cuadrantes)
cuadrantes_f = copy.deepcopy(cuadrantes)

for i in range(len(cuadrantes)):         # Para loopear por las imágenes
    pi = p[i]
    
    for j in range(len(cuadrantes[i])):  # Para loopear por cada una de las zonas que quiero crear en un única imagen
        cuadrante_x = cuadrantes[i][j][0]
        cuadrante_y = cuadrantes[i][j][1]

        fila_i = cuadrante_x*pi     # Fila de inicio del cuadrante
        columna_i = cuadrante_y*pi  # Columna de inicio del cuadrante

        fila_f = fila_i + pi        # Fila de fin del cuadrante
        columna_f = columna_i + pi  # Columna de fin del cuadrantes
        
        cuadrantes_i[i][j] = (fila_i, columna_i)
        cuadrantes_f[i][j] = (fila_f, columna_f)


# In[13]:


del cuadrantes

