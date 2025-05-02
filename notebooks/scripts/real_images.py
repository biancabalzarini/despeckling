#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('..')

import pyreadr # Para leer datos de tipo RData
import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
from matplotlib import colors

from omegaconf import OmegaConf
from scripts.autoencoders import ConfigurableAutoencoder
from scripts.GenrationGI0 import partitioned_gi0_image
import torch
import random


# In[2]:


try:
    OmegaConf.register_new_resolver("eval", eval)
except ValueError:
    pass


# Cargo imágenes SAR reales para visualizarlas

# In[3]:


def Ecualizauint16(img):
    hist, bins = np.histogram(img.flatten(),65536,[0,65536])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)  
    cdf_m = (cdf_m - cdf_m.min())*65535/(cdf_m.max()-cdf_m.min())
    cdf_equ= np.ma.filled(cdf_m,0).astype('uint16')
    return cdf_equ[img]

def plot_ecualized_image(img, title):
    img1 = np.array(img, dtype=np.float64)
    # Normalizar:
    img2 = (img1 - np.min(img1)) * 65536/(np.max(img1) - np.min(img1))
    # Pasar a entero:
    img3 = np.array(img2, dtype=np.uint16)
    # Ecualizar:
    img4 = Ecualizauint16(img3)
    
    plt.imshow(img4,cmap='gray'),plt.title(title), plt.axis('off')


# In[4]:


imageMunich = pyreadr.read_r('data/real_SAR_images/Munich.RData')
Munich = np.array(list(imageMunich.items())[0][1])
plot_ecualized_image(Munich, 'Imagen de Munich')


# In[5]:


imageSanFran = pyreadr.read_r('data/real_SAR_images/AirSAR_SanFrancisc_Enxu.RData')
SanFran = np.array(list(imageSanFran.items())[0][1])
SanFranIm = SanFran[:,:,0] # Esta es la banda HH
plot_ecualized_image(SanFranIm, 'Imagen de San Francisco, California, USA HH')


# Elegir el archivo de configuración correspondiente:

# In[6]:


config_name = 'config_1' # Elegir

config_path = f'configs/{config_name}.yaml'
config = OmegaConf.load(config_path)
config


# Cargo el autoencoder ya entrenado

# In[7]:


# 1. Crear una instancia del modelo (debe tener la misma arquitectura)
autoencoder_cargado = ConfigurableAutoencoder(config=config)
# 2. Carga los parámetros
autoencoder_cargado.load_state_dict(torch.load(f'data/trained_models/{config_name}.pth'))
# 3. Modo evaluación (cuando lo use para inferencia)
autoencoder_cargado.eval()


# In[8]:


ciudades = {
    'Munich': Munich,
    'San Francisco': SanFranIm
}


# In[9]:


# Veo un pedazo (de 50x50) de una imagen real pasado por el autoencoder

ciudad = 'Munich' # Elegir ciudad según el diccionario

i_rand = random.randint(0, ciudades[ciudad].shape[0]-50)
j_rand = random.randint(0, ciudades[ciudad].shape[1]-50)

Image = ciudades[ciudad][i_rand:i_rand+50, j_rand:j_rand+50].astype(np.float64)
Image = Image / np.max(Image)

Im_tensor = torch.from_numpy(Image).double()
Im_tensor = Im_tensor.unsqueeze(0)
Im_tensor = Im_tensor.unsqueeze(1)
autoencoder_cargado = autoencoder_cargado.double()

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(Image)
plt.title("Original")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(autoencoder_cargado(Im_tensor).detach().numpy()[0, 0, :, :])
plt.title("Limpia")
plt.axis('off')

plt.tight_layout()
plt.show()


# ---
# Solo para probar veo como se ve el autoencoder pasado por una imagen del conjunto de test

# In[10]:


a = partitioned_gi0_image(
    p_alphas=[-8],
    p_gammas=[1.0],
    p_looks=[1],
    n_cuad_lado=1,
    pixeles_cuad=50
)

Image = a[2].astype(np.float64)
Image = Image / np.max(Image)
Im_tensor = torch.from_numpy(Image).double()
Im_tensor = Im_tensor.unsqueeze(0)
Im_tensor = Im_tensor.unsqueeze(1)
autoencoder_cargado = autoencoder_cargado.double()

plt.figure(figsize=(8, 4))

plt.subplot(1, 3, 1)
plt.imshow(a[2])
plt.title('Imagen sucia (GI0)')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(a[1])
plt.title('Imagen limpia (GI)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(autoencoder_cargado(Im_tensor).detach().numpy()[0,0,:,:])
plt.title("Imagen pasada por el autoencoder")
plt.axis('off')

plt.tight_layout()
plt.show()


# El problema que se ve aca es que si el modelo fue entrenado con ciertas particiones, parece aprender los bordes de esas particiones, con lo cual no funciona bien cuando se le da imágenes con particiones diferentes. Y probablemente por esta misma razón no siempre funciona muy bien en las imágenes reales.
