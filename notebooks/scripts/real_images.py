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
import torch


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

