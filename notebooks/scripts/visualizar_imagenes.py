#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('..')

from scripts.GenrationGI0 import partitioned_gi0_image
from scripts.autoencoders import generate_multiple_images

import numpy as np
import matplotlib.pyplot as plt


# Genero imágenes para visualizar cómo quedan con las funciones que diseñé

# In[2]:


# Cantidad de imágenes a generar
n = 500
# Cantidad de cuadrados por lado que van a tener las imágenes (cada cuadrado con diferentes parámetros de la GI0)
n_cuad_lado = 3
# Cantidad de píxeles por lado que tiene cada cuadrado de las imágenes
pixeles_cuad = 20


# In[3]:


train_g, train_gi, train_gI0 = generate_multiple_images(n, partitioned_gi0_image, n_cuad_lado, pixeles_cuad)


# In[4]:


index = int(n*np.random.random())


# In[5]:


plt.imshow(train_g[index,:,:], cmap='gray')
plt.title('Distribución Gamma')


# In[6]:


plt.imshow(train_gi[index,:,:], cmap='gray')
plt.title('Distribución Gamma Inversa')


# In[7]:


plt.imshow(train_gI0[index,:,:], cmap='gray')
plt.title('Distribución GI0')


# In[8]:


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

im1 = ax1.imshow(train_g[index,:,:], cmap='gray')
ax1.set_title('Distribución Gamma')

im2 = ax2.imshow(train_gi[index,:,:], cmap='gray')
ax2.set_title('Distribución Gamma Inversa')

im3 = ax3.imshow(train_gI0[index,:,:], cmap='gray')
ax3.set_title('Distribución GI0')

plt.tight_layout()

