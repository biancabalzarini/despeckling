#!/usr/bin/env python
# coding: utf-8

# In[42]:


import sys
sys.path.append('..')

from scripts.GenrationGI0 import partitioned_gi0_image
from scripts.autoencoders import generate_multiple_images

import numpy as np
import matplotlib.pyplot as plt


# Genero imágenes para visualizar cómo quedan con las funciones que diseñé

# In[74]:


# Cantidad de imágenes a generar
n = 500
# Cantidad de cuadrados por lado que van a tener las imágenes (cada cuadrado con diferentes parámetros de la GI0)
n_cuad_lado = 4
# Cantidad de píxeles por lado que tiene cada cuadrado de las imágenes
pixeles_cuad = 25


# In[75]:


train_g, train_gi, train_gI0 = generate_multiple_images(n, partitioned_gi0_image, n_cuad_lado, pixeles_cuad)


# In[76]:


index = int(n*np.random.random())


# In[77]:


plt.imshow(train_g[index,:,:], cmap='gray')
plt.title('Distribución Gamma')


# In[78]:


plt.imshow(train_gi[index,:,:], cmap='gray')
plt.title('Distribución Gamma Inversa')


# In[79]:


plt.imshow(train_gI0[index,:,:], cmap='gray')
plt.title('Distribución GI0')

