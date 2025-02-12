#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
sys.path.append('..')

from scripts.GenrationGI0 import partitioned_gi0_image, generate_multiple_images

import numpy as np
import matplotlib.pyplot as plt
import cv2


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


ecualizar_hist = True # Si se quiere o no ecualizar el histograma de la imagen

imagen = train_g[index,:,:]
titulo = 'Distribución Gamma'

if ecualizar_hist == 1:
    imagen = cv2.equalizeHist(imagen.astype(np.uint8))
    titulo = titulo + '\n(ecualizada)'
    
plt.imshow(imagen, cmap='gray')
plt.title(titulo)


# In[6]:


ecualizar_hist = True # Si se quiere o no ecualizar el histograma de la imagen

imagen = train_gi[index,:,:]
titulo = 'Distribución Gamma Inversa'

if ecualizar_hist == 1:
    imagen = cv2.equalizeHist(imagen.astype(np.uint8))
    titulo = titulo + '\n(ecualizada)'
    
plt.imshow(imagen, cmap='gray')
plt.title(titulo)


# In[7]:


ecualizar_hist = True # Si se quiere o no ecualizar el histograma de la imagen

imagen = train_gI0[index,:,:]
titulo = 'Distribución GI0'

if ecualizar_hist == 1:
    imagen = cv2.equalizeHist(imagen.astype(np.uint8))
    titulo = titulo + '\n(ecualizada)'
    
plt.imshow(imagen, cmap='gray')
plt.title(titulo)


# In[8]:


ecualizar_hist = True  # Si se quiere o no ecualizar el histograma de la imagen

imagenes = [train_g, train_gi, train_gI0]
titulos = ['Distribución Gamma', 'Distribución Gamma Inversa', 'Distribución GI0']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, imagen, titulo in zip(axes, imagenes, titulos):
    if ecualizar_hist:
        imagen = cv2.equalizeHist(imagen[index, :, :].astype(np.uint8))
        titulo += '\n(ecualizada)'
    
    ax.imshow(imagen, cmap='gray')
    ax.set_title(titulo)

plt.tight_layout()

