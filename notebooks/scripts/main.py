#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('..')

from scripts.GenrationGI0 import partitioned_gi0_image
from scripts.autoencoders import InMemoryImageDataset, generate_multiple_images, Autoencoder

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import cv2


# ---
# # Creación del dataset para entrenar

# In[2]:


## PARÁMETROS

# Cantidad de imágenes a generar
n = 50000
# Cantidad de cuadrados por lado que van a tener las imágenes (cada cuadrado con diferentes parámetros de la GI0)
n_cuad_lado = 2
# Cantidad de píxeles por lado que tiene cada cuadrado de las imágenes
pixeles_cuad = 25
# Tamaños de los batches
batch_size = 50


# In[3]:


train_g, train_gi, train_gI0 = generate_multiple_images(n, partitioned_gi0_image, n_cuad_lado, pixeles_cuad)


# In[4]:


normalize_to_01 = transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min()))

transform = transforms.Compose([
    transforms.ToTensor(),
    normalize_to_01
])

dataset_train = InMemoryImageDataset(train_gI0, train_gi, transform=transform)
train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)


# ---
# # Entrenamiento

# In[5]:


# PARÁMETROS

# Dimensión de la capa más interna del autoencoder
encoding_dim = 32
# Learning rate
learning_rate = 1e-3
# Cantidad de épocas
num_epochs = 10


# In[6]:


autoencoder = Autoencoder(image_size=n_cuad_lado*pixeles_cuad, encoding_dim=encoding_dim)
criterion = nn.BCELoss() # Utilizamos Binary Cross Entropy Loss como loss function ya que las imágenes están normalizadas en el rango [0, 1]
optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate) # El optimizador es responsable de ajustar los pesos del modelo con el fin de minimizar la función de pérdida.
                                                                   # Adam es un algoritmo de optimización popular y eficiente que adapta la tasa de aprendizaje de forma dinámica para cada parámetro del modelo.
                                                                   # La tasa de aprendizaje determina qué tan rápido se ajustan los pesos del modelo durante el entrenamiento.


# In[7]:


for epoch in range(num_epochs):
    for data in train_loader:
        entrada, salida = data
        entrada = entrada.view(entrada.size(0), -1).float()
        salida = salida.view(salida.size(0), -1).float()

        # Forward pass
        outputs = autoencoder(entrada) # Se pasa a las imágenes por el autoencoder, en una pasada forward.
        loss = criterion(outputs, salida) # Se calcula la diferencia entre el output y las imágenes sin ruido, según la función de pérdida definida.

        # Backward pass y optimización
        optimizer.zero_grad() # Se restablecen los gradientes acumulados en todos los parámetros del modelo.
                              # Esto es necesario antes de realizar el backward pass, ya que PyTorch acumula los gradientes en cada llamada a loss.backward().
        loss.backward() # Se realiza el backward pass para calcular los gradientes de los parámetros del autoencoder utilizando la función de pérdida.
        optimizer.step() # Finalmente se realiza la optimización de los parámetros del modelo mediante la llamada a optimizer.step(), que actualiza los parámetros en función de los gradientes calculados.

    # Imprimir la pérdida del autoencoder en cada época
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


# ---
# # Evaluación

# In[8]:


# PARÁMETROS

# Cantidad de imágenes a generar para evaluar
n = 1000
# Tamaños de los batches
batch_size = 32


# In[9]:


test_g, test_gi, test_gI0 = generate_multiple_images(n, partitioned_gi0_image, n_cuad_lado, pixeles_cuad)


# In[10]:


dataset_test = InMemoryImageDataset(test_gI0, test_gi, transform=transform)
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)


# In[11]:


total_loss = 0
with torch.no_grad(): # Esto es para asegurarse de que no se realicen cálculos de gradientes durante la evaluación del autoencoder.
                      # Al entrar en este bloque, se desactiva el cálculo y almacenamiento automático de gradientes para reducir el uso de memoria y acelerar la evaluación.
    for data in test_loader:
        entrada, salida = data
        entrada = entrada.view(entrada.size(0), -1).float() # Se modifica la forma de las imágenes para que coincida con el formato esperado por el autoencoder.
                                                            # En este caso, las imágenes se aplanan en un tensor unidimensional. images.size(0) se utiliza para obtener el tamaño del lote.
        salida = salida.view(salida.size(0), -1).float()

        # Forward pass
        outputs = autoencoder(entrada) # Se realiza el forward pass del autoencoder con las imágenes de prueba.
                                       # El autoencoder genera las imágenes reconstruidas utilizando el método forward() que definimos previamente en la clase Autoencoder.
        loss = criterion(outputs, salida) # Se calcula la pérdida entre las imágenes reconstruidas y las imágenes originales utilizando la función de pérdida (criterion).
                                          # Esto proporciona una medida de cuánto difieren las imágenes reconstruidas de las originales.
        total_loss += loss.item() # La pérdida obtenida en cada iteración se suma a la variable total_loss utilizando loss.item(), que devuelve el valor escalar de la pérdida.
                                  # Al final de la iteración, total_loss contendrá la suma acumulada de las pérdidas de todas las muestras del conjunto de datos de prueba.

average_loss = total_loss / len(test_loader) # Se calcula la pérdida promedio dividiendo la suma acumulada de las pérdidas (total_loss) entre el número de lotes en el conjunto de datos de prueba (len(test_loader)).
                                             # Esto proporciona una medida promedio de la discrepancia entre las imágenes originales y las imágenes reconstruidas por el autoencoder en el conjunto de datos de prueba.
print(f"Average Test Loss: {average_loss:.4f}")


# In[12]:


# Aplico el autoencoder a un ejemplo particular del dataset de testeo y veo cómo queda la
# imagen de salida.

ecualizar_hist = True  # Si se quiere o no ecualizar el histograma de la imagen

###

index = int(n*np.random.random()) # Índice del ejemplo puntual que se desea seleccionar
entrada_red, salida_red = dataset_test[index]

example = entrada_red.view(1, -1).float() # Ajusta la forma de la imagen a un lote de tamaño 1

reconstructed = autoencoder(example) # Aplica el autoencoder al ejemplo

tamanio = n_cuad_lado*pixeles_cuad

entrada = entrada_red.view(tamanio, tamanio)
salida_esperada = salida_red.view(tamanio, tamanio)
reconstructed = reconstructed.view(tamanio, tamanio)

imagenes = [entrada, salida_esperada, reconstructed.detach()]
titulos = ['Entrada', 'Salida esperada', 'Salida de la red']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, imagen, titulo in zip(axes, imagenes, titulos):
    if ecualizar_hist:
        im = imagen.cpu().numpy()
        im = ((im - im.min()) * 255) / (im.max() - im.min())
        imagen = cv2.equalizeHist(im.astype(np.uint8))
        titulo += '\n(ecualizada)'
    
    ax.imshow(imagen, cmap='gray')
    ax.set_title(titulo)

plt.tight_layout()

