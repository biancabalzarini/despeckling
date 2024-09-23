#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('..')

from scripts.GenrationGI0 import rGI0, partitioned_gi0_image
from scripts.autoencoders import InMemoryImageDataset, generate_multiple_images

import matplotlib.pyplot as plt

from torchvision import transforms
from torch.utils.data import DataLoader


# ---
# ### Empiezo graficando algunos ejemplos de imagenes

# In[2]:


g, gi, gI0 = rGI0(n=100*100, p_alpha=-1.5, p_gamma=1, p_Looks=1)


# In[3]:


g = g.reshape(100, 100)
gi = gi.reshape(100, 100)
gI0 = gI0.reshape(100, 100)


# In[17]:


plt.imshow(g)
plt.title('Ruido speckle ~ Gamma')


# In[16]:


plt.imshow(gi)
plt.title('Backscatter ~ Gamma inversa')


# In[22]:


plt.imshow(gI0)
plt.title('Imagen + ruido speckle ~ GI0')


# In[12]:


imagen_g, imagen_gi, imagen_gI0 = partitioned_gi0_image(
    p_alphas=[-1.5,-1.6,-1.7,-1.55],
    p_gammas=[1,2,3,4],
    p_looks=[1,2,3,4]
)


# In[18]:


plt.imshow(imagen_g)
plt.title('Imagen particionada - Ruido speckle ~ Gamma')


# In[20]:


plt.imshow(imagen_gi)
plt.title('Imagen particionada - Backscatter ~ Gamma inversa')


# In[21]:


plt.imshow(imagen_gI0)
plt.title('Imagen particionada - Imagen + ruido speckle ~ GI0')


# ---
# ### Genero un dataset para entrenar

# In[14]:


n = 1000
train_g, train_gi, train_gI0 = generate_multiple_images(n, partitioned_gi0_image)


# In[15]:


batch_size = 32


# In[23]:


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_entrada = InMemoryImageDataset(train_gI0, transform=transform)
entrada_loader = DataLoader(train_entrada, batch_size=batch_size, shuffle=True)

train_salida = InMemoryImageDataset(train_gi, transform=transform)
entrada_salida = DataLoader(train_salida, batch_size=batch_size, shuffle=True)


# In[22]:


image = train_entrada[5]
plt.imshow(image[0,:,:])


# In[24]:


image = train_salida[5]
plt.imshow(image[0,:,:])


# ---
# ### Entreno

# 
