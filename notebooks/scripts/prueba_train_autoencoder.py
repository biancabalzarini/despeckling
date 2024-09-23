#!/usr/bin/env python
# coding: utf-8

# In[34]:


import sys
sys.path.append('..')

from scripts.GenrationGI0 import rGI0, partitioned_gi0_image
from scripts.autoencoders import InMemoryImageDataset

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

# In[23]:


imagen_g, imagen_gi, imagen_gI0 = partitioned_gi0_image(
    p_alphas=[-1.5,-1.6,-1.7,-1.55],
    p_gammas=[1,2,3,4],
    p_looks=[1,2,3,4]
)


# In[26]:


imagen_g.shape


# In[36]:


all_images.shape # tiene que tener shape (nr_o_imagenes, 100, 100)


# In[ ]:


batch_size = 32


# In[30]:


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = InMemoryImageDataset(all_images, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# ---
# ### Entreno

# 
