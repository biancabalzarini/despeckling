import numpy as np
from typing import List, Tuple

def rGI0(
  n: int,
  p_alpha: float,
  p_gamma: float,
  p_Looks: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """
  Samplea de la distribución GI0.

  Parameters
  ----------
  n: int
      Cantidad de muestras.
  p_alpha: float
      El valor del parámetro alpha. Solo se aceptan valores negativos.
  p_gamma: float
      El valor del parámetro gamma.
  p_Looks: int
      Cantidad de Looks.

  Returns
  -------
  g: np.ndarray
      Sampleo de tamaño n de la distribución gamma.
  gi: np.ndarray
      Sampleo de tamaño n de la distribución gamma inversa.
  gI0: np.ndarray
      Sampleo de tamaño n de la distribución GI0. Es el producto entre g y gi.
  """
  g = np.random.gamma(p_Looks, 1/p_Looks, n)
  gi = 1 / np.random.gamma(-p_alpha, 1/p_gamma, n)
  gI0 = g * gi
  
  return g, gi, gI0

def partitioned_gi0_image(
    p_alphas: List[float],
    p_gammas: List[float],
    p_looks: List[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Genera una imagen de 100x100 dividida en 4 cuadrados de 25x25 usando la distribución GI0.
    Cada cuadrado tiene sus propios parámetros de alpha, gamma y número de looks.

    Parameters
    ----------
    p_alphas: List[float]
        Lista de 4 valores para el parámetro alpha de cada cuadrado.
    p_gammas: List[float]
        Lista de 4 valores para el parámetro gamma de cada cuadrado.
    p_looks: List[int]
        Lista de 4 valores para el número de Looks de cada cuadrado.

    Returns
    -------
    imagen_g: np.ndarray
        Imagen de 100x100 generada a partir de la distribución gamma.
    imagen_gi: np.ndarray
        Imagen de 100x100 generada a partir de la distribución gamma inversa.
    imagen_gI0: np.ndarray
        Imagen de 100x100 generada a partir de la distribución GI0.
    """
    if len(p_alphas) != 4 or len(p_gammas) != 4 or len(p_looks) != 4:
        raise ValueError("Todas las listas de parámetros deben contener exactamente 4 valores.")

    imagen_g = np.zeros((100, 100))
    imagen_gi = np.zeros((100, 100))
    imagen_gI0 = np.zeros((100, 100))
    n = 25

    ### La parte del idx de estos dos fors parece que solo funcionaría para la imagen
    ### partida en 4 partes. Si lo llego a generalizar, rechequearlo.
    for i in range(2):
        for j in range(2):
            idx = i * 2 + j
            g, gi, gI0 = rGI0(n**2, p_alphas[idx], p_gammas[idx], p_looks[idx])
            
            imagen_g[i*n:(i+1)*n, j*n:(j+1)*n] = g.reshape(n, n)
            imagen_gi[i*n:(i+1)*n, j*n:(j+1)*n] = gi.reshape(n, n)
            imagen_gI0[i*n:(i+1)*n, j*n:(j+1)*n] = gI0.reshape(n, n)

    return imagen_g, imagen_gi, imagen_gI0
