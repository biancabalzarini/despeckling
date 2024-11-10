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
    p_looks: List[int],
    n_cuad_lado: int,
    pixeles_cuad: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Genera una imagen dividida en n_cuad_lado^2 cuadrados, cada uno de pixeles_cuad píxeles por lado,
    usando la distribución GI0 en cada uno. Cada cuadrado tiene sus propios parámetros de alpha, gamma
    y número de looks.

    Parameters
    ----------
    p_alphas: List[float]
        Lista de n_cuad_lado^2 valores para el parámetro alpha de cada cuadrado.
    p_gammas: List[float]
        Lista de n_cuad_lado^2 valores para el parámetro gamma de cada cuadrado.
    p_looks: List[int]
        Lista de n_cuad_lado^2 valores para el número de Looks de cada cuadrado.
    n_cuad_lado: int
        Número de cuadrados por lado de la imagen.
    pixeles_cuad: int
        Número de píxeles por lado de cada cuadrado.

    Returns
    -------
    imagen_g: np.ndarray
        Imagen de n_cuad_lado*pixeles_cuad píxeles por lado, generada a partir de la distribución gamma.
    imagen_gi: np.ndarray
        Imagen de n_cuad_lado*pixeles_cuad píxeles por lado, generada a partir de la distribución gamma inversa.
    imagen_gI0: np.ndarray
        Imagen de n_cuad_lado*pixeles_cuad píxeles por lado, generada a partir de la distribución GI0.
    """
    total_cuadrados = n_cuad_lado ** 2
    if len(p_alphas) != total_cuadrados or len(p_gammas) != total_cuadrados or len(p_looks) != total_cuadrados:
        raise ValueError(f"Todas las listas de parámetros deben contener exactamente {total_cuadrados} valores.")

    tam_imagen = n_cuad_lado * pixeles_cuad
    imagen_g = np.zeros((tam_imagen, tam_imagen))
    imagen_gi = np.zeros((tam_imagen, tam_imagen))
    imagen_gI0 = np.zeros((tam_imagen, tam_imagen))

    count = 0
    for i in range(n_cuad_lado):
        for j in range(n_cuad_lado):
            g, gi, gI0 = rGI0(pixeles_cuad**2, p_alphas[count], p_gammas[count], p_looks[count])
            
            inicio_i = i * pixeles_cuad
            fin_i = (i + 1) * pixeles_cuad
            inicio_j = j * pixeles_cuad
            fin_j = (j + 1) * pixeles_cuad
            
            imagen_g[inicio_i:fin_i, inicio_j:fin_j] = g.reshape(pixeles_cuad, pixeles_cuad)
            imagen_gi[inicio_i:fin_i, inicio_j:fin_j] = gi.reshape(pixeles_cuad, pixeles_cuad)
            imagen_gI0[inicio_i:fin_i, inicio_j:fin_j] = gI0.reshape(pixeles_cuad, pixeles_cuad)

            count += 1
            
    return imagen_g, imagen_gi, imagen_gI0
