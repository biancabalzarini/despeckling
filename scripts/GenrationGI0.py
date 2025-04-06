import numpy as np
from typing import List, Tuple
import random
from typing import List, Tuple, Callable, Optional, Union


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

def generate_multiple_images(
    n: int,
    partitioned_gi0_image: Callable,
    n_cuad_lado: int,
    pixeles_cuad: int,
    alpha_values: Optional[List[float]] = None,
    save_alpha_values: Optional[bool] = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Genera múltiples conjuntos de imágenes utilizando la función partitioned_gi0_image.

    Parameters:
    -----------
    n: int
        Número de conjuntos de imágenes a generar.
    partitioned_gi0_image: function
        Función que genera un conjunto de imágenes (g, gi, gI0).
    n_cuad_lado: int
        Número de cuadrados por lado de cada imagen (ver documentación de partitioned_gi0_image).
    pixeles_cuad: int
        Número de píxeles por lado de cada cuadrado que forma cada imagen (ver documentación de
        partitioned_gi0_image).
    alpha_values: Optional[List[float]]
        Lista de valores posibles para alpha. Si no se proporciona, se usarán los valores por defecto.
    save_alpha_values: Optional[bool]
        Indica si se quiere o no guardar los valores de alpha usados en cada cuadrante de cada imagen.

    Returns:
    --------
    conjunto_g: np.ndarray
        Array de forma (n, n_cuad_lado*pixeles_cuad, n_cuad_lado*pixeles_cuad) con n repeticiones de imagen_g.
    conjunto_gi: np.ndarray
        Array de forma (n, n_cuad_lado*pixeles_cuad, n_cuad_lado*pixeles_cuad) con n repeticiones de imagen_gi.
    conjunto_gI0: np.ndarray
        Array de forma (n, n_cuad_lado*pixeles_cuad, n_cuad_lado*pixeles_cuad) con n repeticiones de imagen_gI0.
    alphas: Optional[np.ndarray]
        Array de forma (n, n_cuad_lado, n_cuad_lado) con los valores alpha de cada imagen en cada cuadrante.
        Solo se devuelve si save_alpha_values=True.
    """
    tam_imagen = n_cuad_lado * pixeles_cuad
    conjunto_g = np.zeros((n, tam_imagen, tam_imagen))
    conjunto_gi = np.zeros((n, tam_imagen, tam_imagen))
    conjunto_gI0 = np.zeros((n, tam_imagen, tam_imagen))
    alphas = np.zeros((n, n_cuad_lado, n_cuad_lado))
    
    total_cuadrados = n_cuad_lado ** 2
    p_gammas: List[float] = [1.0] * total_cuadrados
    p_looks: List[int] = [1] * total_cuadrados
    
    if alpha_values is None:
        alpha_values = [-1.5, -2, -3, -5, -6, -8, -10, -20]
    
    for i in range(n):
        p_alphas: List[float] = random.choices(alpha_values, k=total_cuadrados)
        alphas_imagen_individual = np.array(p_alphas).reshape(2,2)
        
        imagen_g, imagen_gi, imagen_gI0 = partitioned_gi0_image(p_alphas, p_gammas, p_looks, n_cuad_lado, pixeles_cuad)
        
        conjunto_g[i] = imagen_g
        conjunto_gi[i] = imagen_gi
        conjunto_gI0[i] = imagen_gI0
        if save_alpha_values:
            alphas[i] = alphas_imagen_individual
    
    if save_alpha_values:
        return conjunto_g, conjunto_gi, conjunto_gI0, alphas
    else:
        return conjunto_g, conjunto_gi, conjunto_gI0

def mixed_dataset(
    n_total: int,
    generate_multiple_images: Callable,
    conjunto_n_cuad_lado: List[int],
    conjunto_pixeles_cuad: List[int],
    ratios: Optional[List[float]] = [1],
    alpha_values: Optional[List[float]] = None,
    save_alpha_values: Optional[bool] = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Genera múltiples conjuntos de imágenes utilizando la función partitioned_gi0_image, con diferentes particiones
    de las imágenes según se especifica en los parámetros de la función.
    
    Parameters:
    -----------
    n_total: int
        Número de conjuntos de imágenes a generar. Por temas de redondeo es posible que se genere una cantidad
        apenas distinta de imágenes totales.
    generate_multiple_images: function
        Función que genera conjuntos de imágenes con la misma partición.
    conjunto_n_cuad_lado: List[int]
        Lista con las diferentes cantidades de particiones (cuadrados por lado) que se quieren. Por ejemplo, si
        se quiere imágenes sin particionar, otras con 4 subcuadrados, y otras con 9, debería usarse [1,2,3]. Si se
        quisiera un único tipo de partición se debe ingresar una lista de un único elemento.
    conjunto_pixeles_cuad: List[int]
        Análogo a conjunto_n_cuad_lado pero con la cantidad de píxeles de cada subcuadrado de la partición. La
        múltiplicación del elemento n de conjunto_n_cuad_lado por el elemento n de conjunto_pixeles_cuad debe
        coincidir para todo n, porque eso implica que todas las imágenes serán del mismo tamaño.
    ratios: Optional[List[float]]
        Lista que indica el ratio de cada una de las particiones que se quiere tener en el dataset final. Cada
        elemento debe ser menor o igual a 1 (1 en el caso de que sea una única partición), y todos los elementos
        deben sumar 1. En el caso de tener una única partición, se puede ingresar la lista [1] o no ingresar nada.
    alpha_values: Optional[List[float]], opcional
        Lista de valores posibles para alpha. Si no se proporciona, se usarán los valores por defecto.
    save_alpha_values: Optional[bool]
        Indica si se quiere o no guardar los valores de alpha usados en cada cuadrante de cada imagen.
    
    Returns:
    --------
    g_final : np.ndarray
        Array de forma (n_total, tamaño_imagen, tamaño_imagen) con n_total repeticiones de imagen_g, con diferentes
        particiones según lo indicado en los inputs.
    gi_final : np.ndarray
        Array de forma (n_total, tamaño_imagen, tamaño_imagen) con n_total repeticiones de imagen_gi, con diferentes
        particiones según lo indicado en los inputs.
    gI0_final : np.ndarray
        Array de forma (n_total, tamaño_imagen, tamaño_imagen) con n_total repeticiones de imagen_gI0, con diferentes
        particiones según lo indicado en los inputs.
    alphas_final: Optional[List[np.ndarray()]]
        Lista donde cada elemento es un np.ndarray con los valores de alpha de cada cuadrante de cada imagen para
        un único tipo de partición. Las diferentes particiones van en diferentes elementos de la lista. Solo se
        devuelve si save_alpha_values=True.
    """
    conjunto_n_cuad_lado = list(conjunto_n_cuad_lado)
    conjunto_pixeles_cuad = list(conjunto_pixeles_cuad)
    ratios = list(ratios)

    assert (len(conjunto_n_cuad_lado) == 1 and ratios == [1]) or \
           (len(conjunto_n_cuad_lado) == len(conjunto_pixeles_cuad) == len(ratios)), \
           "Los largos de las listas que indican las particiones deben coincidir y ser coherentes"

    assert float(sum(ratios)) == 1.0, "Los ratios tienen que sumar 1"

    tamaños_imagenes = np.array(conjunto_n_cuad_lado) * np.array(conjunto_pixeles_cuad)
    assert np.all(tamaños_imagenes == tamaños_imagenes[0]), \
        f"Los tamaños de las imágenes no son todos iguales: {tamaños_imagenes}"
    
    ns = [int(ratio * n_total) for ratio in ratios]

    g_list = []
    gi_list = []
    gI0_list = []
    alphas_final = []
    
    for n_cuad_lado, pixeles_cuad, n in zip(conjunto_n_cuad_lado, conjunto_pixeles_cuad, ns):
        
        if save_alpha_values:
            g, gi, gI0, alphas = generate_multiple_images(n, partitioned_gi0_image, n_cuad_lado, pixeles_cuad, alpha_values, save_alpha_values)
            alphas_final.append(alphas)
        else:
            g, gi, gI0 = generate_multiple_images(n, partitioned_gi0_image, n_cuad_lado, pixeles_cuad, alpha_values)
        g_list.append(g)
        gi_list.append(gi)
        gI0_list.append(gI0)
    
    g_final = np.concatenate(g_list, axis=0)
    gi_final = np.concatenate(gi_list, axis=0)
    gI0_final = np.concatenate(gI0_list, axis=0)
    
    if save_alpha_values:
        return g_final, gi_final, gI0_final, alphas_final
    else:
        return g_final, gi_final, gI0_final
