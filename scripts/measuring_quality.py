import numpy as np
from typing import List, Tuple
import random
import copy
from skimage.feature import graycomatrix


def selecting_quadrants(
    alphas: List[np.ndarray],
    M: int
) -> List[List]:
    """
    Selecciona cuadrantes dentro de cada imágen en donde el valor de alpha sea menor o igual a -6 (para garantizar
    homogeneidad). Elige M cuadrantes dentro de cada imagen.

    Parameters
    ----------
    alphas: List[np.ndarray]
        Lista donde cada elemento es un np.ndarray con los valores de alpha de cada cuadrante de cada imagen para
        un único tipo de partición. Las diferentes particiones van en diferentes elementos de la lista.
    M: int
        Cantidad de zonas homogéneas que quiero tomar
    
    Returns
    -------
    cuadrantes: List[List]
        Una lista en donde cada elemento es una sublista. Cada sublista corresponde a cada imagen del dataset. Si
        la imagen no tiene cuadrantes con alpha menor o igual a -6, entonces la sublista estará vacía. En caso
        contrario, la sublista tendrá un sampleo de longitud M de los cuadrantes en donde eso sí se cumpla.
    """
    coordenadas = [] # Aca voy a guardar las posiciones de los cuadrantes en donde alpha es menor o igual a -6.
                     # Es decir, los cuadrantes en donde la imagen se considera homogénea.

    for array in alphas: # Loopeo por cada tipo distinto de partición de imágenes
        for element in array: # Loopeo por todas las imágenes con el mismo tipo de partición
            filas, columnas = np.where(element <= -6)
            c = [(int(fila), int(col)) for fila, col in zip(filas, columnas)]
            coordenadas.append(c)

    cuadrantes = []

    for sublista in coordenadas:
        
        if len(sublista) == 0:
            cuadrantes.append([])
            
        elif len(sublista) == M:
            cuadrantes.append(sublista)
            
        elif len(sublista) > M:
            # Elegir M tuplas distintas al azar
            seleccion = random.sample(sublista, M)
            cuadrantes.append(seleccion)
            
        else: # len(sublista) < M
            # Distribuir uniformemente y samplear con repetición balanceada
            nlist = len(sublista)
            repeticiones_base = M // nlist
            repeticiones_extra = M % nlist
            
            lista_distribuida = []
            for i in range(nlist):
                repeticiones = repeticiones_base + (1 if i < repeticiones_extra else 0)
                lista_distribuida.extend([sublista[i]] * repeticiones)
            
            cuadrantes.append(lista_distribuida)
        
    return cuadrantes

def quadrants_to_pixels(
    cuadrantes: List[List],
    pixeles_cuad: List,
    alphas: List[np.ndarray]
) -> Tuple[List, List]:
    """
    Convierte los cuadrantes en coordenadas de píxeles dentro de la imagen. Devuelve el x,y tanto inicial
    como final del cuadrante, medidos en píxeles.

    Parameters
    ----------
    cuadrantes: List[List]
        Lista de listas indicando los cuadrantes a transformar en coordenadas.
    pixeles_cuad: List
        Lista con la cantidad de píxeles de cada cuadrante en cada una de las distintas particiones del
        dataset. Puede obtenerse del config, como config.training.pixeles_cuad.
    alphas: List[np.ndarray]
        Lista donde cada elemento es un np.ndarray con los valores de alpha de cada cuadrante de cada imagen para
        un único tipo de partición. Las diferentes particiones van en diferentes elementos de la lista.

    Returns
    -------
    cuadrantes_i: List[List]
        Análogo al input cuadrantes pero con los x,y iniciales medidos en píxeles de cada cuadrante.
    cuadrantes_f: List[List]
        Análogo al input cuadrantes pero con los x,y finales medidos en píxeles de cada cuadrante.
    """
    partitions = [len(sublista) for sublista in alphas]
    p = [pixel for pixel, count in zip(pixeles_cuad, partitions) for _ in range(count)]

    cuadrantes_i = copy.deepcopy(cuadrantes)
    cuadrantes_f = copy.deepcopy(cuadrantes)

    for i in range(len(cuadrantes)):         # Para loopear por las imágenes
        pi = p[i]
        
        for j in range(len(cuadrantes[i])):  # Para loopear por cada una de las zonas que quiero crear en un única imagen
            cuadrante_x = cuadrantes[i][j][0]
            cuadrante_y = cuadrantes[i][j][1]

            fila_i = cuadrante_x*pi     # Fila de inicio del cuadrante
            columna_i = cuadrante_y*pi  # Columna de inicio del cuadrante

            fila_f = fila_i + pi        # Fila de fin del cuadrante
            columna_f = columna_i + pi  # Columna de fin del cuadrantes
            
            cuadrantes_i[i][j] = (fila_i, columna_i)
            cuadrantes_f[i][j] = (fila_f, columna_f)
    
    return cuadrantes_i, cuadrantes_f

def selecting_homogeneous_areas(
    coord_i: Tuple[int, int],
    coord_f: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """
    A partir de los límites en x e y de un cuadrante, genera una zona aleatoria allí dentro. El tamaño de esa
    zona se elige según el tamaño del cuadrante. De lo posible se usa una zona de 10x10, sino de 9x9, y sino
    de 8x8.

    Parameters
    ----------
    coord_i: Tuple[int, int]
        Tupla con los valores de x e y (en píxeles) iniciales del cuadrante.
    coord_f: Tuple[int, int]
        Tupla con los valores de x e y (en píxeles) finales del cuadrante.

    Returns
    -------
    zona_xi: int
        Valor de x inicial (en píxeles) de la zona aleatoriamente elegida.
    zona_yi: int
        Valor de y inicial (en píxeles) de la zona aleatoriamente elegida.
    zona_xf: int
        Valor de x final (en píxeles) de la zona aleatoriamente elegida.
    zona_yf: int
        Valor de y final (en píxeles) de la zona aleatoriamente elegida.
    """
    M = min(10, max(8, coord_f[0] - coord_i[0]))
    max_x = coord_f[0] - M
    max_y = coord_f[1] - M

    zona_xi = np.random.randint(coord_i[0], max_x+1)
    zona_yi = np.random.randint(coord_i[1], max_y+1)
    zona_xf = zona_xi + M
    zona_yf = zona_yi + M

    return zona_xi, zona_yi, zona_xf, zona_yf

def single_image_first_order_method(
    cuads_i: List[Tuple[int, int]],
    cuads_f: List[Tuple[int, int]],
    original_image: np.ndarray,
    ratio_image: np.ndarray
) -> int:
    """
    Aplica el método de primer orden a una imagen individual para cuantificar la bondad del filtrado.

    Parameters
    ----------
    cuads_i: List[Tuple[int, int]]
        Lista con las coordenadas iniciales de los cuadrantes en donde se van a tomar zonas homogéneas para
        calcular el estadístico de primer orden. Cada cuadrante se repite tantas veces como vaya a ser
        necesario samplearlo.
    cuads_f: List[Tuple[int, int]]
        Lista con las coordenadas finales de los cuadrantes en donde se van a tomar zonas homogéneas para
        calcular el estadístico de primer orden. Cada cuadrante se repite tantas veces como vaya a ser
        necesario samplearlo.
    original_image: np.ndarray
        Imagen original.
    ratio_imagen: np.ndarray
        Imagen que contiene el ratio pixel a pixel de la imagen original a la imagen filtrada.

    Returns
    -------
    r_ENL_mu: int
        Resultado del estadístico de primer orden.
    """
    suma = 0
    for i in range(len(cuads_i)):
        zona_xi, zona_yi, zona_xf, zona_yf = selecting_homogeneous_areas(cuads_i[i], cuads_f[i])
        zona_o = original_image[zona_xi:zona_xf, zona_yi:zona_yf]
        zona_r = ratio_image[zona_xi:zona_xf, zona_yi:zona_yf]

        mu_o = zona_o.mean()
        mu_r = zona_r.mean()
        std_o = zona_o.std()
        std_r = zona_r.std()

        ENL_o = mu_o**2 / std_o**2
        ENL_r = mu_r**2 / std_r**2
        r_ENL = np.abs(ENL_o - ENL_r) / ENL_o
        r_mu = np.abs(1-mu_r)

        suma += r_ENL + r_mu

    r_ENL_mu = suma / (2*len(cuads_i))
    return r_ENL_mu

def first_order_method(
    cuadrant_sizes: List[int],
    alphas: List[np.ndarray],
    inputs: np.ndarray,
    ratios: np.ndarray
) -> np.ndarray:
    """
    Aplica el método de primer orden a todas las imagenes del dataset para cuantificar la bondad del filtrado.
    Si una de las imágenes no tiene cuadrantes con alpha menor o igual a -6, entonces no se la tiene en cuenta.

    Parameters
    ----------
    cuadrant_sizes: List[int]
        Lista con los valores únicos de los tamaños de los cuadrantes. Sale del yaml de configuración
        como config.training.pixeles_cuad.
    alphas: List[np.ndarray]
        Lista donde cada elemento es un np.ndarray con los valores de alpha de cada cuadrante de cada imagen para
        un único tipo de partición. Las diferentes particiones van en diferentes elementos de la lista.
    inputs: np.ndarray
        Dataset de imágenes originales.
    ratios: np.ndarray
        Ratio de imágenes originales a imágenes filtradas. Debe tener la misma forma que inputs.

    Returns
    -------
    r_ENL_mu: np.ndarray
        Array unidimensional con los resultados del estadístico de primer orden para cada imagen.
    """
    assert min(cuadrant_sizes) >= 8, "Existen imágenes muy poco homogéneas en su dataset. Para poder \
    usar el filtro de primer orden se necesita que los cuadrantes de todas las imágenes sean de al menos 8x8 píxeles."

    cuadrantes = selecting_quadrants(alphas, M=5)
    cuadrantes_i, cuadrantes_f = quadrants_to_pixels(cuadrantes, cuadrant_sizes, alphas)

    estadisticos_1er_orden = []

    for i in range(inputs.shape[0]): # Loopeo por todas las imágenes
        if len(cuadrantes_i[i]) == 0:
            pass

        else:
            cuads_i = cuadrantes_i[i]
            cuads_f = cuadrantes_f[i]
            original_image = inputs[i]
            ratio_image = ratios[i]

            s = single_image_first_order_method(cuads_i, cuads_f, original_image, ratio_image)
            estadisticos_1er_orden.append(s)

    return np.array(estadisticos_1er_orden)

def co_ocurrence_matrix(
    image: np.ndarray
) -> np.ndarray:
    """
    Calcula la matriz de co-ocurrencias de una imagen. Lo hace para diferentes distancias y ángulos, y toma
    el promedio sobre todas esas combinaciones.

    Parameters
    ----------
    image: np.ndarray
        Imagen sobre la cual calcular la matriz de co-ocurrencias. Debe ser la imagen de ratio (entrada/output).

    Returns
    -------
    glcm_avg: np.ndarray
        Matriz de co-courrencia.
    """
    image_normalizada = (image - np.min(image)) / (np.max(image) - np.min(image))
    image = (image_normalizada * 255).astype(np.uint8)
    
    # Configurar distancias y ángulos
    distances = [1,2,3]  # Cantidad de vecinos (distancia)
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Ángulos en radianes
    
    # Calcular GLCM para todas las combinaciones de distancia y ángulo
    glcm = graycomatrix(image, distances=distances, angles=angles, symmetric=True, normed=True)
    
    # Promediar la GLCM sobre todas las combinaciones de distancia y ángulo
    glcm_avg = glcm.mean(axis=(2, 3))  # Promedio sobre los ejes de ángulo y distancia
    
    return glcm_avg

def h(
    p: np.ndarray
) -> float:
    """
    Valor de homogeneidad de Haralick calculado a partir de la matriz de co-ocurrencias.
    
    Parameters
    ----------
    p: np.ndarray
        Matriz de co-ocurrencias.

    Returns
    -------
    h: float
        Valor de homogeneidad de Haralick
    """
    if p.shape[0] != p.shape[1]:
        raise ValueError("La matriz de co-ocurrencias debe ser cuadrada.")
    
    M = p.shape[0]
    i = np.arange(M).reshape(-1, 1)
    j = np.arange(M).reshape(1, -1)
    
    weights = 1 / (1 + (i - j)**2)
    return np.sum(weights * p)

def deltah(
    image: np.ndarray,
    g: int = 30
) -> float:
    """
    El valor absoluto de la variación relativa de h0, en porcentaje.
    
    Parameters
    ----------
    image: np.ndarray
        Imagen individual sobre la cual calcular el valor de delta h. Debe ser la imagen de ratio (entrada/output).
    g: int
        Cantidad de permutaciones a tomar en el cálculo de delta h.

    Returns
    -------
    delta_h: float
        Valor de delta h.
    """
    hsum = 0
    for i in range(g):
        
        shuffled_flat = np.random.permutation(image.ravel())
        shuffled_arr = shuffled_flat.reshape(image.shape)
        glcm_avg_shuffled = co_ocurrence_matrix(shuffled_arr)
        hsum += h(glcm_avg_shuffled)

    havg = hsum / g
    h0 = h(co_ocurrence_matrix(image))
    delta_h = 100 * np.abs((h0 - havg) / h0)
    
    return delta_h

def second_order_method(
    ratios: np.ndarray,
    g: int = 30
) -> np.ndarray:
    """
    Aplica el método de segundo orden a todas las imagenes del dataset para cuantificar la bondad del filtrado.

    Parameters
    ----------
    ratios: np.ndarray
        Ratio de imágenes originales a imágenes filtradas.
    g: int
        Cantidad de permutaciones a tomar en el cálculo de delta h para una imagen individual.

    Returns
    -------
    deltah: np.ndarray
        Array unidimensional con los resultados del estadístico de segundo orden para cada imagen.
    """
    estadisticos_2do_orden = []
    
    for i in range(ratios.shape[0]): # Loopeo por todas las imágenes
        dh = deltah(ratios[i], g)
        estadisticos_2do_orden.append(dh)
        
        if i % 500 == 0:
            print(f"Procesadas {i} imágenes de {ratios.shape[0]}")
        
    return np.array(estadisticos_2do_orden)
