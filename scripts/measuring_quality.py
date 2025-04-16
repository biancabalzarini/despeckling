import numpy as np
from typing import List, Tuple
import random
import copy


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

    Parameters:
    -----------
    coord_i: Tuple[int, int]
        Tupla con los valores de x e y (en píxeles) iniciales del cuadrante.
    coord_f: Tuple[int, int]
        Tupla con los valores de x e y (en píxeles) finales del cuadrante.

    Returns:
    --------
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
