import numpy as np
from typing import List
import random


def selecting_homogeneous_areas(
    imagen: np.ndarray,
    alphas: np.ndarray,
    M: int
) -> np.ndarray:
    """
    Genera varias areas homogéneas (al azar) dentro de una imagen.

    Parameters:
    -----------
    imagen: np.ndarray
        Imagen sobre la cual se van a buscar areas homogeneas.
    alphas: np.ndarray
        Valores de alpha de cada cuadarante de la imagen.
    M: int
        Número de áreas homogéneas que se quieren buscar.

    Returns:
    --------
    areash: np.ndarray
        Array con cada una de las areas homogéneas seleccionadas.        
    """

def selecting_cuadrants(
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
