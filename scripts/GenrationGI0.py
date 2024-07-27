import numpy as np

def rGI0(
  n: int,
  p_alpha: float,
  p_gamma: float,
  p_Looks: int,
) -> np.ndarray:
  """Samplea de la distribución GI0.

  Parameters
  ----------
  n: int
      Cantidad de muestras.
  p_alpha: float
      El valor del parámetro alpha.
  p_gamma: float
      El valor del parámetro gamma.
  p_Looks: int
      Cantidad de Looks.

  Returns
  -------
  gI0: np.ndarray
      Sampleo de tamaño n de la distribución GI0.
  """

  gI0 = np.random.gamma(p_Looks, 1/p_Looks, n) / np.random.gamma(-p_alpha, 1/p_gamma, n)
  
  return gI0
