### Sampling from the GI0 distribution
# n: cantidad de muestras
# p_alpha: el valor del parámetro alpha
# p_gamma: el valor del parámetro gamma
# p_Looks: la cantidad de Looks

def rGI0(
  n,
  p_alpha,
  p_gamma,
  p_Looks
):
  return np.random.gamma(p_Looks, 1/p_Looks, n) / np.random.gamma(-p_alpha, 1/p_gamma, n)
