### Sampling from the GI0 distribution
# n: cantidad de muestras
# p_alpha: el valor del parámetro alpha
# p_gamma: el valor del parámetro gamma
# p_Looks: la cantidad de Looks
rGI0 <- function(n, p_alpha, p_gamma, p_Looks) {
  
  return(
    rgamma(n, p_Looks, p_Looks) / rgamma(n, -p_alpha, p_gamma)
  )
  
}