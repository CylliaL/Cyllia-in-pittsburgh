t_test <- function(beta_hat,var_beta_hat){
  # computed t test statistics
  se_beta_hat <- as.matrix(diag(var_beta_hat)^(0.5))
  t <- beta_hat/se_beta_hat
  p_value = 2*(1 - pnorm(abs(t)))
  obj = list(t_stat = t, p_value = p_value)
  return(obj)
}