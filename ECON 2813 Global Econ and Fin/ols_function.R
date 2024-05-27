ols <- function(X,y){
  n_obs <- dim(X)[1]
  p_obs <- dim(X)[2]
  beta <- solve(t(X) %*% X, t(X) %*% y)
  u_hat <- y - X %*% beta
  sigma2_u <- sum(u_hat^2)/(n_obs)
  beta_variance <- solve(t(X) %*% X)*sigma2_u
  obj = list(beta_hat = beta, var_beta_hat = beta_variance, u_hat = u_hat, sigma2_u = sigma2_u)
  return(obj)
}