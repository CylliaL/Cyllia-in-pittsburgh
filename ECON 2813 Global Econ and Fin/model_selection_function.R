model_selection <- function(max_lags,y,x = NaN){
  AIC <- matrix(NA, nrow = max_lags+1, ncol = 1) # plus one is for no lags
  BIC <- matrix(NA, nrow = max_lags+1, ncol = 1) # plus one is for no lags
  n_obs <- dim(y)[1]
  if (sum(!is.nan(x))!=0){
    p <- dim(x)[2] + 1 # plus one is for intercept
  } else {
    p <- 1 # intercept
  }
  lags_y <- matrix(NA, nrow = n_obs, ncol = max_lags)
  for (i in 1:max_lags) {
    lags_y[(i+1):n_obs,i] <- y[1:(n_obs-i),1] 
  }
  if (sum(!is.nan(x))!=0){
    intercept = matrix(1,n_obs)
    X_all <- cbind(intercept,x,lags_y)
  } else {
    intercept = matrix(1,n_obs)
    X_all <- cbind(intercept,lags_y)
  }
  X_reg <- X_all[(max_lags+1):n_obs,]
  y_reg <- as.matrix(y[(max_lags+1):n_obs,1])
  for (i in 0:max_lags) {
    n_param <- p+i
    est_result <- ols(X = as.matrix(X_reg[,1:n_param]), y_reg)
    sigma2_u = est_result$sigma2_u
    AIC[i+1,1] <- log(sigma2_u) + 2*(n_param-1)/n_obs
    BIC[i+1,1] <- log(sigma2_u) + log(n_obs)*(n_param-1)/n_obs
  }
  op_lag_AIC = which.min(AIC) -1 # minus one is for no lags
  op_lag_BIC = which.min(BIC) -1 # minus one is for no lags
  obj = list(AIC = AIC, BIC = BIC, op_lag_AIC = op_lag_AIC, op_lag_BIC = op_lag_BIC)
}
