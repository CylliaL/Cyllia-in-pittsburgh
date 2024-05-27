rolling_window = function(y, init_win_len, pre_sel_num_lags=NA, num_step_ahead = 1,sel_method = 'aic'){
  num_forecasts = length(y) - init_win_len
  sel_num_lags =  matrix(NA, nrow = num_forecasts, ncol = 1)
  y_f = as.matrix(y[(init_win_len+1):length(y)])
  yhat_f = matrix(NA, nrow = num_forecasts, ncol = num_step_ahead)
  for (i in 1:num_forecasts) {
    trian_idx = init_win_len + i - 1
    n_obs = init_win_len
    intercept = matrix(1, nrow = n_obs)
    y_adj = as.matrix(y[i:trian_idx])
    if (is.na(pre_sel_num_lags)){
      if (sel_method == 'aic'){
        results <- model_selection(round(n_obs^(1/3)),y_adj)
        num_lags = results$op_lag_AIC
      } else if(sel_method == 'bic'){
        results <- model_selection(round(n_obs^(1/3)),y_adj)
        num_lags = results$op_lag_BIC
      }
    } else{
      num_lags = pre_sel_num_lags
    }
    if(num_lags==0){
      X_adj <- intercept
      x_reg <- X_adj[(num_lags+1):n_obs,]
      y_reg <- as.matrix(y_adj[(num_lags+1):n_obs,1])
      est_result <- ols(X = x_reg, y = y_reg)
      beta_hat = est_result$beta_hat
      sel_num_lags[i,1] = num_lags
      for (j in 1:num_step_ahead) {
        x_f = 1
        yhat_f[i,j] = x_f %*% beta_hat
      }
    } else {
      lags_y_adj <- matrix(NA, nrow = n_obs, ncol = num_lags)
      for (l in 1:num_lags) {
        lags_y_adj[(l+1):n_obs,l] <- y_adj[1:(n_obs-l),1] 
      }
      X_adj <- cbind(intercept,lags_y_adj)
      x_reg <- X_adj[(num_lags+1):n_obs,]
      y_reg <- as.matrix(y_adj[(num_lags+1):n_obs,1])
      est_result <- ols(X = x_reg, y = y_reg)
      beta_hat = est_result$beta_hat
      sel_num_lags[i,1] = num_lags
      for (j in 1:num_step_ahead) {
        if (i >= j) {
          if (j <= num_lags){
            if (j==1){
              x_f = cbind(1,t(y_adj[(n_obs-num_lags+1):n_obs]))
            } else {
              x_f = cbind(1,t(as.matrix(yhat_f[i,1:(j-1)])),t(as.matrix(y_adj[(n_obs-num_lags+j):n_obs])))
            }
          } else {
            x_f = cbind(1,t(as.matrix(yhat_f[i,(j-num_lags):(j-1)])))
          }
          yhat_f[i,j] = x_f %*% beta_hat
        }
      }
    }
  }
  obj = list(actual_value = y_f, forecast = yhat_f, sel_num_lags = sel_num_lags)
  return(obj)
}
