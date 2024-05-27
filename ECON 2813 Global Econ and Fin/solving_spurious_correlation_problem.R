rm(list = ls()) # clear the data environment
setwd("/Users/mahradvaghefi/Desktop/Econ_1160/Fall_2022/r_scripts")
pwd = getwd() # getting the path for present working directory (pwd)

##########################################################
# Adding required packages and functions
##########################################################
source("r_functions/generate_data_functions.R") # Generate data from AR and "Augmented AR model"
source("r_functions/model_selection_function.R") # function for model selection
source("r_functions/ols_function.R") # function for OLS estimation
source("r_functions/t_test_function.R") # fucntion for t test

##########################################################
# Setting parameters' value
##########################################################

TT <- 500 # number of observations
lam <- 0.9 # determines the degree of serial correlation for y
rho <- 0.9 # determines the degree of serial correlation for x
beta <- 0 # coefficient of x in dgp for y
nsim <- 10000 # number of simulations

reject_null <- matrix(FALSE,nrow = nsim, ncol = 1)

#############################################################################
# Starting for loop simulation to compute size including both lag of x and y 
#############################################################################

for (i in 1:nsim) {
  ##### Generating Data ####
  set.seed(126*i) # set the seed for generating data
  x <- generate_x(TT, rho)
  y <- generate_y(lam, x, beta)
  
  ##### estimate the regression coefficients ####
  y_reg = as.matrix(y[2:TT,1])
  intecept = matrix(1,nrow = TT-1, ncol = 1)
  X_reg = cbind(x[2:TT,1],y[1:TT-1,1],x[1:TT-1,1],intecept)
  est_result = ols(X_reg, y_reg)
  beta_hat = est_result$beta_hat
  var_beta_hat = est_result$var_beta_hat
  
  ##### perform t-test ####
  test_result = t_test(beta_hat,var_beta_hat)
  t_stat <- test_result$t_stat
  p_value <- test_result$p_value
  reject_null[i] <- (p_value < 0.05)[1]
}

estimated_size <- colSums(reject_null)/nsim

estimated_size

#########################################################################################
# Starting for loop simulation to compute size given that the number of lags are unknown
#########################################################################################

for (i in 1:nsim) {
  ##### Generating Data ####
  set.seed(126*i) # set the seed for generating data
  x_data <- generate_x(TT, rho)
  y_data <- generate_y(lam, x_data, beta)
  max_lags <- round(TT^(1/3))
  ##### select the number of lags for x and model checking ####
  sel_result_x <- model_selection(max_lags,x_data)
  num_lag_x = sel_result_x$op_lag_BIC
  lags_x = matrix(NA,nrow = TT, ncol = num_lag_x)
  for (i in 1:num_lag_x) {
    lags_x[(i+1):TT,i] = as.matrix(x_data[1:(TT-i),1])
  }
  intercept = matrix(1,TT)
  X = cbind(intercept,lags_x)
  y = x_data
  reg_result = ols(X[(num_lag_x+1):TT,],as.matrix(y[(num_lag_x+1):TT,1]))
  residuals = reg_result$u_hat # get the AR model residuals
  Box.test(residuals, lag = round(TT^(1/3)), type = "Ljung-Box") # applying Ljung and Box (1978) joint test of auto correlations
  
  ##### select the number of lags for y and model checking ####
  sel_result_y <- model_selection(max_lags,y_data)
  num_lag_y = sel_result_y$op_lag_BIC
  
  lags_y <- matrix(NA, nrow = TT, ncol = num_lag_y)
  for (j in 1:num_lag_y) {
    lags_y[(j+1):TT,j] <- y_data[1:(TT-j),1] 
  }
  intercept = matrix(1,TT)
  X = cbind(intercept,lags_y)
  y = y_data
  reg_result = ols(X[(num_lag_y+1):TT,],as.matrix(y[(num_lag_y+1):TT,1]))
  residuals = reg_result$u_hat # get the AR model residuals
  Box.test(residuals, lag = round(TT^(1/3)), type = "Ljung-Box") # applying Ljung and Box (1978) joint test of auto correlations
  
  ##### estimate the regression coefficients ####
  intecept = matrix(1,nrow = TT, ncol = 1)
  X_all <- cbind(x_data,lags_x,lags_y,intecept)
  num_lag = max(num_lag_x,num_lag_y)
  X_reg <- X_all[(num_lag+1):TT,]
  y_reg <- as.matrix(y_data[(num_lag+1):TT,1])
  
  est_result = ols(X_reg, y_reg)
  beta_hat = est_result$beta_hat
  var_beta_hat = est_result$var_beta_hat
  
  ##### preform t-test ####
  
  test_result = t_test(beta_hat,var_beta_hat)
  t_stat <- test_result$t_stat
  p_value <- test_result$p_value
  reject_null[i] <- (p_value < 0.05)[1]
}

estimated_size <- colSums(reject_null)/nsim

estimated_size
