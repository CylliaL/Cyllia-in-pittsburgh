#rm(list = ls())
#setwd("/Users/mahradvaghefi/Dropbox/Econ_2813/r_scripts")
#pwd = getwd() # getting the path for present working directory (pwd)


# Adding required packages and functions
source("generate_ar_p_function.R")
source("model_selection_function.R")
source("ols_function.R") # add functions from the local R file named ols_function.R
source("t_test_function.R") # add functions from the local R file named t_test_function.R

# Setting the parameters for AR(p) process
TT <- 500
mu <- 1
ar_coefs <- c(0.7,0.2)
sigma <- 1
seed <- 1234

# generate data from AR(p) process
y <- generate_ar_data(TT, ar_coefs, mu, sigma, seed)
plot(1:TT, y, type='l',xlab='time',ylab='y')

# The coefficients of the lag polynomial (including the leading 1)
lag_poly <- c(1, -ar_coefs)

# Find the roots
roots <- polyroot(lag_poly)
roots

# Check if all roots are outside the unit circle
all(abs(roots) > 1)

# Plotting ACF and Partial ACF
acf(y,lag=round(TT^(1/3)))
pacf(y,lag=round(TT^(1/3)))

## testing for joint auto correlation
Box.test(y, lag = round(TT^(1/3)), type = "Ljung-Box")
Box.test(y, lag = round(TT^(1/3)), type = "Box-Pierce")

#Theoretical v.s. empirical ACF
ma_coeff <- 0
ACF = acf(y,lag=round(TT^(1/3)))
TACF <- ARMAacf(ar_coefs, ma_coeff, lag.max = round(TT^(1/3))) # command to obtain theorical ACF
plot(c(0:round(TT^(1/3))),ACF$acf,type='l',xlab='Lag',ylab='ACF',ylim=c(-1,1))
lines(0:round(TT^(1/3)),TACF,lty=2)
grid(nx = 4, ny = 4)


# finding the number of lags using AIC or BIC 
results <- model_selection(round(TT^(1/3)),y)
aic_values = results$AIC
bic_values = results$BIC
num_lags_aic = results$op_lag_AIC  
num_lags_bic = results$op_lag_BIC
num_lags_aic
num_lags_bic

# Estimating the AR model with selected number of lags
num_lags = num_lags_aic
lags_y = matrix(NA,nrow = TT, ncol = num_lags)
for (i in 1:num_lags) {
  lags_y[(i+1):TT,i] = as.matrix(y[1:(TT-i),1])
}
intercept = matrix(1,TT)
X = cbind(intercept,lags_y)
reg_result = ols(X[(num_lags+1):TT,],as.matrix(y[(num_lags+1):TT,1]))
beta_hat = reg_result$beta_hat
beta_hat
var_beta_hat = reg_result$var_beta_hat
test_result = t_test(beta_hat,var_beta_hat)
test_result$t_stat
test_result$p_value

#Theoretical v.s. empirical ACF for the AR model with selected number of lags
ar_coeff <- as.numeric(beta_hat[2:(num_lags+1)])
ma_coeff <- 0
ACF = acf(y,lag=round(TT^(1/3)))
TACF <- ARMAacf(ar_coeff, ma_coeff, lag.max = round(TT^(1/3))) # command to obtain theorical ACF
plot(c(0:round(TT^(1/3))),ACF$acf,type='l',xlab='Lag',ylab='ACF',ylim=c(-1,1))
lines(0:round(TT^(1/3)),TACF,lty=2)
grid(nx = 4, ny = 4)
