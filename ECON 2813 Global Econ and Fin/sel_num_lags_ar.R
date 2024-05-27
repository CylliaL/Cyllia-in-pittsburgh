#rm(list = ls())
#setwd("/Users/mahradvaghefi/Desktop/Econ_1160/Fall_2022/r_scripts")
#pwd = getwd() # getting the path for present working directory (pwd)

##########################################################
# Adding required packages and functions
##########################################################

source("r_functions/create_monthly_price.R")
source("r_functions/model_selection_function.R")
source("r_functions/ols_function.R") # add functions from the local R file named ols_function.R
source("r_functions/t_test_function.R") # add functions from the local R file named t_test_function.R
library("quantmod") # add quantmod to the list of Packages

#########################################################################################################
# Brief analysis of the US Public Debt as Percent of Gross Domestic Product using AR model 
#########################################################################################################

#####################  Fetch and plot the data  #####################

getSymbols(Symbols ="GFDEGDQ188S",src = "FRED") # Download Quarterly data for total Public Debt as Percent of Gross Domestic Product
head(GFDEGDQ188S)
tail(GFDEGDQ188S)
public_debt <- as.matrix(GFDEGDQ188S[,1])
colnames(public_debt) <- "Public Debt"
public_debt_date = as.Date(row.names(public_debt))
n_obs_public_debt = length(public_debt) 
plot(x = public_debt_date, y = public_debt,xlab='time',ylab='Public Debt',type='l',col="black")
acf(public_debt,lag=n_obs_public_debt^(1/3)) # command to obtain sample ACF of the data
pacf(public_debt,lag=round(n_obs_public_debt^(1/3))) # command to obtain sample PACF of the data

#####################  create and plot the data for growth in public debt #####################

public_debt_growth= diff(public_debt)/public_debt[1:n_obs_public_debt-1,1]
public_debt_growth_date = public_debt_date[2:n_obs_public_debt]
n_obs_public_debt_growth = length(public_debt_growth_date)
plot(x = public_debt_growth_date, y = public_debt_growth,xlab='time',ylab='Public Debt Growth',type='l',col="black")
acf(public_debt_growth,lag=round(n_obs_public_debt_growth^(1/3))) # command to obtain sample ACF of the data

#####################  Estimating an AR(1) for growth in public debt #####################

num_lags = 1
lags_public_debt_growth = matrix(NA,nrow = n_obs_public_debt_growth, ncol = num_lags)
for (i in 1:num_lags) {
  lags_public_debt_growth[(i+1):n_obs_public_debt_growth,i] = as.matrix(public_debt_growth[1:(n_obs_public_debt_growth-i),1])
}
intercept = matrix(1,n_obs_public_debt_growth)
X = cbind(intercept,lags_public_debt_growth)
y = public_debt_growth
reg_result = ols(X[(num_lags+1):n_obs_public_debt_growth,],as.matrix(y[(num_lags+1):n_obs_public_debt_growth,1]))
beta_hat = reg_result$beta_hat
var_beta_hat = reg_result$var_beta_hat
test_result = t_test(beta_hat,var_beta_hat)
test_result$t_stat
test_result$p_value

#####################  Theoretical v.s. empirical ACF for the AR(1) #####################

ar_coeff <- as.numeric(beta_hat[2:(num_lags+1)])
ma_coeff <- 0
ACF = acf(public_debt_growth,lag=round(n_obs_public_debt_growth^(1/3)),plot = FALSE) # command to obtain sample ACF of the data
TACF <- ARMAacf(ar_coeff, ma_coeff, lag.max = round(n_obs_public_debt_growth^(1/3))) # command to obtain theorical ACF
plot(c(0:round(n_obs_public_debt_growth^(1/3))),ACF$acf,type='l',xlab='Lag',ylab='ACF',ylim=c(0,1))
lines(0:round(n_obs_public_debt_growth^(1/3)),TACF,lty=2)
grid(nx = 4, ny = 4)

#####################  Selecting the number of lags the AR model #####################

pacf(public_debt_growth,lag=round(n_obs_public_debt_growth^(1/3))) # command to obtain sample ACF of the data
results <- model_selection(n_obs_public_debt_growth^(1/3),public_debt_growth)
aic_values = results$AIC
bic_values = results$BIC
num_lags_aic = results$op_lag_AIC  
num_lags_bic = results$op_lag_BIC


#####################  Estimating the AR mod=el with selected number of lags #####################

num_lags = num_lags_aic
lags_public_debt_growth = matrix(NA,nrow = n_obs_public_debt_growth, ncol = num_lags)
for (i in 1:num_lags) {
  lags_public_debt_growth[(i+1):n_obs_public_debt_growth,i] = as.matrix(public_debt_growth[1:(n_obs_public_debt_growth-i),1])
}
intercept = matrix(1,n_obs_public_debt_growth)
X = cbind(intercept,lags_public_debt_growth)
y = public_debt_growth
reg_result = ols(X[(num_lags+1):n_obs_public_debt_growth,],as.matrix(y[(num_lags+1):n_obs_public_debt_growth,1]))
beta_hat = reg_result$beta_hat
var_beta_hat = reg_result$var_beta_hat
test_result = t_test(beta_hat,var_beta_hat)
test_result$t_stat
test_result$p_value

#####################  Theoretical v.s. empirical ACF for the AR model with selected number of lags #####################

ar_coeff <- as.numeric(beta_hat[2:(num_lags+1)])
ma_coeff <- 0
ACF = acf(public_debt_growth,lag=round(n_obs_public_debt_growth^(1/3)),plot = FALSE) # command to obtain sample ACF of the data
TACF <- ARMAacf(ar_coeff, ma_coeff, lag.max = round(n_obs_public_debt_growth^(1/3))) # command to obtain theorical ACF
plot(c(0:round(n_obs_public_debt_growth^(1/3))),ACF$acf,type='l',xlab='Lag',ylab='ACF',ylim=c(0,1))
lines(0:round(n_obs_public_debt_growth^(1/3)),TACF,lty=2)
grid(nx = 4, ny = 4)
