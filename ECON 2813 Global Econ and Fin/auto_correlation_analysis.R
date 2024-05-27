rm(list = ls())
setwd("E:/Pitts/ECON 2813")
pwd = getwd() # getting the path for present working directory (pwd)

##########################################################
# Adding required packages and functions
##########################################################

##install.packages("quantmod")

source("create_monthly_price.R")
source("create_monthly_return.R")
source("ols_function.R") # add functions from the local R file named ols_function.R
source("t_test_function.R") # add functions from the local R file named t_test_function.R
library("quantmod") # add quantmod to the list of Packages

##########################################################################################
# Testing for single and joint autocorrelation of Daily Return for Apple + AR(1) modeling
##########################################################################################

getSymbols(Symbols ="AAPL",src = "yahoo", from = '2000/01/01/') # Download daily data for Apple from Yahoo Finance

daily_price <- as.matrix(AAPL[,6])
daily_price_date = as.Date(row.names(daily_price))
n_obs_daily_price = length(daily_price_date) 
acf(daily_price,lag=round(n_obs_daily_price^(1/3))) # command to obtain sample ACF of the data

daily_return = diff(daily_price)/daily_price[1:n_obs_daily_price-1,1]
daily_return_date = daily_price_date[2:n_obs_daily_price]
n_obs_daily_return = length(daily_return_date) 
acf(daily_return,lag=round(n_obs_daily_return^(1/3)))

ACF=acf(daily_return,lag=round(n_obs_daily_return^(1/3)), plot = FALSE)
ACF$acf

#####################  individual test for autocorrelation  #####################
t_ratio <- ACF$acf[2]*sqrt(n_obs_daily_return) # individual test for autocorrelation
t_ratio

t_ratio <- ACF$acf[5]*sqrt(n_obs_daily_return)
t_ratio

#####################  joint test for autocorrelation  #####################
Box.test(daily_return, lag = round(n_obs_daily_return^(1/3)), type = "Ljung-Box") # applying Ljung and Box (1978) joint test of auto correlations
Box.test(daily_return, lag = round(n_obs_daily_return^(1/3)), type = "Box-Pierce") # applying Box and Pierce (1970) joint test of auto correlations

#####################  AR(1) model  #####################
lag_daily_return = rbind(NA, as.matrix(daily_return[1:(n_obs_daily_return-1),1]))
intercept = matrix(1,n_obs_daily_return)
X = cbind(intercept,lag_daily_return)
y = daily_return
reg_result = ols(X[2:n_obs_daily_return,],as.matrix(y[2:n_obs_daily_return,1]))
1 - sum(reg_result$u_hat^2)/sum(y^2)
beta_hat = reg_result$beta_hat
beta_hat
var_beta_hat = reg_result$var_beta_hat
test_result = t_test(beta_hat,var_beta_hat)
test_result$t_stat
test_result$p_value

#####################  Theoretical v.s. empirical ACF  #####################

ar_coeff <- as.numeric(beta_hat[2])
ma_coeff <- 0
TACF <- ARMAacf(ar_coeff, ma_coeff, lag.max = round(n_obs_daily_return^(1/3))) # command to obtain theorical ACF
plot(c(0:round(n_obs_daily_return^(1/3))),ACF$acf,type='l',xlab='Lag',ylab='ACF',ylim=c(-0.1,1))
lines(0:round(n_obs_daily_return^(1/3)),TACF,lty=2)
grid(nx = 4, ny = 4)

##########################################################################################
# Testing for single and joint autocorrelation of monthly Return for Apple + AR(1) modeling
##########################################################################################

monthly_price_data <- create_monthly_prices(daily_price_date,daily_price)
monthly_price = as.matrix(monthly_price_data$monthly_price)
monthly_price_date = monthly_price_data$monthly_date
n_obs_monthly_price = length(monthly_price_date)
acf(monthly_price,lag=round(n_obs_monthly_price^(1/3))) # command to obtain sample ACF of the data

monthly_return_data = create_monthly_return(daily_return_date,daily_return)
monthly_return = diff(monthly_price)/monthly_price[1:n_obs_monthly_price-1,1]
monthly_return_date = monthly_price_date[2:n_obs_monthly_price]
n_obs_monthly_return = length(monthly_return_date)

acf(monthly_return,lag=round(n_obs_monthly_return^(1/3))) # command to obtain sample ACF of the data

ACF=acf(monthly_return,lag=round(n_obs_monthly_return^(1/3)), plot = FALSE)
ACF$acf

#####################  individual test for autocorrelation  #####################
t_ratio <- ACF$acf[7]*sqrt(n_obs_monthly_return) # individual test for correlation between returns at months t and t-1  
t_ratio

#####################  joint test for autocorrelation  #####################
Box.test(monthly_return, lag = round(n_obs_monthly_return^(1/3)), type = "Ljung-Box") # applying Ljung and Box (1978) joint test of auto correlations
Box.test(monthly_return, lag = round(n_obs_monthly_return^(1/3)), type = "Box-Pierce") # applying Box and Pierce (1970) joint test of auto correlations

#####################  AR(1) model  #####################
lag_monthly_return = rbind(NA, as.matrix(monthly_return[1:(n_obs_monthly_return-1),1]))
intercept = matrix(1,n_obs_monthly_return)
X = cbind(intercept,lag_monthly_return)
y = monthly_return
reg_result = ols(X[2:n_obs_monthly_return,],as.matrix(y[2:n_obs_monthly_return,1]))
beta_hat = reg_result$beta_hat
beta_hat
var_beta_hat = reg_result$var_beta_hat
test_result = t_test(beta_hat,var_beta_hat)
test_result$t_stat
test_result$p_value

#####################  Theoretical v.s. empirical ACF  #####################
ar_coeff <- as.numeric(beta_hat[2])
ma_coeff <- 0
TACF <- ARMAacf(ar_coeff, ma_coeff, lag.max = round(n_obs_monthly_return^(1/3))) # command to obtain theorical ACF
plot(c(0:round(n_obs_monthly_return^(1/3))),ACF$acf,type='l',xlab='Lag',ylab='ACF',ylim=c(-0.1,1))
lines(0:round(n_obs_monthly_return^(1/3)),TACF,lty=2)
grid(nx = 4, ny = 4)

##########################################################################################
# Testing for single and joint autocorrelation of quarterly real GDP growth + AR(1) modeling
##########################################################################################
getSymbols(Symbols ="GDPC1",src = "FRED") # Download daily data for Apple from Yahoo Finance
quarterly_rgdp <- as.matrix(GDPC1[,1])
quarterly_rgdp_date = as.Date(row.names(quarterly_rgdp))
n_obs_quarterly_rgdp = length(quarterly_rgdp) 
plot(x = quarterly_rgdp_date, y = quarterly_rgdp,xlab='time',ylab='Real GDP',type='l',col="black")
acf(quarterly_rgdp,lag=round(n_obs_quarterly_rgdp^(1/3))) # command to obtain sample ACF of the data

q_rgdp_growth= diff(quarterly_rgdp)/quarterly_rgdp[1:n_obs_quarterly_rgdp-1,1]
q_rgdp_growth_date = quarterly_rgdp_date[2:n_obs_quarterly_rgdp]
n_obs_q_rgdp_growth_date = length(q_rgdp_growth_date) 
acf(q_rgdp_growth,lag=round(n_obs_q_rgdp_growth_date^(1/3))) # command to obtain sample ACF of the data

ACF=acf(q_rgdp_growth,lag=round(n_obs_q_rgdp_growth_date^(1/3)), plot = FALSE)
ACF$acf

#####################  individual test for autocorrelation  #####################
t_ratio <- ACF$acf[3]*sqrt(n_obs_q_rgdp_growth_date) # individual test for correlation between returns at months t and t-1  
t_ratio

#####################  joint test for autocorrelation  #####################
Box.test(monthly_return, lag = 4, type = "Ljung-Box") # applying Ljung and Box (1978) joint test of auto correlations
Box.test(monthly_return, lag = 4, type = "Box-Pierce") # applying Box and Pierce (1970) joint test of auto correlations

#####################  AR(1) model  #####################
lag_q_rgdp_growth = rbind(NA, as.matrix(q_rgdp_growth[1:(n_obs_q_rgdp_growth_date-1),1]))
intercept = matrix(1,n_obs_q_rgdp_growth_date)
X = cbind(intercept,lag_q_rgdp_growth)
y = q_rgdp_growth
reg_result = ols(X[2:n_obs_q_rgdp_growth_date,],as.matrix(y[2:n_obs_q_rgdp_growth_date,1]))
beta_hat = reg_result$beta_hat
var_beta_hat = reg_result$var_beta_hat
test_result = t_test(beta_hat,var_beta_hat)
test_result$t_stat
test_result$p_value

#####################  Theoretical v.s. empirical ACF  #####################
ar_coeff <- as.numeric(beta_hat[2])
ma_coeff <- 0
TACF <- ARMAacf(ar_coeff, ma_coeff, lag.max = round(n_obs_q_rgdp_growth_date^(1/3))) # command to obtain theorical ACF
plot(c(0:round(n_obs_q_rgdp_growth_date^(1/3))),ACF$acf,type='l',xlab='Lag',ylab='ACF',ylim=c(-0.1,1))
lines(0:round(n_obs_q_rgdp_growth_date^(1/3)),TACF,lty=2)
grid(nx = 4, ny = 4)

