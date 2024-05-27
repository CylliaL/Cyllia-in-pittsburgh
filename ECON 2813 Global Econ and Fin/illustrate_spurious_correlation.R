rm(list = ls()) # clear the data environment
setwd("/Users/mahradvaghefi/Desktop/Econ_1160/Fall_2022/r_scripts/")

library("quantmod") # add quantmod to the list of Packages
source("generate_data_functions.R") # add functions from the local R file named generate_data_functions.R
source("ols_function.R") # add functions from the local R file named ols_function.R
source("t_test_function.R") # add functions from the local R file named t_test_function.R

##########################################################
# Visualize time dependence in fanancial asset prices
##########################################################
getSymbols(Symbols ="AAPL",src = "yahoo", from = '2000/01/01/') # Download daily data for Apple from Yahoo Finance
Apple_adj_price <- as.matrix(AAPL[,6])
date = as.Date(row.names(Apple_adj_price))
n_obs = dim(Apple_adj_price)[1]

getSymbols(Symbols ="CVX",src = "yahoo", from = '2000/01/01/') # Download daily data for Shell from Yahoo Finance
Chevron_adj_price <- as.matrix(CVX[,6])

par(mfcol = c(2,1)) # to have multiple figures in one picture 

plot(x = date, y = Apple_adj_price,xlab='time',ylab='Price Apple',type='l',col="black")
plot(x = date, y = Chevron_adj_price,xlab='time',ylab='Price Chevron',type='l',col="black")

intecept = matrix(1,n_obs)

reg_result = ols(X = cbind(intecept,Apple_adj_price),y = Chevron_adj_price)
beta_hat = reg_result$beta_hat
var_beta_hat = reg_result$var_beta_hat
test_result = t_test(beta_hat,var_beta_hat)
test_result$t_stat
test_result$p_value

lag_price_Chevron = Chevron_adj_price[1:n_obs-1]
Chevron_price = Chevron_adj_price[2:n_obs]
intecept = matrix(1,n_obs-1)
reg_result = ols(X = cbind(intecept,lag_price_Chevron),y = Chevron_price)
beta_hat = reg_result$beta_hat
var_beta_hat = reg_result$var_beta_hat
test_result = t_test(beta_hat,var_beta_hat)
test_result$t_stat
test_result$p_value

##########################################################
# Investigating spurious correlation in time series data
##########################################################

#set.seed(124) # set the seed for generating data
TT <- 300 # number of observations
lam = 0.99 # determines the degree of serial correlation for y
rho = 0.99 # determines the degree of serial correlation for x
alpha = 0.05 # size of the t test

e1 <- as.matrix(rnorm(TT,mean = 0, sd = 1),nrow = TT, ncol = 1) # generate observations for e1 from standartd normal distribution
e2 <- as.matrix(rnorm(TT,mean = 0, sd = 1),nrow = TT, ncol = 1) # generate observations for e2 from standartd normal distribution

##########################################################
# generating observations for y_t = lam y_t-1 + e_t 
##########################################################
y <- matrix(data = NA, nrow = TT, ncol = 1)
for (i in 1:TT) {
  if (i == 1) {
    y[i] <- e1[i]
  } else {
    y[i] <- lam * y[i-1] + e1[i]
  }
}
par(mfcol = c(2,1)) # to have multiple figures in one picture 
plot(1:TT,y,type = "l", xlab='time',ylab='y')
abline(a = 0, b = 0) # Add line with zero intercept and zero slope

##########################################################
# generating observations for x_t = rho y_t-1 + e_t 
##########################################################
x <- matrix(NA, nrow = TT, ncol = 1)
for (i in 1:TT) {
  if (i == 1) {
    x[i] <- e2[i]
  } else {
    x[i] <- rho * x[i-1] + e2[i]
  }
}
plot(1:TT,x,type = "l",xlab='time',ylab='x')
abline(a = 0, b = 0) # Add line with zero intercept and zero slope

############################################################################
# run the regression y on x
############################################################################
beta_hat <- sum( x*y )/sum( x^2 ) # estimating the slope coefficient, beta, via OLS
res <- y - beta_hat*x # computing the residuals
sigma2_hat <- mean(res^2)
se_beta <- sqrt(sigma2_hat /sum( x^2 )) # the estimated variance of beta_hat

t_test = beta_hat/se_beta # computed t test statistics
cv = qnorm(1- alpha/2, mean = 0, sd = 1) # critial value for alpha = 0.05
reject_null = abs(t_test) > cv # takes the value of one of we reject the null and zero otherwise

print(reject_null)
