source("create_monthly_price.R")
source("model_selection_function.R")
source("ols_function.R") # add functions from the local R file named ols_function.R
source("t_test_function.R") # add functions from the local R file named t_test_function.R
library("quantmod") # add quantmod to the list of Packages


#########################################################################################################
# Fetch Data for Consumer price index and unemployment rate
#########################################################################################################

getSymbols(Symbols ="CPIAUCSL",src = "FRED")
getSymbols(Symbols ="GDP",src = "FRED")

cpi_all = as.matrix(CPIAUCSL[,1])
n_obs_cpi_all = dim(cpi_all)[1]
inflation_all = as.matrix(cpi_all[13:n_obs_cpi_all,1]/cpi_all[1:(n_obs_cpi_all-12),1] - 1) #generating the inflation

GDP_all = as.matrix(GDP[,1])
n_obs_GDP_all = dim(GDP_all)[1]
GDP_all = as.matrix(GDP_all[5:n_obs_GDP_all,1]/GDP_all[1:(n_obs_GDP_all-4),1] - 1) #generating the inflation

head(inflation_all)
tail(inflation_all)

head(GDP_all)
tail(GDP_all)

keep_data <- seq(from = as.Date("2000-01-01"), to = as.Date("2023-12-01"), by = "quarter")

inflation = as.matrix(inflation_all[as.Date(rownames(inflation_all)) %in% keep_data,])
GDP = as.matrix(GDP_all[as.Date(rownames(GDP_all)) %in% keep_data,])

n_obs = dim(inflation)[1]
date = as.Date(row.names(inflation))

plot(x = date, y = inflation,xlab='time',ylab='inflation',type='l',col="black")
plot(x = date, y = GDP,xlab='time',ylab='GDP',type='l',col="black")

acf(inflation)
acf(GDP)

##### select the number of lags for inflation and model checking ####

max_lags <- round(n_obs^(1/3))

sel_result_inflation <- model_selection(max_lags,inflation)
num_lag_inflation = sel_result_inflation$op_lag_AIC
lags_inflation = matrix(NA,nrow = n_obs, ncol = num_lag_inflation)
for (i in 1:num_lag_inflation) {
  lags_inflation[(i+1):n_obs,i] = as.matrix(inflation[1:(n_obs-i),1])
}
intercept = matrix(1,n_obs)
X = cbind(intercept,lags_inflation)
y = inflation
reg_result = ols(X[(num_lag_inflation+1):n_obs,],as.matrix(y[(num_lag_inflation+1):n_obs,1]))
residuals = reg_result$u_hat # get the AR model residuals
Box.test(residuals, lag = round(n_obs^(1/3)), type = "Ljung-Box") # applying Ljung and Box (1978) joint test of auto correlations

##### select the number of lags for unemployment and model checking ####
sel_result_GDP <- model_selection(max_lags,GDP)
num_lag_GDP = sel_result_GDP$op_lag_AIC

lags_GDP <- matrix(NA, nrow = n_obs, ncol = num_lag_GDP)
for (j in 1:num_lag_GDP) {
  lags_GDP[(j+1):n_obs,j] <- GDP[1:(n_obs-j),1] 
}
intercept = matrix(1,n_obs)
X = cbind(intercept,lags_GDP)
y = GDP
reg_result = ols(X[(num_lag_GDP+1):n_obs,],as.matrix(y[(num_lag_GDP+1):n_obs,1]))
residuals = reg_result$u_hat # get the AR model residuals
Box.test(residuals, lag = round(n_obs^(1/3)), type = "Ljung-Box") # applying Ljung and Box (1978) joint test of auto correlations

##### estimate the regression coefficients ####
intecept = matrix(1,nrow = n_obs, ncol = 1)
X_all <- cbind(inflation,lags_inflation,lags_GDP,intecept)
num_lag = max(num_lag_inflation,num_lag_GDP)
X_reg <- X_all[(num_lag+1):n_obs,]
y_reg <- as.matrix(GDP[(num_lag+1):n_obs,1])

est_result = ols(X_reg, y_reg)
beta_hat = est_result$beta_hat
#how much GDP growth related on past gdp growth
var_beta_hat = est_result$var_beta_hat

##### preform t-test ####

test_result = t_test(beta_hat,var_beta_hat)
t_stat <- test_result$t_stat
p_value <- test_result$p_value
