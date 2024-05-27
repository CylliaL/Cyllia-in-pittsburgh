# this function generate independent data for x and y, and test the null of H0: beta = 0 
# against H_a : beta ~= 0 in regression of y = beta * x + u 

gen_test <- function(TT, lam, rho, alpha = 0.05){
# TT: The number of observations
# lam: determines the degree of serial correlation for y  
# rho: determines the degree of serial correlation for x
# alpha: size of the t test
  
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

  ############################################################################
  # run the regression y on x
  ############################################################################
  beta_hat <- sum( x*y )/sum( x^2 ) # estimating the slope coefficient, beta, via OLS
  res <- y - beta_hat*x # computing the residuals
  sigma2_hat <- mean(res^2)
  se_beta <- sqrt(sigma2_hat /sum( x^2 )) # the estimated variance of beta_hat
  
  t_test = beta_hat/se_beta # computed t test statistics
  cv = qnorm(1- alpha/2, mean = 0, sd = 1) # critial value for alpha = 0.05
  obj = abs(t_test) > cv # takes the value of one of we reject the null and zero otherwise
  return(obj)
}
