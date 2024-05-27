generate_x <- function(TT, rho){
  # this function generate x_t = rho x_t-1 + u_t
  
  # TT: The number of observations
  # rho: determines the degree of serial correlation for x
  u <- as.matrix(rnorm(TT,mean = 0, sd = 1),nrow = TT, ncol = 1) # generate observations for u from standartd normal distribution
  x0 <- 0
  x <- matrix(NA, nrow = TT, ncol = 1)
  for (i in 1:TT) {
    if (i == 1) {
      x[i] <- rho * x0 + u[i]
    } else {
      x[i] <- rho * x[i-1] + u[i]
    }
  }
  return(x)
}

generate_y <- function(lam, x, beta){
  # this function generate y_t = lam y_t-1 + beta x_t + e_t 
  
  # lam: determines the degree of serial correlation for y  
  # x: a covariate of y
  # beta: coeficient of x 

  e <- as.matrix(rnorm(TT,mean = 0, sd = 1),nrow = TT, ncol = 1) # generate observations for e from standartd normal distribution
  y <- matrix(data = NA, nrow = TT, ncol = 1)
  y0 <- 0
  for (i in 1:TT) {
    if (i == 1) {
      y[i] <- lam * y0 +  x[i] * beta + e[i]
    } else {
      y[i] <- lam * y[i-1] + x[i] * beta + e[i]
    }
  }
  return(y)
}

