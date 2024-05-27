generate_ar_data <- function(TT, ar_coefs, sigma = 1, mu = 1, seed = NULL) {
  if (!is.null(seed)) {
    set.seed(seed)
  }
  
  p <- length(ar_coefs)
  total_T <- TT + 100  # Total observations including the initial 100
  data <- matrix(NA,nrow = total_T, ncol = 1)
  
  # Initialize the first p values
  data[1:p,1] <- rnorm(p, mean = 0, sd = sigma)
  
  for (t in (p+1):total_T) {
    data[t,1] <- sum(ar_coefs * data[(t-1):(t-p)]) + rnorm(1, mean = 0, sd = sigma)
  }
  
  # Return only the last n observations
  return(as.matrix(data[(101):(total_T),1]))
}
