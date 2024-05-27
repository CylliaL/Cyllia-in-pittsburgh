rm(list = ls()) # clear the data environment
setwd("/Users/mahradvaghefi/Desktop/Econ_1160/Fall_2022/r_scripts")

source("generate_plus_test.R") # add functions inside the local R file named generate_plus_test

TT <- 300 # number of observations
lam <- 0.999 # determines the degree of serial correlation for y
rho <- 0.999 # determines the degree of serial correlation for x
alpha <- 0.05 # size of the t test
nsim <- 10000 # number of simulations
reject_null <- matrix(FALSE,nrow = nsim, ncol = 1)
for (i in 1:nsim) {
  set.seed(124*i) # set the seed for generating data
  reject_null[i] <- gen_test(TT, lam, rho, alpha)
}
estimated_size <- sum(reject_null)/nsim

print(estimated_size)
