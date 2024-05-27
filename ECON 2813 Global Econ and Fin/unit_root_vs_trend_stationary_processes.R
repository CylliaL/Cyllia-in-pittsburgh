rm(list = ls()) # clear the data environment
setwd("/Users/mahradvaghefi/Desktop/Econ_1160/Fall_2022/r_scripts") # set the working directory
library("fUnitRoots")

set.seed(124) # set the seed for generating data
TT <- 300 # number of observations
mu = 0.5 # intercept of AR models
lam = 1 # cofficient of the first lag of y_t
rho = 0 # cofficient of the first lag of z_t
alpha = 0.5 # cofficient of the trend
trend = 1:TT
e1 <- as.matrix(rnorm(TT,mean = 0, sd = 3),nrow = TT, ncol = 1) # generate observations for e1 from standartd normal distribution

##########################################################################
# generating observations for y_t = mu(1-lam) + lam y_t-1 + e_t with lam = 1
##########################################################################
y <- matrix(data = NA, nrow = TT, ncol = 1)
for (i in 1:TT) {
  if (i == 1) {
    y[i] <- mu*(1-lam) + e1[i]
  } else {
    y[i] <- mu*(1-lam) + lam * y[i-1] + e1[i]
  }
}
plot(1:TT,y,type = "l", xlab='time',ylab='y')

adfTest(y,lags=1,type=c("c"))

#############################################################################
# generating observations for z_t = mu(1-rho) + rho y_t-1 + e_t with |rho|<1
#############################################################################
z <- matrix(data = NA, nrow = TT, ncol = 1)
for (i in 1:TT) {
  if (i == 1) {
    z[i] <- y[1]*(1-rho) + e1[i]
  } else {
    z[i] <- y[1]*(1-rho) + rho * z[i-1] + e1[i]
  }
}
lines(1:TT, z,lty = 1,col="blue")

adfTest(z,lags=1,type=c("c"))

#######################################################################################
# generating observations for y_t = mu + alpha (1-lam)t + lam y_t-1 + e_t with lam = 1
#######################################################################################
y <- matrix(data = NA, nrow = TT, ncol = 1)
for (i in 1:TT) {
  if (i == 1) {
    y[i] <- mu + alpha*(1-lam)*trend[i] + e1[i]
  } else {
    y[i] <- mu + alpha*(1-lam)*trend[i] + lam * y[i-1] + e1[i]
  }
}
plot(1:TT,y,type = "l", xlab='time',ylab='y',ylim = c(-50,150))
adfTest(y,lags=1,type=c("ct"))

########################################################################################
# generating observations for z_t = mu + alpha (1-rho) t + rho z_t-1 + e_t with |rho|<1
########################################################################################

z <- matrix(data = NA, nrow = TT, ncol = 1)
for (i in 1:TT) {
  if (i == 1) {
    z[i] <- y[1] + alpha*(1-rho)*trend[i] + e1[i]
  } else {
    z[i] <- y[1] + alpha*(1-rho)*trend[i] + rho * z[i-1] + e1[i]
  }
}
lines(1:TT, z,lty = 1,col="blue")
adfTest(z,lags=1,type=c("ct"))
