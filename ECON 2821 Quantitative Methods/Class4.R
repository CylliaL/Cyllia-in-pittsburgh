library(tidyverse)
###############
### Additional Willingness To Pay Calculations
###############

##### Data

AWTP <- read.csv("AWTP_MQE2023.csv")
awtp <- AWTP$Q1

##### Calculate Sample size
n<-length(awtp)
n

##### Calculate Sample Mean
Xbar <- mean(awtp)
Xbar

##### Calculate Sample Variance
S2 <- var(awtp)
S2

##### Calculate Sample Standard Deviation
S <- sd(awtp)
S

##### Calculate Standard Error of Xbar
seXbar <- S/sqrt(n)
seXbar

###### Find a 95% confidence interval for the sample mean
##compute critical value for two-sided 95% confidence interval:
##need to use t-distribution with n-1 degrees of freedom, 
##so command qt for the .975 lowertail:
cv <- qt(.975, df=n-1)   
cv
## if we just want to use 1-2-3 rule, we could just set cv <- 2
## if we know the population standard deviation set cv <- qnorm(.975)
#compute two-sided confidence interval *c(-1,1) for +/-:
CI <- Xbar+cv*seXbar*c(-1,1)
CI
#
#alternatively, can use t.test command (but that is no fun):
t.test(awtp, conf.level=0.95)

###### What is the probability that average awtp is larger than $15
1-pnorm(15,Xbar,seXbar)


