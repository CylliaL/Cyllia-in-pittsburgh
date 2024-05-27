library(tidyverse)

####### Question 3
## Xbar=250, S=20, n=1500, H0->mu=252
(254-252)/(20/sqrt(1500))  ## the t-test equals 3.872983
qt(0.95,1499)    ## 1.645871
qt(0.99,1499)    ## 2.328838
qt(0.999,1499)   ## 3.095678
## therefore one can reject the null with 95%, 99%, and 99.9% confidence

####### Question 4
AWTPdata <- read_csv("Digital_Services_AWTP.csv")
AWTPdata <- rename("Digital_Services_AWTP.csv",digitalservice='Digital service')

t.test(AWTPdata$AWTP,mu=36.5,alternative = "greater")
1-0.007445   ## 1-0.007445 = 0.992555
### p-value = 0.007445, so we can reject with confidence up to 0.992555

####### Question 5
###Select the Netflix observations
Netflix <- filter(Netflix, digitalservice == "Netflix")

###Test H0 -> mu=15
t.test(Netflix$AWTP,mu=15)  ## p-value = 0.006134 

####### Question 6
prop.test (c(6123, 4763), c(200000, 150000), conf.level =0.95)
## p-value = 0.05614 so we can reject with 90% confidence
