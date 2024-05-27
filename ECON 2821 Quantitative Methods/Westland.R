### Always remember to load the libraries you will need
library(tidyverse)
library(jtools)

####### Read the data file
Westland <- read_csv("C:/Users/lauraliu/Dropbox/teaching/2023Fall_MQE_Pitt/Classes/Class 13/Westland.csv")
view(Westland)

############## Regression 1: PriceSixMonths vs. Period
Reg.1 <- lm (data=Westland, PriceSixMonths ~ Period)
summ(Reg.1,digits=3)

predictPriceSixMonths <- data.frame(Period=17)
pred <- predict(Reg.1,predictPriceSixMonths,se.fit=T, interval = "prediction")
pred$SE.Fcst <- sqrt(pred$se.fit^2+pred$residual.scale^2)   # This computes SE_forecast
Forecast.Reg.1 <- c(Prediction=pred$fit[1],Lower95=pred$fit[2],Upper95=pred$fit[3], SE_pred=pred$SE.Fcst,Res.SE=pred$residual.scale)
print(Forecast.Reg.1,digits=3)

############## Regression 2: PriceSixMonths vs. Period + PriceNow
Reg.2 <- lm (data=Westland, PriceSixMonths ~ Period + PriceNow)
summ(Reg.2,digits=3)

predictPriceSixMonths <- data.frame(Period=17,PriceNow=Westland$PriceNow[17])
pred <- predict(Reg.2,predictPriceSixMonths,se.fit=T, interval = "prediction")
pred$SE.Fcst <- sqrt(pred$se.fit^2+pred$residual.scale^2)   # This computes SE_forecast
Forecast.Reg.2 <- c(Prediction=pred$fit[1],Lower95=pred$fit[2],Upper95=pred$fit[3], SE_pred=pred$SE.Fcst,Res.SE=pred$residual.scale)
print(Forecast.Reg.2,digits=3)

############## Regression 3: PriceSixMonths vs. Period + logPriceNow
Westland$logPriceNow = log(Westland$PriceNow)
Reg.3 <- lm (data=Westland, PriceSixMonths ~ Period + logPriceNow)
summ(Reg.3,digits=3)

predictPriceSixMonths <- data.frame(Period=17,logPriceNow=Westland$logPriceNow[17])
pred <- predict(Reg.3,predictPriceSixMonths,se.fit=T, interval = "prediction")
pred$SE.Fcst <- sqrt(pred$se.fit^2+pred$residual.scale^2)   # This computes SE_forecast
Forecast.Reg.3 <- c(Prediction=pred$fit[1],Lower95=pred$fit[2],Upper95=pred$fit[3], SE_pred=pred$SE.Fcst,Res.SE=pred$residual.scale)
print(Forecast.Reg.3,digits=3)

############## Regression 4: PriceSixMonths vs. Period + logPriceNow + laglogPriceSixMonthsSimilar
Westland$laglogPriceSixMonthsSimilar = lag(log(Westland$PriceSixMonthsSimilar))
Reg.4 <- lm (data=Westland, PriceSixMonths ~ Period + logPriceNow + laglogPriceSixMonthsSimilar)
summ(Reg.4,digits=3)

predictPriceSixMonths <- data.frame(Period=17,logPriceNow=Westland$logPriceNow[17],laglogPriceSixMonthsSimilar = Westland$laglogPriceSixMonthsSimilar[17])
pred <- predict(Reg.4,predictPriceSixMonths,se.fit=T, interval = "prediction")
pred$SE.Fcst <- sqrt(pred$se.fit^2+pred$residual.scale^2)   # This computes SE_forecast
Forecast.Reg.4 <- c(Prediction=pred$fit[1],Lower95=pred$fit[2],Upper95=pred$fit[3], SE_pred=pred$SE.Fcst,Res.SE=pred$residual.scale)
print(Forecast.Reg.4,digits=3)

############## Forecasts Comparison
##Regression 1
print(Forecast.Reg.1,digits=4)
##Regression 2
print(Forecast.Reg.2,digits=4)
##Regression 3
print(Forecast.Reg.3,digits=4)
##Regression 4
print(Forecast.Reg.4,digits=4)