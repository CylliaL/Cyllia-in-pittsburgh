### Always remember to load the libraries you will need
library(tidyverse)
library(jtools)

####### Read the data file
Amazonl <- read_csv("AmazonQuarterlyRevenues_2023_Data.csv")
#view(Amazonl)

### Let's look at the data, and a linear regression model
ggplot(Amazonl, aes(y=QR, x=Period)) + geom_smooth(method = "lm",se=F,size=2.5) + geom_point(size=2,col=2)

### Create a few extra variables
## Create the log dependent variable
Amazonl$logQR <- log(Amazonl$QR)

## Create the lagged log dependent variables
Amazonl$lag1logQR <- lag(Amazonl$logQR,n=1)
Amazonl$lag2logQR <- lag(Amazonl$logQR,n=2)
Amazonl$lag3logQR <- lag(Amazonl$logQR,n=3)
Amazonl$lag4logQR <- lag(Amazonl$logQR,n=4)

## Make Fiscal into a factor (for use as a categorical variable) and assign Fiscal=3 as reference
Amazonl$Fiscal <- as.factor(Amazonl$FiscalQuarter)
Amazonl$Fiscal = relevel(Amazonl$Fiscal, ref=3)
is.factor(Amazonl$Fiscal)

############### Choose a shorter data set

Amazon <- filter(Amazonl,Period>24)
Amazon$Period <- Amazon$Period-24
### Let's look at the data, and a linear regression model
ggplot(Amazon, aes(y=QR, x=Period)) + geom_smooth(method = "lm",se=F,size=2.5) + geom_point(size=2,col=2)
view(Amazon)

### Create the data frame for predictions
n <- length(Amazon$QR)
predictQR <- data.frame(lag1logQR=Amazon$logQR[n],lag2logQR=Amazon$lag1logQR[n],lag3logQR=Amazon$lag2logQR[n],lag4logQR=Amazon$lag3logQR[n],Period=n+1,Fiscal="3")

#######
####### Given exponential growth focus on natural log of QR
#######

############## logQR vs. Period
logQR.Period <- lm (data=Amazon, logQR~Period)
summ(logQR.Period,digits=3)

####### Prediction in log and millions
pred <- predict(logQR.Period,predictQR,se.fit=T, interval = "prediction")
pred$SE.Fcst <- sqrt(pred$se.fit^2+pred$residual.scale^2)   # This computes SE_forecast
Forecast.logQR <- c(Prediction=pred$fit[1],Lower95=pred$fit[2],Upper95=pred$fit[3],SE_pred=pred$SE.Fcst,Res.SE=pred$residual.scale)
print(Forecast.logQR,digits=3)
Forecast.QR <- c(Prediction.mil=exp(pred$fit[1]),Lower95.mil=exp(pred$fit[2]),Upper95.mil=exp(pred$fit[3]),CIwidth=exp(pred$fit[3])-exp(pred$fit[2]))
print(Forecast.QR,digits=3)

#### Look at the data and regression
ggplot(Amazon, aes(y=logQR, x=Period)) + geom_smooth(method = "lm",se=F,size=2.5) + geom_point(size=2,col=2)

#### Look at error autocorrelation of the residuals
acf(logQR.Period$residuals,lag.max = 5,plot=F)

############## logQR vs. lag4logQR
logQR.lag4logQR <- lm (data=Amazon, logQR~lag4logQR)
summ(logQR.lag4logQR,digits=3)

####### Prediction in log and millions
pred <- predict(logQR.lag4logQR,predictQR,se.fit=T, interval = "prediction")
pred$SE.Fcst <- sqrt(pred$se.fit^2+pred$residual.scale^2)   # This computes SE_forecast
Forecast.logQR <- c(Prediction=pred$fit[1],Lower95=pred$fit[2],Upper95=pred$fit[3],SE_pred=pred$SE.Fcst,Res.SE=pred$residual.scale)
print(Forecast.logQR,digits=3)
Forecast.QR <- c(Prediction.mil=exp(pred$fit[1]),Lower95.mil=exp(pred$fit[2]),Upper95.mil=exp(pred$fit[3]),CIwidth=exp(pred$fit[3])-exp(pred$fit[2]))
print(Forecast.QR,digits=3)

#### Look at error autocorrelation of the residuals
acf(logQR.lag4logQR$residuals,lag.max = 5,plot=F)

#### Look at the data and regression
ggplot(Amazon, aes(y=logQR, x=lag4logQR)) + geom_smooth(method = "lm",se=F,size=2.5) + geom_point(size=2,col=2)

############## logQR vs. lag4logQR+Period
logQR.lag4logQR.Period <- lm (data=Amazon, logQR~lag4logQR+Period)
summ(logQR.lag4logQR.Period,digits=3)

####### Prediction in log and millions
pred <- predict(logQR.lag4logQR.Period,predictQR,se.fit=T, interval = "prediction")
pred$SE.Fcst <- sqrt(pred$se.fit^2+pred$residual.scale^2)   # This computes SE_forecast
Forecast.logQR <- c(Prediction=pred$fit[1],Lower95=pred$fit[2],Upper95=pred$fit[3],SE_pred=pred$SE.Fcst,Res.SE=pred$residual.scale)
print(Forecast.logQR,digits=3)
Forecast.QR <- c(Prediction.mil=exp(pred$fit[1]),Lower95.mil=exp(pred$fit[2]),Upper95.mil=exp(pred$fit[3]),CIwidth=exp(pred$fit[3])-exp(pred$fit[2]))
print(Forecast.QR,digits=3)

#### Look at error autocorrelation of the residuals
acf(logQR.lag4logQR.Period$residuals,lag.max = 5,plot=F)

############## logQR vs. lagslogQR
logQR.lagslogQR <- lm (data=Amazon, logQR~lag1logQR+lag2logQR+lag3logQR+lag4logQR)
summ(logQR.lagslogQR,digits=3)

####### Prediction in log and millions
pred <- predict(logQR.lagslogQR,predictQR,se.fit=T, interval = "prediction")
pred$SE.Fcst <- sqrt(pred$se.fit^2+pred$residual.scale^2)   # This computes SE_forecast
Forecast.logQR <- c(Prediction=pred$fit[1],Lower95=pred$fit[2],Upper95=pred$fit[3],SE_pred=pred$SE.Fcst,Res.SE=pred$residual.scale)
print(Forecast.logQR,digits=3)
Forecast.QR <- c(Prediction.mil=exp(pred$fit[1]),Lower95.mil=exp(pred$fit[2]),Upper95.mil=exp(pred$fit[3]),CIwidth=exp(pred$fit[3])-exp(pred$fit[2]))
print(Forecast.QR,digits=3)

#### Look at error autocorrelation of the residuals
acf(logQR.lagslogQR$residuals,lag.max = 5,plot=F)

############## logQR vs. lagslogQR + Period
logQR.lagslogQR.Period <- lm (data=Amazon, logQR~lag1logQR+lag2logQR+lag3logQR+lag4logQR+Period)
summ(logQR.lagslogQR.Period,digits=3)

####### Prediction in log and millions
pred <- predict(logQR.lagslogQR.Period,predictQR,se.fit=T, interval = "prediction")
pred$SE.Fcst <- sqrt(pred$se.fit^2+pred$residual.scale^2)   # This computes SE_forecast
Forecast.logQR <- c(Prediction=pred$fit[1],Lower95=pred$fit[2],Upper95=pred$fit[3],SE_pred=pred$SE.Fcst,Res.SE=pred$residual.scale)
print(Forecast.logQR,digits=3)
Forecast.QR <- c(Prediction.mil=exp(pred$fit[1]),Lower95.mil=exp(pred$fit[2]),Upper95.mil=exp(pred$fit[3]),CIwidth=exp(pred$fit[3])-exp(pred$fit[2]))
print(Forecast.QR,digits=3)

#### Look at error autocorrelation of the residuals
acf(logQR.lagslogQR.Period$residuals,lag.max = 5,plot=F)

############## logQR vs. lagslogQR + Period + Quarter Dummies
logQR.lagslogQR.Period.Fiscal <- lm (data=Amazon, logQR~lag1logQR+lag2logQR+lag3logQR+lag4logQR+Period+Fiscal)
summ(logQR.lagslogQR.Period.Fiscal,digits=3)

####### Prediction in log and millions
pred <- predict(logQR.lagslogQR.Period.Fiscal,predictQR,se.fit=T, interval = "prediction")
pred$SE.Fcst <- sqrt(pred$se.fit^2+pred$residual.scale^2)   # This computes SE_forecast
Forecast.logQR <- c(Prediction=pred$fit[1],Lower95=pred$fit[2],Upper95=pred$fit[3],SE_pred=pred$SE.Fcst,Res.SE=pred$residual.scale)
print(Forecast.logQR,digits=3)
Forecast.QR <- c(Prediction.mil=exp(pred$fit[1]),Lower95.mil=exp(pred$fit[2]),Upper95.mil=exp(pred$fit[3]),CIwidth=exp(pred$fit[3])-exp(pred$fit[2]))
print(Forecast.QR,digits=3)

#### Look at error autocorrelation of the residuals
acf(logQR.lagslogQR.Period.Fiscal$residuals,lag.max = 5,plot=F)

############## logQR vs. lagslogQR + Quarter Dummies + lag4 interactions
logQR.lagslogQR.lag4xFiscal <- lm (data=Amazon, logQR~lag1logQR+lag2logQR+lag3logQR+lag4logQR*Fiscal)
summ(logQR.lagslogQR.lag4xFiscal,digits=3)

####### Prediction in log and millions
pred <- predict(logQR.lagslogQR.lag4xFiscal,predictQR,se.fit=T, interval = "prediction")
pred$SE.Fcst <- sqrt(pred$se.fit^2+pred$residual.scale^2)   # This computes SE_forecast
Forecast.logQR <- c(Prediction=pred$fit[1],Lower95=pred$fit[2],Upper95=pred$fit[3],SE_pred=pred$SE.Fcst,Res.SE=pred$residual.scale)
print(Forecast.logQR,digits=3)
Forecast.QR <- c(Prediction.mil=exp(pred$fit[1]),Lower95.mil=exp(pred$fit[2]),Upper95.mil=exp(pred$fit[3]),CIwidth=exp(pred$fit[3])-exp(pred$fit[2]))
print(Forecast.QR,digits=3)

#### Look at error autocorrelation of the residuals
acf(logQR.lagslogQR.lag4xFiscal$residuals,lag.max = 5,plot=F)

############## logQR vs. lagslogQR + Quarter Dummies + lag4 interactions + Period
logQR.lagslogQR.lag4xFiscal.Period <- lm (data=Amazon, logQR~lag1logQR+lag2logQR+lag3logQR+lag4logQR*Fiscal+Period)
summ(logQR.lagslogQR.lag4xFiscal.Period,digits=3)

####### Prediction in log and millions
pred <- predict(logQR.lagslogQR.lag4xFiscal.Period,predictQR,se.fit=T, interval = "prediction")
pred$SE.Fcst <- sqrt(pred$se.fit^2+pred$residual.scale^2)   # This computes SE_forecast
Forecast.logQR <- c(Prediction=pred$fit[1],Lower95=pred$fit[2],Upper95=pred$fit[3],SE_pred=pred$SE.Fcst,Res.SE=pred$residual.scale)
print(Forecast.logQR,digits=3)
Forecast.QR <- c(Prediction.mil=exp(pred$fit[1]),Lower95.mil=exp(pred$fit[2]),Upper95.mil=exp(pred$fit[3]),CIwidth=exp(pred$fit[3])-exp(pred$fit[2]))
print(Forecast.QR,digits=3)

#### Look at error autocorrelation of the residuals
acf(logQR.lagslogQR.lag4xFiscal.Period$residuals,lag.max = 5,plot=F)

############## logQR vs. lagslogQR + Quarter Dummies + lag4 interactions + Period interactions
logQR.lagslogQR.lag4xFiscal.PeriodxFiscal <- lm (data=Amazon, logQR~lag1logQR+lag2logQR+lag3logQR+lag4logQR+lag4logQR:Fiscal+Period+Period:Fiscal)
summ(logQR.lagslogQR.lag4xFiscal.PeriodxFiscal,digits=3)

####### Prediction in log and millions
pred <- predict(logQR.lagslogQR.lag4xFiscal.PeriodxFiscal,predictQR,se.fit=T, interval = "prediction")
pred$SE.Fcst <- sqrt(pred$se.fit^2+pred$residual.scale^2)   # This computes SE_forecast
Forecast.logQR <- c(Prediction=pred$fit[1],Lower95=pred$fit[2],Upper95=pred$fit[3],SE_pred=pred$SE.Fcst,Res.SE=pred$residual.scale)
print(Forecast.logQR,digits=3)
Forecast.QR <- c(Prediction.mil=exp(pred$fit[1]),Lower95.mil=exp(pred$fit[2]),Upper95.mil=exp(pred$fit[3]),CIwidth=exp(pred$fit[3])-exp(pred$fit[2]))
print(Forecast.QR,digits=3)

#### Look at error autocorrelation of the residuals
acf(logQR.lagslogQR.lag4xFiscal.PeriodxFiscal$residuals,lag.max = 5,plot=F)

#### Calculate probability of growth > 10%
1-pnorm(log(1.1*Amazon$QR[n-3]),mean=pred$fit[1],sd=pred$SE.Fcst)


