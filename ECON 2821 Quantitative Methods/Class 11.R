### Always remember to load the libraries
library(tidyverse)
library(jtools)

######### Firm Sales Regressions

## load the data
FirmSales <- read_csv("C:/Users/lauraliu/Dropbox/teaching/2023Fall_MQE_Pitt/Classes/Class 11/FirmSales.csv")
view(FirmSales)

qplot(FirmSales$Year,FirmSales$Sales)

## Linear Time Trend: Sales vs. Year
Reg.SalesYear <- lm (data=FirmSales, Sales ~ Year)
summ(Reg.SalesYear,digits=3)
confint(Reg.SalesYear,level=0.95)

# Create the data frame for prediction
predSales <- data.frame(Year=15)

pred <- c(predict(Reg.SalesYear,predSales,se.fit=T,level=0.95,interval = "prediction"),conf="95%")
pred$SE.Forecast <- sqrt(pred$se.fit^2+pred$residual.scale^2)
Forecasts <- data.frame (Year=predSales, Forecast=pred$fit[1], SE.Forecast=pred$SE.Forecast,Lower=pred$fit[2],Upper=pred$fit[3],Confidence=pred$conf)
print(Forecasts,digits=5)

ggplot(FirmSales, aes(y=Sales, x=Year)) + geom_smooth(method = "lm",se=F,size=2) + geom_point(size=3,col=2) +xlim(0,16) +ylim(-200,2000)

## logSales vs. Year
FirmSales$logSales <- log(FirmSales$Sales)
view(FirmSales)

Reg.logSalesYear <- lm (data=FirmSales, logSales ~ Year)
summ(Reg.logSalesYear,digits=3)
confint(Reg.logSalesYear,level=0.95)

pred <- c(predict(Reg.logSalesYear,predSales,se.fit=T,level=0.95,interval = "prediction"),conf="95%")
pred$SE.Forecast <- sqrt(pred$se.fit^2+pred$residual.scale^2)
Forecasts <- data.frame (Year=predSales, Forecast=pred$fit[1], SE.Forecast=pred$SE.Forecast,Lower=pred$fit[2],Upper=pred$fit[3],Confidence=pred$conf)
print(Forecasts,digits=5)
exp(Forecasts$Forecast)
exp(Forecasts$Lower)
exp(Forecasts$Upper)

ggplot(FirmSales, aes(y=logSales, x=Year)) + geom_smooth(method = "lm",se=F,size=2) + geom_point(size=3,col=2) +xlim(0,16)

## Sales vs. lagSales
FirmSales$lagSales <- lag(FirmSales$Sales)
view(FirmSales)

predSales <- data.frame(lagSales=FirmSales$Sales[14])

Reg.lagSales <- lm (data=FirmSales, Sales ~ lagSales)
summ(Reg.lagSales,digits=3)
confint(Reg.lagSales,level=0.95)

pred <- c(predict(Reg.lagSales,predSales,se.fit=T,level=0.95,interval = "prediction"),conf="95%")
pred$SE.Forecast <- sqrt(pred$se.fit^2+pred$residual.scale^2)
Forecasts <- data.frame (lagSales=predSales,Forecast=pred$fit[1], SE.Forecast=pred$SE.Forecast,Lower=pred$fit[2],Upper=pred$fit[3],Confidence=pred$conf)
print(Forecasts,digits=5)

ggplot(FirmSales, aes(y=Sales, x=lagSales)) + geom_smooth(method = "lm",se=F,size=2) + geom_point(size=3,col=2)

## logSales vs. laglogSales
FirmSales$laglogSales <- lag(FirmSales$logSales)

predSales <- data.frame(laglogSales=FirmSales$logSales[14])

Reg.laglogSales <- lm (data=FirmSales, logSales ~ laglogSales)
summ(Reg.laglogSales,digits=3)
confint(Reg.laglogSales,level=0.95)

pred <- c(predict(Reg.laglogSales,predSales,se.fit=T,level=0.95,interval = "prediction"),conf="95%")
pred$SE.Forecast <- sqrt(pred$se.fit^2+pred$residual.scale^2)
Forecasts <- data.frame (lagSales=predSales,Forecast=pred$fit[1], SE.Forecast=pred$SE.Forecast,Lower=pred$fit[2],Upper=pred$fit[3],Confidence=pred$conf)
print(Forecasts,digits=5)
exp(Forecasts$Forecast)
exp(Forecasts$Lower)
exp(Forecasts$Upper)

ggplot(FirmSales, aes(y=logSales, x=laglogSales)) + geom_smooth(method = "lm",se=F,size=2) + geom_point(size=3,col=2)

######### Computer Sales at Campus Store Regressions

## load the data
Computer <- read_csv("C:/Users/lauraliu/Dropbox/teaching/2023Fall_MQE_Pitt/Classes/Class 11/ComputerSales.csv")
view(Computer)

qplot(Computer$Quarter,Computer$Sales)

## Linear Time Trend: Sales vs. Quarter
Reg.ComputerQuarter <- lm (data=Computer, Sales ~ Quarter)
summ(Reg.ComputerQuarter,digits=3)
confint(Reg.ComputerQuarter,level=0.95)

predComputer <- data.frame(Quarter=max(Computer$Quarter)+1)
pred <- c(predict(Reg.ComputerQuarter,predComputer,se.fit=T,level=0.95,interval = "prediction"),conf="95%")
pred$SE.Forecast <- sqrt(pred$se.fit^2+pred$residual.scale^2)
Forecasts <- data.frame (Quarter=predComputer, Forecast=pred$fit[1], SE.Forecast=pred$SE.Forecast,Lower=pred$fit[2],Upper=pred$fit[3],Confidence=pred$conf)
print(Forecasts,digits=5)

ggplot(Computer, aes(y=Sales, x=Quarter)) + geom_smooth(method = "lm",se=F,size=2) + geom_point(size=3,col=2) +xlim(0,45)

## Seasonality: Sales vs. Dummies and Quarter
Computer$Season <- relevel(factor(Computer$Season), ref="summer")
Reg.ComputerQuarterSeason <- lm (data=Computer, Sales ~ Season+Quarter)
summ(Reg.ComputerQuarterSeason,digits=3)
confint(Reg.ComputerQuarterSeason,level=0.95)

predComputer <- data.frame(Quarter=max(Computer$Quarter)+1,Season="fall")
pred <- c(predict(Reg.ComputerQuarterSeason,predComputer,se.fit=T,level=0.95,interval = "prediction"),conf="95%")
pred$SE.Forecast <- sqrt(pred$se.fit^2+pred$residual.scale^2)
Forecasts <- data.frame (Quarter=predComputer,Forecast=pred$fit[1], SE.Forecast=pred$SE.Forecast,Lower=pred$fit[2],Upper=pred$fit[3],Confidence=pred$conf)
print(Forecasts,digits=5)

Computer = cbind(Computer, yhat = predict(Reg.ComputerQuarterSeason))
ggplot(data=Computer, mapping=aes(x=Quarter, y=Sales, color=Season)) + geom_point() + geom_line(mapping=aes(y=yhat),size=1.5)

ggplot(Computer, aes(y=Sales, x=Quarter, color=Season)) + geom_smooth(method = "lm",se=F,size=2) + geom_point(size=3,col=2) +xlim(0,45)

## Seasonality: Sales vs. Lagged Sales
# Create the lagged dependent variables
Computer$lag1Sales <- lag(Computer$Sales,n=1)
Computer$lag2Sales <- lag(Computer$Sales,n=2)
Computer$lag3Sales <- lag(Computer$Sales,n=3)
Computer$lag4Sales <- lag(Computer$Sales,n=4)
view(Computer)

# run the regression
Reg.ComputerLags <- lm (data=Computer, Sales ~ lag1Sales+lag2Sales+lag3Sales+lag4Sales)
summ(Reg.ComputerLags,digits=3)

predComputer <- data.frame(lag1Sales=Computer$Sales[44],
                           lag2Sales=Computer$Sales[43],
                           lag3Sales=Computer$Sales[42],
                           lag4Sales=Computer$Sales[41])
pred <- c(predict(Reg.ComputerLags,predComputer,se.fit=T,level=0.95,interval = "prediction"),conf="95%")
pred$SE.Forecast <- sqrt(pred$se.fit^2+pred$residual.scale^2)
Forecasts <- data.frame (Quarter=predComputer,Forecast=pred$fit[1], SE.Forecast=pred$SE.Forecast,Lower=pred$fit[2],Upper=pred$fit[3],Confidence=pred$conf)
print(Forecasts,digits=5)

## Only lag 4
Reg.ComputerLag4 <- lm (data=Computer, Sales ~ lag4Sales)
summ(Reg.ComputerLag4,digits=3)

ggplot(Computer, aes(y=Sales, x=lag4Sales)) + geom_smooth(method = "lm",se=F,size=2) + geom_point(size=3,col=2)

# Create the data for forecasting
predComputer <- data.frame(lag4Sales=Computer$Sales[41])
pred <- c(predict(Reg.ComputerLag4,predComputer,se.fit=T,level=0.95,interval = "prediction"),conf="95%")
pred$SE.Forecast <- sqrt(pred$se.fit^2+pred$residual.scale^2)
Forecasts <- data.frame (Quarter=predComputer,Forecast=pred$fit[1], SE.Forecast=pred$SE.Forecast,Lower=pred$fit[2],Upper=pred$fit[3],Confidence=pred$conf)
print(Forecasts,digits=5)

## Look at error autocorrelation of the residuals
acf(Reg.ComputerQuarter$residuals,lag.max = 5,plot=F)
acf(Reg.ComputerQuarterSeason$residuals,lag.max = 5,plot=F)
acf(Reg.ComputerLag4$residuals,lag.max = 5,plot=F)

acf(Reg.ComputerQuarter$residuals)
acf(Reg.ComputerQuarterSeason$residuals)
acf(Reg.ComputerLag4$residuals)
