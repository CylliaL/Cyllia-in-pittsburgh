### Always remember to load the libraries you will need
library(tidyverse)
library(jtools)

######### Forecasters Reading Regressions
Forecasters <- read_csv("C:/Users/lauraliu/Dropbox/teaching/2023Fall_MQE_Pitt/Reports/Class 10 - Forecasters/forecastersdata.csv")
view(Forecasters)

### Rose Regression
Rose <- lm (data=Forecasters, `Actual Price` ~ Rose)
summary(Rose)
### Hazel Regression
Hazel <- lm (data=Forecasters, `Actual Price` ~ Hazel)
summary(Hazel)
### Taylor Regression
Taylor <- lm (data=Forecasters, `Actual Price` ~ Taylor)
summary(Taylor)
### Rose & Hazel Regression
Rose.Hazel <- lm (data=Forecasters, `Actual Price` ~ Rose+Hazel)
summary(Rose.Hazel)
### Rose & Taylor Regression
Rose.Taylor <- lm (data=Forecasters, `Actual Price` ~ Rose+Taylor)
summary(Rose.Taylor)
### Hazel & Taylor Regression
Hazel.Taylor <- lm (data=Forecasters, `Actual Price` ~ Hazel+Taylor)
summary(Hazel.Taylor)
### All Three Regression
All3 <- lm (data=Forecasters, `Actual Price` ~ Hazel+Rose+Taylor)
summary(All3)

### Build a Data Frame of Prediction Errors
Errors <- data.frame(Hazel.Errors = Forecasters$`Actual Price` - Forecasters$Hazel )
Errors$Rose.Errors <- Forecasters$`Actual Price` - Forecasters$Rose
Errors$Taylor.Errors <- Forecasters$`Actual Price` - Forecasters$Taylor
### Correlation Matrix for the Forecasters' Errors
cor(Errors)

### Create the data frame for prediction
newdata <- data.frame(Taylor=350,Hazel=320)
fcsts <- predict(Hazel.Taylor, newdata, interval="prediction", se.fit=T, level=0.95)
fcsts

#########                          #########
######### Apple Juice Regressions  #########
#########                          #########

### load the data
AppleJ <- read_csv("C:/Users/lauraliu/Dropbox/teaching/2023Fall_MQE_Pitt/Classes/Class 10/AppleJuice.csv")
view(AppleJ)

### Create the data frame for prediction
predAJ <- data.frame(Price=2.33)

qplot(AppleJ$Price,AppleJ$Sales)

## Sales vs. price
AJ.price <- lm (data=AppleJ, Sales ~ Price)
summ(AJ.price,digits=3)

pred <- c(predict(AJ.price,predAJ,se.fit=T,level=.95, interval = "prediction"),conf=0.95)
pred$SE.Forecast <- sqrt(pred$se.fit^2+pred$residual.scale^2)
Forecasts <- data.frame (predAJ, Forecast=pred$fit[1], SE.Forecast=pred$SE.Forecast,Lower=pred$fit[2],Upper=pred$fit[3],Confidence=pred$conf)
print(Forecasts,digits=5)

AppleJ$predicted <- predict(AJ.price)   # Save the predicted values
AppleJ$residuals <- residuals(AJ.price) # Save the residual values
AppleJ$std_residuals <- (AppleJ$residuals - mean(AppleJ$residuals)) / sd(AppleJ$residuals)

ggplot(AppleJ, aes(y=Sales,x=Price)) + geom_point() + geom_smooth(method = "lm", se=FALSE)

ggplot(AppleJ, aes(x = Price, y = Sales)) +
  geom_smooth(method = "lm", se = FALSE, color = "lightgrey") +     # regression line  
  geom_segment(aes(xend = Price, yend = predicted), alpha = .2) +  # draw line from point to line
  geom_point(aes(color = abs(residuals), size = abs(residuals))) +  # size of the points
  scale_color_continuous(low = "green", high = "red") +             # colour of the points mapped to residual size - green smaller, red larger
  guides(color = FALSE, size = FALSE) +                             # Size legend removed
  geom_point(aes(y = predicted), shape = 1) +
  theme_bw()

#### 1/Price as Independent Variable #####

AppleJ$InvPrice <- 1/(AppleJ$Price)
AJ.InvPrice <- lm (data=AppleJ, Sales ~ InvPrice)
summ(AJ.InvPrice,digits=3)

predAJ <- data.frame(InvPrice=c(1/2.33))

pred <- c(predict(AJ.InvPrice,predAJ,se.fit=T,level=.95, interval = "prediction"),conf=0.95)
pred$SE.Forecast <- sqrt(pred$se.fit^2+pred$residual.scale^2)
Forecasts <- data.frame (predAJ, Forecast=pred$fit[1], SE.Forecast=pred$SE.Forecast,Lower=pred$fit[2],Upper=pred$fit[3],Confidence=pred$conf)
print(Forecasts,digits=5)

ggplot(AppleJ, aes(y=Sales,x=InvPrice)) + geom_point() + geom_smooth(method = "lm", se=FALSE)


#### logSales as Dependent Variable #####

AppleJ$logSales <- log(AppleJ$Sales)
AJ.logSales <- lm (data=AppleJ, logSales ~ Price)
summ(AJ.logSales,digits=3)

predAJ <- data.frame(Price=c(2.33))

pred <- c(predict(AJ.logSales,predAJ,se.fit=T,level=.95, interval = "prediction"),conf=0.95)
pred$SE.Forecast <- sqrt(pred$se.fit^2+pred$residual.scale^2)
Forecasts <- data.frame (predAJ, Forecast=pred$fit[1], SE.Forecast=pred$SE.Forecast,Lower=pred$fit[2],Upper=pred$fit[3],Confidence=pred$conf)
print(Forecasts,digits=5)
exp(Forecasts$Forecast)
exp(Forecasts$Lower)
exp(Forecasts$Upper)

ggplot(AppleJ, aes(y=logSales,x=Price)) + geom_point() + geom_smooth(method = "lm", se=FALSE)

#### logSales vs logPrice #####

AppleJ$logPrice <- log(AppleJ$Price)
AJ.loglog <- lm (data=AppleJ, logSales ~ logPrice)
summ(AJ.loglog,digits=3)

predAJ <- data.frame(logPrice=c(log(2.33)))

pred <- c(predict(AJ.loglog,predAJ,se.fit=T,level=.95, interval = "prediction"),conf=0.95)
pred$SE.Forecast <- sqrt(pred$se.fit^2+pred$residual.scale^2)
Forecasts <- data.frame (predAJ, Forecast=pred$fit[1], SE.Forecast=pred$SE.Forecast,Lower=pred$fit[2],Upper=pred$fit[3],Confidence=pred$conf)
print(Forecasts,digits=4)
exp(Forecasts$Forecast)
exp(Forecasts$Lower)
exp(Forecasts$Upper)

ggplot(AppleJ, aes(y=logSales,x=logPrice)) + geom_point() + geom_smooth(method = "lm", se=FALSE)

plot(AJ.loglog)

########                                     #########
########  Batting Average and Age Regression #####
########                                     #########

MLB16.18 <- read_csv("C:/Users/lauraliu/Dropbox/teaching/2023Fall_MQE_Pitt/Classes/Class 10/MLB BA 2016-18.csv")

BA.reg1 <- lm (data=MLB16.18, BA ~ Age+Agesqrd)
summ(BA.reg1,digits=5)

ggplot(MLB16.18, aes(y=BA.reg1$fitted.values,x=Age)) +geom_line(size=2) + geom_point(col=2,size=5) + ggtitle("MLB Batting Average 2016-2018") +theme(plot.title = element_text(hjust = 0.5))


