### Always remember to load the libraries you will need
library(tidyverse)
library(jtools)

######### Consumption of Durable Goods Regressions

### load the data and create the data frame for prediction
Durable <- read_csv("C:/Users/lauraliu/Dropbox/teaching/2023Fall_MQE_Pitt/Classes/Class 9/DurableGoods.csv")
predictdata <- data.frame(I=35,F=3)
view(Durable)

## Income only regression
Reg.I <-lm(data=Durable, C ~ I)
summ(Reg.I,confidence=T, digits=3)
   #prediction
pred <- predict(Reg.I,predictdata,se.fit=T, interval = "prediction")
pred$SE.Fcst <- sqrt(pred$se.fit^2+pred$residual.scale^2)
Forecast.I <- c(Prediction=pred$fit[1],lower95=pred$fit[2],upper95=pred$fit[3],SE.pred=pred$SE.Fcst,SE.resid=pred$residual.scale)
print(Forecast.I,digits = 5)
   #histogram of residuals
hist(Reg.I$residuals, freq = FALSE, ylab="",xlab="Residuals", main = "Histogram of Residuals Model I", col=7)


## Family Size only regression
Reg.F <-lm(data=Durable, C ~ F)
summ(Reg.F,digits=3)
   #prediction
pred <- predict(Reg.F,predictdata,se.fit=T, interval = "prediction")
pred$SE.Fcst <- sqrt(pred$se.fit^2+pred$residual.scale^2)
Forecast.F <- c(Prediction=pred$fit[1],lower95=pred$fit[2],upper95=pred$fit[3],SE.pred=pred$SE.Fcst,SE.resid=pred$residual.scale)
print(Forecast.F,digits = 5)
   #histogram of residuals
hist(Reg.F$residuals, freq = FALSE, ylab="",xlab="Residuals", main = "Histogram of Residuals Model F", col=7)


## Income and Family Size regression
Reg.IF <-lm(data=Durable, C ~ I+F)
summ(Reg.IF,digits=3)
   #prediction
pred <- predict(Reg.IF,predictdata,se.fit=T, interval = "prediction")
pred$SE.Fcst <- sqrt(pred$se.fit^2+pred$residual.scale^2)
Forecast.IF <- c(Prediction=pred$fit[1],lower95=pred$fit[2],upper95=pred$fit[3],SE.pred=pred$SE.Fcst,SE.resid=pred$residual.scale)
print(Forecast.IF,digits = 5)
   #histogram of residuals
hist(Reg.IF$residuals, freq = FALSE, ylab="",xlab="Residuals", main = "Histogram of Residuals Model IF", col=7)

######### Dummy Variables with Gender Regressions

###### CPS Data
CPSSW8 <- read_csv("C:/Users/lauraliu/Dropbox/teaching/2023Fall_MQE_Pitt/Classes/Class 9/CPSSW8.csv")
View(CPSSW8)

## Age regression
Wage.age <-lm(data=CPSSW8, earnings ~ age)
summ(Wage.age,digits=3)
ggplot(CPSSW8, aes(y=earnings,x=age)) + geom_smooth(method = "lm", se=FALSE) + ylim(0,22)

## verify gender can be made into a factor variable
CPSSW8$gender.m <- factor(CPSSW8$gender)
##CPSSW8$gender.f <- relevel(CPSSW8$gender.m,ref="male")
is.factor(CPSSW8$gender.m)

## Male Dummy regression
Wage.gender <- lm(data=CPSSW8, earnings ~ gender)
summ(Wage.gender,digits=3)
ggplot(CPSSW8, aes(y=fitted.values(Wage.gender),x=age,color=gender)) + geom_line() + ylim(0,22)

## Age and Male Dummy regression
Wage.age.gender <- lm(data=CPSSW8, earnings ~ age+gender)
summ(Wage.age.gender,digits=3)
ggplot(CPSSW8, aes(y=fitted.values(Wage.age.gender),x=age,color=gender)) + geom_line() +labs(y= "earnings")

## Age and Male Dummy with interaction regression
Age.gender.int <- lm(data=CPSSW8, earnings ~ age+age*gender)
summ(Age.gender.int,digits=3)
ggplot(CPSSW8, aes(y=earnings, x=age,color=gender)) + geom_smooth(method = "lm",se=F)

## Prediction with Dummy Variables
predictdata <- data.frame(education=c(17,12),age=c(25,26),gender=c("female","male"))
pred <- predict(Age.gender.int,predictdata,se.fit=T, interval = "prediction")
pred

## Age and Region Dummies regression
Wage.age.region <- lm(data=CPSSW8, earnings ~ age+region)
summ(Wage.age.region,digits=3)

## Education, Age and Male Dummy with interaction regression
Edu.age.gender.int <- lm(data=CPSSW8, earnings ~ education+age+age*gender)
summ(Edu.age.gender.int,digits=3)
ggplot(CPSSW8, aes(y=earnings, x=age,color=gender)) + geom_smooth(method = "lm",se=F)
