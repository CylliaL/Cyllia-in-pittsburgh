install.packages("tidyverse")
library(tidyverse)
library(dplyr)
###The following tries to avoid scientific notation
options(scipen = 1) 

### load the data (use correct directory)
housedata <- read.csv("housedata.csv")

### name the variables
Income <- housedata$Income
Price <- housedata$Price

### quick scatterplot
qplot(x=Income,y=Price)

### linear regression model
reg.1 <-lm(Price ~ Income)
print(summary(reg.1),digits=3)

### confidence intervals for intercept and slope
print(confint(reg.1,level = 0.95),digits=3)

### plot regression line and data
ggplot(housedata, aes(Income, Price)) + geom_point() + geom_smooth(method = "lm", se=FALSE)

### predictions/forecasts for income=50000,75000,1000000
## create the X values
newIncome <- data.frame(Income=c(50000,75000,1000000))
## Prediction with Confidence Intervals
predict(reg.1,newIncome, interval = "prediction", level=0.95)

### Calculate the SE of Prediction
## calculate from scratch as SE of line + SE of regression 
fcsts <- predict(reg.1,newIncome, interval = "prediction",se.fit =T, level=0.95)
sqrt((fcsts$se.fit)^2+(fcsts$residual.scale)^2)

####Pictures
# 1. Add predicted confidence band values 
pred.int <- predict(reg.1, interval = "prediction")
mydata <- cbind(housedata, pred.int)

# 2. Regression line + band + prediction interval 
ggplot(mydata, aes(Income, Price)) + geom_point() + geom_line(aes(y=fit),color="blue") +
  geom_smooth(method=lm,se=TRUE)+ geom_line(aes(y=lwr),color="red")+geom_line(aes(y=upr),color="red")




