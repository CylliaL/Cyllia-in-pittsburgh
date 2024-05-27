library(tidyverse)
library(stargazer)
options(scipen = 1) 

### Load the same house pricing data as Class 7
getwd()
housedata <- read_csv("housedata.csv")
Income <- housedata$Income
Price <- housedata$Price

### Check for Linear relationship
qplot(x=Income,y=Price)
##ggplot(housedata, aes(Income, Price)) + geom_point() + geom_smooth(method = "lm", se=FALSE)

### Run the simple regression
reg.1 <-lm(data=housedata, Price ~ Income)
summary(reg.1)

### Construct the data needed for assumptions plots
housedata$predicted <- predict(reg.1)   # Save the predicted values of y (this is yhat)
housedata$residuals <- residuals(reg.1) # Save the residuals as (this is y - yhat)
housedata$std_residuals <- (housedata$residuals - mean(housedata$residuals)) / sd(housedata$residuals) ## Standardize the residuals

### Bubbles plot where size and color of a bubble is related to size of residual
ggplot(housedata, aes(x = Income, y = Price)) +
  geom_smooth(method = "lm", se = FALSE, color = "lightgrey") +     # regression line  
  geom_segment(aes(xend = Income, yend = predicted), alpha = .2) +  # draw line from point to line
  geom_point(aes(color = abs(residuals), size = abs(residuals))) +  # size of the points
  scale_color_continuous(low = "green", high = "red") +             # colour of the points mapped to residual size - green smaller, red larger
  guides(color = "none", size ="none") +                             # Size legend removed
  geom_point(aes(y = predicted), shape = 1) +
  theme_bw()

### Histogram of Residuals with "perfect Normal" comparison
hist(housedata$residuals, freq = FALSE, ylab="",xlab="Residuals", main = "Histogram of Residuals", col=7,ylim=c(0,8e-06))
curve(dnorm(x, mean=mean(housedata$residuals), sd=sd(housedata$residuals)), col="darkblue", lwd=4, add=TRUE, yaxt="n")
### Standardized residuals are easier to look at
hist(housedata$std_residuals, freq = FALSE, col=7) 
curve(dnorm, add = TRUE, col="darkblue", lwd=4,ylim=0.5)

### Normality check plot with red line for theoretical comparison
qqnorm(housedata$residuals, ylab="Residual Quantiles")
qqline(housedata$residuals,col="red")

### All plots: 1. residuals vs. fitted plot, 2. QQ plot, 3. scale-location plot
plot(reg.1)

### ==================================================
### Financial Markets Regression
FMdata <- read_csv("C:/Users/lauraliu/Dropbox/teaching/2023Fall_MQE_Pitt/Classes/Class 8/beta_apple.csv")
ggplot(FMdata, aes(SP500,Apple)) + geom_point(col=6,size=3) + geom_smooth(method = "lm", se=FALSE)

FMreg.1 <-lm(data=FMdata, Apple ~ SP500)
print(summary(FMreg.1)$coefficients,digits=3)
stargazer(FMreg.1,type="text",single.row=T, intercept.bottom=F,star.cutoffs=NA,style="all")

print(confint.lm(FMreg.1),digits=1)

newSP500 <- data.frame(SP500=c(0.10))
fcsts <- predict(FMreg.1,newSP500, interval = "prediction",se.fit =T, level=0.95)
fcsts$se.fcst <- sqrt((fcsts$se.fit)^2+(fcsts$residual.scale)^2)
fcsts

### ==================================================
### CPS data multiple regressions
CPSSW8 <- read_csv("C:/Users/lauraliu/Dropbox/teaching/2023Fall_MQE_Pitt/Classes/Class 8/CPSSW8.csv")
View(CPSSW8)

Wage.reg.1 <-lm(data=CPSSW8, earnings ~ education + age + experience)
print(summary(Wage.reg.1),digits=3)

Wage.reg.1 <-lm(data=CPSSW8, earnings ~ education + age)
print(summary(Wage.reg.1),digits=3)
