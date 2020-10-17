#####
data_longitudinal <- read.csv("~/Desktop/GlucoNet/test_longitudinal")
data_reshaped <- read.csv("~/Desktop/GlucoNet/test_reshaped")
# le 29 osservazioni di differenza sono perse nel reshaping

#per ora eliminio la variabile dell'ora del giorno
data_longitudinal = subset(data_longitudinal, select = -datetime )
data_reshaped = subset(data_reshaped, select = -datetime )

#train test split
long_tr = data_longitudinal[1:2629,]
long_te = data_longitudinal[2630:3629,]

resh_tr = data_reshaped[1:2600,]
resh_te = data_reshaped[2601:3600,]

#verifica dell'indipendenza dei residui
model_resh <- lm(y ~ ., data = resh_tr)
model_resh
par(mfrow = c(2, 2))
plot(model_resh)

test_pred <- predict.lm(model_resh, resh_te)
residuals = resh_te[,73] - test_pred
par(mfrow = c(1, 1))
plot(test_pred, residuals)

plot(residuals)

library(forecast)
par(mfrow = c(1, 1))
Acf(residuals, lag.max=NULL, type="correlation", 
    plot=TRUE, main="Autocorrelation function of residual", ylim=NULL)


#####
library(tidyverse)
library(lubridate)
library(ranger)
library(MetricsWeighted) # AUC

# Import
raw <- read.csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv")

# Explore
str(raw)
head(raw)
summary(raw)
hist(raw$Temp, breaks = "FD")

# Prepare and add binary response
prep <- raw %>% 
  mutate(Date = ymd(Date),
         y = year(Date),
         m = month(Date),
         d = day(Date),
         increase = 0 + (Temp > lag(Temp)))

with(prep, table(y))
summary(prep)

# Plot full data -> year as seasonality
ggplot(data = prep, aes(x = Date, y = Temp))+
  geom_line(color = "#00AFBB", size = 2) +
  scale_x_date()

# No visible within year seasonality
prep %>% 
  filter(y == 1987) %>% 
  ggplot(aes(x = Date, y = Temp))+
  geom_line(color = "#00AFBB", size = 2) +
  scale_x_date()

# Add some lags and diffs & remove incomplete rows
prep <- prep %>% 
  mutate(lag1 = lag(Temp),
         lag2 = lag(Temp, 2L),
         lag3 = lag(Temp, 3L),
         dif1 = lag1 - lag2,
         dif2 = lag2 - lag3) %>% 
  filter(complete.cases(.))

# Train/valid split in blocks
valid <- prep %>% 
  filter(y == 1990)
train <- prep %>% 
  filter(y < 1990)

# Models
y <- "increase" # response
x <- c("lag1", "lag2", "lag3", "dif1", "dif2", "y", "m", "d") # covariables
form <- reformulate(x, y)

# Logistic model: Linear dependence between difs and lags
fit_glm <- glm(form, 
               data = train, 
               family = binomial()) 
summary(fit_glm)

# Random forest
fit_rf <- ranger(form, 
                 data = train,
                 seed = 345345, 
                 importance = "impurity", 
                 probability = TRUE)
fit_rf
barplot(-sort(-importance(fit_rf))) # Variable importance

# Evaluate on 1990 for glm by looking at ROC AUC
pred_glm <- predict(fit_glm, valid, type = "response")
AUC(valid[[y]], pred_glm) # 0.684 ROC AUC

# Then for rf
pred_rf <- predict(fit_rf, valid)$predictions[, 2]
AUC(valid[[y]], pred_rf)    # 0.702 ROC AUC

# view OOB residuals of rf within one month to see if structure is left over
random_month <- train %>% 
  mutate(residuals = increase - fit_rf$predictions[, 2]) %>% 
  filter(y == 1987, m == 3) 

ggplot(random_month, aes(x = Date, y = residuals))+
  geom_line(color = "#00AFBB", size = 2) +
  scale_x_date()

#il fatto che ci sia del residuo rimasto è positivo nel senso che la random forest ha lavorato
#preservando la struttura temporale nonostante la fase di bootstrap oppure
#è negativo perchè vuol dire che nel dataset è rimasta una dinamica non spiegata?

#cambia qualcosa se lo faccio per classificazione? Magari si...! 
#perchè è come se stessi classificando istanze separate?


#####
normal_test <- resh_te
resh_test <- resh_te[sample(nrow(resh_te)),]
model_resh <- lm(y ~ ., data = resh_tr)

test_pred <- predict.lm(model_resh, normal_test)
test_pred2 <- predict.lm(model_resh, resh_test)


#####
# default RF model
library(randomForest)
#set.seed(123)
rf <- randomForest(
  formula = y ~ .,
  data    = resh_tr
)

rf
prediction <-predict(rf, resh_te)
res <- prediction - resh_te[73]
res2 <- res^2
mean(unlist(res2))

resh_tr <- resh_tr[sample(nrow(resh_tr)),]
#set.seed(123)
rf <- randomForest(
  formula = y ~ .,
  data    = resh_tr
)

rf
prediction <-predict(rf, resh_te)
res <- prediction - resh_te[73]
res2 <- res^2
mean(unlist(res2))