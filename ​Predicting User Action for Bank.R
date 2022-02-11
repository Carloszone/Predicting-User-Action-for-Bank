#getwd()
#setwd("/Users/juliachu/6020")

# load libraries
library(lattice)
library(ggplot2)
library(tidyverse)
library(dplyr)
library(caret)
library(MASS)
library(vif)
library(car)
library(class)
library(randomForest)

#install.packages('fastDummies')
library(fastDummies)
#install.packages("corrplot")
library(corrplot)

# read data set from github
url <- 'https://media.githubusercontent.com/media/Carloszone/ALY-6020/master/Final%20proposal/bank.csv'
bank<- read.csv(url)


# check data set
summary(bank$deposit)
str(bank)
anyNA(bank) # no missinig value

# data processing
# encode categorical columns: binary to 0-1, other to dummy variable
bank$default <- ifelse(bank$default == "yes", 1, 0)
bank$housing <- ifelse(bank$housing == "yes", 1, 0)
bank$loan <- ifelse(bank$loan == "yes", 1, 0)
bank$deposit <- ifelse(bank$deposit == "yes", 1, 0)

bank <- bank[c(17,1:16)]

stringCol <- c('job', 'marital', 'education', 'contact', 'month', 'poutcome')
bank <- dummy_cols(bank, select_columns = stringCol, remove_selected_columns = TRUE)


# drop columns based on corr matrix
droplist <- c('poutcome_unknown', 'month_sep', 'contact_unknown', 'education_unknown',
              'marital_single', 'job_unknown')
bank = bank[,!colnames(bank) %in% droplist]


#Split the data into training and test data sets.
set.seed(1234)
index <- sample((1:nrow(bank)), round(0.8*nrow(bank)))
train_data <-bank[index,]
test_data <-bank[-index,]

######
# Logistic Regression
######
bank_logit <- glm(deposit ~.,family = binomial(link = 'logit'), data = train_data)

#check model metrix
summary(bank_logit)
summary(bank_logit)$coefficients
AIC(bank_logit)
vif(bank_logit)

# check accuracy
pred <- predict(bank_logit, test_data, type = 'response')
pred <- ifelse(pred >= 0.5, 1, 0)
confusionMatrix(as.factor(pred), as.factor(test_data$deposit))

##########
# KNN
##########

# scale data set
mean <- apply(train_data[,-1], 2, mean)
sd <- apply(train_data[,-1], 2, sd)

scale_train <- scale(train_data[,-1], center = mean, scale = sd)
scale_test <- scale(test_data[,-1], center = mean, scale = sd)

# store results based on different k
res <- c()
for(i in 1:100){
  knn_predict <- knn(train = scale_train, test = scale_test, cl = as.factor(train_data[,1]), k = i)
  res = append(res,confusionMatrix(knn_predict, as.factor(test_data$deposit))$overall["Accuracy"])
}

# plot and find the best k
k = c(1:100)
plot(x = k, y = res, type = 'l', lwd = 3) # best k is 10

res[10]

######
# Random Forest
######

#model building 
library(randomForest)
rf<-randomForest(deposit~.,data = train_data,importane = T, proximity = T, do.trace = 100)
#check basic information about model
rf
summary(rf)

#error rate in 0-500 trees
plot(rf)

#measure the importance of each variable to "deposit"
round(importance(rf), 2)

#prediction
result <- predict(rf, newdata = test_data)
result_Approved <- ifelse(result > 0.5, 1, 0)

#build confusion matrix
cm <- table(test_data$deposit, result_Approved, dnn = c("Actual", "Predict"))
cm

#correction rate
confusionMatrix(as.factor(result_Approved), as.factor(test_data$deposit))
