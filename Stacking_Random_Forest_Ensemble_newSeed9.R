train_df
#######################################################
# Statistical Data Mining I - EAS 506
# Capstone Project
#
# Exploring Ensemble methods for Classification
# Created: 11/25/2019
# Modified: 
#######################################################

rm(list = ls())

# set working directory
setwd("C:/UB/Sem 1/Stat Data Mining/Project/Data")
# setwd("C:/Academics/1st Semester/Statistical Data Mining I/Project/Ensemble_Learning-master")

# Library
library(e1071)

# Data
Train_Stack <- read.table(file ="Stacking_Train.csv", sep = ",",header = TRUE)
CV_Stack <- read.table(file ="Stacking_CV.csv", sep = ",",header = TRUE)
Test_Stack <- read.table(file ="Stacking_Test.csv", sep = ",",header = TRUE)

# Removing the index column
Train_Stack$X <- NULL
CV_Stack$X <- NULL
Test_Stack$X <- NULL

# Make Response a factor variable
Train_Stack$y_true <- as.factor(Train_Stack$y_true)
Test_Stack$y_true <- as.factor(Test_Stack$y_true)
CV_Stack$y_true <- as.factor(CV_Stack$y_true)

# Build Random Forest
rf_fit <- randomForest(y_true ~ ., data=Test_Stack, n.trees=500, type='class')

x11()
varImpPlot(rf_fit) # shows which variables are important
importance(rf_fit)

y_hat_train_rf <- predict(rf_fit, newdata= Train_Stack, type="class")

misclass_tree <- mean(y_hat_train_rf != Train_Stack$y_true)
misclass_tree # 0.12

rf_train_accuracy <- 1 - misclass_tree
rf_train_accuracy # 0.87

#CV sample
CV_Stack <- rbind(Train_Stack[1, ] , CV_Stack)
CV_Stack <- CV_Stack[-1,]
y_hat_cv_rf <- predict(rf_fit, newdata= CV_Stack, type="class")

misclass_tree_rf_cv <- mean(y_hat_cv_rf != CV_Stack$y_true)
misclass_tree_rf_cv # 0.14

rf_cv_accuracy <- 1 - misclass_tree_rf_cv
rf_cv_accuracy # 0.8542

y_hat_test_rf <- predict(rf_fit, newdata= Test_Stack, type="class")


misclass_tree_rf_test <- mean(y_hat_test_rf != Test_Stack$y_true)
misclass_tree_rf_test # 0.011

rf_test_accuracy <- 1 - misclass_tree_rf_test
rf_test_accuracy # 0.988


