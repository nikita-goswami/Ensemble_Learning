#######################################################
# Statistical Data Mining I - EAS 506
# Capstone Project
#
# Exploring Ensemble methods for Classification
# Created: 11/24/2019
# Modified: 
#######################################################

rm(list = ls())

# set working directory
setwd("C:/UB/Sem 1/Stat Data Mining/Project/Data")


housing_data_train <- read.table(file ="train.csv", sep = ",",header = TRUE)
housing_data_test <- read.table(file ="test.csv", sep = ",",header = TRUE)
Y_train <- housing_data_train$SalePrice
Y_test <- read.table(file ="sample_submission.csv", sep = ",",header = TRUE)
