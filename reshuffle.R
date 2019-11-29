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


# Loading Library
library("randomForest") 
library(DataExplorer) # For cool visualizations



# Loading Data and combining test and train data into one
housing_data_train <- read.table(file ="train.csv", sep = ",",header = TRUE)
housing_data_test <- read.table(file ="test.csv", sep = ",",header = TRUE)

Y_data <- read.table(file ="sample_submission.csv", sep = ",",header = TRUE)
Y_test <- as.integer(Y_data$SalePrice)

Y_train <- housing_data_train$SalePrice
housing_data_train$SalePrice <- NULL

Y <- c(Y_train,Y_test)
Y <- as.data.frame(Y)
# There are 1460 observations and 80 variables.
# The response variable is SalePrice

# Converting response variable,  from quantitative to categorical for classification 
Cutoff <- quantile(Y, probs = c(0.80)) # Cutoff point 
Price_Category <- ifelse(Y$Y<Cutoff, "No", "Yes")
# category_Price <- function(x) { if (x > Cutoff) 1 else 0 }
# Price_Category <- sapply(Y,category_Price) # For response variable of train data

housing_data <- rbind(housing_data_train , housing_data_test)
housing_data$Id <- NULL
housing_data$Price_Category <- Price_Category
dim(housing_data) #2919*80

# Check for missing values
count_na <- sapply(housing_data, function(x) sum(is.na(x)))
plot_missing(housing_data)
sort((count_na),decreasing = TRUE)

# Fitting Random Forest to check variable importance and make decision
rf.fit <- randomForest(Price_Category~.,data=housing_data, n.trees=500,na.action=na.roughfix)
x11()
varImpPlot(rf.fit) # shows which variables are important



# Removing all columns with Null values
X <- Filter(function(x)!any(is.na(x)), housing_data)
dim(X) #2919*46

# Dividing into test and train
set.seed(123)
index <- sample(1:nrow(X),0.30*nrow(X))
X_test <- X[index,]
X_train <- X[-index,]
y_true <- X_test$Price_Category
y_true <- as.factor(y_true)
y_true <- as.numeric(y_true)-1

# Fitting Random Forest
X_train$Price_Category=as.factor(X_train$Price_Category)
rf.fit <- randomForest(Price_Category~.,data=X_train, n.trees=500, type='class')

x11()
varImpPlot(rf.fit) # shows which variables are important
my_pred <- predict(rf.fit, newdata= X_test, type="class")
y_hat <- as.numeric(my_pred)-1

misclass_tree <- sum(abs(y_true-y_hat))/length(y_hat)
misclass_tree #0.1474286

