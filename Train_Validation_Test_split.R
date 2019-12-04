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
library("gbm") # For boosting algorithm
library("MASS") # For LDA
library(caret)
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
Cutoff <- quantile(Y$Y, probs = c(0.80)) # Cutoff point 
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
#rf.fit <- randomForest(Price_Category~.,data=housing_data, n.trees=500,na.action=na.roughfix)
#x11()
#varImpPlot(rf.fit) # shows which variables are important



# Removing all columns with Null values
X <- Filter(function(x)!any(is.na(x)), housing_data)
dim(X) #2919*46

# Dividing into test and train
set.seed(123)
fractionTraining   <- 0.60
fractionValidation <- 0.20
fractionTest       <- 0.20

# Compute sample sizes.
sampleSizeTraining   <- floor(fractionTraining   * nrow(X))
sampleSizeValidation <- floor(fractionValidation * nrow(X))
sampleSizeTest       <- floor(fractionTest       * nrow(X))

# Create the randomly-sampled indices for the dataframe. Use setdiff() to
# avoid overlapping subsets of indices.
indicesTraining    <- sort(sample(seq_len(nrow(X)), size=sampleSizeTraining))
indicesNotTraining <- setdiff(seq_len(nrow(X)), indicesTraining)
indicesValidation  <- sort(sample(indicesNotTraining, size=sampleSizeValidation))
indicesTest        <- setdiff(indicesNotTraining, indicesValidation)

# Finally, output the three dataframes for training, validation and test.
X_Train   <- X[indicesTraining, ]
X_Validation <- X[indicesValidation, ]
X_Test       <- X[indicesTest, ]


y_true_train <- as.numeric(as.factor(X_Train$Price_Category))-1
y_true_validation <- as.numeric(as.factor(X_Validation$Price_Category))-1
y_true_test <- as.numeric(as.factor(X_Test$Price_Category))-1


##################################################################################
### Fitting Logistic Regression
##################################################################################
X_Train$Price_Category <- as.factor(X_Train$Price_Category)
X_Validation$Price_Category <- as.factor(X_Validation$Price_Category)
X_Test$Price_Category <- as.factor(X_Test$Price_Category)
glm.fit <- glm(Price_Category~GrLivArea+Neighborhood+LotArea+X1stFlrSF+OverallQual+X2ndFlrSF+TotRmsAbvGrd,data=X_Train,family = "binomial")


# Predict
glm.probs.train <- predict(glm.fit, newdata = X_Train, type = "response")
y_hat_train_lr <- round(glm.probs.train)
glm.probs.validation <- predict(glm.fit, newdata = X_Validation, type = "response")
y_hat_validation_lr <- round(glm.probs.validation)
glm.probs.test <- predict(glm.fit, newdata = X_Test, type = "response")
y_hat_test_lr <- round(glm.probs.test)

#########################################
#  Calculate the error rates
########################################
train_err <- sum(abs(y_hat_train_lr- y_true_train))/length(y_true_train)
validation_err <- sum(abs(y_hat_validation_lr- y_true_validation))/length(y_true_validation)
test_err <- sum(abs(y_hat_test_lr- y_true_test))/length(y_true_test)

train_err #14.8%
validation_err #16.6%
test_err #12.99%


######################################################################################################
### LDA
######################################################################################################
lda.fit <- lda(Price_Category~GrLivArea+Neighborhood+LotArea+X1stFlrSF+OverallQual+X2ndFlrSF+TotRmsAbvGrd,data = X_Train)
names(lda.fit)
summary(lda.fit)


lda_pred_train <- predict(lda.fit, newdata = X_Train)
y_hat_train <- as.numeric(lda_pred_train$class)-1
lda_pred_validation <- predict(lda.fit, newdata = X_Validation)
y_hat_validation <- as.numeric(lda_pred_validation$class)-1
lda_pred_test <- predict(lda.fit, newdata = X_Test)
y_hat_test <- as.numeric(lda_pred_test$class)-1



#  Calculate the error rates
train_err <- sum(abs(y_hat_train- y_true_train))/length(y_true_train)
validation_err <- sum(abs(y_hat_validation- y_true_validation))/length(y_true_validation)
test_err <- sum(abs(y_hat_test- y_true_test))/length(y_true_test)

train_err #15.24%
validation_err #18.35%
test_err #13.84%

###########################################################################
Write to CSV

Compiled <- data.frame(y_true_train,y_hat_train_lr)

write.csv(MyData, file = "MyData.csv")
