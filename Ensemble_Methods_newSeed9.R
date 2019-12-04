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


# Loading Library
library(randomForest) 
library(gbm) # For boosting algorithm
library(MASS) # For LDA
library(DataExplorer) # For cool visualizations
library(gbm)
library(e1071)

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

housing_data <- rbind(housing_data_train , housing_data_test)
housing_data$Id <- NULL
housing_data$Price_Category <- Price_Category
dim(housing_data) #2919*80

# Check for missing values
count_na <- sapply(housing_data, function(x) sum(is.na(x)))
plot_missing(housing_data)
sort((count_na),decreasing = TRUE)

# Fitting Random Forest to check variable importance and make decision
housing_data$Price_Category=as.factor(housing_data$Price_Category)
rf_fit <- randomForest(Price_Category~.,data=housing_data, n.trees=500,na.action=na.roughfix)
#x11()
varImpPlot(rf_fit) # shows which variables are important

# Removing all columns with Null values
X <- Filter(function(x)!any(is.na(x)), housing_data)
dim(X) #2919*46

# Dividing into test and train
#set.seed(123)
set.seed(9)
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

y_true_train <- X_Train$Price_Category
y_true_train <- as.factor(y_true_train)
y_true_train <- as.numeric(y_true_train)-1

y_true_cv <- X_Validation$Price_Category
y_true_cv <- as.factor(y_true_cv)
y_true_cv <- as.numeric(y_true_cv)-1

y_true_test <- X_Test$Price_Category
y_true_test <- as.factor(y_true_test)
y_true_test <- as.numeric(y_true_test)-1

# Fitting Random Forest
rf_fit <- randomForest(Price_Category ~ ., data=X_Train, n.trees=500, type='class')

x11()
varImpPlot(rf_fit) # shows which variables are important
importance(rf_fit)

y_hat_train_rf <- predict(rf_fit, newdata= X_Train, type="class")
y_hat_train_rf <- as.numeric(y_hat_train_rf)-1

misclass_tree_rf_train <- sum(abs(y_true_train-y_hat_train_rf))/length(y_hat_train_rf)
misclass_tree_rf_train # 0

rf_train_accuracy <- 1 - misclass_tree_rf_train
rf_train_accuracy # 1

y_hat_cv_rf <- predict(rf_fit, newdata= X_Validation, type="class")
y_hat_cv_rf <- as.numeric(y_hat_cv_rf)-1

misclass_tree_rf_cv <- sum(abs(y_true_cv-y_hat_cv_rf))/length(y_hat_cv_rf)
misclass_tree_rf_cv # 0.1646655

rf_cv_accuracy <- 1 - misclass_tree_rf_cv
rf_cv_accuracy # 0.8353345

y_hat_test_rf <- predict(rf_fit, newdata= X_Test, type="class")
y_hat_test_rf <- as.numeric(y_hat_test_rf)-1

misclass_tree_rf_test <- sum(abs(y_true_test-y_hat_test_rf))/length(y_hat_test_rf)
misclass_tree_rf_test # 0.1384615

rf_test_accuracy <- 1 - misclass_tree_rf_test
rf_test_accuracy # 0.8615385

#########################################################
### SVM #########
#########################################################

# SVM with a linear kernel
svm_fit_linear <- tune(svm, Price_Category ~ ., data = X_Train, kernel = "linear",
                       ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10)))
svm_fit_linear
summary(svm_fit_linear)

best_mod_linear <- svm_fit_linear$best.model
best_mod_linear

# predict the test data
y_hat_train_linear_svm <- predict(best_mod_linear, newdata = X_Train)
y_hat_train_linear_svm <- as.numeric(y_hat_train_linear_svm)-1

svm_linear_train_error <- length(which(y_hat_train_linear_svm != y_true_train))/length(y_true_train)
svm_linear_train_error # 0.1267847

svm_linear_train_accuracy <- 1 - svm_linear_train_error
svm_linear_train_accuracy # 0.8732153

table(Prediction = y_hat_train_linear_svm, Actual = y_true_train)

y_hat_cv_linear_svm <- predict(best_mod_linear, newdata = X_Validation)
y_hat_cv_linear_svm <- as.numeric(y_hat_cv_linear_svm)-1

svm_linear_cv_error <- length(which(y_hat_cv_linear_svm != y_true_cv))/length(y_true_cv)
svm_linear_cv_error # 0.1543739

svm_linear_cv_accuracy <- 1 - svm_linear_cv_error
svm_linear_cv_accuracy # 0.8456261

table(Prediction = y_hat_cv_linear_svm, Actual = y_true_cv)

y_hat_test_linear_svm <- predict(best_mod_linear, newdata = X_Test)
y_hat_test_linear_svm <- as.numeric(y_hat_test_linear_svm)-1

svm_linear_test_error <- length(which(y_hat_test_linear_svm != y_true_test))/length(y_true_test)
svm_linear_test_error # 0.1213675

svm_linear_test_accuracy <- 1 - svm_linear_test_error
svm_linear_test_accuracy # 0.8786325

table(Prediction = y_hat_test_linear_svm, Actual = y_true_test)

# SVM with a radial kernel
svm_fit_radial <- tune(svm, Price_Category~., data = X_Train, kernel = "radial",
                       ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10), gamma = c(0.5, 1, 2, 3, 4)))
svm_fit_radial
summary(svm_fit_radial)

best_mod_radial <- svm_fit_radial$best.model
best_mod_radial

# predict the test data
y_hat_train_radial_svm <- predict(best_mod_radial, newdata = X_Train)
y_hat_train_radial_svm <- as.numeric(y_hat_train_radial_svm)-1

svm_radial_train_error <- length(which(y_hat_train_radial_svm != y_true_train))/length(y_true_train)
svm_radial_train_error # 0.001142204

svm_radial_train_accuracy <- 1 - svm_radial_train_error
svm_radial_train_accuracy # 0.9988578

table(Prediction = y_hat_train_radial_svm, Actual = y_true_train)

y_hat_cv_radial_svm <- predict(best_mod_radial, newdata = X_Validation)
y_hat_cv_radial_svm <- as.numeric(y_hat_cv_radial_svm)-1

svm_radial_cv_error <- length(which(y_hat_cv_radial_svm != y_true_cv))/length(y_true_cv)
svm_radial_cv_error # 0.212693

svm_radial_cv_accuracy <- 1 - svm_radial_cv_error
svm_radial_cv_accuracy # 0.787307

table(Prediction = y_hat_cv_radial_svm, Actual = y_true_cv)

y_hat_test_radial_svm <- predict(best_mod_radial, newdata = X_Test)
y_hat_test_radial_svm <- as.numeric(y_hat_test_radial_svm)-1

svm_radial_test_error <- length(which(y_hat_test_radial_svm != y_true_test))/length(y_true_test)
svm_radial_test_error # 0.1897436

svm_radial_test_accuracy <- 1 - svm_radial_test_error
svm_radial_test_accuracy # 0.8102564

table(Prediction = y_hat_test_radial_svm, Actual = y_true_test)

###############################################################################
# Boosting
###############################################################################

# The below line changing Price category to numeric should only be run once or the subsequant results will be incorrect
# as it the Price_Category will
?gbm
X_Train$Price_Category <-  as.numeric(X_Train$Price_Category)-1
boost_fit <- gbm(Price_Category~.,data=X_Train, n.trees = 500, shrinkage = .1, interaction.depth = 3, 
                 distribution = "adaboost")
boost_fit2 <- gbm(Price_Category~.,data=X_Train, n.trees = 500, shrinkage = .6, interaction.depth = 3,
                  distribution = "adaboost")

# Look at error for shrinkage=0.1
y_hat_train_boost <- predict(boost_fit, newdata=X_Train, n.trees = 500, type = "response")
misclass_boost_train <- sum(abs(y_true_train-y_hat_train_boost))/length(y_hat_train_boost)
misclass_boost_train # 0.07878743

boost_train_accuracy <- 1 - misclass_boost_train
boost_train_accuracy # 0.9212126

y_hat_cv_boost <- predict(boost_fit, newdata=X_Validation, n.trees = 500, type = "response")
misclass_boost_cv <- sum(abs(y_true_cv-y_hat_cv_boost))/length(y_hat_cv_boost)
misclass_boost_cv # 0.1882814

boost_cv_accuracy <- 1 - misclass_boost_cv
boost_cv_accuracy # 0.8117186

y_hat_test_boost <- predict(boost_fit, newdata=X_Test, n.trees = 500, type = "response")
misclass_boost_test <- sum(abs(y_true_test-y_hat_test_boost))/length(y_hat_test_boost)
misclass_boost_test # 0.1474642

boost_test_accuracy <- 1 - misclass_boost_test
boost_test_accuracy # 0.8525358

# Look at error for shrinkage=0.6
y_hat_train_boost2 <- predict(boost_fit2, newdata=X_Train, n.trees = 500, type = "response")
misclass_boost2_train <- sum(abs(y_true_train-y_hat_train_boost2))/length(y_hat_train_boost2)
misclass_boost2_train # 0.006271495

boost_train_accuracy2 <- 1 - misclass_boost2_train
boost_train_accuracy2 # 0.9937285

y_hat_cv_boost2 <- predict(boost_fit2, newdata=X_Validation, n.trees = 500, type = "response")
misclass_boost2_cv <- sum(abs(y_true_cv-y_hat_cv_boost2))/length(y_hat_cv_boost2)
misclass_boost2_cv # 0.1955198

boost_cv_accuracy2 <- 1 - misclass_boost2_cv
boost_cv_accuracy2 # 0.8044802

y_hat_test_boost2 <- predict(boost_fit2, newdata=X_Test, n.trees = 500, type = "response")
misclass_boost2_test <- sum(abs(y_true_test-y_hat_test_boost2))/length(y_hat_test_boost2)
misclass_boost2_test # 0.1560217

boost_test_accuracy2 <- 1 - misclass_boost2_test
boost_test_accuracy2 # 0.8439783

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

logistic_train_err <- sum(abs(y_hat_train_lr- y_true_train))/length(y_true_train)
logistic_train_err # 0.1484866

logistic_train_accuracy <- 1 - logistic_train_err
logistic_train_accuracy # 0.8515134

glm.probs.cv <- predict(glm.fit, newdata = X_Validation, type = "response")
y_hat_cv_lr <- round(glm.probs.cv)

logistic_cv_err <- sum(abs(y_hat_cv_lr- y_true_cv))/length(y_true_cv)
logistic_cv_err # 0.1663808

logistic_cv_accuracy <- 1 - logistic_cv_err
logistic_cv_accuracy # 0.8336192

glm.probs.test <- predict(glm.fit, newdata = X_Test, type = "response")
y_hat_test_lr <- round(glm.probs.test)

logistic_test_err <- sum(abs(y_hat_test_lr- y_true_test))/length(y_true_test)
logistic_test_err # 0.1299145

logistic_test_accuracy <- 1 - logistic_test_err
logistic_test_accuracy # 0.8700855

######################################################################################################
### LDA
######################################################################################################

lda.fit <- lda(Price_Category~GrLivArea+Neighborhood+LotArea+X1stFlrSF+OverallQual+X2ndFlrSF+TotRmsAbvGrd,data = X_Train)
names(lda.fit)
summary(lda.fit)

lda_pred_train <- predict(lda.fit, newdata = X_Train)
y_hat_train_lda <- as.numeric(lda_pred_train$class)-1
lda_pred_validation <- predict(lda.fit, newdata = X_Validation)
y_hat_cv_lda <- as.numeric(lda_pred_validation$class)-1
lda_pred_test <- predict(lda.fit, newdata = X_Test)
y_hat_test_lda <- as.numeric(lda_pred_test$class)-1

#  Calculate the error rates
lda_train_err <- sum(abs(y_hat_train_lda- y_true_train))/length(y_true_train)
lda_train_err # 0.1524843

lda_train_accuracy <- 1 - lda_train_err
lda_train_accuracy # 0.8475157

lda_cv_err <- sum(abs(y_hat_cv_lda- y_true_cv))/length(y_true_cv)
lda_cv_err # 0.1835334

lda_cv_accuracy <- 1 - lda_cv_err
lda_cv_accuracy # 0.8164666

lda_test_err <- sum(abs(y_hat_test_lda- y_true_test))/length(y_true_test)
lda_test_err # 0.1384615

lda_test_accuracy <- 1 - lda_test_err
lda_test_accuracy # 0.8615385

train_df_var <- data.frame(GrLivArea = X_Train$GrLivArea, Neighborhood =X_Train$Neighborhood, LotArea=X_Train$LotArea, X1stFlrSF=X_Train$X1stFlrSF, OverallQual = X_Train$OverallQual ,y_true = y_true_train, y_rf = y_hat_train_rf, y_svm_linear = y_hat_train_linear_svm,
                       y_svm_radial = y_hat_train_radial_svm, y_lda = y_hat_train_lda, y_logistic = y_hat_train_lda,
                       y_boosting = round(y_hat_train_boost))
train_df_var

cv_df_var <- data.frame(GrLivArea = X_Validation$GrLivArea, Neighborhood =X_Validation$Neighborhood, LotArea=X_Validation$LotArea, X1stFlrSF=X_Validation$X1stFlrSF, OverallQual = X_Validation$OverallQual,y_true = y_true_cv, y_rf = y_hat_cv_rf, y_svm_linear = y_hat_cv_linear_svm,
                    y_svm_radial = y_hat_cv_radial_svm, y_lda = y_hat_cv_lda, y_logistic = y_hat_cv_lda,
                    y_boosting = round(y_hat_cv_boost))
cv_df_var

test_df_var <- data.frame(GrLivArea = X_Test$GrLivArea, Neighborhood =X_Test$Neighborhood, LotArea=X_Test$LotArea, X1stFlrSF=X_Test$X1stFlrSF, OverallQual = X_Test$OverallQual, y_true = y_true_test, y_rf = y_hat_test_rf, y_svm_linear = y_hat_test_linear_svm,
                      y_svm_radial = y_hat_test_radial_svm, y_lda = y_hat_test_lda, y_logistic = y_hat_test_lda,
                      y_boosting = round(y_hat_test_boost))
test_df_var

write.csv(train_df_var, "Stacking_Train.csv")
write.csv(cv_df_var, "Stacking_CV.csv")
write.csv(test_df_var, "Stacking_Test.csv")



############################################################################################
### Stacking
############################################################################################

train_ensemble <- read.csv(file ="Ensemble Train Data.csv", sep = ",",header = TRUE, stringsAsFactors = FALSE)
test_ensemble <- read.csv(file ="Ensemble Test Data.csv", sep = ",",header = TRUE, stringsAsFactors = FALSE)
cv_ensemble <- read.csv(file ="Ensemble CV Data.csv", sep = ",",header = TRUE, stringsAsFactors = FALSE)

# Removing the index column
train_ensemble$X <- NULL
test_ensemble$X <- NULL
cv_ensemble$X <- NULL

### Fitting Logistic Regression

# train_ensemble$y_true <- as.factor(train_ensemble$y_true)
# test_ensemble$y_true <- as.factor(test_ensemble$y_true)
# cv_ensemble$y_true <- as.factor(cv_ensemble$y_true)
glm.fit <- glm(y_true~.,data=train_ensemble,family = "binomial")


# Predict
glm.probs.train <- predict(glm.fit, newdata = train_ensemble, type = "response")
y_hat_train_lr <- round(glm.probs.train)

logistic_train_err <- sum(abs(y_hat_train_lr- train_ensemble$y_true))/length(y_hat_train_lr)
logistic_train_err # 0

logistic_train_accuracy <- 1 - logistic_train_err
logistic_train_accuracy # 1

glm.probs.cv <- predict(glm.fit, newdata = cv_ensemble, type = "response")
y_hat_cv_lr <- round(glm.probs.cv)

logistic_cv_err <- sum(abs(y_hat_cv_lr- cv_ensemble$y_true))/length(y_hat_cv_lr)
logistic_cv_err # 0.1646

logistic_cv_accuracy <- 1 - logistic_cv_err
logistic_cv_accuracy # 0.8353

glm.probs.test <- predict(glm.fit, newdata = test_ensemble, type = "response")
y_hat_test_lr <- round(glm.probs.test)

logistic_test_err <- sum(abs(y_hat_test_lr- test_ensemble$y_true))/length(y_hat_test_lr)
logistic_test_err # 0.1316

logistic_test_accuracy <- 1 - logistic_test_err
logistic_test_accuracy # 0.8683

# Beta estimates from Logistic Regression Stack 
#(Intercept)          y_rf  y_svm_linear  y_svm_radial         y_lda    y_logistic    y_boosting 
#-2.656607e+01  5.313214e+01  2.116646e-11 -1.333248e-09 -5.388857e-12            NA -2.239468e-11 

# NA as a coefficient in a regression indicates that the variable in question is linearly related to the other variables



# Trying KNN for stacking
y_true <- train_ensemble$y_true
KNN <- knn(train_ensemble, train_ensemble, y_true, 3, use.all = FALSE)


# Trying PCR for stacking
pcr.mod <- pcr(y_true~., data = train_ensemble, scale = TRUE, validation = "CV")
summary(pcr.mod)
names(pcr.mod)
coef(pcr.mod$finalModel)

# Plot the root mean squared error
validationplot(pcr.mod, val.type = "MSEP")
validationplot(pcr.mod, val.type = "R2")
# 90% variability in the data can be explained by using 9 PC

pred_pcr_test=predict(pcr.mod,test_ensemble,ncomp=2)

pred_pcr_test_class<-ifelse (pred_pcr<0.35,1,0)
sum(abs(pred_pcr_test_class-test_ensemble$y_true))/length(pred_pcr_test_class)

# 86.49%


pred_pcr_cv=predict(pcr.mod,cv_ensemble,ncomp=2)

pred_pcr_cv_class<-ifelse (pred_pcr_cv<0.5,1,0)
sum(abs(pred_pcr_cv_class-cv_ensemble$y_true))/length(pred_pcr_cv_class)


# 81.98%



  

