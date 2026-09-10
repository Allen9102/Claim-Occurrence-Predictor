# Claim-Frequency-Predictor

An auto insurance claim frequency predictor
A team project to predict auto insurance claim occurrence by comparing machine learning models on structured policyholder data.

## Packages and Libraries
install.packages('randomForest')
install.packages('rpart.plot')
library(caret) # For stratified sampling and training XGboost
library(Matrix) # Create a sparse matrix for use by XGboost
library(MLmetrics) # Machine learning evaluation metrics
library(smotefamily) # Solve class imbalance
library(ggplot2)
library(dplyr)
library(PRROC)
library(xgboost)
library(randomForest)
library(rpart)
library(rpart.plot)
