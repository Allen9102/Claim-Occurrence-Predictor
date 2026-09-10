## Packages
install.packages('randomForest')
install.packages('rpart.plot')
install.packages('caret')
install.packages('ggplot2')
install.packages('dplyr')
library(caret)
library(Matrix)
library(MLmetrics)
library(smotefamily)
library(ggplot2)
library(dplyr)
library(PRROC)
library(xgboost)
library(randomForest)
library(rpart)
library(rpart.plot)

## EDA
data <- read.csv("C:/Users/USER/Downloads/train.csv") # Adjust file path if needed
str(data)
names(data)
sum(is.na(data))
data <- data %>% select(-policy_id) # Remove irrelevant identifier
data$is_claim <- factor(data$is_claim, levels = c(0, 1))
table(data$is_claim)
prop.table(table(data$is_claim))

## Stratified Data Preparation
# One-hot encode categorical variables while preserving the target variable
dummy_model <- dummyVars(~ ., data = data, fullRank = TRUE)
data_processed <- predict(dummy_model, newdata = data) %>%
  as.data.frame() %>%
  select(-starts_with("is_claim")) %>% # Remove one-hot encoded target columns
  mutate(is_claim = data$is_claim) %>% # Restore the original target variable
  setNames(make.names(names(.)))

# Train-test split
set.seed(123)
train_index <- createDataPartition(data_processed$is_claim, p = 0.8, list = FALSE)
train_raw <- data_processed[train_index, ]
test_raw  <- data_processed[-train_index, ]
test_data <- select(test_raw, -is_claim) # Remove target variable from test features
test_label <- test_raw$is_claim # Store test labels separately

# Address class imbalance with SMOTE
x_train <- select(train_raw, -is_claim)
y_train <- as.numeric(train_raw$is_claim) - 1

set.seed(123)
train_balanced <- SMOTE(x_train, y_train, K = 5, dup_size = 7)$data %>%
  mutate(is_claim = as.factor(class)) %>%
  select(-class) # Remove the temporary class column created by SMOTE

table(train_balanced$is_claim)
prop.table(table(train_balanced$is_claim))

### Logistic Regression
## Step 1: Train the logistic regression model
set.seed(123)
logistic_model <- glm(is_claim ~ ., 
                      family = binomial(link = "logit"), 
                      data = train_balanced)

## Step 2: Extract p-values
summary_model <- summary(logistic_model)
coefficients_table <- as.data.frame(summary_model$coefficients)

# Extract variable names and corresponding p-values
results <- data.frame(
  variable = rownames(coefficients_table)[-1], # Exclude the intercept
  p_value = coefficients_table[-1, "Pr(>|z|)"],
  stringsAsFactors = FALSE
)

## Step 3: Identify significant variables
significant_results <- subset(results, p_value < 0.05)
print("Significant variables (p < 0.05):"); print(significant_results)

## Step 4: Prediction and evaluation
# Predict probabilities on the test set using a threshold of 0.5
pred_prob_log <- predict(logistic_model, newdata = test_data, type = "response")
pred_label_log <- ifelse(pred_prob_log >= 0.5, 1, 0)

# Evaluation metrics
f1_log <- F1_Score(y_pred = pred_label_log, y_true = test_label, positive = "1")
recall_log <- Recall(y_pred = pred_label_log, y_true = test_label, positive = "1")
cat("Out-of-sample F1 Score: ", round(f1_log, 4), "\n")
cat("Out-of-sample Recall: ", round(recall_log, 4), "\n")

## Step 5: Plot Recall vs. Threshold
thresholds <- seq(0.1, 0.9, by = 0.01)
recall_scores_log <- sapply(thresholds, function(thresh) {
  pred_label_log <- ifelse(pred_prob_log >= thresh, 1, 0)
  Recall(y_true = test_label, y_pred = pred_label_log, positive = "1")
})

best_thresh_log <- thresholds[which.max(recall_scores_log)] # Find the threshold with the highest recall
best_recall_log <- max(na.omit(recall_scores_log))

# Keep NA values until after creating the data frame to preserve row alignment
df_plot_log <- data.frame(Threshold = thresholds, Recall = recall_scores_log)
df_plot_log <- na.omit(df_plot_log)

threshrold_sum_log <- sum(na.omit(recall_scores_log) == 1) # Count thresholds with perfect recall
thresh_log <- ifelse(threshrold_sum_log < 1, 1, threshrold_sum_log) # Set the reference index
maxthresh_log <- thresh_log * 0.01 + 0.1 - 0.01 # Find the highest threshold with perfect recall

# Plot
ggplot(df_plot_log, aes(x = Threshold, y = Recall)) +
  geom_line(color = "blue", linewidth = 1) +
  geom_vline(xintercept = thresholds[thresh_log], linetype = "dashed", color = "red") +
  annotate("text", x = maxthresh_log + 0.13, y = 0.05, 
           label = paste("Best recall @ threshold =", maxthresh_log),
           color = "red") +
  labs(title = "Recall vs. Threshold: Logistic Regression",
       x = "Threshold", y = "Recall") +
  theme_minimal() + 
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_hline(yintercept = recall_scores_log[41], linetype = "dotted", color = "darkgreen") +
  annotate("text", x = 0.8, y = recall_scores_log[41], 
           label = paste0("Recall @ 0.5 = ", round(recall_scores_log[41], 4)),
           vjust = -1, hjust = 1, color = "darkgreen")

## Step 6: Plot confusion matrix
conf_mat_log <- table(Predicted = pred_label_log, Actual = test_label)
conf_df_log <- as.data.frame(conf_mat_log)
names(conf_df_log) <- c("Predicted", "Actual", "Freq")

ggplot(conf_df_log, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 6) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(title = "Confusion Matrix at threshold = 0.35: Logistic Regression",
       x = "Actual Label",
       y = "Predicted Label") +
  theme_minimal()

### XGBoost
## Step 1: Prepare data for XGBoost
train_matrix <- sparse.model.matrix(is_claim ~ . -1, data = train_raw)
test_matrix  <- sparse.model.matrix(is_claim ~ . -1, data = test_raw)

## Step 2: Create XGBoost DMatrix objects
dtrain <- xgb.DMatrix(data = train_matrix, label = as.numeric(train_raw$is_claim) - 1)
dtest  <- xgb.DMatrix(data = test_matrix, label = as.numeric(test_raw$is_claim) - 1)

## Step 3: Cross-validation and hyperparameter tuning
# Convert the training data to caret format by adding the target variable
train_df <- as.data.frame(as.matrix(train_matrix))
train_df$is_claim <- factor(train_raw$is_claim, levels = c(0, 1), 
                            labels = c("No", "Yes")) # Required by caret

# Set cross-validation parameters
ctrl <- trainControl(
  method = "cv", # Use cross-validation
  number = 5, # 5-fold cross-validation
  classProbs = TRUE, # Return class probabilities
  summaryFunction = prSummary, # Use PR-based performance metrics
  verboseIter = TRUE, # Display training progress
  allowParallel = TRUE # Enable parallel processing
)

# Define the hyperparameter grid
grid <- expand.grid(
  nrounds = 100,
  max_depth = c(3, 4, 5),
  eta = 0.1,
  gamma = 1,
  colsample_bytree = 0.6,
  min_child_weight = 1,
  subsample = 0.6)

# Train and tune the model using cross-validation
set.seed(123)
xgb_cv_model <- train(
  is_claim ~ .,
  data = train_df,
  method = "xgbTree",
  trControl = ctrl,
  tuneGrid = grid,
  metric = "Recall")

## Step 4: Set parameters and train the final model
# Calculate the class imbalance ratio
ratio <- sum(train_raw$is_claim == 0) / sum(train_raw$is_claim == 1)

# Get the best hyperparameters
best_params <- xgb_cv_model$bestTune

# Set XGBoost training parameters
params <- list(
  objective = "binary:logistic",
  eval_metric = "aucpr",
  eta = best_params$eta,
  max_depth = best_params$max_depth,
  gamma = best_params$gamma,
  subsample = best_params$subsample,
  colsample_bytree = best_params$colsample_bytree,
  min_child_weight = best_params$min_child_weight,
  scale_pos_weight = ratio
)

# Train the final XGBoost model
set.seed(123)
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = best_params$nrounds,
  watchlist = list(train = dtrain, eval = dtest),
  early_stopping_rounds = 10,
  verbose = 1
)

## Step 5: Prediction and evaluation
# Predict probabilities on the test set using a threshold of 0.5
pred_prob_xg <- predict(xgb_model, newdata = dtest,
                        iteration_range = c(0, xgb_model$best_iteration))
pred_label_xg <- ifelse(pred_prob_xg >= 0.5, 1, 0)

# Evaluation metrics
f1_xg <- F1_Score(y_pred = pred_label_xg, y_true = test_label, positive = "1")
recall_xg <- Recall(y_pred = pred_label_xg, y_true = test_label, positive = "1")
cat("Out-of-sample F1 Score: ", round(f1_xg, 4), "\n")
cat("Out-of-sample Recall: ", round(recall_xg, 4), "\n")

## Step 6: Plot Recall vs. Threshold
recall_scores_xg <- sapply(thresholds, function(thresh) {
  pred_label_xg <- ifelse(pred_prob_xg >= thresh, 1, 0)
  Recall(y_true = test_label, y_pred = pred_label_xg, positive = "1")
})

best_thresh_xg <- thresholds[which.max(recall_scores_xg)] # Find the threshold with the highest recall
best_recall_xg <- max(na.omit(recall_scores_xg))

# Keep NA values until after creating the data frame to preserve row alignment
df_plot_xg <- data.frame(Threshold = thresholds, Recall = recall_scores_xg)
df_plot_xg <- na.omit(df_plot_xg)

threshrold_sum_xg <- sum(na.omit(recall_scores_xg) == 1) # Count thresholds with perfect recall
thresh_xg <- ifelse(threshrold_sum_xg < 1, 1, threshrold_sum_xg) # Set the reference index
maxthresh_xg <- thresh_xg * 0.01 + 0.1 - 0.01 # Find the highest threshold with perfect recall

# Plot
ggplot(df_plot_xg, aes(x = Threshold, y = Recall)) +
  geom_line(color = "blue", linewidth = 1) +
  geom_vline(xintercept = thresholds[thresh_xg], linetype = "dashed", color = "red") +
  annotate("text", x = maxthresh_xg + 0.13, y = 0.05, 
           label = paste("Best recall @ threshold =", maxthresh_xg),
           color = "red") +
  labs(title = "Recall vs. Threshold: XGboost",
       x = "Threshold", y = "Recall") +
  theme_minimal() + 
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_hline(yintercept = recall_scores_xg[41], linetype = "dotted", color = "darkgreen") +
  annotate("text", x = 0.8, y = recall_scores_xg[41], 
           label = paste0("Recall @ 0.5 = ", round(recall_scores_xg[41], 4)),
           vjust = -1, hjust = 1, color = "darkgreen")

## Step 7: Plot confusion matrix
conf_mat_xg <- table(Predicted = pred_label_xg, Actual = test_label)
conf_df_xg <- as.data.frame(conf_mat_xg)
names(conf_df_xg) <- c("Predicted", "Actual", "Freq")

ggplot(conf_df_xg, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 6) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(title = "Confusion Matrix at threshold = 0.5: XGboost",
       x = "Actual Label",
       y = "Predicted Label") +
  theme_minimal()

## Step 8: Plot PR curve
pr_xg <- pr.curve(scores.class0 = pred_prob_xg[test_label == 1],
                  scores.class1 = pred_prob_xg[test_label == 0],
                  curve = TRUE)
plot(pr_xg)

### Decision Tree
## Step 1: Select the top 10 important variables
decision_tree_model <- rpart(is_claim ~ ., data = data_processed, method = "class",
                             control = rpart.control(minsplit = 2, cp = 0))
importance <- sort(decision_tree_model$variable.importance, decreasing = TRUE)
top10 <- names(importance)[1:10]
selected_data <- data_processed[, c("is_claim", top10)]

## Step 2: Train and prune the decision tree
# Train the initial tree
set.seed(123)
decision_tree_model <- rpart(is_claim ~ ., data = train_balanced,
                             method = "class", 
                             control = rpart.control(minsplit = 2, cp = 0))

# Inspect the complexity parameter table
plotcp(decision_tree_model)

# Extract the cp table
cp_table <- printcp(decision_tree_model)

# Find the minimum cross-validation error and its standard error
min_xerror <- min(cp_table[, "xerror"])
min_xerror_se <- cp_table[which.min(cp_table[, "xerror"]), "xstd"]

# Apply the 1-SE rule
one_se_threshold <- min_xerror + min_xerror_se
cp_candidates <- cp_table[cp_table[, "xerror"] <= one_se_threshold, ]
best_cp_1se <- cp_candidates[which.min(cp_candidates[, "nsplit"]), "CP"]
cat("Best cp (1-SE Rule):", best_cp_1se, "\n")

decision_tree_model <- prune(decision_tree_model, cp = best_cp_1se)

## Step 3: Prediction and evaluation
test_data_DT <- select(test_raw, -is_claim)
true_label_DT <- factor(test_raw$is_claim, levels = c(0, 1))
pred_label_DT <- predict(decision_tree_model, newdata = test_data_DT,
                         type = "class")

# Evaluation metrics
f1_DT <- F1_Score(y_pred = pred_label_DT, y_true = true_label_DT, positive = "1")
recall_DT <- Recall(y_pred = pred_label_DT, y_true = true_label_DT, positive = "1")

cat("Out-of-sample F1 Score: ", round(f1_DT, 4), "\n")
cat("Out-of-sample Recall: ", round(recall_DT, 4), "\n")

## Step 4: Plot confusion matrix
conf_mat_DT <- table(Predicted = pred_label_DT, Actual = true_label_DT)
conf_df_DT <- as.data.frame(conf_mat_DT)
names(conf_df_DT) <- c("Predicted", "Actual", "Freq")

# Plot
ggplot(conf_df_DT, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 6) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(title = "Confusion Matrix : Decision Tree",
       x = "Actual Label",
       y = "Predicted Label") +
  theme_minimal() 

## Step 5: Plot PR curve
# Plot the PR curve using predicted probabilities
pred_prob_DT <- predict(decision_tree_model, newdata = test_data_DT,
                        type = "prob")[, 2]
pr <- pr.curve(scores.class0 = pred_prob_DT[true_label_DT == 1],
               scores.class1 = pred_prob_DT[true_label_DT == 0],
               curve = TRUE)
plot(pr)

### Random Forest
## Step 1: Train the Random Forest model
set.seed(123)
random_forest_model <- randomForest(
  is_claim ~ ., 
  data = train_balanced, 
  ntree = 100,
  mtry = floor(sqrt(ncol(train_balanced) - 1)),
  classwt = c("0" = 1, "1" = 1),
  importance = TRUE
)

## Step 2: Evaluate the model on the test set
# Generate predictions
pred_prob_rf <- predict(random_forest_model, newdata = test_data)
pred_prob_rf <- factor(pred_prob_rf, levels = levels(test_label))

# Compute the confusion matrix
conf_mat_rf <- confusionMatrix(pred_prob_rf, test_label, positive = "1")
print(conf_mat_rf)

# Extract Recall and F1-score
recall_rf   <- conf_mat_rf$byClass["Recall"]
f1_rf      <- conf_mat_rf$byClass["F1"]

## Step 3: Variable importance
varImpPlot(random_forest_model)

## Step 4: Report out-of-sample performance
cat("Out-of-sample F1 Score: ", round(f1_rf, 4), "\n")
cat("Out-of-sample Recall: ", round(recall_rf, 4), "\n")

## Step 5: Plot confusion matrix

# Plot
conf_df_rf <- as.data.frame(conf_mat_rf$table)
names(conf_df_rf) <- c("Predicted", "Actual", "Freq")

ggplot(conf_df_rf, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 6) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(title = "Confusion Matrix (Random Forest)",
       x = "Actual Label",
       y = "Predicted Label") +
  theme_minimal()

## Step 6: Plot PR curve
# Obtain predicted probabilities
pred_prob <- predict(random_forest_model, newdata = test_data, type = "prob")[,2]

# Compute the PR curve for the positive class
pr_rf <- pr.curve(scores.class0 = pred_prob[test_label == 1],
                  scores.class1 = pred_prob[test_label == 0],
                  curve = TRUE)
plot(pr_rf)

### Summary
# Create a results table
result_table <- data.frame(
  Model = c("Logistic Regression","XGBoost", "Decision Tree","Random Forest"),
  F1_Score = c(round(f1_log, 4),round(f1_xg, 4), round(f1_DT, 4),round(f1_rf, 4)),
  Recall = c(round(recall_log, 4),round(recall_xg, 4), round(recall_DT, 4),round(recall_rf, 4))
)

# Display the results
print(result_table)
