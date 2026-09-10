### 共用套件載入與資料前處理
## Step 1：載入所有所需套件
# 資料處理套件
install.packages('randomForest')
install.packages('rpart.plot')
library(caret) #用於分層抽樣與訓練 XGboost
library(Matrix) # 建立稀疏矩陣供 XGboost 使用
library(MLmetrics) # 機器學習評估指標
library(smotefamily)
library(ggplot2)
library(dplyr)
library(PRROC)

# 機器學習套件
library(xgboost)
library(randomForest)
library(rpart)
library(rpart.plot)

## Step 2：讀取資料與基本處理
data <- read.csv("C:/Users/USER/Downloads/train.csv") # 依實際路徑調整
str(data)
names(data)
sum(is.na(data))
data <- data %>% select(-policy_id) # 移除無關(policy_id)欄位
data$is_claim <- factor(data$is_claim, levels = c(0, 1)) # 將 is_claim 轉為類別變數

## Step 3：資料結構檢查與類別分佈觀察
table(data$is_claim) # 查看 is_claim 的分佈
prop.table(table(data$is_claim)) # 查看 is_claim 的比例

## Step 4：資料切分（80%：20%）
# One-hot encoding，並保留目標變數 is_claim
dummy_model <- dummyVars(~ ., data = data, fullRank = TRUE)
data_processed <- predict(dummy_model, newdata = data) %>%
  as.data.frame() %>%
  select(-starts_with("is_claim")) %>% # 移除 one-hot 之後的 is_claim
  mutate(is_claim = data$is_claim) %>% # data_processed 的 is_claim 是正常的
  setNames(make.names(names(.)))

# 資料切分
set.seed(123)
train_index <- createDataPartition(data_processed$is_claim, p = 0.8, list = FALSE)
train_raw <- data_processed[train_index, ]
test_raw  <- data_processed[-train_index, ]
test_data <- select(test_raw, -is_claim) # 將訓練資料屏蔽 is_claim 資訊
test_label <- test_raw$is_claim # test_raw 的 is_claim 資料

# Step 5：SMOTE 平衡樣本
x_train <- select(train_raw, -is_claim)
y_train <- as.numeric(train_raw$is_claim) - 1

# SMOTE 本來就會多一欄"class"出來
set.seed(123)
train_balanced <- SMOTE(x_train, y_train, K = 5, dup_size = 7)$data %>%
  mutate(is_claim = as.factor(class)) %>%
  select(-class)

# 檢查平衡後類別分布
table(train_balanced$is_claim)
prop.table(table(train_balanced$is_claim))

### Logistic Regression
## Step 1：訓練 logistic regression 模型
set.seed(123)
logistic_model <- glm(is_claim ~ ., 
                      family = binomial(link = "logit"), 
                      data = train_balanced)

## Step 2：提取所有變數的 p-value
summary_model <- summary(logistic_model)
coefficients_table <- as.data.frame(summary_model$coefficients)

# 提取變數名稱和對應的 p-value
results <- data.frame(
  variable = rownames(coefficients_table)[-1], # 排除截距項
  p_value = coefficients_table[-1, "Pr(>|z|)"],
  stringsAsFactors = FALSE
)

## Step 3：查看 p-value < 0.05 的變數
significant_results <- subset(results, p_value < 0.05)
print("p-value < 0.05 的顯著變數:"); print(significant_results)

## Step 4：預測與評估
# 預測 test 資料的機率，此次使用的門檻值為 0.5
pred_prob_log <- predict(logistic_model, newdata = test_data, type = "response")
pred_label_log <- ifelse(pred_prob_log >= 0.5, 1, 0)

# 評估指標
f1_log <- F1_Score(y_pred = pred_label_log, y_true = test_label, positive = "1")
recall_log <- Recall(y_pred = pred_label_log, y_true = test_label, positive = "1")
cat("樣本外 F1 Score: ", round(f1_log, 4), "\n")
cat("樣本外 Recall: ", round(recall_log, 4), "\n")

## Step 5：製作 Recall vs Threshold 圖 
thresholds <- seq(0.1, 0.9, by = 0.01)
recall_scores_log <- sapply(thresholds, function(thresh) {
  pred_label_log <- ifelse(pred_prob_log >= thresh, 1, 0)
  Recall(y_true = test_label, y_pred = pred_label_log, positive = "1")
})

best_thresh_log <- thresholds[which.max(recall_scores_log)] # 尋找最佳 recall 下，threshold 會是多少
best_recall_log <- max(na.omit(recall_scores_log))

# data.frame 在做的時候不能省略 na，因為這樣會跟 0.1 ~ 0.9 的個數不一樣
df_plot_log <- data.frame(Threshold = thresholds, Recall = recall_scores_log)
df_plot_log <- na.omit(df_plot_log)

threshrold_sum_log <- sum(na.omit(recall_scores_log) == 1) # 有可能 recall score = 1 的有很多個
thresh_log <- ifelse(threshrold_sum_log < 1, 1, threshrold_sum_log) # for 參照位置使用
maxthresh_log <- thresh_log * 0.01 + 0.1 - 0.01 # 找到 recall = 1 的最大門檻

# 做圖
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

## Step 6: 畫混淆矩陣
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

### XGboost
## Step 1：Dummy encoding for XGboost
train_matrix <- sparse.model.matrix(is_claim ~ . -1, data = train_raw)
test_matrix  <- sparse.model.matrix(is_claim ~ . -1, data = test_raw)

## Step 2：建立 DMatrix 格式 for XGboost
dtrain <- xgb.DMatrix(data = train_matrix, label = as.numeric(train_raw$is_claim) - 1)
dtest  <- xgb.DMatrix(data = test_matrix, label = as.numeric(test_raw$is_claim) - 1)

## Step 3：交叉驗證（k-fold 樣本內驗證）
# 轉為 caret 格式：將 label 合併進 feature matrix
train_df <- as.data.frame(as.matrix(train_matrix))
train_df$is_claim <- factor(train_raw$is_claim, levels = c(0, 1), 
                            labels = c("No", "Yes")) # 沒有labels，就跑不了

# 設定交叉驗證參數
ctrl <- trainControl(
  method = "cv", # Cross Validation
  number = 5, # 代表有5-fold
  classProbs = TRUE, # 回傳機率值（例如：預測為「1」的機率）。
  summaryFunction = prSummary, # 指定模型評估的指標，之後才能用 Recall 優化
  verboseIter = TRUE, # 可讓使用者即時看到訓練過程
  allowParallel = TRUE # 平行運算加快速度
)

# 調參
grid <- expand.grid(
  nrounds = 100,
  max_depth = c(3, 4, 5),
  eta = 0.1,
  gamma = 1,
  colsample_bytree = 0.6,
  min_child_weight = 1,
  subsample = 0.6)

# 建立模型（樣本內交叉驗證）
set.seed(123)
xgb_cv_model <- train(
  is_claim ~ .,
  data = train_df,
  method = "xgbTree",
  trControl = ctrl,
  tuneGrid = grid,
  metric = "Recall")

## Step 4：設定參數與訓練模型
# 計算 scale_pos_weight，告知模型類別的比例
ratio <- sum(train_raw$is_claim == 0) / sum(train_raw$is_claim == 1)

# 最佳參數
best_params <- xgb_cv_model$bestTune

# 訓練用的參數
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

# 訓練實際要用到的模型
set.seed(123)
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = best_params$nrounds,
  watchlist = list(train = dtrain, eval = dtest),
  early_stopping_rounds = 10,
  verbose = 1
)

## Step 5：預測與評估
# 預測 test 資料的機率，此次使用的門檻值為 0.5
pred_prob_xg <- predict(xgb_model, newdata = dtest,
                        iteration_range = c(0, xgb_model$best_iteration))
pred_label_xg <- ifelse(pred_prob_xg >= 0.5, 1, 0)

# 評估指標
f1_xg <- F1_Score(y_pred = pred_label_xg, y_true = test_label, positive = "1")
recall_xg <- Recall(y_pred = pred_label_xg, y_true = test_label, positive = "1")
cat("樣本外 F1 Score: ", round(f1_xg, 4), "\n")
cat("樣本外 Recall: ", round(recall_xg, 4), "\n")

## Step 6：製作 Recall vs Threshold 圖 
recall_scores_xg <- sapply(thresholds, function(thresh) {
  pred_label_xg <- ifelse(pred_prob_xg >= thresh, 1, 0)
  Recall(y_true = test_label, y_pred = pred_label_xg, positive = "1")
})

best_thresh_xg <- thresholds[which.max(recall_scores_xg)] # 尋找最佳 recall 下，threshold 會是多少
best_recall_xg <- max(na.omit(recall_scores_xg))

# data.frame 在做的時候不能省略 na，因為這樣會跟 0.1 ~ 0.9 的個數不一樣
df_plot_xg <- data.frame(Threshold = thresholds, Recall = recall_scores_xg)
df_plot_xg <- na.omit(df_plot_xg)

threshrold_sum_xg <- sum(na.omit(recall_scores_xg) == 1) # 有可能 recall score = 1 的有很多個
thresh_xg <- ifelse(threshrold_sum_xg < 1, 1, threshrold_sum_xg) # for 參照位置使用
maxthresh_xg <- thresh_xg * 0.01 + 0.1 - 0.01 # 找到 recall = 1 的最大門檻

# 做圖
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

## Step 7：畫混淆矩陣
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

## Step 8：畫 PR curve
pr_xg <- pr.curve(scores.class0 = pred_prob_xg[test_label == 1],
                  scores.class1 = pred_prob_xg[test_label == 0],
                  curve = TRUE)
plot(pr_xg)

### Decision Tree
## Step 1：取前10重要變數
decision_tree_model <- rpart(is_claim ~ ., data = data_processed, method = "class",
                             control = rpart.control(minsplit = 2, cp = 0))
importance <- sort(decision_tree_model$variable.importance, decreasing = TRUE)
top10 <- names(importance)[1:10]
selected_data <- data_processed[, c("is_claim", top10)]

## Step 2：決策樹建模與修剪
# 訓練模型（以前10重要特徵為基底）
set.seed(123)
decision_tree_model <- rpart(is_claim ~ ., data = train_balanced,
                             method = "class", 
                             control = rpart.control(minsplit = 2, cp = 0))
# 有把train_balanced_DT改成無DT無DT
# 修剪樹枝並優化
plotcp(decision_tree_model)

# 取得 cp table
cp_table <- printcp(decision_tree_model)

# 最小 xerror 及其標準誤
min_xerror <- min(cp_table[, "xerror"])
min_xerror_se <- cp_table[which.min(cp_table[, "xerror"]), "xstd"]

# 1-SE Rule：選擇 xerror <= min_xerror + xstd 最小的 cp
one_se_threshold <- min_xerror + min_xerror_se
cp_candidates <- cp_table[cp_table[, "xerror"] <= one_se_threshold, ]
best_cp_1se <- cp_candidates[which.min(cp_candidates[, "nsplit"]), "CP"]
cat("Best cp (1-SE Rule):", best_cp_1se, "\n")

decision_tree_model <- prune(decision_tree_model, cp = best_cp_1se)

## Step 3：預測與評估
test_data_DT <- select(test_raw, -is_claim)
true_label_DT <- factor(test_raw$is_claim, levels = c(0, 1))
pred_label_DT <- predict(decision_tree_model, newdata = test_data_DT,
                         type = "class")

# 評估結果
f1_DT <- F1_Score(y_pred = pred_label_DT, y_true = true_label_DT, positive = "1")
recall_DT <- Recall(y_pred = pred_label_DT, y_true = true_label_DT, positive = "1")

cat("樣本外 F1 Score: ", round(f1_DT, 4), "\n")
cat("樣本外 Recall: ", round(recall_DT, 4), "\n")

## Step 4：畫混淆矩陣
conf_mat_DT <- table(Predicted = pred_label_DT, Actual = true_label_DT)
conf_df_DT <- as.data.frame(conf_mat_DT)
names(conf_df_DT) <- c("Predicted", "Actual", "Freq")

# 作圖
ggplot(conf_df_DT, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 6) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(title = "Confusion Matrix : Decision Tree",
       x = "Actual Label",
       y = "Predicted Label") +
  theme_minimal() 

## Step 5：畫PR curve
# PR Curve（使用類別 1 的預測機率）
pred_prob_DT <- predict(decision_tree_model, newdata = test_data_DT,
                        type = "prob")[, 2]
pr <- pr.curve(scores.class0 = pred_prob_DT[true_label_DT == 1],
               scores.class1 = pred_prob_DT[true_label_DT == 0],
               curve = TRUE)
plot(pr)

### Random Forest
## Step 1：訓練Random Forest模型
set.seed(123)
random_forest_model <- randomForest(
  is_claim ~ ., 
  data = train_balanced, 
  ntree = 100,
  mtry = floor(sqrt(ncol(train_balanced) - 1)),
  classwt = c("0" = 1, "1" = 1),
  importance = TRUE
)

## Step 2：樣本外模型評估
# 預測測試集
pred_prob_rf <- predict(random_forest_model, newdata = test_data)
pred_prob_rf <- factor(pred_prob_rf, levels = levels(test_label))

# 混淆矩陣（測試集）
conf_mat_rf <- confusionMatrix(pred_prob_rf, test_label, positive = "1")
print(conf_mat_rf)

# recall 與 F1-score（測試集）
recall_rf   <- conf_mat_rf$byClass["Recall"]
f1_rf      <- conf_mat_rf$byClass["F1"]

## Step 3：變數重要性
varImpPlot(random_forest_model)

## Step 4：比較樣本外及樣本內F1 Score, Recall, 以及 MSE
cat("樣本外 F1 Score: ", round(f1_rf, 4), "\n")
cat("樣本外 Recall: ", round(recall_rf, 4), "\n")

## Step 5：混淆矩陣表格

# 作圖
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

## Step 6：畫 PR curve
# 取得機率預測值
pred_prob <- predict(random_forest_model, newdata = test_data, type = "prob")[,2]

# 計算 PR 曲線（正類別為 1）
pr_rf <- pr.curve(scores.class0 = pred_prob[test_label == 1],
                  scores.class1 = pred_prob[test_label == 0],
                  curve = TRUE)
plot(pr_rf)

### 統整
# 建立結果資料框
result_table <- data.frame(
  Model = c("Logistic Regression","XGBoost", "Decision Tree","Random Forest"),
  F1_Score = c(round(f1_log, 4),round(f1_xg, 4), round(f1_DT, 4),round(f1_rf, 4)),
  Recall = c(round(recall_log, 4),round(recall_xg, 4), round(recall_DT, 4),round(recall_rf, 4))
)

# 顯示結果
print(result_table)

