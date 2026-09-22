## =============================================================================
##  Insurance Claim Prediction: No SMOTE vs. With SMOTE
##  Models: Logistic Regression | XGBoost | Decision Tree | Random Forest
## =============================================================================
##
##  WHAT THIS SCRIPT DOES
##  ----------------------------------------------------------------------------
##  Trains and compares four binary classifiers that predict whether an
##  insurance policy will file a claim (target column `is_claim`), once
##  WITHOUT class-balancing and once WITH SMOTE oversampling, so the two
##  scenarios can be compared side by side on the same held-out test set.
##
##  HOW TO RUN
##  ----------------------------------------------------------------------------
##  1. Set DATA_PATH below to the location of your train.csv
##     (expects one row per policy, a binary `is_claim` column, and a
##     `policy_id` identifier column that is dropped before modelling).
##  2. Run the whole script top to bottom (e.g. `Rscript
##     insurance_claim_smote_comparison.R`, or source it in RStudio).
##  3. Missing packages are installed automatically on first run.
##  4. Optional: run report_extras.R afterwards, IN THE SAME R SESSION, for
##     Brier score and calibration-plot diagnostics (see that file's header).
##
##  RUNTIME NOTE
##  ----------------------------------------------------------------------------
##  Most of the time goes into the XGBoost grid search in Section 3
##  (nrow(xgb_grid) x CV_FOLDS x 2 scenarios full model fits). Shrink
##  `xgb_grid` for a quick smoke-test run before committing to the full grid.
##
##  METHODOLOGY / WHY THE WORKFLOW IS SET UP THIS WAY
##  ----------------------------------------------------------------------------
##  (identical for both scenarios, only the SMOTE switch differs)
##    1. Split train / test FIRST  -> the test set is never touched until the end
##    2. Fit all preprocessing on the training set only
##    3. Tune with 5-fold CV using the SAME folds for every model;
##       SMOTE (if on) is applied INSIDE each CV training fold only,
##       so synthetic rows never leak into a validation fold
##    4. Choose the classification threshold on out-of-fold training predictions
##    5. Score the test set exactly once
##  This avoids the most common sources of data leakage in imbalanced-class
##  problems: fitting preprocessing/SMOTE before the split, tuning a decision
##  threshold on the test set, and letting SMOTE-duplicated rows spill into a
##  validation fold.
## =============================================================================


## 0. Setup ---------------------------------------------------------------------

pkgs <- c(
  "caret", "MLmetrics", "smotefamily", "PRROC", "xgboost",
  "randomForest", "rpart", "ggplot2", "dplyr"
)
missing_pkgs <- setdiff(pkgs, rownames(installed.packages()))
if (length(missing_pkgs) > 0) install.packages(missing_pkgs)
invisible(lapply(pkgs, library, character.only = T))

## ---- Configuration ----
DATA_PATH  <- "/Users/allen/Downloads/Coding & Apps/R/R for DS/project/train.csv"
SEED       <- 123
TRAIN_P    <- 0.8                            # train / test split ratio
CV_FOLDS   <- 5
SMOTE_K    <- 5                              # nearest neighbours used by SMOTE
SMOTE_DUP  <- 7                              # synthetic copies per minority row
BETA       <- 2                              # F-beta for threshold choice (2 = favour Recall)
THRESHOLDS <- seq(0.01, 0.90, by = 0.005)    # starts low: un-balanced models output small probabilities
POS        <- "Yes"                          # positive class MUST be the first factor level
NEG        <- "No"

## ---- Hyperparameter grids (shrink these for a quick test run) ----
xgb_grid <- expand.grid(
  nrounds          = 100,
  max_depth        = c(3, 4, 5),
  eta              = 0.1,
  gamma            = c(0, 0.5, 1, 2),
  colsample_bytree = c(0.6),
  min_child_weight = c(1),
  subsample        = c(0.8)
)                                            # nrow(xgb_grid) combinations x 5 folds (slowest step)

tree_grid <- data.frame(cp = c(1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2))

## Optional: parallel backend (uncomment; remember to call stopCluster(cl) at the end)
# library(doParallel)
# cl <- makePSOCKcluster(parallel::detectCores() - 1)
# registerDoParallel(cl)


## 1. Load Data & Exploratory Analysis ------------------------------------------

claims <- read.csv(DATA_PATH, stringsAsFactors = F)

str(claims)
cat("Missing values:", sum(is.na(claims)), "\n")

claims <- claims %>%
  dplyr::select(-policy_id) %>%                       # remove the identifier
  mutate(
    across(where(is.character), as.factor),
    is_claim = factor(ifelse(is_claim == 1, POS, NEG), levels = c(POS, NEG))
  )

table(claims$is_claim)
round(prop.table(table(claims$is_claim)), 4)


## 2. Train-Test Split, THEN Preprocessing (fit on training data only) ----------

set.seed(SEED)
train_idx <- createDataPartition(claims$is_claim, p = TRAIN_P, list = F)

train_raw <- claims[train_idx, ]
test_raw  <- claims[-train_idx, ]
y_train   <- train_raw$is_claim
y_test    <- test_raw$is_claim

train_pred <- dplyr::select(train_raw, -is_claim)
test_pred  <- dplyr::select(test_raw,  -is_claim)

# Drop single-valued predictors (decided on the training set)
multi_valued <- sapply(train_pred, function(x) length(unique(x)) > 1)
train_pred   <- train_pred[, multi_valued, drop = F]
test_pred    <- test_pred[,  multi_valued, drop = F]

# One-hot encoding rules are learned from the training predictors only
dummy_fit <- dummyVars(~ ., data = train_pred, fullRank = TRUE)

encode <- function(df) {
  out <- as.data.frame(predict(dummy_fit, newdata = df))
  names(out) <- make.names(names(out))
  out
}
X_train <- encode(train_pred)
X_test  <- encode(test_pred)

# Drop all-zero dummy columns and exact linear combinations (decided on training data)
X_train <- X_train[, sapply(X_train, function(x) var(x) > 0), drop = FALSE]
lin_combo <- findLinearCombos(as.matrix(X_train))$remove
if (length(lin_combo) > 0) X_train <- X_train[, -lin_combo, drop = FALSE]
X_test <- X_test[, names(X_train), drop = FALSE]

# Rescale to [0, 1]: SMOTE relies on Euclidean distance, so scales must be comparable
scaler  <- preProcess(X_train, method = "range")
X_train <- predict(scaler, X_train)
X_test  <- predict(scaler, X_test)

cat("Train:", nrow(X_train), "rows |", "Test:", nrow(X_test), "rows |",
    "Features:", ncol(X_train), "\n")

# Data size summary (numbers for the write-up)
data_size <- data.frame(
  Item = c(
    "Total policies", "Train policies", "Test policies",
    "Model input columns after one-hot (and dropping constant/collinear ones)",
    "Claim rate: overall", "Claim rate: train",
    "Claim rate: test (= no-skill PR-AUC baseline)"
  ),
  Value = c(
    format(nrow(claims),  big.mark = ","),
    format(nrow(X_train), big.mark = ","),
    format(nrow(X_test),  big.mark = ","),
    ncol(X_train),
    sprintf("%.2f%%", 100 * mean(claims$is_claim == POS)),
    sprintf("%.2f%%", 100 * mean(y_train == POS)),
    sprintf("%.2f%%", 100 * mean(y_test == POS))
  )
)
print(data_size, row.names = FALSE)

# The same CV folds are reused for every model and both scenarios (fair comparison)
set.seed(SEED)
cv_folds <- createFolds(y_train, k = CV_FOLDS, returnTrain = T)


## 3. Helper Functions ----------------------------------------------------------

## ---- SMOTE sampler that caret runs inside each CV training fold ----
make_smote <- function(K, dup_size) {
  list(
    name  = "SMOTE (smotefamily)",
    first = TRUE,
    func  = function(x, y) {
      out <- smotefamily::SMOTE(
        X = as.data.frame(x), target = as.character(y),
        K = K, dup_size = dup_size
      )$data
      list(
        x = out[, setdiff(names(out), "class"), drop = FALSE],
        y = factor(out$class, levels = levels(y))
      )
    }
  )
}

# Sanity check only (NOT used for modelling): what SMOTE does to the class balance
set.seed(SEED)
smote_demo <- make_smote(SMOTE_K, SMOTE_DUP)$func(X_train, y_train)
cat("\nClass balance after SMOTE (illustration on the full training set):\n")
print(table(smote_demo$y))
print(round(prop.table(table(smote_demo$y)), 4))
rm(smote_demo)

## ---- Precision / Recall / F-scores at a given threshold ----
metrics_at <- function(thr, prob, obs, beta = BETA) {
  pred_pos <- prob >= thr
  is_pos   <- obs == POS
  tp <- sum(pred_pos & is_pos)
  fp <- sum(pred_pos & !is_pos)
  fn <- sum(!pred_pos & is_pos)
  
  precision <- if (tp + fp == 0) 0 else tp / (tp + fp)
  recall    <- if (tp + fn == 0) 0 else tp / (tp + fn)
  f1        <- if (precision + recall == 0) 0 else
    2 * precision * recall / (precision + recall)
  f_beta    <- if (precision + recall == 0) 0 else
    (1 + beta^2) * precision * recall / (beta^2 * precision + recall)
  
  c(threshold = thr, precision = precision, recall = recall,
    f1 = f1, f_beta = f_beta)
}

## ---- Wrap a caret model so every model exposes the same interface ----
wrap_caret <- function(fit) {
  list(
    pred         = fit$pred,                       # out-of-fold predictions
    bestTune     = fit$bestTune,
    fit          = fit,
    predict_prob = function(X) predict(fit, newdata = X, type = "prob")[[POS]]
  )
}

## ---- XGBoost: grid search with CV (SMOTE only inside training folds) ----
##  Written with xgboost's own API instead of caret's "xgbTree", because caret
##  (6.0-94) fails with xgboost >= 3.0. Bonus: SMOTE runs once per fold rather
##  than once per fold x grid row.
tune_xgboost <- function(use_smote, scenario) {
  sampler <- if (use_smote) make_smote(SMOTE_K, SMOTE_DUP)$func else NULL
  to_dmatrix <- function(x, y = NULL) {
    xgb.DMatrix(as.matrix(x), label = if (is.null(y)) NULL else as.integer(y == POS))
  }
  augment <- function(x, y) if (is.null(sampler)) list(x = x, y = y) else sampler(x, y)
  
  set.seed(SEED)
  fold_data <- lapply(cv_folds, function(train_id) {
    tr   <- augment(X_train[train_id, , drop = FALSE], y_train[train_id])
    hold <- setdiff(seq_len(nrow(X_train)), train_id)
    list(
      dtrain = to_dmatrix(tr$x, tr$y),
      hold   = hold,
      dhold  = to_dmatrix(X_train[hold, , drop = FALSE])
    )
  })
  
  make_params <- function(g) list(
    objective = "binary:logistic", seed = SEED,
    eta = g$eta, max_depth = g$max_depth, gamma = g$gamma,
    subsample = g$subsample, colsample_bytree = g$colsample_bytree,
    min_child_weight = g$min_child_weight
  )
  
  oof_list <- vector("list", nrow(xgb_grid))
  pr_auc   <- numeric(nrow(xgb_grid))
  for (i in seq_len(nrow(xgb_grid))) {
    g   <- xgb_grid[i, ]
    oof <- numeric(nrow(X_train))
    for (f in fold_data) {
      m <- xgb.train(params = make_params(g), data = f$dtrain,
                     nrounds = g$nrounds, verbose = 0)
      oof[f$hold] <- predict(m, f$dhold)
    }
    oof_list[[i]] <- oof
    pr_auc[i]     <- MLmetrics::PRAUC(y_pred = oof, y_true = as.integer(y_train == POS))
    message(sprintf("  [%s] XGBoost grid %d/%d  CV PR-AUC = %.4f",
                    scenario, i, nrow(xgb_grid), pr_auc[i]))
  }
  best_i <- which.max(pr_auc)
  best_g <- xgb_grid[best_i, ]
  
  # Refit on the full training set (SMOTE-augmented if requested)
  full  <- augment(X_train, y_train)
  final <- xgb.train(params = make_params(best_g), data = to_dmatrix(full$x, full$y),
                     nrounds = best_g$nrounds, verbose = 0)
  
  oof_df <- data.frame(obs = y_train)
  oof_df[[POS]] <- oof_list[[best_i]]
  list(
    pred         = oof_df,
    bestTune     = best_g,
    fit          = final,
    predict_prob = function(X) predict(final, to_dmatrix(X))
  )
}

## ---- Train all four models for one scenario ----
fit_all_models <- function(use_smote, scenario) {
  ctrl <- trainControl(
    method          = "cv",
    number          = CV_FOLDS,
    index           = cv_folds,                 # identical folds for every model
    classProbs      = TRUE,
    summaryFunction = prSummary,                # PR-AUC, Precision, Recall, F
    savePredictions = "final",                  # keeps out-of-fold predictions
    allowParallel   = TRUE,
    sampling        = if (use_smote) make_smote(SMOTE_K, SMOTE_DUP) else NULL
  )
  ctrl_1se <- ctrl
  ctrl_1se$selectionFunction <- "oneSE"         # 1-SE rule for the decision tree
  
  n_feat  <- ncol(X_train)
  rf_grid <- data.frame(
    mtry = sort(unique(pmax(1, round(sqrt(n_feat) * c(0.5, 1, 2)))))
  )
  
  fits <- list()
  
  message("[", scenario, "] 1/4 Logistic Regression")
  set.seed(SEED)
  fits[["Logistic Regression"]] <- wrap_caret(train(
    x = X_train, y = y_train, method = "glm", family = binomial(link = "logit"),
    trControl = ctrl, metric = "AUC"
  ))
  
  message("[", scenario, "] 2/4 XGBoost (slowest step)")
  fits[["XGBoost"]] <- tune_xgboost(use_smote, scenario)
  
  message("[", scenario, "] 3/4 Decision Tree")
  set.seed(SEED)
  fits[["Decision Tree"]] <- wrap_caret(train(
    x = X_train, y = y_train, method = "rpart",
    tuneGrid = tree_grid, trControl = ctrl_1se, metric = "AUC",
    control = rpart.control(minsplit = 2)
  ))
  
  message("[", scenario, "] 4/4 Random Forest")
  set.seed(SEED)
  fits[["Random Forest"]] <- wrap_caret(train(
    x = X_train, y = y_train, method = "rf",
    tuneGrid = rf_grid, trControl = ctrl, metric = "AUC",
    ntree = 100, importance = TRUE
  ))
  
  fits
}

## ---- Threshold from OOF predictions -> one-time scoring on the test set ----
evaluate_model <- function(model, model_name, scenario) {
  # (a) choose the threshold on out-of-fold TRAINING predictions
  oof       <- model$pred
  thr_curve <- as.data.frame(t(sapply(
    THRESHOLDS, metrics_at, prob = oof[[POS]], obs = oof$obs
  )))
  best_thr  <- thr_curve$threshold[which.max(thr_curve$f_beta)]
  
  # (b) score the untouched TEST set once
  prob_test  <- model$predict_prob(X_test)
  at_default <- as.list(metrics_at(0.5,      prob_test, y_test))
  at_tuned   <- as.list(metrics_at(best_thr, prob_test, y_test))
  
  pr <- PRROC::pr.curve(
    scores.class0 = prob_test[y_test == POS],
    scores.class1 = prob_test[y_test == NEG],
    curve = TRUE
  )
  roc <- PRROC::roc.curve(
    scores.class0 = prob_test[y_test == POS],
    scores.class1 = prob_test[y_test == NEG]
  )
  
  pred_label <- factor(ifelse(prob_test >= best_thr, POS, NEG), levels = c(POS, NEG))
  cm <- as.data.frame(table(Predicted = pred_label, Actual = y_test))
  cm$Model    <- sprintf("%s\n(threshold = %.3f)", model_name, best_thr)
  cm$Scenario <- scenario
  
  thr_curve$Model    <- model_name
  thr_curve$Scenario <- scenario
  
  list(
    summary = data.frame(
      Scenario      = scenario,
      Model         = model_name,
      PR_AUC        = pr$auc.integral,
      ROC_AUC       = roc$auc,
      Threshold     = best_thr,
      Precision     = at_tuned$precision,
      Recall        = at_tuned$recall,
      F1            = at_tuned$f1,
      Precision_at_0.5 = at_default$precision,
      Recall_at_0.5    = at_default$recall,
      F1_at_0.5        = at_default$f1
    ),
    thr_curve = thr_curve,
    cm        = cm,
    pr_curve  = data.frame(
      Scenario = scenario, Model = model_name,
      Recall = pr$curve[, 1], Precision = pr$curve[, 2]
    )
  )
}

run_experiment <- function(use_smote) {
  scenario <- if (use_smote) "With SMOTE" else "No SMOTE"
  fits     <- fit_all_models(use_smote, scenario)
  evals    <- Map(function(m, nm) evaluate_model(m, nm, scenario), fits, names(fits))
  list(scenario = scenario, fits = fits, evals = evals)
}

collect <- function(results, field) {
  bind_rows(lapply(results, function(r) bind_rows(lapply(r$evals, `[[`, field))))
}

## ---- Fold-by-fold CV metrics rebuilt from the out-of-fold predictions ----
cv_fold_report <- function(model, thr) {
  pred <- model$pred
  if (!is.null(pred$rowIndex)) pred <- pred[order(pred$rowIndex), ]  # caret models
  
  per_fold <- bind_rows(lapply(cv_folds, function(train_id) {
    hold <- setdiff(seq_len(nrow(X_train)), train_id)
    prob <- pred[[POS]][hold]
    obs  <- pred$obs[hold]
    m    <- metrics_at(thr, prob, obs)
    data.frame(
      PR_AUC    = MLmetrics::PRAUC(y_pred = prob, y_true = as.integer(obs == POS)),
      Precision = m[["precision"]],
      Recall    = m[["recall"]],
      F1        = m[["f1"]]
    )
  }), .id = "Fold")
  
  list(
    per_fold = per_fold,
    summary  = sapply(per_fold[-1], function(x) sprintf("%.4f +/- %.4f", mean(x), sd(x)))
  )
}

## ---- Plot helpers ----
plot_threshold_curves <- function(thr_all, summary_tbl, scenario) {
  f_label <- paste0("F", BETA)
  best    <- filter(summary_tbl, Scenario == scenario)
  
  ggplot(filter(thr_all, Scenario == scenario), aes(x = threshold)) +
    geom_line(aes(y = precision, colour = "Precision"), linewidth = 0.8) +
    geom_line(aes(y = recall,    colour = "Recall"),    linewidth = 0.8) +
    geom_line(aes(y = f_beta,    colour = f_label),     linewidth = 1.1) +
    geom_vline(
      data = best, aes(xintercept = Threshold),
      linetype = "dashed", colour = "red", linewidth = 0.7
    ) +
    facet_wrap(~ Model) +
    scale_colour_manual(
      values = setNames(c("darkorange", "blue", "darkgreen"),
                        c("Precision", "Recall", f_label))
    ) +
    labs(
      title = paste("Out-of-fold metrics vs. threshold:", scenario),
      subtitle = "Red dashed line = threshold chosen on training folds",
      x = "Threshold", y = "Score", colour = NULL
    ) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5),
          plot.subtitle = element_text(hjust = 0.5))
}

plot_confusion <- function(cm_all, scenario) {
  ggplot(filter(cm_all, Scenario == scenario),
         aes(x = Actual, y = Predicted, fill = Freq)) +
    geom_tile(colour = "white") +
    geom_text(aes(label = format(Freq, big.mark = ",")), size = 5) +
    scale_fill_gradient(low = "white", high = "steelblue") +
    scale_y_discrete(limits = rev) +
    facet_wrap(~ Model, nrow = 1) +
    labs(
      title = paste("Test-set confusion matrices:", scenario),
      x = "Actual Label", y = "Predicted Label"
    ) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5), legend.position = "none")
}

plot_pr_curves <- function(pr_all, base_rate) {
  ggplot(pr_all, aes(x = Recall, y = Precision, colour = Model)) +
    geom_line(linewidth = 0.8) +
    geom_hline(yintercept = base_rate, linetype = "dotted", colour = "grey40") +
    facet_wrap(~ Scenario) +
    labs(
      title = "Precision-Recall Curves (test set)",
      subtitle = "Dotted line = no-skill baseline (claim rate)"
    ) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5),
          plot.subtitle = element_text(hjust = 0.5))
}

plot_comparison <- function(summary_tbl) {
  long <- bind_rows(
    transmute(summary_tbl, Scenario, Model, Metric = "PR-AUC", Value = PR_AUC),
    transmute(summary_tbl, Scenario, Model, Metric = "Recall", Value = Recall),
    transmute(summary_tbl, Scenario, Model, Metric = "F1",     Value = F1)
  )
  ggplot(long, aes(x = Model, y = Value, fill = Scenario)) +
    geom_col(position = position_dodge(width = 0.9)) +
    geom_text(
      aes(label = round(Value, 3)),
      position = position_dodge(width = 0.9), vjust = -0.3, size = 3
    ) +
    facet_wrap(~ Metric) +
    labs(title = "No SMOTE vs. With SMOTE (test set, tuned threshold)", y = NULL) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5),
          axis.text.x = element_text(angle = 30, hjust = 1))
}


## 4. Run Both Scenarios --------------------------------------------------------

results <- list(
  "No SMOTE"   = run_experiment(use_smote = FALSE),
  "With SMOTE" = run_experiment(use_smote = TRUE)
)

# Selected hyperparameters
for (r in results) {
  cat("\n=====", r$scenario, ": selected hyperparameters =====\n")
  print(lapply(r$fits, function(f) f$bestTune))
}


## 5. Logistic Regression: Statistical Significance -----------------------------
##    Uses the ORIGINAL training data only. SMOTE creates synthetic rows, which
##    inflates the sample size and makes p-values meaningless.

sig_data <- cbind(X_train, claim = as.integer(y_train == POS))
sig_fit  <- glm(claim ~ ., data = sig_data, family = binomial(link = "logit"))
coefs    <- as.data.frame(summary(sig_fit)$coefficients)

significant_results <- data.frame(
  variable = rownames(coefs),
  estimate = coefs$Estimate,
  p_value  = coefs[["Pr(>|z|)"]]
) %>%
  filter(variable != "(Intercept)", p_value < 0.05) %>%
  arrange(p_value)

cat("\nSignificant variables (p < 0.05), logistic regression on original data:\n")
print(significant_results, row.names = FALSE)


## 6. Results: No SMOTE vs. With SMOTE ------------------------------------------

summary_tbl <- collect(results, "summary")
thr_all     <- collect(results, "thr_curve")
cm_all      <- collect(results, "cm")
pr_all      <- collect(results, "pr_curve")

base_rate <- mean(y_test == POS)
cat("\nTest-set claim rate (no-skill PR-AUC baseline):", round(base_rate, 4), "\n")

cat("\n===== Test-set results (threshold tuned on training folds) =====\n")
print(
  summary_tbl %>% mutate(across(where(is.numeric), ~ round(.x, 4))),
  row.names = FALSE
)

# Effect of SMOTE = (With SMOTE) - (No SMOTE)
no_smote   <- filter(summary_tbl, Scenario == "No SMOTE")
with_smote <- filter(summary_tbl, Scenario == "With SMOTE")
smote_effect <- data.frame(
  Model        = no_smote$Model,
  delta_PR_AUC = with_smote$PR_AUC - no_smote$PR_AUC,
  delta_Recall = with_smote$Recall - no_smote$Recall,
  delta_F1     = with_smote$F1     - no_smote$F1
)
cat("\n===== Effect of SMOTE (With - No) =====\n")
print(smote_effect %>% mutate(across(where(is.numeric), ~ round(.x, 4))),
      row.names = FALSE)

for (scn in names(results)) {
  print(plot_threshold_curves(thr_all, summary_tbl, scn))
  print(plot_confusion(cm_all, scn))
}
print(plot_pr_curves(pr_all, base_rate))
print(plot_comparison(summary_tbl))

# Random forest variable importance
for (scn in names(results)) {
  print(plot(varImp(results[[scn]]$fits[["Random Forest"]]$fit), top = 15,
             main = paste("Random Forest Variable Importance:", scn)))
}


## 7. Cross-Validation Detail (training set, 5 folds) ---------------------------
##    PR-AUC does not depend on a threshold. Precision / Recall / F1 use the
##    threshold chosen on the pooled out-of-fold predictions, so they are mildly
##    optimistic. Hyperparameters and threshold were both selected on these same
##    folds (no nested CV), so treat the test-set numbers as the clean ones.

cv_detail <- lapply(results, function(r) {
  setNames(
    lapply(names(r$fits), function(nm) {
      thr <- summary_tbl$Threshold[
        summary_tbl$Scenario == r$scenario & summary_tbl$Model == nm
      ]
      cv_fold_report(r$fits[[nm]], thr)
    }),
    names(r$fits)
  )
})

cv_summary <- bind_rows(lapply(names(cv_detail), function(scn) {
  bind_rows(lapply(names(cv_detail[[scn]]), function(nm) {
    data.frame(Scenario = scn, Model = nm, as.list(cv_detail[[scn]][[nm]]$summary))
  }))
}))
cat("\n===== CV mean +/- sd over 5 folds (all models) =====\n")
print(cv_summary, row.names = FALSE)

for (scn in names(cv_detail)) {
  cat("\n===== XGBoost per-fold CV:", scn, "=====\n")
  print(
    cv_detail[[scn]][["XGBoost"]]$per_fold %>%
      mutate(across(where(is.numeric), ~ round(.x, 4))),
    row.names = FALSE
  )
}