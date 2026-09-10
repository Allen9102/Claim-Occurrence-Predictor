# Claim-Frequency-Predictor

An auto insurance claim frequency predictor using machine learning.

A team project that predicts whether an auto insurance policyholder will file a claim by comparing multiple classification models on structured policyholder data.

## Packages and Libraries

The project uses the following R packages:

- **randomForest** — Random Forest classification
- **rpart** — Decision Tree modeling
- **rpart.plot** — Decision Tree visualization
- **xgboost** — XGBoost modeling
- **caret** — Stratified data splitting, cross-validation, and hyperparameter tuning
- **smotefamily** — SMOTE for handling class imbalance
- **MLmetrics** — F1 Score and Recall evaluation
- **PRROC** — Precision-Recall curve analysis
- **Matrix** — Sparse matrix representation
- **ggplot2** — Data visualization
- **dplyr** — Data manipulation

## Project Structure

```text
Claim-Frequency-Predictor/
│
├── README.md
├── [R script]
├── [Dataset]
└── [Figures]
````

## Key Takeaways

This project provided hands-on experience with:

* Binary classification for insurance claims
* Data preprocessing and one-hot encoding
* Stratified train-test splitting
* Class imbalance and SMOTE
* Logistic Regression
* Decision Trees
* Random Forest
* XGBoost
* Cross-validation and hyperparameter tuning
* Classification threshold analysis
* Model evaluation using Recall, F1 Score, and PR curves

## Team Project

This was completed as a team project, with responsibilities divided across data processing, modeling, visualization, and model evaluation.


## Methodology

### 1. Exploratory Data Analysis

The dataset was first examined for:
- Data structure and variable types
- Variable names
- Missing values
- Class distribution of the target variable

The `policy_id` identifier was removed as it does not provide predictive information.

### 2. Data Preprocessing

Categorical variables were transformed using one-hot encoding.

The dataset was then split into:
- **80% training data**
- **20% test data**

Stratified sampling was used to preserve the class distribution between the training and test sets.

### 3. Class Imbalance

Because claim occurrences were relatively rare, **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to the training data to address class imbalance.

The test set was kept separate for out-of-sample evaluation.

### 4. Classification Models

Four classification models were trained and compared:

- Logistic Regression
- XGBoost
- Decision Tree
- Random Forest

### 5. XGBoost Cross-Validation and Tuning

For XGBoost, the `caret` framework was used to perform **5-fold cross-validation and hyperparameter tuning**.

A predefined grid of XGBoost hyperparameters was evaluated, with **Recall** used as the primary metric for selecting the best parameter combination.

The selected hyperparameters were then used to train the final XGBoost model using the `xgboost` package.

### 6. Model Evaluation

Model performance was evaluated on the held-out test set using:

- **Recall**
- **F1 Score**
- **Precision-Recall (PR) Curve**
- **Confusion Matrix**

Because claim prediction is an imbalanced classification problem, Recall was emphasized to measure the model's ability to identify policyholders who actually filed claims.

For Logistic Regression and XGBoost, the relationship between the classification threshold and Recall was also examined.

### 7. Model Comparison

The final performance of the four models was summarized and compared using Recall and F1 Score.

<!-- Insert your final comparison table here -->

| Model | F1 Score | Recall |
|---|---:|---:|
| Logistic Regression | TBD | TBD |
| XGBoost | TBD | TBD |
| Decision Tree | TBD | TBD |
| Random Forest | TBD | TBD |

## Results

<!-- Insert your model comparison result or visualization here -->

The models were compared based on their out-of-sample performance. The analysis focused on the trade-off between identifying actual claims and maintaining overall classification performance.

### Suggested Visualizations

The following visualizations can be included here:

1. **Recall vs. Threshold — Logistic Regression**
2. **Recall vs. Threshold — XGBoost**
3. **PR Curve — XGBoost**
4. **PR Curve — Decision Tree**
5. **PR Curve — Random Forest**
6. **Confusion Matrices**
7. **Model Performance Comparison**

