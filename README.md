# Claim-Occurrence-Predictor

An auto insurance claim prediction project using machine learning.

A team project that predicts whether an auto insurance policyholder will file a claim by comparing multiple classification models on structured policyholder data.

## Project Structure
```text
Claim-Occurrence-Predictor/
├── .gitignore
├── Claim Occurrence Predictor.R
├── LICENSE
└── README.md
```

Because our data is classified, I decided not to upload our dataset in case I disclose something sensitive.

## Methodology

### 1. Exploratory Data Analysis

The dataset was first examined for:

- Data structure and variable types
- Variable names
- Missing values
- Class distribution of the target variable

The `policy_id` identifier was removed because it does not provide predictive information.

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

Logistic Regression was used as an interpretable baseline, while tree-based methods were used to capture potentially nonlinear relationships.

### 5. Cross-Validation and Hyperparameter Tuning

For XGBoost, the `caret` framework was used to perform **5-fold cross-validation and hyperparameter tuning**.

A predefined grid of XGBoost hyperparameters was evaluated, with **Recall** used as the primary metric for selecting the best parameter combination.

The selected hyperparameters were then used to train the final XGBoost model using the `xgboost` package.

For the Decision Tree, cost-complexity pruning was performed using the **1-SE rule** based on cross-validation error.

<img width="1340" height="857" alt="決策樹" src="https://github.com/user-attachments/assets/cc28c4d2-b131-4a85-91c4-5510386221be" />


### 6. Classification Threshold Analysis

The models produce predicted probabilities that must be converted into binary predictions using a classification threshold.

For example, with a threshold of 0.50:

```text
Predicted probability >= 0.50 → Claim
Predicted probability < 0.50  → No Claim
```
Because the claim class is highly imbalanced, we examined how different thresholds affected Recall.

For Logistic Regression and XGBoost, thresholds from 0.10 to 0.90 were evaluated.

For XGBoost, Recall reached 1.00 at thresholds around 0.10–0.16. However, maximizing Recall alone can increase false positives and does not necessarily maximize F1 Score.


<img width="1375" height="715" alt="xgboost-threshold" src="https://github.com/user-attachments/assets/3e4f84c6-dd76-4ef6-a3be-1f734f25fd5a" />


This analysis illustrates the trade-off between identifying more actual claims and maintaining overall classification performance.

### 7. Model Evaluation

Model performance was evaluated on the held-out test set using:

- Recall
- F1 Score
- Precision-Recall (PR) Curve
- Confusion Matrix

Recall was emphasized because the project focuses on identifying policyholders who actually filed claims.

### 8. Model Comparison

The final performance of the four models was summarized and compared using Recall and F1 Score.

| Model | Recall | F1 Score |
|---|---:|---:|
| Logistic Regression | 0.1135 | 0.1183 |
| XGBoost | 0.6662 | **0.1648** |
| Decision Tree | 0.0951 | 0.1064 |
| Random Forest | **0.7276** | 0.1513 |

XGBoost achieved the highest F1 Score, while Random Forest achieved the highest Recall under the reported final evaluation setting.

## Results
### Threshold and Class-Weight Analysis

Additional experiments were conducted to examine how classification thresholds and class weights affect model performance.

| Model | Setting | Recall | F1 Score |
|---|---| ---:|---:|
| XGBoost | threshold = 0.45 | **0.8371** | 0.1502 |
| XGBoost | threshold = 0.50 | 0.7076 | 0.1626 |
| Random Forest | class weight = 1:1 | 0.5648 | **0.1628** |
| Random Forest | class weight = 1:1.5 | 0.7597 | 0.1479 |

For XGBoost, lowering the threshold from 0.50 to 0.45 increased Recall from 0.7076 to 0.8371, but reduced F1 Score from 0.1626 to 0.1502.

For Random Forest, increasing the relative weight of the claim class from 1:1 to 1:1.5 increased Recall from 0.5648 to 0.7597, while F1 Score decreased from 0.1628 to 0.1479.

These results demonstrate the trade-off between improving claim detection and controlling false positives.

### Confusion Matrix

Confusion matrices were used to examine the model's True Positives, True Negatives, False Positives, and False Negatives.

<img width="1471" height="738" alt="XGBoost at 05" src="https://github.com/user-attachments/assets/69061038-8361-4e30-86f3-ccdce2a1ef8c" />


### Precision-Recall Curves

Precision-Recall curves were used to further evaluate model performance under the imbalanced classification setting.

## Key Takeaways

This project provided hands-on experience with:

- Binary classification for insurance claims
- Exploratory data analysis and data preprocessing
- One-hot encoding of categorical variables
- Stratified train-test splitting
- Class imbalance and SMOTE
- Logistic Regression
- Decision Trees and cost-complexity pruning
- Random Forest
- XGBoost
- 5-fold cross-validation
- Hyperparameter tuning
- Classification threshold analysis
- Model evaluation using Recall and F1 Score
- Precision-Recall curve analysis
- Model comparison

The analysis showed that different evaluation objectives can lead to different preferred models. XGBoost achieved the highest F1 Score, while Random Forest achieved the highest Recall under the final reported setting.

## Packages and Libraries

The project uses the following R packages:

- randomForest — Random Forest classification and variable importance analysis
- rpart — Decision Tree modeling
- rpart.plot — Decision Tree visualization
- xgboost — XGBoost modeling
- caret — Stratified data splitting, cross-validation, and hyperparameter tuning
- smotefamily — SMOTE for handling class imbalance
- MLmetrics — F1 Score and Recall evaluation
- PRROC — Precision-Recall curve analysis
- Matrix — Sparse matrix representation
- ggplot2 — Data visualization
- dplyr — Data manipulation
