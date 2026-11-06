# Insurance Claim Prediction: Does SMOTE Help?

Predicts whether an auto-insurance policy will file a claim (`is_claim`), comparing four models under two class-imbalance strategies — **no rebalancing** and **SMOTE oversampling** — on discrimination (ROC-AUC, PR-AUC) *and* probability quality (Brier score), with a leakage-conscious protocol and bootstrap confidence intervals.

Public Kaggle "Car Insurance Claim Prediction" dataset: 58,592 policies, 42 predictors, 6.4% claim rate.

## Project structure

```
.
├── R/
│   └── claim_occurrence_predictor.R   # main pipeline: train, tune, evaluate
├── figures/                           # key result figures (used above)
├── data/
│   └── train.csv                      
├── claim_occurrence_project.pdf       # full write-up
└── README.md
```

## My Contribution

- **Project leadership:** Led our team's investigation comparing supervised models for auto-insurance claim prediction.
- **Evaluation protocol:** In consultation with my teammates, designed the leakage-conscious evaluation protocol used throughout the study (train/test split before preprocessing, SMOTE confined to CV training folds, thresholds chosen on out-of-fold predictions).
- **Model building:** Responsible for the logistic regression and XGBoost models, including the XGBoost hyperparameter search under five-fold cross-validation.
- **Preprocessing:** Contributed to the SMOTE implementation and the one-hot encoding step.
- **Research:** Investigated why SMOTE may distort predicted probabilities, which became Section 5.2 of the report.
- **Statistical analysis:** Designed the analysis of Brier score, calibration analysis, and paired bootstrap confidence intervals and significance tests reported throughout the paper.
- **Visualization:** Designed confusion matrix.
- **Report writing:** Co-wrote the report with my teammates.

## Models

- Logistic Regression
- Decision Tree (`rpart`, 1-SE-pruned)
- Random Forest
- XGBoost

## Key results

**XGBoost without SMOTE gave the best discrimination** (ROC-AUC 0.666, PR-AUC 0.114), but discrimination was modest for every model — every PR-AUC interval sits above the 0.064 base rate, none very far above it.


<img width="2400" height="1200" alt="discrimination_calibration" src="https://github.com/user-attachments/assets/7fa490dd-e5e4-4c2e-8f95-482169af58af" />



| Scenario   | Model               | ROC-AUC | PR-AUC | Brier |
|------------|---------------------|:-------:|:------:|:-----:|
| No SMOTE   | Logistic Regression | 0.614   | 0.094  | 0.059 |
| No SMOTE   | **XGBoost**          | **0.666** | **0.114** | 0.059 |
| No SMOTE   | Decision Tree        | 0.596   | 0.087  | 0.073 |
| No SMOTE   | Random Forest        | 0.569   | 0.078  | 0.062 |
| With SMOTE | Logistic Regression | 0.612   | 0.092  | 0.143 |
| With SMOTE | XGBoost              | 0.647   | 0.101  | 0.071 |
| With SMOTE | Decision Tree        | 0.607   | 0.090  | 0.067 |
| With SMOTE | Random Forest        | 0.625   | 0.093  | 0.066 |

No-skill references: ROC-AUC 0.5, PR-AUC 0.064 (test-set claim rate), Brier 0.060. 95% bootstrap confidence intervals and the full paired-comparison tables are in the [project report](claim_occurrence_project.pdf).

### Does SMOTE help? It depends on the model.


<img width="2400" height="1200" alt="smote_effect" src="https://github.com/user-attachments/assets/33ed70db-079e-4b65-ad58-be3a84302f6b" />



| Model               | Δ ROC-AUC     | Δ PR-AUC      | Δ Brier    |
|----------------------|:-------------:|:-------------:|:----------:|
| Logistic Regression  | −0.002 (n.s.) | **−0.002**    | **+0.083** |
| XGBoost               | **−0.019**    | **−0.0125**   | **+0.012** |
| Decision Tree         | +0.011 (n.s.) | +0.003 (n.s.) | **−0.006** |
| Random Forest         | **+0.056**    | **+0.016**    | **+0.004** |

*(Δ = With SMOTE − No SMOTE; bold = 95% bootstrap CI excludes 0; n.s. = not
significant)*

- **Logistic regression:** ranking essentially unchanged (the PR-AUC change is statistically detectable but negligible), but the Brier score more than doubled (0.059 → 0.143) — SMOTE badly distorts the predicted probabilities.
- **XGBoost:** SMOTE made both ranking *and* probability quality worse.
- **Decision tree:** essentially unchanged; its no-SMOTE Brier score was already inflated by overfit leaf probabilities, so the small improvement isn't really "SMOTE helping calibration."
- **Random forest:** the only model where SMOTE clearly helped ranking — but its no-SMOTE baseline was also the weakest of the four, so part of the gain may reflect a poorly tuned baseline rather than SMOTE itself.

SMOTE raises the minority share of the training data from ~6% to roughly a third. A model fit to that inflated rate learns a claim prior several times too high, which shifts predicted probabilities upward without necessarily reordering policies — visible directly in the calibration plot below.

<img width="2400" height="1200" alt="calibration" src="https://github.com/user-attachments/assets/af85226b-29bc-4696-bb16-b410ba1db1f1" />

Points on the dashed diagonal mean predicted probability matches the observed claim rate. The SMOTE curves generally sit below it — systematic overprediction, most strongly for logistic regression. Without SMOTE, logistic regression and XGBoost track the diagonal reasonably well, while the decision tree's top bin (mean predicted probability 0.63, observed claim rate about 0.11) and the flat random-forest curve show clear miscalibration.

<img width="2400" height="1200" alt="PR curve" src="https://github.com/user-attachments/assets/b1cf3817-d5b6-46d2-8e73-60a716bb807e" />

### Bottom line

For **ranking** uses (e.g. flagging policies for underwriting review), XGBoost without SMOTE is the best of the four models tested, and SMOTE is not a reliable way to improve ranking — it helped one model, hurt another, and left two essentially unchanged. For **pricing or reserving**, where the probability itself is used, SMOTE should not be used without recalibration (e.g. Platt scaling or isotonic regression): it substantially worsened the Brier score for logistic regression and XGBoost.

## Full report

The complete methodology, all bootstrap confidence intervals, per-model comparison tables, operating-point tables, variable-importance rankings, and limitations are in **[`claim_occurrence_project.pdf`](claim_occurrence_project.pdf)**.

## Requirements

- R (developed and tested on R 4.5)
- Packages: `caret`, `MLmetrics`, `smotefamily`, `PRROC`, `xgboost`, `randomForest`, `rpart`, `ggplot2`, `dplyr`, `tidyr` — installed automatically on first run if missing.

## Data

`data/train.csv` is not included in this repo (public Kaggle "Car Insurance Claim Prediction" dataset). It should contain one row per policy, with:

- `policy_id` — an identifier column (dropped before modelling)
- `is_claim` — binary target (`0` / `1`)
- any number of predictor columns (numeric or categorical)

Update `DATA_PATH` at the top of `claim_occurrence_predictor.R` to point at your copy of the file.

## Methodology

The same workflow is applied to both scenarios so the only real difference between them is whether SMOTE is used:

1. **Split train/test first** — the test set is untouched until final scoring.
2. **Fit all preprocessing on the training set only** (one-hot encoding, constant/collinear column removal, min–max scaling).
3. **Tune with 5-fold CV using identical folds for every model.** SMOTE (when enabled) is applied *inside* each training fold only, so synthetic rows never leak into a validation fold.
4. **Choose the classification threshold from out-of-fold training predictions**, maximizing an F-beta score (`BETA = 2` favors Recall).
5. **Score the test set exactly once**, at the end, with the chosen threshold.

This avoids the most common leakage pitfalls in imbalanced classification: preprocessing fit before the split, thresholds tuned on the test set, and SMOTE-duplicated rows leaking into validation folds.

## Known limitations

- Single train/test split and a single random seed — repeated CV or several seeds would give a stronger design.
- Hyperparameters and the classification threshold are selected on the same CV folds (no nested CV), so CV numbers are mildly optimistic; the test-set numbers are the fair comparison.
- The random forest and decision tree were tuned over narrower search spaces than XGBoost — the finding that SMOTE helps the random forest may partly reflect a weaker baseline rather than the method itself.
- No separate validation set or early stopping is used for XGBoost; `nrounds` is chosen by grid search instead.
- No recalibration (Platt scaling, isotonic regression) was evaluated — see the report's Future Work section.
- Several selected hyperparameters lie at the edge of their grids, so wider grids might change the results.
- Six integer-coded vehicle attributes were used as numeric variables rather than one-hot encoded, which imposes a linear effect on the logit and was not tested.
- Calibration was assessed descriptively with the Brier score and a decile plot, without confidence bands.

See the [full report](claim_occurrence_project.pdf) for the complete limitations discussion.
