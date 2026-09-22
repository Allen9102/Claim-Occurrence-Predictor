# Insurance Claim Prediction: Does SMOTE Help?

Predicts whether an auto-insurance policy will file a claim (`is_claim`), comparing four models under two class-imbalance strategies — **no rebalancing** and **SMOTE oversampling** — on discrimination (ROC-AUC, PR-AUC) *and* probability quality (Brier score), with a leakage-conscious protocol and bootstrap confidence intervals.

Public Kaggle "Car Insurance Claim Prediction" dataset: 58,592 policies, 42 predictors, 6.4% claim rate.

## Project structure

```
.
├── R/
│   ├── insurance_claim_smote_comparison.R   # main pipeline: train, tune, evaluate
│   └── report_extras.R                      # Brier score + calibration plot
├── figures/                                 # key result figures (used above)
├── data/
│   └── train.csv                            # not included; see Data below
├── claim_occurence_project.pdf              # full write-up
└── README.md
```

## My Contribution

This started as a four-person course project; the analysis below reflects
my own follow-up work re-analyzing our original results with a
leakage-conscious protocol.

- **Model tuning:** Led the tuning of Logistic Regression and XGBoost,
  including the XGBoost hyperparameter grid search (`nrounds`, `max_depth`,
  `min_child_weight`) under the 5-fold CV protocol.
- **Preprocessing:** Contributed to the one-hot encoding step and the SMOTE
  integration (applying it inside CV training folds only, so it never
  leaks into a validation fold).
- **Statistical analysis:** Designed and implemented the
  bootstrap confidence intervals and paired significance tests comparing
  models and SMOTE scenarios.
- **Report writing:** Co-wrote the report with my teammates.

## Models

- Logistic Regression
- Decision Tree (`rpart`, 1-SE-pruned)
- Random Forest
- XGBoost

## Key results

**XGBoost without SMOTE gave the best discrimination** (ROC-AUC 0.664,
PR-AUC 0.113), but discrimination was modest for every model — every
PR-AUC interval sits above the 0.064 base rate, none very far above it.


<img width="1584" height="660" alt="fig1_discrimination_calibration" src="https://github.com/user-attachments/assets/15f26317-ede4-416a-bec7-6c08c4617150" />



| Scenario   | Model               | ROC-AUC | PR-AUC | Brier |
|------------|---------------------|:-------:|:------:|:-----:|
| No SMOTE   | Logistic Regression | 0.614   | 0.094  | 0.059 |
| No SMOTE   | **XGBoost**          | **0.664** | **0.113** | 0.059 |
| No SMOTE   | Decision Tree        | 0.596   | 0.087  | 0.073 |
| No SMOTE   | Random Forest        | 0.569   | 0.078  | 0.062 |
| With SMOTE | Logistic Regression | 0.612   | 0.092  | 0.143 |
| With SMOTE | XGBoost              | 0.642   | 0.101  | 0.079 |
| With SMOTE | Decision Tree        | 0.607   | 0.090  | 0.067 |
| With SMOTE | Random Forest        | 0.625   | 0.093  | 0.066 |

No-skill references: ROC-AUC 0.5, PR-AUC 0.064 (test-set claim rate),
Brier 0.060. 95% bootstrap confidence intervals and the full paired-comparison
tables are in the [project report](claim_occurence_project.pdf).

### Does SMOTE help? It depends on the model.


<img width="1584" height="638" alt="fig4_smote_effect" src="https://github.com/user-attachments/assets/bc27b510-7139-4bf4-b285-61f8f432bf94" />



| Model               | Δ ROC-AUC     | Δ PR-AUC      | Δ Brier    |
|----------------------|:-------------:|:-------------:|:----------:|
| Logistic Regression  | −0.002 (n.s.) | −0.002 (n.s.) | **+0.083** |
| XGBoost               | **−0.022**    | **−0.012**    | **+0.021** |
| Decision Tree         | +0.011 (n.s.) | +0.003 (n.s.) | **−0.006** |
| Random Forest         | **+0.056**    | **+0.016**    | **+0.004** |

*(Δ = With SMOTE − No SMOTE; bold = 95% bootstrap CI excludes 0; n.s. = not
significant)*

- **Logistic regression:** ranking unchanged, but the Brier score nearly
  tripled (0.059 → 0.143) — SMOTE badly distorts the predicted probabilities.
- **XGBoost:** SMOTE made both ranking *and* probability quality worse.
- **Decision tree:** essentially unchanged; its no-SMOTE Brier score was
  already inflated by overfit leaf probabilities, so the small improvement
  isn't really "SMOTE helping calibration."
- **Random forest:** the only model where SMOTE clearly helped ranking —
  but its no-SMOTE baseline was also the weakest of the four, so part of the
  gain may reflect a poorly tuned baseline rather than SMOTE itself.

SMOTE raises the minority share of the training data from ~6% to roughly a
third. A model fit to that inflated rate learns a claim prior several times
too high, which shifts predicted probabilities upward without necessarily
reordering policies — visible directly in the calibration plot below.

<img width="2400" height="1200" alt="calibration" src="https://github.com/user-attachments/assets/1eb47a48-74ec-4567-9179-490d4f2ee490" />

Points on the dashed diagonal mean predicted probability matches the
observed claim rate. The no-SMOTE curves track the diagonal reasonably well;
the SMOTE curves sit well below it — systematic overprediction.

<img width="2400" height="1200" alt="PR curve" src="https://github.com/user-attachments/assets/056fbd4f-8d18-42be-81fc-624338bba126" />

### Bottom line

For **ranking** uses (e.g. flagging policies for underwriting review),
XGBoost without SMOTE is the best of the four models tested, and SMOTE is
not a reliable way to improve ranking — it helped one model, hurt another,
and left two unchanged. For **pricing or reserving**, where the probability
itself is used, SMOTE should not be used without recalibration (e.g. Platt
scaling or isotonic regression): it substantially worsened the Brier score
for logistic regression and XGBoost.

## Full report

The complete methodology, all bootstrap confidence intervals, per-model
comparison tables, operating-point tables, variable-importance rankings, and
limitations are in **[`claim_occurence_project.pdf`](claim_occurence_project.pdf)**.

## Requirements

- R (developed and tested on R 4.5)
- Packages: `caret`, `MLmetrics`, `smotefamily`, `PRROC`, `xgboost`,
  `randomForest`, `rpart`, `ggplot2`, `dplyr` — installed automatically on
  first run if missing.

## Data

`data/train.csv` is not included in this repo (public Kaggle "Car Insurance
Claim Prediction" dataset). It should contain one row per policy, with:

- `policy_id` — an identifier column (dropped before modelling)
- `is_claim` — binary target (`0` / `1`)
- any number of predictor columns (numeric or categorical)

Update `DATA_PATH` at the top of `insurance_claim_smote_comparison.R` to
point at your copy of the file.

## Usage

```r
# 1. Run the main pipeline (trains all four models, both scenarios)
source("R/insurance_claim_smote_comparison.R")

# 2. Brier score + calibration plot (reuses the fitted models, no retraining)
source("R/report_extras.R")
```

Most of the runtime goes into the XGBoost grid search. Shrink `xgb_grid` in
the Configuration section for a quick smoke-test before committing to a full
run.

## Methodology

The same workflow is applied to both scenarios so the only real difference
between them is whether SMOTE is used:

1. **Split train/test first** — the test set is untouched until final scoring.
2. **Fit all preprocessing on the training set only** (one-hot encoding,
   constant/collinear column removal, min–max scaling).
3. **Tune with 5-fold CV using identical folds for every model.** SMOTE (when
   enabled) is applied *inside* each training fold only, so synthetic rows
   never leak into a validation fold.
4. **Choose the classification threshold from out-of-fold training
   predictions**, maximizing an F-beta score (`BETA = 2` favors Recall).
5. **Score the test set exactly once**, at the end, with the chosen threshold.

This avoids the most common leakage pitfalls in imbalanced classification:
preprocessing fit before the split, thresholds tuned on the test set, and
SMOTE-duplicated rows leaking into validation folds.

## Known limitations

- Single train/test split and a single random seed — repeated CV or several
  seeds would give a stronger design.
- Hyperparameters and the classification threshold are selected on the same
  CV folds (no nested CV), so CV numbers are mildly optimistic; the test-set
  numbers are the fair comparison.
- The random forest and decision tree were tuned over narrower search spaces
  than XGBoost — the finding that SMOTE helps the random forest may partly
  reflect a weaker baseline rather than the method itself.
- No separate validation set or early stopping is used for XGBoost;
  `nrounds` is chosen by grid search instead.
- No recalibration (Platt scaling, isotonic regression) was evaluated — see
  the report's Future Work section.

See the [full report](claim_occurence_project.pdf) for the complete
limitations discussion.
