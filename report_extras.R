## =============================================================================
##  Extras: Brier Score + Calibration Plot
## =============================================================================
##
##  WHAT THIS SCRIPT DOES
##  ----------------------------------------------------------------------------
##  Adds probability-quality diagnostics on top of the main comparison:
##    - Brier score per model/scenario (are the predicted probabilities close
##      to the actual outcomes, not just good at ranking policies?)
##    - A calibration plot per scenario (do policies given ~20% predicted risk
##      actually claim about 20% of the time?)
##
##  HOW TO RUN
##  ----------------------------------------------------------------------------
##  Run insurance_claim_smote_comparison.R FIRST, then run this file in the
##  SAME R session. It needs `results`, `summary_tbl`, `X_test`, `y_test`, and
##  `POS`, all created by that script. Nothing is retrained here -- the
##  already-fitted models are only asked for predictions again, so this runs
##  in a few seconds.
##
##  HOW TO READ THE OUTPUT
##  ----------------------------------------------------------------------------
##  Brier score  = average of (predicted probability - actual 0/1 outcome)^2.
##                 Lower is better. It judges how good the PROBABILITIES are,
##                 not just how well policies are ranked.
##  Calibration  = among policies the model gives ~20%, do ~20% really claim?
##                 The plot groups test policies by predicted probability and
##                 compares the average prediction with the actual claim rate.
##                 Points on the dashed diagonal = well calibrated.
##  SMOTE tends to push predicted probabilities upward, so SMOTE models often
##  show worse Brier scores and calibration even when their ranking of
##  high-risk vs. low-risk policies (PR-AUC) is just as good or better.
## =============================================================================


## 1. Brier score ---------------------------------------------------------------

y01       <- as.integer(y_test == POS)
base_rate <- mean(y01)

brier_table <- bind_rows(lapply(results, function(r) {
  bind_rows(lapply(names(r$fits), function(nm) {
    prob <- r$fits[[nm]]$predict_prob(X_test)
    data.frame(Scenario = r$scenario, Model = nm, Brier = mean((prob - y01)^2))
  }))
}))

cat("\n===== Brier score (test set, lower = better) =====\n")
cat("Reference: always predicting the claim rate gives",
    round(base_rate * (1 - base_rate), 4), "\n")
print(brier_table %>% mutate(Brier = round(Brier, 4)), row.names = FALSE)


## 2. Calibration plot ----------------------------------------------------------

calibration_df <- function(model, model_name, scenario, bins = 10) {
  prob   <- model$predict_prob(X_test)
  breaks <- unique(quantile(prob, probs = seq(0, 1, length.out = bins + 1)))
  data.frame(prob = prob, y = as.integer(y_test == POS)) %>%
    mutate(bin = cut(prob, breaks, include.lowest = TRUE)) %>%
    group_by(bin) %>%
    summarise(mean_pred = mean(prob), obs_rate = mean(y), n = n(), .groups = "drop") %>%
    mutate(Model = model_name, Scenario = scenario)
}

cal_all <- bind_rows(lapply(results, function(r) {
  bind_rows(lapply(names(r$fits), function(nm) {
    calibration_df(r$fits[[nm]], nm, r$scenario)
  }))
}))

print(
  ggplot(cal_all, aes(x = mean_pred, y = obs_rate, colour = Model)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey40") +
    geom_line() +
    geom_point() +
    facet_wrap(~ Scenario) +
    labs(
      title = "Calibration (test set, 10 equal-size bins)",
      x = "Mean predicted probability", y = "Observed claim rate"
    ) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
)