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
