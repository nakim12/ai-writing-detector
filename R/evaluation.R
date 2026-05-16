# Cross-validation and metric helpers used across the notebooks.
#
# All notebooks share `cv_folds()` and `cv_metrics()` so model comparisons
# are run against identical folds and the same metric set.

suppressPackageStartupMessages({
  library(rsample)
  library(yardstick)
  library(tune)
  library(workflows)
  library(dplyr)
  library(tibble)
  library(tidyr)
})

RANDOM_SEED <- 42L

#' Stratified 5-fold CV that all candidate models share for a fair comparison.
cv_folds <- function(train_data, v = 5, seed = RANDOM_SEED) {
  set.seed(seed)
  rsample::vfold_cv(train_data, v = v, strata = generated)
}

#' Metric set used during model selection and final test-set evaluation.
#' AI is treated as the positive class (event_level = "second" because the
#' factor levels are c("human", "ai")).
cv_metrics <- function() {
  yardstick::metric_set(
    yardstick::accuracy,
    yardstick::roc_auc,
    yardstick::f_meas,
    yardstick::precision,
    yardstick::recall
  )
}

#' Fit a workflow across CV folds and return a mean/std summary.
cv_summary <- function(workflow, folds, metrics = cv_metrics()) {
  res <- tune::fit_resamples(
    workflow,
    resamples = folds,
    metrics = metrics,
    control = tune::control_resamples(save_pred = FALSE, event_level = "second")
  )

  tune::collect_metrics(res) |>
    dplyr::select(metric = .metric, mean, std_err, n)
}

#' Final-test-set report for a fitted workflow.
#'
#' @param fitted_wf A workflow already fit on the full training set.
#' @param test_data The held-out test tibble (must contain `text`, `generated`).
#' @return A list with the metrics tibble, confusion matrix, and predictions.
test_set_report <- function(fitted_wf, test_data, metrics = cv_metrics()) {
  preds <- predict(fitted_wf, test_data, type = "class") |>
    dplyr::bind_cols(predict(fitted_wf, test_data, type = "prob")) |>
    dplyr::bind_cols(test_data |> dplyr::select(generated))

  metric_tbl <- metrics(
    preds,
    truth = generated,
    estimate = .pred_class,
    .pred_ai,
    event_level = "second"
  )

  list(
    metrics = metric_tbl,
    confusion_matrix = yardstick::conf_mat(preds, truth = generated, estimate = .pred_class),
    predictions = preds
  )
}
