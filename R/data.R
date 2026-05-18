# Loading the Human-vs-AI essays dataset and producing a stratified split.
#
# The dataset is expected to live at data/raw/AI_Human.csv (downloaded
# manually from HuggingFace). The loader normalizes column names so the rest of
# the codebase can rely on `text` and `generated` being present.

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(rsample)
})

RANDOM_SEED <- 42L
TEST_PROP <- 0.20

DEFAULT_RAW_PATH <- file.path("data", "raw", "AI_Human.csv")

#' Load the raw essays CSV and return a tidy tibble.
#'
#' @param path Path to the CSV. Defaults to data/raw/AI_Human.csv.
#' @return A tibble with columns `text` (chr) and `generated` (factor: "human", "ai").
load_essays <- function(path = DEFAULT_RAW_PATH) {
  if (!file.exists(path)) {
    stop(sprintf(
      "Could not find dataset at %s. Download it from Kaggle and place it at data/raw/AI_Human.csv.",
      path
    ))
  }

  df <- readr::read_csv(path, show_col_types = FALSE)
  names(df) <- tolower(trimws(names(df)))

  # Some Kaggle versions ship the label under a different name.
  if (!"generated" %in% names(df)) {
    for (alt in c("label", "is_ai", "ai_generated")) {
      if (alt %in% names(df)) {
        df <- dplyr::rename(df, generated = !!alt)
        break
      }
    }
  }

  if (!all(c("text", "generated") %in% names(df))) {
    stop(sprintf(
      "Expected columns 'text' and 'generated' in %s, got: %s",
      path, paste(names(df), collapse = ", ")
    ))
  }

  df |>
    dplyr::select(text, generated) |>
    tidyr::drop_na() |>
    dplyr::mutate(
      text = as.character(text),
      generated = factor(
        ifelse(as.integer(generated) == 1L, "ai", "human"),
        levels = c("human", "ai")
      )
    )
}

#' Stratified train/test split that preserves the 50/50 class balance.
#'
#' @param df Tibble produced by `load_essays()`.
#' @param prop Proportion to retain in training. Defaults to 0.80.
#' @param seed Random seed.
#' @return An `rsplit` object from rsample.
stratified_split <- function(df, prop = 1 - TEST_PROP, seed = RANDOM_SEED) {
  set.seed(seed)
  rsample::initial_split(df, prop = prop, strata = generated)
}
