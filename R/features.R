# Feature engineering helpers.
#
# Two families of features are supported:
#   1. `tfidf_recipe()`  — a textrecipes recipe that tokenizes and computes
#      TF-IDF inside the modeling workflow, so the vocabulary is learned only
#      from training folds (no leakage during cross-validation).
#   2. `stylometric_features()` — simple hand-crafted features (sentence
#      length, punctuation rates, type-token ratio, etc.) that the team can
#      use as descriptive variables or as additional inputs to a classifier.

suppressPackageStartupMessages({
  library(recipes)
  library(textrecipes)
  library(dplyr)
  library(stringr)
  library(tibble)
})

#' Build a TF-IDF recipe with sensible defaults for essay-length text.
#'
#' @param formula A formula such as `generated ~ text`.
#' @param data Training data used to declare the recipe roles.
#' @param max_tokens Vocabulary cap after token filtering.
#' @param min_times Minimum number of essays a token must appear in.
#' @param ngram Maximum n-gram length (1 = unigrams only, 2 = uni+bigrams).
#' @return A `recipes::recipe` object.
tfidf_recipe <- function(formula, data,
                         max_tokens = 20000L,
                         min_times = 2L,
                         ngram = 2L) {
  rec <- recipes::recipe(formula, data = data) |>
    textrecipes::step_tokenize(text) |>
    textrecipes::step_stopwords(text)

  if (ngram > 1) {
    rec <- rec |> textrecipes::step_ngram(text, num_tokens = ngram, min_num_tokens = 1L)
  }

  rec |>
    textrecipes::step_tokenfilter(text, max_tokens = max_tokens, min_times = min_times) |>
    textrecipes::step_tfidf(text)
}

#' Compute readability scores for each essay.
#'
#' Wraps `quanteda.textstats::textstat_readability()` so that the team can
#' use industry-standard readability indices as additional descriptive or
#' predictive features.
#'
#' @param texts Character vector of essays.
#' @param measures Which indices to compute. Defaults to Flesch reading ease,
#'   Flesch-Kincaid grade level, and the Automated Readability Index.
#' @return A tibble with one row per essay and one column per measure.
readability_features <- function(texts,
                                 measures = c("Flesch", "Flesch.Kincaid", "ARI")) {
  if (!requireNamespace("quanteda.textstats", quietly = TRUE)) {
    stop("Package 'quanteda.textstats' is required. Run setup.R to install it.")
  }
  out <- quanteda.textstats::textstat_readability(
    as.character(texts),
    measure = measures
  )
  tibble::as_tibble(out[, measures, drop = FALSE])
}

#' Compute a small table of hand-crafted stylometric features.
#'
#' @param texts Character vector of essays.
#' @param include_readability If TRUE, also compute Flesch / Flesch-Kincaid / ARI.
#' @return A tibble with one row per essay.
stylometric_features <- function(texts, include_readability = FALSE) {
  texts <- as.character(texts)

  word_lists <- stringr::str_extract_all(texts, "\\b\\w+\\b")
  sentence_lists <- stringr::str_split(texts, "[.!?]+")
  sentence_lists <- lapply(sentence_lists, function(s) s[nchar(trimws(s)) > 0])

  n_chars <- pmax(nchar(texts), 1L)
  n_words <- pmax(lengths(word_lists), 1L)
  n_sentences <- pmax(lengths(sentence_lists), 1L)

  avg_word_len <- vapply(word_lists, function(w) {
    if (length(w) == 0) 0 else mean(nchar(w))
  }, numeric(1))

  unique_word_count <- vapply(word_lists, function(w) length(unique(tolower(w))), integer(1))

  punct_count <- stringr::str_count(texts, "[[:punct:]]")
  comma_count <- stringr::str_count(texts, ",")
  upper_count <- stringr::str_count(texts, "[A-Z]")

  out <- tibble::tibble(
    n_chars = as.numeric(nchar(texts)),
    n_words = as.numeric(lengths(word_lists)),
    n_sentences = as.numeric(lengths(sentence_lists)),
    avg_word_len = avg_word_len,
    avg_sentence_len_words = n_words / n_sentences,
    type_token_ratio = unique_word_count / n_words,
    punct_rate = punct_count / n_chars,
    comma_rate = comma_count / n_chars,
    uppercase_rate = upper_count / n_chars
  )

  if (isTRUE(include_readability)) {
    out <- dplyr::bind_cols(out, readability_features(texts))
  }
  out
}
