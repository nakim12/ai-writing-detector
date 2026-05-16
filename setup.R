# One-time setup: install the R packages this project depends on.
# Run once after cloning:  source("setup.R")

required_packages <- c(
  # Core
  "tidyverse",
  # Modeling
  "tidymodels",
  "textrecipes",
  "glmnet",        # logistic regression with regularization
  "discrim",       # parsnip wrapper for naive Bayes
  "naivebayes",    # naive Bayes engine
  "LiblineaR",     # linear SVM engine
  "xgboost",       # boosted trees engine
  "vip",           # variable importance for the final model
  # NLP utilities
  "tidytext",
  "stopwords",
  # Notebook rendering
  "knitr",
  "rmarkdown"
)

missing <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]

if (length(missing)) {
  message("Installing missing packages: ", paste(missing, collapse = ", "))
  install.packages(missing, repos = "https://cloud.r-project.org")
} else {
  message("All required packages are already installed.")
}
