\name{RWiL}
\alias{RWiL}
\title{Random Walk-Informed LSTM (RWiL)}
\description{
Trains an LSTM on the FULL dataset using the Random Walk-informed composite loss:

\preformatted{
  Loss = (1-lambda) x MSE(y_hat, y) + lambda x MSE(y_hat, y_RW)
}

Trains on ALL data -- no train/test split.
Use predictRWiL() for out-of-sample forecasting.
An internal 20 percent validation split handles early stopping.

The Random Walk model is auto-selected between:
\enumerate{
  \item Simple RW: y_hat[t] = y[t-1]
  \item Drift  RW: y_hat[t] = y[t-1] + d, where d = mean(diff(y))
}
Selection is by lowest in-sample RMSE.
}
\usage{
RWiL(
  data,
  mode = c("auto", "manual"),
  lag,
  hidden_size = 32,
  num_layers = 1,
  lr = 0.001,
  dropout = 0.0,
  epochs = 500,
  patience = 30,
  tune_epochs = 50,
  batch_size = 16,
  lambda_list = seq(0.1, 0.9, 0.1),
  hidden_grid = c(8, 16, 24, 32, 40),
  layers_grid = c(1L, 2L),
  lr_grid = c(0.001, 0.003, 0.005),
  dropout_grid = c(0.0, 0.1, 0.2),
  verbose = TRUE,
  seed = NULL
)
}
\arguments{
  \item{data}{Numeric vector. Full time series -- ALL observations used for training.}

  \item{mode}{\code{"auto"} runs grid-search LSTM tuning.
    \code{"manual"} uses user-supplied hyperparameters.}

  \item{lag}{Look-back window length. COMPULSORY -- no default.
    Example: lag = 3 means model uses last 3 values to predict next.}

  \item{hidden_size}{LSTM hidden units (manual mode). Default 32.}

  \item{num_layers}{Stacked LSTM layers (manual mode). Default 1.}

  \item{lr}{Learning rate (manual mode). Default 0.001.}

  \item{dropout}{Dropout rate (manual mode). Default 0.0.}

  \item{epochs}{Maximum training epochs. Default 500.}

  \item{patience}{Early-stopping patience. Default 30.}

  \item{tune_epochs}{Epochs per grid-search combo (auto). Default 50.}

  \item{batch_size}{Mini-batch size. Default 16.}

  \item{lambda_list}{Lambda values to test. Default seq(0.1, 0.9, 0.1).
    lambda = 0.0: Standard LSTM (no RW guidance).
    lambda = 1.0: Fully RW-guided LSTM.}

  \item{hidden_grid}{Hidden-unit candidates (auto mode).}

  \item{layers_grid}{Layer-count candidates (auto mode).}

  \item{lr_grid}{Learning-rate candidates (auto mode).}

  \item{dropout_grid}{Dropout candidates (auto mode).}

  \item{verbose}{Print progress. Default TRUE.}

  \item{seed}{Optional integer random seed for reproducible results. Default
    \code{NULL}: no seed is set (call \code{set.seed()} and
    \code{torch::torch_manual_seed()} beforehand, or pass a value here).}
}
\value{
S3 object of class \code{"RWiL"} with elements:
\describe{
  \item{results}{Named list per lambda. Each contains: tr_rmse, tr_mae,
    tr_mape, tr_smape, tr_preds, tr_actuals, val_rmse, loss_hist,
    time, lambda, model.}
  \item{best_model}{Complete best lambda summary: lambda, hyperparams,
    rw (type, drift, rmse_simple, rmse_drift), metrics, predictions,
    loss_history, model object, train_time.}
  \item{rw}{Random Walk results: rw_type, drift, rmse_simple, rmse_drift.}
  \item{best_hp}{Best LSTM hyperparameters.}
  \item{tuning_log}{Grid-search log (auto mode only).}
  \item{meta}{Settings: lag, lambda_list, mode, best_lam_nm,
    selection = "Train RMSE".}
  \item{data_info}{data, N, scaler, last_seq_scaled, lag.}
}
}
\examples{
# Small toy example (a few seconds)
# Runs only when the libtorch backend is installed (torch::install_torch())
if (torch::torch_is_installed()) {
  set.seed(1)
  y <- cumsum(rnorm(40, 0.5, 1))
  fit <- RWiL(y, mode = "manual", lag = 3, hidden_size = 4, lr = 0.01,
              epochs = 5, patience = 5, lambda_list = 0.5,
              verbose = FALSE, seed = 1)
  print(fit)
  summary(fit)
  predictRWiL(fit, h = 6)
  fit$rw$rw_type     # "simple" or "drift"
}

\donttest{
# Auto mode: grid search over the LSTM hyperparameters (takes longer)
if (torch::torch_is_installed()) {
  set.seed(1)
  y <- cumsum(rnorm(60, 0.5, 2))
  fit_auto <- RWiL(y, mode = "auto", lag = 3,
                   hidden_grid = c(4, 8), layers_grid = 1L,
                   lr_grid = 0.01, dropout_grid = 0,
                   tune_epochs = 5, epochs = 20, patience = 5,
                   lambda_list = c(0.3, 0.7), verbose = FALSE, seed = 1)
  summary(fit_auto)
  plotRWiL(fit_auto, h = 12)
  plotlambda_rwil(fit_auto)
}
}
}
