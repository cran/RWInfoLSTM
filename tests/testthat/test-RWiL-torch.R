# Tests that need the libtorch backend. Each one is skipped automatically when
# torch is not fully installed; the ones that actually train a network are also
# skipped on CRAN to keep check times short.

# ---- Loss, model and preprocessing building blocks -------------------------

test_that(".rwil_loss blends the two MSE terms with weight lambda", {
  skip_if_no_torch()

  y_pred   <- torch::torch_tensor(matrix(c(1, 2, 3), ncol = 1))
  y_actual <- torch::torch_tensor(matrix(c(1, 2, 5), ncol = 1))
  y_rw     <- torch::torch_tensor(matrix(c(2, 2, 3), ncol = 1))

  mse_actual <- 4 / 3   # mean(c(0, 0, 4))
  mse_rw     <- 1 / 3   # mean(c(1, 0, 0))

  out <- .rwil_loss(y_pred, y_actual, y_rw, lambda = 0.25)
  expect_equal(out$actual$item(), mse_actual, tolerance = 1e-5)
  expect_equal(out$rw$item(),     mse_rw,     tolerance = 1e-5)
  expect_equal(out$total$item(),
               0.75 * mse_actual + 0.25 * mse_rw, tolerance = 1e-5)

  # lambda = 0 is the plain LSTM loss, lambda = 1 is fully RW-guided
  expect_equal(.rwil_loss(y_pred, y_actual, y_rw, 0)$total$item(),
               mse_actual, tolerance = 1e-5)
  expect_equal(.rwil_loss(y_pred, y_actual, y_rw, 1)$total$item(),
               mse_rw, tolerance = 1e-5)
})

test_that(".rwil_lstm_model returns one prediction per input sequence", {
  skip_if_no_torch()

  x <- torch::torch_randn(5, 3, 1)         # (batch, sequence, features)
  for (nl in 1:2) {
    model <- .rwil_lstm_model(hidden_size = 4L, num_layers = nl, dropout = 0.1)
    expect_equal(as.integer(model(x)$shape), c(5L, 1L))
  }
})

test_that(".rwil_preprocess_full aligns sequences and the validation split", {
  skip_if_no_torch()

  y  <- cumsum(rnorm(30, mean = 0.5))
  rw <- .fit_rw_full(y, verbose = FALSE)
  pp <- .rwil_preprocess_full(y, rw, seq_len = 3, batch_size = 8)

  expect_equal(pp$n_seqs, 27)               # 30 observations - lag of 3
  expect_equal(pp$val_split, 21)            # first 80% of the sequences
  expect_equal(nrow(pp$X_tr), 21)
  expect_equal(nrow(pp$X_val), 6)
  expect_equal(ncol(pp$X_tr), 3)
  expect_equal(pp$BATCH, 8)

  expect_length(pp$last_seq_scaled, 3)
  expect_equal(pp$last_seq_scaled, pp$scaler$scale(tail(y, 3)))
  expect_equal(pp$scaler$unscale(pp$y_all), y[-(1:3)])
})

# ---- End-to-end training on tiny problems ----------------------------------

test_that("RWiL() trains in manual mode and returns a well-formed fit", {
  skip_if_no_torch()
  skip_on_cran()

  y <- toy_series()
  expect_output(
    fit <- RWiL(y, mode = "manual", lag = 3, hidden_size = 4, num_layers = 1,
                lr = 0.01, epochs = 5, patience = 3, lambda_list = 0.5,
                verbose = FALSE),
    NA                                       # verbose = FALSE prints nothing
  )

  expect_s3_class(fit, "RWiL")
  expect_named(fit$results, c("standard", "lambda_0.5"))
  expect_true(fit$rw$rw_type %in% c("simple", "drift"))
  expect_equal(fit$data_info$N, length(y))
  expect_equal(fit$data_info$lag, 3)

  # the best lambda is the one with the lowest training RMSE
  tr_rmse <- vapply(fit$results, function(r) r$tr_rmse, numeric(1))
  expect_equal(fit$meta$best_lam_nm, names(which.min(tr_rmse)))

  for (r in fit$results) {
    expect_true(is.finite(r$tr_rmse))
    expect_true(is.finite(r$tr_mae))
    expect_length(r$tr_preds, length(y) - 3)
  }

  expect_equal(nrow(fit$best_model$metrics), 4)
  expect_equal(nrow(fit$best_model$predictions), length(y) - 3)
  expect_named(fit$best_model$loss_history,
               c("Total_Loss", "LSTM_Loss", "RW_Guided_Loss"))
})

test_that("RWiL() grid search records its tuning log", {
  skip_if_no_torch()
  skip_on_cran()

  fit <- RWiL(toy_series(), mode = "auto", lag = 3,
              hidden_grid = 4, layers_grid = 1L, lr_grid = 0.01,
              dropout_grid = 0, tune_epochs = 2,
              epochs = 5, patience = 3, lambda_list = 0.5, verbose = FALSE)

  expect_s3_class(fit, "RWiL")
  expect_equal(nrow(fit$tuning_log), 1)
  expect_true(all(c("hidden", "layers", "lr", "dropout", "val_rmse")
                  %in% names(fit$tuning_log)))
  expect_equal(fit$best_hp$hidden, 4)
})

test_that("predictRWiL() returns a forecast of the requested length", {
  skip_if_no_torch()
  skip_on_cran()

  y   <- toy_series()
  fit <- RWiL(y, mode = "manual", lag = 3, hidden_size = 4, lr = 0.01,
              epochs = 5, patience = 3, lambda_list = 0.5, verbose = FALSE)

  fc <- predictRWiL(fit, h = 4)
  expect_s3_class(fc, "RWiL_forecast")
  expect_named(fc, c("h", "time_idx", "forecast"))
  expect_equal(fc$h, 1:4)
  expect_equal(fc$time_idx, length(y) + 1:4)
  expect_true(all(is.finite(fc$forecast)))
  expect_equal(attr(fc, "lambda"), fit$meta$best_lam_nm)

  # choosing a specific lambda
  expect_equal(attr(predictRWiL(fit, h = 2, lambda = 0),   "lambda"), "standard")
  expect_equal(attr(predictRWiL(fit, h = 2, lambda = 0.5), "lambda"), "lambda_0.5")
  expect_error(predictRWiL(fit, h = 2, lambda = 0.3), "lambda not found")

  # new observations shift the time index of the forecast
  fc_new <- predictRWiL(fit, h = 2, newdata = tail(y, 2) + 1)
  expect_equal(fc_new$time_idx, length(y) + 2 + 1:2)

  # predict() is a thin wrapper around predictRWiL()
  expect_equal(nrow(predict(fit, h = 3)), 3)
})

test_that("AccuracyRWiL() can be silenced with verbose = FALSE", {
  skip_if_no_torch()
  skip_on_cran()

  y   <- toy_series()
  fit <- RWiL(y, mode = "manual", lag = 3, hidden_size = 4, lr = 0.01,
              epochs = 5, patience = 3, lambda_list = 0.5, verbose = FALSE)
  actual <- tail(y, 1) + cumsum(rep(0.5, 4))

  expect_output(acc <- AccuracyRWiL(fit, actual = actual), "AccuracyRWiL")
  expect_output(quiet <- AccuracyRWiL(fit, actual = actual, verbose = FALSE), NA)

  expect_s3_class(quiet, "data.frame")
  expect_named(quiet, c("Lambda", "h", "RMSE", "MAE", "SMAPE", "MAPE"))
  expect_equal(quiet$h, 4)
  expect_equal(quiet$RMSE, acc$RMSE)         # same numbers either way
  expect_equal(nrow(attr(quiet, "comparison")), 4)
})

test_that("print(), summary() and the plots work on a fitted model", {
  skip_if_no_torch()
  skip_on_cran()

  y   <- toy_series()
  fit <- RWiL(y, mode = "manual", lag = 3, hidden_size = 4, lr = 0.01,
              epochs = 5, patience = 3, lambda_list = 0.5, verbose = FALSE)

  expect_output(print(fit), "Random Walk-Informed LSTM")
  expect_output(summary(fit), "RWiL Summary")

  expect_s3_class(plotRWiL(fit, h = 4), "ggplot")
  expect_s3_class(plotRWiL(fit, h = 4, actual = tail(y, 1) + 1:4, n_hist = 10),
                  "ggplot")
  expect_s3_class(plotlambda_rwil(fit), "ggplot")
  expect_s3_class(plot(fit, which = "lambda"), "ggplot")
  expect_s3_class(plot(fit), "ggplot")
})
