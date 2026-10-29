# ============================================================
# RWiL -- Out-of-Sample Forecasting: predictRWiL + AccuracyRWiL
# ============================================================
#' @importFrom utils tail
NULL

#' Forecast h steps ahead from a fitted RWiL model
#'
#' @param fit     A RWiL object returned by RWiL().
#' @param h       Number of steps to forecast. Default 1.
#' @param lambda  Lambda value to use. NULL = best (lowest Train RMSE).
#' @param newdata Optional numeric vector of new observations after training.
#'
#' @return Data frame with columns: h, time_idx, forecast.
#' @examples
#' # Runs only when the libtorch backend is installed (torch::install_torch())
#' if (torch::torch_is_installed()) {
#'   set.seed(1)
#'   y <- cumsum(rnorm(40, 0.5, 1))
#'   fit <- RWiL(y, mode = "manual", lag = 3, hidden_size = 4, lr = 0.01,
#'               epochs = 5, patience = 5, lambda_list = 0.5,
#'               verbose = FALSE, seed = 1)
#'   predictRWiL(fit, h = 6)
#' }
#' @export
predictRWiL <- function(fit, h = 1, lambda = NULL, newdata = NULL) {

  nm <- if (is.null(lambda)) fit$meta$best_lam_nm else {
    if (lambda == 0) "standard" else paste0("lambda_", lambda)
  }
  if (!(nm %in% names(fit$results)))
    stop("lambda not found. Available: ",
         paste(sapply(fit$results, function(r) r$lambda), collapse = ", "))

  model  <- fit$results[[nm]]$model
  scaler <- fit$data_info$scaler
  seq_ln <- fit$data_info$lag

  # Starting sequence
  cur_seq <- fit$data_info$last_seq_scaled

  if (!is.null(newdata)) {
    nd_scaled  <- scaler$scale(as.numeric(newdata))
    all_scaled <- c(scaler$scale(fit$data_info$data), nd_scaled)
    cur_seq    <- tail(all_scaled, seq_ln)
  }

  # Recursive h-step forecast
  forecasts_scaled <- numeric(h)
  model$eval()
  for (t in seq_len(h)) {
    x_t <- torch::torch_tensor(
      array(cur_seq, dim = c(1L, seq_ln, 1L)),
      dtype = torch::torch_float()
    )
    pred_sc <- torch::with_no_grad({
      as.numeric(model(x_t)$detach()$reshape(list(1L)))
    })
    forecasts_scaled[t] <- pred_sc
    cur_seq <- c(cur_seq[-1L], pred_sc)
  }

  forecasts <- scaler$unscale(forecasts_scaled)

  n_new    <- if (!is.null(newdata)) length(newdata) else 0L
  last_idx <- fit$data_info$N + n_new
  time_idx <- last_idx + seq_len(h)

  df <- data.frame(
    h        = seq_len(h),
    time_idx = time_idx,
    forecast = round(forecasts, 6)
  )

  attr(df, "lambda") <- nm
  class(df) <- c("RWiL_forecast", "data.frame")
  df
}

#' Out-of-sample accuracy for RWiL forecasts vs actual values
#'
#' @param fit     A RWiL object returned by RWiL().
#' @param actual  Numeric vector of actual values after training.
#' @param lambda  Lambda to use. NULL = best.
#' @param verbose Logical. If \code{TRUE} (default), prints a formatted
#'                accuracy report to the console. Set to \code{FALSE} to
#'                suppress console output and only receive the returned
#'                data frame.
#'
#' @return Data frame with RMSE, MAE, SMAPE, MAPE.
#' @examples
#' # Runs only when the libtorch backend is installed (torch::install_torch())
#' if (torch::torch_is_installed()) {
#'   set.seed(1)
#'   y <- cumsum(rnorm(40, 0.5, 1))
#'   fit <- RWiL(y, mode = "manual", lag = 3, hidden_size = 4, lr = 0.01,
#'               epochs = 5, patience = 5, lambda_list = 0.5,
#'               verbose = FALSE, seed = 1)
#'   future <- tail(y, 1) + cumsum(rnorm(6, 0.5, 1))
#'   AccuracyRWiL(fit, actual = future)
#' }
#' @export
AccuracyRWiL <- function(fit, actual, lambda = NULL, verbose = TRUE) {
  actual <- as.numeric(actual)
  h      <- length(actual)
  fc     <- predictRWiL(fit, h = h, lambda = lambda)
  pred   <- fc$forecast
  act    <- actual

  rmse  <- sqrt(mean((act - pred)^2))
  mae   <- mean(abs(act - pred))
  denom <- abs(act) + abs(pred); mask <- denom != 0
  smape <- if (any(mask))
    mean(2 * abs(act[mask] - pred[mask]) / denom[mask]) * 100 else 0
  mape  <- if (any(act != 0))
    mean(abs((act[act != 0] - pred[act != 0]) / act[act != 0])) * 100 else NA

  lam_label <- gsub("lambda_", "", attr(fc, "lambda"))

  if (verbose) {
    cat("\n=== AccuracyRWiL -- Out-of-Sample Accuracy ===\n")
    cat(sprintf("  Lambda   : %s\n",   lam_label))
    cat(sprintf("  Horizon  : %d steps\n", h))
    cat(sprintf("  RMSE     : %.6f\n", rmse))
    cat(sprintf("  MAE      : %.6f\n", mae))
    cat(sprintf("  SMAPE    : %.4f%%\n", smape))
    if (!is.na(mape)) cat(sprintf("  MAPE     : %.4f%%\n", mape))
    cat("===============================================\n\n")
  }

  df <- data.frame(
    Lambda = lam_label, h = h,
    RMSE   = round(rmse,  6),
    MAE    = round(mae,   6),
    SMAPE  = round(smape, 4),
    MAPE   = if (!is.na(mape)) round(mape, 4) else NA_real_
  )

  comparison <- data.frame(
    Step      = seq_len(h),
    Actual    = round(act,  4),
    Forecast  = round(pred, 4),
    Error     = round(act - pred, 4),
    Abs_Error = round(abs(act - pred), 4)
  )
  attr(df, "comparison") <- comparison
  invisible(df)
}

#' @export
print.RWiL_forecast <- function(x, ...) {
  lam <- gsub("lambda_", "", attr(x, "lambda"))
  cat(sprintf("\n=== RWiL Forecast  |  lambda=%s  |  h=%d steps ===\n",
              lam, nrow(x)))
  print.data.frame(x, ...)
  cat("\n")
  invisible(x)
}
