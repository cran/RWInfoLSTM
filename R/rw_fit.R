# ============================================================
# RWiL -- Random Walk Fitting (auto-select: simple vs drift)
# ============================================================
#' @importFrom stats approx
NULL

# -- Interpolate NAs ------------------------------------------
.rwil_interp <- function(x) {
  idx <- which(!is.na(x))
  if (length(idx) == length(x)) return(x)
  approx(idx, x[idx], xout = seq_along(x), rule = 2)$y
}

# -- Fit Random Walk on FULL data -----------------------------
# Simple RW  : y_hat[t] = y[t-1]
# Drift RW   : y_hat[t] = y[t-1] + d,  d = mean(diff(y))
# Auto-select: lower in-sample RMSE wins
.fit_rw_full <- function(data, verbose = TRUE) {
  N <- length(data)

  # -- Simple Random Walk -----------------------------------
  fitted_simple        <- c(data[1], data[-N])
  resid_simple         <- data[-1] - fitted_simple[-1]
  rmse_simple          <- sqrt(mean(resid_simple^2))

  # -- Random Walk with Drift -------------------------------
  d                    <- mean(diff(data))
  fitted_drift         <- c(data[1], data[-N] + d)
  resid_drift          <- data[-1] - fitted_drift[-1]
  rmse_drift           <- sqrt(mean(resid_drift^2))

  if (verbose) {
    cli::cli_alert_info(
      "[RW Simple] RMSE={round(rmse_simple, 6)}")
    cli::cli_alert_info(
      "[RW Drift]  RMSE={round(rmse_drift, 6)}  drift={round(d, 6)}")
  }

  # -- Auto-select ------------------------------------------
  if (rmse_simple <= rmse_drift) {
    rw_type   <- "simple"
    fitted    <- fitted_simple
    best_rmse <- rmse_simple
    drift_val <- 0
  } else {
    rw_type   <- "drift"
    fitted    <- fitted_drift
    best_rmse <- rmse_drift
    drift_val <- d
  }

  if (verbose)
    cli::cli_alert_success(
      "[RW] Best: {rw_type}  RMSE={round(best_rmse, 6)}  drift={round(drift_val, 6)}")

  list(
    rw_type     = rw_type,
    fitted      = fitted,      # length N, aligned to data
    drift       = drift_val,
    rmse_simple = rmse_simple,
    rmse_drift  = rmse_drift,
    best_rmse   = best_rmse
  )
}
