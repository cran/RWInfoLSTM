# Shared helpers for the RWiL tests (sourced automatically by testthat).

# Skip a test unless the libtorch backend is actually available.
skip_if_no_torch <- function() {
  testthat::skip_if_not_installed("torch")
  testthat::skip_if_not(
    torch::torch_is_installed(),
    "libtorch backend is not installed"
  )
}

# A short, reproducible random-walk-like series.
toy_series <- function(n = 40) {
  set.seed(1)
  cumsum(rnorm(n, mean = 0.5, sd = 1))
}

# A minimal object shaped like a fitted "RWiL" model. It carries no torch
# model, so it can only be used to test argument handling and S3 dispatch.
make_stub_fit <- function() {
  structure(
    list(
      meta      = list(best_lam_nm = "lambda_0.5"),
      results   = list(
        standard   = list(lambda = 0),
        lambda_0.5 = list(lambda = 0.5)
      ),
      data_info = list(N = 30L, lag = 3L)
    ),
    class = "RWiL"
  )
}

# A hand-built forecast object, as returned by predictRWiL().
make_forecast <- function() {
  fc <- data.frame(h = 1:3, time_idx = 31:33, forecast = c(1.5, 2.5, 3.5))
  attr(fc, "lambda") <- "lambda_0.5"
  class(fc) <- c("RWiL_forecast", "data.frame")
  fc
}
