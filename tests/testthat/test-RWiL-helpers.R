# Tests for the pure-R building blocks. None of these need torch, so they
# run everywhere (including CRAN).

# ---- Random Walk fitting ---------------------------------------------------

test_that(".fit_rw_full selects the drift walk for a steadily trending series", {
  y   <- 2 * seq_len(30)               # constant increment of 2
  res <- .fit_rw_full(y, verbose = FALSE)

  expect_equal(res$rw_type, "drift")
  expect_equal(res$drift, 2)
  expect_equal(res$rmse_simple, 2)
  expect_equal(res$rmse_drift, 0)
  expect_length(res$fitted, length(y))
  expect_equal(res$fitted[-1], y[-1])
})

test_that(".fit_rw_full selects the simple walk for a constant series", {
  y   <- rep(5, 30)
  res <- .fit_rw_full(y, verbose = FALSE)

  expect_equal(res$rw_type, "simple")
  expect_equal(res$drift, 0)
  expect_equal(res$best_rmse, 0)
  expect_equal(res$fitted, y)
})

test_that(".fit_rw_full never reports a better simple walk than drift walk", {
  set.seed(123)
  y   <- cumsum(rnorm(50, mean = 0.3))
  res <- .fit_rw_full(y, verbose = FALSE)

  expect_lte(res$rmse_drift, res$rmse_simple + 1e-12)
  expect_equal(res$best_rmse, min(res$rmse_simple, res$rmse_drift))
  expect_length(res$fitted, length(y))
})

test_that(".fit_rw_full is silent when verbose = FALSE", {
  expect_silent(.fit_rw_full(cumsum(rep(1, 25)), verbose = FALSE))
})

# ---- Missing-value interpolation ------------------------------------------

test_that(".rwil_interp fills interior and edge gaps", {
  expect_equal(.rwil_interp(c(1, NA, 3, NA, 5)), c(1, 2, 3, 4, 5))
  expect_equal(.rwil_interp(c(NA, 2, 3)), c(2, 2, 3))   # rule = 2 copies the edge
  expect_equal(.rwil_interp(c(1, 2, NA)), c(1, 2, 2))
})

test_that(".rwil_interp leaves complete data untouched", {
  x <- c(4, 5, 6)
  expect_identical(.rwil_interp(x), x)
})

# ---- Min-max scaler --------------------------------------------------------

test_that(".rwil_make_scaler maps data to [0, 1] and back", {
  x  <- c(10, 20, 30, 40)
  sc <- .rwil_make_scaler(x)

  expect_equal(sc$min, 10)
  expect_equal(sc$max, 40)
  expect_equal(sc$range, 30)
  expect_equal(sc$scale(x), c(0, 1 / 3, 2 / 3, 1))
  expect_equal(sc$unscale(sc$scale(x)), x)
  expect_equal(sc$unscale(0.5), 25)
})

test_that(".rwil_make_scaler copes with a constant series", {
  sc <- .rwil_make_scaler(rep(7, 5))

  expect_equal(sc$range, 1)                      # avoids division by zero
  expect_equal(sc$scale(7), 0)
  expect_equal(sc$unscale(sc$scale(7)), 7)
})

# ---- Supervised sequences --------------------------------------------------

test_that(".rwil_make_sequences builds look-back windows and targets", {
  series <- as.numeric(1:6)
  out    <- .rwil_make_sequences(series, 3)

  expect_equal(
    out$X,
    matrix(c(1, 2, 3,
             2, 3, 4,
             3, 4, 5), nrow = 3, byrow = TRUE)
  )
  expect_equal(out$y, c(4, 5, 6))
})

test_that(".rwil_make_sequences needs a series longer than the window", {
  expect_error(.rwil_make_sequences(as.numeric(1:3), 3), "series length must be")
})

# ---- Time formatting -------------------------------------------------------

test_that(".fmt_time_rwil picks a sensible unit", {
  expect_equal(.fmt_time_rwil(30),   "30 seconds")
  expect_equal(.fmt_time_rwil(120),  "2 minutes")
  expect_equal(.fmt_time_rwil(7200), "2 hours")
})
