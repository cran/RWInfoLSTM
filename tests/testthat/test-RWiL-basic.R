# Package-level checks: exports, S3 registration, input validation and the
# printing behaviour. None of these train a network, so they need no torch.

test_that("the package loads and exports the public API", {
  expect_true("RWInfoLSTM" %in% loadedNamespaces())

  exports <- getNamespaceExports("RWInfoLSTM")
  expect_true(all(
    c("RWiL", "predictRWiL", "AccuracyRWiL", "plotRWiL", "plotlambda_rwil")
    %in% exports
  ))
  expect_false(".fmt_time_rwil" %in% exports)    # internal helper stays private
})

test_that("S3 methods are registered", {
  methods <- list(
    c("print",   "RWiL"),
    c("summary", "RWiL"),
    c("plot",    "RWiL"),
    c("predict", "RWiL"),
    c("print",   "RWiL_forecast")
  )
  for (m in methods) {
    expect_false(
      is.null(getS3method(m[1], m[2], optional = TRUE)),
      info = paste(m, collapse = ".")
    )
  }
})

# ---- Input validation (all of this fails before any torch code runs) -------

test_that("RWiL() requires at least 20 observations", {
  expect_error(
    RWiL(rnorm(10), lag = 3, verbose = FALSE),
    "at least 20 observations"
  )
})

test_that("RWiL() requires a positive lag", {
  expect_error(RWiL(rnorm(30), lag = 0,  verbose = FALSE), "lag must be a positive integer")
  expect_error(RWiL(rnorm(30), lag = -2, verbose = FALSE), "lag must be a positive integer")
})

test_that("RWiL() insists that lag is supplied", {
  expect_error(RWiL(rnorm(30), verbose = FALSE), "lag")
})

test_that("RWiL() rejects an unknown mode", {
  expect_error(
    RWiL(rnorm(30), mode = "bogus", lag = 3, verbose = FALSE),
    "should be one of"
  )
})

# ---- Forecast helpers on a stub fit ---------------------------------------

test_that("predictRWiL() and friends reject an unknown lambda", {
  fit <- make_stub_fit()

  expect_error(predictRWiL(fit, h = 3, lambda = 0.77), "lambda not found. Available: 0, 0.5")
  expect_error(AccuracyRWiL(fit, actual = c(1, 2, 3), lambda = 0.77), "lambda not found")
  expect_error(plotRWiL(fit, lambda = 0.77), "lambda not found")
})

test_that("predict() dispatches to predictRWiL() for RWiL objects", {
  fit <- make_stub_fit()
  expect_error(predict(fit, h = 2, lambda = 0.77), "lambda not found")
})

# ---- Output control --------------------------------------------------------

test_that("AccuracyRWiL() has a verbose switch that defaults to TRUE", {
  expect_true("verbose" %in% names(formals(AccuracyRWiL)))
  expect_true(formals(AccuracyRWiL)$verbose)
})

test_that("print() for a forecast shows the header and returns the object", {
  fc <- make_forecast()

  expect_output(res <- print(fc), "RWiL Forecast")
  expect_output(print(fc), "lambda=0.5")
  expect_output(print(fc), "h=3 steps")
  expect_identical(res, fc)
})
