# ============================================================
# RWiL -- LSTM Model & Random Walk-Informed Loss
# ============================================================

# -- LSTM model (defined inside function to satisfy torch) ---
.rwil_lstm_model <- function(hidden_size = 32L, num_layers = 1L,
                              dropout = 0.0) {
  hs   <- as.integer(hidden_size)
  nl   <- as.integer(num_layers)
  do_r <- if (nl > 1L) as.double(dropout) else 0.0

  RWiLNet <- torch::nn_module(
    classname = "RWiLNet",
    initialize = function(hs, nl, do_r) {
      self$lstm <- torch::nn_lstm(
        input_size  = 1L,
        hidden_size = hs,
        num_layers  = nl,
        batch_first = TRUE,
        dropout     = do_r
      )
      self$fc <- torch::nn_linear(hs, 1L)
    },
    forward = function(x) {
      out  <- self$lstm(x)[[1]]
      last <- out$select(2L, out$shape[2] - 1L)
      self$fc(last)
    }
  )
  RWiLNet(hs = hs, nl = nl, do_r = do_r)
}

# -- RW-Informed Loss ----------------------------------------
# Loss = (1-lambda) x MSE(y_hat, y) + lambda x MSE(y_hat, y_RW)
# lambda = 0.0 -> Standard LSTM (no RW guidance)
# lambda = 1.0 -> Fully RW-guided
.rwil_loss <- function(y_pred, y_actual, y_rw, lambda) {
  mse_fn      <- torch::nn_mse_loss()
  loss_actual <- mse_fn(y_pred, y_actual)
  loss_rw     <- mse_fn(y_pred, y_rw)
  total       <- (1 - lambda) * loss_actual + lambda * loss_rw
  list(total = total, actual = loss_actual, rw = loss_rw)
}
