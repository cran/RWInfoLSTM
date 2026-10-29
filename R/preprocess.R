# ============================================================
# RWiL -- Preprocessing: scaling, sequences, tensors
# ============================================================
#' @importFrom utils tail head
NULL

# -- MinMax scaler -------------------------------------------
.rwil_make_scaler <- function(train_vec) {
  mn  <- min(train_vec, na.rm = TRUE)
  mx  <- max(train_vec, na.rm = TRUE)
  rng <- mx - mn
  if (rng == 0) rng <- 1
  list(
    scale   = function(x) (x - mn) / rng,
    unscale = function(x) x * rng + mn,
    min = mn, max = mx, range = rng
  )
}

# -- Supervised sequences ------------------------------------
.rwil_make_sequences <- function(series, seq_len) {
  n <- length(series)
  if (n <= seq_len) stop("series length must be > seq_len")
  X <- matrix(NA_real_, nrow = n - seq_len, ncol = seq_len)
  y <- numeric(n - seq_len)
  for (i in seq_len(n - seq_len)) {
    X[i, ] <- series[i:(i + seq_len - 1L)]
    y[i]   <- series[i + seq_len]
  }
  list(X = X, y = y)
}

# -- 3-D torch tensor (batch, seq, 1) ------------------------
.rwil_to_seq_tensor <- function(mat) {
  arr <- array(mat, dim = c(nrow(mat), ncol(mat), 1L))
  torch::torch_tensor(arr, dtype = torch::torch_float())
}

# -- Column tensor (N, 1) ------------------------------------
.rwil_to_col_tensor <- function(vec) {
  torch::torch_tensor(
    matrix(vec, ncol = 1L),
    dtype = torch::torch_float()
  )
}

# -- Full-data preprocessing (no test split) -----------------
.rwil_preprocess_full <- function(data, rw_res, seq_len, batch_size) {
  N <- length(data)

  # Scale on full data
  scaler      <- .rwil_make_scaler(data)
  data_scaled <- scaler$scale(data)

  # All sequences
  seqs   <- .rwil_make_sequences(data_scaled, seq_len)
  X_all  <- seqs$X
  y_all  <- seqs$y
  n_seqs <- nrow(X_all)

  # RW guidance aligned to sequences
  # rw_res$fitted has length N; sequences start at index seq_len+1
  rw_fitted        <- rw_res$fitted
  req              <- seq_len + n_seqs
  if (length(rw_fitted) < req)
    rw_fitted <- c(rw_fitted,
                   rep(tail(rw_fitted, 1), req - length(rw_fitted)))

  rw_guide_raw    <- rw_fitted[(seq_len + 1L):(seq_len + n_seqs)]
  rw_guide_scaled <- scaler$scale(rw_guide_raw)

  # Internal val split: last 20% for early stopping
  val_start <- floor(n_seqs * 0.8) + 1L
  val_split <- val_start - 1L

  X_tr  <- X_all[seq_len(val_split), , drop = FALSE]
  y_tr  <- y_all[seq_len(val_split)]
  X_val <- X_all[val_start:n_seqs, , drop = FALSE]
  y_val <- y_all[val_start:n_seqs]
  guide_tr <- rw_guide_scaled[seq_len(val_split)]

  # Torch tensors
  Xtr_t   <- .rwil_to_seq_tensor(X_tr)
  ytr_t   <- .rwil_to_col_tensor(y_tr)
  Xval_t  <- .rwil_to_seq_tensor(X_val)
  yval_t  <- .rwil_to_col_tensor(y_val)
  Xall_t  <- .rwil_to_seq_tensor(X_all)
  yall_t  <- .rwil_to_col_tensor(y_all)
  guide_t <- .rwil_to_col_tensor(as.numeric(guide_tr))

  BATCH <- min(batch_size, nrow(X_tr))

  # Last seq_len scaled values -- seed for forecasting
  last_seq_scaled <- data_scaled[(N - seq_len + 1L):N]

  list(
    scaler          = scaler,
    BATCH           = BATCH,
    X_tr            = X_tr,
    y_tr            = y_tr,
    X_val           = X_val,
    y_val           = y_val,
    X_all           = X_all,
    y_all           = y_all,
    val_split       = val_split,
    n_seqs          = n_seqs,
    Xtr_t           = Xtr_t,
    ytr_t           = ytr_t,
    Xval_t          = Xval_t,
    yval_t          = yval_t,
    Xall_t          = Xall_t,
    yall_t          = yall_t,
    guide_t         = guide_t,
    last_seq_scaled = last_seq_scaled,
    seq_len         = seq_len
  )
}
