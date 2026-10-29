# ============================================================
# RWiL -- Grid-Search Tuning + Full Training
# ============================================================

# -- Evaluate model ------------------------------------------
.rwil_evaluate_model <- function(model, X_t, y_t, scaler) {
  model$eval()
  torch::with_no_grad({
    ps <- as.numeric(model(X_t)$detach()$reshape(list(-1L)))
  })
  preds   <- scaler$unscale(ps)
  actuals <- scaler$unscale(as.numeric(y_t$squeeze()))
  rmse    <- sqrt(mean((actuals - preds)^2))
  mae     <- mean(abs(actuals - preds))
  denom   <- abs(actuals) + abs(preds); mask <- denom != 0
  smape   <- if (any(mask))
    mean(2 * abs(actuals[mask] - preds[mask]) / denom[mask]) * 100 else 0
  mape    <- if (any(actuals != 0))
    mean(abs((actuals[actuals != 0] - preds[actuals != 0]) /
               actuals[actuals != 0])) * 100 else NA
  list(rmse = rmse, mae = mae, smape = smape, mape = mape,
       preds = preds, actuals = actuals)
}

# -- DataLoader helpers --------------------------------------
.rwil_make_dl <- function(X_t, y_t, batch_size, shuffle = TRUE) {
  ds <- torch::tensor_dataset(X_t, y_t)
  torch::dataloader(ds, batch_size = batch_size,
                    shuffle = shuffle, drop_last = TRUE)
}

.rwil_make_dl_guided <- function(X_t, y_t, guide_t, batch_size) {
  bs <- min(batch_size, dim(X_t)[1])
  ds <- torch::tensor_dataset(X_t, y_t, guide_t)
  torch::dataloader(ds, batch_size = bs,
                    shuffle = FALSE, drop_last = TRUE)
}

# -- Grid Search ---------------------------------------------
.rwil_tune_lstm <- function(pp, hidden_grid, layers_grid, lr_grid,
                             dropout_grid, tune_epochs, batch_size,
                             seed, verbose) {
  grid   <- expand.grid(hidden  = hidden_grid,
                        layers  = layers_grid,
                        lr      = lr_grid,
                        dropout = dropout_grid,
                        stringsAsFactors = FALSE)
  n_comb <- nrow(grid)
  if (verbose)
    cli::cli_alert_info("{n_comb} hyperparameter combinations to evaluate")

  BATCH <- min(batch_size, nrow(pp$X_tr))
  dl    <- .rwil_make_dl(pp$Xtr_t, pp$ytr_t, BATCH, shuffle = TRUE)
  log   <- vector("list", n_comb)

  for (i in seq_len(n_comb)) {
    h <- grid$hidden[i]; l <- grid$layers[i]
    r <- grid$lr[i];     d <- grid$dropout[i]

    .rwil_set_seed(seed)
    m   <- .rwil_lstm_model(hidden_size = h, num_layers = l, dropout = d)
    opt <- torch::optim_adam(m$parameters, lr = r)
    m$train()

    for (ep in seq_len(tune_epochs)) {
      coro::loop(for (batch in dl) {
        bX <- batch[[1]]; by <- batch[[2]]
        opt$zero_grad()
        out <- .rwil_loss(m(bX), by, by, lambda = 0.0)
        out$total$backward()
        opt$step()
      })
    }

    ev <- .rwil_evaluate_model(m, pp$Xval_t, pp$yval_t, pp$scaler)
    log[[i]] <- list(hidden=h, layers=l, lr=r, dropout=d, val_rmse=ev$rmse)
    if (verbose)
      cli::cli_alert_info(
        "  [{i}/{n_comb}] h={h} l={l} lr={r} d={d}  Val RMSE={round(ev$rmse,4)}")
  }

  log_df <- do.call(rbind, lapply(log, as.data.frame))
  best_i <- which.min(log_df$val_rmse)
  best   <- as.list(log_df[best_i, ])

  if (verbose)
    cli::cli_alert_success(
      "Best HP: h={best$hidden}  l={best$layers}  lr={best$lr}  d={best$dropout}  Val RMSE={round(best$val_rmse,4)}")

  list(best = best, log = log_df)
}

# -- Train on FULL data (stores model for forecasting) -------
.rwil_train_lstm_full <- function(pp, best_hp, lam, epochs, patience,
                                   batch_size, verbose) {
  BATCH <- min(batch_size, nrow(pp$X_tr))
  model <- .rwil_lstm_model(
    hidden_size = best_hp$hidden,
    num_layers  = best_hp$layers,
    dropout     = best_hp$dropout
  )
  opt    <- torch::optim_adam(model$parameters, lr = best_hp$lr)
  mse_fn <- torch::nn_mse_loss()
  best_v <- Inf; counter <- 0L
  best_params <- lapply(model$parameters, function(p) p$clone()$detach())
  hist <- list(total = numeric(), actual = numeric(), rw = numeric())
  t0   <- proc.time()["elapsed"]

  if (lam == 0.0) {
    # -- Standard LSTM (no RW guidance) ---------------------
    dl <- .rwil_make_dl(pp$Xtr_t, pp$ytr_t, BATCH, shuffle = TRUE)
    for (epoch in seq_len(epochs)) {
      model$train(); ep <- 0.0
      coro::loop(for (batch in dl) {
        bX <- batch[[1]]; by <- batch[[2]]
        opt$zero_grad()
        out <- .rwil_loss(model(bX), by, by, lambda = 0.0)
        out$total$backward()
        torch::nn_utils_clip_grad_norm_(model$parameters, 1.0)
        opt$step(); ep <- ep + out$total$item()
      })
      ep <- ep / max(length(dl), 1L)
      hist$total  <- c(hist$total,  ep)
      hist$actual <- c(hist$actual, ep)
      hist$rw     <- c(hist$rw,     0.0)
      model$eval()
      vl <- torch::with_no_grad({
        mse_fn(model(pp$Xval_t), pp$yval_t)$item()
      })
      if (vl < best_v) {
        best_v <- vl; counter <- 0L
        best_params <- lapply(model$parameters,
                              function(p) p$clone()$detach())
      } else counter <- counter + 1L
      if (counter >= patience) {
        if (verbose)
          cli::cli_alert_info("  Early stopping @ epoch {epoch}")
        break
      }
    }

  } else {
    # -- RW-Informed LSTM -----------------------------------
    dl <- .rwil_make_dl_guided(pp$Xtr_t, pp$ytr_t, pp$guide_t, BATCH)
    for (epoch in seq_len(epochs)) {
      model$train(); et <- ea <- er <- 0.0
      coro::loop(for (batch in dl) {
        bX <- batch[[1]]; by <- batch[[2]]; bg <- batch[[3]]
        opt$zero_grad()
        out <- .rwil_loss(model(bX), by, bg, lambda = lam)
        out$total$backward()
        torch::nn_utils_clip_grad_norm_(model$parameters, 1.0)
        opt$step()
        et <- et + out$total$item()
        ea <- ea + out$actual$item()
        er <- er + out$rw$item()
      })
      n <- max(length(dl), 1L)
      hist$total  <- c(hist$total,  et / n)
      hist$actual <- c(hist$actual, ea / n)
      hist$rw     <- c(hist$rw,     er / n)
      model$eval()
      vl <- torch::with_no_grad({
        mse_fn(model(pp$Xval_t), pp$yval_t)$item()
      })
      if (vl < best_v) {
        best_v <- vl; counter <- 0L
        best_params <- lapply(model$parameters,
                              function(p) p$clone()$detach())
      } else counter <- counter + 1L
      if (counter >= patience) {
        if (verbose)
          cli::cli_alert_info("  Early stopping @ epoch {epoch}")
        break
      }
    }
  }

  # Restore best weights
  torch::with_no_grad({
    pnames <- names(model$parameters)
    for (i in seq_along(pnames))
      model$parameters[[pnames[i]]]$copy_(best_params[[i]])
  })

  elapsed <- proc.time()["elapsed"] - t0
  ev      <- .rwil_evaluate_model(model, pp$Xall_t, pp$yall_t, pp$scaler)
  val_ev  <- .rwil_evaluate_model(model, pp$Xval_t, pp$yval_t, pp$scaler)

  list(
    tr_rmse    = ev$rmse,
    tr_mae     = ev$mae,
    tr_smape   = ev$smape,
    tr_mape    = ev$mape,
    tr_preds   = ev$preds,
    tr_actuals = ev$actuals,
    val_rmse   = val_ev$rmse,
    loss_hist  = hist,
    time       = elapsed,
    lambda     = lam,
    model      = model
  )
}
