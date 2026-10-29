# ============================================================
# RWiL -- Random Walk-Informed LSTM  |  Main Entry Point
# Trains on FULL data. Use predictRWiL() for out-of-sample forecast.
# ============================================================

#' @importFrom stats approx
#' @importFrom utils head tail
#' @importFrom torch nn_module nn_lstm nn_linear nn_mse_loss torch_tensor
#' @importFrom torch torch_float with_no_grad optim_adam dataloader
#' @importFrom torch tensor_dataset torch_manual_seed nn_utils_clip_grad_norm_
#' @importFrom ggplot2 ggplot aes geom_line geom_bar geom_vline geom_text
#' @importFrom ggplot2 annotate labs theme_minimal theme element_text
#' @importFrom ggplot2 element_line element_blank element_rect facet_wrap vars
#' @importFrom ggplot2 scale_fill_manual scale_colour_manual
#' @importFrom ggplot2 scale_x_continuous coord_cartesian expansion margin rel
#' @importFrom cli cli_h1 cli_h2 cli_alert_info cli_alert_success
#' @importFrom cli cli_alert_warning
NULL

# Set the R and torch seeds only when the user supplies one.
.rwil_set_seed <- function(seed) {
  if (!is.null(seed)) {
    set.seed(seed)
    torch::torch_manual_seed(seed)
  }
  invisible(NULL)
}

.fmt_time_rwil <- function(t) {
  if      (t < 60)   paste0(round(t,     2), " seconds")
  else if (t < 3600) paste0(round(t/60,  2), " minutes")
  else               paste0(round(t/3600, 2), " hours")
}

#' @export
RWiL <- function(
  data,
  mode         = c("auto", "manual"),
  lag,                      # compulsory -- user must specify
  hidden_size  = 32,
  num_layers   = 1,
  lr           = 0.001,
  dropout      = 0.0,
  epochs       = 500,
  patience     = 30,
  tune_epochs  = 50,
  batch_size   = 16,
  lambda_list  = seq(0.1, 0.9, 0.1),
  hidden_grid  = c(8, 16, 24, 32, 40),
  layers_grid  = c(1L, 2L),
  lr_grid      = c(0.001, 0.003, 0.005),
  dropout_grid = c(0.0, 0.1, 0.2),
  verbose      = TRUE,
  seed         = NULL
) {
  mode <- match.arg(mode)
  data <- as.numeric(data)
  if (any(is.na(data))) {
    if (verbose) cli::cli_alert_info("Missing values -- interpolating.")
    data <- .rwil_interp(data)
  }
  N <- length(data)
  if (N < 20) stop("RWiL requires at least 20 observations.")
  seq_len <- as.integer(lag)
  if (seq_len < 1L) stop("lag must be a positive integer (e.g. lag = 3).")
  .rwil_set_seed(seed)

  if (verbose) {
    cli::cli_h1("RWiL -- Random Walk-Informed LSTM")
    cli::cli_alert_info("Mode      : {toupper(mode)}")
    cli::cli_alert_info("N         : {N}  |  lag={lag}  |  Training on FULL data")
    cli::cli_alert_info("Lambda    : [{paste(lambda_list, collapse=', ')}]")
    cli::cli_alert_info("Best lambda: selected by Training RMSE")
  }

  # -- STEP 1: Fit Random Walk (auto-select simple vs drift) --
  rw_res <- .fit_rw_full(data = data, verbose = verbose)

  # -- STEP 2: Preprocess full data ---------------------------
  pp <- .rwil_preprocess_full(data, rw_res, seq_len, batch_size)

  # -- STEP 3: Hyperparameter selection ----------------------
  if (mode == "auto") {
    if (verbose) cli::cli_h2("Grid Search -- LSTM Hyperparameters")
    tune_out <- .rwil_tune_lstm(
      pp           = pp,
      hidden_grid  = hidden_grid,
      layers_grid  = layers_grid,
      lr_grid      = lr_grid,
      dropout_grid = dropout_grid,
      tune_epochs  = tune_epochs,
      batch_size   = batch_size,
      seed         = seed,
      verbose      = verbose
    )
    best_hp    <- tune_out$best
    tuning_log <- tune_out$log
  } else {
    best_hp    <- list(hidden  = hidden_size, layers = num_layers,
                       lr      = lr,          dropout = dropout)
    tuning_log <- NULL
    if (verbose) {
      cli::cli_h2("Manual Hyperparameters")
      cli::cli_alert_info(
        "hidden={best_hp$hidden}  layers={best_hp$layers}  lr={best_hp$lr}  dropout={best_hp$dropout}")
    }
  }

  # -- STEP 4: Training ---------------------------------------
  if (verbose)
    cli::cli_h2(
      "Training on Full Data  (Standard + {length(lambda_list)} lambda values)")

  all_lams <- c(0.0, lambda_list)
  results  <- vector("list", length(all_lams))
  names(results) <- ifelse(all_lams == 0, "standard",
                           paste0("lambda_", all_lams))

  for (i in seq_along(all_lams)) {
    lam   <- all_lams[i]
    label <- if (lam == 0) "Standard (lambda=0)" else
      paste0("RW-Informed (lambda=", lam, ")")
    if (verbose) cli::cli_alert_info("[LSTM] {label}")
    .rwil_set_seed(seed)

    res <- .rwil_train_lstm_full(
      pp         = pp,
      best_hp    = best_hp,
      lam        = lam,
      epochs     = epochs,
      patience   = patience,
      batch_size = batch_size,
      verbose    = verbose
    )
    results[[i]] <- res

    if (verbose)
      cli::cli_alert_success(
        "  Train RMSE={round(res$tr_rmse,4)}  Val RMSE={round(res$val_rmse,4)}  SMAPE={round(res$tr_smape,2)}%")
  }

  # -- Best lambda by Train RMSE ------------------------------
  tr_rmses    <- sapply(results, function(r) r$tr_rmse)
  best_lam_nm <- names(which.min(tr_rmses))

  # -- Build best_model summary ------------------------------
  bm <- results[[best_lam_nm]]
  best_model <- list(
    lambda      = bm$lambda,
    lambda_name = best_lam_nm,
    hyperparams = list(
      hidden_size = best_hp$hidden,
      num_layers  = best_hp$layers,
      lr          = best_hp$lr,
      dropout     = best_hp$dropout,
      lag         = lag,
      epochs_run  = length(bm$loss_hist$total)
    ),
    rw = list(
      rw_type     = rw_res$rw_type,
      drift       = rw_res$drift,
      rmse_simple = rw_res$rmse_simple,
      rmse_drift  = rw_res$rmse_drift
    ),
    metrics = data.frame(
      Metric = c("RMSE", "MAE", "MAPE%", "SMAPE%"),
      Value  = c(round(bm$tr_rmse,  4), round(bm$tr_mae,   4),
                 round(bm$tr_mape,  2), round(bm$tr_smape, 2)),
      stringsAsFactors = FALSE
    ),
    predictions = data.frame(
      Actual    = round(bm$tr_actuals, 4),
      Predicted = round(bm$tr_preds,   4)
    ),
    loss_history = data.frame(
      Total_Loss    = round(bm$loss_hist$total,  8),
      LSTM_Loss     = round(bm$loss_hist$actual, 8),
      RW_Guided_Loss = round(bm$loss_hist$rw,    8)
    ),
    model      = bm$model,
    train_time = .fmt_time_rwil(bm$time)
  )

  out <- structure(
    list(
      results    = results,
      best_model = best_model,
      rw         = rw_res,
      best_hp    = best_hp,
      tuning_log = tuning_log,
      meta = list(
        mode        = mode,
        lag         = lag,
        lambda_list = lambda_list,
        all_lams    = all_lams,
        epochs      = epochs,
        patience    = patience,
        tune_epochs = tune_epochs,
        batch_size  = batch_size,
        seed        = seed,
        best_lam_nm = best_lam_nm,
        selection   = "Train RMSE"
      ),
      data_info = list(
        data            = data,
        N               = N,
        scaler          = pp$scaler,
        last_seq_scaled = pp$last_seq_scaled,
        lag             = lag
      )
    ),
    class = "RWiL"
  )

  if (verbose)
    cli::cli_alert_success(
      "RWiL complete. Best lambda: {gsub('lambda_','',best_lam_nm)}")
  out
}
