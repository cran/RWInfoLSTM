# ============================================================
# RWiL -- S3 Methods: print, summary, plot, predict
# ============================================================
utils::globalVariables(c("Epoch", "Loss", "Type", "Lambda", "self"))

#' @export
print.RWiL <- function(x, ...) {
  cat("\n========================================\n")
  cat("  RWiL -- Random Walk-Informed LSTM\n")
  cat("========================================\n")
  cat(sprintf("  Mode       : %s\n", toupper(x$meta$mode)))
  cat(sprintf("  N (full)   : %d\n", x$data_info$N))
  cat(sprintf("  Lag        : %d\n", x$meta$lag))
  cat(sprintf("  RW type    : %s  (drift=%.6f)\n",
              x$rw$rw_type, x$rw$drift))
  cat(sprintf("  Best HP    : hidden=%d  layers=%d  lr=%.4f  dropout=%.1f\n",
              x$best_hp$hidden, x$best_hp$layers,
              x$best_hp$lr, x$best_hp$dropout))
  cat(sprintf("  Best lambda: %s\n\n",
              gsub("lambda_", "", x$meta$best_lam_nm)))

  cat(sprintf("  %-10s | %12s | %10s | %10s | %12s\n",
              "Lambda", "RMSE", "MAE", "MAPE%", "SMAPE"))
  cat(paste(rep("-", 62), collapse = ""), "\n")

  for (nm in names(x$results)) {
    rd  <- x$results[[nm]]
    lam <- rd$lambda
    lbl <- if (lam == 0) "Standard  " else sprintf("%-10.1f", lam)
    cat(sprintf("  %s| %12.4f | %10.4f | %10.2f | %12.2f\n",
                lbl, rd$tr_rmse, rd$tr_mae, rd$tr_mape, rd$tr_smape))
  }
  cat("========================================\n")
  cat("  Use predictRWiL(fit, h=12) to forecast\n")
  cat("========================================\n\n")
  invisible(x)
}

#' @export
summary.RWiL <- function(object, ...) {
  cat("\n=== RWiL Summary (Full-Data Training) ===\n")
  rows <- lapply(names(object$results), function(nm) {
    rd <- object$results[[nm]]
    data.frame(
      Lambda = rd$lambda,
      RMSE   = rd$tr_rmse,
      MAE    = rd$tr_mae,
      MAPE   = round(rd$tr_mape,  2),
      SMAPE  = round(rd$tr_smape, 2),
      Time_s = round(rd$time, 2),
      stringsAsFactors = FALSE
    )
  })
  df <- do.call(rbind, rows)
  print(df, row.names = FALSE)
  cat(sprintf("\nBest lambda: %s  (Train RMSE = %.4f)\n",
              gsub("lambda_", "", object$meta$best_lam_nm),
              min(sapply(object$results, function(r) r$tr_rmse))))
  cat("Use predictRWiL(fit, h=12) for out-of-sample forecast\n\n")
  invisible(df)
}

#' @export
plot.RWiL <- function(x, which = c("loss", "lambda"), ...) {
  which <- match.arg(which)
  if (which == "lambda") return(plotlambda_rwil(x))

  df_loss <- do.call(rbind, lapply(names(x$results), function(nm) {
    rd   <- x$results[[nm]]
    hist <- rd$loss_hist
    lbl  <- if (rd$lambda == 0) "Standard" else
      paste0("lambda=", rd$lambda)
    ep   <- seq_along(hist$total)
    rbind(
      data.frame(Epoch=ep, Loss=hist$total,  Type="Total",        Lambda=lbl),
      data.frame(Epoch=ep, Loss=hist$actual, Type="LSTM Loss",    Lambda=lbl),
      data.frame(Epoch=ep, Loss=hist$rw,     Type="RW Guided",   Lambda=lbl)
    )
  }))

  p <- ggplot2::ggplot(df_loss,
                        ggplot2::aes(x=Epoch, y=Loss, colour=Type)) +
    ggplot2::geom_line(linewidth=0.5, alpha=0.8) +
    ggplot2::facet_wrap(ggplot2::vars(Lambda), scales="free_y") +
    ggplot2::labs(title="RWiL - Training Loss History",
                  x="Epoch", y="Loss") +
    ggplot2::theme_minimal(base_size=9)
  print(p)
  invisible(p)
}

#' @export
predict.RWiL <- function(object, h = 12, lambda = NULL, ...) {
  predictRWiL(object, h = h, lambda = lambda)
}

# Internal helper (not exported): per-lambda training accuracy table
accuracy.RWiL <- function(x) {
  rows <- lapply(names(x$results), function(nm) {
    rd <- x$results[[nm]]
    data.frame(Model="RWiL_LSTM", Lambda=rd$lambda,
               RMSE=rd$tr_rmse, MAE=rd$tr_mae,
               MAPE=rd$tr_mape, SMAPE=rd$tr_smape)
  })
  do.call(rbind, rows)
}
