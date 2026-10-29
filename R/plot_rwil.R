# ============================================================
# RWiL -- plotRWiL + plotlambda_rwil
# ============================================================
#' @importFrom utils tail
utils::globalVariables(c(
  "time_idx", "value", "Series", "forecast",
  "Lambda", "RMSE", "Best"
))

# -- Shared theme --------------------------------------------
theme_rwil <- function(base_size = 12) {
  ggplot2::theme_minimal(base_size = base_size) +
    ggplot2::theme(
      plot.title       = ggplot2::element_text(
        face="bold", size=ggplot2::rel(1.15),
        margin=ggplot2::margin(b=6)),
      plot.subtitle    = ggplot2::element_text(
        colour="grey35", size=ggplot2::rel(0.95),
        margin=ggplot2::margin(b=10)),
      plot.caption     = ggplot2::element_text(
        colour="grey45", size=ggplot2::rel(0.75),
        hjust=0, margin=ggplot2::margin(t=10)),
      axis.title       = ggplot2::element_text(
        colour="grey25", size=ggplot2::rel(0.95)),
      axis.text        = ggplot2::element_text(colour="grey30"),
      panel.grid.major = ggplot2::element_line(
        colour="grey90", linewidth=0.4),
      panel.grid.minor = ggplot2::element_blank(),
      legend.position  = "bottom",
      legend.title     = ggplot2::element_text(
        face="bold", size=ggplot2::rel(0.9)),
      legend.text      = ggplot2::element_text(size=ggplot2::rel(0.85)),
      plot.background  = ggplot2::element_rect(fill="white", colour=NA),
      plot.margin      = ggplot2::margin(14, 18, 10, 14)
    )
}

#' Plot RWiL: Actual vs Predicted (in-sample) + Out-of-sample Forecast
#'
#' @param fit     RWiL object from RWiL().
#' @param h       Forecast horizon. Default 12.
#' @param lambda  Lambda to use. NULL = best.
#' @param actual  Optional numeric vector of actual future values.
#' @param newdata Optional new observations to update starting sequence.
#' @param n_hist  Number of historical points to show. NULL = all.
#'
#' @return ggplot2 object (invisibly).
#' @examples
#' # Runs only when the libtorch backend is installed (torch::install_torch())
#' if (torch::torch_is_installed()) {
#'   set.seed(1)
#'   y <- cumsum(rnorm(40, 0.5, 1))
#'   fit <- RWiL(y, mode = "manual", lag = 3, hidden_size = 4, lr = 0.01,
#'               epochs = 5, patience = 5, lambda_list = 0.5,
#'               verbose = FALSE, seed = 1)
#'   plotRWiL(fit, h = 6)
#' }
#' @export
plotRWiL <- function(fit, h = 12, lambda = NULL, actual = NULL,
                      newdata = NULL, n_hist = NULL) {

  nm <- if (is.null(lambda)) fit$meta$best_lam_nm else {
    if (lambda == 0) "standard" else paste0("lambda_", lambda)
  }
  if (!(nm %in% names(fit$results))) stop("lambda not found.")

  lam_label <- gsub("lambda_", "", nm)
  rd        <- fit$results[[nm]]

  tr_act  <- rd$tr_actuals
  tr_pred <- rd$tr_preds
  n_tr    <- length(tr_act)

  seq_ln    <- fit$data_info$lag
  start_idx <- seq_ln + 1L
  end_idx   <- start_idx + n_tr - 1L

  if (!is.null(n_hist) && n_hist < n_tr) {
    keep      <- (n_tr - n_hist + 1L):n_tr
    tr_act    <- tr_act[keep]
    tr_pred   <- tr_pred[keep]
    start_idx <- end_idx - n_hist + 1L
  }
  t_idx <- start_idx:end_idx

  df_actual <- data.frame(time_idx=t_idx, value=tr_act,
                           Series="Actual (Full Data)")
  df_pred   <- data.frame(time_idx=t_idx, value=tr_pred,
                           Series="Predicted (In-sample)")

  fc    <- predictRWiL(fit, h=h, lambda=lambda, newdata=newdata)
  fc_df <- data.frame(time_idx=fc$time_idx, forecast=fc$forecast)
  vline_x <- fit$data_info$N

  cols <- c(
    "Actual (Full Data)"    = "#1B2A38",
    "Predicted (In-sample)" = "#2E86DE",
    "Forecast"              = "#E63946",
    "Actual (Future)"       = "#1DB954"
  )

  p <- ggplot2::ggplot() +
    ggplot2::geom_line(
      data=df_actual,
      ggplot2::aes(x=time_idx, y=value, colour=Series),
      linewidth=0.9, lineend="round") +
    ggplot2::geom_line(
      data=df_pred,
      ggplot2::aes(x=time_idx, y=value, colour=Series),
      linewidth=0.9, lineend="round") +
    ggplot2::geom_line(
      data=fc_df,
      ggplot2::aes(x=time_idx, y=forecast, colour="Forecast"),
      linewidth=1.2, lineend="round") +
    ggplot2::geom_vline(
      xintercept=vline_x,
      linetype="dashed", colour="grey50", linewidth=0.6) +
    ggplot2::annotate("text",
      x=vline_x, y=Inf, label=" Forecast",
      hjust=0, vjust=1.5, size=3, colour="grey40") +
    ggplot2::scale_colour_manual(values=cols, name="Series") +
    ggplot2::labs(
      title    = paste0("RWiL -- Actual vs Predicted + ",
                        h, "-Step Forecast  [lambda=", lam_label,
                        "]  [RW=", fit$rw$rw_type, "]"),
      subtitle = paste0("Random Walk type: ", fit$rw$rw_type,
                        "  |  Drift = ", round(fit$rw$drift, 6)),
      x = "Time Index",
      y = "Value",
      caption = if (!is.null(actual))
        "Dark = Actual | Blue = Predicted | Red = Forecast | Green = Actual (future)"
      else
        "Dark = Actual (full data) | Blue = Predicted (in-sample) | Red = Forecast"
    ) +
    theme_rwil()

  p <- p + ggplot2::scale_x_continuous(
    expand = ggplot2::expansion(mult=c(0.02, 0.05)))

  if (!is.null(actual)) {
    act_df <- data.frame(
      time_idx = fit$data_info$N + seq_along(actual),
      value    = as.numeric(actual),
      Series   = "Actual (Future)"
    )
    p <- p + ggplot2::geom_line(
      data=act_df,
      ggplot2::aes(x=time_idx, y=value, colour=Series),
      linewidth=0.95, lineend="round")
  }

  print(p)
  invisible(p)
}

#' Plot Training RMSE Across All Lambda Values (RWiL)
#'
#' @param fit RWiL object.
#' @return ggplot2 object (invisibly).
#' @examples
#' # Runs only when the libtorch backend is installed (torch::install_torch())
#' if (torch::torch_is_installed()) {
#'   set.seed(1)
#'   y <- cumsum(rnorm(40, 0.5, 1))
#'   fit <- RWiL(y, mode = "manual", lag = 3, hidden_size = 4, lr = 0.01,
#'               epochs = 5, patience = 5, lambda_list = 0.5,
#'               verbose = FALSE, seed = 1)
#'   plotlambda_rwil(fit)
#' }
#' @export
plotlambda_rwil <- function(fit) {

  best_nm   <- fit$meta$best_lam_nm
  all_lams  <- fit$meta$all_lams
  lam_names <- names(fit$results)

  rmse_vals  <- sapply(lam_names, function(nm) fit$results[[nm]]$tr_rmse)
  lam_labels <- ifelse(all_lams == 0, "Std\n(0)", paste0(all_lams))

  df <- data.frame(
    Lambda = factor(lam_labels, levels=lam_labels),
    RMSE   = rmse_vals,
    Best   = lam_names == best_nm,
    stringsAsFactors = FALSE
  )

  best_rmse <- min(rmse_vals)
  best_lbl  <- gsub("lambda_", "", best_nm)

  p <- ggplot2::ggplot(df, ggplot2::aes(x=Lambda, y=RMSE, fill=Best)) +
    ggplot2::geom_bar(stat="identity", alpha=0.9, width=0.35,
                      colour="white", linewidth=0.3) +
    ggplot2::scale_fill_manual(
      values=c("FALSE"="#2E86DE", "TRUE"="#F4B400"), guide="none") +
    ggplot2::geom_hline(
      yintercept=best_rmse, linetype="dashed",
      colour="#E74C3C", linewidth=0.7) +
    ggplot2::geom_text(
      ggplot2::aes(label=format(round(RMSE, 4), scientific=FALSE)),
      vjust=-0.4, size=2.8, colour="grey30") +
    ggplot2::coord_cartesian(
      ylim=c(min(rmse_vals)*0.985, max(rmse_vals)*1.03)) +
    ggplot2::labs(
      title   = "RWiL - Training RMSE by Lambda",
      x       = "Lambda",
      y       = "Training RMSE",
      caption = "Red dashed = minimum RMSE  |  Gold bar = best lambda"
    ) +
    theme_rwil() +
    ggplot2::theme(panel.grid.major.x = ggplot2::element_blank())

  print(p)
  invisible(p)
}

`%||%` <- function(a, b) if (!is.null(a)) a else b
