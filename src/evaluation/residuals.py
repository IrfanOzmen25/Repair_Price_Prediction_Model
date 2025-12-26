# src/evaluation/residuals.py

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats


def compute_residuals_dollars(trues_log: np.ndarray, preds_log: np.ndarray):
    y_true = trues_log.reshape(-1)
    y_pred = preds_log.reshape(-1)
    y_true_cost = np.expm1(y_true)
    y_pred_cost = np.expm1(y_pred)
    residuals = y_true_cost - y_pred_cost
    return y_true_cost, y_pred_cost, residuals


def empirical_residual_interval(residuals: np.ndarray, lo_q: float, hi_q: float):
    res_lo = float(np.quantile(residuals, lo_q))
    res_hi = float(np.quantile(residuals, hi_q))
    return res_lo, res_hi


def qq_plot_log_residuals(trues_log: np.ndarray, preds_log: np.ndarray, title="Q–Q Plot of Log-Space Residuals"):
    log_residuals = (trues_log - preds_log).reshape(-1)
    log_residuals = log_residuals[np.isfinite(log_residuals)]

    plt.figure(figsize=(6, 6))
    stats.probplot(log_residuals, dist="norm", plot=plt)
    plt.title(title)
    plt.grid(True)
    plt.show()


def residual_plots(y_pred_cost: np.ndarray, residuals: np.ndarray):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred_cost, residuals, alpha=0.4)
    plt.axhline(0, linewidth=2)
    plt.xlabel("Predicted Cost ($)")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Residuals vs Predicted Cost")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=40, edgecolor="black", alpha=0.8)
    plt.axvline(0, linewidth=2)
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Count")
    plt.title("Residual Histogram")
    plt.grid(True)
    plt.show()


def attach_diagnostics(test_df: pd.DataFrame, y_pred_cost: np.ndarray, residuals: np.ndarray) -> pd.DataFrame:
    df = test_df.copy().reset_index(drop=True)
    df["y_pred_cost"] = y_pred_cost.reshape(-1)
    df["residual"] = residuals.reshape(-1)
    return df


def find_bad_points(df_diag: pd.DataFrame, pred_thresh=800.0, resid_thresh=-800.0) -> pd.DataFrame:
    bad = df_diag.loc[(df_diag["y_pred_cost"] > pred_thresh) & (df_diag["residual"] < resid_thresh)]
    return bad
