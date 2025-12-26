# src/evaluation/metrics.py

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def regression_metrics(y_true_cost: np.ndarray, y_pred_cost: np.ndarray) -> dict:
    mse = mean_squared_error(y_true_cost, y_pred_cost)
    mae = mean_absolute_error(y_true_cost, y_pred_cost)
    r2 = r2_score(y_true_cost, y_pred_cost)
    return {"mse": float(mse), "mae": float(mae), "r2": float(r2)}
