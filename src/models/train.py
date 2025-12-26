# src/models/train.py

from __future__ import annotations
import os
import json
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.config import TrainConfig
from src.data.preprocessing import load_data, infer_feature_types, basic_cleaning, fill_missing, filter_target
from src.data.feature_engineering import (
    split_train_test,
    build_num_cols,
    train_only_clip_target,
    scale_numeric,
    build_categoricals,
)
from src.models.model import RepairDataset, RepairEmbNet
from src.evaluation.metrics import regression_metrics
from src.evaluation.residuals import compute_residuals_dollars, empirical_residual_interval


def main():
    # --- CONFIG (edit these or replace with argparse later) ---
    cfg = TrainConfig()
    file_path = os.environ.get("REPAIR_DATA_PATH", "data/MTRX1000000.csv")
    artifacts_dir = os.environ.get("ARTIFACTS_DIR", "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    # --- LOAD + CLEAN ---
    df = load_data(file_path)
    df = basic_cleaning(df)

    numeric_cols, categorical_cols = infer_feature_types(df)
    df = fill_missing(df, numeric_cols, categorical_cols)
    df = filter_target(df, cfg.target)

    # --- FEATURES ---
    num_cols = build_num_cols(df, cfg.target, cfg.id_like)

    train_df, test_df = split_train_test(df, test_size=cfg.test_size, random_state=cfg.random_state)

    # train-only clipping (winsor-style)
    train_df, test_df, low_q, high_q = train_only_clip_target(
        train_df, test_df, cfg.target, cfg.clip_lo, cfg.clip_hi
    )

    scaler, X_num_train, X_num_test = scale_numeric(train_df, test_df, num_cols)

    # log-transform target
    y_train = np.log1p(train_df[cfg.target].astype(np.float32).values).reshape(-1, 1)
    y_test = np.log1p(test_df[cfg.target].astype(np.float32).values).reshape(-1, 1)

    vocabs, X_cat_train, X_cat_test, cat_cardinalities, emb_dims = build_categoricals(
        train_df, test_df, cfg.cat_cols, min_freq=2
    )

    # --- DATALOADERS ---
    train_ds = RepairDataset(X_cat_train, X_num_train, y_train)
    test_ds = RepairDataset(X_cat_test, X_num_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_train, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_test, shuffle=False)

    # --- MODEL ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RepairEmbNet(cat_cardinalities, emb_dims, n_num=len(num_cols), hidden=cfg.hidden, p=cfg.dropout).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=cfg.huber_beta)

    train_loss_hist, test_loss_hist = [], []

    # --- TRAIN LOOP ---
    for epoch in range(cfg.epochs):
        model.train()
        batch_losses = []

        for xc, xn, yb in train_loader:
            xc, xn, yb = xc.to(device), xn.to(device), yb.to(device)
            pred = model(xc, xn)
            loss = loss_fn(pred, yb)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            batch_losses.append(loss.item())

        train_epoch_loss = float(np.mean(batch_losses))
        train_loss_hist.append(train_epoch_loss)

        # eval
        model.eval()
        eval_losses = []
        preds, trues = [], []
        with torch.no_grad():
            for xc, xn, yb in test_loader:
                xc, xn, yb = xc.to(device), xn.to(device), yb.to(device)
                pred = model(xc, xn)
                loss = loss_fn(pred, yb)
                eval_losses.append(loss.item())
                preds.append(pred.cpu().numpy())
                trues.append(yb.cpu().numpy())

        test_epoch_loss = float(np.mean(eval_losses))
        test_loss_hist.append(test_epoch_loss)

        preds = np.vstack(preds)
        trues = np.vstack(trues)

        # metrics in dollars
        y_true_cost = np.expm1(trues)
        y_pred_cost = np.expm1(preds)
        m = regression_metrics(y_true_cost, y_pred_cost)

        print(
            f"Epoch {epoch+1:02d} | "
            f"train huber {train_epoch_loss:.4f} | "
            f"test huber {test_epoch_loss:.4f} | "
            f"MAE ${m['mae']:.2f} | R2 {m['r2']:.3f}"
        )

    # --- FINAL residual interval (empirical) ---
    _, y_pred_cost, residuals = compute_residuals_dollars(trues, preds)
    res_lo, res_hi = empirical_residual_interval(residuals, cfg.interval_lo, cfg.interval_hi)

    # --- SAVE ARTIFACTS ---
    torch.save(model.state_dict(), os.path.join(artifacts_dir, "model.pt"))
    joblib.dump(scaler, os.path.join(artifacts_dir, "scaler.joblib"))

    with open(os.path.join(artifacts_dir, "metadata.json"), "w") as f:
        json.dump(
            {
                "target": cfg.target,
                "cat_cols": cfg.cat_cols,
                "num_cols": num_cols,
                "id_like": cfg.id_like,
                "cat_cardinalities": cat_cardinalities,
                "emb_dims": emb_dims,
                "clip_low_q": low_q,
                "clip_high_q": high_q,
                "interval_lo_q": cfg.interval_lo,
                "interval_hi_q": cfg.interval_hi,
                "residual_lo": res_lo,
                "residual_hi": res_hi,
                "huber_beta": cfg.huber_beta,
            },
            f,
            indent=2,
        )

    with open(os.path.join(artifacts_dir, "vocabs.json"), "w") as f:
        json.dump(vocabs, f)

    print(f"\nSaved artifacts to: {artifacts_dir}")
    print(f"Empirical residual interval ({int((cfg.interval_hi-cfg.interval_lo)*100)}%): "
          f"[{res_lo:.2f}, {res_hi:.2f}] dollars")


if __name__ == "__main__":
    main()
