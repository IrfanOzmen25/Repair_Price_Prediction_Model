# src/models/predict.py

from __future__ import annotations
import json
import joblib
import numpy as np
import pandas as pd
import torch

from src.data.feature_engineering import encode_cats
from src.models.model import RepairEmbNet


def main():
    artifacts_dir = "artifacts"

    # --- load metadata ---
    with open(f"{artifacts_dir}/metadata.json", "r") as f:
        meta = json.load(f)

    with open(f"{artifacts_dir}/vocabs.json", "r") as f:
        vocabs = json.load(f)

    scaler = joblib.load(f"{artifacts_dir}/scaler.joblib")

    cat_cols = meta["cat_cols"]
    num_cols = meta["num_cols"]
    cat_cardinalities = {k: int(v) for k, v in meta["cat_cardinalities"].items()}
    emb_dims = {k: int(v) for k, v in meta["emb_dims"].items()}

    res_lo = float(meta["residual_lo"])
    res_hi = float(meta["residual_hi"])
    lo_q = float(meta["interval_lo_q"])
    hi_q = float(meta["interval_hi_q"])

    # --- model ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RepairEmbNet(cat_cardinalities, emb_dims, n_num=len(num_cols)).to(device)
    model.load_state_dict(torch.load(f"{artifacts_dir}/model.pt", map_location=device))
    model.eval()

    # --- example input (replace with your own) ---
    example = pd.DataFrame([{
        "VEHICLE_YEAR": 2017,
        "ITEM_NAME": "FRONT",
        "VEHICLE_MODEL": "F150",
        "REPAIRER_STATE": "California",
        "SERVICE": "PDR",
    }])

    # numeric: align columns (missing -> 0)
    example_num = example.reindex(columns=num_cols, fill_value=0.0).astype(float)
    ex_num = scaler.transform(example_num)

    # categorical
    ex_cat = encode_cats(example, vocabs, cat_cols)

    xc = torch.tensor(ex_cat, dtype=torch.long).to(device)
    xn = torch.tensor(ex_num, dtype=torch.float32).to(device)

    with torch.no_grad():
        pred_log = model(xc, xn).item()

    pred_cost = float(np.expm1(pred_log))

    pred_low = max(0.0, pred_cost + res_lo)
    pred_high = max(0.0, pred_cost + res_hi)

    print(f"Predicted repair cost (point): ${pred_cost:.2f}")
    print(f"Likely range ({int((hi_q-lo_q)*100)}%): ${pred_low:.2f} – ${pred_high:.2f}")
    print(f"Uncertainty (half-width): ±${(pred_high - pred_low)/2:.2f}")


if __name__ == "__main__":
    main()
