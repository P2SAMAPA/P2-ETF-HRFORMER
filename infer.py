"""
infer.py
Generate next-day signal using best walk-forward model.
Reads walk_forward_results.json, picks best mode by ann. return,
uses that model for today's signal.
"""

import os, sys, json, argparse
from datetime import datetime
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import (
    load_raw_df, engineer_features, make_labels, get_feature_names,
    normalise, TARGET_ETFS, SEQ_LEN, PRED_HORIZON
)
from hrformer import build_model

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UP_THRESHOLD = 0.65
OUTPUT_PATH  = "latest.json"


def generate_signal(model, feature_df, mean, std):
    feat_names = get_feature_names()
    t = len(feature_df) - SEQ_LEN
    
    # CRITICAL FIX: Ensure we have enough data
    if t < 0:
        raise ValueError(f"Not enough data. Need {SEQ_LEN} rows, have {len(feature_df)}")
    
    x = np.stack([normalise(
        feature_df[tk][feat_names].iloc[t:t+SEQ_LEN].values, mean, std)
        for tk in TARGET_ETFS], axis=0).astype(np.float32)
    
    with torch.no_grad():
        proba = model.predict_proba(
            torch.from_numpy(x).unsqueeze(0).to(DEVICE)).cpu().numpy()[0]

    last_data_date = feature_df.index[t + SEQ_LEN - 1]
    from pandas.tseries.offsets import BDay
    next_trade_date = (last_data_date + BDay(1)).strftime("%Y-%m-%d")

    pick_i = int(np.argmax(proba))
    
    # CRITICAL FIX: Added detailed logging for debugging
    print(f"  Probabilities: {dict(zip(TARGET_ETFS, [round(p, 4) for p in proba]))}")
    print(f"  Selected: {TARGET_ETFS[pick_i]} with P(up)={proba[pick_i]:.4f}")
    
    return {
        "signal_date": next_trade_date,
        "data_date": last_data_date.strftime("%Y-%m-%d"),
        "recommended_etf": TARGET_ETFS[pick_i],
        "confidence": round(float(proba[pick_i]), 4),
        "will_trade": bool(proba[pick_i] > UP_THRESHOLD),
        "probabilities": {t: round(float(p), 4)
                         for t, p in zip(TARGET_ETFS, proba)},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    # Load walk-forward results
    if not os.path.exists("walk_forward_results.json"):
        print("walk_forward_results.json not found — run train.py first.")
        return

    with open("walk_forward_results.json") as f:
        wf = json.load(f)

    best_mode = wf.get("best_mode", "fixed")
    model_path = f"model_{best_mode}.pt"

    if not os.path.exists(model_path):
        print(f"{model_path} not found.")
        return

    ann_return = wf.get(best_mode, {}).get('aggregate', {}).get('summary', {}).get('annualised_return', 0)
    print(f"Best mode: {best_mode.upper()} (ann. return: {ann_return:.1%})")

    print("Loading data...")
    raw_df = load_raw_df(args.hf_token)
    feature_df = engineer_features(raw_df)
    
    # CRITICAL FIX: Ensure feature_df is aligned with model expectations
    # Remove last PRED_HORIZON rows to match training data preparation
    feature_df_trim = feature_df.iloc[:-PRED_HORIZON] if len(feature_df) > PRED_HORIZON else feature_df

    print("Loading model...")
    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # Compute mean/std from last FIXED_YEARS*252 rows of the best mode's last fold
    # CRITICAL FIX: Use consistent normalization with training
    feat_names = get_feature_names()
    
    # Use last 504 trading days (~2 years) for normalization stats
    norm_window = min(504, len(feature_df_trim))
    recent_data = np.concatenate(
        [feature_df_trim[tk][feat_names].iloc[-norm_window:].values
         for tk in TARGET_ETFS], axis=0)
    mean = recent_data.mean(0)
    std = recent_data.std(0)
    
    # CRITICAL FIX: Prevent division by zero in normalization
    std = np.where(std == 0, 1.0, std)

    print("Generating signal...")
    signal = generate_signal(model, feature_df_trim, mean, std)
    print(f"  Trade decision: {'EXECUTE' if signal['will_trade'] else 'NO TRADE'} "
          f"(threshold: {UP_THRESHOLD})")

    output = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "best_mode": best_mode,
        "signal": signal,
        "expanding": wf.get("expanding", {}),
        "fixed": wf.get("fixed", {}),
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Signal saved → {OUTPUT_PATH}")

    if args.hf_token:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=args.hf_token)
            api.upload_file(path_or_fileobj=OUTPUT_PATH,
                            path_in_repo="latest.json",
                            repo_id="P2SAMAPA/etf-hrformer-model",
                            repo_type="model")
            print("Pushed to HF Hub.")
        except Exception as e:
            print(f"HF push failed: {e}")


if __name__ == "__main__":
    main()
