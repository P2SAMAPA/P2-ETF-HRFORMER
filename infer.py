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
    x = np.stack([normalise(
        feature_df[tk][feat_names].iloc[t:t+SEQ_LEN].values, mean, std)
        for tk in TARGET_ETFS], axis=0).astype(np.float32)
    with torch.no_grad():
        proba = model.predict_proba(
            torch.from_numpy(x).unsqueeze(0).to(DEVICE)).cpu().numpy()[0]

    last_data_date  = feature_df.index[t + SEQ_LEN - 1]
    from pandas.tseries.offsets import BDay
    next_trade_date = (last_data_date + BDay(1)).strftime("%Y-%m-%d")

    pick_i = int(np.argmax(proba))
    return {
        "signal_date":     next_trade_date,
        "data_date":       last_data_date.strftime("%Y-%m-%d"),
        "recommended_etf": TARGET_ETFS[pick_i],
        "confidence":      round(float(proba[pick_i]), 4),
        "will_trade":      bool(proba[pick_i] > UP_THRESHOLD),
        "probabilities":   {t: round(float(p), 4)
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

    print(f"Best mode: {best_mode.upper()} "
          f"(ann. return: {wf[best_mode]['aggregate']['summary']['annualised_return']:.1%})")

    print("Loading data...")
    raw_df     = load_raw_df(args.hf_token)
    feature_df = engineer_features(raw_df)

    print("Loading model...")
    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # Compute mean/std from last FIXED_YEARS*252 rows of the best mode's last fold
    last_fold   = wf[best_mode]["folds"][-1]
    feat_names  = get_feature_names()
    feature_df_trim = feature_df.iloc[:-PRED_HORIZON]
    recent_data = np.concatenate(
        [feature_df_trim[tk][feat_names].iloc[-504:].values   # ~2 years
         for tk in TARGET_ETFS], axis=0)
    mean = recent_data.mean(0)
    std  = recent_data.std(0)

    print("Generating signal...")
    signal = generate_signal(model, feature_df, mean, std)
    print(f"  Pick: {signal['recommended_etf']}  "
          f"P(up)={signal['confidence']:.1%}  "
          f"trade={signal['will_trade']}")

    output = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "best_mode":    best_mode,
        "signal":       signal,
        "expanding":    wf.get("expanding", {}),
        "fixed":        wf.get("fixed", {}),
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
