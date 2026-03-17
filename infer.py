"""
infer.py
Generate next-day signal using best walk-forward model.
"""

import os, sys, json, argparse
from datetime import datetime
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import (
    load_raw_df, engineer_features, get_feature_names,
    normalise, TARGET_ETFS, SEQ_LEN, PRED_HORIZON
)
from hrformer import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_PATH = "latest.json"


def generate_signal(model, feature_df, mean, std, threshold=0.50):
    """Generate trading signal."""
    feat_names = get_feature_names()
    t = len(feature_df) - SEQ_LEN
    
    if t < 0:
        raise ValueError(f"Need {SEQ_LEN} rows, have {len(feature_df)}")

    x = np.stack([
        normalise(feature_df[tk][feat_names].iloc[t:t+SEQ_LEN].values, mean, std)
        for tk in TARGET_ETFS
    ], axis=0).astype(np.float32)
    
    model.eval()
    with torch.no_grad():
        x_tensor = torch.from_numpy(x).unsqueeze(0).to(DEVICE)
        proba = model.predict_proba(x_tensor).cpu().numpy()[0]
        
        print("  Probabilities (P-up):")
        for i, ticker in enumerate(TARGET_ETFS):
            print(f"    {ticker}: {proba[i]:.4f}")

    last_data_date = feature_df.index[t + SEQ_LEN - 1]
    from pandas.tseries.offsets import BDay
    next_trade_date = (last_data_date + BDay(1)).strftime("%Y-%m-%d")

    pick_i = int(np.argmax(proba))
    confidence = float(proba[pick_i])
    
    print(f"  Selected: {TARGET_ETFS[pick_i]} (P-up={confidence:.4f})")
    print(f"  Will trade: {confidence > threshold} (threshold={threshold})")

    return {
        "signal_date": next_trade_date,
        "data_date": last_data_date.strftime("%Y-%m-%d"),
        "recommended_etf": TARGET_ETFS[pick_i],
        "confidence": round(confidence, 4),
        "will_trade": bool(confidence > threshold),
        "probabilities": {ticker: round(float(proba[i]), 4) 
                         for i, ticker in enumerate(TARGET_ETFS)},
        "threshold": threshold,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--threshold", type=float, default=0.50)
    args = parser.parse_args()

    # Load results
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

    ann_ret = wf.get(best_mode, {}).get("aggregate", {}).get("summary", {}).get("annualised_return", 0)
    print(f"Best mode: {best_mode.upper()} (ann. return: {ann_ret:.2%})")

    print("Loading data...")
    raw_df = load_raw_df(args.hf_token)
    feature_df = engineer_features(raw_df)
    feature_df_trim = feature_df.iloc[:-PRED_HORIZON]

    print("Loading model...")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model = build_model().to(DEVICE)
    model.load_state_dict(checkpoint['model'])
    
    mean = checkpoint['mean']
    std = checkpoint['std']
    std = np.where(std == 0, 1.0, std)

    print("Generating signal...")
    signal = generate_signal(model, feature_df_trim, mean, std, threshold=args.threshold)

    output = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "best_mode": best_mode,
        "signal": signal,
        "expanding": wf.get("expanding", {}),
        "fixed": wf.get("fixed", {}),
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSignal saved → {OUTPUT_PATH}")

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
