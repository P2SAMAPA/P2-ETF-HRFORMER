"""
infer.py
Generate next-day ETF trading signal from the latest trained model.

Outputs latest.json (root) with:
  - recommended ETF
  - P(up) for all 6 ETFs
  - backtest equity curve on test set
  - per-ETF test metrics
  - timestamp
"""

import os
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from data_utils import (
    build_dataloaders, load_raw_df, engineer_features,
    make_labels, get_feature_names, normalise,
    TARGET_ETFS, SEQ_LEN, TRAIN_RATIO, VAL_RATIO,
)
from hrformer import build_model

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH  = "model.pt"
OUTPUT_PATH = "latest.json"


# ── Backtest helpers ──────────────────────────────────────────────────────────

def run_backtest(
    model:        torch.nn.Module,
    feature_df:   pd.DataFrame,
    label_df:     pd.DataFrame,
    test_idx:     np.ndarray,
    mean:         np.ndarray,
    std:          np.ndarray,
    trading_cost: float = 0.001,
) -> dict:
    """
    Long-only strategy: each day pick the ETF with highest P(up),
    hold for 1 day, subtract trading cost on entry + exit.
    """
    feat_names    = get_feature_names()
    model.eval()

    equity        = [1.0]
    daily_returns = []
    picks         = []
    all_proba     = []

    with torch.no_grad():
        for t in test_idx:
            x = np.stack(
                [
                    normalise(
                        feature_df[ticker][feat_names].iloc[t: t + SEQ_LEN].values,
                        mean, std,
                    )
                    for ticker in TARGET_ETFS
                ],
                axis=0,
            ).astype(np.float32)

            x_t   = torch.tensor(x).unsqueeze(0).to(DEVICE)
            proba = model.predict_proba(x_t).cpu().numpy()[0]
            all_proba.append(proba.tolist())

            pick_idx  = int(np.argmax(proba))
            pick_etf  = TARGET_ETFS[pick_idx]
            picks.append(pick_etf)

            close_now  = feature_df[pick_etf]["Close"].iloc[t + SEQ_LEN]
            close_next = (
                feature_df[pick_etf]["Close"].iloc[t + SEQ_LEN + 1]
                if t + SEQ_LEN + 1 < len(feature_df) else close_now
            )
            ret = (close_next - close_now) / (close_now + 1e-8)
            ret -= 2 * trading_cost
            daily_returns.append(float(ret))
            equity.append(equity[-1] * (1 + ret))

    daily_returns = np.array(daily_returns)
    equity        = np.array(equity)

    ann_return = float((equity[-1] ** (252 / max(len(daily_returns), 1))) - 1)
    ann_vol    = float(daily_returns.std() * np.sqrt(252))
    sharpe     = float(ann_return / (ann_vol + 1e-8))
    drawdowns  = equity / np.maximum.accumulate(equity) - 1
    max_dd     = float(drawdowns.min())
    total_ret  = float(equity[-1] - 1)

    test_dates = feature_df.index[test_idx + SEQ_LEN].strftime("%Y-%m-%d").tolist()

    return {
        "equity_curve":  equity[1:].tolist(),
        "dates":         test_dates,
        "daily_returns": daily_returns.tolist(),
        "picks":         picks,
        "all_proba":     all_proba,
        "summary": {
            "total_return":       round(total_ret,   4),
            "annualised_return":  round(ann_return,  4),
            "annualised_vol":     round(ann_vol,     4),
            "sharpe_ratio":       round(sharpe,      4),
            "max_drawdown":       round(max_dd,      4),
            "num_days":           len(daily_returns),
        },
    }


# ── Next-day signal ───────────────────────────────────────────────────────────

def generate_signal(
    model:      torch.nn.Module,
    feature_df: pd.DataFrame,
    mean:       np.ndarray,
    std:        np.ndarray,
) -> dict:
    feat_names = get_feature_names()
    t = len(feature_df) - SEQ_LEN - 1

    x = np.stack(
        [
            normalise(
                feature_df[ticker][feat_names].iloc[t: t + SEQ_LEN].values,
                mean, std,
            )
            for ticker in TARGET_ETFS
        ],
        axis=0,
    ).astype(np.float32)

    x_t = torch.tensor(x).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        proba = model.predict_proba(x_t).cpu().numpy()[0]

    signal_date = feature_df.index[t + SEQ_LEN].strftime("%Y-%m-%d")
    pick_idx    = int(np.argmax(proba))
    pick_etf    = TARGET_ETFS[pick_idx]

    return {
        "signal_date":     signal_date,
        "recommended_etf": pick_etf,
        "confidence":      round(float(proba[pick_idx]), 4),
        "probabilities":   {t: round(float(p), 4) for t, p in zip(TARGET_ETFS, proba)},
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print("model.pt not found — run train.py first.")
        return

    print("Loading data...")
    _, _, _, meta = build_dataloaders(hf_token=args.hf_token, batch_size=32)
    feature_df = meta["feature_df"]
    label_df   = meta["label_df"]
    test_idx   = meta["test_idx"]
    mean, std  = meta["mean"], meta["std"]

    print("Loading model weights...")
    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print("Running backtest...")
    backtest = run_backtest(model, feature_df, label_df, test_idx, mean, std)

    print("Generating next-day signal...")
    signal = generate_signal(model, feature_df, mean, std)

    # Load training metrics
    train_metrics = {}
    if os.path.exists("train_metrics.json"):
        with open("train_metrics.json") as f:
            train_metrics = json.load(f)

    output = {
        "generated_at":  datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "signal":        signal,
        "backtest":      backtest,
        "model_metrics": train_metrics.get("test_metrics", {}),
        "training_info": {
            "best_val_f1": train_metrics.get("best_val_f1"),
            "epochs_run":  train_metrics.get("epochs_run"),
            "history":     train_metrics.get("history", []),
        },
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSignal saved → {OUTPUT_PATH}")
    print(f"  Recommended ETF : {signal['recommended_etf']}")
    print(f"  Confidence      : {signal['confidence']:.1%}")
    print(f"  Sharpe Ratio    : {backtest['summary']['sharpe_ratio']:.2f}")
    print(f"  Total Return    : {backtest['summary']['total_return']:.1%}")

    # Push to HF Hub
    if args.hf_token:
        try:
            from huggingface_hub import HfApi
            api  = HfApi(token=args.hf_token)
            repo = "P2SAMAPA/etf-hrformer-model"
            api.upload_file(path_or_fileobj=OUTPUT_PATH,
                            path_in_repo="latest.json",
                            repo_id=repo, repo_type="model")
            print("Signal pushed to HF Hub.")
        except Exception as e:
            print(f"HF push failed (non-fatal): {e}")


if __name__ == "__main__":
    main()
