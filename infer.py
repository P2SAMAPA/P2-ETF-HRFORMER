"""
infer.py
Generate next-day ETF trading signal from the latest trained model.

Backtest logic:
  - Uses feature_df_train (same df used during training) so indices are correct
  - Only enters a trade when model predicts "up" (P(up) > 0.5)
  - Otherwise stays in cash (0% return that day)
  - 0.1% round-trip trading cost only applied when a trade is made
"""

import os
import sys
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import (
    build_dataloaders, engineer_features, load_raw_df,
    get_feature_names, normalise,
    TARGET_ETFS, SEQ_LEN, TRAIN_RATIO, VAL_RATIO, PRED_HORIZON,
)
from hrformer import build_model

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH  = "model.pt"
OUTPUT_PATH = "latest.json"
UP_THRESHOLD = 0.55   # minimum P(up) to enter a trade — raised to reduce false signals


# ── Backtest ──────────────────────────────────────────────────────────────────

def run_backtest(
    model:        torch.nn.Module,
    feature_df:   pd.DataFrame,   # training-aligned df (trimmed by PRED_HORIZON)
    test_idx:     np.ndarray,
    mean:         np.ndarray,
    std:          np.ndarray,
    trading_cost: float = 0.001,
) -> dict:
    """
    For each day in test_idx:
      - Window  = rows [t : t+SEQ_LEN]   in feature_df
      - Label   = row  [t+SEQ_LEN]       close vs row [t+SEQ_LEN-1] close
      - Pick ETF with highest P(up); only trade if P(up) > UP_THRESHOLD
      - Return  = (close[t+SEQ_LEN] - close[t+SEQ_LEN-1]) / close[t+SEQ_LEN-1]
    This matches exactly how labels are defined in data_utils.make_labels().
    """
    feat_names = get_feature_names()
    model.eval()

    equity        = [1.0]
    daily_returns = []
    picks         = []
    trade_flags   = []
    correct_flags = []   # True/False/None per day
    all_proba     = []

    with torch.no_grad():
        for t in test_idx:
            # ── Build input ──────────────────────────────────────────────────
            x = np.stack(
                [normalise(
                    feature_df[ticker][feat_names].iloc[t: t + SEQ_LEN].values,
                    mean, std)
                 for ticker in TARGET_ETFS],
                axis=0,
            ).astype(np.float32)

            x_t   = torch.tensor(x).unsqueeze(0).to(DEVICE)
            proba = model.predict_proba(x_t).cpu().numpy()[0]   # (6,)
            all_proba.append(proba.tolist())

            pick_idx = int(np.argmax(proba))
            pick_etf = TARGET_ETFS[pick_idx]
            picks.append(pick_etf)

            # ── Realised return ──────────────────────────────────────────────
            # Prediction target: will close[t+SEQ_LEN] > close[t+SEQ_LEN-1]?
            # So the return we realise is exactly that day's move.
            idx_today = t + SEQ_LEN - 1
            idx_next  = t + SEQ_LEN

            if idx_next >= len(feature_df):
                daily_returns.append(0.0)
                trade_flags.append(False)
                equity.append(equity[-1])
                continue

            close_today = feature_df[pick_etf]["Close"].iloc[idx_today]
            close_next  = feature_df[pick_etf]["Close"].iloc[idx_next]
            raw_ret     = (close_next - close_today) / (close_today + 1e-8)

            # Track whether the pick was actually correct (for hit rate)
            actual_up = raw_ret > 0
            pick_correct = bool(actual_up)  # model picked this ETF; was it right?

            # Only trade if model is confident enough
            if proba[pick_idx] > UP_THRESHOLD:
                ret = raw_ret - 2 * trading_cost
                trade_flags.append(True)
            else:
                ret = 0.0   # stay in cash
                trade_flags.append(False)
                pick_correct = None  # no trade, not counted

            correct_flags.append(pick_correct)
            daily_returns.append(float(ret))
            equity.append(equity[-1] * (1 + ret))

    daily_returns = np.array(daily_returns)
    equity        = np.array(equity)
    n_trades      = sum(trade_flags)
    traded_correct = [c for c in correct_flags if c is True]
    traded_total   = [c for c in correct_flags if c is not None]
    hit_rate       = len(traded_correct) / max(len(traded_total), 1)

    ann_return = float((equity[-1] ** (252 / max(len(daily_returns), 1))) - 1)
    ann_vol    = float(daily_returns.std() * np.sqrt(252))
    sharpe     = float(ann_return / (ann_vol + 1e-8))
    drawdowns  = equity / np.maximum.accumulate(equity) - 1
    max_dd     = float(drawdowns.min())
    total_ret  = float(equity[-1] - 1)

    # Dates: the prediction is made BEFORE day t+SEQ_LEN opens
    test_dates = feature_df.index[
        [t + SEQ_LEN for t in test_idx if t + SEQ_LEN < len(feature_df)]
    ].strftime("%Y-%m-%d").tolist()

    # Trim arrays to match dates length
    n = len(test_dates)

    return {
        "equity_curve":  equity[1:n+1].tolist(),
        "dates":         test_dates,
        "daily_returns": daily_returns[:n].tolist(),
        "picks":         picks[:n],
        "all_proba":     all_proba[:n],
        "n_trades":      n_trades,
        "summary": {
            "total_return":      round(total_ret,  4),
            "annualised_return": round(ann_return, 4),
            "annualised_vol":    round(ann_vol,    4),
            "sharpe_ratio":      round(sharpe,     4),
            "max_drawdown":      round(max_dd,     4),
            "num_days":          len(daily_returns),
            "num_trades":        n_trades,
            "trade_rate":        round(n_trades / max(len(daily_returns), 1), 4),
            "hit_rate":          round(hit_rate, 4),
        },
    }


# ── Next-day signal ───────────────────────────────────────────────────────────

def generate_signal(
    model:      torch.nn.Module,
    feature_df: pd.DataFrame,    # full df including latest date
    mean:       np.ndarray,
    std:        np.ndarray,
) -> dict:
    """Use the very last SEQ_LEN rows for tomorrow's signal."""
    feat_names = get_feature_names()

    # Last valid window: rows [-SEQ_LEN:]
    t = len(feature_df) - SEQ_LEN

    x = np.stack(
        [normalise(
            feature_df[ticker][feat_names].iloc[t: t + SEQ_LEN].values,
            mean, std)
         for ticker in TARGET_ETFS],
        axis=0,
    ).astype(np.float32)

    x_t = torch.tensor(x).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        proba = model.predict_proba(x_t).cpu().numpy()[0]

    # Signal date = last date in the window (data used to make prediction)
    signal_date = feature_df.index[t + SEQ_LEN - 1].strftime("%Y-%m-%d")
    pick_idx    = int(np.argmax(proba))
    pick_etf    = TARGET_ETFS[pick_idx]

    return {
        "signal_date":     signal_date,
        "recommended_etf": pick_etf,
        "confidence":      round(float(proba[pick_idx]), 4),
        "will_trade":      bool(proba[pick_idx] > UP_THRESHOLD),
        "probabilities":   {t: round(float(p), 4)
                            for t, p in zip(TARGET_ETFS, proba)},
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

    # feature_df_train: trimmed (used for backtest — same alignment as training)
    # feature_df:       full (used for latest signal)
    feature_df_full  = meta["feature_df"]          # full, for signal generation
    mean, std        = meta["mean"], meta["std"]
    test_idx         = meta["test_idx"]

    # Reconstruct the trimmed df for backtest (drop last PRED_HORIZON rows)
    feature_df_train = feature_df_full.iloc[:-PRED_HORIZON]

    print("Loading model weights...")
    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print("Running backtest on test set...")
    backtest = run_backtest(model, feature_df_train, test_idx, mean, std)
    print(f"  Total return  : {backtest['summary']['total_return']:.1%}")
    print(f"  Sharpe ratio  : {backtest['summary']['sharpe_ratio']:.2f}")
    print(f"  Trades made   : {backtest['summary']['num_trades']} / {backtest['summary']['num_days']} days")

    print("Generating next-day signal...")
    signal = generate_signal(model, feature_df_full, mean, std)
    print(f"  Pick: {signal['recommended_etf']}  P(up)={signal['confidence']:.1%}  will_trade={signal['will_trade']}")

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
        },
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Signal saved → {OUTPUT_PATH}")

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
