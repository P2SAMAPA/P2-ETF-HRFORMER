"""
infer_equity.py
Generate 48-day ahead signal using trained equity ETF models.
Mirrors infer.py exactly — same dual-mode hero logic, same history schema.
Equity-specific files use the suffix _equity to avoid collision with FI files.
"""

import os, sys, json, argparse
from datetime import datetime
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils_equity import (
    load_raw_df, engineer_features, get_feature_names,
    normalise, TARGET_ETFS, SEQ_LEN, PRED_HORIZON
)
from hrformer import build_model

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_PATH  = "latest_equity.json"
HISTORY_PATH = "prediction_history_equity.json"
NUM_ETFS     = len(TARGET_ETFS)   # 13


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def generate_signal(model, feature_df, mean, std, mode_name=""):
    feat_names = get_feature_names()
    t = len(feature_df) - SEQ_LEN

    if t < 0:
        raise ValueError(f"Need {SEQ_LEN} rows, have {len(feature_df)}")

    x = np.stack([
        normalise(feature_df[tk][feat_names].iloc[t:t + SEQ_LEN].values, mean, std)
        for tk in TARGET_ETFS
    ], axis=0).astype(np.float32)

    model.eval()
    with torch.no_grad():
        x_tensor   = torch.from_numpy(x).unsqueeze(0).to(DEVICE)
        pred_returns = model.predict_returns(x_tensor).cpu().numpy()[0]

        if mode_name:
            print(f"  {mode_name} predicted 48-day returns:")
            for i, ticker in enumerate(TARGET_ETFS):
                print(f"    {ticker}: {pred_returns[i]*100:+.2f}%")

    top_idx = int(np.argmax(pred_returns))
    top_etf = TARGET_ETFS[top_idx]
    top_return = float(pred_returns[top_idx])

    last_data_date = feature_df.index[t + SEQ_LEN - 1]
    from pandas.tseries.offsets import BDay
    entry_date = (last_data_date + BDay(1)).strftime("%Y-%m-%d")
    target_48  = (last_data_date + BDay(PRED_HORIZON)).strftime("%Y-%m-%d")

    return {
        "entry_date":       entry_date,
        "target_48_date":   target_48,
        "data_date":        last_data_date.strftime("%Y-%m-%d"),
        "recommended_etf":  top_etf,
        "predicted_return": round(top_return, 4),
        "predicted_returns": {ticker: round(float(pred_returns[i]), 4)
                              for i, ticker in enumerate(TARGET_ETFS)},
        "strategy": "daily_rebalance",
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(mode, device, hf_token=None):
    model_path = f"model_equity_{mode}.pt"
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except FileNotFoundError:
        from huggingface_hub import hf_hub_download
        local_path = hf_hub_download(
            repo_id="P2SAMAPA/etf-hrformer-model",
            filename=model_path,
            repo_type="model",
            token=hf_token,
        )
        checkpoint = torch.load(local_path, map_location=device, weights_only=False)

    # Build model for 13 equity ETFs
    model = build_model(num_etfs=NUM_ETFS, seq_len=SEQ_LEN, num_features=8).to(device)
    model.load_state_dict(checkpoint["model"])
    mean = checkpoint["mean"]
    std  = checkpoint["std"]
    std  = np.where(std == 0, 1.0, std)
    return model, mean, std


# ---------------------------------------------------------------------------
# Combined results loader
# ---------------------------------------------------------------------------

def load_combined_results(hf_token=None):
    combined    = {}
    mode_metrics = {}

    for mode in ["expanding", "shrinking"]:
        local_file = f"walk_forward_results_equity_{mode}.json"
        data = None

        if os.path.exists(local_file):
            try:
                with open(local_file) as f:
                    data = json.load(f)
            except Exception as e:
                print(f"  Local {mode}: {e}")

        if data is None:
            try:
                import requests
                url = (f"https://huggingface.co/P2SAMAPA/etf-hrformer-model/resolve/main/"
                       f"walk_forward_results_equity_{mode}.json")
                r = requests.get(url, timeout=10)
                if r.status_code == 200:
                    data = r.json()
            except Exception as e:
                print(f"  HF {mode}: {e}")

        if data:
            combined[mode] = data
            agg     = data.get("aggregate", {})
            summary = agg.get("summary", {})
            mode_metrics[mode] = {
                "annualised_return": summary.get("annualised_return", -999),
                "sharpe_ratio":      summary.get("sharpe_ratio",      -999),
                "max_drawdown":      summary.get("max_drawdown",      -999),
                "annualised_vol":    summary.get("annualised_vol",    -999),
                "total_return":      summary.get("total_return",      -999),
            }

    # Pick best mode by Sharpe
    best_mode, best_score = None, -float("inf")
    for mode, m in mode_metrics.items():
        score = m.get("sharpe_ratio", -999)
        if score != -999 and not np.isnan(score) and score > best_score:
            best_score = score
            best_mode  = mode

    if best_mode:
        combined["best_historical_mode"]   = best_mode
        combined["historical_performance"] = mode_metrics[best_mode]
        combined["all_metrics"]            = mode_metrics

    return combined, best_mode, mode_metrics


# ---------------------------------------------------------------------------
# History management  (identical schema to FI history)
# ---------------------------------------------------------------------------

def load_history(hf_token=None):
    history = {"predictions": []}
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH) as f:
                history = json.load(f)
            print(f"Loaded equity history: {len(history.get('predictions', []))} entries.")
        except Exception as e:
            print(f"Error loading equity history: {e}")
    else:
        try:
            from huggingface_hub import hf_hub_download
            local_path = hf_hub_download(
                repo_id="P2SAMAPA/etf-hrformer-model",
                filename=HISTORY_PATH,
                repo_type="model",
                token=hf_token,
            )
            with open(local_path) as f:
                history = json.load(f)
            print(f"Downloaded equity history: {len(history.get('predictions', []))} entries.")
        except Exception:
            print("No equity history found, starting fresh.")
    return history


def update_history(history, new_signal, feature_df, hero_mode):
    predictions = history.get("predictions", [])

    # Deduplicate
    seen, unique = {}, []
    for p in predictions:
        key = (p.get("entry_date"), p.get("recommended_etf"))
        if key not in seen:
            seen[key] = True
            unique.append(p)
    predictions = unique

    # Fill in actual 1-day returns for matured entries
    today = datetime.now().date()
    for p in predictions:
        if p.get("actual_return") is not None:
            continue
        entry_date_str = p.get("entry_date")
        if not entry_date_str or entry_date_str not in feature_df.index:
            continue
        entry_idx = feature_df.index.get_loc(entry_date_str)
        if entry_idx + 1 >= len(feature_df):
            continue
        next_date     = feature_df.index[entry_idx + 1]
        next_date_str = next_date.strftime("%Y-%m-%d")
        if next_date.date() <= today:
            ticker = p.get("recommended_etf")
            try:
                entry_close = feature_df[ticker]["Close"].loc[entry_date_str]
                exit_close  = feature_df[ticker]["Close"].loc[next_date_str]
                actual = (exit_close - entry_close) / entry_close
                p["actual_return"] = round(float(actual), 4)
                print(f"Updated {entry_date_str} {ticker}: {actual*100:+.2f}%")
            except Exception as e:
                print(f"Could not compute actual return for {entry_date_str}: {e}")

    # Append new signal
    predictions.append({
        "entry_date":       new_signal["entry_date"],
        "target_48_date":   new_signal["target_48_date"],
        "data_date":        new_signal["data_date"],
        "recommended_etf":  new_signal["recommended_etf"],
        "predicted_return": new_signal["predicted_return"],
        "actual_return":    None,
        "hero_mode":        hero_mode,
    })

    predictions.sort(key=lambda x: x["entry_date"], reverse=True)
    history["predictions"]  = predictions
    history["last_updated"] = datetime.utcnow().isoformat()
    return history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    results, historical_best_mode, mode_metrics = load_combined_results(args.hf_token)

    if not historical_best_mode:
        print("[EQUITY] No results found. Run train_equity.py first.")
        return

    print(f"\n[EQUITY] Best historical mode: {historical_best_mode.upper()}")

    print("\nLoading equity data...")
    raw_df     = load_raw_df(args.hf_token)
    feature_df = engineer_features(raw_df)

    if len(feature_df) < SEQ_LEN + 10:
        print("[EQUITY] Insufficient data")
        return

    # Generate signals for both modes
    mode_signals = {}
    for mode in ["expanding", "shrinking"]:
        print(f"\nLoading equity {mode} model...")
        try:
            model, mean, std = load_model(mode, DEVICE, args.hf_token)
            sig = generate_signal(model, feature_df, mean, std, mode_name=mode)
            mode_signals[mode] = sig
        except Exception as e:
            print(f"Failed equity signal for {mode}: {e}")
            mode_signals[mode] = None

    # Hero = highest predicted return across both models
    hero_mode, hero_signal, max_return = None, None, -float("inf")
    for mode, sig in mode_signals.items():
        if sig is None:
            continue
        ret = sig.get("predicted_return", -float("inf"))
        if ret > max_return:
            max_return = ret
            hero_mode  = mode
            hero_signal = sig

    if hero_signal is None:
        print("[EQUITY] No valid signals generated.")
        return

    print(f"\n[EQUITY] Hero: {hero_mode.upper()} predicted={max_return*100:+.2f}%")

    history = load_history(args.hf_token)
    history = update_history(history, hero_signal, feature_df, hero_mode)

    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved → {HISTORY_PATH}")

    output = {
        "generated_at":         datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "best_historical_mode": historical_best_mode,
        "hero_mode":            hero_mode,
        "signal":               hero_signal,
        "mode_predictions":     mode_signals,
        "historical_performance": results.get("historical_performance", {}),
        "mode_comparison":      mode_metrics,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Signal saved → {OUTPUT_PATH}")

    if args.hf_token:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=args.hf_token)
            repo_id = "P2SAMAPA/etf-hrformer-model"
            for path in [OUTPUT_PATH, HISTORY_PATH]:
                api.upload_file(path_or_fileobj=path, path_in_repo=path,
                                repo_id=repo_id, repo_type="model")
            print("Pushed equity artefacts to HF Hub.")
        except Exception as e:
            print(f"HF push failed: {e}")


if __name__ == "__main__":
    main()
