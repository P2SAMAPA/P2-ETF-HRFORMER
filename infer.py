"""
infer.py
Generate 48-day ahead signal using trained model.
Selects ETF with highest predicted return across all models.
Maintains prediction history with actual returns.
"""

import os, sys, json, argparse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import (
    load_raw_df, engineer_features, get_feature_names,
    normalise, TARGET_ETFS, SEQ_LEN, PRED_HORIZON
)
from hrformer import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_PATH = "latest.json"
HISTORY_PATH = "prediction_history.json"


def generate_signal(model, feature_df, mean, std, mode_name=""):
    """Generate 48-day ahead signal for SINGLE ETF, optionally print mode name."""
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
        pred_returns = model.predict_returns(x_tensor).cpu().numpy()[0]
        
        if mode_name:
            print(f"  {mode_name} predicted 48-day returns:")
            for i, ticker in enumerate(TARGET_ETFS):
                print(f"    {ticker}: {pred_returns[i]*100:+.2f}%")

    # SINGLE ETF: pick highest predicted return
    top_idx = int(np.argmax(pred_returns))
    top_etf = TARGET_ETFS[top_idx]
    top_return = float(pred_returns[top_idx])
    
    last_data_date = feature_df.index[t + SEQ_LEN - 1]
    from pandas.tseries.offsets import BDay
    next_trade_date = (last_data_date + BDay(1)).strftime("%Y-%m-%d")
    hold_until = (last_data_date + BDay(PRED_HORIZON)).strftime("%Y-%m-%d")

    return {
        "signal_date": next_trade_date,
        "hold_until": hold_until,
        "data_date": last_data_date.strftime("%Y-%m-%d"),
        "recommended_etf": top_etf,
        "predicted_return": round(top_return, 4),
        "predicted_returns": {ticker: round(float(pred_returns[i]), 4) 
                             for i, ticker in enumerate(TARGET_ETFS)},
        "strategy": "buy_hold_48d_single",
    }


def load_model(mode, device, hf_token=None):
    """Load model and stats for a given mode, from local or HF."""
    model_path = f"model_{mode}.pt"
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except FileNotFoundError:
        from huggingface_hub import hf_hub_download
        local_path = hf_hub_download(
            repo_id="P2SAMAPA/etf-hrformer-model",
            filename=model_path,
            repo_type="model",
            token=hf_token
        )
        checkpoint = torch.load(local_path, map_location=device, weights_only=False)
    
    model = build_model().to(device)
    model.load_state_dict(checkpoint['model'])
    mean = checkpoint['mean']
    std = checkpoint['std']
    std = np.where(std == 0, 1.0, std)
    return model, mean, std


def load_combined_results():
    """Load and combine results from both modes with PROPER metric extraction."""
    combined = {}
    mode_metrics = {}
    
    for mode in ["expanding", "shrinking"]:
        try:
            with open(f"walk_forward_results_{mode}.json", "r") as f:
                data = json.load(f)
                combined[mode] = data
                
                # PROPER extraction: check multiple possible locations for metrics
                metrics = {}
                
                # Try aggregate.summary first (correct location)
                agg = data.get("aggregate", {})
                if "summary" in agg:
                    summary = agg["summary"]
                    metrics = {
                        "annualised_return": summary.get("annualised_return", -999),
                        "sharpe_ratio": summary.get("sharpe_ratio", -999),
                        "max_drawdown": summary.get("max_drawdown", -999),
                        "annualised_vol": summary.get("annualised_vol", -999),
                        "total_return": summary.get("total_return", -999),
                    }
                # Fallback to direct keys
                else:
                    metrics = {
                        "annualised_return": data.get("annualised_return", -999),
                        "sharpe_ratio": data.get("sharpe_ratio", -999),
                        "max_drawdown": data.get("max_drawdown", -999),
                        "annualised_vol": data.get("annualised_vol", -999),
                        "total_return": data.get("total_return", -999),
                    }
                
                mode_metrics[mode] = metrics
                print(f"\n  {mode.upper()} metrics found:")
                for k, v in metrics.items():
                    print(f"    {k}: {v}")
                    
        except FileNotFoundError:
            print(f"  {mode}: local file not found, trying HF...")
            # Try HF Hub
            try:
                import requests
                url = f"https://huggingface.co/P2SAMAPA/etf-hrformer-model/resolve/main/walk_forward_results_{mode}.json"
                r = requests.get(url, timeout=10)
                if r.status_code == 200:
                    data = r.json()
                    combined[mode] = data
                    
                    # Same extraction logic for HF data
                    agg = data.get("aggregate", {})
                    if "summary" in agg:
                        summary = agg["summary"]
                        mode_metrics[mode] = {
                            "annualised_return": summary.get("annualised_return", -999),
                            "sharpe_ratio": summary.get("sharpe_ratio", -999),
                            "max_drawdown": summary.get("max_drawdown", -999),
                            "annualised_vol": summary.get("annualised_vol", -999),
                        }
                    else:
                        mode_metrics[mode] = {
                            "annualised_return": data.get("annualised_return", -999),
                            "sharpe_ratio": data.get("sharpe_ratio", -999),
                            "max_drawdown": data.get("max_drawdown", -999),
                            "annualised_vol": data.get("annualised_vol", -999),
                        }
            except Exception as e:
                print(f"  HF fetch failed for {mode}: {e}")
    
    # Determine best mode based on historical metrics (for display only, not used for hero)
    best_historical_mode = None
    best_historical_score = -float('inf')
    
    print("\n  Historical performance comparison:")
    for mode in ["expanding", "shrinking"]:
        if mode in mode_metrics:
            m = mode_metrics[mode]
            sharpe = m.get("sharpe_ratio", -999)
            if sharpe != -999 and not np.isnan(sharpe):
                score = sharpe
                metric_used = "sharpe"
            else:
                score = m.get("annualised_return", -999)
                metric_used = "return"
            
            print(f"    {mode}: score={score:.4f} (using {metric_used})")
            
            if score > best_historical_score:
                best_historical_score = score
                best_historical_mode = mode
    
    if best_historical_mode:
        combined["best_historical_mode"] = best_historical_mode
        combined["historical_performance"] = mode_metrics[best_historical_mode]
        combined["all_metrics"] = mode_metrics
        print(f"\n  Best historical mode: {best_historical_mode.upper()}")
        print(f"  Performance: {combined['historical_performance']}")
    
    return combined, best_historical_mode, mode_metrics


def load_history(hf_token=None):
    """Load prediction history from local file or HF Hub."""
    history = {"predictions": []}
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, "r") as f:
                history = json.load(f)
            print(f"Loaded history with {len(history.get('predictions', []))} entries.")
        except Exception as e:
            print(f"Error loading local history: {e}")
    else:
        # Try to download from HF
        try:
            from huggingface_hub import hf_hub_download
            local_path = hf_hub_download(
                repo_id="P2SAMAPA/etf-hrformer-model",
                filename=HISTORY_PATH,
                repo_type="model",
                token=hf_token
            )
            with open(local_path, "r") as f:
                history = json.load(f)
            print(f"Downloaded history with {len(history.get('predictions', []))} entries.")
        except Exception as e:
            print(f"No history file found on HF, starting fresh.")
    return history


def update_history(history, new_signal, feature_df):
    """
    Update history:
    - Compute actual returns for any matured predictions.
    - Append new prediction.
    """
    predictions = history.get("predictions", [])
    today = datetime.now().date()
    
    # Update matured predictions
    for p in predictions:
        if p.get("actual_return") is not None:
            continue  # already computed
        exit_date_str = p.get("hold_until")
        if not exit_date_str:
            continue
        exit_date = datetime.strptime(exit_date_str, "%Y-%m-%d").date()
        if exit_date <= today:
            # Compute actual return
            entry_date_str = p.get("signal_date")
            ticker = p.get("recommended_etf")
            try:
                entry_close = feature_df[ticker]["Close"].loc[entry_date_str]
                exit_close = feature_df[ticker]["Close"].loc[exit_date_str]
                actual_return = (exit_close - entry_close) / entry_close
                p["actual_return"] = round(float(actual_return), 4)
                print(f"Updated {entry_date_str} {ticker}: actual return = {actual_return*100:+.2f}%")
            except Exception as e:
                print(f"Could not compute actual return for {entry_date_str}: {e}")
                p["actual_return"] = None  # mark as failed (maybe missing data)
    
    # Append new prediction (actual_return = None)
    new_entry = {
        "signal_date": new_signal["signal_date"],
        "hold_until": new_signal["hold_until"],
        "data_date": new_signal["data_date"],
        "recommended_etf": new_signal["recommended_etf"],
        "predicted_return": new_signal["predicted_return"],
        "actual_return": None,
        "hero_mode": hero_mode,  # we'll set this later
    }
    predictions.append(new_entry)
    
    # Sort by signal_date descending (most recent first)
    predictions.sort(key=lambda x: x["signal_date"], reverse=True)
    history["predictions"] = predictions
    history["last_updated"] = datetime.utcnow().isoformat()
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    # Load combined results to get historical best mode (not used for hero)
    results, historical_best_mode, mode_metrics = load_combined_results()
    
    if not historical_best_mode:
        print("No results found. Run training first.")
        return

    print(f"\nBest historical mode: {historical_best_mode.upper()}")
    print(f"Historical performance metrics: {results.get('historical_performance', {})}")

    print("\nLoading data...")
    raw_df = load_raw_df(args.hf_token)
    feature_df = engineer_features(raw_df)
    
    if len(feature_df) < SEQ_LEN + 10:
        print("Insufficient data")
        return

    # Generate signals for both modes
    mode_signals = {}
    for mode in ["expanding", "shrinking"]:
        print(f"\nLoading {mode} model...")
        try:
            model, mean, std = load_model(mode, DEVICE, args.hf_token)
            signal = generate_signal(model, feature_df, mean, std, mode_name=mode)
            mode_signals[mode] = signal
        except Exception as e:
            print(f"Failed to generate signal for {mode}: {e}")
            mode_signals[mode] = None

    # Determine hero model and signal by comparing predicted returns across both models
    hero_mode = None
    hero_signal = None
    max_return = -float('inf')
    for mode, sig in mode_signals.items():
        if sig is None:
            continue
        ret = sig.get("predicted_return", -float('inf'))
        if ret > max_return:
            max_return = ret
            hero_mode = mode
            hero_signal = sig

    if hero_signal is None:
        print("No valid signals generated.")
        return

    print(f"\nHero mode (highest predicted return): {hero_mode.upper()} with predicted return {max_return*100:+.2f}%")

    # Load and update history
    history = load_history(args.hf_token)
    history = update_history(history, hero_signal, feature_df)

    # Save history locally and push to HF
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved → {HISTORY_PATH}")

    # Prepare output JSON
    output = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "best_historical_mode": historical_best_mode,
        "hero_mode": hero_mode,
        "signal": hero_signal,
        "mode_predictions": mode_signals,
        "historical_performance": results.get("historical_performance", {}),
        "mode_comparison": mode_metrics,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSignal saved → {OUTPUT_PATH}")

    # Push to HF
    if args.hf_token:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=args.hf_token)
            repo_id = "P2SAMAPA/etf-hrformer-model"
            
            # Upload latest.json
            api.upload_file(path_or_fileobj=OUTPUT_PATH,
                            path_in_repo=OUTPUT_PATH,
                            repo_id=repo_id, repo_type="model")
            print("Pushed latest.json to HF Hub.")
            
            # Upload history
            api.upload_file(path_or_fileobj=HISTORY_PATH,
                            path_in_repo=HISTORY_PATH,
                            repo_id=repo_id, repo_type="model")
            print("Pushed prediction_history.json to HF Hub.")
        except Exception as e:
            print(f"HF push failed: {e}")


if __name__ == "__main__":
    main()
