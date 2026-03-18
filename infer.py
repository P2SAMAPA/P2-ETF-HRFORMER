"""
infer.py
Generate 48-day ahead signal using trained model.
Selects SINGLE ETF with highest predicted return.
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


def generate_signal(model, feature_df, mean, std):
    """Generate 48-day ahead signal for SINGLE ETF."""
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
        
        print("  Predicted 48-day returns:")
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

    print(f"\n  Selected: {top_etf} (predicted: {top_return*100:+.2f}%)")
    print(f"  Hold: {next_trade_date} → {hold_until}")

    return {
        "signal_date": next_trade_date,
        "hold_until": hold_until,
        "data_date": last_data_date.strftime("%Y-%m-%d"),
        "recommended_etf": top_etf,  # SINGLE ETF - only the best one
        "predicted_return": round(top_return, 4),
        "predicted_returns": {ticker: round(float(pred_returns[i]), 4) 
                             for i, ticker in enumerate(TARGET_ETFS)},
        "strategy": "buy_hold_48d_single",
    }


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
    
    # Determine best mode using SHARPE RATIO (more robust than just return)
    # or composite score: return / (volatility + |drawdown|)
    best_mode = None
    best_score = -float('inf')
    
    print("\n  Comparing modes:")
    for mode in ["expanding", "shrinking"]:
        if mode in mode_metrics:
            m = mode_metrics[mode]
            ann_ret = m.get("annualised_return", -999)
            sharpe = m.get("sharpe_ratio", -999)
            
            # Use Sharpe ratio as primary selector, fallback to return if Sharpe invalid
            if sharpe != -999 and not np.isnan(sharpe):
                score = sharpe
                metric_used = "sharpe"
            else:
                score = ann_ret
                metric_used = "return"
            
            print(f"    {mode}: score={score:.4f} (using {metric_used})")
            
            if score > best_score:
                best_score = score
                best_mode = mode
    
    # Validate best_mode has reasonable metrics
    if best_mode and best_mode in mode_metrics:
        perf = mode_metrics[best_mode]
        # Sanity check: if max_drawdown is -1.0 or total_return is absurd, use other mode
        if perf.get("max_drawdown") == -1.0 or perf.get("total_return", 0) > 1e10:
            print(f"\n  WARNING: {best_mode} has corrupt metrics, switching to other mode")
            other_mode = "shrinking" if best_mode == "expanding" else "expanding"
            if other_mode in mode_metrics:
                best_mode = other_mode
                perf = mode_metrics[best_mode]
    
    if best_mode:
        combined["best_mode"] = best_mode
        combined["performance"] = mode_metrics[best_mode]
        combined["all_metrics"] = mode_metrics  # Include both for comparison
        print(f"\n  Selected best mode: {best_mode.upper()}")
        print(f"  Performance: {combined['performance']}")
    
    return combined, best_mode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    # Load combined results
    results, best_mode = load_combined_results()
    
    if not best_mode:
        print("No results found. Run training first.")
        return

    print(f"\nBest mode: {best_mode.upper()}")
    print(f"Performance metrics: {results.get('performance', {})}")

    print("\nLoading data...")
    raw_df = load_raw_df(args.hf_token)
    feature_df = engineer_features(raw_df)
    
    if len(feature_df) < SEQ_LEN + 10:
        print("Insufficient data")
        return

    print("Loading model...")
    model_path = f"model_{best_mode}.pt"
    
    # Try local first, then HF
    try:
        # FIX: Add weights_only=False for PyTorch 2.6+ compatibility
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    except FileNotFoundError:
        # Download from HF
        from huggingface_hub import hf_hub_download
        local_path = hf_hub_download(
            repo_id="P2SAMAPA/etf-hrformer-model",
            filename=model_path,
            repo_type="model",
            token=args.hf_token
        )
        # FIX: Add weights_only=False here too
        checkpoint = torch.load(local_path, map_location=DEVICE, weights_only=False)
    
    model = build_model().to(DEVICE)
    model.load_state_dict(checkpoint['model'])
    
    mean = checkpoint['mean']
    std = checkpoint['std']
    std = np.where(std == 0, 1.0, std)

    print("Generating signal...")
    signal = generate_signal(model, feature_df, mean, std)

    output = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "best_mode": best_mode,
        "signal": signal,
        "performance": results.get("performance", {}),
        "mode_comparison": results.get("all_metrics", {}),  # Include both modes for reference
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSignal saved → {OUTPUT_PATH}")

    # Push to HF
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
