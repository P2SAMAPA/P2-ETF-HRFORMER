"""
train_equity.py
Walk-forward training for equity ETFs — 48-day return prediction.
Mirrors train.py exactly; uses data_utils_equity + 13-ETF model.
"""

import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils_equity import (
    load_raw_df, engineer_features, make_labels, get_feature_names,
    normalise, ETFDataset, TARGET_ETFS, SEQ_LEN, PRED_HORIZON
)
from hrformer import build_model
from torch.utils.data import DataLoader

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS      = 30
PATIENCE    = 10
LR          = 1e-3
BATCH_SIZE  = 32
FOLD_YEARS  = 1
MIN_TRAIN_YEARS = 3

NUM_ETFS    = len(TARGET_ETFS)   # 13


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def ranking_loss(pred_returns, true_returns, margin=0.1):
    pred_diff = pred_returns.unsqueeze(2) - pred_returns.unsqueeze(1)
    true_diff = true_returns.unsqueeze(2) - true_returns.unsqueeze(1)
    mask = (true_diff > margin).float()
    loss = torch.relu(0.1 - pred_diff) * mask
    return loss.sum() / (mask.sum() + 1e-8)


# ---------------------------------------------------------------------------
# Epoch runner
# ---------------------------------------------------------------------------

def run_epoch(model, loader, optimiser=None):
    training = optimiser is not None
    model.train() if training else model.eval()

    total_loss = total_mse = total_rank = 0.0
    n_batches = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)

            mse  = nn.functional.mse_loss(pred, y)
            rank = ranking_loss(pred, y)
            loss = mse + 0.5 * rank

            if training:
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()

            total_loss += loss.item()
            total_mse  += mse.item()
            total_rank += rank.item()
            n_batches  += 1

    return total_loss / n_batches, total_mse / n_batches, total_rank / n_batches


# ---------------------------------------------------------------------------
# Train on one split
# ---------------------------------------------------------------------------

def train_on_split(feature_df, label_df, train_idx, val_idx, model_path):
    feat_names = get_feature_names()

    train_data = np.concatenate([
        feature_df[t][feat_names].iloc[train_idx[0]:train_idx[-1] + SEQ_LEN].values
        for t in TARGET_ETFS
    ], axis=0)
    mean = train_data.mean(0)
    std  = train_data.std(0)
    std  = np.where(std == 0, 1.0, std)

    def make_loader(idx, shuffle):
        ds = ETFDataset(feature_df, label_df, idx, mean, std)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, drop_last=shuffle)

    train_loader = make_loader(train_idx, True)
    val_loader   = make_loader(val_idx,   False)

    # Build model sized for 13 equity ETFs
    model = build_model(num_etfs=NUM_ETFS, seq_len=SEQ_LEN, num_features=8).to(DEVICE)
    optimiser = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimiser, mode="min", factor=0.5, patience=5)

    best_loss, patience_ctr, best_state = float("inf"), 0, None

    for epoch in range(1, EPOCHS + 1):
        tr_loss, _, _ = run_epoch(model, train_loader, optimiser)
        va_loss, _, _ = run_epoch(model, val_loader)

        scheduler.step(va_loss)

        if va_loss < best_loss:
            best_loss = va_loss
            best_state = {"model": model.state_dict(), "mean": mean, "std": std,
                          "val_loss": va_loss}
            torch.save(best_state, model_path)
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"      Early stopping at epoch {epoch}")
                break

        if epoch % 5 == 0:
            print(f"      ep{epoch:02d} tr_loss={tr_loss:.4f} va_loss={va_loss:.4f}")

    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])
    print(f"    Best val_loss: {checkpoint['val_loss']:.4f}")
    return model, checkpoint["mean"], checkpoint["std"]


# ---------------------------------------------------------------------------
# Backtest one fold
# ---------------------------------------------------------------------------

def backtest_fold(model, feature_df, test_idx, mean, std, trading_cost=0.001):
    """Single ETF backtest: pick highest predicted 48-day return."""
    feat_names = get_feature_names()
    model.eval()

    portfolio_returns, picks_list = [], []

    with torch.no_grad():
        for t in test_idx[::PRED_HORIZON]:
            if t + SEQ_LEN + PRED_HORIZON > len(feature_df):
                continue

            x = np.stack([
                normalise(feature_df[tk][feat_names].iloc[t:t + SEQ_LEN].values, mean, std)
                for tk in TARGET_ETFS
            ], axis=0).astype(np.float32)

            x_tensor = torch.from_numpy(x).unsqueeze(0).to(DEVICE)
            pred_returns = model.predict_returns(x_tensor).cpu().numpy()[0]

            top_idx = int(np.argmax(pred_returns))
            top_etf = TARGET_ETFS[top_idx]
            picks_list.append([top_etf])

            i_today  = t + SEQ_LEN - 1
            i_future = min(t + SEQ_LEN + PRED_HORIZON - 1, len(feature_df) - 1)

            price_today  = feature_df[top_etf]["Close"].iloc[i_today]
            price_future = feature_df[top_etf]["Close"].iloc[i_future]
            raw_ret = (price_future - price_today) / price_today
            portfolio_ret = raw_ret - 2 * trading_cost

            if abs(portfolio_ret) > 0.5:
                portfolio_ret = 0.0

            portfolio_returns.append(portfolio_ret)

    equity = [1.0]
    for r in portfolio_returns:
        equity.append(equity[-1] * (1 + r))
    equity = np.array(equity)

    n_periods = len(portfolio_returns)
    periods_per_year = 252 / PRED_HORIZON

    ann_r  = (equity[-1] ** (periods_per_year / n_periods)) - 1 if n_periods > 0 else 0.0
    ann_v  = np.std(portfolio_returns) * np.sqrt(periods_per_year) if n_periods > 0 else 0.0
    sharpe = ann_r / (ann_v + 1e-8)
    dd     = (equity / np.maximum.accumulate(equity) - 1).min() if len(equity) > 0 else 0.0
    total_ret = max(min(equity[-1] - 1.0, 10.0), -0.99)

    return {
        "equity":  equity.tolist(),
        "returns": portfolio_returns,
        "picks":   picks_list,
        "summary": {
            "annualised_return": round(float(ann_r),  4),
            "annualised_vol":    round(float(ann_v),  4),
            "sharpe_ratio":      round(float(sharpe), 4),
            "max_drawdown":      round(float(dd),     4),
            "total_return":      round(float(total_ret), 4),
        },
    }


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------

def run_walk_forward(feature_df, label_df, mode, model_path):
    trading_days_per_year = 252
    fold_days = FOLD_YEARS * trading_days_per_year
    min_days  = MIN_TRAIN_YEARS * trading_days_per_year

    n = len(feature_df) - SEQ_LEN - PRED_HORIZON
    cutoffs, c = [], min_days
    while c + fold_days <= n:
        cutoffs.append(c)
        c += fold_days

    print(f"  Mode: {mode} | Folds: {len(cutoffs)}")
    fold_results = []
    final_model = final_mean = final_std = None

    for fi, cutoff in enumerate(cutoffs):
        test_end = min(cutoff + fold_days, n)
        if mode == "expanding":
            train_start = 0
        else:
            train_start = fi * fold_days

        train_len = cutoff - train_start
        val_start = train_start + int(train_len * 0.8)

        train_idx = np.arange(train_start, val_start)
        val_idx   = np.arange(val_start,   cutoff)
        test_idx  = np.arange(cutoff,      test_end)

        if len(train_idx) < 50 or len(val_idx) < 10 or len(test_idx) < 10:
            continue

        fold_label = f"{feature_df.index[train_start].year}–{feature_df.index[cutoff-1].year}"
        test_label = (f"{feature_df.index[cutoff].strftime('%Y-%m-%d')} → "
                      f"{feature_df.index[min(test_end + SEQ_LEN - 1, len(feature_df)-1)].strftime('%Y-%m-%d')}")
        print(f"  Fold {fi+1}/{len(cutoffs)}: train {fold_label} | test {test_label}")

        model, mean, std = train_on_split(feature_df, label_df, train_idx, val_idx, model_path)

        bt = backtest_fold(model, feature_df, test_idx, mean, std)
        bt["fold"]       = fi + 1
        bt["train_range"] = fold_label
        bt["test_range"]  = test_label
        fold_results.append(bt)

        print(f"    → ann_ret={bt['summary']['annualised_return']:.2%} "
              f"sharpe={bt['summary']['sharpe_ratio']:.2f}")

        final_model, final_mean, final_std = model, mean, std

    return fold_results, final_model, final_mean, final_std


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def aggregate_folds(fold_results):
    if not fold_results:
        return {}

    all_rets = []
    for fold in fold_results:
        rets = fold.get("returns", [])
        if rets:
            all_rets.extend(rets)

    if not all_rets:
        return {}

    all_rets = [r if abs(r) < 0.5 else 0.0 for r in all_rets]

    equity = [1.0]
    for r in all_rets:
        equity.append(equity[-1] * (1 + r))
    equity     = np.array(equity)
    rets_array = np.array(all_rets)

    n_periods      = len(all_rets)
    periods_per_year = 252 / PRED_HORIZON

    ann_r  = (equity[-1] ** (periods_per_year / n_periods)) - 1 if n_periods > 0 else 0.0
    ann_v  = rets_array.std() * np.sqrt(periods_per_year) if len(rets_array) > 0 else 0.0
    sharpe = ann_r / (ann_v + 1e-8)
    dd     = (equity / np.maximum.accumulate(equity) - 1).min() if len(equity) > 0 else 0.0
    total_ret = max(min(equity[-1] - 1.0, 10.0), -0.99)

    return {
        "equity":  equity.tolist(),
        "returns": all_rets,
        "summary": {
            "annualised_return": round(float(ann_r),  4),
            "annualised_vol":    round(float(ann_v),  4),
            "sharpe_ratio":      round(float(sharpe), 4),
            "max_drawdown":      round(float(dd),     4),
            "total_return":      round(float(total_ret), 4),
            "num_folds":         len(fold_results),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--mode", type=str, required=True,
                        choices=["expanding", "shrinking"])
    args = parser.parse_args()

    print(f"[EQUITY] Device: {DEVICE} | Mode: {args.mode}")
    print("Loading equity data...")

    raw_df     = load_raw_df(args.hf_token)
    feature_df = engineer_features(raw_df)
    label_df   = make_labels(feature_df)

    feature_df = feature_df.iloc[:-PRED_HORIZON]
    label_df   = label_df.iloc[:-PRED_HORIZON]

    print(f"  Data: {feature_df.index[0].date()} → {feature_df.index[-1].date()} "
          f"({len(feature_df)} rows, {NUM_ETFS} ETFs)")

    print(f"\n{'='*50}")
    print(f"Running EQUITY {args.mode.upper()} walk-forward...")
    t0 = time.time()

    model_path   = f"model_equity_{args.mode}.pt"
    fold_results, final_model, mean, std = run_walk_forward(
        feature_df, label_df, args.mode, model_path)

    elapsed = round(time.time() - t0, 1)
    agg     = aggregate_folds(fold_results)

    results = {"folds": fold_results, "aggregate": agg, "elapsed_s": elapsed}

    print(f"\n[EQUITY] {args.mode} complete in {elapsed}s")
    print(f"  Ann. return: {agg['summary']['annualised_return']:.2%}")
    print(f"  Sharpe:      {agg['summary']['sharpe_ratio']:.2f}")
    print(f"  Total return:{agg['summary']['total_return']:.2%}")

    output_file = f"walk_forward_results_equity_{args.mode}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {output_file}")

    checkpoint = {"model": final_model.state_dict(), "mean": mean, "std": std,
                  "mode": args.mode, "num_etfs": NUM_ETFS}
    torch.save(checkpoint, model_path)
    print(f"Model saved  → {model_path}")

    if args.hf_token:
        try:
            from huggingface_hub import HfApi
            api    = HfApi(token=args.hf_token)
            repo   = "P2SAMAPA/etf-hrformer-model"
            api.create_repo(repo, repo_type="model", exist_ok=True)
            for path in [model_path, output_file]:
                api.upload_file(path_or_fileobj=path, path_in_repo=path,
                                repo_id=repo, repo_type="model")
            print("Pushed equity artefacts to HF Hub.")
        except Exception as e:
            print(f"HF push failed: {e}")


if __name__ == "__main__":
    main()
