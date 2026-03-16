"""
train.py
Walk-forward training for P2-ETF-HRFORMER.

Modes:
  single    — original single 70/20/10 split (kept for reference)
  expanding — expanding window: train on all data up to fold cutoff
  fixed     — fixed 2-year window: train on most recent 2 years before fold cutoff

Each fold trains on [train_start:cutoff], validates on last 20% of that,
tests on [cutoff:cutoff+1year]. Folds roll forward 1 year at a time.

Outputs:
  model_expanding.pt   / model_fixed.pt      — best model per mode
  metrics_expanding.json / metrics_fixed.json — per-fold + aggregate metrics
  walk_forward_results.json                  — combined results for infer.py + UI
"""

import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import (
    load_raw_df, engineer_features, make_labels, get_feature_names,
    normalise, ETFDataset, TARGET_ETFS, SEQ_LEN, PRED_HORIZON
)
from hrformer import build_model
from torch.utils.data import DataLoader

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS       = 80
PATIENCE     = 10
LR           = 5e-4
BATCH_SIZE   = 64
FIXED_YEARS  = 2      # train window for fixed mode
FOLD_YEARS   = 1      # test period per fold
MIN_TRAIN_YEARS = 3   # expanding: minimum years before first fold


def run_epoch(model, loader, criterion, optimiser=None):
    training = optimiser is not None
    model.train() if training else model.eval()
    total_loss, preds_all, labels_all = 0.0, [], []
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            B, M, _ = logits.shape
            loss = criterion(logits.view(B*M, 2), y.view(B*M))
            if training:
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
            total_loss += loss.item() * B
            preds_all.extend(logits.argmax(-1).cpu().numpy().flatten())
            labels_all.extend(y.cpu().numpy().flatten())
    f1  = f1_score(labels_all, preds_all, average="macro", zero_division=0)
    acc = accuracy_score(labels_all, preds_all)
    return total_loss / (len(loader.dataset) + 1e-8), acc, f1


def train_on_split(feature_df, label_df, train_idx, val_idx, model_path):
    feat_names = get_feature_names()
    train_data = np.concatenate(
        [feature_df[t][feat_names].iloc[train_idx[0]: train_idx[-1]+SEQ_LEN+1].values
         for t in TARGET_ETFS], axis=0)
    mean = train_data.mean(0)
    std  = train_data.std(0)

    def make_loader(idx, shuffle):
        ds = ETFDataset(feature_df, label_df, idx, mean, std)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, drop_last=shuffle)

    train_loader = make_loader(train_idx, True)
    val_loader   = make_loader(val_idx,   False)

    model     = build_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimiser = Adam(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = ReduceLROnPlateau(optimiser, mode="max", factor=0.5, patience=5)

    best_f1, patience_ctr = -1, 0
    for epoch in range(1, EPOCHS+1):
        _, _, tr_f1 = run_epoch(model, train_loader, criterion, optimiser)
        _, _, va_f1 = run_epoch(model, val_loader,   criterion)
        scheduler.step(va_f1)
        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save(model.state_dict(), model_path)
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break
        if epoch % 5 == 0:
            print(f"      ep{epoch:03d} tr_f1={tr_f1:.3f} va_f1={va_f1:.3f} best={best_f1:.3f}")

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    return model, mean, std, best_f1


def backtest_fold(model, feature_df, test_idx, mean, std, trading_cost=0.0005, threshold=0.65):
    feat_names = get_feature_names()
    model.eval()
    equity, rets, picks, probas = [1.0], [], [], []
    with torch.no_grad():
        for t in test_idx:
            x = np.stack([normalise(
                feature_df[tk][feat_names].iloc[t:t+SEQ_LEN].values, mean, std)
                for tk in TARGET_ETFS], axis=0).astype(np.float32)
            proba = model.predict_proba(
                torch.from_numpy(x).unsqueeze(0).to(DEVICE)).cpu().numpy()[0]
            probas.append(proba.tolist())
            pick_i = int(np.argmax(proba))
            pick   = TARGET_ETFS[pick_i]
            picks.append(pick)

            i_today = t + SEQ_LEN - 1
            i_next  = t + SEQ_LEN
            if i_next >= len(feature_df):
                rets.append(0.0); equity.append(equity[-1]); continue

            raw_ret = ((feature_df[pick]["Close"].iloc[i_next] -
                        feature_df[pick]["Close"].iloc[i_today]) /
                       (feature_df[pick]["Close"].iloc[i_today] + 1e-8))
            ret = (raw_ret - 2*trading_cost) if proba[pick_i] > threshold else 0.0
            rets.append(float(ret))
            equity.append(equity[-1] * (1+ret))

    rets   = np.array(rets)
    equity = np.array(equity)
    ann_r  = float((equity[-1] ** (252/max(len(rets),1))) - 1)
    ann_v  = float(rets.std() * np.sqrt(252))
    sharpe = float(ann_r / (ann_v + 1e-8))
    dd     = float((equity / np.maximum.accumulate(equity) - 1).min())
    dates  = feature_df.index[[t+SEQ_LEN for t in test_idx
                                if t+SEQ_LEN < len(feature_df)]].strftime("%Y-%m-%d").tolist()
    n = len(dates)
    return {
        "equity":   equity[1:n+1].tolist(),
        "dates":    dates,
        "picks":    picks[:n],
        "probas":   probas[:n],
        "summary":  {
            "annualised_return": round(ann_r, 4),
            "annualised_vol":    round(ann_v, 4),
            "sharpe_ratio":      round(sharpe, 4),
            "max_drawdown":      round(dd, 4),
            "total_return":      round(float(equity[-1]-1), 4),
        }
    }


def run_walk_forward(feature_df, label_df, mode, model_path):
    """
    Build folds and train/test sequentially.
    Returns list of fold results + the final trained model + mean/std.
    """
    trading_days_per_year = 252
    fold_days  = FOLD_YEARS  * trading_days_per_year
    fixed_days = FIXED_YEARS * trading_days_per_year
    min_days   = MIN_TRAIN_YEARS * trading_days_per_year

    n = len(feature_df) - SEQ_LEN - PRED_HORIZON

    # Build fold cutoffs: first cutoff at min_train_years, then every fold_days
    cutoffs = []
    c = min_days
    while c + fold_days <= n:
        cutoffs.append(c)
        c += fold_days

    print(f"  Mode: {mode} | Folds: {len(cutoffs)}")
    fold_results = []
    final_model  = None
    final_mean   = None
    final_std    = None

    for fi, cutoff in enumerate(cutoffs):
        test_end = min(cutoff + fold_days, n)
        if mode == "expanding":
            train_start = 0
        else:  # fixed
            train_start = max(0, cutoff - fixed_days)

        # 80/20 split of train window for val
        train_len = cutoff - train_start
        val_start = train_start + int(train_len * 0.8)

        train_idx = np.arange(train_start, val_start)
        val_idx   = np.arange(val_start,   cutoff)
        test_idx  = np.arange(cutoff,      test_end)

        if len(train_idx) < 50 or len(val_idx) < 10 or len(test_idx) < 10:
            continue

        fold_label = (f"{feature_df.index[train_start].year}–"
                      f"{feature_df.index[cutoff-1].year}")
        test_label = (f"{feature_df.index[cutoff].strftime('%Y-%m-%d')} → "
                      f"{feature_df.index[min(test_end+SEQ_LEN-1, len(feature_df)-1)].strftime('%Y-%m-%d')}")
        print(f"  Fold {fi+1}/{len(cutoffs)}: train {fold_label} | test {test_label}")

        model, mean, std, best_val_f1 = train_on_split(
            feature_df, label_df, train_idx, val_idx, model_path)

        bt = backtest_fold(model, feature_df, test_idx, mean, std)
        bt["fold"]       = fi + 1
        bt["train_range"] = fold_label
        bt["test_range"]  = test_label
        bt["best_val_f1"] = round(best_val_f1, 4)
        fold_results.append(bt)

        print(f"    → ann_return={bt['summary']['annualised_return']:.1%} "
              f"sharpe={bt['summary']['sharpe_ratio']:.2f} "
              f"val_f1={best_val_f1:.3f}")

        # Keep track of the most recent model for live signals
        final_model = model
        final_mean  = mean
        final_std   = std

    return fold_results, final_model, final_mean, final_std


def aggregate_folds(fold_results):
    """Concatenate all fold equity curves and compute aggregate metrics."""
    if not fold_results:
        return {}
    all_equity = [1.0]
    all_dates, all_picks = [], []
    for fold in fold_results:
        n = len(fold["dates"])
        for r in fold["equity"][:n]:
            all_equity.append(all_equity[-1] * (1 + (r - all_equity[-1]) /
                              max(all_equity[-1], 1e-8) if False else
                              (r / (all_equity[-1] + 1e-8) - 1 + 1)))
        # simpler: just chain the equity directly
        pass

    # Proper chaining: scale each fold's equity to start where previous ended
    chained = [1.0]
    for fold in fold_results:
        eq = fold["equity"]
        if not eq: continue
        scale = chained[-1]
        for v in eq:
            chained.append(v * scale / (fold["equity"][0] if fold["equity"][0] != 0 else 1))
    chained = np.array(chained[1:])

    all_dates = [d for f in fold_results for d in f["dates"]]
    all_picks = [p for f in fold_results for p in f["picks"]]

    rets   = np.diff(np.concatenate([[1.0], chained])) / (np.concatenate([[1.0], chained[:-1]]) + 1e-8)
    ann_r  = float((chained[-1] ** (252/max(len(rets),1))) - 1)
    ann_v  = float(rets.std() * np.sqrt(252))
    sharpe = float(ann_r / (ann_v + 1e-8))
    dd     = float((chained / np.maximum.accumulate(chained) - 1).min())

    return {
        "equity":       chained.tolist(),
        "dates":        all_dates[:len(chained)],
        "picks":        all_picks[:len(chained)],
        "summary": {
            "annualised_return": round(ann_r, 4),
            "annualised_vol":    round(ann_v, 4),
            "sharpe_ratio":      round(sharpe, 4),
            "max_drawdown":      round(dd, 4),
            "total_return":      round(float(chained[-1]-1), 4),
            "num_folds":         len(fold_results),
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--mode", type=str, default="both",
                        choices=["single", "expanding", "fixed", "both"])
    args = parser.parse_args()

    print(f"Device: {DEVICE} | Mode: {args.mode}")
    print("Loading data...")
    from data_utils import load_raw_df, engineer_features, make_labels
    raw_df     = load_raw_df(args.hf_token)
    feature_df = engineer_features(raw_df)
    label_df   = make_labels(feature_df)
    feature_df = feature_df.iloc[:-PRED_HORIZON]
    label_df   = label_df.iloc[:-PRED_HORIZON]
    print(f"  Data: {feature_df.index[0].date()} → {feature_df.index[-1].date()} "
          f"({len(feature_df)} rows)")

    results = {}
    modes_to_run = ["expanding", "fixed"] if args.mode == "both" else \
                   [args.mode] if args.mode != "single" else ["single"]

    for mode in modes_to_run:
        print(f"\n{'='*50}")
        print(f"Running {mode.upper()} walk-forward...")
        t0 = time.time()
        model_path = f"model_{mode}.pt"
        fold_results, final_model, mean, std = run_walk_forward(
            feature_df, label_df, mode, model_path)
        elapsed = round(time.time() - t0, 1)
        agg = aggregate_folds(fold_results)
        results[mode] = {
            "folds":     fold_results,
            "aggregate": agg,
            "elapsed_s": elapsed,
        }
        print(f"\n{mode} complete in {elapsed}s")
        print(f"  Aggregate ann_return : {agg['summary']['annualised_return']:.1%}")
        print(f"  Aggregate sharpe     : {agg['summary']['sharpe_ratio']:.2f}")
        print(f"  Aggregate total_ret  : {agg['summary']['total_return']:.1%}")

    # Determine best mode by annualised return
    if len(results) == 2:
        best_mode = max(results, key=lambda m: results[m]["aggregate"]["summary"]["annualised_return"])
        print(f"\nBest mode by ann. return: {best_mode.upper()}")
    else:
        best_mode = modes_to_run[0]

    results["best_mode"] = best_mode

    # Save combined results
    with open("walk_forward_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved → walk_forward_results.json")

    # Push to HF Hub
    if args.hf_token:
        try:
            from huggingface_hub import HfApi
            api  = HfApi(token=args.hf_token)
            repo = "P2SAMAPA/etf-hrformer-model"
            try: api.create_repo(repo, repo_type="model", exist_ok=True)
            except: pass
            for mode in modes_to_run:
                api.upload_file(path_or_fileobj=f"model_{mode}.pt",
                                path_in_repo=f"model_{mode}.pt",
                                repo_id=repo, repo_type="model")
            api.upload_file(path_or_fileobj="walk_forward_results.json",
                            path_in_repo="walk_forward_results.json",
                            repo_id=repo, repo_type="model")
            print("Pushed to HF Hub.")
        except Exception as e:
            print(f"HF push failed (non-fatal): {e}")


if __name__ == "__main__":
    main()
