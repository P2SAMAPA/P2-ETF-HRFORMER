"""
train.py
Walk-forward training for P2-ETF-HRFORMER.
"""

import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import (
    load_raw_df, engineer_features, make_labels, get_feature_names,
    normalise, ETFDataset, TARGET_ETFS, SEQ_LEN, PRED_HORIZON
)
from hrformer import build_model
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 30           # Paper uses 30
PATIENCE = 10
LR = 5e-4
BATCH_SIZE = 64
FIXED_YEARS = 2
FOLD_YEARS = 1
MIN_TRAIN_YEARS = 3


def run_epoch(model, loader, criterion, optimiser=None):
    training = optimiser is not None
    model.train() if training else model.eval()
    
    total_loss, preds_all, labels_all, probs_all = 0.0, [], [], []
    
    with torch.enable_grad() if training else torch.no_grad():
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
            probs = torch.softmax(logits, dim=-1)
            preds_all.extend(logits.argmax(-1).cpu().numpy().flatten())
            labels_all.extend(y.cpu().numpy().flatten())
            probs_all.extend(probs[:, :, 1].detach().cpu().numpy().flatten())
    
    f1 = f1_score(labels_all, preds_all, average="macro", zero_division=0)
    acc = accuracy_score(labels_all, preds_all)
    
    try:
        auc = roc_auc_score(labels_all, probs_all)
    except:
        auc = 0.5
    
    return total_loss / len(loader.dataset), acc, f1, auc


def train_on_split(feature_df, label_df, train_idx, val_idx, model_path):
    feat_names = get_feature_names()
    
    # Compute normalization stats
    train_data = np.concatenate([
        feature_df[t][feat_names].iloc[train_idx[0]:train_idx[-1]+SEQ_LEN].values
        for t in TARGET_ETFS
    ], axis=0)
    mean = train_data.mean(0)
    std = train_data.std(0)
    std = np.where(std == 0, 1.0, std)

    def make_loader(idx, shuffle):
        ds = ETFDataset(feature_df, label_df, idx, mean, std)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, drop_last=shuffle)

    train_loader = make_loader(train_idx, True)
    val_loader = make_loader(val_idx, False)

    model = build_model().to(DEVICE)
    
    # Simple class weighting
    pos_weight = 1.0
    neg_weight = 1.0
    class_weights = torch.tensor([neg_weight, pos_weight]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimiser = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimiser, mode="max", factor=0.5, patience=5)

    best_f1, patience_ctr = -1, 0
    best_state = None
    
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, tr_f1, tr_auc = run_epoch(model, train_loader, criterion, optimiser)
        va_loss, va_acc, va_f1, va_auc = run_epoch(model, val_loader, criterion)
        
        scheduler.step(va_f1)
        
        if va_f1 > best_f1:
            best_f1 = va_f1
            best_state = {
                'model': model.state_dict(),
                'mean': mean,
                'std': std,
                'val_f1': va_f1,
                'val_auc': va_auc
            }
            torch.save(best_state, model_path)
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"      Early stopping at epoch {epoch}")
                break
        
        if epoch % 5 == 0:
            print(f"      ep{epoch:02d} tr_f1={tr_f1:.3f} va_f1={va_f1:.3f} va_auc={va_auc:.3f}")

    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model'])
    
    print(f"    Best: F1={checkpoint['val_f1']:.3f}, AUC={checkpoint['val_auc']:.3f}")
    return model, checkpoint['mean'], checkpoint['std'], best_f1


def backtest_fold(model, feature_df, test_idx, mean, std, trading_cost=0.0005, threshold=0.50):
    """
    Backtest with LOW threshold (0.50) to ensure trades happen.
    """
    feat_names = get_feature_names()
    model.eval()
    
    equity, rets, picks, probas = [1.0], [], [], []
    all_confidences = []
    
    with torch.no_grad():
        for t in test_idx:
            x = np.stack([
                normalise(feature_df[tk][feat_names].iloc[t:t+SEQ_LEN].values, mean, std)
                for tk in TARGET_ETFS
            ], axis=0).astype(np.float32)
            
            x_tensor = torch.from_numpy(x).unsqueeze(0).to(DEVICE)
            proba = model.predict_proba(x_tensor).cpu().numpy()[0]
            
            probas.append(proba.tolist())
            pick_i = int(np.argmax(proba))
            pick = TARGET_ETFS[pick_i]
            picks.append(pick)

            i_today = t + SEQ_LEN - 1
            i_next = t + SEQ_LEN
            if i_next >= len(feature_df):
                rets.append(0.0)
                equity.append(equity[-1])
                continue

            price_today = feature_df[pick]["Close"].iloc[i_today]
            price_next = feature_df[pick]["Close"].iloc[i_next]
            raw_ret = (price_next - price_today) / (price_today + 1e-8)
            
            confidence = proba[pick_i]
            all_confidences.append(confidence)
            
            # CRITICAL: Trade if confidence > threshold (0.50 = always trade best pick)
            if confidence > threshold:
                ret = raw_ret - 2 * trading_cost
            else:
                ret = 0.0
                
            rets.append(float(ret))
            equity.append(equity[-1] * (1 + ret))

    # Print stats
    if all_confidences:
        conf_array = np.array(all_confidences)
        print(f"    Conf: mean={conf_array.mean():.3f}, max={conf_array.max():.3f}, "
              f"trades={(conf_array > threshold).sum()}/{len(conf_array)}")

    rets = np.array(rets)
    equity = np.array(equity)
    
    total_ret = equity[-1] - 1.0
    ann_r = (equity[-1] ** (252 / max(len(rets), 1))) - 1
    ann_v = rets.std() * np.sqrt(252)
    sharpe = ann_r / (ann_v + 1e-8)
    dd = (equity / np.maximum.accumulate(equity) - 1).min()
    
    dates = feature_df.index[[t + SEQ_LEN for t in test_idx
                              if t + SEQ_LEN < len(feature_df)]].strftime("%Y-%m-%d").tolist()
    
    return {
        "equity": equity[1:].tolist(),
        "dates": dates[:len(equity)-1],
        "picks": picks,
        "probas": probas,
        "summary": {
            "annualised_return": round(float(ann_r), 4),
            "annualised_vol": round(float(ann_v), 4),
            "sharpe_ratio": round(float(sharpe), 4),
            "max_drawdown": round(float(dd), 4),
            "total_return": round(float(total_ret), 4),
        }
    }


def run_walk_forward(feature_df, label_df, mode, model_path):
    trading_days_per_year = 252
    fold_days = FOLD_YEARS * trading_days_per_year
    fixed_days = FIXED_YEARS * trading_days_per_year
    min_days = MIN_TRAIN_YEARS * trading_days_per_year

    n = len(feature_df) - SEQ_LEN - PRED_HORIZON
    cutoffs = []
    c = min_days
    while c + fold_days <= n:
        cutoffs.append(c)
        c += fold_days

    print(f"  Mode: {mode} | Folds: {len(cutoffs)}")
    fold_results = []
    final_model, final_mean, final_std = None, None, None

    for fi, cutoff in enumerate(cutoffs):
        test_end = min(cutoff + fold_days, n)
        if mode == "expanding":
            train_start = 0
        else:
            train_start = max(0, cutoff - fixed_days)

        train_len = cutoff - train_start
        val_start = train_start + int(train_len * 0.8)

        train_idx = np.arange(train_start, val_start)
        val_idx = np.arange(val_start, cutoff)
        test_idx = np.arange(cutoff, test_end)

        if len(train_idx) < 50 or len(val_idx) < 10 or len(test_idx) < 10:
            continue

        fold_label = f"{feature_df.index[train_start].year}–{feature_df.index[cutoff-1].year}"
        test_label = (f"{feature_df.index[cutoff].strftime('%Y-%m-%d')} → "
                      f"{feature_df.index[min(test_end+SEQ_LEN-1, len(feature_df)-1)].strftime('%Y-%m-%d')}")
        print(f"  Fold {fi+1}/{len(cutoffs)}: train {fold_label} | test {test_label}")

        model, mean, std, best_val_f1 = train_on_split(
            feature_df, label_df, train_idx, val_idx, model_path)

        # Backtest with threshold=0.50 (always trade)
        bt = backtest_fold(model, feature_df, test_idx, mean, std, threshold=0.50)
        bt["fold"] = fi + 1
        bt["train_range"] = fold_label
        bt["test_range"] = test_label
        bt["best_val_f1"] = round(best_val_f1, 4)
        fold_results.append(bt)

        print(f"    → ann_ret={bt['summary']['annualised_return']:.2%} "
              f"sharpe={bt['summary']['sharpe_ratio']:.2f}")

        final_model, final_mean, final_std = model, mean, std

    return fold_results, final_model, final_mean, final_std


def aggregate_folds(fold_results):
    if not fold_results:
        return {}
    
    chained = [1.0]
    all_dates, all_picks = [], []
    
    for fold in fold_results:
        eq = fold["equity"]
        if not eq:
            continue
            
        if len(chained) == 1:
            for v in eq:
                chained.append(v)
        else:
            scale = chained[-1]
            for v in eq:
                chained.append(v * scale / eq[0] if eq[0] != 0 else scale)
                
        all_dates.extend(fold["dates"])
        all_picks.extend(fold["picks"])

    chained = np.array(chained[1:])
    n = min(len(chained), len(all_dates))
    chained = chained[:n]
    all_dates = all_dates[:n]
    all_picks = all_picks[:n]

    rets = np.diff(chained) / chained[:-1]
    
    ann_r = (chained[-1] ** (252 / max(len(rets), 1))) - 1 if len(rets) > 0 else 0.0
    ann_v = rets.std() * np.sqrt(252) if len(rets) > 0 else 0.0
    sharpe = ann_r / (ann_v + 1e-8)
    dd = (chained / np.maximum.accumulate(chained) - 1).min() if len(chained) > 0 else 0.0

    return {
        "equity": chained.tolist(),
        "dates": all_dates,
        "picks": all_picks,
        "summary": {
            "annualised_return": round(float(ann_r), 4),
            "annualised_vol": round(float(ann_v), 4),
            "sharpe_ratio": round(float(sharpe), 4),
            "max_drawdown": round(float(dd), 4),
            "total_return": round(float(chained[-1] - 1), 4) if len(chained) > 0 else 0.0,
            "num_folds": len(fold_results),
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
    
    raw_df = load_raw_df(args.hf_token)
    feature_df = engineer_features(raw_df)
    label_df = make_labels(feature_df)
    feature_df = feature_df.iloc[:-PRED_HORIZON]
    label_df = label_df.iloc[:-PRED_HORIZON]
    
    print(f"  Data: {feature_df.index[0].date()} → {feature_df.index[-1].date()} "
          f"({len(feature_df)} rows)")

    results = {}
    modes_to_run = ["expanding", "fixed"] if args.mode == "both" else [args.mode]

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
            "folds": fold_results,
            "aggregate": agg,
            "elapsed_s": elapsed,
        }
        
        print(f"\n{mode} complete in {elapsed}s")
        print(f"  Ann. return: {agg['summary']['annualised_return']:.2%}")
        print(f"  Sharpe: {agg['summary']['sharpe_ratio']:.2f}")
        print(f"  Total return: {agg['summary']['total_return']:.2%}")

    # Determine best mode
    if len(results) == 2:
        best_mode = max(results, key=lambda m: results[m]["aggregate"]["summary"]["annualised_return"])
    else:
        best_mode = modes_to_run[0]
    
    results["best_mode"] = best_mode
    print(f"\nBest mode: {best_mode.upper()}")

    # Save results
    with open("walk_forward_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved → walk_forward_results.json")

    # Save final model
    checkpoint = {
        'model': final_model.state_dict(),
        'mean': final_mean,
        'std': final_std,
        'mode': best_mode,
    }
    torch.save(checkpoint, f"model_{best_mode}.pt")
    print(f"Model saved → model_{best_mode}.pt")

    # Push to HF Hub
    if args.hf_token:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=args.hf_token)
            repo = "P2SAMAPA/etf-hrformer-model"
            api.create_repo(repo, repo_type="model", exist_ok=True)
            
            for mode in modes_to_run:
                api.upload_file(path_or_fileobj=f"model_{mode}.pt",
                                path_in_repo=f"model_{mode}.pt",
                                repo_id=repo, repo_type="model")
            api.upload_file(path_or_fileobj="walk_forward_results.json",
                            path_in_repo="walk_forward_results.json",
                            repo_id=repo, repo_type="model")
            print("Pushed to HF Hub.")
        except Exception as e:
            print(f"HF push failed: {e}")


if __name__ == "__main__":
    main()
