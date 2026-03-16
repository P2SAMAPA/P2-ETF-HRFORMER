"""
train.py
Daily retraining entry point for P2-ETF-HRFORMER.

Workflow:
  1. Load + preprocess data from HuggingFace
  2. Train HRformer with early stopping on validation F1
  3. Evaluate on held-out test set
  4. Save model weights + training metrics JSON
  5. Push artifacts back to HuggingFace Hub (model repo)
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import build_dataloaders, TARGET_ETFS, SEQ_LEN
from hrformer import build_model

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS       = 40
PATIENCE     = 7
LR           = 1e-3
BATCH_SIZE   = 32
MODEL_PATH   = "model.pt"
METRICS_PATH = "train_metrics.json"


# ── Training helpers ──────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimiser=None):
    train = optimiser is not None
    model.train() if train else model.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y   = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)                            # (B, M, 2)
            B, M, _ = logits.shape
            loss   = criterion(logits.view(B * M, 2), y.view(B * M))

            if train:
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()

            total_loss += loss.item() * B
            preds = logits.argmax(dim=-1).cpu().numpy().flatten()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy().flatten())

    avg_loss = total_loss / (len(loader.dataset) + 1e-8)
    f1       = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    acc      = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, f1


def evaluate_full(model, loader):
    """Per-ETF + aggregate metrics on any loader."""
    model.eval()
    all_preds_per_etf  = {t: [] for t in TARGET_ETFS}
    all_labels_per_etf = {t: [] for t in TARGET_ETFS}

    with torch.no_grad():
        for x, y in loader:
            x, y   = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            preds  = logits.argmax(dim=-1)
            for i, ticker in enumerate(TARGET_ETFS):
                all_preds_per_etf[ticker].extend(preds[:, i].cpu().numpy())
                all_labels_per_etf[ticker].extend(y[:, i].cpu().numpy())

    results = {}
    for ticker in TARGET_ETFS:
        p = all_preds_per_etf[ticker]
        l = all_labels_per_etf[ticker]
        results[ticker] = {
            "accuracy":  round(accuracy_score(l, p), 4),
            "precision": round(precision_score(l, p, zero_division=0), 4),
            "recall":    round(recall_score(l, p, zero_division=0), 4),
            "f1":        round(f1_score(l, p, zero_division=0), 4),
        }

    all_p = [v for vals in all_preds_per_etf.values() for v in vals]
    all_l = [v for vals in all_labels_per_etf.values() for v in vals]
    results["aggregate"] = {
        "accuracy":  round(accuracy_score(all_l, all_p), 4),
        "precision": round(precision_score(all_l, all_p, zero_division=0), 4),
        "recall":    round(recall_score(all_l, all_p, zero_division=0), 4),
        "f1":        round(f1_score(all_l, all_p, zero_division=0), 4),
    }
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print("Loading data...")
    train_loader, val_loader, test_loader, meta = build_dataloaders(
        hf_token=args.hf_token, batch_size=BATCH_SIZE
    )
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val   batches : {len(val_loader)}")
    print(f"  Test  batches : {len(test_loader)}")

    model     = build_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimiser = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimiser, mode="max", factor=0.5,
                                  patience=3, verbose=True)

    best_val_f1  = -1.0
    patience_ctr = 0
    history      = []

    print("\nTraining...")
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, tr_f1 = run_epoch(model, train_loader, criterion, optimiser)
        va_loss, va_acc, va_f1 = run_epoch(model, val_loader,   criterion)
        scheduler.step(va_f1)

        history.append({
            "epoch":      epoch,
            "train_loss": round(tr_loss, 4), "train_acc": round(tr_acc, 4),
            "train_f1":   round(tr_f1,   4),
            "val_loss":   round(va_loss, 4), "val_acc":   round(va_acc, 4),
            "val_f1":     round(va_f1,   4),
        })
        print(f"  Epoch {epoch:02d} | "
              f"tr_loss={tr_loss:.4f} tr_f1={tr_f1:.4f} | "
              f"va_loss={va_loss:.4f} va_f1={va_f1:.4f}")

        if va_f1 > best_val_f1:
            best_val_f1  = va_f1
            torch.save(model.state_dict(), MODEL_PATH)
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    elapsed = round(time.time() - t0, 1)
    print(f"\nTraining complete in {elapsed}s | Best val F1: {best_val_f1:.4f}")

    # Evaluate best model on test set
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    test_metrics = evaluate_full(model, test_loader)
    print("\nTest metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    # Save metrics to root
    payload = {
        "train_duration_s": elapsed,
        "best_val_f1":      round(best_val_f1, 4),
        "epochs_run":       len(history),
        "history":          history,
        "test_metrics":     test_metrics,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Metrics saved → {METRICS_PATH}")

    # Push model + metrics to HF Hub
    if args.hf_token:
        try:
            from huggingface_hub import HfApi
            api  = HfApi(token=args.hf_token)
            repo = "P2SAMAPA/etf-hrformer-model"
            try:
                api.create_repo(repo, repo_type="model", exist_ok=True)
            except Exception:
                pass
            api.upload_file(path_or_fileobj=MODEL_PATH,
                            path_in_repo="model.pt",
                            repo_id=repo, repo_type="model")
            api.upload_file(path_or_fileobj=METRICS_PATH,
                            path_in_repo="train_metrics.json",
                            repo_id=repo, repo_type="model")
            print("Model + metrics pushed to HF Hub.")
        except Exception as e:
            print(f"HF push failed (non-fatal): {e}")


if __name__ == "__main__":
    main()
