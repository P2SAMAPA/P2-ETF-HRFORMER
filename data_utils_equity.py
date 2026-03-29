"""
data_utils_equity.py
Load OHLCV data for 13 equity ETFs from HuggingFace.
Mirrors data_utils.py exactly — only ETF list, HF path, and module name change.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

TARGET_ETFS = [
    "SPY",   # S&P 500
    "QQQ",   # NASDAQ 100
    "XLK",   # Technology
    "XLF",   # Financials
    "XLE",   # Energy
    "XLV",   # Health Care
    "XLI",   # Industrials
    "XLY",   # Consumer Disc
    "XLP",   # Consumer Staples
    "XLU",   # Utilities
    "XME",   # Metal and Mining
    "GDX",   # Gold Miners
    "IWM",   # Russell 2000 Small Cap
]

FEATURES  = ["Open", "High", "Low", "Close", "Volume"]
HF_REPO   = "P2SAMAPA/etf-dlinear-cross-data"
HF_DATA_FILE = "equity/ohlcv_equity.parquet"   # <-- equity path in the same HF dataset repo

SEQ_LEN      = 48   # Input: past 48 trading days
PRED_HORIZON = 48   # Predict 48-day forward return
START_DATE   = "2008-01-01"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_raw_df(hf_token=None):
    """Download equity parquet from HuggingFace."""
    from huggingface_hub import hf_hub_download
    import ast

    local_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=HF_DATA_FILE,
        repo_type="dataset",
        token=hf_token,
        force_download=True,
    )
    df = pd.read_parquet(local_path)

    # Normalise index
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    elif df.index.name == "Date":
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Normalise columns to MultiIndex
    if not isinstance(df.columns, pd.MultiIndex):
        df.columns = pd.MultiIndex.from_tuples(
            [ast.literal_eval(c) if c.startswith("(") else (c, "")
             for c in df.columns]
        )

    available = [t for t in TARGET_ETFS if t in df.columns.get_level_values(0)]
    missing   = set(TARGET_ETFS) - set(available)
    if missing:
        raise ValueError(f"Equity ETFs not found in parquet: {missing}")

    df = df[available].loc[START_DATE:].dropna(how="all")
    return df


# ---------------------------------------------------------------------------
# Feature engineering  (identical logic to FI module)
# ---------------------------------------------------------------------------

def engineer_features(df):
    """Add derived features for each equity ETF."""
    frames = []
    for ticker in TARGET_ETFS:
        if ticker not in df.columns.get_level_values(0):
            continue
        t = df[ticker].copy()
        t["daily_return"] = t["Close"].pct_change()
        t["log_volume"]   = np.log1p(t["Volume"])
        t["hl_range"]     = (t["High"] - t["Low"]) / t["Close"]
        frames.append(t)

    result = pd.concat(frames, axis=1, keys=TARGET_ETFS)
    result = result.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna(how="all")

    if len(result) < SEQ_LEN + PRED_HORIZON + 10:
        raise ValueError(f"Insufficient equity data: {len(result)} rows")

    return result


def get_feature_names():
    return ["Open", "High", "Low", "Close", "Volume",
            "daily_return", "log_volume", "hl_range"]


def normalise(arr, mean, std):
    std = np.where(np.isnan(std), 1.0, std)
    std = np.where(std == 0,      1.0, std)
    return (arr - mean) / std


# ---------------------------------------------------------------------------
# Labels  (48-day forward return — same as FI)
# ---------------------------------------------------------------------------

def make_labels(df):
    """48-day forward return as regression target."""
    labels = {}
    for ticker in TARGET_ETFS:
        if ticker not in df.columns.get_level_values(0):
            continue
        close = df[ticker]["Close"]
        future_return = (close.shift(-PRED_HORIZON) - close) / close
        labels[ticker] = future_return

    label_df = pd.DataFrame(labels, index=df.index)

    for ticker in labels.keys():
        returns = label_df[ticker].dropna()
        print(f"  {ticker}: mean={returns.mean():.4f}, std={returns.std():.4f}")

    return label_df


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ETFDataset(Dataset):
    """Dataset for equity ETF sequences with 48-day return targets."""

    def __init__(self, feature_df, label_df, indices, mean, std):
        self.feature_df = feature_df
        self.label_df   = label_df
        self.indices    = indices
        self.mean       = mean
        self.std        = std
        self.feat_names = get_feature_names()

        max_idx = len(feature_df) - SEQ_LEN - PRED_HORIZON
        self.indices = indices[(indices >= 0) & (indices <= max_idx)]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]

        x_list = []
        for ticker in TARGET_ETFS:
            vals = self.feature_df[ticker][self.feat_names].iloc[t:t + SEQ_LEN].values
            x_list.append(normalise(vals, self.mean, self.std))

        x = np.stack(x_list, axis=0).astype(np.float32)
        y = self.label_df[TARGET_ETFS].iloc[t + SEQ_LEN].values.astype(np.float32)

        return torch.from_numpy(x), torch.from_numpy(y)
