"""
data_utils.py
Load OHLCV data for 6 target ETFs from HuggingFace.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

TARGET_ETFS = ["TLT", "VNQ", "SLV", "GLD", "LQD", "HYG"]
FEATURES = ["Open", "High", "Low", "Close", "Volume"]
HF_REPO = "P2SAMAPA/etf-dlinear-cross-data"
HF_DATA_FILE = "fixed_income/ohlcv_fixed_income.parquet"
SEQ_LEN = 48
PRED_HORIZON = 1
START_DATE = "2008-01-01"


def load_raw_df(hf_token=None):
    """Download parquet from HuggingFace."""
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

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    elif df.index.name == "Date":
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    if not isinstance(df.columns, pd.MultiIndex):
        df.columns = pd.MultiIndex.from_tuples(
            [ast.literal_eval(c) if c.startswith("(") else (c, "")
             for c in df.columns]
        )

    available = [t for t in TARGET_ETFS if t in df.columns.get_level_values(0)]
    missing = set(TARGET_ETFS) - set(available)
    if missing:
        raise ValueError(f"ETFs not found: {missing}")

    df = df[available].loc[START_DATE:].dropna(how="all")
    return df


def engineer_features(df):
    """Add derived features."""
    frames = []
    for ticker in TARGET_ETFS:
        if ticker not in df.columns.get_level_values(0):
            continue
        t = df[ticker].copy()
        t["daily_return"] = t["Close"].pct_change()
        t["log_volume"] = np.log1p(t["Volume"])
        t["hl_range"] = (t["High"] - t["Low"]) / t["Close"]
        frames.append(t)

    result = pd.concat(frames, axis=1, keys=TARGET_ETFS)
    result = result.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna(how="all")
    
    if len(result) < SEQ_LEN + PRED_HORIZON + 10:
        raise ValueError(f"Insufficient data: {len(result)} rows")
    
    return result


def get_feature_names():
    return ["Open", "High", "Low", "Close", "Volume",
            "daily_return", "log_volume", "hl_range"]


def normalise(arr, mean, std):
    """Normalize array."""
    std = np.where(np.isnan(std), 1.0, std)
    std = np.where(std == 0, 1.0, std)
    return (arr - mean) / std


def make_labels(df):
    """Binary label: 1 if next-day close > today close."""
    labels = {}
    for ticker in TARGET_ETFS:
        if ticker not in df.columns.get_level_values(0):
            continue
        close = df[ticker]["Close"]
        labels[ticker] = (close.shift(-PRED_HORIZON) > close).astype(int)
    
    label_df = pd.DataFrame(labels, index=df.index)
    
    # Print distribution once
    for ticker in labels.keys():
        up_ratio = label_df[ticker].mean()
        print(f"  {ticker}: {up_ratio:.1%} up, {1-up_ratio:.1%} down")
    
    return label_df


class ETFDataset(Dataset):
    """Dataset for ETF sequences."""
    
    def __init__(self, feature_df, label_df, indices, mean, std):
        self.feature_df = feature_df
        self.label_df = label_df
        self.indices = indices
        self.mean = mean
        self.std = std
        self.feat_names = get_feature_names()
        
        # Validate
        max_idx = len(feature_df) - SEQ_LEN - 1
        self.indices = indices[(indices >= 0) & (indices <= max_idx)]
        
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        
        x_list = []
        for ticker in TARGET_ETFS:
            vals = self.feature_df[ticker][self.feat_names].iloc[t:t+SEQ_LEN].values
            x_list.append(normalise(vals, self.mean, self.std))
        
        x = np.stack(x_list, axis=0).astype(np.float32)
        y = self.label_df[TARGET_ETFS].iloc[t + SEQ_LEN].values.astype(np.int64)
        
        return torch.from_numpy(x), torch.from_numpy(y)
