"""
data_utils.py
Load OHLCV data for 6 target ETFs from HuggingFace, build sliding-window
tensors and binary labels for next-day direction prediction.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# ── Constants ────────────────────────────────────────────────────────────────
TARGET_ETFS   = ["TLT", "VNQ", "SLV", "GLD", "LQD", "HYG"]
FEATURES      = ["Open", "High", "Low", "Close", "Volume"]
HF_REPO       = "P2SAMAPA/etf-dlinear-cross-data"
HF_DATA_FILE  = "fixed_income/ohlcv_fixed_income.parquet"
SEQ_LEN       = 48          # look-back window (trading days)
PRED_HORIZON  = 1           # next-day prediction
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.20
# TEST_RATIO  = 0.10  (remainder)
START_DATE    = "2008-01-01"


# ── Data loading ─────────────────────────────────────────────────────────────

def load_raw_df(hf_token: str | None = None) -> pd.DataFrame:
    """
    Download parquet directly via huggingface_hub (bypasses datasets schema
    inference which conflicts when multiple parquet configs exist in the repo).
    Returns a DataFrame with:
        index  : DatetimeIndex (daily)
        columns: MultiIndex (ticker, feature)
    Filtered to TARGET_ETFS and dates >= START_DATE.
    """
    from huggingface_hub import hf_hub_download
    import ast

    local_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=HF_DATA_FILE,
        repo_type="dataset",
        token=hf_token,
    )
    df = pd.read_parquet(local_path)

    # Ensure Date is the index
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    elif df.index.name == "Date":
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Rebuild MultiIndex columns from flat strings like "('TLT','Close')"
    if not isinstance(df.columns, pd.MultiIndex):
        df.columns = pd.MultiIndex.from_tuples(
            [ast.literal_eval(c) if c.startswith("(") else (c, "")
             for c in df.columns]
        )

    # Keep only target ETFs
    available = [t for t in TARGET_ETFS if t in df.columns.get_level_values(0)]
    missing   = set(TARGET_ETFS) - set(available)
    if missing:
        raise ValueError(f"ETFs not found in dataset: {missing}")

    df = df[available]
    df = df.loc[START_DATE:]
    df = df.dropna(how="all")

    return df


# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features per ETF:
        - daily_return : (close_t - close_{t-1}) / close_{t-1}
        - log_volume   : log(volume + 1)
        - hl_range     : (high - low) / close  (normalised intraday range)
    Returns DataFrame with same MultiIndex structure.
    """
    frames = []
    for ticker in TARGET_ETFS:
        t = df[ticker].copy()
        t["daily_return"] = t["Close"].pct_change()
        t["log_volume"]   = np.log1p(t["Volume"])
        t["hl_range"]     = (t["High"] - t["Low"]) / t["Close"]
        frames.append(t)

    result = pd.concat(frames, axis=1, keys=TARGET_ETFS)
    result = result.replace([np.inf, -np.inf], np.nan).dropna()
    return result


def get_feature_names() -> list[str]:
    return ["Open", "High", "Low", "Close", "Volume",
            "daily_return", "log_volume", "hl_range"]


# ── Normalisation ─────────────────────────────────────────────────────────────

def normalise(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (arr - mean) / (std + 1e-8)


# ── Labels ────────────────────────────────────────────────────────────────────

def make_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary label per ETF per day:
        1  if close_{t+1} > close_t
        0  otherwise
    """
    labels = {}
    for ticker in TARGET_ETFS:
        close = df[ticker]["Close"]
        labels[ticker] = (close.shift(-PRED_HORIZON) > close).astype(int)
    return pd.DataFrame(labels, index=df.index)


# ── Dataset ───────────────────────────────────────────────────────────────────

class ETFDataset(Dataset):
    """
    Each sample:
        x : (num_etfs, seq_len, num_features)  float32
        y : (num_etfs,)                         long
    """

    def __init__(
        self,
        feature_df: pd.DataFrame,
        label_df:   pd.DataFrame,
        indices:    np.ndarray,
        mean:       np.ndarray,
        std:        np.ndarray,
    ):
        self.feature_df = feature_df
        self.label_df   = label_df
        self.indices    = indices      # row indices into feature_df
        self.mean       = mean         # (num_features,)
        self.std        = std
        self.feat_names = get_feature_names()
        self.num_etfs   = len(TARGET_ETFS)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        # x: shape (num_etfs, seq_len, num_features)
        x = np.stack(
            [
                normalise(
                    self.feature_df[ticker][self.feat_names]
                    .iloc[t : t + SEQ_LEN]
                    .values,
                    self.mean,
                    self.std,
                )
                for ticker in TARGET_ETFS
            ],
            axis=0,
        ).astype(np.float32)

        # y: shape (num_etfs,)
        y = self.label_df[TARGET_ETFS].iloc[t + SEQ_LEN].values.astype(np.int64)

        return torch.tensor(x), torch.tensor(y)


# ── Main builder ──────────────────────────────────────────────────────────────

def build_dataloaders(
    hf_token: str | None = None,
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Returns train_loader, val_loader, test_loader, meta_dict.
    meta_dict contains: mean, std, dates, feature_df, label_df
    """
    raw_df     = load_raw_df(hf_token)
    feature_df = engineer_features(raw_df)
    label_df   = make_labels(feature_df)

    feat_names = get_feature_names()

    # For training we need valid labels, so trim the last PRED_HORIZON rows
    # But we keep full feature_df in meta so inference can use the latest date
    feature_df_train = feature_df.iloc[:-PRED_HORIZON]
    label_df_train   = label_df.iloc[:-PRED_HORIZON]
    n_rows     = len(feature_df_train) - SEQ_LEN

    # Chronological splits
    n_train = int(n_rows * TRAIN_RATIO)
    n_val   = int(n_rows * VAL_RATIO)

    train_idx = np.arange(0,               n_train)
    val_idx   = np.arange(n_train,         n_train + n_val)
    test_idx  = np.arange(n_train + n_val, n_rows)

    # Fit normalisation stats on training data only
    train_data = np.concatenate(
        [
            feature_df_train[ticker][feat_names].iloc[: n_train + SEQ_LEN].values
            for ticker in TARGET_ETFS
        ],
        axis=0,
    )
    mean = train_data.mean(axis=0)
    std  = train_data.std(axis=0)

    def make_ds(idx):
        return ETFDataset(feature_df_train, label_df_train, idx, mean, std)

    train_loader = DataLoader(make_ds(train_idx), batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(make_ds(val_idx),   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(make_ds(test_idx),  batch_size=batch_size, shuffle=False)

    meta = {
        "mean":       mean,
        "std":        std,
        "feature_df": feature_df,          # full df including latest date for inference
        "label_df":   label_df,
        "test_idx":   test_idx,
        "dates":      feature_df.index,
    }
    return train_loader, val_loader, test_loader, meta
