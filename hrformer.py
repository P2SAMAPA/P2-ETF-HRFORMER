"""
hrformer.py
Compact HRformer adapted for 6 ETFs × next-day binary classification.

Architecture (per the paper, scaled down for CPU training):
  1. Multi-Component Decomposition Layer  (MCDL)
  2. Component-wise Temporal Encoder      (CTE)
       ├── Trend     → Transformer encoder
       ├── Cyclic    → Fourier-attention encoder
       └── Volatility → RevIN + MLP + LSTM
  3. Adaptive Multi-Component Integration (AMCI)
  4. Inter-Stock Correlation Attention    (ISCA)
  5. Linear classifier → P(up) per ETF
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 1. Multi-Component Decomposition Layer ────────────────────────────────────

class MovingAvg(nn.Module):
    """Centred moving average for trend extraction."""
    def __init__(self, kernel_size: int = 13):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d)
        pad = self.kernel_size // 2
        x_t = x.permute(0, 2, 1)                        # (B, d, L)
        x_t = F.pad(x_t, (pad, pad), mode="replicate")
        x_t = self.avg(x_t).permute(0, 2, 1)            # (B, L, d)
        return x_t


class SpectralDecomposition(nn.Module):
    """
    FFT-based separation of cyclic vs volatility components.
    Adaptive threshold: keep frequencies where |X_f| > θ * local_mean(|X_f|).
    θ is a learnable scalar.
    """
    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x_w: torch.Tensor):
        # x_w: (B, L, d)
        X_f  = torch.fft.rfft(x_w, dim=1)               # (B, F, d)  F = L//2+1
        mag  = X_f.abs()                                 # (B, F, d)
        F_   = mag.shape[1]

        # Local spectral threshold via 1-d avg pool over frequency axis
        mag_t = mag.permute(0, 2, 1)                     # (B, d, F)
        k     = max(3, F_ // 4)
        # padding=k//2 keeps output length == input length, trim to F_ to be safe
        local = F.avg_pool1d(mag_t, kernel_size=k, stride=1,
                             padding=k // 2)             # (B, d, F or F+1)
        local = local[:, :, :F_]                         # (B, d, F)
        local = local.permute(0, 2, 1)                   # (B, F, d)

        mask  = (mag > self.theta * local).float()       # (B, F, d)
        X_cyc = torch.fft.irfft(X_f * mask,       n=x_w.shape[1], dim=1)
        X_vol = torch.fft.irfft(X_f * (1 - mask), n=x_w.shape[1], dim=1)
        return X_cyc, X_vol


class MultiComponentDecomposition(nn.Module):
    def __init__(self, kernel_size: int = 13):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)
        self.spectral   = SpectralDecomposition()

    def forward(self, x: torch.Tensor):
        # x: (B, L, d)
        x_trend = self.moving_avg(x)
        x_w     = x - x_trend
        x_cyclic, x_vol = self.spectral(x_w)
        return x_trend, x_cyclic, x_vol


# ── 2a. Trend encoder — standard Transformer ─────────────────────────────────

class TrendEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)          # (B, L, d)


# ── 2b. Cyclic encoder — Fourier attention ────────────────────────────────────

class FourierAttention(nn.Module):
    """
    Attention in the frequency domain: transform Q,K,V → frequency space,
    compute scaled dot-product, inverse-transform.
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, d = x.shape
        H, Dh   = self.n_heads, self.d_head

        def proj_fft(W, inp):
            out = W(inp)                                 # (B, L, d)
            out = out.view(B, L, H, Dh).permute(0, 2, 1, 3)  # (B,H,L,Dh)
            return torch.fft.rfft(out, dim=2)            # (B,H,F,Dh)

        Q = proj_fft(self.Wq, x)
        K = proj_fft(self.Wk, x)
        V = proj_fft(self.Wv, x)

        scale  = math.sqrt(self.d_head)
        scores = (Q * K.conj()).real / scale             # (B,H,F,Dh)
        attn   = torch.softmax(scores, dim=-1)
        out_f  = attn * V                                # (B,H,F,Dh)

        out = torch.fft.irfft(out_f, n=L, dim=2)        # (B,H,L,Dh)
        out = out.permute(0, 2, 1, 3).reshape(B, L, d)
        return self.Wo(out)


class CyclicEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn  = FourierAttention(d_model, n_heads)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x + self.drop(self.attn(x)))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class CyclicEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [CyclicEncoderLayer(d_model, n_heads, dropout) for _ in range(n_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ── 2c. Volatility encoder — RevIN + MLP + LSTM ──────────────────────────────

class RevIN(nn.Module):
    """Reversible Instance Normalisation."""
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps      = eps
        self.affine_w = nn.Parameter(torch.ones(num_features))
        self.affine_b = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self._mean = x.mean(dim=1, keepdim=True)
            self._std  = x.std(dim=1, keepdim=True) + self.eps
            x = (x - self._mean) / self._std
            x = x * self.affine_w + self.affine_b
        elif mode == "denorm":
            x = (x - self.affine_b) / (self.affine_w + self.eps)
            x = x * self._std + self._mean
        return x


class VolatilityEncoder(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.revin1 = RevIN(d_model)
        self.mlp1   = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.lstm   = nn.LSTM(d_model, d_model, batch_first=True)  # single layer, no dropout
        self.revin2 = RevIN(d_model)
        self.mlp2   = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.drop   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x     = self.revin1(x, "norm")
        x     = self.drop(self.mlp1(x))
        x, _  = self.lstm(x)
        x     = self.revin2(x, "norm")
        x     = self.drop(self.mlp2(x))
        return x


# ── 3. Adaptive Multi-Component Integration ──────────────────────────────────

class AMCI(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.gate_t = nn.Linear(d_model, d_model)
        self.gate_c = nn.Linear(d_model, d_model)
        self.gate_v = nn.Linear(d_model, d_model)

    def forward(self, h_t, h_c, h_v):
        g_t = torch.sigmoid(self.gate_t(h_t))
        g_c = torch.sigmoid(self.gate_c(h_c))
        g_v = torch.sigmoid(self.gate_v(h_v))
        return g_t * h_t + g_c * h_c + g_v * h_v   # (B, L, d)


# ── 4. Inter-Stock Correlation Attention ─────────────────────────────────────

class ISCA(nn.Module):
    """
    Treats each ETF as a token; learns cross-ETF dependencies.
    Input:  (B, M, d)  where M = num_etfs
    Output: (B, M, d)
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


# ── 5. Full HRformer ──────────────────────────────────────────────────────────

class HRformer(nn.Module):
    """
    HRformer for next-day ETF direction classification.

    Args:
        num_etfs      : number of ETFs (6)
        seq_len       : look-back window (48)
        num_features  : input features per ETF per day (8)
        d_model       : hidden dimension (64)
        n_heads       : attention heads (4)
        n_layers      : encoder depth (2)
        dropout       : dropout rate (0.1)
        kernel_size   : moving-average kernel for trend extraction (13)
    """

    def __init__(
        self,
        num_etfs:     int = 6,
        seq_len:      int = 48,
        num_features: int = 8,
        d_model:      int = 64,
        n_heads:      int = 4,
        n_layers:     int = 2,
        dropout:      float = 0.1,
        kernel_size:  int = 13,
    ):
        super().__init__()
        self.num_etfs = num_etfs
        self.seq_len  = seq_len
        self.d_model  = d_model

        # Input projection: map num_features → d_model
        self.input_proj = nn.Linear(num_features, d_model)

        # 1. Decomposition
        self.mcdl = MultiComponentDecomposition(kernel_size)

        # 2. Component encoders
        self.trend_enc  = TrendEncoder(d_model, n_heads, n_layers, dropout)
        self.cyclic_enc = CyclicEncoder(d_model, n_heads, n_layers, dropout)
        self.vol_enc    = VolatilityEncoder(d_model, dropout)

        # 3. AMCI
        self.amci = AMCI(d_model)

        # 4. Temporal aggregation: use attention pooling instead of flattening
        # CRITICAL FIX: Replace Linear(seq_len * d_model, d_model) with proper pooling
        self.temporal_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.temporal_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        
        # Alternative: Use mean pooling with a learnable projection
        self.temporal_pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # 5. ISCA
        self.isca = ISCA(d_model, n_heads, dropout)

        # 6. Classifier with dropout for regularization
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, 2)   # 2 classes: down / up
        
        # CRITICAL FIX: Initialize classifier weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, num_etfs, seq_len, num_features)
        returns logits : (B, num_etfs, 2)
        """
        B, M, L, F = x.shape

        # Process each ETF independently through decomp + CTE + AMCI
        etf_embeddings = []
        for i in range(M):
            xi = x[:, i, :, :]                      # (B, L, F)
            xi = self.input_proj(xi)                 # (B, L, d)

            x_t, x_c, x_v = self.mcdl(xi)

            h_t = self.trend_enc(x_t)               # (B, L, d)
            h_c = self.cyclic_enc(x_c)
            h_v = self.vol_enc(x_v)

            z = self.amci(h_t, h_c, h_v)            # (B, L, d)
            
            # CRITICAL FIX: Use mean pooling across time dimension instead of flatten
            # This preserves scale independence and is more stable
            z_pooled = z.mean(dim=1)                 # (B, d)
            z_pooled = self.temporal_pool(z_pooled)  # (B, d)
            
            etf_embeddings.append(z_pooled)

        # Stack: (B, M, d)
        Z = torch.stack(etf_embeddings, dim=1)

        # ISCA: cross-ETF attention
        Z = self.isca(Z)                             # (B, M, d)
        
        # Apply dropout before classification
        Z = self.dropout(Z)

        # Classify
        logits = self.classifier(Z)                  # (B, M, 2)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns P(up) per ETF, shape (B, M)."""
        # CRITICAL FIX: Ensure model is in eval mode
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            # CRITICAL FIX: Check for NaN/Inf in logits
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("WARNING: NaN or Inf detected in logits!")
                logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
            probs = torch.softmax(logits, dim=-1)
            # Return P(class=1) = P(up)
            return probs[:, :, 1]


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(
    num_etfs:     int = 6,
    seq_len:      int = 48,
    num_features: int = 8,
) -> HRformer:
    # CRITICAL FIX: Use lower dropout for small dataset to prevent underfitting
    return HRformer(
        num_etfs=num_etfs,
        seq_len=seq_len,
        num_features=num_features,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dropout=0.1,  # Reduced from 0.3 to prevent underfitting
        kernel_size=13,
    )
