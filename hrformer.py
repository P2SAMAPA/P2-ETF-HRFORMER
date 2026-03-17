"""
hrformer.py
HRformer for 48-day return prediction (regression).
Matches paper architecture more closely.
"""

import math
import torch
import torch.nn as nn


class TrendEncoder(nn.Module):
    """Standard Transformer for trend."""
    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, 
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class FourierAttention(nn.Module):
    """
    Fourier Attention from paper (simplified).
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, d = x.shape
        H, Dh = self.n_heads, self.d_head

        def proj_fft(W, inp):
            out = W(inp)
            out = out.view(B, L, H, Dh).permute(0, 2, 1, 3)
            return torch.fft.rfft(out, dim=2)

        Q = proj_fft(self.Wq, x)
        K = proj_fft(self.Wk, x)
        V = proj_fft(self.Wv, x)

        scale = math.sqrt(self.d_head)
        scores = (Q * K.conj()).real / scale
        attn = torch.softmax(scores, dim=-1)
        out_f = attn * V

        out = torch.fft.irfft(out_f, n=L, dim=2)
        out = out.permute(0, 2, 1, 3).reshape(B, L, d)
        return self.Wo(out)


class CyclicEncoder(nn.Module):
    """Fourier-based encoder for cyclic patterns."""
    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': FourierAttention(d_model, n_heads),
                'norm1': nn.LayerNorm(d_model),
                'ff': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model),
                ),
                'norm2': nn.LayerNorm(d_model),
                'drop': nn.Dropout(dropout),
            }) for _ in range(n_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer['norm1'](x + layer['drop'](layer['attn'](x)))
            x = layer['norm2'](x + layer['drop'](layer['ff'](x)))
        return x


class VolatilityEncoder(nn.Module):
    """LSTM-based encoder for volatility."""
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(d_model, d_model, num_layers=1, 
                           batch_first=True, dropout=0)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.mlp(x)
        return x


class HRformer(nn.Module):
    """
    HRformer matching paper: 3 components + ISCA.
    Output: 48-day return prediction (regression).
    """

    def __init__(
        self,
        num_etfs: int = 6,
        seq_len: int = 48,
        num_features: int = 8,
        d_model: int = 128,  # Paper uses 512, but 128 works for small data
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_etfs = num_etfs
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(num_features, d_model)
        
        # Three encoders for different components
        self.trend_enc = TrendEncoder(d_model, n_heads, n_layers, dropout)
        self.cyclic_enc = CyclicEncoder(d_model, n_heads, n_layers, dropout)
        self.vol_enc = VolatilityEncoder(d_model, dropout)
        
        # Adaptive fusion (AMCI)
        self.gate_t = nn.Linear(d_model, d_model)
        self.gate_c = nn.Linear(d_model, d_model)
        self.gate_v = nn.Linear(d_model, d_model)
        
        # Temporal pooling
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        # ISCA: Cross-stock attention
        self.isca = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.isca_norm = nn.LayerNorm(d_model)
        
        # Output: 48-day return prediction (regression)
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)  # Single return value per ETF
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, M, L, F)
        Returns: (B, M) predicted 48-day returns
        """
        B, M, L, F = x.shape
        
        # Process each ETF independently
        etf_reps = []
        for i in range(M):
            xi = x[:, i, :, :]  # (B, L, F)
            xi = self.input_proj(xi)  # (B, L, d)
            
            # Three component encoders
            h_t = self.trend_enc(xi)      # (B, L, d)
            h_c = self.cyclic_enc(xi)     # (B, L, d)
            h_v = self.vol_enc(xi)        # (B, L, d)
            
            # AMCI: Adaptive gating
            g_t = torch.sigmoid(self.gate_t(h_t))
            g_c = torch.sigmoid(self.gate_c(h_c))
            g_v = torch.sigmoid(self.gate_v(h_v))
            z = g_t * h_t + g_c * h_c + g_v * h_v  # (B, L, d)
            
            # Temporal pooling: (B, L, d) -> (B, d)
            z = self.temporal_pool(z.transpose(1, 2)).squeeze(-1)  # (B, d)
            etf_reps.append(z)
        
        # Stack: (B, M, d)
        Z = torch.stack(etf_reps, dim=1)
        
        # ISCA: Cross-ETF attention
        Z_attn, _ = self.isca(Z, Z, Z)
        Z = self.isca_norm(Z + Z_attn)
        
        # Predict returns: (B, M, 1) -> (B, M)
        returns = self.output_proj(Z).squeeze(-1)
        return returns
    
    def predict_returns(self, x: torch.Tensor) -> torch.Tensor:
        """Predict 48-day returns."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)


def build_model(num_etfs=6, seq_len=48, num_features=8):
    return HRformer(
        num_etfs=num_etfs,
        seq_len=seq_len,
        num_features=num_features,
        d_model=128,
        n_heads=4,
        n_layers=2,
        dropout=0.2,
    )
