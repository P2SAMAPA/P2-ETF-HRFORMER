"""
hrformer.py
Simplified HRformer for next-day ETF direction prediction.
Removes unstable components (Fourier attention, decomposition) 
that hurt small-dataset performance.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TrendEncoder(nn.Module):
    """Standard Transformer encoder for temporal dependencies."""
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


class StockCorrelationAttention(nn.Module):
    """
    ISCA: Cross-ETF attention. Treats each ETF as a token.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, M, d) where M = num_etfs
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class HRformer(nn.Module):
    """
    Simplified HRformer for next-day ETF classification.
    """

    def __init__(
        self,
        num_etfs: int = 6,
        seq_len: int = 48,
        num_features: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_etfs = num_etfs
        self.seq_len = seq_len
        self.d_model = d_model
        
        # Input normalization
        self.input_norm = nn.LayerNorm(num_features)
        
        # Feature projection
        self.input_proj = nn.Linear(num_features, d_model)
        
        # Positional encoding (learnable)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        
        # Temporal encoding
        self.temporal_encoder = TrendEncoder(d_model, n_heads, n_layers, dropout)
        
        # Temporal pooling: attention-based aggregation
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.pool_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        
        # Cross-ETF correlation
        self.isca = StockCorrelationAttention(d_model, n_heads, dropout)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)
        )
        
        self._init_weights()
        
    def _init_weights(self):
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
        x: (B, M, L, F)
        Returns: (B, M, 2) logits
        """
        B, M, L, F = x.shape
        
        # Process each ETF
        etf_tokens = []
        for i in range(M):
            xi = x[:, i, :, :]  # (B, L, F)
            
            # Normalize and project
            xi = self.input_norm(xi)
            xi = self.input_proj(xi)  # (B, L, d)
            
            # Add positional encoding
            xi = xi + self.pos_encoding[:, :L, :]
            
            # Temporal encoding
            xi = self.temporal_encoder(xi)  # (B, L, d)
            
            # Attention-based pooling
            query = self.pool_query.expand(B, -1, -1)  # (B, 1, d)
            pooled, _ = self.pool_attn(query, xi, xi)  # (B, 1, d)
            etf_tokens.append(pooled.squeeze(1))  # (B, d)
        
        # Stack: (B, M, d)
        Z = torch.stack(etf_tokens, dim=1)
        
        # Cross-ETF correlation
        Z = self.isca(Z)
        
        # Classify
        logits = self.classifier(Z)  # (B, M, 2)
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns P(up) for each ETF: (B, M)"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
            # Check for NaN
            if torch.isnan(probs).any():
                print("WARNING: NaN in probabilities, returning uniform")
                return torch.ones_like(probs[:, :, 1]) / self.num_etfs
            return probs[:, :, 1]


def build_model(num_etfs=6, seq_len=48, num_features=8):
    return HRformer(
        num_etfs=num_etfs,
        seq_len=seq_len,
        num_features=num_features,
        d_model=128,
        n_heads=4,
        n_layers=2,
        dropout=0.3,
    )
