# P2-ETF-HRFORMER

> Next-day ETF trading signal generation using a compact adaptation of the **HRformer** architecture.

---

## Research Foundation

This project adapts the model proposed in:

**HRformer: A Hybrid Relational Transformer for Stock Time Series Forecasting**
Xu, H.; Wan, H.; Wu, Y.; Zheng, J.; Xie, L.
*Electronics* 2025, 14, 4459. [https://doi.org/10.3390/electronics14224459](https://doi.org/10.3390/electronics14224459)

---

## What This Project Does

Six fixed-income and commodity ETFs are modelled jointly:

| Ticker | Description |
|--------|-------------|
| TLT | iShares 20+ Year Treasury Bond ETF |
| VNQ | Vanguard Real Estate ETF |
| SLV | iShares Silver Trust |
| GLD | SPDR Gold Shares |
| LQD | iShares iBoxx $ Investment Grade Corporate Bond ETF |
| HYG | iShares iBoxx $ High Yield Corporate Bond ETF |

Every trading day the pipeline:
1. Loads latest OHLCV data from HuggingFace Datasets
2. Decomposes each ETF series into **trend**, **cyclic**, and **volatility** components
3. Encodes each component with a specialist encoder (Transformer / Fourier-attention / LSTM)
4. Fuses components via **Adaptive Multi-Component Integration (AMCI)**
5. Learns cross-ETF dependencies via **Inter-Stock Correlation Attention (ISCA)**
6. Outputs P(up) for each ETF → selects the **single highest-confidence long signal**

---

## Architecture Overview

```
OHLCV (6 ETFs × 48-day window)
        │
        ▼
Multi-Component Decomposition Layer
  ├── Moving Average  →  Trend  (Xt)
  └── FFT / IFFT      →  Cyclic (Xc) + Volatility (Xv)
        │
        ▼
Component-wise Temporal Encoder (CTE)
  ├── Transformer Encoder     →  Trend representation
  ├── Fourier Attention       →  Cyclic representation
  └── RevIN + MLP + LSTM      →  Volatility representation
        │
        ▼
Adaptive Multi-Component Integration (AMCI)
  └── Learnable sigmoid gating → Fused embedding Zt
        │
        ▼
Inter-Stock Correlation Attention (ISCA)
  └── Multi-head attention across 6 ETF tokens
        │
        ▼
Linear classifier → P(up) per ETF → Top pick signal
```

---

## Pipeline

```
HuggingFace Dataset  ──►  GitHub Actions (daily)
                               │
                         train.py  (retrain model)
                               │
                         model.pt  saved to repo
                               │
                         infer.py  (generate signal)
                               │
                         signals/latest.json
                               │
                         Streamlit app  (display UI)
```

---

## Project Structure

```
P2-ETF-HRFORMER/
├── .github/
│   └── workflows/
│       └── daily_train.yml      # Daily retrain + inference
├── model/
│   └── hrformer.py              # HRformer model definition
├── data/
│   └── data_utils.py            # HF dataset loader + feature engineering
├── signals/
│   └── latest.json              # Latest trading signal (auto-updated)
├── train.py                     # Training entry point
├── infer.py                     # Inference entry point
├── app.py                       # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Setup

### Secrets required in GitHub repo settings

| Secret | Purpose |
|--------|---------|
| `HF_TOKEN` | Read/write access to HuggingFace dataset and model hub |

### Local development

```bash
pip install -r requirements.txt
python train.py
python infer.py
streamlit run app.py
```

---

## Disclaimer

This project is for **research and educational purposes only**. It is not financial advice. Past model performance does not guarantee future returns.
