"""
app.py
Professional Streamlit dashboard for P2-ETF-HRFORMER trading signals.
Light theme, clean financial UI with full metrics and backtest visualisation.
"""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ETF Oracle · HRformer Signals",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #f7f8fa;
    color: #1a1d23;
  }

  #MainMenu, footer, header { visibility: hidden; }

  .hero {
    background: linear-gradient(135deg, #1a1d23 0%, #2d3748 100%);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 28px;
    color: #fff;
  }
  .hero h1 { font-size: 2rem; font-weight: 600; letter-spacing: -0.5px; margin: 0 0 6px 0; color: #fff; }
  .hero p  { font-size: 0.95rem; color: #a0aec0; margin: 0; }
  .hero .badge {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    color: #e2e8f0;
    margin-right: 8px;
    margin-top: 12px;
  }

  .signal-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 28px 32px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 0 0 1px rgba(0,0,0,0.04);
    margin-bottom: 20px;
  }
  .signal-card .label  { font-size: 0.75rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #718096; margin-bottom: 6px; }
  .signal-card .ticker { font-size: 3rem; font-weight: 600; color: #1a1d23; letter-spacing: -1px; line-height: 1; font-family: 'DM Mono', monospace; }
  .signal-card .conf   { font-size: 1rem; color: #48bb78; font-weight: 500; margin-top: 4px; }
  .signal-card .date   { font-size: 0.82rem; color: #a0aec0; margin-top: 10px; font-family: 'DM Mono', monospace; }

  .metric-tile {
    background: #ffffff;
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.07), 0 0 0 1px rgba(0,0,0,0.04);
    margin-bottom: 16px;
  }
  .metric-tile .m-label { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #a0aec0; margin-bottom: 4px; }
  .metric-tile .m-value { font-size: 1.7rem; font-weight: 600; color: #1a1d23; font-family: 'DM Mono', monospace; letter-spacing: -0.5px; }
  .metric-tile .m-sub   { font-size: 0.78rem; color: #718096; margin-top: 2px; }
  .pos { color: #38a169 !important; }
  .neg { color: #e53e3e !important; }

  .section-title {
    font-size: 1.05rem; font-weight: 600; color: #1a1d23;
    margin: 28px 0 14px 0; padding-bottom: 8px;
    border-bottom: 2px solid #edf2f7;
  }

  .disclaimer {
    background: #fffbeb; border: 1px solid #fbd38d; border-radius: 10px;
    padding: 14px 18px; font-size: 0.8rem; color: #744210; margin-top: 32px;
  }
</style>
""", unsafe_allow_html=True)

# ── Colour palette ────────────────────────────────────────────────────────────
ETF_COLORS = {
    "TLT": "#4299e1",
    "VNQ": "#48bb78",
    "SLV": "#a0aec0",
    "GLD": "#ecc94b",
    "LQD": "#9f7aea",
    "HYG": "#f6ad55",
}
CHART_LAYOUT = dict(
    font_family="DM Sans",
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(l=0, r=0, t=24, b=0),
    xaxis=dict(showgrid=True, gridcolor="#edf2f7", linecolor="#e2e8f0"),
    yaxis=dict(showgrid=True, gridcolor="#edf2f7", linecolor="#e2e8f0"),
)

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_signal() -> dict | None:
    # 1. Try local file (works when running alongside the repo)
    if os.path.exists("latest.json"):
        with open("latest.json") as f:
            d = json.load(f)
            if d.get("signal", {}).get("recommended_etf"):
                return d

    # 2. Fallback: fetch from HF Hub model repo
    try:
        import requests
        url = ("https://huggingface.co/P2SAMAPA/etf-hrformer-model"
               "/resolve/main/latest.json")
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


# ── Chart helpers ─────────────────────────────────────────────────────────────

def equity_chart(dates, equity, picks):
    df  = pd.DataFrame({"date": pd.to_datetime(dates), "equity": equity, "pick": picks})
    fig = go.Figure()
    for etf in ETF_COLORS:
        sub = df[df["pick"] == etf]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["date"], y=sub["equity"], mode="markers",
            marker=dict(size=4, color=ETF_COLORS[etf], opacity=0.6),
            name=etf,
        ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["equity"], mode="lines",
        line=dict(color="#1a1d23", width=2), name="Portfolio",
    ))
    fig.add_hline(y=1.0, line_dash="dot", line_color="#cbd5e0", line_width=1)
    fig.update_layout(
        **CHART_LAYOUT, height=340,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title="Portfolio Value", hovermode="x unified",
    )
    return fig


def prob_bar_chart(probabilities: dict):
    tickers = list(probabilities.keys())
    probs   = [probabilities[t] for t in tickers]
    colors  = [ETF_COLORS.get(t, "#4299e1") for t in tickers]
    fig = go.Figure(go.Bar(
        x=tickers, y=probs,
        marker_color=colors,
        text=[f"{p:.1%}" for p in probs],
        textposition="outside",
    ))
    fig.update_layout(
        **CHART_LAYOUT, height=260, showlegend=False,
    )
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    fig.add_hline(y=0.5, line_dash="dot", line_color="#cbd5e0", line_width=1)
    return fig


def drawdown_chart(equity):
    eq  = np.array(equity)
    dd  = eq / np.maximum.accumulate(eq) - 1
    fig = go.Figure(go.Scatter(
        y=dd, mode="lines", fill="tozeroy",
        line=dict(color="#fc8181", width=1.5),
        fillcolor="rgba(252,129,129,0.15)",
    ))
    fig.update_layout(
        **CHART_LAYOUT, height=200,
        yaxis_tickformat=".0%", yaxis_title="Drawdown", showlegend=False,
    )
    return fig


def training_history_chart(history: list):
    if not history:
        return None
    df  = pd.DataFrame(history)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df["epoch"], y=df["train_loss"],
        name="Train loss", line=dict(color="#4299e1", width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=df["epoch"], y=df["val_loss"],
        name="Val loss", line=dict(color="#9f7aea", width=1.5)))
    fig.add_trace(go.Scatter(x=df["epoch"], y=df["val_f1"],
        name="Val F1", line=dict(color="#48bb78", width=2)), secondary_y=True)
    fig.update_layout(
        **CHART_LAYOUT, height=260,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Loss", secondary_y=False)
    fig.update_yaxes(title_text="F1",   secondary_y=True, range=[0, 1])
    return fig


def pick_distribution_chart(picks: list):
    counts = pd.Series(picks).value_counts().reindex(list(ETF_COLORS.keys()), fill_value=0)
    fig = go.Figure(go.Pie(
        labels=counts.index.tolist(),
        values=counts.values.tolist(),
        marker_colors=[ETF_COLORS[t] for t in counts.index],
        hole=0.55, textinfo="label+percent", textfont_size=12,
    ))
    fig.update_layout(
        **CHART_LAYOUT, height=260, showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def metric_tile(label, value, sub="", positive_good=True):
    if isinstance(value, float):
        c        = "pos" if (value >= 0) == positive_good else "neg"
        val_html = f'<div class="m-value {c}">{value:+.2%}</div>'
    else:
        val_html = f'<div class="m-value">{value}</div>'
    return f"""
    <div class="metric-tile">
      <div class="m-label">{label}</div>
      {val_html}
      <div class="m-sub">{sub}</div>
    </div>"""


# ── Main app ──────────────────────────────────────────────────────────────────

def main():
    data = load_signal()

    # Hero
    st.markdown("""
    <div class="hero">
      <h1>📡 ETF Oracle</h1>
      <p>Next-day trading signal · Hybrid Relational Transformer (HRformer)</p>
      <span class="badge">TLT</span><span class="badge">VNQ</span>
      <span class="badge">SLV</span><span class="badge">GLD</span>
      <span class="badge">LQD</span><span class="badge">HYG</span>
    </div>
    """, unsafe_allow_html=True)

    if data is None:
        st.warning("No signal data found yet. The model may still be training.")
        st.info("Trigger manually: GitHub repo → Actions → Daily Train & Signal → Run workflow.")
        return

    sig     = data.get("signal", {})
    bt      = data.get("backtest", {})
    bt_sum  = bt.get("summary", {})
    metrics = data.get("model_metrics", {})
    tr_info = data.get("training_info", {})
    gen_at  = data.get("generated_at", "")

    # ── Signal + probabilities ─────────────────────────────────────────────
    col_sig, col_prob = st.columns([1, 2], gap="large")
    with col_sig:
        rec   = sig.get("recommended_etf", "—")
        conf  = sig.get("confidence", 0)
        sdate = sig.get("signal_date", "—")
        st.markdown(f"""
        <div class="signal-card">
          <div class="label">Tomorrow's Pick</div>
          <div class="ticker">{rec}</div>
          <div class="conf">P(up) = {conf:.1%}</div>
          <div class="date">Signal date: {sdate}</div>
          <div class="date">Generated: {gen_at[:10] if gen_at else '—'}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_prob:
        st.markdown('<div class="section-title">P(up) — All ETFs</div>', unsafe_allow_html=True)
        probs = sig.get("probabilities", {})
        if probs:
            st.plotly_chart(prob_bar_chart(probs), use_container_width=True,
                            config={"displayModeBar": False})

    # ── Backtest metrics ───────────────────────────────────────────────────
    st.markdown('<div class="section-title">Backtest Performance</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5, gap="small")
    tiles = [
        (c1, "Total Return",     bt_sum.get("total_return"),      "Test period",      True),
        (c2, "Ann. Return",      bt_sum.get("annualised_return"),  "Annualised",       True),
        (c4, "Max Drawdown",     bt_sum.get("max_drawdown"),       "Peak-to-trough",   False),
        (c5, "Ann. Volatility",  bt_sum.get("annualised_vol"),     "Daily std × √252", False),
    ]
    for col, label, val, sub, pg in tiles:
        with col:
            st.markdown(metric_tile(label, val, sub, pg), unsafe_allow_html=True)
    with c3:
        v = bt_sum.get("sharpe_ratio")
        c = "pos" if v and v > 0 else "neg"
        st.markdown(f"""
        <div class="metric-tile">
          <div class="m-label">Sharpe Ratio</div>
          <div class="m-value {c}">{f'{v:.2f}' if v is not None else '—'}</div>
          <div class="m-sub">Risk-adjusted</div>
        </div>""", unsafe_allow_html=True)

    # ── Equity curve ───────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Equity Curve</div>', unsafe_allow_html=True)
    dates  = bt.get("dates", [])
    equity = bt.get("equity_curve", [])
    picks  = bt.get("picks", [])
    if dates and equity:
        st.plotly_chart(equity_chart(dates, equity, picks),
                        use_container_width=True, config={"displayModeBar": False})

    # ── Drawdown + Pick distribution ───────────────────────────────────────
    col_dd, col_pie = st.columns([2, 1], gap="large")
    with col_dd:
        st.markdown('<div class="section-title">Drawdown</div>', unsafe_allow_html=True)
        if equity:
            st.plotly_chart(drawdown_chart(equity), use_container_width=True,
                            config={"displayModeBar": False})
    with col_pie:
        st.markdown('<div class="section-title">Pick Distribution</div>', unsafe_allow_html=True)
        if picks:
            st.plotly_chart(pick_distribution_chart(picks), use_container_width=True,
                            config={"displayModeBar": False})

    # ── Model metrics table ────────────────────────────────────────────────
    if metrics:
        st.markdown('<div class="section-title">Model Metrics (Test Set)</div>',
                    unsafe_allow_html=True)
        rows = []
        for ticker, m in metrics.items():
            if ticker == "aggregate":
                continue
            rows.append({
                "ETF":       ticker,
                "Accuracy":  f"{m.get('accuracy', 0):.1%}",
                "Precision": f"{m.get('precision', 0):.1%}",
                "Recall":    f"{m.get('recall', 0):.1%}",
                "F1":        f"{m.get('f1', 0):.1%}",
            })
        if "aggregate" in metrics:
            agg = metrics["aggregate"]
            rows.append({
                "ETF":       "ALL (macro)",
                "Accuracy":  f"{agg.get('accuracy', 0):.1%}",
                "Precision": f"{agg.get('precision', 0):.1%}",
                "Recall":    f"{agg.get('recall', 0):.1%}",
                "F1":        f"{agg.get('f1', 0):.1%}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Training history ───────────────────────────────────────────────────
    history = tr_info.get("history", [])
    if history:
        st.markdown('<div class="section-title">Training History</div>', unsafe_allow_html=True)
        fig_h = training_history_chart(history)
        if fig_h:
            col_th, col_ti = st.columns([3, 1], gap="large")
            with col_th:
                st.plotly_chart(fig_h, use_container_width=True,
                                config={"displayModeBar": False})
            with col_ti:
                st.markdown(f"""
                <div class="metric-tile" style="margin-top:24px">
                  <div class="m-label">Best Val F1</div>
                  <div class="m-value">{tr_info.get('best_val_f1', 0):.3f}</div>
                  <div class="m-sub">Epochs: {tr_info.get('epochs_run', '—')}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Disclaimer ─────────────────────────────────────────────────────────
    st.markdown("""
    <div class="disclaimer">
      <strong>Disclaimer:</strong> This dashboard is for research and educational purposes only.
      It does not constitute financial advice. Model outputs are statistical predictions and
      carry no guarantee of future returns. Always conduct your own due diligence before
      making investment decisions.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
