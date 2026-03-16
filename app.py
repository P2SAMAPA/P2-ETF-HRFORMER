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

st.set_page_config(
    page_title="ETF Oracle · HRformer Signals",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,400;0,500;0,600;0,700&family=DM+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #f4f6f9;
    color: #111827;
    font-size: 15px;
  }
  #MainMenu, footer, header { visibility: hidden; }

  /* Hero */
  .hero {
    background: #ffffff;
    border: 1px solid #d1d9e0;
    border-radius: 14px;
    padding: 26px 34px;
    margin-bottom: 22px;
  }
  .hero h1 { font-size: 1.8rem; font-weight: 700; letter-spacing: -0.5px; margin: 0 0 4px 0; color: #111827; }
  .hero p  { font-size: 0.95rem; color: #374151; margin: 0; font-weight: 500; }
  .hero .badge {
    display: inline-block; background: #f1f5f9; border: 1px solid #cbd5e1;
    border-radius: 20px; padding: 4px 13px;
    font-size: 0.8rem; font-weight: 600; color: #1e293b;
    margin-right: 6px; margin-top: 10px;
  }
  .hero .badge.active { background: #dbeafe; color: #1e40af; border-color: #93c5fd; }

  /* Signal card */
  .signal-card {
    background: #ffffff; border-radius: 14px; padding: 26px 30px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08), 0 0 0 1px rgba(0,0,0,0.05);
    margin-bottom: 18px;
  }
  .signal-card .label {
    font-size: 0.75rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: #6b7280; margin-bottom: 8px;
  }
  .signal-card .ticker {
    font-size: 3.4rem; font-weight: 700; color: #111827;
    letter-spacing: -2px; line-height: 1; font-family: 'DM Mono', monospace;
  }
  .signal-card .conf { font-size: 1.1rem; color: #15803d; font-weight: 700; margin-top: 8px; }
  .signal-card .date { font-size: 0.85rem; color: #374151; margin-top: 8px; font-family: 'DM Mono', monospace; font-weight: 500; }

  /* Metric tiles */
  .metric-tile {
    background: #ffffff; border-radius: 12px; padding: 18px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.07), 0 0 0 1px rgba(0,0,0,0.04);
    margin-bottom: 12px;
  }
  .metric-tile .m-label {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.07em;
    text-transform: uppercase; color: #6b7280; margin-bottom: 6px;
  }
  .metric-tile .m-value {
    font-size: 1.9rem; font-weight: 700; color: #111827;
    font-family: 'DM Mono', monospace; letter-spacing: -0.5px; line-height: 1.1;
  }
  .metric-tile .m-sub { font-size: 0.8rem; color: #374151; margin-top: 5px; font-weight: 500; }
  .pos { color: #15803d !important; }
  .neg { color: #b91c1c !important; }

  /* Section headers */
  .section-title {
    font-size: 1.1rem; font-weight: 700; color: #111827;
    margin: 22px 0 10px 0; padding-bottom: 8px; border-bottom: 2px solid #e2e8f0;
  }
  .section-sub {
    font-size: 0.9rem; color: #374151; font-weight: 400;
    margin: -6px 0 14px 0; line-height: 1.6;
  }

  /* Info box */
  .info-box {
    background: #eff6ff; border: 1px solid #bfdbfe;
    border-radius: 10px; padding: 16px 20px;
    font-size: 0.9rem; color: #1e3a5f; margin-bottom: 16px; line-height: 1.7;
  }
  .info-box strong { color: #1e3a8a; }

  /* Disclaimer */
  .disclaimer {
    background: #fffbeb; border: 1px solid #fbbf24;
    border-radius: 10px; padding: 14px 18px;
    font-size: 0.85rem; color: #78350f; margin-top: 28px; font-weight: 500;
  }
</style>
""", unsafe_allow_html=True)

ETF_COLORS = {
    "TLT": "#4299e1", "VNQ": "#48bb78", "SLV": "#a0aec0",
    "GLD": "#ecc94b", "LQD": "#9f7aea", "HYG": "#f6ad55",
}
BASE_LAYOUT = dict(font_family="DM Sans", paper_bgcolor="white", plot_bgcolor="white")
AXES        = dict(xaxis=dict(showgrid=True, gridcolor="#edf2f7", linecolor="#e2e8f0"),
                   yaxis=dict(showgrid=True, gridcolor="#edf2f7", linecolor="#e2e8f0"))

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_signal():
    if os.path.exists("latest.json"):
        with open("latest.json") as f:
            d = json.load(f)
            if d.get("signal", {}).get("recommended_etf"):
                return d
    try:
        import requests
        r = requests.get(
            "https://huggingface.co/P2SAMAPA/etf-hrformer-model/resolve/main/latest.json",
            timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

# ── Charts ────────────────────────────────────────────────────────────────────

def prob_bar_chart(probabilities: dict):
    tickers = list(probabilities.keys())
    probs   = [probabilities[t] for t in tickers]
    colors  = [ETF_COLORS.get(t, "#4299e1") for t in tickers]
    fig = go.Figure(go.Bar(
        x=tickers, y=probs, marker_color=colors,
        text=[f"{p:.1%}" for p in probs], textposition="outside",
    ))
    fig.update_layout(**BASE_LAYOUT, **AXES, height=260,
                      showlegend=False, margin=dict(l=0, r=0, t=24, b=0))
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    fig.add_hline(y=0.5, line_dash="dot", line_color="#cbd5e0", line_width=1)
    return fig


def equity_chart(dates, equity, picks):
    df  = pd.DataFrame({"date": pd.to_datetime(dates), "equity": equity, "pick": picks})
    fig = go.Figure()
    for etf, col in ETF_COLORS.items():
        sub = df[df["pick"] == etf]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["date"], y=sub["equity"], mode="markers",
            marker=dict(size=4, color=col, opacity=0.6), name=etf,
        ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["equity"], mode="lines",
        line=dict(color="#1a1d23", width=2), name="Portfolio",
    ))
    fig.add_hline(y=1.0, line_dash="dot", line_color="#cbd5e0", line_width=1)
    fig.update_layout(**BASE_LAYOUT, **AXES, height=320,
                      hovermode="x unified", margin=dict(l=0, r=0, t=24, b=0),
                      yaxis_title="Portfolio Value (starting = 1.0)")
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  xanchor="right", x=1))
    return fig


def drawdown_chart(dates, equity):
    eq  = np.array(equity)
    dd  = eq / np.maximum.accumulate(eq) - 1
    fig = go.Figure(go.Scatter(
        x=pd.to_datetime(dates), y=dd, mode="lines", fill="tozeroy",
        line=dict(color="#fc8181", width=1.5),
        fillcolor="rgba(252,129,129,0.15)",
    ))
    fig.update_layout(**BASE_LAYOUT, **AXES, height=200,
                      yaxis_tickformat=".0%", yaxis_title="Drawdown",
                      showlegend=False, margin=dict(l=0, r=0, t=24, b=0))
    return fig


def pick_distribution_chart(picks: list):
    counts = pd.Series(picks).value_counts().reindex(
        list(ETF_COLORS.keys()), fill_value=0)
    fig = go.Figure(go.Pie(
        labels=counts.index.tolist(), values=counts.values.tolist(),
        marker_colors=[ETF_COLORS[t] for t in counts.index],
        hole=0.55, textinfo="label+percent", textfont_size=12,
    ))
    fig.update_layout(**BASE_LAYOUT, height=260, showlegend=False,
                      margin=dict(l=10, r=10, t=10, b=10))
    return fig


def metric_tile(label, value, sub="", positive_good=True):
    if isinstance(value, float):
        c        = "pos" if (value >= 0) == positive_good else "neg"
        val_html = f'<div class="m-value {c}">{value:+.2%}</div>'
    else:
        val_html = f'<div class="m-value">{value}</div>'
    return f"""<div class="metric-tile">
      <div class="m-label">{label}</div>{val_html}
      <div class="m-sub">{sub}</div></div>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    data = load_signal()

    # ── Hero — light ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
      <div class="hero-top">
        <div>
          <h1>📡 ETF Oracle</h1>
          <p>Next-day trading signal powered by Hybrid Relational Transformer (HRformer)</p>
        </div>
        <div class="powered-by">
          <strong>HRformer</strong><br>
          Xu et al., Electronics 2025<br>
          Adapted for fixed-income ETFs
        </div>
      </div>
      <div class="badge-row">
        <span class="badge">TLT · 20Y Treasury</span>
        <span class="badge">VNQ · Real Estate</span>
        <span class="badge">SLV · Silver</span>
        <span class="badge">GLD · Gold</span>
        <span class="badge">LQD · IG Corporate</span>
        <span class="badge">HYG · High Yield</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if data is None:
        st.warning("No signal data found yet — trigger the GitHub Actions workflow manually to generate the first signal.")
        return

    sig    = data.get("signal", {})
    bt     = data.get("backtest", {})
    bt_sum = bt.get("summary", {})
    metrics = data.get("model_metrics", {})
    gen_at  = data.get("generated_at", "")

    dates  = bt.get("dates", [])
    equity = bt.get("equity_curve", [])
    picks  = bt.get("picks", [])

    bt_start = dates[0][:10]  if dates else "—"
    bt_end   = dates[-1][:10] if dates else "—"
    bt_label = f"{bt_start} → {bt_end}"

    # ── Signal + probabilities ────────────────────────────────────────────────
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
          <div class="date">Based on data to: {sdate}</div>
          <div class="date">Generated: {gen_at[:10] if gen_at else '—'}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_prob:
        st.markdown('<div class="section-title">P(up) — All ETFs</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Model\'s estimated probability each ETF closes higher tomorrow. The highest P(up) becomes the day\'s pick.</div>',
                    unsafe_allow_html=True)
        probs = sig.get("probabilities", {})
        if probs:
            st.plotly_chart(prob_bar_chart(probs), width='stretch',
                            config={"displayModeBar": False})

    # ── Backtest ──────────────────────────────────────────────────────────────
    st.markdown(f'<div class="section-title">Backtest Performance <span style="font-weight:400;font-size:0.85rem;color:#718096;">· {bt_label} · long-only, 0.1% trading cost each side</span></div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5, gap="small")
    with c1:
        st.markdown(metric_tile("Total Return", bt_sum.get("total_return"),
                                sub=bt_label), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_tile("Ann. Return", bt_sum.get("annualised_return"),
                                sub="Annualised"), unsafe_allow_html=True)
    with c3:
        v = bt_sum.get("sharpe_ratio")
        c = "pos" if v and v > 0 else "neg"
        st.markdown(f"""<div class="metric-tile">
          <div class="m-label">Sharpe Ratio</div>
          <div class="m-value {c}">{f'{v:.2f}' if v is not None else '—'}</div>
          <div class="m-sub">Return ÷ risk (ann.)</div></div>""",
                    unsafe_allow_html=True)
    with c4:
        st.markdown(metric_tile("Max Drawdown", bt_sum.get("max_drawdown"),
                                sub="Worst peak-to-trough", positive_good=False),
                    unsafe_allow_html=True)
    with c5:
        st.markdown(metric_tile("Ann. Volatility", bt_sum.get("annualised_vol"),
                                sub="Daily std × √252", positive_good=False),
                    unsafe_allow_html=True)

    # ── Equity curve ──────────────────────────────────────────────────────────
    st.markdown(f'<div class="section-title">Equity Curve <span style="font-weight:400;font-size:0.85rem;color:#718096;">· {bt_label}</span></div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Portfolio value starting at 1.0. Coloured dots show which ETF was held each day.</div>',
                unsafe_allow_html=True)
    if dates and equity:
        st.plotly_chart(equity_chart(dates, equity, picks),
                        width='stretch', config={"displayModeBar": False})

    # ── Drawdown + Pick distribution ──────────────────────────────────────────
    col_dd, col_pie = st.columns([2, 1], gap="large")
    with col_dd:
        st.markdown('<div class="section-title">Drawdown</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">How far the portfolio fell from its previous peak at each point in time.</div>',
                    unsafe_allow_html=True)
        if equity:
            st.plotly_chart(drawdown_chart(dates, equity), width='stretch',
                            config={"displayModeBar": False})
    with col_pie:
        st.markdown(f'<div class="section-title">Pick Distribution</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="section-sub">How often each ETF was selected during the backtest period ({bt_label}).</div>',
                    unsafe_allow_html=True)
        if picks:
            st.plotly_chart(pick_distribution_chart(picks), width='stretch',
                            config={"displayModeBar": False})

    # ── Model metrics ─────────────────────────────────────────────────────────
    if metrics:
        st.markdown('<div class="section-title">Model Classification Metrics (Test Set)</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-box">
          These metrics measure how well the model predicts next-day <strong>up or down</strong>
          direction for each ETF on the held-out test set ({bt_label}), before any trading costs.<br><br>
          · <strong>Accuracy</strong> — % of days where the predicted direction was correct.<br>
          · <strong>Precision</strong> — of days the model predicted "up", how often it was actually up.<br>
          · <strong>Recall</strong> — of all actual up-days, how many did the model correctly flag.<br>
          · <strong>F1</strong> — harmonic mean of precision and recall; the primary model quality score.<br><br>
          A random coin-flip scores ~50% accuracy. Consistent accuracy above 52–53% is meaningful for daily ETF prediction.
        </div>
        """, unsafe_allow_html=True)

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
                "ETF":       "ALL (macro avg)",
                "Accuracy":  f"{agg.get('accuracy', 0):.1%}",
                "Precision": f"{agg.get('precision', 0):.1%}",
                "Recall":    f"{agg.get('recall', 0):.1%}",
                "F1":        f"{agg.get('f1', 0):.1%}",
            })
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="disclaimer">
      <strong>Disclaimer:</strong> This dashboard is for research and educational purposes only.
      It does not constitute financial advice. Model outputs are statistical predictions and carry
      no guarantee of future returns. Always conduct your own due diligence before making
      investment decisions.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
