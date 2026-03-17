"""
app.py — ETF Oracle · 48-Day HRformer Dashboard (Single ETF)
"""

import json, os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="ETF Oracle · HRformer 48D", page_icon="📡",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #f4f6f9; }
  .sig-card { background:#fff; border-radius:14px; padding:26px 30px;
              border:1px solid #d1d9e0; margin-bottom:4px; }
  .sig-label { font-size:.75rem; font-weight:700; letter-spacing:.1em;
               text-transform:uppercase; color:#6b7280; margin-bottom:8px; }
  .sig-ticker { font-size:4rem; font-weight:800; color:#111827;
                letter-spacing:-3px; line-height:1; font-family:monospace; }
  .sig-ret   { font-size:1.4rem; font-weight:700; color:#15803d; margin-top:8px; }
  .sig-ret-neg { font-size:1.4rem; font-weight:700; color:#b91c1c; margin-top:8px; }
  .sig-date   { font-size:.88rem; font-weight:600; color:#374151;
                margin-top:6px; font-family:monospace; }
  .m-card { background:#fff; border-radius:12px; padding:18px 20px;
            border:1px solid #d1d9e0; }
  .m-label { font-size:.72rem; font-weight:700; letter-spacing:.08em;
             text-transform:uppercase; color:#6b7280; margin-bottom:6px; }
  .m-pos  { font-size:2rem; font-weight:800; color:#15803d;
            font-family:monospace; letter-spacing:-1px; line-height:1.1; }
  .m-neg  { font-size:2rem; font-weight:800; color:#b91c1c;
            font-family:monospace; letter-spacing:-1px; line-height:1.1; }
  .m-neu  { font-size:2rem; font-weight:800; color:#111827;
            font-family:monospace; letter-spacing:-1px; line-height:1.1; }
  .best-badge { display:inline-block; background:#dcfce7; border:1px solid #86efac;
                border-radius:20px; padding:3px 14px; font-size:.8rem;
                font-weight:700; color:#15803d; margin-left:10px; }
</style>
""", unsafe_allow_html=True)

ETF_COLORS = {"TLT":"#3b82f6","VNQ":"#22c55e","SLV":"#94a3b8",
              "GLD":"#eab308","LQD":"#a855f7","HYG":"#f97316"}
ETF_NAMES  = {"TLT":"20Y Treasury","VNQ":"Real Estate","SLV":"Silver",
              "GLD":"Gold","LQD":"IG Corporate","HYG":"High Yield"}
MODE_COLORS = {"expanding": "#6366f1", "fixed": "#f59e0b"}
MODE_LABELS = {"expanding": "Expanding window", "fixed": "Fixed 2-year window"}


@st.cache_data(ttl=3600)
def load_data():
    """Load latest signal."""
    if os.path.exists("latest.json"):
        with open("latest.json") as f:
            return json.load(f)
    
    try:
        import requests
        r = requests.get(
            "https://huggingface.co/P2SAMAPA/etf-hrformer-model/resolve/main/latest.json",
            timeout=10
        )
        if r.status_code == 200:
            return r.json()
    except:
        pass
    
    return None


def mcard(label, value, sub="", good=True):
    if isinstance(value, (int, float)):
        if abs(value) > 1e6:
            value_str, cls = "Error", "m-neu"
        else:
            cls = "m-pos" if (value >= 0) == good else "m-neg"
            value_str = f"{value:+.2%}" if abs(value) < 10 else f"{value:.2f}"
    else:
        cls, value_str = "m-neu", str(value) if value else "—"
    
    return f'<div class="m-card"><div class="m-label">{label}</div><div class="{cls}">{value_str}</div><div class="m-sub">{sub}</div></div>'


def chart_returns_bar(predicted_returns, selected_etf):
    tickers = list(predicted_returns.keys())
    vals = [predicted_returns[t] for t in tickers]
    
    colors = [ETF_COLORS.get(t, "#3b82f6") if t == selected_etf else "#d1d5db" 
              for t in tickers]
    
    fig = go.Figure(go.Bar(
        x=tickers,
        y=vals,
        marker_color=colors,
        text=[f"{v*100:+.1f}%" for v in vals],
        textposition="outside",
        textfont=dict(size=13, color="#111827"),
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", tickformat=".0%", 
                   title="Predicted 48-day return"),
        showlegend=False
    )
    
    fig.add_hline(y=0, line_color="#111827", line_width=1)
    
    return fig


def main():
    data = load_data()
    
    st.markdown("## 📡 ETF Oracle · 48-Day Horizon")
    st.markdown("**Single ETF selection · 48-day hold · Highest predicted return**")
    
    # ETF legend
    cols_b = st.columns(6)
    for i, (t, name) in enumerate(ETF_NAMES.items()):
        bg = "#dbeafe" if data and data.get("signal", {}).get("recommended_etf") == t else "#f1f5f9"
        bc = "#93c5fd" if data and data.get("signal", {}).get("recommended_etf") == t else "#cbd5e1"
        cols_b[i].markdown(
            f'<div style="background:{bg};border:1px solid {bc};border-radius:20px;'
            f'padding:5px 12px;text-align:center;font-size:.82rem;font-weight:600;">'
            f'{t}</div>', unsafe_allow_html=True
        )
    
    st.divider()
    
    if data is None:
        st.warning("⏳ No signal data. Run infer.py first.")
        return
    
    sig = data.get("signal", {})
    best_mode = data.get("best_mode", "fixed")
    perf = data.get("performance", {})
    
    rec_etf = sig.get("recommended_etf", "—")
    pred_ret = sig.get("predicted_return", 0)
    pred_returns = sig.get("predicted_returns", {})
    sdate = sig.get("signal_date", "—")
    hold_until = sig.get("hold_until", "—")
    data_date = sig.get("data_date", "—")
    
    # Signal card
    col_s, col_p = st.columns([1, 2], gap="large")
    
    with col_s:
        ret_class = "sig-ret" if pred_ret >= 0 else "sig-ret-neg"
        st.markdown(f"""
        <div class="sig-card">
          <div class="sig-label">48-Day Signal</div>
          <div class="sig-ticker">{rec_etf}</div>
          <div class="{ret_class}">{pred_ret*100:+.2f}% predicted</div>
          <div class="sig-date">Entry: {sdate}</div>
          <div class="sig-date">Exit: {hold_until}</div>
          <div class="sig-date">Data: {data_date}</div>
          <div style="margin-top:10px;color:#6366f1;font-weight:600;">
            {MODE_LABELS.get(best_mode, best_mode)} ★
          </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_p:
        st.markdown("### Predicted Returns by ETF")
        st.caption("Highlighted = selected ETF")
        if pred_returns:
            st.plotly_chart(chart_returns_bar(pred_returns, rec_etf), use_container_width=True)
    
    # Performance metrics
    st.divider()
    st.markdown("### Walk-Forward Performance")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(mcard("Ann. Return", perf.get("annualised_return", 0)), unsafe_allow_html=True)
    c2.markdown(mcard("Sharpe Ratio", perf.get("sharpe_ratio", 0)), unsafe_allow_html=True)
    c3.markdown(mcard("Max Drawdown", perf.get("max_drawdown", 0), good=False), unsafe_allow_html=True)
    c4.markdown(mcard("Ann. Volatility", perf.get("annualised_vol", 0), good=False), unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div style="background:#fffbeb;border:1px solid #fbbf24;border-radius:10px;
                padding:14px 18px;font-size:.88rem;color:#78350f;margin-top:24px;">
    <strong>Disclaimer:</strong> Research purposes only. Not financial advice.
    Past performance does not guarantee future returns.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
