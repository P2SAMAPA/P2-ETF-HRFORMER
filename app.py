"""
app.py — ETF Oracle · 48-Day HRformer Dashboard
Shows 48-day return predictions and walk-forward results.
"""

import json, os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="ETF Oracle · HRformer 48D", page_icon="📡",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #f4f6f9; }
  [data-testid="block-container"]    { padding-top: 1.5rem; }
  .sig-card { background:#fff; border-radius:14px; padding:26px 30px;
              border:1px solid #d1d9e0; margin-bottom:4px; }
  .sig-label { font-size:.75rem; font-weight:700; letter-spacing:.1em;
               text-transform:uppercase; color:#6b7280; margin-bottom:8px; }
  .sig-ticker { font-size:3rem; font-weight:800; color:#111827;
                letter-spacing:-2px; line-height:1; font-family:monospace; }
  .sig-ret   { font-size:1.2rem; font-weight:700; color:#15803d; margin-top:8px; }
  .sig-ret-neg { font-size:1.2rem; font-weight:700; color:#b91c1c; margin-top:8px; }
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
  .m-sub  { font-size:.8rem; font-weight:500; color:#374151; margin-top:5px; }
  .best-badge { display:inline-block; background:#dcfce7; border:1px solid #86efac;
                border-radius:20px; padding:3px 14px; font-size:.8rem;
                font-weight:700; color:#15803d; margin-left:10px; }
  .info-box { background:#eff6ff; border:1px solid #bfdbfe; border-radius:10px;
              padding:16px 20px; font-size:.9rem; color:#1e3a5f;
              line-height:1.75; margin-bottom:16px; }
  .disc { background:#fffbeb; border:1px solid #fbbf24; border-radius:10px;
          padding:14px 18px; font-size:.88rem; color:#78350f;
          font-weight:500; margin-top:24px; }
</style>
""", unsafe_allow_html=True)

ETF_COLORS = {"TLT":"#3b82f6","VNQ":"#22c55e","SLV":"#94a3b8",
              "GLD":"#eab308","LQD":"#a855f7","HYG":"#f97316"}
ETF_NAMES  = {"TLT":"20Y Treasury","VNQ":"Real Estate","SLV":"Silver",
              "GLD":"Gold","LQD":"IG Corporate","HYG":"High Yield"}
PLOT = dict(font=dict(family="DM Sans, sans-serif", size=13, color="#111827"),
            paper_bgcolor="white", plot_bgcolor="white")
GRID = dict(showgrid=True, gridcolor="#f1f5f9", linecolor="#e2e8f0", zeroline=False)
MODE_COLORS = {"expanding": "#6366f1", "fixed": "#f59e0b"}
MODE_LABELS = {"expanding": "Expanding window", "fixed": "Fixed 2-year window"}


@st.cache_data(ttl=3600)
def load_data():
    """Load latest signal and walk-forward results."""
    # Try local file first
    if os.path.exists("latest.json"):
        with open("latest.json") as f:
            return json.load(f)
    
    # Try HuggingFace
    try:
        import requests
        r = requests.get(
            "https://huggingface.co/P2SAMAPA/etf-hrformer-model/resolve/main/latest.json",
            timeout=10
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    
    return None


def mcard(label, value, sub="", good=True, neutral=False):
    """Metric card HTML."""
    if isinstance(value, (int, float)):
        # Handle crazy large numbers
        if abs(value) > 1e6:
            value_str = "Error"
            cls = "m-neu"
        else:
            cls = "m-neu" if neutral else ("m-pos" if (value>=0)==good else "m-neg")
            value_str = f"{value:+.2%}" if abs(value) < 10 else f"{value:.2f}"
    else:
        cls, value_str = "m-neu", str(value) if value is not None else "—"
    
    return f'<div class="m-card"><div class="m-label">{label}</div><div class="{cls}">{value_str}</div><div class="m-sub">{sub}</div></div>'


def chart_returns_bar(predicted_returns):
    """Bar chart of predicted 48-day returns."""
    tickers = list(predicted_returns.keys())
    vals = [predicted_returns[t] for t in tickers]
    
    colors = ["#15803d" if v >= 0 else "#b91c1c" for v in vals]
    
    fig = go.Figure(go.Bar(
        x=tickers,
        y=vals,
        marker_color=colors,
        text=[f"{v*100:+.1f}%" for v in vals],
        textposition="outside",
        textfont=dict(size=13, color="#111827"),
    ))
    
    fig.update_layout(
        **PLOT,
        height=300,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis={**GRID, "tickfont": dict(size=14)},
        yaxis={**GRID, "tickformat": ".0%", "title": "Predicted 48-day return"},
        showlegend=False
    )
    
    fig.add_hline(y=0, line_dash="solid", line_color="#111827", line_width=1)
    
    return fig


def chart_equity_both(data, best_mode):
    """Combined equity curves for both modes."""
    fig = go.Figure()
    
    for mode in ["expanding", "fixed"]:
        if mode not in data:
            continue
        
        agg = data[mode].get("aggregate", {})
        eq = agg.get("equity", [])
        
        if not eq or len(eq) < 2:
            continue
        
        # Sanity check equity values
        if any(e <= 0 or e > 1e6 for e in eq):
            st.warning(f"Invalid equity values in {mode} mode")
            continue
        
        ann_r = agg.get("summary", {}).get("annualised_return", 0)
        
        dash = "solid" if mode == best_mode else "dot"
        width = 3 if mode == best_mode else 1.5
        
        fig.add_trace(go.Scatter(
            y=eq,
            mode="lines",
            line=dict(color=MODE_COLORS[mode], width=width, dash=dash),
            name=f"{MODE_LABELS[mode]} ({ann_r:+.1%} ann.)",
        ))
    
    fig.add_hline(y=1.0, line_dash="dot", line_color="#94a3b8", line_width=1)
    
    fig.update_layout(
        **PLOT,
        height=400,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis={**GRID, "title": "Trading Period (48-day steps)"},
        yaxis={**GRID, "title": "Portfolio Value", "tickformat": ".2f"},
        hovermode="x unified",
        legend=dict(orientation="h", y=1.06, x=1, xanchor="right")
    )
    
    return fig


def chart_fold_returns(folds, color):
    """Bar chart of per-fold returns."""
    if not folds:
        return None
    
    vals = [f.get("summary", {}).get("annualised_return", 0) for f in folds]
    colors = ["#15803d" if v >= 0 else "#b91c1c" for v in vals]
    
    fig = go.Figure(go.Bar(
        x=list(range(len(folds))),
        y=vals,
        marker_color=colors,
        text=[f"{v*100:+.0f}%" for v in vals],
        textposition="outside",
        textfont=dict(size=11)
    ))
    
    fig.update_layout(
        **PLOT,
        height=220,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis={**GRID, "tickvals": list(range(len(folds))), 
               "ticktext": [f"F{f.get('fold', i+1)}" for i, f in enumerate(folds)]},
        yaxis={**GRID, "tickformat": ".0%", "title": "Ann. return"},
        showlegend=False
    )
    
    fig.add_hline(y=0, line_dash="dot", line_color="#94a3b8", line_width=1)
    
    return fig


def main():
    data = load_data()
    
    # Header
    st.markdown("## 📡 ETF Oracle · 48-Day Horizon")
    st.markdown("**48-day return prediction · Hybrid Relational Transformer · Walk-Forward Validation**")
    
    # ETF legend
    cols_b = st.columns(6)
    for i, (t, name) in enumerate(ETF_NAMES.items()):
        cols_b[i].markdown(
            f'<div style="background:#f1f5f9;border:1px solid #cbd5e1;border-radius:20px;'
            f'padding:5px 12px;text-align:center;font-size:.82rem;font-weight:600;">'
            f'{t} · {name}</div>',
            unsafe_allow_html=True
        )
    
    st.divider()
    
    if data is None:
        st.warning("⏳ No signal data. Run infer.py first.")
        return
    
    # Extract data
    sig = data.get("signal", {})
    best_mode = data.get("best_mode", "fixed")
    perf = data.get("performance", {})
    
    # Signal section
    rec_etfs = sig.get("recommended_etfs", [])
    pred_returns = sig.get("predicted_returns", {})
    sdate = sig.get("signal_date", "—")
    hold_until = sig.get("hold_until", "—")
    data_date = sig.get("data_date", "—")
    
    col_s, col_p = st.columns([1, 2], gap="large")
    
    with col_s:
        # Show top 2 picks
        picks_str = " + ".join(rec_etfs) if rec_etfs else "—"
        
        # Get predicted returns for picks
        if rec_etfs and pred_returns:
            pick_rets = [pred_returns.get(e, 0) for e in rec_etfs]
            avg_ret = sum(pick_rets) / len(pick_rets)
            ret_class = "sig-ret" if avg_ret >= 0 else "sig-ret-neg"
            ret_str = f"{avg_ret*100:+.2f}%"
        else:
            ret_class = "sig-ret"
            ret_str = "—"
        
        st.markdown(f"""
        <div class="sig-card">
          <div class="sig-label">48-Day Signal</div>
          <div class="sig-ticker">{picks_str}</div>
          <div class="{ret_class}">Avg. predicted: {ret_str}</div>
          <div class="sig-date">Entry: {sdate}</div>
          <div class="sig-date">Exit: {hold_until}</div>
          <div class="sig-date">Based on data to: {data_date}</div>
          <div style="margin-top:10px;color:#6366f1;font-weight:600;">
            Using: {MODE_LABELS.get(best_mode, best_mode)} ★
          </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_p:
        st.markdown("### Predicted 48-Day Returns by ETF")
        if pred_returns:
            st.plotly_chart(chart_returns_bar(pred_returns), use_container_width=True)
    
    # Performance metrics
    st.divider()
    st.markdown("### Walk-Forward Performance")
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(mcard("Ann. Return", perf.get("annualised_return", 0)), unsafe_allow_html=True)
    c2.markdown(mcard("Sharpe Ratio", perf.get("sharpe_ratio", 0), good=True), unsafe_allow_html=True)
    c3.markdown(mcard("Max Drawdown", perf.get("max_drawdown", 0), good=False), unsafe_allow_html=True)
    c4.markdown(mcard("Ann. Volatility", perf.get("annualised_vol", 0), good=False), unsafe_allow_html=True)
    # Fix crazy total_return
    tot_ret = perf.get("total_return", 0)
    if abs(tot_ret) > 1e6:
        tot_ret = "Error"
    c5.markdown(mcard("Total Return", tot_ret), unsafe_allow_html=True)
    
    # Equity curves
    st.markdown("#### Equity Curves")
    st.caption("Solid = selected mode, Dotted = other mode")
    fig_eq = chart_equity_both(data, best_mode)
    if fig_eq:
        st.plotly_chart(fig_eq, use_container_width=True)
    
    # Per-mode details
    st.divider()
    col_exp, col_fix = st.columns(2)
    
    for col, mode in [(col_exp, "expanding"), (col_fix, "fixed")]:
        with col:
            if mode not in data:
                st.info(f"No {mode} data")
                continue
            
            mode_data = data[mode]
            agg = mode_data.get("aggregate", {})
            folds = mode_data.get("folds", [])
            
            badge = " ★" if mode == best_mode else ""
            st.markdown(f"**{MODE_LABELS[mode]}{badge}**")
            
            # Fold returns chart
            fig_f = chart_fold_returns(folds, MODE_COLORS[mode])
            if fig_f:
                st.plotly_chart(fig_f, use_container_width=True)
            
            # Summary stats
            summ = agg.get("summary", {})
            st.caption(f"Folds: {summ.get('num_folds', 0)} | "
                      f"Ann. Return: {summ.get('annualised_return', 0):.2%} | "
                      f"Sharpe: {summ.get('sharpe_ratio', 0):.2f}")
    
    # Fold details table
    st.divider()
    st.markdown("### Fold-by-Fold Results")
    
    rows = []
    for mode in ["expanding", "fixed"]:
        if mode not in data:
            continue
        for f in data[mode].get("folds", []):
            s = f.get("summary", {})
            rows.append({
                "Mode": MODE_LABELS[mode],
                "Fold": f.get("fold", "—"),
                "Train": f.get("train_range", "—"),
                "Test": f.get("test_range", "—"),
                "Ann. Return": f"{s.get('annualised_return', 0):+.1%}",
                "Sharpe": f"{s.get('sharpe_ratio', 0):.2f}",
                "Max DD": f"{s.get('max_drawdown', 0):.1%}",
            })
    
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    
    # Disclaimer
    st.markdown("""
    <div class="disc">
    <strong>Disclaimer:</strong> Research and educational purposes only. 
    Not financial advice. Past performance does not guarantee future returns.
    48-day predictions are experimental and may not reflect actual market behavior.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
