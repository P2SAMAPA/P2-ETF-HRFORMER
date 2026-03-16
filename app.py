"""
app.py  —  ETF Oracle · HRformer Signal Dashboard
Uses native Streamlit components for reliable rendering on Streamlit Cloud.
"""

import json, os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(
    page_title="ETF Oracle · HRformer",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Minimal CSS — only for custom HTML cards and overrides Streamlit won't do natively
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #f4f6f9; }
  [data-testid="block-container"]    { padding-top: 1.5rem; }
  .sig-card {
    background: #fff; border-radius: 14px;
    padding: 28px 32px;
    border: 1px solid #d1d9e0;
    margin-bottom: 4px;
  }
  .sig-label { font-size: 0.78rem; font-weight: 700; letter-spacing: 0.1em;
               text-transform: uppercase; color: #6b7280; margin-bottom: 6px; }
  .sig-ticker { font-size: 4rem; font-weight: 800; color: #111827;
                letter-spacing: -3px; line-height: 1; font-family: monospace; }
  .sig-conf   { font-size: 1.25rem; font-weight: 700; color: #15803d; margin-top: 10px; }
  .sig-date   { font-size: 0.9rem;  font-weight: 600; color: #374151; margin-top: 8px; font-family: monospace; }
  .cash-warn  { margin-top: 12px; padding: 10px 14px; background: #fef2f2;
                border: 1px solid #fca5a5; border-radius: 8px;
                font-size: 0.88rem; font-weight: 600; color: #991b1b; }
  .m-card {
    background: #fff; border-radius: 12px; padding: 20px 22px;
    border: 1px solid #d1d9e0; height: 100%;
  }
  .m-label { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em;
             text-transform: uppercase; color: #6b7280; margin-bottom: 8px; }
  .m-val-pos { font-size: 2.1rem; font-weight: 800; color: #15803d;
               font-family: monospace; letter-spacing: -1px; line-height: 1.1; }
  .m-val-neg { font-size: 2.1rem; font-weight: 800; color: #b91c1c;
               font-family: monospace; letter-spacing: -1px; line-height: 1.1; }
  .m-val-neu { font-size: 2.1rem; font-weight: 800; color: #111827;
               font-family: monospace; letter-spacing: -1px; line-height: 1.1; }
  .m-sub  { font-size: 0.82rem; font-weight: 500; color: #374151; margin-top: 6px; }
  .info-box { background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 10px;
              padding: 18px 22px; font-size: 0.92rem; color: #1e3a5f;
              line-height: 1.75; margin-bottom: 16px; }
  .info-box strong { color: #1e3a8a; }
  .disc { background: #fffbeb; border: 1px solid #fbbf24; border-radius: 10px;
          padding: 14px 18px; font-size: 0.88rem; color: #78350f;
          font-weight: 500; margin-top: 24px; }
</style>
""", unsafe_allow_html=True)

ETF_COLORS = {
    "TLT": "#3b82f6", "VNQ": "#22c55e", "SLV": "#94a3b8",
    "GLD": "#eab308", "LQD": "#a855f7", "HYG": "#f97316",
}
ETF_NAMES = {
    "TLT": "20Y Treasury", "VNQ": "Real Estate", "SLV": "Silver",
    "GLD": "Gold", "LQD": "IG Corporate", "HYG": "High Yield",
}
PLOT = dict(
    font=dict(family="DM Sans, sans-serif", size=13, color="#111827"),
    paper_bgcolor="white", plot_bgcolor="white",
    margin=dict(l=0, r=0, t=10, b=0),
)
GRID = dict(showgrid=True, gridcolor="#f1f5f9", linecolor="#e2e8f0", zeroline=False)


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


def mcard(label, value, sub="", good=True, neutral=False):
    if isinstance(value, float):
        cls = "m-val-neu" if neutral else ("m-val-pos" if (value >= 0) == good else "m-val-neg")
        val_str = f"{value:+.2%}"
    else:
        cls = "m-val-neu"
        val_str = str(value)
    return f"""<div class="m-card">
  <div class="m-label">{label}</div>
  <div class="{cls}">{val_str}</div>
  <div class="m-sub">{sub}</div>
</div>"""


def chart_equity(dates, equity, picks):
    df = pd.DataFrame({"d": pd.to_datetime(dates), "e": equity, "p": picks})
    fig = go.Figure()
    for etf, col in ETF_COLORS.items():
        s = df[df.p == etf]
        if s.empty: continue
        fig.add_trace(go.Scatter(x=s.d, y=s.e, mode="markers",
            marker=dict(size=5, color=col, opacity=0.6), name=etf))
    fig.add_trace(go.Scatter(x=df.d, y=df.e, mode="lines",
        line=dict(color="#1e293b", width=2.5), name="Portfolio"))
    fig.add_hline(y=1.0, line_dash="dot", line_color="#94a3b8", line_width=1)
    fig.update_layout(**PLOT, height=320,
        xaxis={**GRID}, yaxis={**GRID, "title": "Portfolio value (start = 1.0)"},
        hovermode="x unified",
        legend=dict(orientation="h", y=1.06, x=1, xanchor="right", yanchor="bottom"))
    return fig


def chart_probs(probs):
    tickers = list(probs.keys())
    vals    = [probs[t] for t in tickers]
    fig = go.Figure(go.Bar(
        x=tickers, y=vals,
        marker_color=[ETF_COLORS.get(t, "#3b82f6") for t in tickers],
        text=[f"{v:.1%}" for v in vals], textposition="outside",
        textfont=dict(size=14, color="#111827"),
    ))
    fig.update_layout(**PLOT, height=280,
        xaxis={**GRID, "tickfont": dict(size=14, color="#111827")},
        yaxis={**GRID, "range": [0, 1.05], "tickformat": ".0%",
               "tickfont": dict(size=13)},
        showlegend=False)
    fig.add_hline(y=0.5, line_dash="dot", line_color="#94a3b8", line_width=1)
    return fig


def chart_drawdown(equity, dates):
    eq = np.array(equity)
    dd = eq / np.maximum.accumulate(eq) - 1
    fig = go.Figure(go.Scatter(
        x=pd.to_datetime(dates), y=dd, mode="lines", fill="tozeroy",
        line=dict(color="#ef4444", width=1.5),
        fillcolor="rgba(239,68,68,0.1)"))
    fig.update_layout(**PLOT, height=220,
        xaxis={**GRID}, yaxis={**GRID, "tickformat": ".0%", "title": "Drawdown"},
        showlegend=False)
    return fig


def chart_picks(picks):
    counts = pd.Series(picks).value_counts().reindex(list(ETF_COLORS.keys()), fill_value=0)
    fig = go.Figure(go.Pie(
        labels=counts.index.tolist(), values=counts.values.tolist(),
        marker_colors=[ETF_COLORS[t] for t in counts.index],
        hole=0.52, textinfo="label+percent",
        textfont=dict(size=13, color="#111827"),
    ))
    fig.update_layout(**PLOT, height=280, showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10))
    return fig


# ════════════════════════════════════════════════════════
def main():
    data = load_signal()

    # ── Header ──────────────────────────────────────────
    st.markdown("## 📡 ETF Oracle")
    st.markdown("**Next-day trading signal powered by Hybrid Relational Transformer (HRformer)**  \n"
                "Xu et al., *Electronics* 2025 · Adapted for fixed-income ETFs")

    badge_row = "  ".join(
        [f"**{t}** · {ETF_NAMES[t]}" for t in ETF_NAMES]
    )
    cols_b = st.columns(6)
    for i, (t, name) in enumerate(ETF_NAMES.items()):
        cols_b[i].markdown(
            f"<div style='background:#f1f5f9;border:1px solid #cbd5e1;border-radius:20px;"
            f"padding:5px 12px;text-align:center;font-size:0.85rem;font-weight:600;"
            f"color:#1e293b'>{t} · {name}</div>",
            unsafe_allow_html=True)

    st.divider()

    if data is None:
        st.warning("⏳ No signal data yet. Run the GitHub Actions workflow to generate the first signal.")
        return

    sig    = data.get("signal", {})
    bt     = data.get("backtest", {})
    bts    = bt.get("summary", {})
    mets   = data.get("model_metrics", {})
    rec    = sig.get("recommended_etf", "—")
    conf   = sig.get("confidence", 0)
    sdate  = sig.get("signal_date", "—")
    gen_at = data.get("generated_at", "")[:10]
    will_trade = sig.get("will_trade", True)
    dates  = bt.get("dates", [])
    equity = bt.get("equity_curve", [])
    picks  = bt.get("picks", [])
    bt_label = f"{dates[0]} → {dates[-1]}" if dates else "—"

    # ── Signal row ───────────────────────────────────────
    col_s, col_p = st.columns([1, 2], gap="large")

    with col_s:
        warn = f'<div class="cash-warn">⚠ P(up) below 50% — hold cash tomorrow</div>' \
               if not will_trade else ""
        st.markdown(f"""<div class="sig-card">
          <div class="sig-label">Tomorrow's Pick</div>
          <div class="sig-ticker">{rec}</div>
          <div class="sig-conf">P(up) = {conf:.1%}</div>
          <div class="sig-date">Based on data to: {sdate}</div>
          <div class="sig-date">Generated: {gen_at}</div>
          {warn}
        </div>""", unsafe_allow_html=True)

    with col_p:
        st.markdown("### P(up) — All ETFs")
        st.caption("Model's estimated probability each ETF closes higher tomorrow. "
                   "The highest P(up) becomes the day's pick.")
        probs = sig.get("probabilities", {})
        if probs:
            st.plotly_chart(chart_probs(probs), width="stretch",
                            config={"displayModeBar": False})

    # ── Backtest metrics ──────────────────────────────────
    st.divider()
    st.markdown(f"### Backtest Performance "
                f"<span style='font-size:1rem;font-weight:400;color:#6b7280'>"
                f"· {bt_label} · long-only, 0.1% trading cost each side</span>",
                unsafe_allow_html=True)
    n_tr  = bts.get("num_trades", "—")
    n_dy  = bts.get("num_days", "—")
    t_rt  = bts.get("trade_rate", 0)
    hit   = bts.get("hit_rate", None)
    st.caption(f"Strategy: picks ETF with highest P(up) each day; only trades if P(up) > 50%, "
               f"otherwise holds cash. **Traded {n_tr} of {n_dy} days ({t_rt:.0%})**.")

    c1,c2,c3,c4,c5,c6 = st.columns(6, gap="small")
    c1.markdown(mcard("Total Return",    bts.get("total_return"),      bt_label),        unsafe_allow_html=True)
    c2.markdown(mcard("Ann. Return",     bts.get("annualised_return"), "Annualised"),     unsafe_allow_html=True)
    sr = bts.get("sharpe_ratio")
    sr_str = f"{sr:.2f}" if sr is not None else "—"
    sr_cls = "m-val-pos" if sr and sr > 0 else "m-val-neg"
    c3.markdown(f'<div class="m-card"><div class="m-label">Sharpe Ratio</div><div class="{sr_cls}">{sr_str}</div><div class="m-sub">Return ÷ risk (ann.)</div></div>',
                unsafe_allow_html=True)
    hr_str = f"{hit:.1%}" if hit is not None else "—"
    hr_cls = "m-val-pos" if hit and hit > 0.5 else "m-val-neg"
    c6.markdown(f'<div class="m-card"><div class="m-label">Pick Hit Rate</div>'
                f'<div class="{hr_cls}">{hr_str}</div>'
                f'<div class="m-sub">Correct direction on trades</div></div>',
                unsafe_allow_html=True)

    # ── Equity curve ──────────────────────────────────────
    st.divider()
    st.markdown(f"### Equity Curve")
    st.caption(f"Portfolio value over the backtest period ({bt_label}). "
               f"Dots show which ETF was held each day.")
    if dates and equity:
        st.plotly_chart(chart_equity(dates, equity, picks), width="stretch",
                        config={"displayModeBar": False})

    # ── Drawdown + picks ──────────────────────────────────
    col_d, col_pi = st.columns([2, 1], gap="large")
    with col_d:
        st.markdown("### Drawdown")
        st.caption("How far the portfolio fell from its previous peak at each point in time.")
        if equity:
            st.plotly_chart(chart_drawdown(equity, dates), width="stretch",
                            config={"displayModeBar": False})
    with col_pi:
        st.markdown("### Pick Distribution")
        st.caption(f"How often each ETF was selected during the backtest ({bt_label}).")
        if picks:
            st.plotly_chart(chart_picks(picks), width="stretch",
                            config={"displayModeBar": False})

    # ── Model metrics ──────────────────────────────────────
    if mets:
        st.divider()
        st.markdown("### Model Classification Metrics (Test Set)")
        st.markdown(f"""<div class="info-box">
These metrics measure how well the model predicts next-day <strong>up or down</strong>
direction for each ETF on the held-out test set ({bt_label}), before any trading costs.<br><br>
· <strong>Accuracy</strong> — % of days where the predicted direction was correct.<br>
· <strong>Precision</strong> — of days the model predicted "up", how often it was actually up.<br>
· <strong>Recall</strong> — of all actual up-days, how many did the model correctly flag.<br>
· <strong>F1</strong> — harmonic mean of precision and recall; the primary model quality score.<br><br>
A random coin-flip scores ~50% accuracy. Consistent accuracy above 52–53% is meaningful for daily ETF prediction.
</div>""", unsafe_allow_html=True)

        rows = []
        for ticker in ["TLT","VNQ","SLV","GLD","LQD","HYG"]:
            if ticker not in mets: continue
            m = mets[ticker]
            rows.append({"ETF": ticker,
                         "Accuracy":  f"{m.get('accuracy',0):.1%}",
                         "Precision": f"{m.get('precision',0):.1%}",
                         "Recall":    f"{m.get('recall',0):.1%}",
                         "F1":        f"{m.get('f1',0):.1%}"})
        if "aggregate" in mets:
            a = mets["aggregate"]
            rows.append({"ETF": "ALL (macro avg)",
                         "Accuracy":  f"{a.get('accuracy',0):.1%}",
                         "Precision": f"{a.get('precision',0):.1%}",
                         "Recall":    f"{a.get('recall',0):.1%}",
                         "F1":        f"{a.get('f1',0):.1%}"})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # ── Disclaimer ──────────────────────────────────────────
    st.markdown("""<div class="disc">
<strong>Disclaimer:</strong> This dashboard is for research and educational purposes only.
It does not constitute financial advice. Model outputs are statistical predictions and carry
no guarantee of future returns. Always conduct your own due diligence before making investment decisions.
</div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
