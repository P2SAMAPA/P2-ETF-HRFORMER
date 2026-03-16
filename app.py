"""
app.py — ETF Oracle · Walk-Forward HRformer Dashboard
Shows both expanding and fixed window backtest results side by side.
"""

import json, os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="ETF Oracle · HRformer", page_icon="📡",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #f4f6f9; }
  [data-testid="block-container"]    { padding-top: 1.5rem; }
  .sig-card { background:#fff; border-radius:14px; padding:26px 30px;
              border:1px solid #d1d9e0; margin-bottom:4px; }
  .sig-label { font-size:.75rem; font-weight:700; letter-spacing:.1em;
               text-transform:uppercase; color:#6b7280; margin-bottom:8px; }
  .sig-ticker { font-size:4rem; font-weight:800; color:#111827;
                letter-spacing:-3px; line-height:1; font-family:monospace; }
  .sig-conf   { font-size:1.2rem; font-weight:700; color:#15803d; margin-top:8px; }
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
  .info-box strong { color:#1e3a8a; }
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
    for path in ["latest.json"]:
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
                if d.get("signal", {}).get("recommended_etf"):
                    return d
    try:
        import requests
        r = requests.get("https://huggingface.co/P2SAMAPA/etf-hrformer-model"
                         "/resolve/main/latest.json", timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def mcard(label, value, sub="", good=True, neutral=False):
    if isinstance(value, float):
        cls = "m-neu" if neutral else ("m-pos" if (value>=0)==good else "m-neg")
        val_str = f"{value:+.2%}"
    else:
        cls, val_str = "m-neu", str(value) if value is not None else "—"
    return f'<div class="m-card"><div class="m-label">{label}</div><div class="{cls}">{val_str}</div><div class="m-sub">{sub}</div></div>'


def chart_equity_both(data, best_mode):
    fig = go.Figure()
    for mode in ["expanding", "fixed"]:
        if mode not in data: continue
        agg = data[mode].get("aggregate", {})
        eq  = agg.get("equity", [])
        dts = agg.get("dates", [])
        if not eq or not dts: continue
        n = min(len(eq), len(dts))
        dash = "solid" if mode == best_mode else "dot"
        width = 2.5 if mode == best_mode else 1.5
        ann_r = agg["summary"]["annualised_return"]
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(dts[:n]), y=eq[:n], mode="lines",
            line=dict(color=MODE_COLORS[mode], width=width, dash=dash),
            name=f"{MODE_LABELS[mode]} ({ann_r:+.1%} ann.)",
        ))
    fig.add_hline(y=1.0, line_dash="dot", line_color="#94a3b8", line_width=1)
    fig.update_layout(**PLOT, height=360, margin=dict(l=0,r=0,t=10,b=0),
        xaxis={**GRID}, yaxis={**GRID, "title":"Portfolio value (start=1.0)"},
        hovermode="x unified",
        legend=dict(orientation="h", y=1.06, x=1, xanchor="right", yanchor="bottom"))
    return fig


def chart_equity_single(mode_data, color):
    agg = mode_data.get("aggregate", {})
    eq  = agg.get("equity", [])
    dts = agg.get("dates", [])
    pks = agg.get("picks", [])
    if not eq or not dts: return None
    n   = min(len(eq), len(dts), len(pks))
    df  = pd.DataFrame({"d":pd.to_datetime(dts[:n]), "e":eq[:n], "p":pks[:n]})
    fig = go.Figure()
    for etf, col in ETF_COLORS.items():
        s = df[df.p==etf]
        if s.empty: continue
        fig.add_trace(go.Scatter(x=s.d, y=s.e, mode="markers",
            marker=dict(size=4, color=col, opacity=0.5), name=etf))
    fig.add_trace(go.Scatter(x=df.d, y=df.e, mode="lines",
        line=dict(color=color, width=2.5), name="Portfolio"))
    fig.add_hline(y=1.0, line_dash="dot", line_color="#94a3b8", line_width=1)
    fig.update_layout(**PLOT, height=300, margin=dict(l=0,r=0,t=10,b=0),
        xaxis={**GRID}, yaxis={**GRID, "title":"Portfolio value"},
        hovermode="x unified",
        legend=dict(orientation="h", y=1.06, x=1, xanchor="right", yanchor="bottom"))
    return fig


def chart_drawdown(mode_data, color):
    eq = mode_data.get("aggregate", {}).get("equity", [])
    if not eq: return None
    e  = np.array(eq)
    dd = e / np.maximum.accumulate(e) - 1
    dts = mode_data.get("aggregate", {}).get("dates", [])
    x = pd.to_datetime(dts[:len(dd)]) if dts else list(range(len(dd)))
    r = int(color[1:3],16)
    g = int(color[3:5],16)
    b = int(color[5:7],16)
    fig = go.Figure(go.Scatter(x=x, y=dd, mode="lines", fill="tozeroy",
        line=dict(color=color, width=1.5),
        fillcolor=f"rgba({r},{g},{b},0.12)",
    ))
    fig.update_layout(**PLOT, height=200, margin=dict(l=0,r=0,t=10,b=0),
        xaxis={**GRID}, yaxis={**GRID,"tickformat":".0%","title":"Drawdown"},
        showlegend=False)
    return fig


def chart_probs(probs):
    tickers = list(probs.keys())
    vals    = [probs[t] for t in tickers]
    fig = go.Figure(go.Bar(
        x=tickers, y=vals,
        marker_color=[ETF_COLORS.get(t,"#3b82f6") for t in tickers],
        text=[f"{v:.1%}" for v in vals], textposition="outside",
        textfont=dict(size=14, color="#111827"),
    ))
    fig.update_layout(**PLOT, height=260, margin=dict(l=0,r=0,t=10,b=0),
        xaxis={**GRID,"tickfont":dict(size=14,color="#111827")},
        yaxis={**GRID,"range":[0,1.05],"tickformat":".0%"},
        showlegend=False)
    fig.add_hline(y=0.5, line_dash="dot", line_color="#94a3b8", line_width=1)
    return fig


def chart_fold_returns(folds, color):
    labels = [f"Fold {f['fold']}\n{f['test_range'][:7]}" for f in folds]
    vals   = [f["summary"]["annualised_return"] for f in folds]
    colors = ["#15803d" if v>=0 else "#b91c1c" for v in vals]
    fig = go.Figure(go.Bar(x=list(range(len(folds))), y=vals,
        marker_color=colors,
        text=[f"{v:+.1%}" for v in vals], textposition="outside",
        textfont=dict(size=12)))
    fig.update_layout(**PLOT, height=220, margin=dict(l=0,r=0,t=10,b=0),
        xaxis={**GRID,"tickvals":list(range(len(folds))),
               "ticktext":[f"F{f['fold']}" for f in folds]},
        yaxis={**GRID,"tickformat":".0%","title":"Ann. return"},
        showlegend=False)
    fig.add_hline(y=0, line_dash="dot", line_color="#94a3b8", line_width=1)
    return fig


def metrics_row(summary, cols, period=""):
    c1,c2,c3,c4,c5 = cols
    c1.markdown(mcard("Total Return",    summary.get("total_return"),      period),        unsafe_allow_html=True)
    c2.markdown(mcard("Ann. Return",     summary.get("annualised_return"), "Annualised"),  unsafe_allow_html=True)
    sr = summary.get("sharpe_ratio")
    sr_cls = "m-pos" if sr and sr>0 else "m-neg"
    sr_str = f"{sr:.2f}" if sr is not None else "—"
    c3.markdown(f'<div class="m-card"><div class="m-label">Sharpe Ratio</div>'
                f'<div class="{sr_cls}">{sr_str}</div>'
                f'<div class="m-sub">Return ÷ risk</div></div>', unsafe_allow_html=True)
    c4.markdown(mcard("Max Drawdown",    summary.get("max_drawdown"),    "Peak-to-trough", good=False), unsafe_allow_html=True)
    c5.markdown(mcard("Ann. Volatility", summary.get("annualised_vol"),  "Daily std×√252", good=False), unsafe_allow_html=True)


def main():
    data = load_data()

    # Header
    st.markdown("## 📡 ETF Oracle")
    st.markdown("**Next-day trading signal · Hybrid Relational Transformer (HRformer) · Walk-Forward Validation**")
    cols_b = st.columns(6)
    for i,(t,name) in enumerate(ETF_NAMES.items()):
        rec = data.get("signal",{}).get("recommended_etf") if data else None
        bg  = "#dbeafe" if t==rec else "#f1f5f9"
        bc  = "#93c5fd" if t==rec else "#cbd5e1"
        fc  = "#1e40af" if t==rec else "#1e293b"
        cols_b[i].markdown(
            f'<div style="background:{bg};border:1px solid {bc};border-radius:20px;'
            f'padding:5px 12px;text-align:center;font-size:.82rem;font-weight:600;'
            f'color:{fc}">{t} · {name}</div>', unsafe_allow_html=True)

    st.divider()

    if data is None:
        st.warning("⏳ No signal data yet. Run the GitHub Actions workflow first.")
        return

    sig       = data.get("signal", {})
    best_mode = data.get("best_mode", "fixed")
    rec       = sig.get("recommended_etf", "—")
    conf      = sig.get("confidence", 0)
    sdate     = sig.get("signal_date", "—")
    data_date = sig.get("data_date", "—")
    gen_at    = data.get("generated_at","")[:10]
    will_trade = sig.get("will_trade", True)

    # Signal row
    col_s, col_p = st.columns([1,2], gap="large")
    with col_s:
        st.markdown(f"""<div class="sig-card">
          <div class="sig-label">Signal for</div>
          <div class="sig-ticker">{rec}</div>
          <div class="sig-conf">P(up) = {conf:.1%}</div>
          <div class="sig-date">Predicting: {sdate}</div>
          <div class="sig-date">Based on data to: {data_date}</div>
          <div class="sig-date">Generated: {gen_at}</div>
          <div class="sig-date" style="margin-top:10px;color:#6366f1">
            Using: {MODE_LABELS.get(best_mode,best_mode)} ★</div>
        </div>""", unsafe_allow_html=True)
        if not will_trade:
            st.warning("⚠️ P(up) below threshold — model suggests holding cash")

    with col_p:
        st.markdown("### P(up) — All ETFs")
        st.caption("Probability each ETF closes higher on the signal date. Highest bar = today's pick.")
        probs = sig.get("probabilities", {})
        if probs:
            st.plotly_chart(chart_probs(probs), width="stretch",
                            config={"displayModeBar":False})

    # Walk-forward comparison
    st.divider()
    st.markdown("### Walk-Forward Backtest — Both Modes")
    st.caption(f"Each fold: train → test on next 1 year. Signals only when P(up) > 65%. "
               f"0.05% trading cost each side. "
               f"**{'★ ' + MODE_LABELS.get(best_mode,'') + ' selected for live signal (higher ann. return)'}**")

    # Side-by-side summary metrics
    for mode in ["expanding", "fixed"]:
        if mode not in data: continue
        agg  = data[mode].get("aggregate", {})
        summ = agg.get("summary", {})
        badge = '<span class="best-badge">★ Live signal</span>' if mode==best_mode else ""
        st.markdown(f"#### {MODE_LABELS[mode]}{badge} &nbsp;"
                    f"<span style='font-size:.9rem;font-weight:400;color:#6b7280'>"
                    f"· {summ.get('num_folds','?')} folds</span>",
                    unsafe_allow_html=True)
        c1,c2,c3,c4,c5 = st.columns(5, gap="small")
        metrics_row(summ, (c1,c2,c3,c4,c5))

    st.markdown("#### Combined equity curves")
    st.caption("Solid line = selected mode. Dotted = other mode.")
    st.plotly_chart(chart_equity_both(data, best_mode), width="stretch",
                    config={"displayModeBar":False})

    # Per-mode detail
    st.divider()
    col_exp, col_fix = st.columns(2, gap="large")

    for col, mode, color in [(col_exp,"expanding","#6366f1"),(col_fix,"fixed","#f59e0b")]:
        if mode not in data:
            col.info(f"No {mode} results.")
            continue
        mode_data = data[mode]
        agg   = mode_data.get("aggregate",{})
        folds = mode_data.get("folds",[])
        badge = " ★" if mode==best_mode else ""
        with col:
            st.markdown(f"**{MODE_LABELS[mode]}{badge}**")
            fig_eq = chart_equity_single(mode_data, color)
            if fig_eq:
                st.plotly_chart(fig_eq, width="stretch", config={"displayModeBar":False})

            st.caption("Annual return per fold")
            fig_f = chart_fold_returns(folds, color)
            if fig_f:
                st.plotly_chart(fig_f, width="stretch", config={"displayModeBar":False})

            st.caption("Drawdown")
            fig_dd = chart_drawdown(mode_data, color)
            if fig_dd:
                st.plotly_chart(fig_dd, width="stretch", config={"displayModeBar":False})

    # Fold details table
    st.divider()
    st.markdown("### Fold-by-Fold Results")
    st.markdown(f"""<div class="info-box">
Each row is one fold. The model was trained on all data up to the cutoff,
then tested on the following 1-year period — it never saw the test data during training.
<strong>Expanding</strong> grows the training window back to 2008.
<strong>Fixed</strong> always uses the most recent 2 years only.
The mode with the higher aggregate annualised return drives the live signal.
</div>""", unsafe_allow_html=True)

    rows = []
    for mode in ["expanding","fixed"]:
        if mode not in data: continue
        for f in data[mode].get("folds",[]):
            s = f["summary"]
            rows.append({
                "Mode":       MODE_LABELS[mode],
                "Fold":       f["fold"],
                "Train":      f.get("train_range","—"),
                "Test period":f.get("test_range","—"),
                "Ann. return":f"{s['annualised_return']:+.1%}",
                "Sharpe":     f"{s['sharpe_ratio']:.2f}",
                "Max DD":     f"{s['max_drawdown']:.1%}",
                "Val F1":     f"{f.get('best_val_f1',0):.3f}",
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.markdown("""<div class="disc">
<strong>Disclaimer:</strong> Research and educational purposes only.
Not financial advice. Past model performance does not guarantee future returns.
</div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
