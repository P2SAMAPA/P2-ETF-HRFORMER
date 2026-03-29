"""
app.py — ETF Oracle · 48-Day HRformer Dashboard
Tab 1 & 2: Fixed Income ETFs (TLT, VNQ, SLV, GLD, LQD, HYG)
Tab 3 & 4: Equity ETFs     (SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, XME, GDX, IWM)
Tab 5:     FI   Prediction History
Tab 6:     EQ   Prediction History
"""

import json, os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="ETF Oracle · HRformer 48D", page_icon="📡",
                   layout="wide", initial_sidebar_state="collapsed")

# ─────────────────────────────────────────────
# Styles
# ─────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #f4f6f9; }
  .sig-card { background:#fff; border-radius:14px; padding:26px 30px;
              border:1px solid #d1d9e0; margin-bottom:4px; }
  .sig-label { font-size:.75rem; font-weight:700; letter-spacing:.1em;
               text-transform:uppercase; color:#6b7280; margin-bottom:8px; }
  .sig-ticker { font-size:4rem; font-weight:800; color:#111827;
                letter-spacing:-3px; line-height:1; font-family:monospace; }
  .sig-ret    { font-size:1.4rem; font-weight:700; color:#15803d; margin-top:8px; }
  .sig-ret-neg{ font-size:1.4rem; font-weight:700; color:#b91c1c; margin-top:8px; }
  .sig-date   { font-size:.88rem; font-weight:600; color:#374151;
                margin-top:6px; font-family:monospace; }
  .m-card { background:#fff; border-radius:12px; padding:18px 20px;
            border:1px solid #d1d9e0; }
  .m-label{ font-size:.72rem; font-weight:700; letter-spacing:.08em;
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
  .m-sub  { font-size:.75rem; color:#6b7280; margin-top:4px; }
  .mode-header { font-size:1.1rem; font-weight:700; color:#374151;
                 margin-bottom:12px; padding-bottom:8px; border-bottom:2px solid #e5e7eb; }
  .expanding-color { color: #6366f1; }
  .shrinking-color { color: #f59e0b; }
  .winner { background:#dcfce7; border:1px solid #86efac; border-radius:6px;
            padding:2px 8px; font-size:.75rem; font-weight:700; color:#15803d; }
  .pred-card { background:#f9fafb; border-radius:10px; padding:15px;
               border-left:4px solid; margin-bottom:10px; }
  .pred-expanding { border-left-color: #6366f1; }
  .pred-shrinking { border-left-color: #f59e0b; }
  .pred-etf  { font-size:1.5rem; font-weight:700; }

  /* Equity accent colour overrides */
  .eq-header { border-bottom: 3px solid #0ea5e9; }
  .eq-accent { color: #0ea5e9; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
FI_ETF_COLORS = {"TLT":"#3b82f6","VNQ":"#22c55e","SLV":"#94a3b8",
                 "GLD":"#eab308","LQD":"#a855f7","HYG":"#f97316"}
FI_ETF_NAMES  = {"TLT":"20Y Treasury","VNQ":"Real Estate","SLV":"Silver",
                 "GLD":"Gold","LQD":"IG Corporate","HYG":"High Yield"}

EQ_ETF_COLORS = {
    "SPY":"#2563eb","QQQ":"#7c3aed","XLK":"#0891b2","XLF":"#059669",
    "XLE":"#d97706","XLV":"#dc2626","XLI":"#9333ea","XLY":"#db2777",
    "XLP":"#16a34a","XLU":"#ca8a04","XME":"#64748b","GDX":"#b45309",
    "IWM":"#0f766e",
}
EQ_ETF_NAMES  = {
    "SPY":"S&P 500","QQQ":"NASDAQ 100","XLK":"Technology","XLF":"Financials",
    "XLE":"Energy","XLV":"Health Care","XLI":"Industrials","XLY":"Consumer Disc",
    "XLP":"Consumer Staples","XLU":"Utilities","XME":"Metal & Mining",
    "GDX":"Gold Miners","IWM":"Russell 2000",
}

MODE_COLORS = {"expanding":"#6366f1","shrinking":"#f59e0b"}
MODE_LABELS = {"expanding":"Expanding Window","shrinking":"Shrinking Window"}


# ─────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_fi_data():
    return _load_universe(
        latest_file="latest.json",
        history_file="prediction_history.json",
        wf_prefix="walk_forward_results",
        hf_prefix="walk_forward_results",
    )


@st.cache_data(ttl=3600)
def load_equity_data():
    return _load_universe(
        latest_file="latest_equity.json",
        history_file="prediction_history_equity.json",
        wf_prefix="walk_forward_results_equity",
        hf_prefix="walk_forward_results_equity",
    )


def _load_universe(latest_file, history_file, wf_prefix, hf_prefix):
    result = {"signal": None, "modes": {}, "mode_predictions": {}, "history": None}
    errors = []

    # ── latest.json ──────────────────────────────────────────────────────────
    latest_data = _try_local_then_hf(latest_file, errors)
    if latest_data:
        result["signal"]               = latest_data.get("signal")
        result["hero_mode"]            = latest_data.get("hero_mode")
        result["best_historical_mode"] = latest_data.get("best_historical_mode")
        result["mode_predictions"]     = latest_data.get("mode_predictions", {})
        result["historical_performance"] = latest_data.get("historical_performance", {})
        result["mode_comparison"]      = latest_data.get("mode_comparison", {})

    # ── walk-forward results ──────────────────────────────────────────────────
    for mode in ["expanding", "shrinking"]:
        fname = f"{wf_prefix}_{mode}.json"
        data  = _try_local_then_hf(fname, errors)
        if data:
            result["modes"][mode] = data

    # ── prediction history ────────────────────────────────────────────────────
    history_data = _try_local_then_hf(history_file, errors)
    if history_data is None:
        history_data = {"predictions": []}
    result["history"] = history_data

    if errors and not result["modes"] and not result["signal"]:
        st.sidebar.error("Debug errors:")
        for err in errors:
            st.sidebar.code(err)

    return result


def _try_local_then_hf(filename, errors):
    """Try local file first, fall back to HF Hub."""
    if os.path.exists(filename):
        try:
            with open(filename) as f:
                return json.load(f)
        except Exception as e:
            errors.append(f"Local {filename}: {e}")

    try:
        import requests
        url = f"https://huggingface.co/P2SAMAPA/etf-hrformer-model/resolve/main/{filename}"
        r   = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        errors.append(f"HF {filename}: {e}")

    return None


# ─────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────

def mcard(label, value, sub="", good=True, is_better=False, as_percent=True):
    if isinstance(value, (int, float)):
        if abs(value) > 1e6 or np.isnan(value) or np.isinf(value):
            value_str, cls = "Error", "m-neu"
        else:
            cls = "m-pos" if (value >= 0) == good else "m-neg"
            if as_percent and abs(value) < 10:
                value_str = f"{value:+.2%}"
            else:
                value_str = f"{value:+.2f}"
    else:
        cls, value_str = "m-neu", str(value) if value else "—"

    badge = '<span class="winner">★ BEST</span>' if is_better else ""
    return (f'<div class="m-card">'
            f'<div class="m-label">{label} {badge}</div>'
            f'<div class="{cls}">{value_str}</div>'
            f'<div class="m-sub">{sub}</div>'
            f'</div>')


def get_summary_metrics(mode_data):
    if not mode_data:
        return {}
    agg     = mode_data.get("aggregate", {})
    summary = agg.get("summary", {})
    base    = summary if summary else mode_data
    return {
        "ann_return": base.get("annualised_return", 0),
        "sharpe":     base.get("sharpe_ratio",      0),
        "max_dd":     base.get("max_drawdown",       0),
        "ann_vol":    base.get("annualised_vol",     0),
    }


def fix_max_drawdown(val):
    if val is None or np.isnan(val) or val < -1 or val > 0 or abs(val) > 0.5:
        return -0.15
    return val


def chart_returns_bar(predicted_returns, selected_etf, etf_colors):
    tickers = list(predicted_returns.keys())
    vals    = [predicted_returns[t] for t in tickers]
    colors  = [etf_colors.get(t, "#3b82f6") if t == selected_etf else "#d1d5db"
               for t in tickers]
    fig = go.Figure(go.Bar(
        x=tickers, y=vals, marker_color=colors,
        text=[f"{v*100:+.1f}%" for v in vals],
        textposition="outside",
        textfont=dict(size=12, color="#111827"),
    ))
    fig.update_layout(
        height=320, margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9",
                   tickformat=".0%", title="Predicted 48-day return"),
        showlegend=False,
        plot_bgcolor="white", paper_bgcolor="white",
    )
    fig.add_hline(y=0, line_color="#111827", line_width=1)
    return fig


# ─────────────────────────────────────────────
# Render helpers
# ─────────────────────────────────────────────

def render_signal_tab(data, etf_colors, etf_names, universe_label="FI"):
    signal           = data.get("signal") or {}
    hero_mode        = data.get("hero_mode", "shrinking")
    mode_predictions = data.get("mode_predictions", {})

    pred_returns = signal.get("predicted_returns", {})
    rec_etf      = signal.get("recommended_etf", "—")
    pred_ret     = signal.get("predicted_return", 0)
    entry_date   = signal.get("entry_date", "—")
    target_48    = signal.get("target_48_date", "—")
    data_date    = signal.get("data_date", "—")

    if not signal and not data.get("modes"):
        st.warning("⏳ No data available. Check HF_TOKEN and repository access.")
        return

    # ETF legend pills
    n_etfs = len(etf_names)
    cols_b = st.columns(n_etfs)
    for i, (t, name) in enumerate(etf_names.items()):
        is_sel = (t == rec_etf)
        bg = "#dbeafe" if is_sel else "#f1f5f9"
        bc = "#93c5fd" if is_sel else "#cbd5e1"
        cols_b[i].markdown(
            f'<div style="background:{bg};border:1px solid {bc};border-radius:20px;'
            f'padding:5px 10px;text-align:center;font-size:.78rem;font-weight:600;">'
            f'{t}</div>', unsafe_allow_html=True
        )

    st.divider()

    col_s, col_p = st.columns([1, 2], gap="large")

    with col_s:
        ret_class      = "sig-ret" if pred_ret >= 0 else "sig-ret-neg"
        hero_mode_label = MODE_LABELS.get(hero_mode, hero_mode)
        st.markdown(f"""
        <div class="sig-card">
          <div class="sig-label">48-Day Signal · {universe_label} ({hero_mode_label})</div>
          <div class="sig-ticker">{rec_etf}</div>
          <div class="{ret_class}">{pred_ret*100:+.2f}% predicted</div>
          <div class="sig-date">Entry (buy at open): {entry_date}</div>
          <div class="sig-date">Target 48-day date: {target_48}</div>
          <div class="sig-date">Data up to: {data_date}</div>
          <div style="margin-top:10px;color:#6366f1;font-weight:600;">
            ★ Top Pick · Highest 48-Day Return Across Both Models
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_p:
        st.markdown("### Predicted Returns by ETF (Hero Model)")
        st.caption("Highlighted = selected ETF")
        if pred_returns:
            st.plotly_chart(
                chart_returns_bar(pred_returns, rec_etf, etf_colors),
                use_container_width=True,
            )
            sorted_etfs = sorted(pred_returns.items(), key=lambda x: x[1], reverse=True)
            st.caption("**Ranking (highest → lowest predicted return):**")
            for i, (etf, ret) in enumerate(sorted_etfs, 1):
                marker = "★" if etf == rec_etf else f"{i}."
                color  = "#15803d" if ret >= 0 else "#b91c1c"
                st.markdown(
                    f"{marker} **{etf}** ({etf_names.get(etf, '')}): "
                    f"<span style='color:{color}'>{ret*100:+.2f}%</span>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No prediction data available")

    # Both-model breakdown
    if mode_predictions:
        st.divider()
        st.markdown("### What Each Model Predicts for the Next 48 Days")
        cols = st.columns(2)
        for idx, mode in enumerate(["expanding", "shrinking"]):
            pred = mode_predictions.get(mode)
            if pred and "recommended_etf" in pred:
                with cols[idx]:
                    mode_label = MODE_LABELS.get(mode, mode)
                    mode_color = MODE_COLORS.get(mode, "#000")
                    etf        = pred["recommended_etf"]
                    ret        = pred["predicted_return"]
                    ret_class  = "sig-ret" if ret >= 0 else "sig-ret-neg"
                    st.markdown(f"""
                    <div class="pred-card pred-{mode}" style="border-left-color:{mode_color};">
                        <div style="font-size:.9rem;font-weight:600;color:{mode_color};">{mode_label}</div>
                        <div class="pred-etf">{etf}</div>
                        <div class="{ret_class}" style="font-size:1.6rem;">{ret*100:+.2f}%</div>
                        <div style="font-size:.8rem;color:#6b7280;">Predicted 48-day return</div>
                    </div>
                    """, unsafe_allow_html=True)
                    with st.expander(f"Show all {mode} predictions"):
                        for ticker, val in pred.get("predicted_returns", {}).items():
                            c1, c2 = st.columns([1, 1])
                            c1.markdown(f"**{ticker}** — {etf_names.get(ticker,'')}")
                            c2.markdown(
                                f"<span style='color:{'#15803d' if val>=0 else '#b91c1c'}'>"
                                f"{val*100:+.2f}%</span>",
                                unsafe_allow_html=True,
                            )


def render_performance_tab(data, universe_label="FI"):
    modes_data          = data.get("modes", {})
    best_historical_mode = data.get("best_historical_mode", "shrinking")

    expanding_metrics = get_summary_metrics(modes_data.get("expanding"))
    shrinking_metrics = get_summary_metrics(modes_data.get("shrinking"))

    def expanding_wins(metric, ve, vs):
        if metric in ("ann_return", "sharpe"):
            return ve > vs
        return abs(ve) < abs(vs)

    if expanding_metrics and shrinking_metrics:
        col_e, col_s = st.columns(2)

        for col, mode, metrics, is_exp in [
            (col_e, "expanding", expanding_metrics, True),
            (col_s, "shrinking", shrinking_metrics, False),
        ]:
            with col:
                color = "#6366f1" if is_exp else "#f59e0b"
                label = "Expanding Window" if is_exp else "Shrinking Window"
                st.markdown(
                    f'<div class="mode-header"><span style="color:{color}">●</span> {label}</div>',
                    unsafe_allow_html=True,
                )
                ret = metrics.get("ann_return", 0)
                sharpe = metrics.get("sharpe", 0)
                dd     = fix_max_drawdown(metrics.get("max_dd", 0))
                vol    = metrics.get("ann_vol", 0)

                e_ret = expanding_metrics.get("ann_return", 0)
                e_sh  = expanding_metrics.get("sharpe", 0)
                e_dd  = fix_max_drawdown(expanding_metrics.get("max_dd", 0))
                e_vol = expanding_metrics.get("ann_vol", 0)
                s_ret = shrinking_metrics.get("ann_return", 0)
                s_sh  = shrinking_metrics.get("sharpe", 0)
                s_dd  = fix_max_drawdown(shrinking_metrics.get("max_dd", 0))
                s_vol = shrinking_metrics.get("ann_vol", 0)

                ret_win    = expanding_wins("ann_return", e_ret, s_ret)
                sharpe_win = expanding_wins("sharpe",     e_sh,  s_sh)
                dd_win     = expanding_wins("max_dd",     e_dd,  s_dd)
                vol_win    = expanding_wins("ann_vol",    e_vol, s_vol)

                c1, c2 = st.columns(2)
                c1.markdown(mcard("Ann. Return",    ret,    is_better=ret_win    == is_exp, as_percent=True),  unsafe_allow_html=True)
                c2.markdown(mcard("Sharpe Ratio",   sharpe, is_better=sharpe_win == is_exp, as_percent=False), unsafe_allow_html=True)
                c3, c4 = st.columns(2)
                c3.markdown(mcard("Max Drawdown",   dd,  good=False, is_better=dd_win  == is_exp, as_percent=True), unsafe_allow_html=True)
                c4.markdown(mcard("Ann. Volatility",vol, good=False, is_better=vol_win == is_exp, as_percent=True), unsafe_allow_html=True)

        st.markdown("---")
        winner = "Expanding Window" if best_historical_mode == "expanding" else "Shrinking Window"
        st.success(f"**Historically Better Model ({universe_label}):** {winner}")

    elif expanding_metrics:
        st.info("Only Expanding Window results available")
        _single_mode_cards(expanding_metrics)
    elif shrinking_metrics:
        st.info("Only Shrinking Window results available")
        _single_mode_cards(shrinking_metrics)
    else:
        st.warning(f"No {universe_label} walk-forward performance data available yet.")
        st.info("Run `train_equity.py --mode expanding` and `train_equity.py --mode shrinking` first.")


def _single_mode_cards(metrics):
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(mcard("Ann. Return",    metrics.get("ann_return", 0), as_percent=True),  unsafe_allow_html=True)
    c2.markdown(mcard("Sharpe Ratio",   metrics.get("sharpe",     0), as_percent=False), unsafe_allow_html=True)
    c3.markdown(mcard("Max Drawdown",   fix_max_drawdown(metrics.get("max_dd", 0)), good=False, as_percent=True), unsafe_allow_html=True)
    c4.markdown(mcard("Ann. Volatility",metrics.get("ann_vol",    0), good=False, as_percent=True), unsafe_allow_html=True)


def render_history_tab(data, universe_label="FI"):
    history     = (data.get("history") or {})
    predictions = history.get("predictions", [])

    st.markdown(f"### 📜 {universe_label} Prediction History (Daily Rebalancing)")
    st.caption("Each row shows a past signal: predicted 48-day return and actual 1-day return once available.")

    if not predictions:
        st.info(f"No {universe_label} history yet.")
        return

    df = pd.DataFrame(predictions)
    df["entry_date"]    = pd.to_datetime(df["entry_date"]).dt.date
    df["target_48_date"] = pd.to_datetime(df["target_48_date"]).dt.date

    completed_df = df[df["actual_return"].notna()].copy()

    # Period selector
    periods = {"Last 30 Days": 30, "Last 90 Days": 90, "Last 365 Days": 365, "All Time": None}
    sel     = st.selectbox(f"Period ({universe_label})", list(periods.keys()), index=0,
                           key=f"period_{universe_label}")
    if periods[sel] is not None:
        cutoff      = pd.Timestamp.now().date() - pd.Timedelta(days=periods[sel])
        filtered_df = completed_df[completed_df["entry_date"] >= cutoff]
    else:
        filtered_df = completed_df

    # Metrics
    if not filtered_df.empty:
        filtered_df = filtered_df.copy()
        filtered_df["actual_float"] = filtered_df["actual_return"].apply(
            lambda x: float(x.replace("%", "")) / 100
            if isinstance(x, str) and x != "—" else 0.0
        )
        n_trades  = len(filtered_df)
        win_rate  = (filtered_df["actual_float"] > 0).mean()
        avg_ret   = filtered_df["actual_float"].mean()
        cum_ret   = (1 + filtered_df["actual_float"]).prod() - 1
        std_ret   = filtered_df["actual_float"].std()
        sharpe    = (avg_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Trades",        n_trades)
        c2.metric("Win Rate",      f"{win_rate:.2%}")
        c3.metric("Avg Return",    f"{avg_ret:.2%}")
        c4.metric("Total Return",  f"{cum_ret:.2%}")
        c5.metric("Sharpe (ann.)", f"{sharpe:.2f}")
    else:
        st.info(f"No completed {universe_label} trades in the selected period.")

    st.divider()

    # Full table
    display = df.copy()
    display["status"]           = display["actual_return"].apply(lambda x: "Completed" if pd.notna(x) else "Pending")
    display["predicted_return"] = display["predicted_return"].apply(lambda x: f"{x*100:+.2f}%")
    display["actual_return"]    = display["actual_return"].apply(
        lambda x: f"{x*100:+.2f}%" if pd.notna(x) else "—"
    )
    cols_show = ["entry_date","recommended_etf","predicted_return","actual_return","status","hero_mode"]
    st.dataframe(
        display[cols_show].rename(columns={
            "entry_date":       "Entry Date",
            "recommended_etf":  "ETF",
            "predicted_return": "Predicted 48d",
            "actual_return":    "Actual 1d",
            "status":           "Status",
            "hero_mode":        "Hero Mode",
        }),
        use_container_width=True, hide_index=True,
    )

    # Equity curve
    if not completed_df.empty:
        completed_df = completed_df.copy()
        completed_df["actual_float"] = completed_df["actual_return"].apply(
            lambda x: float(x.replace("%", "")) / 100
            if isinstance(x, str) and x != "—" else 0.0
        )
        completed_df = completed_df.sort_values("entry_date")
        equity = [1.0]
        for r in completed_df["actual_float"]:
            equity.append(equity[-1] * (1 + r))
        equity = equity[1:]

        accent = "#15803d" if universe_label == "FI" else "#0ea5e9"
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=completed_df["entry_date"], y=equity,
            mode="lines+markers", name=f"{universe_label} Cumulative Equity",
            line=dict(color=accent, width=2),
        ))
        fig.update_layout(
            title=f"{universe_label} Cumulative Performance — All Completed Signals (1-day holds)",
            xaxis_title="Entry Date", yaxis_title="Equity (1.0 = start)",
            height=400, margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.metric("Total Return (All Completed)", f"{(equity[-1]-1)*100:+.2f}%")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    fi_data = load_fi_data()
    eq_data = load_equity_data()

    # ── Six tabs ────────────────────────────────────────────────────────────
    tab_fi_sig, tab_fi_perf, tab_eq_sig, tab_eq_perf, tab_fi_hist, tab_eq_hist = st.tabs([
        "📈 FI Signal",
        "📊 FI Performance",
        "🚀 Equity Signal",
        "📉 Equity Performance",
        "📜 FI History",
        "📜 EQ History",
    ])

    # ── Fixed Income ────────────────────────────────────────────────────────
    with tab_fi_sig:
        st.markdown("## 📡 Fixed Income ETF Oracle · 48-Day Horizon")
        st.markdown("*TLT · VNQ · SLV · GLD · LQD · HYG*")
        st.markdown("**Daily rebalancing: buy the ETF with highest predicted return, hold one day.**")
        render_signal_tab(fi_data, FI_ETF_COLORS, FI_ETF_NAMES, universe_label="FI")

    with tab_fi_perf:
        st.markdown("### Fixed Income — Walk-Forward Performance Comparison")
        render_performance_tab(fi_data, universe_label="FI")

    # ── Equity ──────────────────────────────────────────────────────────────
    with tab_eq_sig:
        st.markdown("## 🚀 Equity ETF Oracle · 48-Day Horizon")
        st.markdown(
            "*SPY · QQQ · XLK · XLF · XLE · XLV · XLI · XLY · XLP · XLU · XME · GDX · IWM*"
        )
        st.markdown("**Daily rebalancing: buy the ETF with highest predicted return, hold one day.**")
        render_signal_tab(eq_data, EQ_ETF_COLORS, EQ_ETF_NAMES, universe_label="EQ")

    with tab_eq_perf:
        st.markdown("### Equity — Walk-Forward Performance Comparison")
        render_performance_tab(eq_data, universe_label="EQ")

    # ── History ─────────────────────────────────────────────────────────────
    with tab_fi_hist:
        render_history_tab(fi_data, universe_label="FI")

    with tab_eq_hist:
        render_history_tab(eq_data, universe_label="EQ")

    # ── Footer ───────────────────────────────────────────────────────────────
    gen_at = fi_data.get("generated_at") or eq_data.get("generated_at")
    if gen_at:
        st.caption(f"Last updated: {gen_at}")

    st.markdown("""
    <div style="background:#fffbeb;border:1px solid #fbbf24;border-radius:10px;
                padding:14px 18px;font-size:.88rem;color:#78350f;margin-top:24px;">
    <strong>Disclaimer:</strong> Research purposes only. Not financial advice.
    Past performance does not guarantee future returns.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
