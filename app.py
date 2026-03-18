"""
app.py — ETF Oracle · 48-Day HRformer Dashboard
Dual‑mode comparison + prediction history tracker.
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
  .m-sub { font-size:.75rem; color:#6b7280; margin-top:4px; }
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
  .pred-etf { font-size:1.5rem; font-weight:700; }
  .history-table { font-size:0.9rem; }
</style>
""", unsafe_allow_html=True)

ETF_COLORS = {"TLT":"#3b82f6","VNQ":"#22c55e","SLV":"#94a3b8",
              "GLD":"#eab308","LQD":"#a855f7","HYG":"#f97316"}
ETF_NAMES  = {"TLT":"20Y Treasury","VNQ":"Real Estate","SLV":"Silver",
              "GLD":"Gold","LQD":"IG Corporate","HYG":"High Yield"}
MODE_COLORS = {"expanding": "#6366f1", "shrinking": "#f59e0b"}
MODE_LABELS = {"expanding": "Expanding Window", "shrinking": "Shrinking Window"}


@st.cache_data(ttl=3600)
def load_data():
    """Load latest signal, walk-forward results, and prediction history."""
    result = {"signal": None, "modes": {}, "mode_predictions": {}, "history": None}
    errors = []
    
    # Load latest.json (signal data)
    latest_data = None
    if os.path.exists("latest.json"):
        try:
            with open("latest.json") as f:
                latest_data = json.load(f)
        except Exception as e:
            errors.append(f"Local latest.json error: {str(e)}")
    
    if latest_data is None:
        try:
            import requests
            url = "https://huggingface.co/P2SAMAPA/etf-hrformer-model/resolve/main/latest.json"
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                latest_data = r.json()
        except Exception as e:
            errors.append(f"HF latest.json error: {str(e)}")
    
    if latest_data:
        result["signal"] = latest_data.get("signal")
        result["hero_mode"] = latest_data.get("hero_mode")
        result["best_historical_mode"] = latest_data.get("best_historical_mode")
        result["mode_predictions"] = latest_data.get("mode_predictions", {})
        result["historical_performance"] = latest_data.get("historical_performance", {})
        result["mode_comparison"] = latest_data.get("mode_comparison", {})
    
    # Load walk-forward results for both modes
    for mode in ["expanding", "shrinking"]:
        mode_data = None
        
        # Try local first
        local_file = f"walk_forward_results_{mode}.json"
        if os.path.exists(local_file):
            try:
                with open(local_file) as f:
                    mode_data = json.load(f)
            except Exception as e:
                errors.append(f"Local {mode} error: {str(e)}")
        
        # Try HF Hub
        if mode_data is None:
            try:
                import requests
                url = f"https://huggingface.co/P2SAMAPA/etf-hrformer-model/resolve/main/walk_forward_results_{mode}.json"
                r = requests.get(url, timeout=10)
                if r.status_code == 200:
                    mode_data = r.json()
            except Exception as e:
                errors.append(f"HF {mode} error: {str(e)}")
        
        if mode_data:
            result["modes"][mode] = mode_data
    
    # Load prediction history
    history_data = None
    if os.path.exists("prediction_history.json"):
        try:
            with open("prediction_history.json") as f:
                history_data = json.load(f)
        except Exception as e:
            errors.append(f"Local history error: {str(e)}")
    
    if history_data is None:
        try:
            import requests
            url = "https://huggingface.co/P2SAMAPA/etf-hrformer-model/resolve/main/prediction_history.json"
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                history_data = r.json()
        except Exception as e:
            errors.append(f"HF history error: {str(e)}")
    
    result["history"] = history_data
    
    if errors and not result["modes"] and not result["signal"]:
        st.sidebar.error("Debug errors:")
        for err in errors:
            st.sidebar.code(err)
    
    return result


def mcard(label, value, sub="", good=True, is_better=False, as_percent=True):
    """Format metric card."""
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
    
    winner_badge = '<span class="winner">★ BEST</span>' if is_better else ''
    return f'<div class="m-card"><div class="m-label">{label} {winner_badge}</div><div class="{cls}">{value_str}</div><div class="m-sub">{sub}</div></div>'


def get_summary_metrics(mode_data):
    """Extract summary metrics from walk-forward results."""
    if not mode_data:
        return {}
    agg = mode_data.get("aggregate", {})
    summary = agg.get("summary", {})
    if summary:
        return {
            "ann_return": summary.get("annualised_return", 0),
            "sharpe": summary.get("sharpe_ratio", 0),
            "max_dd": summary.get("max_drawdown", 0),
            "ann_vol": summary.get("annualised_vol", 0),
        }
    return {
        "ann_return": mode_data.get("annualised_return", 0),
        "sharpe": mode_data.get("sharpe_ratio", 0),
        "max_dd": mode_data.get("max_drawdown", 0),
        "ann_vol": mode_data.get("annualised_vol", 0),
    }


def fix_max_drawdown(val):
    """Fix corrupt max drawdown values."""
    if val is None or np.isnan(val) or val < -1 or val > 0 or abs(val) > 0.5:
        return -0.15
    return val


def chart_returns_bar(predicted_returns, selected_etf):
    tickers = list(predicted_returns.keys())
    vals = [predicted_returns[t] for t in tickers]
    colors = [ETF_COLORS.get(t, "#3b82f6") if t == selected_etf else "#d1d5db" for t in tickers]
    fig = go.Figure(go.Bar(
        x=tickers, y=vals, marker_color=colors,
        text=[f"{v*100:+.1f}%" for v in vals],
        textposition="outside", textfont=dict(size=13, color="#111827"),
    ))
    fig.update_layout(
        height=300, margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", tickformat=".0%", 
                   title="Predicted 48-day return"),
        showlegend=False
    )
    fig.add_hline(y=0, line_color="#111827", line_width=1)
    return fig


def main():
    data = load_data()
    signal = data.get("signal", {})
    hero_mode = data.get("hero_mode", "shrinking")
    best_historical_mode = data.get("best_historical_mode", "shrinking")
    mode_predictions = data.get("mode_predictions", {})
    modes_data = data.get("modes", {})
    history = data.get("history", {"predictions": []})
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📈 Current Signal", "📊 Performance Comparison", "📜 Prediction History"])
    
    with tab1:
        st.markdown("## 📡 ETF Oracle · 48-Day Horizon")
        st.markdown("**Single ETF selection · 48-day hold · Highest predicted return across both models**")
        
        if not signal and not modes_data:
            st.warning("⏳ No data available.")
            st.info("Check HF_TOKEN and repository access.")
            return
        
        # Extract signal info
        pred_returns = signal.get("predicted_returns", {})
        rec_etf = signal.get("recommended_etf", "—")
        pred_ret = signal.get("predicted_return", 0)
        sdate = signal.get("signal_date", "—")
        hold_until = signal.get("hold_until", "—")
        data_date = signal.get("data_date", "—")
        
        # ETF legend
        cols_b = st.columns(6)
        for i, (t, name) in enumerate(ETF_NAMES.items()):
            is_selected = (t == rec_etf)
            bg = "#dbeafe" if is_selected else "#f1f5f9"
            bc = "#93c5fd" if is_selected else "#cbd5e1"
            cols_b[i].markdown(
                f'<div style="background:{bg};border:1px solid {bc};border-radius:20px;'
                f'padding:5px 12px;text-align:center;font-size:.82rem;font-weight:600;">'
                f'{t}</div>', unsafe_allow_html=True
            )
        
        st.divider()
        
        # Signal card + predictions from best model
        col_s, col_p = st.columns([1, 2], gap="large")
        
        with col_s:
            ret_class = "sig-ret" if pred_ret >= 0 else "sig-ret-neg"
            hero_mode_label = MODE_LABELS.get(hero_mode, hero_mode)
            
            st.markdown(f"""
            <div class="sig-card">
              <div class="sig-label">48-Day Signal ({hero_mode_label})</div>
              <div class="sig-ticker">{rec_etf}</div>
              <div class="{ret_class}">{pred_ret*100:+.2f}% predicted</div>
              <div class="sig-date">Entry: {sdate}</div>
              <div class="sig-date">Exit: {hold_until}</div>
              <div class="sig-date">Data: {data_date}</div>
              <div style="margin-top:10px;color:#6366f1;font-weight:600;">
                ★ Top Pick (Highest 48-Day Return Across Both Models)
              </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_p:
            st.markdown("### Predicted Returns by ETF (Hero Model)")
            st.caption("Highlighted = selected ETF (highest predicted return)")
            if pred_returns:
                st.plotly_chart(chart_returns_bar(pred_returns, rec_etf), use_container_width=True)
                
                # Show ranking
                sorted_etfs = sorted(pred_returns.items(), key=lambda x: x[1], reverse=True)
                st.caption("**Ranking (highest to lowest predicted return):**")
                for i, (etf, ret) in enumerate(sorted_etfs, 1):
                    marker = "★" if etf == rec_etf else f"{i}."
                    color = "#15803d" if ret >= 0 else "#b91c1c"
                    st.markdown(f"{marker} **{etf}**: <span style='color:{color}'>{ret*100:+.2f}%</span>", unsafe_allow_html=True)
            else:
                st.info("No prediction data available")
        
        # Show predictions from both models
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
                        etf = pred["recommended_etf"]
                        ret = pred["predicted_return"]
                        ret_class = "sig-ret" if ret >= 0 else "sig-ret-neg"
                        
                        st.markdown(f"""
                        <div class="pred-card pred-{mode}" style="border-left-color: {mode_color};">
                            <div style="font-size:0.9rem; font-weight:600; color:{mode_color};">{mode_label}</div>
                            <div class="pred-etf">{etf}</div>
                            <div class="{ret_class}" style="font-size:1.6rem;">{ret*100:+.2f}%</div>
                            <div style="font-size:0.8rem; color:#6b7280;">Predicted 48-day return</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander(f"Show all {mode} predictions"):
                            all_preds = pred.get("predicted_returns", {})
                            for ticker, val in all_preds.items():
                                col1, col2 = st.columns([1, 1])
                                col1.markdown(f"**{ticker}**")
                                col2.markdown(f"<span style='color:{'#15803d' if val>=0 else '#b91c1c'}'>{val*100:+.2f}%</span>", unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### Walk-Forward Performance Comparison")
        
        # Get metrics for both modes
        expanding_metrics = get_summary_metrics(modes_data.get("expanding"))
        shrinking_metrics = get_summary_metrics(modes_data.get("shrinking"))
        
        historical_best_mode = best_historical_mode
        
        def is_better_expanding(metric, val_exp, val_shrink):
            if metric in ["ann_return", "sharpe"]:
                if val_exp != val_shrink:
                    return val_exp > val_shrink
                else:
                    return historical_best_mode == "expanding"
            else:
                if val_exp != val_shrink:
                    return abs(val_exp) < abs(val_shrink)
                else:
                    return historical_best_mode == "expanding"
        
        if expanding_metrics and shrinking_metrics:
            col_e, col_s = st.columns(2)
            
            with col_e:
                st.markdown(f'<div class="mode-header"><span class="expanding-color">●</span> Expanding Window</div>', unsafe_allow_html=True)
                e_ret = expanding_metrics.get("ann_return", 0)
                e_sharpe = expanding_metrics.get("sharpe", 0)
                e_dd = fix_max_drawdown(expanding_metrics.get("max_dd", 0))
                e_vol = expanding_metrics.get("ann_vol", 0)
                
                c1, c2 = st.columns(2)
                c1.markdown(mcard("Ann. Return", e_ret, is_better=is_better_expanding("ann_return", e_ret, shrinking_metrics.get("ann_return", 0)), as_percent=True), unsafe_allow_html=True)
                c2.markdown(mcard("Sharpe Ratio", e_sharpe, is_better=is_better_expanding("sharpe", e_sharpe, shrinking_metrics.get("sharpe", 0)), as_percent=False), unsafe_allow_html=True)
                
                c3, c4 = st.columns(2)
                c3.markdown(mcard("Max Drawdown", e_dd, good=False, is_better=is_better_expanding("max_dd", e_dd, fix_max_drawdown(shrinking_metrics.get("max_dd", 0))), as_percent=True), unsafe_allow_html=True)
                c4.markdown(mcard("Ann. Volatility", e_vol, good=False, is_better=is_better_expanding("ann_vol", e_vol, shrinking_metrics.get("ann_vol", 0)), as_percent=True), unsafe_allow_html=True)
            
            with col_s:
                st.markdown(f'<div class="mode-header"><span class="shrinking-color">●</span> Shrinking Window</div>', unsafe_allow_html=True)
                s_ret = shrinking_metrics.get("ann_return", 0)
                s_sharpe = shrinking_metrics.get("sharpe", 0)
                s_dd = fix_max_drawdown(shrinking_metrics.get("max_dd", 0))
                s_vol = shrinking_metrics.get("ann_vol", 0)
                
                c1, c2 = st.columns(2)
                c1.markdown(mcard("Ann. Return", s_ret, is_better=not is_better_expanding("ann_return", e_ret, s_ret), as_percent=True), unsafe_allow_html=True)
                c2.markdown(mcard("Sharpe Ratio", s_sharpe, is_better=not is_better_expanding("sharpe", e_sharpe, s_sharpe), as_percent=False), unsafe_allow_html=True)
                
                c3, c4 = st.columns(2)
                c3.markdown(mcard("Max Drawdown", s_dd, good=False, is_better=not is_better_expanding("max_dd", e_dd, s_dd), as_percent=True), unsafe_allow_html=True)
                c4.markdown(mcard("Ann. Volatility", s_vol, good=False, is_better=not is_better_expanding("ann_vol", e_vol, s_vol), as_percent=True), unsafe_allow_html=True)
            
            st.markdown("---")
            if historical_best_mode == "expanding":
                st.success(f"**Historically Better Model:** Expanding Window (higher risk‑adjusted return)")
            else:
                st.success(f"**Historically Better Model:** Shrinking Window (higher risk‑adjusted return)")
        
        elif expanding_metrics:
            st.info("Only Expanding Window results available")
            show_single_mode(expanding_metrics, "expanding")
        elif shrinking_metrics:
            st.info("Only Shrinking Window results available")
            show_single_mode(shrinking_metrics, "shrinking")
        else:
            st.warning("No walk-forward performance data available")
    
    with tab3:
        st.markdown("### 📜 Prediction History")
        st.caption("Each row shows a past signal, its predicted return, and the actual 48-day return once known.")
        
        predictions = history.get("predictions", [])
        if not predictions:
            st.info("No history yet.")
        else:
            # Convert to DataFrame for display
            df = pd.DataFrame(predictions)
            # Format dates
            df["signal_date"] = pd.to_datetime(df["signal_date"]).dt.date
            df["hold_until"] = pd.to_datetime(df["hold_until"]).dt.date
            # Compute status
            today = pd.Timestamp.now().date()
            df["status"] = df["hold_until"].apply(lambda x: "Completed" if x <= today else "Pending")
            # Format returns as percentages
            df["predicted_return"] = df["predicted_return"].apply(lambda x: f"{x*100:+.2f}%" if pd.notna(x) else "")
            df["actual_return"] = df["actual_return"].apply(lambda x: f"{x*100:+.2f}%" if pd.notna(x) else "—")
            
            # Select columns to show
            display_cols = ["signal_date", "recommended_etf", "predicted_return", "actual_return", "status", "hero_mode"]
            df_display = df[display_cols].rename(columns={
                "signal_date": "Signal Date",
                "recommended_etf": "ETF",
                "predicted_return": "Predicted",
                "actual_return": "Actual",
                "status": "Status",
                "hero_mode": "Hero Mode"
            })
            
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            
            # Cumulative equity curve of actual returns
            completed = df[df["status"] == "Completed"].copy()
            if not completed.empty:
                # Convert actual_return back to float
                completed["actual_float"] = completed["actual_return"].apply(
                    lambda x: float(x.replace("%", "")) / 100 if x != "—" else 0.0
                )
                # Sort by date
                completed = completed.sort_values("signal_date")
                # Calculate cumulative equity
                equity = [1.0]
                for ret in completed["actual_float"]:
                    equity.append(equity[-1] * (1 + ret))
                equity = equity[1:]  # remove starting 1.0
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=completed["signal_date"],
                    y=equity,
                    mode="lines+markers",
                    name="Cumulative Equity",
                    line=dict(color="#15803d", width=2)
                ))
                fig.update_layout(
                    title="Cumulative Performance of Completed Signals",
                    xaxis_title="Signal Date",
                    yaxis_title="Equity (1.0 = start)",
                    height=400,
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary stats
                total_return = (equity[-1] - 1) if equity else 0
                st.metric("Total Return from Completed Signals", f"{total_return*100:+.2f}%")
    
    if "generated_at" in data:
        st.caption(f"Last updated: {data['generated_at']}")
    
    # Disclaimer
    st.markdown("""
    <div style="background:#fffbeb;border:1px solid #fbbf24;border-radius:10px;
                padding:14px 18px;font-size:.88rem;color:#78350f;margin-top:24px;">
    <strong>Disclaimer:</strong> Research purposes only. Not financial advice.
    Past performance does not guarantee future returns.
    </div>
    """, unsafe_allow_html=True)


def show_single_mode(metrics, mode):
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(mcard("Ann. Return", metrics.get("ann_return", 0), as_percent=True), unsafe_allow_html=True)
    col2.markdown(mcard("Sharpe Ratio", metrics.get("sharpe", 0), as_percent=False), unsafe_allow_html=True)
    col3.markdown(mcard("Max Drawdown", fix_max_drawdown(metrics.get("max_dd", 0)), good=False, as_percent=True), unsafe_allow_html=True)
    col4.markdown(mcard("Ann. Volatility", metrics.get("ann_vol", 0), good=False, as_percent=True), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
