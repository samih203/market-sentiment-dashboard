import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

from pipeline import (
    run_pipeline,
    fetch_ohlc,
    fetch_prices,
    compute_fear_greed,
    fear_greed_label,
    get_sentiment_pipeline,
    COINS,
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="SignalDesk — Crypto Sentiment Terminal",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=DM+Sans:wght@400;500;600&display=swap');

:root {
    --bg:        #0d0f14;
    --bg2:       #13161e;
    --bg3:       #1a1e29;
    --border:    rgba(255,255,255,0.07);
    --border2:   rgba(255,255,255,0.13);
    --green:     #00d4a8;
    --green-dim: rgba(0,212,168,0.10);
    --red:       #ff4d6a;
    --red-dim:   rgba(255,77,106,0.10);
    --amber:     #f5a623;
    --amber-dim: rgba(245,166,35,0.10);
    --blue:      #4d9fff;
    --text:      #e2e4ed;
    --muted:     #6b7280;
    --mono:      'JetBrains Mono', monospace;
    --sans:      'DM Sans', system-ui, sans-serif;
}

html, body, [class*="css"] { font-family: var(--sans); }
.stApp { background: var(--bg) !important; color: var(--text); }
section[data-testid="stSidebar"] { background: var(--bg2) !important; }
.block-container { padding: 1.6rem 2rem 4rem !important; max-width: 1500px; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

h1, h2, h3 { font-family: var(--mono) !important; letter-spacing: -0.5px; }
h1 { font-size: 1.4rem !important; font-weight: 700 !important; color: var(--green) !important; }
h2 { font-size: 1rem  !important; font-weight: 600 !important; color: var(--text) !important; }
h3 { font-size: 0.8rem !important; font-weight: 600 !important; color: var(--muted) !important;
     text-transform: uppercase; letter-spacing: 1.5px; margin-top: 1.2rem !important; }

[data-testid="metric-container"] {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 0.8rem 1rem !important;
    font-family: var(--mono) !important;
}
[data-testid="metric-container"] label {
    font-size: 0.6rem !important; letter-spacing: 1.2px !important;
    color: var(--muted) !important; text-transform: uppercase;
    font-family: var(--sans) !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.2rem !important; font-weight: 700 !important;
    font-family: var(--mono) !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] svg { display: none; }
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 0.68rem !important; font-family: var(--sans) !important;
}
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important; border-radius: 8px !important;
}
.stButton > button {
    background: var(--bg3) !important; color: var(--muted) !important;
    border: 1px solid var(--border2) !important; border-radius: 6px !important;
    font-family: var(--mono) !important; font-size: 0.72rem !important;
    letter-spacing: 0.5px !important; padding: 0.35rem 0.9rem !important;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    border-color: var(--green) !important; color: var(--green) !important;
    background: var(--green-dim) !important;
}
.stDownloadButton > button {
    background: var(--bg3) !important; color: var(--blue) !important;
    border: 1px solid rgba(77,159,255,0.25) !important; border-radius: 6px !important;
    font-family: var(--mono) !important; font-size: 0.72rem !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg2) !important; border-radius: 8px !important;
    border: 1px solid var(--border) !important; padding: 3px !important; gap: 2px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; border-radius: 5px !important;
    color: var(--muted) !important; font-family: var(--mono) !important;
    font-size: 0.7rem !important; letter-spacing: 0.8px !important;
    padding: 5px 14px !important;
}
.stTabs [aria-selected="true"] {
    background: var(--bg3) !important; color: var(--text) !important;
    border: 1px solid var(--border2) !important;
}
.stSelectbox > div > div, .stRadio > div {
    background: var(--bg2) !important; border-color: var(--border2) !important;
    color: var(--text) !important; border-radius: 6px !important;
}
hr { border-color: var(--border) !important; margin: 1rem 0 !important; }

.price-chip {
    background: var(--bg2); border: 1px solid var(--border);
    border-radius: 8px; padding: 0.75rem 1rem;
    position: relative; overflow: hidden; cursor: pointer;
    transition: border-color 0.15s ease;
}
.price-chip:hover { border-color: var(--border2); }
.price-chip.selected { border-color: var(--green) !important;
                       background: rgba(0,212,168,0.05) !important; }
.price-chip::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
}
.price-chip.up::before    { background: var(--green); }
.price-chip.down::before  { background: var(--red); }
.chip-label { font-size: 0.58rem; color: var(--muted); letter-spacing: 1.2px;
               text-transform: uppercase; font-family: var(--sans); margin-bottom: 3px; }
.chip-value { font-size: 1.1rem; font-weight: 700; font-family: var(--mono); }
.chip-delta { font-size: 0.65rem; margin-top: 2px; font-family: var(--sans); }
.chip-delta.up   { color: var(--green); }
.chip-delta.down { color: var(--red); }
.chip-sig  { font-size: 0.6rem; margin-top: 2px; font-family: var(--mono);
              color: var(--muted); }

.live-badge {
    display: inline-block; background: var(--green-dim); color: var(--green);
    border: 1px solid rgba(0,212,168,0.22); border-radius: 20px;
    padding: 2px 10px; font-size: 0.62rem; letter-spacing: 1px;
    font-family: var(--mono); vertical-align: middle; margin-left: 10px;
}
.alert-banner {
    background: var(--amber-dim); border: 1px solid rgba(245,166,35,0.22);
    border-radius: 8px; padding: 0.5rem 0.9rem;
    font-size: 0.72rem; font-family: var(--sans); color: var(--amber);
    margin-bottom: 0.6rem;
}
.coin-pill {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 0.65rem; font-family: var(--mono); font-weight: 600;
    letter-spacing: 0.5px; margin-right: 4px;
}
.pill-bull { background: rgba(0,212,168,0.12); color: #00d4a8;
              border: 1px solid rgba(0,212,168,0.2); }
.pill-bear { background: rgba(255,77,106,0.12); color: #ff4d6a;
              border: 1px solid rgba(255,77,106,0.2); }
.pill-neu  { background: rgba(107,114,128,0.15); color: #9ca3af;
              border: 1px solid rgba(107,114,128,0.2); }
</style>
""", unsafe_allow_html=True)

# ============================================================
# PLOTLY THEME
# ============================================================
PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono, monospace", color="#6b7280", size=11),
    margin=dict(l=44, r=16, t=28, b=40),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False,
               showline=False, tickfont=dict(size=10)),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False,
               showline=False, tickfont=dict(size=10)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="#1a1e29", bordercolor="#2a2e3d",
                    font=dict(family="JetBrains Mono", size=11, color="#e2e4ed")),
)

# Per-coin accent colors
COIN_COLORS = {
    "BTC":  "#f7931a",
    "ETH":  "#627eea",
    "SOL":  "#9945ff",
    "BNB":  "#f0b90b",
    "XRP":  "#00aae4",
    "AVAX": "#e84142",
    "DOGE": "#c2a633",
    "ADA":  "#0033ad",
}
GREEN  = "#00d4a8"
RED    = "#ff4d6a"
AMBER  = "#f5a623"
BLUE   = "#4d9fff"
MUTED  = "#6b7280"

# ============================================================
# LOAD MODEL (session-cached)
# ============================================================
@st.cache_resource(show_spinner="Loading FinBERT…")
def load_model():
    return get_sentiment_pipeline()

load_model()

# ============================================================
# PIPELINE CACHE (60s TTL for NLP work)
# ============================================================
@st.cache_data(ttl=60, show_spinner="Analysing news…")
def load_pipeline():
    return run_pipeline()

try:
    _df_cached, _prices_cached, market_signal, coin_momentum = load_pipeline()
except Exception as e:
    st.error(f"Pipeline error: {e}")
    st.stop()

df = _df_cached.copy(deep=True) if not _df_cached.empty else _df_cached

# Live prices every rerun (fast, no NLP)
try:
    prices = fetch_prices(ttl=20)
except Exception:
    prices = _prices_cached

# ============================================================
# SESSION STATE
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(
        columns=["time"] + [f"{t}_price" for t in COINS] + [f"{t}_signal" for t in COINS]
    )
if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "selected_coin" not in st.session_state:
    st.session_state.selected_coin = "BTC"

# Build new history row
new_row_data = {"time": pd.Timestamp.now()}
for ticker in COINS:
    new_row_data[f"{ticker}_price"]  = float(prices.get(ticker, {}).get("price", 0))
    new_row_data[f"{ticker}_signal"] = float(coin_momentum.get(ticker, 0))

new_row = pd.DataFrame([new_row_data])
_h = st.session_state.history

# Only append when something actually changed
_changed = _h.empty or any(
    float(prices.get(t, {}).get("price", 0)) != float(_h.get(f"{t}_price", pd.Series([0])).iloc[-1])
    for t in ["BTC", "ETH", "SOL"]
)
if _changed:
    st.session_state.history = pd.concat([_h, new_row], ignore_index=True).tail(200)

history = st.session_state.history.copy()

# ============================================================
# ALERT DETECTION
# ============================================================
ALERT_THRESHOLD = st.session_state.get("signal_thresh", 0.45)
if not df.empty:
    high_impact = df[df["signal"].abs() > ALERT_THRESHOLD]
    for _, row in high_impact.head(3).iterrows():
        coin_tag = f"[{row['coin']}] " if row.get("coin") and row["coin"] != "MACRO" else ""
        msg = f"{coin_tag}{'🟢' if row['signal'] > 0 else '🔴'} \"{row['title'][:75]}…\" ({row['signal']:+.2f})"
        if msg not in st.session_state.alerts:
            st.session_state.alerts.append(msg)
            st.toast(msg, icon="⚡")
st.session_state.alerts = st.session_state.alerts[-8:]

# ============================================================
# HELPERS
# ============================================================
def get_signal_label(sig):
    if   sig >  0.15: return "BUY",        GREEN
    elif sig < -0.15: return "SELL",       RED
    else:             return "NEUTRAL",    AMBER

def position_size(score):
    if   score >  0.3: return  1.0
    elif score >  0.1: return  0.5
    elif score < -0.3: return -1.0
    elif score < -0.1: return -0.5
    else:              return  0.0

def build_coin_history(ticker):
    """Extract price + signal history for a single coin."""
    p_col = f"{ticker}_price"
    s_col = f"{ticker}_signal"
    if p_col not in history.columns:
        return pd.DataFrame()
    h = history[["time", p_col, s_col]].copy().rename(
        columns={p_col: "price", s_col: "signal"}
    )
    h = h[h["price"] > 0].copy()
    if len(h) < 2:
        return h
    h["returns"]     = h["price"].pct_change()
    p_range          = h["price"].max() - h["price"].min()
    h["price_norm"]  = (h["price"] - h["price"].min()) / (p_range + 1e-9)
    h["signal_norm"] = (h["signal"] + 1) / 2
    h["rolling_corr"]= h["price_norm"].rolling(5).corr(h["signal_norm"]).fillna(0)
    h["momentum"]    = h["signal_norm"].diff().fillna(0)
    h["volatility"]  = h["signal"].rolling(3).std().fillna(0)
    h["pred_score"]  = (
        h["signal_norm"] * 0.5 +
        h["momentum"]    * 0.3 +
        ((h["rolling_corr"] + 1) / 2) * 0.2
    )
    h["position"]      = h["pred_score"].apply(position_size)
    h["strat_return"]  = h["position"].shift(1) * h["returns"]
    h["strat_curve"]   = (1 + h["strat_return"].fillna(0)).cumprod()
    h["hold_curve"]    = (1 + h["returns"].fillna(0)).cumprod()
    return h

# ============================================================
# HEADER
# ============================================================
hcol1, hcol2, hcol3 = st.columns([5, 3, 2])
with hcol1:
    st.markdown(
        '<h1>SIGNAL<span style="color:#e2e4ed">DESK</span>'
        '<span class="live-badge">● LIVE</span></h1>',
        unsafe_allow_html=True,
    )
with hcol2:
    st.markdown(
        f'<p style="font-size:0.68rem;color:#6b7280;font-family:\'JetBrains Mono\','
        f'monospace;margin-top:0.85rem">'
        f'Tracking {len(COINS)} assets  ·  {pd.Timestamp.now().strftime("%H:%M:%S")}</p>',
        unsafe_allow_html=True,
    )
with hcol3:
    if st.button("↻  Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.markdown("<hr>", unsafe_allow_html=True)

# ============================================================
# PRICE TICKER — all 8 coins, click to select
# ============================================================
ticker_cols = st.columns(len(COINS))
for i, (ticker, meta) in enumerate(COINS.items()):
    p_data   = prices.get(ticker, {})
    price    = p_data.get("price", 0)
    change   = p_data.get("change_24h", 0.0)
    sig      = coin_momentum.get(ticker, 0.0)
    sig_lbl, sig_col = get_signal_label(sig)
    direction = "up" if change >= 0 else "down"
    selected  = "selected" if st.session_state.selected_coin == ticker else ""

    with ticker_cols[i]:
        # Clicking the button selects the coin
        if st.button(
            f"{ticker}",
            key=f"coin_btn_{ticker}",
            use_container_width=True,
            help=f"Select {meta['name']}",
        ):
            st.session_state.selected_coin = ticker
            st.rerun()

        fmt_price = f"${price:,.2f}" if price < 1000 else f"${price:,.0f}"
        st.markdown(f"""
        <div class="price-chip {direction} {selected}">
            <div class="chip-label">{meta['name']}</div>
            <div class="chip-value" style="color:{COIN_COLORS[ticker]}">{fmt_price}</div>
            <div class="chip-delta {direction}">{'▲' if change >= 0 else '▼'} {change:+.2f}%</div>
            <div class="chip-sig" style="color:{sig_col}">Signal: {sig:+.3f} · {sig_lbl}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── alerts ──
for a in st.session_state.alerts[-2:]:
    st.markdown(f'<div class="alert-banner">⚡ {a}</div>', unsafe_allow_html=True)

# ============================================================
# ACTIVE COIN
# ============================================================
coin       = st.session_state.selected_coin
coin_meta  = COINS[coin]
coin_color = COIN_COLORS[coin]
coin_hist  = build_coin_history(coin)
coin_sig   = coin_momentum.get(coin, 0.0)
coin_price = prices.get(coin, {}).get("price", 0)
coin_chg   = prices.get(coin, {}).get("change_24h", 0.0)
sig_label, sig_color = get_signal_label(coin_sig)

# ============================================================
# TABS
# ============================================================
tab_overview, tab_signals, tab_heatmap, tab_strategy, tab_settings = st.tabs([
    f"📊  {coin} Overview",
    "📰  Signals",
    "🔥  Market Heatmap",
    "📈  Strategy",
    "⚙️  Settings",
])

# ─────────────────────────────────────────────
# TAB 1 — COIN OVERVIEW
# ─────────────────────────────────────────────
with tab_overview:
    st.markdown(
        f'<h2 style="color:{coin_color};font-family:\'JetBrains Mono\',monospace">'
        f'{coin_meta["name"]} ({coin})</h2>',
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Price",        f"${coin_price:,.2f}" if coin_price < 1000 else f"${coin_price:,.0f}",
              delta=f"{coin_chg:+.2f}% 24h")
    m2.metric("Signal",       f"{coin_sig:+.3f}", delta=sig_label)
    pred_now = float(coin_hist["pred_score"].iloc[-1]) if len(coin_hist) > 0 else 0.0
    pred_lbl, _ = get_signal_label(pred_now)
    m3.metric("Pred. Score",  f"{pred_now:+.3f}", delta=pred_lbl)
    mom_now  = float(coin_hist["momentum"].iloc[-1])  if len(coin_hist) > 1 else 0.0
    vol_now  = float(coin_hist["volatility"].iloc[-1]) if len(coin_hist) > 3 else 0.0
    m4.metric("Momentum",     f"{mom_now:+.3f}")
    m5.metric("Volatility",   f"{vol_now:.3f}")

    left, right = st.columns([2, 1], gap="medium")

    with left:
        st.markdown("### Signal & Price History")

        if len(coin_hist) >= 2:
            tf = st.radio("Range", ["1 H", "4 H", "All"], horizontal=True,
                          label_visibility="collapsed", key="tf_overview")
            disp = coin_hist.copy()
            if tf == "1 H":
                disp = disp[disp["time"] >= pd.Timestamp.now() - pd.Timedelta(hours=1)]
            elif tf == "4 H":
                disp = disp[disp["time"] >= pd.Timestamp.now() - pd.Timedelta(hours=4)]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=disp["time"], y=disp["price_norm"],
                name="Price (norm)", line=dict(color=coin_color, width=1.2, dash="dot"),
                hovertemplate="Price norm: %{y:.3f}<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=disp["time"], y=disp["signal_norm"],
                name="Sentiment", line=dict(color=GREEN, width=2),
                fill="tozeroy", fillcolor="rgba(0,212,168,0.06)",
                hovertemplate="Signal: %{y:.3f}<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=disp["time"], y=disp["pred_score"],
                name="Pred. score", line=dict(color=AMBER, width=1.4, dash="dash"),
                hovertemplate="Pred: %{y:.3f}<extra></extra>",
            ))
            fig.add_hline(y=0.5, line=dict(color="rgba(255,255,255,0.1)", width=1, dash="dot"))
            fig.update_layout(**PLOTLY_BASE, height=260)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Accumulating history — check back in a few refreshes.")

        # OHLC candlestick
        st.markdown("### OHLC Chart")
        ohlc_days = st.radio("OHLC Period", ["1 day", "7 days", "30 days"],
                             horizontal=True, label_visibility="collapsed",
                             key="ohlc_period")
        days_map = {"1 day": 1, "7 days": 7, "30 days": 30}
        ohlc = fetch_ohlc(coin, days=days_map[ohlc_days])
        if not ohlc.empty:
            fig2 = go.Figure(go.Candlestick(
                x=ohlc["time"],
                open=ohlc["open"], high=ohlc["high"],
                low=ohlc["low"],  close=ohlc["close"],
                increasing_line_color=GREEN, decreasing_line_color=RED,
                name=f"{coin}/USD",
            ))
            fig2.update_layout(**PLOTLY_BASE, height=240,
                               xaxis_rangeslider_visible=False)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("OHLC unavailable — API rate limit may apply.")

    with right:
        st.markdown("### Composite Score")

        fg_score = compute_fear_greed(
            coin_sig,
            float(coin_hist["rolling_corr"].iloc[-1]) if len(coin_hist) > 5 else 0.0,
            vol_now,
        )
        fg_label_str, fg_color = fear_greed_label(fg_score)

        fig_fg = go.Figure(go.Pie(
            values=[fg_score, 100 - fg_score],
            hole=0.72,
            marker_colors=[fg_color, "#1a1e29"],
            textinfo="none", hoverinfo="skip", showlegend=False,
        ))
        fig_fg.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=10, b=10), height=180,
            annotations=[
                dict(text=f'<b>{fg_score:.0f}</b>', x=0.5, y=0.55,
                     showarrow=False,
                     font=dict(size=30, color=fg_color, family="JetBrains Mono")),
                dict(text=fg_label_str.upper(), x=0.5, y=0.35,
                     showarrow=False,
                     font=dict(size=10, color="#6b7280", family="DM Sans")),
            ],
        )
        st.plotly_chart(fig_fg, use_container_width=True)

        # Rolling correlation gauge
        corr_val = float(coin_hist["rolling_corr"].iloc[-1]) if len(coin_hist) > 5 else 0.0
        corr_color = GREEN if corr_val > 0.3 else (RED if corr_val < -0.3 else AMBER)

        fig_corr = go.Figure(go.Indicator(
            mode="gauge+number",
            value=corr_val,
            number=dict(font=dict(size=22, color=corr_color, family="JetBrains Mono")),
            gauge=dict(
                axis=dict(range=[-1, 1], tickfont=dict(size=8, color="#6b7280")),
                bar=dict(color=corr_color, thickness=0.5),
                bgcolor="#1a1e29", borderwidth=0,
                steps=[
                    dict(range=[-1, -0.3],  color="rgba(255,77,106,0.12)"),
                    dict(range=[-0.3, 0.3], color="rgba(26,30,41,1)"),
                    dict(range=[0.3, 1],    color="rgba(0,212,168,0.12)"),
                ],
            ),
        ))
        fig_corr.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=10, b=10), height=150,
            font=dict(family="JetBrains Mono", color="#6b7280"),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        if   corr_val >  0.5: st.success("Trend confirmation")
        elif corr_val < -0.5: st.warning("Possible reversal")
        else:                 st.info("No directional edge")

# ─────────────────────────────────────────────
# TAB 2 — SIGNALS (all coins)
# ─────────────────────────────────────────────
with tab_signals:

    if df.empty:
        st.warning("No articles loaded.")
    else:
        top = df.copy()
        top["abs_signal"] = top["signal"].abs()
        top = top.sort_values("abs_signal", ascending=False).head(40).reset_index(drop=True)
        top["direction"] = top["signal"].apply(lambda x: "🟢 Bullish" if x > 0 else "🔴 Bearish")
        top["published"]  = pd.to_datetime(
            top["published_at"], utc=True, errors="coerce"
        ).dt.tz_localize(None)

        # Filters
        fc1, fc2, fc3, fc4 = st.columns([2, 2, 2, 2])
        with fc1:
            coin_filter = st.selectbox("Coin", ["All"] + list(COINS.keys()) + ["MACRO"])
        with fc2:
            src_filter = st.selectbox("Source", ["All"] + sorted(top["source"].unique().tolist()))
        with fc3:
            dir_filter = st.selectbox("Direction", ["All", "Bullish", "Bearish"])
        with fc4:
            min_sig = st.slider("Min |signal|", 0.0, 1.0, 0.0, 0.05)

        mask = top["abs_signal"] >= min_sig
        if coin_filter != "All":
            mask &= top["coin"] == coin_filter
        if src_filter != "All":
            mask &= top["source"] == src_filter
        if dir_filter == "Bullish":
            mask &= top["signal"] > 0
        elif dir_filter == "Bearish":
            mask &= top["signal"] < 0
        filtered = top[mask].reset_index(drop=True)

        st.markdown("### Top Signals")
        if filtered.empty:
            st.info("No signals match the current filters.")
        else:
            display = filtered[["coin", "source", "title", "direction",
                                 "signal", "confidence", "importance", "published"]].copy()
            display["signal"] = display["signal"].apply(lambda x: f"{x:+.3f}")
            display["confidence"] = display["confidence"].apply(lambda x: f"{x:.2f}")
            display["importance"] = display["importance"].apply(lambda x: f"{x:.2f}")
            st.dataframe(display, use_container_width=True, hide_index=True)

        # Per-coin bar chart
        st.markdown("### Signal Strength by Coin")
        if not filtered.empty:
            bar_data = filtered.nlargest(20, "abs_signal")
            bar_colors = [
                COIN_COLORS.get(r["coin"], GREEN) if r["signal"] > 0
                else RED
                for _, r in bar_data.iterrows()
            ]
            fig_bar = go.Figure(go.Bar(
                x=bar_data["signal"],
                y=bar_data["title"].str[:50] + "…",
                orientation="h",
                marker_color=bar_colors,
                customdata=bar_data["coin"],
                hovertemplate="[%{customdata}] %{y}<br>Signal: %{x:.3f}<extra></extra>",
            ))
            fig_bar.add_vline(x=0, line=dict(color="rgba(255,255,255,0.12)", width=1))
            layout = {**PLOTLY_BASE, "margin": dict(l=300, r=20, t=20, b=40)}
            fig_bar.update_layout(**layout, height=max(300, len(bar_data) * 30))
            fig_bar.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_bar, use_container_width=True)

        # Sentiment breakdown per coin (stacked bar)
        st.markdown("### Sentiment Breakdown by Coin")
        if not df.empty:
            sent_df = df[df["coin"] != "MACRO"].copy()
            sent_df["sent_label"] = sent_df["sentiment"].map(
                {1: "Positive", -1: "Negative", 0: "Neutral"}
            )
            pivot = sent_df.groupby(["coin", "sent_label"]).size().unstack(fill_value=0)
            pivot = pivot.reindex(columns=["Positive", "Neutral", "Negative"], fill_value=0)

            fig_sent = go.Figure()
            for col, color in [("Positive", GREEN), ("Neutral", MUTED), ("Negative", RED)]:
                if col in pivot.columns:
                    fig_sent.add_trace(go.Bar(
                        name=col, x=pivot.index, y=pivot[col],
                        marker_color=color,
                        hovertemplate=f"{col}: %{{y}}<extra></extra>",
                    ))
            fig_sent.update_layout(
                **PLOTLY_BASE, barmode="stack", height=240,
                legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(fig_sent, use_container_width=True)

        # CSV export
        csv = filtered[["coin", "source", "title", "direction", "signal",
                         "confidence", "importance", "published"]].to_csv(index=False) \
              if not filtered.empty else ""
        if csv:
            st.download_button("⬇  Export CSV", data=csv,
                               file_name=f"signals_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                               mime="text/csv")

# ─────────────────────────────────────────────
# TAB 3 — MARKET HEATMAP
# ─────────────────────────────────────────────
with tab_heatmap:

    st.markdown("### Live Signal Heatmap")

    tickers     = list(COINS.keys())
    sig_values  = [coin_momentum.get(t, 0.0) for t in tickers]
    chg_values  = [prices.get(t, {}).get("change_24h", 0.0) for t in tickers]
    price_vals  = [prices.get(t, {}).get("price", 0) for t in tickers]
    mcap_vals   = [prices.get(t, {}).get("mcap", 1e9) for t in tickers]

    # Heatmap grid — 2 rows × 4 cols, sized by market cap
    fig_heat = go.Figure()
    cols_per_row = 4
    cell_w, cell_h = 140, 100
    pad = 10

    for i, ticker in enumerate(tickers):
        row   = i // cols_per_row
        col   = i % cols_per_row
        sig   = sig_values[i]
        chg   = chg_values[i]
        price = price_vals[i]
        color = COIN_COLORS[ticker]

        # Signal-tinted background
        intensity = min(abs(sig) * 0.8, 0.5)
        if sig > 0.05:
            fill = f"rgba(0,212,168,{intensity:.2f})"
        elif sig < -0.05:
            fill = f"rgba(255,77,106,{intensity:.2f})"
        else:
            fill = "rgba(26,30,41,0.9)"

        x0 = col * (cell_w + pad)
        y0 = row * (cell_h + pad)
        x1 = x0 + cell_w
        y1 = y0 + cell_h

        fig_heat.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                           fillcolor=fill, line=dict(color=color, width=1))
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

        sig_lbl, _ = get_signal_label(sig)
        fmt_p = f"${price:,.2f}" if price < 100 else f"${price:,.0f}"

        fig_heat.add_annotation(x=cx, y=cy + 28, text=f"<b>{ticker}</b>",
                                showarrow=False, font=dict(size=16, color=color,
                                family="JetBrains Mono"))
        fig_heat.add_annotation(x=cx, y=cy + 8, text=fmt_p,
                                showarrow=False, font=dict(size=11, color="#e2e4ed",
                                family="JetBrains Mono"))
        chg_col = "#00d4a8" if chg >= 0 else "#ff4d6a"
        fig_heat.add_annotation(x=cx, y=cy - 10,
                                text=f"{'▲' if chg >= 0 else '▼'} {chg:+.2f}%",
                                showarrow=False, font=dict(size=10, color=chg_col,
                                family="DM Sans"))
        fig_heat.add_annotation(x=cx, y=cy - 28,
                                text=f"Sig: {sig:+.3f}  {sig_lbl}",
                                showarrow=False, font=dict(size=9, color="#9ca3af",
                                family="JetBrains Mono"))

    total_w = cols_per_row * (cell_w + pad)
    total_h = (len(tickers) // cols_per_row) * (cell_h + pad) + 20
    fig_heat.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, range=[0, total_w]),
        yaxis=dict(visible=False, range=[0, total_h], autorange="reversed"),
        margin=dict(l=0, r=0, t=10, b=0), height=total_h + 20,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # 24h change bar chart
    st.markdown("### 24h Price Change")
    chg_colors = [GREEN if c >= 0 else RED for c in chg_values]
    fig_chg = go.Figure(go.Bar(
        x=tickers, y=chg_values,
        marker_color=chg_colors,
        text=[f"{c:+.2f}%" for c in chg_values],
        textposition="outside",
        textfont=dict(size=10, family="JetBrains Mono"),
        hovertemplate="%{x}: %{y:+.2f}%<extra></extra>",
    ))
    fig_chg.add_hline(y=0, line=dict(color="rgba(255,255,255,0.15)", width=1))
    fig_chg.update_layout(**PLOTLY_BASE, height=220)
    st.plotly_chart(fig_chg, use_container_width=True)

    # Sentiment signal comparison across all coins
    st.markdown("### Signal Comparison")
    sig_colors = [GREEN if s > 0.05 else (RED if s < -0.05 else MUTED) for s in sig_values]
    fig_sig = go.Figure(go.Bar(
        x=tickers, y=sig_values,
        marker_color=sig_colors,
        text=[f"{s:+.3f}" for s in sig_values],
        textposition="outside",
        textfont=dict(size=10, family="JetBrains Mono"),
        hovertemplate="%{x} signal: %{y:+.3f}<extra></extra>",
    ))
    fig_sig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.15)", width=1))
    fig_sig.update_layout(**PLOTLY_BASE, height=220,
                          yaxis=dict(**PLOTLY_BASE["yaxis"], range=[-1, 1]))
    st.plotly_chart(fig_sig, use_container_width=True)

# ─────────────────────────────────────────────
# TAB 4 — STRATEGY
# ─────────────────────────────────────────────
with tab_strategy:

    st.markdown(
        f'<h2 style="color:{coin_color};font-family:\'JetBrains Mono\','
        f'monospace">{coin} Strategy</h2>',
        unsafe_allow_html=True,
    )

    pred_now = float(coin_hist["pred_score"].iloc[-1]) if len(coin_hist) > 0 else 0.0
    pred_lbl, pred_col = get_signal_label(pred_now)
    pos_now  = position_size(pred_now)
    pos_label = "LONG" if pos_now > 0 else ("SHORT" if pos_now < 0 else "FLAT")

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Pred. Score",  f"{pred_now:+.3f}", delta=pred_lbl)
    sc2.metric("Position",     f"{pos_label} {abs(pos_now)*100:.0f}%")
    corr_h = float(coin_hist["rolling_corr"].iloc[-1]) if len(coin_hist) > 5 else 0.0
    sc3.metric("Corr (5-pt)",  f"{corr_h:+.3f}")
    vol_h  = float(coin_hist["volatility"].iloc[-1]) if len(coin_hist) > 3 else 0.0
    sc4.metric("Volatility",   f"{vol_h:.3f}")

    if len(coin_hist) > 2:
        # Position history
        st.markdown("### Position Timeline")
        pos_colors = [GREEN if p > 0 else (RED if p < 0 else MUTED)
                      for p in coin_hist["position"]]
        fig_pos = go.Figure(go.Bar(
            x=coin_hist["time"], y=coin_hist["position"],
            marker_color=pos_colors,
            hovertemplate="Position: %{y:.1f}<extra></extra>",
        ))
        fig_pos.add_hline(y=0, line=dict(color="rgba(255,255,255,0.15)", width=1))
        fig_pos.update_layout(**PLOTLY_BASE, height=180)
        fig_pos.update_yaxes(range=[-1.3, 1.3], tickvals=[-1, -0.5, 0, 0.5, 1])
        st.plotly_chart(fig_pos, use_container_width=True)

        # Equity curve
        st.markdown("### Strategy vs Buy & Hold")
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=coin_hist["time"], y=coin_hist["hold_curve"],
            name=f"{coin} Buy & Hold",
            line=dict(color=coin_color, width=1.5, dash="dot"),
            hovertemplate="Hold: %{y:.4f}<extra></extra>",
        ))
        fig_eq.add_trace(go.Scatter(
            x=coin_hist["time"], y=coin_hist["strat_curve"],
            name="Signal Strategy",
            line=dict(color=GREEN, width=2),
            fill="tonexty", fillcolor="rgba(0,212,168,0.05)",
            hovertemplate="Strategy: %{y:.4f}<extra></extra>",
        ))
        fig_eq.update_layout(**PLOTLY_BASE, height=240)
        st.plotly_chart(fig_eq, use_container_width=True)

        # Multi-coin equity comparison
        st.markdown("### Multi-Coin Strategy Comparison")
        fig_multi = go.Figure()
        for t in COINS:
            h = build_coin_history(t)
            if len(h) > 2 and "strat_curve" in h.columns:
                fig_multi.add_trace(go.Scatter(
                    x=h["time"], y=h["strat_curve"],
                    name=t, line=dict(color=COIN_COLORS[t], width=1.5),
                    hovertemplate=f"{t}: %{{y:.4f}}<extra></extra>",
                ))
        fig_multi.update_layout(**PLOTLY_BASE, height=280)
        st.plotly_chart(fig_multi, use_container_width=True)
    else:
        st.info("Accumulating history for strategy charts.")

    # Export
    if len(coin_hist) > 0:
        hist_csv = coin_hist[["time", "price", "signal", "pred_score",
                               "position", "strat_curve", "hold_curve"]].to_csv(index=False)
        st.download_button(
            f"⬇  Export {coin} history CSV", data=hist_csv,
            file_name=f"{coin}_history_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )

# ─────────────────────────────────────────────
# TAB 5 — SETTINGS
# ─────────────────────────────────────────────
with tab_settings:
    st.markdown("### Alert Thresholds")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        signal_thresh = st.slider("Signal alert threshold", 0.1, 1.0,
                                   st.session_state.get("signal_thresh", 0.45), 0.05)
        st.session_state["signal_thresh"] = signal_thresh
    with col_s2:
        fg_alert = st.slider("Fear & Greed extreme alert", 0, 50,
                              st.session_state.get("fg_alert", 20), 5)
        st.session_state["fg_alert"] = fg_alert

    st.markdown("### Recent Alerts")
    if st.session_state.alerts:
        for a in reversed(st.session_state.alerts):
            st.markdown(f'<div class="alert-banner">{a}</div>', unsafe_allow_html=True)
    else:
        st.caption("No alerts this session.")

    st.markdown("### Pipeline Info")
    i1, i2, i3 = st.columns(3)
    i1.metric("Articles loaded",  len(df) if not df.empty else 0)
    i2.metric("Coins tracked",    len(COINS))
    i3.metric("History points",   len(history))
    st.caption("Model: ProsusAI/FinBERT  ·  Pipeline TTL: 60s  ·  Price TTL: 20s")

# ============================================================
# AUTO-REFRESH
# ============================================================
time.sleep(15)
st.rerun()
