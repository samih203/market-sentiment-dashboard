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
@st.cache_data(ttl=60, show_spinner="Analyzing news…")
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
    """
    Extract price + signal history for a single coin.
    Always returns a DataFrame with ALL derived columns present,
    even when there are fewer than 2 rows — callers can safely
    access any column and guard only on len().
    """
    DERIVED_COLS = [
        "returns", "price_norm", "signal_norm", "rolling_corr",
        "momentum", "volatility", "pred_score",
        "position", "strat_return", "strat_curve", "hold_curve",
    ]

    def _empty():
        return pd.DataFrame(columns=["time", "price", "signal"] + DERIVED_COLS)

    p_col = f"{ticker}_price"
    s_col = f"{ticker}_signal"
    if p_col not in history.columns or s_col not in history.columns:
        return _empty()

    h = history[["time", p_col, s_col]].copy().rename(
        columns={p_col: "price", s_col: "signal"}
    )
    h = h[h["price"] > 0].copy().reset_index(drop=True)

    # Always add all derived columns, defaulting to 0.0
    for col in DERIVED_COLS:
        h[col] = 0.0

    if len(h) < 2:
        return h

    h["returns"]      = h["price"].pct_change()
    p_range           = h["price"].max() - h["price"].min()
    h["price_norm"]   = (h["price"] - h["price"].min()) / (p_range + 1e-9)
    h["signal_norm"]  = (h["signal"] + 1) / 2
    h["rolling_corr"] = h["price_norm"].rolling(5).corr(h["signal_norm"]).fillna(0)
    h["momentum"]     = h["signal_norm"].diff().fillna(0)
    h["volatility"]   = h["signal"].rolling(3).std().fillna(0)
    h["pred_score"]   = (
        h["signal_norm"] * 0.5 +
        h["momentum"]    * 0.3 +
        ((h["rolling_corr"] + 1) / 2) * 0.2
    )
    h["position"]     = h["pred_score"].apply(position_size)
    h["strat_return"] = h["position"].shift(1) * h["returns"]
    h["strat_curve"]  = (1 + h["strat_return"].fillna(0)).cumprod()
    h["hold_curve"]   = (1 + h["returns"].fillna(0)).cumprod()
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
tab_overview, tab_signals, tab_heatmap, tab_strategy, tab_settings, tab_learn = st.tabs([
    f"📊  {coin} Overview",
    "📰  Signals",
    "🔥  Market Heatmap",
    "📈  Strategy",
    "⚙️  Settings",
    "📚  Learn",
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
    pred_now = float(coin_hist["pred_score"].iloc[-1]) if len(coin_hist) > 1 else 0.0
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
            fig_sent.update_layout(**PLOTLY_BASE, barmode="stack", height=240)
            fig_sent.update_layout(legend=dict(orientation="h", y=1.1))
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
    fig_sig.update_layout(**PLOTLY_BASE, height=220)
    fig_sig.update_yaxes(range=[-1, 1])
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

    pred_now = float(coin_hist["pred_score"].iloc[-1]) if len(coin_hist) > 1 else 0.0
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


# ─────────────────────────────────────────────
# TAB 6 — LEARN
# ─────────────────────────────────────────────
with tab_learn:

    # ── CSS additions for learn tab cards ──
    st.markdown("""
    <style>
    .learn-card {
        background: var(--bg2);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.75rem;
    }
    .learn-card-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        font-weight: 700;
        color: var(--text);
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .learn-card p, .learn-card li {
        font-size: 0.8rem;
        color: #9ca3af;
        line-height: 1.7;
        font-family: 'DM Sans', sans-serif;
        margin: 0.3rem 0;
    }
    .learn-card ul { padding-left: 1.2rem; margin: 0.4rem 0; }
    .term-chip {
        display: inline-block;
        background: var(--bg3);
        border: 1px solid var(--border2);
        border-radius: 5px;
        padding: 3px 10px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        color: var(--green);
        margin: 3px 3px 3px 0;
    }
    .level-badge {
        font-size: 0.58rem;
        padding: 2px 8px;
        border-radius: 20px;
        font-family: 'DM Sans', sans-serif;
        letter-spacing: 0.8px;
        text-transform: uppercase;
    }
    .level-beginner { background: rgba(0,212,168,0.1); color: #00d4a8;
                       border: 1px solid rgba(0,212,168,0.2); }
    .level-intermediate { background: rgba(245,166,35,0.1); color: #f5a623;
                           border: 1px solid rgba(245,166,35,0.2); }
    .level-advanced { background: rgba(255,77,106,0.1); color: #ff4d6a;
                       border: 1px solid rgba(255,77,106,0.2); }
    .learn-divider {
        border: none; border-top: 1px solid var(--border);
        margin: 1.4rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Section selector ──
    section = st.radio(
        "Section",
        ["🌐  What is Crypto?", "📊  Reading This Dashboard",
         "🧠  How the AI Works", "📈  Trading Concepts", "🔤  Glossary"],
        horizontal=True, label_visibility="collapsed",
    )

    # ══════════════════════════════════════════
    # SECTION 1 — WHAT IS CRYPTO
    # ══════════════════════════════════════════
    if section == "🌐  What is Crypto?":

        st.markdown("""
        <div class="learn-card">
            <div class="learn-card-title">
                🌐 What is Cryptocurrency?
                <span class="level-badge level-beginner">Beginner</span>
            </div>
            <p>Cryptocurrency is digital money secured by cryptography — mathematical codes that
            make it nearly impossible to counterfeit. Unlike traditional currencies (dollars, euros),
            no government or bank controls it. Transactions are recorded on a <strong style="color:#e2e4ed">blockchain</strong>:
            a public ledger shared across thousands of computers worldwide.</p>
            <p>Think of the blockchain as a Google Doc that everyone can read, nobody can delete,
            and new lines can only be added — never edited.</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="learn-card">
                <div class="learn-card-title">₿ Bitcoin (BTC)</div>
                <p>The original cryptocurrency, created in 2009 by the pseudonymous
                Satoshi Nakamoto. Bitcoin was designed as a peer-to-peer electronic
                cash system — a way to send value anywhere in the world without a bank.</p>
                <ul>
                    <li>Fixed supply of 21 million coins ever</li>
                    <li>New BTC minted via "mining" — solving complex math problems</li>
                    <li>"Halving" events cut the mining reward in half every ~4 years,
                    reducing new supply (historically bullish)</li>
                    <li>Widely considered "digital gold" — a store of value</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="learn-card">
                <div class="learn-card-title">◎ Solana (SOL)</div>
                <p>A high-speed blockchain focused on speed and low fees. Where Ethereum
                can handle ~15 transactions per second, Solana targets 65,000+.</p>
                <ul>
                    <li>Popular for NFTs, gaming, and DeFi apps</li>
                    <li>Uses "Proof of History" — a clock built into the blockchain</li>
                    <li>Has suffered notable network outages in the past</li>
                    <li>Strong developer community and ecosystem growth</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="learn-card">
                <div class="learn-card-title">🔴 Avalanche (AVAX)</div>
                <p>A platform for building custom blockchains ("subnets") with near-instant
                finality. Designed to be highly scalable without sacrificing decentralization.</p>
                <ul>
                    <li>Transaction finality in under 2 seconds</li>
                    <li>Subnets let institutions build private or permissioned chains</li>
                    <li>Competes directly with Ethereum for DeFi activity</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="learn-card">
                <div class="learn-card-title">🐕 Dogecoin (DOGE)</div>
                <p>Started as a joke in 2013 based on a popular meme, DOGE has become
                a top-10 cryptocurrency by market cap — largely driven by social media
                and celebrity attention (notably Elon Musk).</p>
                <ul>
                    <li>No supply cap — inflationary by design</li>
                    <li>Extremely low transaction fees, fast confirmations</li>
                    <li>Highly sentiment-driven — news and tweets move the price dramatically</li>
                    <li>Used for tipping and microtransactions</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="learn-card">
                <div class="learn-card-title">⟠ Ethereum (ETH)</div>
                <p>The world's programmable blockchain. While Bitcoin is mainly for
                transferring value, Ethereum runs <strong style="color:#e2e4ed">smart contracts</strong>
                — self-executing code that powers apps (dApps), DeFi protocols, NFTs, and more.</p>
                <ul>
                    <li>Powers most of the DeFi (decentralized finance) ecosystem</li>
                    <li>Switched to "Proof of Stake" in 2022 (The Merge), cutting energy use ~99%</li>
                    <li>ETH is used to pay "gas fees" — the cost of running computations</li>
                    <li>EIP upgrades continuously improve performance and economics</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="learn-card">
                <div class="learn-card-title">🔶 BNB (Binance)</div>
                <p>The native token of Binance — the world's largest crypto exchange —
                and its own blockchain (BNB Chain). Originally a discount token for
                trading fees, it now powers a large DeFi ecosystem.</p>
                <ul>
                    <li>Used for transaction fees on BNB Chain (cheap and fast)</li>
                    <li>Binance periodically "burns" (destroys) BNB to reduce supply</li>
                    <li>Heavily tied to Binance's business health and regulatory status</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="learn-card">
                <div class="learn-card-title">💧 XRP (Ripple)</div>
                <p>XRP is designed for fast, cheap cross-border payments between banks
                and financial institutions. Ripple Labs (the company behind XRP) has
                had a long-running legal battle with the SEC over whether XRP is a security.</p>
                <ul>
                    <li>Transactions settle in 3–5 seconds for fractions of a cent</li>
                    <li>Adopted by banks and payment processors worldwide</li>
                    <li>The SEC lawsuit outcome is a major price driver — watch for news</li>
                    <li>Pre-mined — no mining, controlled distribution by Ripple Labs</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="learn-card">
                <div class="learn-card-title">🔵 Cardano (ADA)</div>
                <p>An academically-driven blockchain built with peer-reviewed research.
                Founded by Ethereum co-founder Charles Hoskinson, Cardano emphasizes
                formal verification and security above speed.</p>
                <ul>
                    <li>Research-first philosophy — slower to ship, but rigorously tested</li>
                    <li>Energy efficient Proof of Stake since launch</li>
                    <li>Growing smart contract and DeFi ecosystem (still maturing)</li>
                    <li>Strong community focus on developing-world financial inclusion</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<hr class="learn-divider">', unsafe_allow_html=True)
        st.markdown("""
        <div class="learn-card">
            <div class="learn-card-title">⚡ Key Crypto Concepts</div>
            <p><strong style="color:#e2e4ed">Market Cap</strong> — Total value of all coins in circulation
            (price × supply). Used to rank coins by size. BTC and ETH dominate.</p>
            <p><strong style="color:#e2e4ed">Volatility</strong> — Crypto is far more volatile than stocks.
            A 10% daily move is normal. 50%+ drawdowns happen in bear markets. Never invest more than you can afford to lose.</p>
            <p><strong style="color:#e2e4ed">Bull / Bear Market</strong> — Bull = prices rising, optimism
            high. Bear = prices falling, pessimism dominates. Crypto cycles are historically 3–4 years,
            loosely tied to Bitcoin halving events.</p>
            <p><strong style="color:#e2e4ed">Liquidity</strong> — How easily you can buy or sell without
            moving the price. BTC and ETH are very liquid. Smaller coins can be illiquid — a big sell order
            tanks the price.</p>
            <p><strong style="color:#e2e4ed">Whale</strong> — An individual or entity holding a very large
            amount of a coin. Whale transactions can move markets. On-chain analytics track wallet movements.</p>
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════
    # SECTION 2 — READING THE DASHBOARD
    # ══════════════════════════════════════════
    elif section == "📊  Reading This Dashboard":

        st.markdown("""
        <div class="learn-card">
            <div class="learn-card-title">
                📡 What SignalDesk Does
                <span class="level-badge level-beginner">Beginner</span>
            </div>
            <p>SignalDesk reads crypto news in real time, runs each headline through an AI model,
            and converts the sentiment into a numerical signal. The goal: surface whether the
            current news flow is bullish (positive) or bearish (negative) for each coin —
            faster than you could read the headlines yourself.</p>
            <p>It does <em>not</em> predict prices. It measures the emotional tone of the market
            narrative. Sentiment is one input into a trading decision, not a complete strategy.</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="learn-card">
                <div class="learn-card-title">🟢 Signal Score (−1 to +1)</div>
                <p>The core output of the AI pipeline for each coin.</p>
                <ul>
                    <li><strong style="color:#00d4a8">+0.5 to +1.0</strong> — Strong bullish news flow.
                    Multiple positive headlines, high AI confidence.</li>
                    <li><strong style="color:#f5a623">−0.15 to +0.15</strong> — Neutral. Mixed or
                    macro/irrelevant news. No clear directional edge.</li>
                    <li><strong style="color:#ff4d6a">−0.5 to −1.0</strong> — Strong bearish news flow.
                    Negative headlines, crashes, hacks, bans.</li>
                </ul>
                <p>The signal combines the AI sentiment score, keyword boosts (e.g. "ETF approval"
                adds +0.7), and article importance weighting.</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="learn-card">
                <div class="learn-card-title">🔮 Predictive Score</div>
                <p>A composite forward-looking number built from three inputs:</p>
                <ul>
                    <li><strong style="color:#e2e4ed">Signal (50%)</strong> — the current sentiment score</li>
                    <li><strong style="color:#e2e4ed">Momentum (30%)</strong> — is the signal trending up or down?</li>
                    <li><strong style="color:#e2e4ed">Correlation (20%)</strong> — how well has sentiment
                    predicted price movement recently?</li>
                </ul>
                <p>Above +0.15 → BUY signal. Below −0.15 → SELL signal. In between → no edge.</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="learn-card">
                <div class="learn-card-title">📈 OHLC Candlestick Chart</div>
                <p>Each candle represents a time period. The body shows open and close price;
                the wicks show the high and low.</p>
                <ul>
                    <li><strong style="color:#00d4a8">Green candle</strong> — price closed higher than it opened (bullish)</li>
                    <li><strong style="color:#ff4d6a">Red candle</strong> — price closed lower than it opened (bearish)</li>
                    <li>Long wicks = high volatility / indecision in that period</li>
                    <li>Many small candles in a tight range = consolidation (a breakout may follow)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="learn-card">
                <div class="learn-card-title">😨 Fear & Greed Index (0–100)</div>
                <p>A single number summarising market sentiment for a given coin.</p>
                <ul>
                    <li><strong style="color:#00d4a8">75–100 Extreme Greed</strong> — market may be
                    overbought. Historically a caution signal.</li>
                    <li><strong style="color:#4ade80">55–74 Greed</strong> — positive momentum, bullish bias</li>
                    <li><strong style="color:#f5a623">45–54 Neutral</strong> — no clear edge</li>
                    <li><strong style="color:#fb923c">25–44 Fear</strong> — negative sentiment, bearish bias</li>
                    <li><strong style="color:#ff4d6a">0–24 Extreme Fear</strong> — historically a contrarian
                    buy signal (market may be oversold)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="learn-card">
                <div class="learn-card-title">📉 Rolling Correlation Gauge</div>
                <p>Measures how closely the sentiment signal has tracked the price over the last
                5 data points.</p>
                <ul>
                    <li><strong style="color:#00d4a8">Above +0.5</strong> — sentiment and price are
                    moving together. The signal has predictive value right now.</li>
                    <li><strong style="color:#6b7280">Near 0</strong> — no relationship. Sentiment is
                    noisy relative to price.</li>
                    <li><strong style="color:#ff4d6a">Below −0.5</strong> — sentiment and price are
                    moving opposite. Could signal a reversal.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="learn-card">
                <div class="learn-card-title">🔥 Market Heatmap</div>
                <p>The heatmap tab shows all 8 coins at once. Each tile's background intensity
                encodes the strength of its sentiment signal:</p>
                <ul>
                    <li><strong style="color:#00d4a8">Bright green</strong> — strong bullish signal</li>
                    <li><strong style="color:#ff4d6a">Bright red</strong> — strong bearish signal</li>
                    <li><strong style="color:#6b7280">Dark / neutral</strong> — no clear sentiment edge</li>
                </ul>
                <p>Use the heatmap to quickly spot which coins have the strongest news momentum
                without switching tabs.</p>
            </div>
            """, unsafe_allow_html=True)

    # ══════════════════════════════════════════
    # SECTION 3 — HOW THE AI WORKS
    # ══════════════════════════════════════════
    elif section == "🧠  How the AI Works":

        st.markdown("""
        <div class="learn-card">
            <div class="learn-card-title">
                🤖 The FinBERT Model
                <span class="level-badge level-intermediate">Intermediate</span>
            </div>
            <p>SignalDesk uses <strong style="color:#e2e4ed">ProsusAI/FinBERT</strong> —
            a BERT-based language model fine-tuned specifically on financial news text.
            Unlike general-purpose sentiment models, FinBERT understands financial
            vocabulary: words like "hawkish", "liquidity", "default", and "halving"
            carry their correct financial meanings.</p>
            <p>For each headline it outputs three probabilities summing to 1.0:
            <strong style="color:#00d4a8">positive</strong>,
            <strong style="color:#6b7280">neutral</strong>,
            <strong style="color:#ff4d6a">negative</strong>.
            The winning label becomes the sentiment direction; the winning
            probability becomes the confidence score.</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="learn-card">
                <div class="learn-card-title">⚙️ Signal Pipeline Step-by-Step</div>
                <p><strong style="color:#e2e4ed">1. Fetch</strong> — RSS feeds from CoinDesk,
                Cointelegraph, Reuters, The Block, and Decrypt are fetched every 60 seconds.
                Up to 80 headlines are processed per run.</p>
                <p><strong style="color:#e2e4ed">2. Tag</strong> — Each headline is scanned for
                coin-specific keywords. "Ethereum gas fees spike" → tagged ETH.
                Generic macro articles ("Fed raises rates") → tagged MACRO, applied to all
                coins at 40% weight.</p>
                <p><strong style="color:#e2e4ed">3. Score</strong> — FinBERT assigns sentiment
                (−1 / 0 / +1) and confidence (0–1). The ML signal = sentiment × confidence.</p>
                <p><strong style="color:#e2e4ed">4. Boost</strong> — Keyword overrides are applied.
                Words like "crash", "hack", "ban" add −0.7. Words like "ETF", "approval",
                "rally" add +0.7. These capture directional facts the model might miss.</p>
                <p><strong style="color:#e2e4ed">5. Weight</strong> — Each article is multiplied by
                its importance score (source credibility + keyword significance) and a time decay
                factor (recent news counts more).</p>
                <p><strong style="color:#e2e4ed">6. Aggregate</strong> — All weighted signals for
                a coin are summed with exponential weighting → final signal score.</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="learn-card">
                <div class="learn-card-title">🏋️ Importance Score</div>
                <p>Not all news is equal. Each article gets an importance multiplier before
                its signal is counted:</p>
                <ul>
                    <li><strong style="color:#e2e4ed">Source weight</strong> — Reuters/Bloomberg
                    carry 0.4 weight; smaller crypto blogs get 0.15. More credible sources
                    move the signal more.</li>
                    <li><strong style="color:#e2e4ed">Keyword weight</strong> — "crash" and "surge"
                    add 2.0× (high impact events). "ETF" and "SEC" add 1.2× (regulatory news
                    is historically price-moving). "Inflation" and "Fed" add 1.5× (macro drivers).</li>
                    <li><strong style="color:#e2e4ed">Cap at 1.0</strong> — Importance is capped so
                    no single article can dominate the signal entirely.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="learn-card">
                <div class="learn-card-title">⏱️ Time Decay</div>
                <p>A 2-hour-old article matters less than a 10-minute-old one.
                The decay formula is:</p>
                <p style="font-family:'JetBrains Mono',monospace; font-size:0.75rem;
                color:#00d4a8; background:#1a1e29; padding:8px 12px; border-radius:6px;
                margin:8px 0">weight = max(0.1, 1 / (1 + hours_old / 6))</p>
                <p>At 0 hours → weight = 1.0. At 6 hours → weight = 0.5.
                At 24 hours → weight ≈ 0.2. The floor of 0.1 ensures even old news
                has a tiny influence.</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="learn-card">
                <div class="learn-card-title">⚠️ AI Limitations</div>
                <p>FinBERT reads headlines, not full articles. It can be fooled by:</p>
                <ul>
                    <li>Sarcasm or irony in headlines</li>
                    <li>Clickbait framing ("Bitcoin CRASHES to… a new all-time high")</li>
                    <li>Context-dependent news (a "lawsuit" might be good or bad depending
                    on who filed it)</li>
                    <li>Rumours and unverified reports that move sentiment before the truth emerges</li>
                </ul>
                <p>Always read the underlying article before acting on any signal.
                This tool is for research and education — not financial advice.</p>
            </div>
            """, unsafe_allow_html=True)

    # ══════════════════════════════════════════
    # SECTION 4 — TRADING CONCEPTS
    # ══════════════════════════════════════════
    elif section == "📈  Trading Concepts":

        st.markdown("""
        <div class="learn-card">
            <div class="learn-card-title">
                ⚠️ Disclaimer
                <span class="level-badge level-intermediate">Important</span>
            </div>
            <p>Nothing on this dashboard is financial advice. Crypto markets are highly volatile
            and speculative. Past signal performance does not guarantee future results.
            Always do your own research (DYOR) and consult a licensed financial advisor
            before making investment decisions.</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="learn-card">
                <div class="learn-card-title">📐 Position Sizing</div>
                <p>The strategy tab uses a simple rule to decide how much of a position to take
                based on the predictive score:</p>
                <ul>
                    <li><strong style="color:#00d4a8">Score > 0.3</strong> → Full long (100%)</li>
                    <li><strong style="color:#4ade80">Score > 0.1</strong> → Half long (50%)</li>
                    <li><strong style="color:#6b7280">−0.1 to 0.1</strong> → Flat (0% — no position)</li>
                    <li><strong style="color:#fb923c">Score < −0.1</strong> → Half short (−50%)</li>
                    <li><strong style="color:#ff4d6a">Score < −0.3</strong> → Full short (−100%)</li>
                </ul>
                <p>In real trading you would also factor in risk management rules,
                stop-losses, and portfolio correlation.</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="learn-card">
                <div class="learn-card-title">📊 Equity Curve</div>
                <p>The equity curve in the Strategy tab shows the hypothetical cumulative
                return of following the signal strategy vs simply holding (buy & hold).</p>
                <ul>
                    <li>Starts at 1.0 (100% of starting capital)</li>
                    <li>A value of 1.18 means the strategy has returned +18%</li>
                    <li>Strategy curve beating the hold curve = signal added value</li>
                    <li>Strategy curve lagging = the signal hurt performance</li>
                </ul>
                <p><strong style="color:#f5a623">Warning:</strong> With few data points (minutes of
                history) this is highly noisy. Equity curves need hundreds of trades
                to be statistically meaningful.</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="learn-card">
                <div class="learn-card-title">🔁 Long vs Short</div>
                <p><strong style="color:#00d4a8">Long</strong> — you buy and profit when price goes up.
                This is the standard "buy low, sell high" trade.</p>
                <p><strong style="color:#ff4d6a">Short</strong> — you borrow an asset and sell it,
                hoping to buy it back cheaper later and pocket the difference.
                You profit when price goes down.</p>
                <p>Shorting crypto is typically done via derivatives (futures/perpetuals)
                on exchanges like Binance or Bybit. It amplifies both gains and losses.
                Beginners should stick to long-only spot trading.</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="learn-card">
                <div class="learn-card-title">📉 Momentum</div>
                <p>Momentum measures whether the signal is accelerating or decelerating —
                the <em>rate of change</em> of sentiment, not its level.</p>
                <ul>
                    <li><strong style="color:#00d4a8">Positive momentum</strong> — sentiment is
                    getting more bullish. A rising tide.</li>
                    <li><strong style="color:#ff4d6a">Negative momentum</strong> — sentiment is
                    deteriorating even if still positive overall.</li>
                </ul>
                <p>In markets, momentum tends to persist in the short term (trending)
                but mean-revert over the medium term.</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="learn-card">
                <div class="learn-card-title">📏 Volatility</div>
                <p>The volatility metric here measures how much the signal itself is
                jumping around — signal noise, not price volatility.</p>
                <ul>
                    <li><strong style="color:#f5a623">High signal volatility</strong> — conflicting
                    headlines, uncertain market. Signals are less reliable.</li>
                    <li><strong style="color:#00d4a8">Low signal volatility</strong> — consistent
                    news flow in one direction. More trustworthy signal.</li>
                </ul>
                <p>The dashboard shows "NO CLEAR TREND" when signal volatility is below
                0.02 — the noise floor where direction is indistinguishable from randomness.</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="learn-card">
                <div class="learn-card-title">🎯 Sentiment vs Price</div>
                <p>Sentiment does not always lead price. Sometimes it lags. The relationship
                changes depending on market conditions:</p>
                <ul>
                    <li><strong style="color:#e2e4ed">Bull market</strong> — good news pushes prices
                    up quickly. Sentiment leads well.</li>
                    <li><strong style="color:#e2e4ed">Bear market</strong> — bad news may already be
                    priced in. Sentiment can lag or diverge.</li>
                    <li><strong style="color:#e2e4ed">Ranging market</strong> — sentiment and price
                    are uncorrelated. Watch the correlation gauge.</li>
                </ul>
                <p>The "Lead Indicator" metric on the Overview tab shows whether yesterday's
                sentiment predicted today's price. When positive and above 0.4, the signal
                has been genuinely predictive recently.</p>
            </div>
            """, unsafe_allow_html=True)

    # ══════════════════════════════════════════
    # SECTION 5 — GLOSSARY
    # ══════════════════════════════════════════
    elif section == "🔤  Glossary":

        st.markdown("""
        <div class="learn-card">
            <div class="learn-card-title">
                🔤 Full Glossary
                <span class="level-badge level-beginner">Reference</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        terms = {
            "All-Time High (ATH)": "The highest price a coin has ever reached. Breaking ATH is considered very bullish.",
            "Altcoin": "Any cryptocurrency that is not Bitcoin. ETH, SOL, ADA, etc. are all altcoins.",
            "Bearish": "Expecting prices to fall. Negative sentiment. Bear markets are prolonged downtrends.",
            "Block": "A batch of verified transactions added to the blockchain. Each block links to the previous one, forming the chain.",
            "Blockchain": "A distributed ledger of transactions shared across a network of computers. Immutable and transparent.",
            "Bullish": "Expecting prices to rise. Positive sentiment. Bull markets are prolonged uptrends.",
            "Burn": "Permanently destroying tokens by sending them to an unspendable address. Reduces supply, often bullish.",
            "Confidence Score": "How certain the AI model is about its sentiment classification. 0.9+ = very confident; 0.5 = coin flip.",
            "Correlation": "A measure of how two things move together. +1 = perfectly in sync; 0 = no relationship; −1 = opposite.",
            "DeFi": "Decentralized Finance — financial services (lending, trading, earning yield) built on blockchains without banks.",
            "DYOR": "Do Your Own Research. Always verify information before acting on it.",
            "Equity Curve": "A chart showing the cumulative performance of a trading strategy over time.",
            "ETF (Spot)": "Exchange-Traded Fund. A regulated product that lets traditional investors gain crypto exposure through a stock exchange. Bitcoin spot ETFs launched in the US in 2024, a major milestone.",
            "Fear & Greed Index": "A 0–100 composite sentiment score. Extreme Fear can indicate oversold conditions; Extreme Greed can indicate overbought.",
            "Fiat": "Government-issued currency (USD, EUR, GBP). Not backed by a commodity.",
            "Gas Fee": "The cost to execute a transaction or smart contract on Ethereum. Paid in ETH.",
            "Halving": "A programmed Bitcoin event (every ~210,000 blocks, ~4 years) that cuts the mining reward in half, reducing new BTC supply.",
            "Importance Score": "SignalDesk's measure of how market-moving an article is likely to be, based on source credibility and keyword significance.",
            "Liquidity": "How easily an asset can be bought or sold without significantly moving the price.",
            "Long": "A trade that profits when price rises. Buying an asset with the expectation it will go up.",
            "MACRO": "Articles with no coin-specific keywords. Applied to all coins at reduced weight as general market sentiment.",
            "Market Cap": "Total market value = price × circulating supply. Used to rank cryptocurrencies by size.",
            "Mining": "Using computing power to validate transactions and earn new Bitcoin as a reward.",
            "Momentum": "The rate of change of the signal — whether sentiment is accelerating positively or negatively.",
            "NFT": "Non-Fungible Token. A unique digital asset on a blockchain, commonly used for digital art and collectibles.",
            "Position": "How much exposure you have to an asset. Long 100% = fully invested; Short 50% = half-sized short bet.",
            "Predictive Score": "SignalDesk's composite forward-looking signal combining current sentiment (50%), momentum (30%), and correlation (20%).",
            "Proof of Stake (PoS)": "A consensus mechanism where validators lock up (stake) coins as collateral to validate transactions. Energy-efficient.",
            "Proof of Work (PoW)": "Bitcoin's consensus mechanism. Miners compete to solve math problems, consuming energy to validate transactions.",
            "Rolling Correlation": "Correlation calculated over a recent window (last 5 points here), showing the current relationship between sentiment and price.",
            "SEC": "U.S. Securities and Exchange Commission. Its regulatory actions (approvals, lawsuits) heavily impact crypto prices.",
            "Sentiment": "The emotional tone of a piece of text — Positive, Negative, or Neutral — as classified by FinBERT.",
            "Short": "A trade that profits when price falls. Borrowing and selling an asset expecting to buy it back cheaper.",
            "Signal": "SignalDesk's composite score from −1 (maximum bearish) to +1 (maximum bullish) for each coin.",
            "Smart Contract": "Self-executing code stored on a blockchain. Enables trustless agreements without intermediaries.",
            "Time Decay": "The reduction in an article's influence as it gets older. Recent news matters more.",
            "Volatility": "How much the signal jumps around. High volatility = conflicting/noisy news; low = consistent directional flow.",
            "Whale": "An entity holding a very large amount of a cryptocurrency. Whale moves can significantly impact markets.",
        }

        # Render in two columns, alphabetical
        sorted_terms = sorted(terms.items())
        half = len(sorted_terms) // 2
        gc1, gc2 = st.columns(2)

        for col, chunk in [(gc1, sorted_terms[:half]), (gc2, sorted_terms[half:])]:
            with col:
                for term, definition in chunk:
                    st.markdown(f"""
                    <div style="border-left: 2px solid var(--border2); padding: 6px 0 6px 12px; margin-bottom: 10px;">
                        <div style="font-family:'JetBrains Mono',monospace; font-size:0.72rem;
                                    font-weight:700; color:var(--green); margin-bottom:3px;">{term}</div>
                        <div style="font-size:0.75rem; color:#9ca3af; font-family:'DM Sans',sans-serif;
                                    line-height:1.6">{definition}</div>
                    </div>
                    """, unsafe_allow_html=True)

# ============================================================
# AUTO-REFRESH
# ============================================================
time.sleep(15)
st.rerun()
