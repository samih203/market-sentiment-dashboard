import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time

from pipeline import (
    run_pipeline,
    fetch_ohlc,
    compute_fear_greed,
    fear_greed_label,
    get_sentiment_pipeline,
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
# GLOBAL CSS — dark terminal aesthetic
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=DM+Sans:wght@400;500;600&display=swap');

/* ── root tokens ── */
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

/* ── page & app shell ── */
html, body, [class*="css"] { font-family: var(--sans); }
.stApp { background: var(--bg) !important; color: var(--text); }
section[data-testid="stSidebar"] { background: var(--bg2) !important; }
.block-container { padding: 1.6rem 2rem 4rem !important; max-width: 1400px; }

/* ── hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── headings ── */
h1, h2, h3 { font-family: var(--mono) !important; letter-spacing: -0.5px; }
h1 { font-size: 1.4rem !important; font-weight: 700 !important; color: var(--green) !important; }
h2 { font-size: 1rem  !important; font-weight: 600 !important; color: var(--text)  !important; }
h3 { font-size: 0.85rem !important; font-weight: 600 !important; color: var(--muted) !important;
     text-transform: uppercase; letter-spacing: 1.5px; margin-top: 1.4rem !important; }

/* ── metric cards ── */
[data-testid="metric-container"] {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 1rem 1.1rem !important;
    font-family: var(--mono) !important;
}
[data-testid="metric-container"] label {
    font-size: 0.65rem !important; letter-spacing: 1.2px !important;
    color: var(--muted) !important; text-transform: uppercase; font-family: var(--sans) !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.35rem !important; font-weight: 700 !important; font-family: var(--mono) !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] svg { display: none; }
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 0.72rem !important; font-family: var(--sans) !important;
}

/* ── dataframe ── */
[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 8px !important; }
.dvn-scroller { background: var(--bg2) !important; }

/* ── buttons ── */
.stButton > button {
    background: var(--bg3) !important; color: var(--muted) !important;
    border: 1px solid var(--border2) !important; border-radius: 6px !important;
    font-family: var(--mono) !important; font-size: 0.75rem !important;
    letter-spacing: 0.5px !important; padding: 0.4rem 1rem !important;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    border-color: var(--green) !important; color: var(--green) !important;
    background: var(--green-dim) !important;
}

/* ── download button ── */
.stDownloadButton > button {
    background: var(--bg3) !important; color: var(--blue) !important;
    border: 1px solid rgba(77,159,255,0.25) !important; border-radius: 6px !important;
    font-family: var(--mono) !important; font-size: 0.75rem !important;
}

/* ── tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg2) !important; border-radius: 8px !important;
    border: 1px solid var(--border) !important; padding: 3px !important; gap: 2px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; border-radius: 5px !important;
    color: var(--muted) !important; font-family: var(--mono) !important;
    font-size: 0.72rem !important; letter-spacing: 0.8px !important;
    padding: 6px 18px !important;
}
.stTabs [aria-selected="true"] {
    background: var(--bg3) !important; color: var(--text) !important;
    border: 1px solid var(--border2) !important;
}

/* ── selectbox / radio ── */
.stSelectbox > div > div, .stRadio > div {
    background: var(--bg2) !important; border-color: var(--border2) !important;
    color: var(--text) !important; border-radius: 6px !important;
}

/* ── slider ── */
.stSlider .thumb { background: var(--green) !important; }

/* ── alert / info boxes ── */
.stAlert { border-radius: 8px !important; font-family: var(--sans) !important; }

/* ── section divider ── */
hr { border-color: var(--border) !important; margin: 1.2rem 0 !important; }

/* ── custom price ticker card ── */
.price-chip {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    position: relative;
    overflow: hidden;
}
.price-chip::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
}
.price-chip.up::before   { background: var(--green); }
.price-chip.down::before { background: var(--red); }
.price-chip.neutral::before { background: var(--amber); }
.chip-label { font-size: 0.6rem; color: var(--muted); letter-spacing: 1.2px;
               text-transform: uppercase; font-family: var(--sans); margin-bottom: 4px; }
.chip-value { font-size: 1.4rem; font-weight: 700; font-family: var(--mono); }
.chip-delta { font-size: 0.7rem; margin-top: 3px; font-family: var(--sans); }
.chip-delta.up   { color: var(--green); }
.chip-delta.down { color: var(--red); }
.chip-delta.neutral { color: var(--amber); }

/* ── live badge ── */
.live-badge {
    display: inline-block;
    background: var(--green-dim);
    color: var(--green);
    border: 1px solid rgba(0,212,168,0.22);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.65rem;
    letter-spacing: 1px;
    font-family: var(--mono);
    vertical-align: middle;
    margin-left: 10px;
}

/* ── signal bar in table ── */
.sig-bar-wrap { display: flex; align-items: center; gap: 6px; }
.sig-bar-track { width: 52px; height: 4px; background: var(--bg3);
                  border-radius: 2px; overflow: hidden; display: inline-block; }
.sig-bar-fill  { height: 100%; border-radius: 2px; }

/* ── fear & greed ring label ── */
.fg-ring { text-align: center; padding: 0.5rem 0; }
.fg-score { font-size: 2.4rem; font-weight: 700; font-family: var(--mono); }
.fg-label { font-size: 0.72rem; letter-spacing: 1px; text-transform: uppercase;
             font-family: var(--sans); color: var(--muted); margin-top: 2px; }

/* ── alert banner ── */
.alert-banner {
    background: var(--amber-dim);
    border: 1px solid rgba(245,166,35,0.22);
    border-radius: 8px;
    padding: 0.6rem 1rem;
    font-size: 0.75rem;
    font-family: var(--sans);
    color: var(--amber);
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# PLOTLY DARK THEME DEFAULTS
# ============================================================
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono, monospace", color="#6b7280", size=11),
    margin=dict(l=40, r=20, t=28, b=40),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False,
               showline=False, tickfont=dict(size=10)),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False,
               showline=False, tickfont=dict(size=10)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="#1a1e29", bordercolor="#2a2e3d",
                    font=dict(family="JetBrains Mono", size=11, color="#e2e4ed")),
)

GREEN  = "#00d4a8"
RED    = "#ff4d6a"
AMBER  = "#f5a623"
BLUE   = "#4d9fff"
PURPLE = "#a78bfa"
MUTED  = "#6b7280"

# ============================================================
# WARM-UP MODEL (cached for the whole session)
# ============================================================
@st.cache_resource(show_spinner="Loading FinBERT model…")
def load_model():
    return get_sentiment_pipeline()

load_model()

# ============================================================
# PIPELINE CACHE  (5-minute TTL — FinBERT is slow)
# ============================================================
# Pipeline (NLP-heavy) cached for 60s — prices fetched live every rerun
@st.cache_data(ttl=60, show_spinner="Fetching signals…")
def load_pipeline():
    return run_pipeline()

# ============================================================
# SAFE LOAD
# ============================================================
try:
    _df_cached, _prices_cached, market_signal = load_pipeline()
except Exception as e:
    st.error(f"Pipeline error: {e}")
    st.stop()

# Deep-copy df so Streamlit cache mutation protection never silently no-ops
df = _df_cached.copy(deep=True) if not _df_cached.empty else _df_cached

# Always fetch fresh prices (fast, no NLP) so chart updates every rerun
from pipeline import fetch_prices as _fetch_prices
try:
    prices = _fetch_prices(ttl=30)
except Exception:
    prices = _prices_cached

btc_price  = prices.get("btc", 0)
eth_price  = prices.get("eth", 0)
btc_change = prices.get("btc_change", 0.0)
eth_change = prices.get("eth_change", 0.0)

# ============================================================
# SESSION HISTORY
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=[
        "time", "btc_price", "eth_price", "signal"
    ])
if "alerts" not in st.session_state:
    st.session_state.alerts = []

new_row = pd.DataFrame([{
    "time":      pd.Timestamp.now(),
    "btc_price": float(btc_price),
    "eth_price": float(eth_price),
    "signal":    float(market_signal),
}])

# Only append if price or signal actually changed (avoids flat duplicate rows)
_h = st.session_state.history
if _h.empty or float(btc_price) != float(_h["btc_price"].iloc[-1]) or float(market_signal) != float(_h["signal"].iloc[-1]):
    st.session_state.history = pd.concat(
        [_h, new_row], ignore_index=True
    ).tail(100)

history = st.session_state.history.copy()

# ============================================================
# DERIVED COLUMNS (strict order)
# ============================================================
history["btc_returns"]  = history["btc_price"].pct_change()
history["eth_returns"]  = history["eth_price"].pct_change()

btc_range = history["btc_price"].max() - history["btc_price"].min()
history["btc_norm"]    = (history["btc_price"] - history["btc_price"].min()) / (btc_range + 1e-9)
history["signal_norm"] = (history["signal"] + 1) / 2

history["rolling_corr"]       = history["btc_norm"].rolling(5).corr(history["signal_norm"]).fillna(0)
history["rolling_corr_scaled"]= (history["rolling_corr"] + 1) / 2

history["signal_momentum"]    = history["signal_norm"].diff().fillna(0)
history["signal_volatility"]  = history["signal"].rolling(3).std().fillna(0)

mom_max = history["signal_momentum"].abs().max()
history["momentum_norm"] = history["signal_momentum"] / (mom_max + 1e-9)

history["predictive_score"] = (
    history["signal_norm"]        * 0.5 +
    history["momentum_norm"]      * 0.3 +
    history["rolling_corr_scaled"]* 0.2
)
history["pred_scaled"] = (history["predictive_score"] + 1) / 2

def position_size(score):
    if   score >  0.3: return  1.0
    elif score >  0.1: return  0.5
    elif score < -0.3: return -1.0
    elif score < -0.1: return -0.5
    else:              return  0.0

history["position"]       = history["predictive_score"].apply(position_size)
history["btc_strategy"]   = history["position"].shift(1) * history["btc_returns"]
history["eth_strategy"]   = history["position"].shift(1) * history["eth_returns"]
history["future_return"]  = history["btc_returns"].shift(-1)
history["btc_curve"]      = (1 + history["btc_returns"].fillna(0)).cumprod()
history["eth_curve"]      = (1 + history["eth_returns"].fillna(0)).cumprod()
history["btc_strat_curve"]= (1 + history["btc_strategy"].fillna(0)).cumprod()
history["eth_strat_curve"]= (1 + history["eth_strategy"].fillna(0)).cumprod()
history["strategy_return"]= history["position"].shift(1) * history["btc_returns"]
history["strategy_curve"] = (1 + history["strategy_return"].fillna(0)).cumprod()
history["buy_hold"]       = (1 + history["btc_returns"].fillna(0)).cumprod()

# ============================================================
# SCALARS
# ============================================================
momentum    = float(history["signal_momentum"].iloc[-1]) if len(history) > 1 else 0.0
volatility  = float(history["signal_volatility"].iloc[-1]) if len(history) > 3 else 0.0
confidence  = abs(market_signal) * (1 + abs(momentum))
pred_score  = float(history["predictive_score"].iloc[-1]) if len(history) > 0 else 0.0
latest_corr = float(history["rolling_corr"].iloc[-1]) if len(history) > 5 else 0.0
corr        = float(history["btc_norm"].corr(history["signal_norm"]))
history["signal_shifted"] = history["signal_norm"].shift(1)
lead_corr   = float(history["signal_shifted"].corr(history["btc_norm"]))

valid = history.dropna(subset=["predictive_score", "future_return"])
prediction_corr = (
    float(valid["predictive_score"].corr(valid["future_return"])) if len(valid) > 5 else 0.0
)

fg_score           = compute_fear_greed(market_signal, latest_corr, volatility)
fg_label, fg_color = fear_greed_label(fg_score)

# ── signal label ──
if   volatility < 0.02:                                          signal_label = "NO CLEAR TREND"
elif market_signal > 0.15 and momentum > 0 and confidence > 0.2: signal_label = "STRONG BUY"
elif market_signal > 0:                                           signal_label = "BUY"
elif market_signal < -0.15 and momentum < 0 and confidence > 0.2:signal_label = "STRONG SELL"
elif market_signal < 0:                                           signal_label = "SELL"
else:                                                             signal_label = "HOLD"

# ── prediction label ──
if   pred_score >  0.15: pred_label = "PREDICT BUY"
elif pred_score < -0.15: pred_label = "PREDICT SELL"
else:                    pred_label = "NO EDGE"

# ── alert detection ──
ALERT_THRESHOLD = 0.5
if not df.empty:
    high_impact = df[df["signal"].abs() > ALERT_THRESHOLD]
    for _, row in high_impact.iterrows():
        alert_msg = f"{'🟢' if row['signal'] > 0 else '🔴'} High-impact signal: \"{row['title'][:80]}…\" ({row['signal']:+.2f})"
        if alert_msg not in st.session_state.alerts:
            st.session_state.alerts.append(alert_msg)
            st.toast(alert_msg, icon="⚡")
st.session_state.alerts = st.session_state.alerts[-5:]

# ============================================================
# HEADER
# ============================================================
col_logo, col_ts, col_btn = st.columns([5, 3, 2])
with col_logo:
    st.markdown(
        f'<h1>SIGNAL<span style="color:#e2e4ed">DESK</span>'
        f'<span class="live-badge">● LIVE</span></h1>',
        unsafe_allow_html=True,
    )
with col_ts:
    st.markdown(
        f'<p style="font-size:0.7rem;color:#6b7280;font-family:\'JetBrains Mono\',monospace;'
        f'margin-top:0.9rem">Last refresh: {pd.Timestamp.now().strftime("%H:%M:%S")}</p>',
        unsafe_allow_html=True,
    )
with col_btn:
    if st.button("↻  Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.markdown("<hr>", unsafe_allow_html=True)

# ============================================================
# PRICE TICKER ROW
# ============================================================
def price_chip(label, value, change, prefix="$", decimals=0):
    direction = "up" if change >= 0 else "down"
    arrow     = "▲" if change >= 0 else "▼"
    fmt_val   = f"{prefix}{value:,.{decimals}f}"
    return f"""
    <div class="price-chip {direction}">
        <div class="chip-label">{label}</div>
        <div class="chip-value">{fmt_val}</div>
        <div class="chip-delta {direction}">{arrow} {change:+.2f}%  24h</div>
    </div>"""

sig_dir   = "up" if market_signal >= 0 else "down"
pred_dir  = "up" if pred_score >= 0 else "down"

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(price_chip("BTC / USD", btc_price, btc_change), unsafe_allow_html=True)
with c2:
    st.markdown(price_chip("ETH / USD", eth_price, eth_change), unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="price-chip {sig_dir}">
        <div class="chip-label">MARKET SIGNAL</div>
        <div class="chip-value" style="color:{'#00d4a8' if market_signal>=0 else '#ff4d6a'}">{market_signal:+.3f}</div>
        <div class="chip-delta {sig_dir}">{signal_label}</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div class="price-chip {pred_dir}">
        <div class="chip-label">PRED. SCORE</div>
        <div class="chip-value" style="color:{'#00d4a8' if pred_score>=0 else '#ff4d6a'}">{pred_score:+.3f}</div>
        <div class="chip-delta {pred_dir}">{pred_label}</div>
    </div>""", unsafe_allow_html=True)
with c5:
    st.markdown(f"""
    <div class="price-chip {'up' if fg_score>=50 else 'down'}">
        <div class="chip-label">FEAR & GREED</div>
        <div class="chip-value" style="color:{fg_color}">{fg_score:.0f}</div>
        <div class="chip-delta" style="color:{fg_color}">{fg_label}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── alert banners ──
for a in st.session_state.alerts[-2:]:
    st.markdown(f'<div class="alert-banner">⚡ {a}</div>', unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================
tab_overview, tab_signals, tab_strategy, tab_settings = st.tabs([
    "📊  Overview", "📰  Signals", "📈  Strategy", "⚙️  Settings"
])

# ─────────────────────────────────────────────
# TAB 1 — OVERVIEW
# ─────────────────────────────────────────────
with tab_overview:

    left, right = st.columns([2, 1], gap="medium")

    # ── Left: signal + BTC chart ──
    with left:
        st.markdown("### Signal History & BTC Overlay")

        timeframe_opts = {"1 H": "1H", "4 H": "4H", "All": "all"}
        tf = st.radio("Timeframe", list(timeframe_opts.keys()),
                      horizontal=True, label_visibility="collapsed")

        disp = history.copy()
        if tf == "1 H":
            cutoff = pd.Timestamp.now() - pd.Timedelta(hours=1)
            disp = disp[disp["time"] >= cutoff]
        elif tf == "4 H":
            cutoff = pd.Timestamp.now() - pd.Timedelta(hours=4)
            disp = disp[disp["time"] >= cutoff]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=disp["time"], y=disp["btc_norm"],
            name="BTC (norm)", line=dict(color=BLUE, width=1.2, dash="dot"),
            hovertemplate="BTC norm: %{y:.3f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=disp["time"], y=disp["signal_norm"],
            name="Sentiment", line=dict(color=GREEN, width=2),
            fill="tozeroy", fillcolor="rgba(0,212,168,0.07)",
            hovertemplate="Signal: %{y:.3f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=disp["time"], y=disp["predictive_score"],
            name="Pred. score", line=dict(color=AMBER, width=1.5, dash="dash"),
            hovertemplate="Pred: %{y:.3f}<extra></extra>",
        ))
        fig.add_hline(y=0.5, line=dict(color="rgba(255,255,255,0.12)",
                                        width=1, dash="dot"))
        fig.update_layout(**PLOTLY_LAYOUT, height=280)
        st.plotly_chart(fig, use_container_width=True)

        # ── BTC candlestick ──
        st.markdown("### BTC Price (OHLC)")
        ohlc = fetch_ohlc("bitcoin", days=1)
        if not ohlc.empty:
            fig2 = go.Figure(go.Candlestick(
                x=ohlc["time"],
                open=ohlc["open"], high=ohlc["high"],
                low=ohlc["low"],  close=ohlc["close"],
                increasing_line_color=GREEN,
                decreasing_line_color=RED,
                name="BTC/USD",
            ))
            fig2.update_layout(**PLOTLY_LAYOUT, height=240,
                               xaxis_rangeslider_visible=False)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("OHLC data unavailable — API rate limit may apply.")

    # ── Right: Fear & Greed gauge + metrics ──
    with right:
        st.markdown("### Composite Score")

        # Fear & Greed donut
        fig_fg = go.Figure(go.Pie(
            values=[fg_score, 100 - fg_score],
            hole=0.72,
            marker_colors=[fg_color, "#1a1e29"],
            textinfo="none",
            hoverinfo="skip",
            showlegend=False,
        ))
        fig_fg.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=20, b=10),
            height=190,
            annotations=[dict(
                text=f'<b style="font-family:JetBrains Mono">{fg_score:.0f}</b>',
                x=0.5, y=0.55, showarrow=False,
                font=dict(size=32, color=fg_color, family="JetBrains Mono"),
            ), dict(
                text=fg_label.upper(),
                x=0.5, y=0.35, showarrow=False,
                font=dict(size=11, color="#6b7280", family="DM Sans"),
            )],
        )
        st.plotly_chart(fig_fg, use_container_width=True)

        # 4-metric mini grid
        ma, mb = st.columns(2)
        ma.metric("Momentum",   f"{momentum:+.3f}")
        mb.metric("Volatility", f"{volatility:.3f}")
        ma.metric("Confidence", f"{confidence:.3f}")
        mb.metric("Pred. Corr", f"{prediction_corr:+.3f}")

        st.markdown("<hr>", unsafe_allow_html=True)

        # Rolling correlation gauge
        corr_color = GREEN if latest_corr > 0.3 else (RED if latest_corr < -0.3 else AMBER)
        st.markdown("### Rolling Correlation")
        fig_corr = go.Figure(go.Indicator(
            mode="gauge+number",
            value=latest_corr,
            number=dict(font=dict(size=26, color=corr_color,
                                  family="JetBrains Mono")),
            gauge=dict(
                axis=dict(range=[-1, 1], tickcolor="#6b7280",
                          tickfont=dict(size=9, color="#6b7280")),
                bar=dict(color=corr_color, thickness=0.5),
                bgcolor="#1a1e29",
                borderwidth=0,
                steps=[
                    dict(range=[-1, -0.3],  color="rgba(255,77,106,0.13)"),
                    dict(range=[-0.3, 0.3], color="rgba(26,30,41,1)"),
                    dict(range=[0.3, 1],    color="rgba(0,212,168,0.13)"),
                ],
                threshold=dict(line=dict(color="#6b7280", width=1),
                               thickness=0.6, value=0),
            ),
        ))
        fig_corr.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=10, b=10),
            height=160,
            font=dict(family="JetBrains Mono", color="#6b7280"),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        if   latest_corr >  0.5: st.success("Strong positive correlation — trend confirmation")
        elif latest_corr < -0.5: st.warning("Strong negative correlation — possible reversal")
        else:                    st.info("Weak correlation — no directional edge")

        st.metric("BTC-Sentiment Corr",    f"{corr:+.3f}")
        st.metric("Lead Indicator (1-lag)", f"{lead_corr:+.3f}")

# ─────────────────────────────────────────────
# TAB 2 — SIGNALS
# ─────────────────────────────────────────────
with tab_signals:

    if df.empty:
        st.warning("No articles loaded — check your RSS feeds.")
    else:
        # Work on a clean copy — never mutate the cached df
        top = df.copy()
        top["abs_signal"] = top["signal"].abs()
        top = top.sort_values("abs_signal", ascending=False).head(20).reset_index(drop=True)
        top["direction"] = top["signal"].apply(
            lambda x: "🟢 Bullish" if x > 0 else "🔴 Bearish"
        )
        # Strip timezone so st.dataframe renders the column cleanly
        top["published"] = pd.to_datetime(
            top["published_at"], utc=True, errors="coerce"
        ).dt.tz_localize(None)

        # Filter controls
        fc1, fc2, fc3 = st.columns([2, 2, 3])
        with fc1:
            src_filter = st.selectbox("Source", ["All"] + sorted(top["source"].unique().tolist()))
        with fc2:
            dir_filter = st.selectbox("Direction", ["All", "Bullish", "Bearish"])
        with fc3:
            min_sig = st.slider("Min |signal|", 0.0, 1.0, 0.0, 0.05)

        mask = top["abs_signal"] >= min_sig
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
            # Build display df: format signal as string for display only
            display = filtered[["source", "title", "direction",
                                 "signal", "confidence", "importance", "published"]].copy()
            display["signal"] = display["signal"].apply(lambda x: f"{x:+.3f}")
            st.dataframe(display, use_container_width=True, hide_index=True)

        # ── horizontal bar chart ──
        st.markdown("### Signal Strength")
        if not filtered.empty:
            bar_data = filtered.nlargest(15, "abs_signal")
            bar_colors = [GREEN if v > 0 else RED for v in bar_data["signal"]]
            fig_bar = go.Figure(go.Bar(
                x=bar_data["signal"],
                y=bar_data["title"].str[:55] + "…",
                orientation="h",
                marker_color=bar_colors,
                hovertemplate="%{y}<br>Signal: %{x:.3f}<extra></extra>",
            ))
            fig_bar.add_vline(x=0, line=dict(color="rgba(255,255,255,0.15)", width=1))
            layout = {**PLOTLY_LAYOUT}
            layout["margin"] = dict(l=320, r=20, t=20, b=40)
            fig_bar.update_layout(**layout, height=max(300, len(bar_data) * 32))
            fig_bar.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_bar, use_container_width=True)

        # ── CSV export — use numeric signal, not formatted string ──
        csv_df = filtered[["source", "title", "direction", "signal",
                            "confidence", "importance", "published"]].copy() if not filtered.empty else top[["source", "title", "direction", "signal", "confidence", "importance", "published"]].copy()
        csv = csv_df.to_csv(index=False)
        st.download_button(
            label="⬇  Export signals CSV",
            data=csv,
            file_name=f"signals_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )

        # ── sentiment distribution pie ──
        st.markdown("### Sentiment Distribution")
        dist = df["sentiment"].map({1: "Positive", -1: "Negative", 0: "Neutral"}).value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=dist.index, values=dist.values,
            marker_colors=[GREEN, RED, MUTED],
            hole=0.55,
            textinfo="label+percent",
            textfont=dict(family="JetBrains Mono", size=11),
            hoverinfo="label+value",
        ))
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            height=260,
            margin=dict(l=20, r=20, t=10, b=10),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# ─────────────────────────────────────────────
# TAB 3 — STRATEGY
# ─────────────────────────────────────────────
with tab_strategy:

    st.markdown("### Prediction Signal")
    pred_col, pos_col = st.columns(2)

    with pred_col:
        pred_color = GREEN if pred_score > 0.15 else (RED if pred_score < -0.15 else AMBER)
        fig_pred = go.Figure(go.Indicator(
            mode="number+delta",
            value=pred_score,
            delta=dict(reference=0, valueformat="+.3f",
                       increasing_color=GREEN, decreasing_color=RED),
            number=dict(font=dict(size=42, color=pred_color,
                                  family="JetBrains Mono"),
                        valueformat="+.3f"),
            title=dict(text=pred_label,
                       font=dict(size=14, color="#6b7280",
                                 family="DM Sans")),
        ))
        fig_pred.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            height=160, margin=dict(l=20, r=20, t=20, b=10),
        )
        st.plotly_chart(fig_pred, use_container_width=True)

    with pos_col:
        cur_position = position_size(pred_score)
        pos_label    = ("LONG" if cur_position > 0 else
                        "SHORT" if cur_position < 0 else "FLAT")
        pos_color    = GREEN if cur_position > 0 else (RED if cur_position < 0 else MUTED)
        st.metric("Current Position", f"{pos_label} {abs(cur_position)*100:.0f}%",
                  delta=f"Score {pred_score:+.3f}")
        st.metric("Prediction Accuracy", f"{prediction_corr:+.3f}",
                  delta="correlation vs future return")

    # ── position timeline ──
    st.markdown("### Position History")
    if len(history) > 2:
        fig_pos = go.Figure()
        pos_colors = [GREEN if p > 0 else (RED if p < 0 else MUTED)
                      for p in history["position"]]
        fig_pos.add_trace(go.Bar(
            x=history["time"], y=history["position"],
            marker_color=pos_colors,
            name="Position",
            hovertemplate="Position: %{y:.1f}<extra></extra>",
        ))
        fig_pos.add_hline(y=0, line=dict(color="rgba(255,255,255,0.15)", width=1))
        fig_pos.update_layout(**PLOTLY_LAYOUT, height=200)
        fig_pos.update_yaxes(range=[-1.3, 1.3], tickvals=[-1, -0.5, 0, 0.5, 1])
        st.plotly_chart(fig_pos, use_container_width=True)

    # ── equity curves ──
    st.markdown("### Strategy vs Buy & Hold")
    if len(history) > 2:
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=history["time"], y=history["buy_hold"],
            name="BTC Buy & Hold",
            line=dict(color=MUTED, width=1.5, dash="dot"),
            hovertemplate="Hold: %{y:.4f}<extra></extra>",
        ))
        fig_eq.add_trace(go.Scatter(
            x=history["time"], y=history["strategy_curve"],
            name="Signal Strategy",
            line=dict(color=GREEN, width=2),
            fill="tonexty", fillcolor="rgba(0,212,168,0.05)",
            hovertemplate="Strategy: %{y:.4f}<extra></extra>",
        ))
        fig_eq.update_layout(**PLOTLY_LAYOUT, height=260)
        st.plotly_chart(fig_eq, use_container_width=True)

    # ── multi-asset ──
    st.markdown("### Multi-Asset Equity Curves")
    if len(history) > 2:
        fig_ma = go.Figure()
        curves = [
            ("BTC Hold",     "btc_curve",       BLUE,   "dot"),
            ("BTC Strategy", "btc_strat_curve",  GREEN,  "solid"),
            ("ETH Hold",     "eth_curve",        PURPLE, "dot"),
            ("ETH Strategy", "eth_strat_curve",  AMBER,  "solid"),
        ]
        for name, col, color, dash in curves:
            fig_ma.add_trace(go.Scatter(
                x=history["time"], y=history[col],
                name=name,
                line=dict(color=color, width=1.8 if dash == "solid" else 1.2, dash=dash),
                hovertemplate=f"{name}: %{{y:.4f}}<extra></extra>",
            ))
        fig_ma.update_layout(**PLOTLY_LAYOUT, height=280)
        st.plotly_chart(fig_ma, use_container_width=True)

    # ── full signal history export ──
    hist_csv = history[["time", "btc_price", "eth_price", "signal",
                         "predictive_score", "position",
                         "btc_strat_curve", "buy_hold"]].to_csv(index=False)
    st.download_button(
        label="⬇  Export full history CSV",
        data=hist_csv,
        file_name=f"history_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )

# ─────────────────────────────────────────────
# TAB 4 — SETTINGS
# ─────────────────────────────────────────────
with tab_settings:

    st.markdown("### Alert Thresholds")
    st.info("Alert settings are stored in session state and reset on page reload.")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        signal_thresh = st.slider(
            "Signal alert threshold", 0.1, 1.0,
            st.session_state.get("signal_thresh", 0.5), 0.05,
        )
        st.session_state["signal_thresh"] = signal_thresh
    with col_s2:
        fg_alert = st.slider(
            "Fear & Greed extreme alert", 0, 50,
            st.session_state.get("fg_alert", 20), 5,
        )
        st.session_state["fg_alert"] = fg_alert

    if fg_score <= fg_alert:
        st.error(f"⚠️ Extreme Fear detected — F&G index at {fg_score:.0f}")
    if fg_score >= 100 - fg_alert:
        st.success(f"🚀 Extreme Greed detected — F&G index at {fg_score:.0f}")

    st.markdown("### Recent Alerts")
    if st.session_state.alerts:
        for a in reversed(st.session_state.alerts):
            st.markdown(f'<div class="alert-banner">{a}</div>', unsafe_allow_html=True)
    else:
        st.caption("No alerts triggered this session.")

    st.markdown("### Pipeline Info")
    info_col1, info_col2 = st.columns(2)
    info_col1.metric("Articles loaded", len(df) if not df.empty else 0)
    info_col2.metric("History points",  len(history))
    st.caption("Model: ProsusAI/FinBERT  |  Pipeline TTL: 300s  |  Price TTL: 60s")

# ============================================================
# AUTO-REFRESH every 15s — prices update each rerun, NLP cached separately
# ============================================================
time.sleep(15)
st.rerun()
