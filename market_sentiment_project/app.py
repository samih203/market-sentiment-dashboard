import streamlit as st
import pandas as pd
import numpy as np
import time
from pipeline import run_pipeline

# ---------------------------
# PAGE SETUP
# ---------------------------
st.set_page_config(page_title="Market Sentiment Dashboard", layout="wide")
st.title("📊 Crypto Market Sentiment Dashboard")

# ---------------------------
# CACHE PIPELINE
# ---------------------------
@st.cache_data(ttl=60)
def load_data():
    return run_pipeline()

# ---------------------------
# SAFE DATA LOAD
# ---------------------------
try:
    df, btc_price, eth_price, market_signal = load_data()
except Exception as e:
    st.error(f"Pipeline error: {e}")
    st.stop()

# ---------------------------
# STORE HISTORY
# ---------------------------
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=[
        "time", "btc_price", "eth_price", "signal"
    ])

new_row = pd.DataFrame([{
    "time": pd.Timestamp.now(),
    "btc_price": btc_price,
    "eth_price": eth_price,
    "signal": market_signal
}])

st.session_state.history = pd.concat(
    [st.session_state.history, new_row],
    ignore_index=True
).tail(50)

history = st.session_state.history.copy()

# ---------------------------
# TOP SIGNALS TABLE
# ---------------------------
if not df.empty:
    df["abs_signal"] = df["signal"].abs()
    top = df.sort_values("abs_signal", ascending=False).head(10)

    st.subheader("Top Market Signals")
    top["direction"] = top["signal"].apply(
        lambda x: "🟢 Bullish" if x > 0 else "🔴 Bearish"
    )
    st.dataframe(top[["source", "title", "direction", "signal"]])
    st.bar_chart(top.set_index("title")["signal"])
else:
    st.warning("No data available from pipeline.")

# ---------------------------
# BUILD DERIVED COLUMNS (ORDER MATTERS)
# ---------------------------

# 1. Returns — must come first, everything else depends on this
history["btc_returns"] = history["btc_price"].pct_change()
history["eth_returns"] = history["eth_price"].pct_change()

# 2. Normalise price and signal
btc_range = history["btc_price"].max() - history["btc_price"].min()
history["btc_norm"] = (
    (history["btc_price"] - history["btc_price"].min()) /
    (btc_range + 1e-9)
)
history["signal_norm"] = (history["signal"] + 1) / 2

# 3. Rolling correlation
history["rolling_corr"] = (
    history["btc_norm"].rolling(5).corr(history["signal_norm"]).fillna(0)
)
history["rolling_corr_scaled"] = (history["rolling_corr"] + 1) / 2

# 4. Momentum & volatility
history["signal_momentum"] = history["signal_norm"].diff().fillna(0)
history["signal_volatility"] = history["signal"].rolling(3).std().fillna(0)

momentum_max = history["signal_momentum"].abs().max()
history["momentum_norm"] = (
    history["signal_momentum"] / (momentum_max + 1e-9)
)

# 5. Predictive score (single definition)
history["predictive_score"] = (
    history["signal_norm"] * 0.5 +
    history["momentum_norm"].fillna(0) * 0.3 +
    history["rolling_corr_scaled"] * 0.2
)
history["pred_scaled"] = (history["predictive_score"] + 1) / 2

# 6. Position sizing (must come before strategy returns)
def position_size(score):
    if score > 0.3:
        return 1.0
    elif score > 0.1:
        return 0.5
    elif score < -0.3:
        return -1.0
    elif score < -0.1:
        return -0.5
    else:
        return 0.0

history["position"] = history["predictive_score"].apply(position_size)

# 7. Strategy returns (position from prior row × current return)
history["btc_strategy"] = history["position"].shift(1) * history["btc_returns"]
history["eth_strategy"] = history["position"].shift(1) * history["eth_returns"]

# 8. Future return for correlation (uses btc_returns, which now exists)
history["future_return"] = history["btc_returns"].shift(-1)

# 9. Equity curves
history["btc_curve"] = (1 + history["btc_returns"].fillna(0)).cumprod()
history["eth_curve"] = (1 + history["eth_returns"].fillna(0)).cumprod()
history["btc_strategy_curve"] = (1 + history["btc_strategy"].fillna(0)).cumprod()
history["eth_strategy_curve"] = (1 + history["eth_strategy"].fillna(0)).cumprod()

# 10. Legacy strategy curve (signal-based, no asset split)
history["strategy_return"] = history["position"].shift(1) * history["btc_returns"]
history["strategy_curve"] = (1 + history["strategy_return"].fillna(0)).cumprod()
history["buy_hold"] = (1 + history["btc_returns"].fillna(0)).cumprod()

# ---------------------------
# SENTIMENT COLOUR
# ---------------------------
def get_color(signal):
    if signal > 0.1:
        return "green"
    elif signal < -0.1:
        return "red"
    else:
        return "gray"

st.markdown(
    f"### Market Sentiment: :{get_color(market_signal)}[{round(market_signal, 3)}]"
)

# ---------------------------
# PREDICTION
# ---------------------------
pred_score = history["predictive_score"].iloc[-1] if len(history) > 0 else 0

if pred_score > 0.15:
    prediction_label = "🟢 PREDICT BUY"
elif pred_score < -0.15:
    prediction_label = "🔴 PREDICT SELL"
else:
    prediction_label = "🟡 NO EDGE"

st.metric("Prediction", prediction_label, delta=round(pred_score, 3))

# Prediction accuracy vs actual future return
valid = history.dropna(subset=["predictive_score", "future_return"])
prediction_corr = (
    valid["predictive_score"].corr(valid["future_return"])
    if len(valid) > 5 else 0
)
st.metric("Prediction Accuracy (corr)", round(prediction_corr, 3))

# ---------------------------
# POSITIONING
# ---------------------------
st.subheader("📌 Positioning")
if len(history) > 1:
    st.line_chart(history.set_index("time")["position"])
else:
    st.write("Collecting data...")

# ---------------------------
# COMBINED SIGNALS CHART
# ---------------------------
if len(history) > 2:
    chart_data = history.set_index("time")[[
        "btc_norm",
        "signal_norm",
        "rolling_corr_scaled",
        "pred_scaled"
    ]]
    st.subheader("📊 Combined Signals View")
    st.line_chart(chart_data)

    st.subheader("📈 BTC Price")
    st.line_chart(history.set_index("time")["btc_price"])

    st.subheader("🧠 Sentiment Signal")
    st.line_chart(history.set_index("time")["signal"])

    st.subheader("🔮 Predictive Score")
    st.line_chart(history.set_index("time")["predictive_score"])
else:
    st.write("Collecting data...")

# ---------------------------
# CORRELATION METRICS
# ---------------------------
momentum = history["signal_momentum"].iloc[-1] if len(history) > 1 else 0
volatility = history["signal_volatility"].iloc[-1] if len(history) > 3 else 0
confidence = abs(market_signal) * (1 + abs(momentum))

corr = history["btc_norm"].corr(history["signal_norm"])
st.metric("BTC vs Sentiment Correlation", round(corr, 3))

history["signal_shifted"] = history["signal_norm"].shift(1)
lead_corr = history["signal_shifted"].corr(history["btc_norm"])
st.metric("Sentiment Leading Indicator", round(lead_corr, 3))

st.metric("Sentiment Momentum", round(momentum, 3))

if momentum > 0:
    st.success("📈 Sentiment Trending Up")
elif momentum < 0:
    st.error("📉 Sentiment Trending Down")
else:
    st.info("➡️ Sentiment Flat")

latest_corr = history["rolling_corr"].iloc[-1] if len(history) > 5 else 0

if latest_corr > 0.5:
    st.success("📈 Strong positive correlation (trend confirmation)")
elif latest_corr < -0.5:
    st.warning("⚠️ Strong negative correlation (possible reversal)")
else:
    st.info("➖ Weak correlation (no clear signal)")

# ---------------------------
# STRATEGY vs BUY-AND-HOLD
# ---------------------------
st.subheader("💰 Strategy vs BTC Buy & Hold")
st.line_chart(
    history.set_index("time")[["buy_hold", "strategy_curve"]]
)

# ---------------------------
# MULTI-ASSET PERFORMANCE
# ---------------------------
st.subheader("💰 Multi-Asset Strategy Performance")
st.line_chart(
    history.set_index("time")[[
        "btc_curve",
        "btc_strategy_curve",
        "eth_curve",
        "eth_strategy_curve"
    ]]
)

# ---------------------------
# SIGNAL LABEL & METRICS
# ---------------------------
if volatility < 0.02:
    signal_label = "⚪ NO CLEAR TREND"
elif market_signal > 0.15 and momentum > 0 and confidence > 0.2:
    signal_label = "🟢 STRONG BUY"
elif market_signal > 0:
    signal_label = "🟢 BUY"
elif market_signal < -0.15 and momentum < 0 and confidence > 0.2:
    signal_label = "🔴 STRONG SELL"
elif market_signal < 0:
    signal_label = "🔴 SELL"
else:
    signal_label = "🟡 HOLD"

st.metric("Market Signal", signal_label, delta=round(market_signal, 3))

col1, col2, col3 = st.columns(3)
col1.metric("Momentum", round(momentum, 3))
col2.metric("Volatility", round(volatility, 3))
col3.metric("Confidence", round(confidence, 3))

# ---------------------------
# REFRESH
# ---------------------------
if st.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()

time.sleep(5)
st.rerun()
