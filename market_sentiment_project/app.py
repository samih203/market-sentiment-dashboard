import streamlit as st
import pandas as pd
import time
from pipeline import run_pipeline

# ---------------------------
# PAGE SETUP
# ---------------------------
st.set_page_config(page_title="Market Sentiment Dashboard", layout="wide")

st.title("📊 Crypto Market Sentiment Dashboard")

# ---------------------------
# CACHE PIPELINE (prevents API spam)
# ---------------------------
@st.cache_data(ttl=10)
def load_data():
    return run_pipeline()

# ---------------------------
# SAFE DATA LOAD
# ---------------------------
try:
    df, btc_price, market_signal = load_data()
except Exception as e:
    st.error(f"Pipeline error: {e}")
    st.stop()


# ---------------------------
# STORE HISTORY (for chart)
# ---------------------------
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["time", "btc_price", "signal"])

new_row = pd.DataFrame([{
    "time": pd.Timestamp.now(),
    "btc_price": btc_price,
    "signal": market_signal
}])

st.session_state.history = pd.concat(
    [st.session_state.history, new_row],
    ignore_index=True
)

# limit size (prevents lag)
st.session_state.history = st.session_state.history.tail(50)

# ---------------------------
# TOP SIGNALS TABLE
# ---------------------------
if not df.empty:
    df["abs_signal"] = df["signal"].abs()
    top = df.sort_values("abs_signal", ascending=False).head(10)

    st.subheader("Top Market Signals")
    st.dataframe(top[["source","title","sentiment","confidence","signal"]])

    st.bar_chart(top.set_index("title")["signal"])
else:
    st.warning("No data available from pipeline.")

# ---------------------------
# BTC vs SENTIMENT CHART
# ---------------------------
st.subheader("📈 BTC Price vs Sentiment (Normalized)")

history = st.session_state.history.copy()

history["btc_norm"] = (
    history["btc_price"] - history["btc_price"].min()
) / (
    history["btc_price"].max() - history["btc_price"].min() + 1e-9
)

history["signal_norm"] = (history["signal"] + 1) / 2

# Rolling correlation (window = 5 points)
history["rolling_corr"] = history["btc_norm"].rolling(5).corr(history["signal_norm"])
history["rolling_corr_scaled"] = (history["rolling_corr"] + 1) / 2


history["signal_momentum"] = history["signal_norm"].diff()
history["signal_volatility"] = history["signal_norm"].rolling(5).std()

if len(history) > 2:
    # Normalize
    history["btc_norm"] = (
        history["btc_price"] - history["btc_price"].min()
    ) / (
        history["btc_price"].max() - history["btc_price"].min() + 1e-9
    )

chart_data = history.set_index("time")[["btc_norm", "signal_norm", "rolling_corr_scaled"]]    
st.line_chart(chart_data)

else:
    st.write("Collecting data...")


momentum = history["signal_momentum"].iloc[-1]
volatility = history["signal_volatility"].iloc[-1]
confidence = abs(market_signal) * (1 + abs(momentum))

# Correlation
corr = history["btc_norm"].corr(history["signal_norm"])

st.metric("BTC vs Sentiment Correlation", round(corr, 3))

# Lead indicator
history["signal_shifted"] = history["signal_norm"].shift(1)
lead_corr = history["signal_shifted"].corr(history["btc_norm"])

st.metric("Sentiment Leading Indicator", round(lead_corr, 3))

# Momentum display
st.metric("Sentiment Momentum", round(momentum, 3))


latest_corr = history["rolling_corr"].iloc[-1]

if latest_corr > 0.5:
    st.success("📈 Strong positive correlation (trend confirmation)")
elif latest_corr < -0.5:
    st.warning("⚠️ Strong negative correlation (possible reversal)")
else:
    st.info("➖ Weak correlation (no clear signal)")
# ---------------------------
# METRICS
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
# MANUAL REFRESH BUTTON
# ---------------------------
if st.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# ---------------------------
# AUTO REFRESH (every 5 sec)
# ---------------------------
import time
time.sleep(5)
st.rerun()
