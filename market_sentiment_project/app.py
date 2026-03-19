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
# METRICS
# ---------------------------
if market_signal > 0.1 and momentum > 0:
    signal_label = "🟢 STRONG BUY"
elif market_signal > 0:
    signal_label = "🟢 BUY"
elif market_signal < -0.1 and momentum < 0:
    signal_label = "🔴 STRONG SELL"
elif market_signal < 0:
    signal_label = "🔴 SELL"
else:
    signal_label = "🟡 HOLD"

st.metric("Market Signal", signal_label, delta=round(market_signal, 3))

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

if len(history) > 2:
    history = history.copy()

    # Normalize BTC (0 → 1 scale)
    history["btc_norm"] = (
        history["btc_price"] - history["btc_price"].min()
    ) / (
        history["btc_price"].max() - history["btc_price"].min() + 1e-9
    )

    # Normalize Signal (-1 → 1 → shift to 0 → 1)
    history["signal_norm"] = (history["signal"] + 1) / 2

    chart_data = history.set_index("time")[["btc_norm", "signal_norm"]]

    st.line_chart(chart_data)

else:
    st.write("Collecting data...")


    corr = history["btc_norm"].corr(history["signal_norm"])

    st.metric("BTC vs Sentiment Correlation", round(corr, 3))

    history["signal_shifted"] = history["signal_norm"].shift(1)
    
    lead_corr = history["signal_shifted"].corr(history["btc_norm"])

    st.metric("Sentiment Leading Indicator", round(lead_corr, 3))


    history["signal_momentum"] = history["signal_norm"].diff()

    momentum = history["signal_momentum"].iloc[-1]

    st.metric("Sentiment Momentum", round(momentum, 3))
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
