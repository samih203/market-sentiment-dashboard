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
st.metric("Market Sentiment Signal", round(market_signal, 3))
st.subheader(f"BTC Price: ${btc_price}")

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
st.subheader("📈 BTC Price vs Sentiment")

history = st.session_state.history.copy()

if len(history) > 2:
    # Smooth BTC for better visuals
    history["btc_smooth"] = history["btc_price"].rolling(3).mean()

    chart_data = history.set_index("time")[["btc_smooth", "signal"]]

    st.line_chart(chart_data)
else:
    st.write("Collecting data... refresh in a few seconds.")

# ---------------------------
# MANUAL REFRESH BUTTON
# ---------------------------
if st.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# ---------------------------
# AUTO REFRESH (every 5 sec)
# ---------------------------
time.sleep(5)
st.rerun()
