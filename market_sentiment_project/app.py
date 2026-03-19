import streamlit as st
from pipeline import run_pipeline

import pandas as pd
import time
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["time", "btc_price", "signal"])

st.set_page_config(page_title="Market Sentiment Dashboard", layout="wide")

st.title("📊 Crypto Market Sentiment Dashboard")
import time

time.sleep(5)
st.rerun()

# ✅ CACHE YOUR PIPELINE
@st.cache_data(ttl=10)
def load_data():
    return run_pipeline()

df, btc_price, market_signal = load_data()
new_row = pd.DataFrame([{
    "time": pd.Timestamp.now(),
    "btc_price": btc_price,
    "signal": market_signal
}])

st.session_state.history = pd.concat(
    [st.session_state.history, new_row],
    ignore_index=True
)
st.session_state.history = st.session_state.history.tail(50)

# ✅ Metrics
st.metric("Market Sentiment Signal", round(market_signal, 3))
st.subheader(f"BTC Price: ${btc_price}")

# ✅ Process data
df["abs_signal"] = df["signal"].abs()
top = df.sort_values("abs_signal", ascending=False).head(10)

# ✅ Table
st.dataframe(top[["source","title","sentiment","confidence","signal"]])

# ✅ Chart
st.bar_chart(top.set_index("title")["signal"])
st.subheader("📈 BTC vs Sentiment (Normalized)")

history = st.session_state.history.copy()

# Normalize
history["btc_norm"] = history["btc_price"] / history["btc_price"].iloc[0]
history["signal_norm"] = history["signal"] / max(abs(history["signal"].max()), 1)

chart_data = history.set_index("time")[["btc_norm", "signal_norm"]]

st.line_chart(chart_data)

# ✅ Optional manual refresh
if st.button("Refresh Data"):
    st.cache_data.clear()
