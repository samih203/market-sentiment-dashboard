import streamlit as st
from pipeline import run_pipeline

st.set_page_config(page_title="Market Sentiment Dashboard", layout="wide")

st.title("📊 Crypto Market Sentiment Dashboard")

# ✅ CACHE YOUR PIPELINE
@st.cache_data(ttl=60)
def load_data():
    return run_pipeline()

df, btc_price, market_signal = load_data()

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

# ✅ Optional manual refresh
if st.button("Refresh Data"):
    st.cache_data.clear()
