import streamlit as st
from market_sentiment_project.pipeline import run_pipeline

st.set_page_config(page_title="Market Sentiment Dashboard", layout="wide")

st.title("📊 Crypto Market Sentiment Dashboard")

df, btc_price = run_pipeline()

st.subheader(f"BTC Price: ${btc_price}")

df["abs_signal"] = df["signal"].abs()
top = df.sort_values("abs_signal", ascending=False).head(10)

st.dataframe(top[["source","title","sentiment","confidence","signal"]])

st.bar_chart(top.set_index("title")["signal"])
