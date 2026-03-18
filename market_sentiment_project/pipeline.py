# =========================
# IMPORTS
# =========================
import pandas as pd
import matplotlib.pyplot as plt
import feedparser
from transformers import pipeline
from pycoingecko import CoinGeckoAPI
import torch

torch.set_grad_enabled(False)


import json
import os

CACHE_FILE = "article_cache.json"

# Load cache if it exists
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        ARTICLE_CACHE = json.load(f)
else:
    ARTICLE_CACHE = {}

def save_cache():
    with open(CACHE_FILE, "w") as f:
        json.dump(ARTICLE_CACHE, f)
# =========================
# MODEL
# =========================
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment",
    device=-1
)
#==========================
# TIME DECAY
#==========================
from datetime import datetime
import pandas as pd

def time_decay(published_at):
    if pd.isna(published_at):
        return 0.5

    try:
        # convert to pandas datetime (handles timezone safely)
        published_at = pd.to_datetime(published_at, utc=True)
        now = pd.Timestamp.utcnow()

        hours_old = (now - published_at).total_seconds() / 3600

        return max(0.1, 1 / (1 + hours_old / 6))

    except:
        return 0.5
# =========================
# TEXT HELPERS
# =========================
def shorten_text(text, max_words=200):
    return " ".join(text.split()[:max_words])

def get_article_content(article):
    url = article.get("url")

    if not url:
        return article.get("title", "")

    text = article.get("title")

    if not text or len(text.split()) < 50:
        return article.get("title", "")

    return text

# =========================
# BATCH SENTIMENT (FAST)
# =========================
def batch_sentiment(texts, batch_size=16):
    return sentiment_pipeline(
        texts,
        batch_size=batch_size,
        truncation=True
    )

# =========================
# FETCH RSS
# =========================
COINDESK_RSS = "https://www.coindesk.com/arc/outboundfeeds/rss/"
COINTELEGRAPH_RSS = "https://cointelegraph.com/rss"
REUTERS_RSS = "https://feeds.reuters.com/reuters/businessNews"

def fetch_rss_news(url, source_name):
    feed = feedparser.parse(url)
    articles = []

    for entry in feed.entries:
        articles.append({
            "title": entry.get("title"),
            "url": entry.get("link"),
            "published_at": pd.to_datetime(entry.get("published", None)),
            "source": source_name
        })

    return articles

# =========================
# ANALYZE ARTICLES
# =========================
def analyze_news_batch(articles):
    texts = []
    metadata = []

    articles = articles[:25]  # limit for speed

    for a in articles:
        text = a.get("title", "")

        if not text:
            continue

        short_text = shorten_text(text)

        texts.append(short_text)
        metadata.append(a)

    # Run sentiment in batch
    results = batch_sentiment(texts)

    label_map = {
        "POSITIVE": "positive",
        "NEGATIVE": "negative",
        "NEUTRAL": "neutral"
    }

    rows = []

    for r, a in zip(results, metadata):
        sentiment = label_map.get(r["label"], "neutral")
        confidence = float(r["score"])

        rows.append({
            "title": a.get("title"),
            "sentiment": sentiment,
            "confidence": confidence,
            "published_at": a.get("published_at"),
            "source": a.get("source")
        })

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Safety check
    if df.empty:
        return df

    # Compute importance + signal
    df["importance"] = df.apply(compute_importance, axis=1)
    df["signal"] = df.apply(signal_strength, axis=1)

    df["time_weight"] = df["published_at"].apply(time_decay)
    df["weighted_signal"] = df["signal"] * df["time_weight"]
    
    return df

# ---------------------------
# Importance & Signal
# ---------------------------

IMPORTANT_KEYWORDS = {
    "bitcoin": 0.3,
    "btc": 0.3,
    "ethereum": 0.3,
    "crypto": 0.2,
    "market": 0.2,
    "etf": 0.4,
    "sec": 0.4,
    "regulation": 0.3,
    "fed": 0.4,
    "inflation": 0.3,
    "interest": 0.3,
    "rates": 0.3,
    "crash": 0.5,
    "drop": 0.4,
    "surge": 0.5,
    "rally": 0.5,
    "bull": 0.4,
    "bear": 0.4,
    "ban": 0.4,
    "approval": 0.3
}

SOURCE_WEIGHTS = {
    "Reuters": 0.4,
    "Bloomberg": 0.4,
    "CoinDesk": 0.2,
    "Reddit/CryptoCurrency": 0.15
}

def compute_importance(row):
    text = str(row["title"]).lower()

    score = 0.2  # ✅ base so nothing is zero

    for keyword, weight in IMPORTANT_KEYWORDS.items():
        if keyword in text:
            score += weight

    score += SOURCE_WEIGHTS.get(row["source"], 0.1)

    return min(score, 1.0)


def signal_strength(row):
    sentiment_map = {
        "positive": 1,
        "neutral": 0,
        "negative": -1
    }

    return (
        sentiment_map.get(row["sentiment"], 0)
        * row["confidence"]
        * row["importance"]
    )
def compute_momentum(df):
    if df.empty:
        return 0

    return df["weighted_signal"].sum()
# =========================
# BTC PRICE
# =========================
cg = CoinGeckoAPI()

import time

BTC_CACHE = {
    "price": None,
    "timestamp": 0
}

def fetch_btc_price():
    now = time.time()

    # cache for 60 seconds
    if BTC_CACHE["price"] is not None and now - BTC_CACHE["timestamp"] < 60:
        return BTC_CACHE["price"]

    try:
        price_data = cg.get_price(ids='bitcoin', vs_currencies='usd')
        price = price_data['bitcoin']['usd']

        BTC_CACHE["price"] = price
        BTC_CACHE["timestamp"] = now

        return price

    except Exception as e:
        print("BTC fetch error:", e)

        # fallback to last known price
        return BTC_CACHE["price"] if BTC_CACHE["price"] else 0

# =========================
# MAIN PIPELINE
# =========================
def run_pipeline():
    articles = []

    articles += fetch_rss_news(COINDESK_RSS, "CoinDesk")
    articles += fetch_rss_news(COINTELEGRAPH_RSS, "Cointelegraph")
    articles += fetch_rss_news(REUTERS_RSS, "Reuters")

    df = analyze_news_batch(articles)
    print(df[["title", "sentiment", "confidence", "importance", "signal"]].head())
    market_signal = compute_momentum(df)
    btc_price = fetch_btc_price()

    print(f"\nBTC Price: ${btc_price}\n")

    df["abs_signal"] = df["signal"].abs()
    top = df.sort_values("abs_signal", ascending=False).head(10)

    colors = ['green' if x > 0 else 'red' for x in top["signal"]]

    plt.figure(figsize=(12,6))
    plt.barh(top["title"], top["signal"], color=colors)
    plt.xlabel("Signal Strength")
    plt.title("Top News Sentiment Signals")
    plt.gca().invert_yaxis()
    plt.show()

    save_cache()

    return df, btc_price, market_signal
