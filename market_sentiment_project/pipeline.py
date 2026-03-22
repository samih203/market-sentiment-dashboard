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
    model="ProsusAI/finbert",
    device=-1
)

# ==========================
# TIME DECAY
# ==========================
from datetime import datetime
import pandas as pd

def time_decay(published_at):
    if pd.isna(published_at):
        return 0.5
    try:
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
    return article.get("title", "")

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
# IMPORTANCE & SIGNAL
# =========================
IMPORTANT_KEYWORDS = {
    "bitcoin": 1.0,
    "ethereum": 1.0,
    "etf": 1.2,
    "sec": 1.2,
    "crash": 2.0,
    "surge": 2.0,
    "inflation": 1.5,
    "fed": 1.5
}

SOURCE_WEIGHTS = {
    "Reuters": 0.4,
    "Bloomberg": 0.4,
    "CoinDesk": 0.2,
    "Cointelegraph": 0.2,
    "Reddit/CryptoCurrency": 0.15
}

def compute_importance(row):
    text = str(row["title"]).lower()
    score = 0.4
    for keyword, weight in IMPORTANT_KEYWORDS.items():
        if keyword in text:
            score += weight
    score += SOURCE_WEIGHTS.get(row["source"], 0.1)
    return min(score, 1.0)


def signal_strength(row):
    text = str(row["title"]).lower()

    # Base ML signal
    ml_signal = row["sentiment"] * row["confidence"]

    # Keyword override (STRONG directional bias)
    keyword_boost = 0
    if any(k in text for k in ["crash", "collapse", "lawsuit", "ban", "hack"]):
        keyword_boost -= 0.7
    if any(k in text for k in ["surge", "rally", "approval", "etf", "adoption"]):
        keyword_boost += 0.7

    # Combine
    raw = (ml_signal * 0.6) + (keyword_boost * 0.4)

    # Apply importance AFTER direction is established
    final = raw * row["importance"]

    return final  # FIX: removed dead `return raw` below

# =========================
# ANALYZE ARTICLES
# =========================
def analyze_news_batch(articles):
    texts = []
    metadata = []

    for a in articles[:50]:
        text = a.get("title", "")
        if text:
            texts.append(text)
            metadata.append(a)

    if not texts:
        return pd.DataFrame()

    results = batch_sentiment(texts)
    rows = []

    for i in range(len(results)):
        r = results[i]
        a = metadata[i]
        try:
            sentiment = {
                "positive": 1,
                "negative": -1,
                "neutral": 0
            }.get(r["label"].lower(), 0)

            rows.append({
                "title": a["title"],
                "sentiment": sentiment,
                "confidence": float(r["score"]),
                "published_at": a["published_at"],
                "source": a["source"]
            })
        except Exception as e:
            print("ROW ERROR:", e)

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df["importance"] = df.apply(compute_importance, axis=1)
    df["signal"] = df.apply(signal_strength, axis=1)
    df["time_weight"] = df["published_at"].apply(time_decay)
    df["weighted_signal"] = df["signal"] * df["time_weight"]

    return df

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
        return BTC_CACHE["price"] if BTC_CACHE["price"] else 0

def fetch_prices():
    try:
        data = cg.get_price(
            ids="bitcoin,ethereum",
            vs_currencies="usd"
        )
        return {
            "btc": data["bitcoin"]["usd"],
            "eth": data["ethereum"]["usd"]
        }
    except Exception as e:
        print("Price fetch error:", e)
        return {"btc": 0, "eth": 0}

# =========================
# MOMENTUM
# =========================
def compute_momentum(df):
    if df.empty:
        return 0
    weights = pd.Series(range(1, len(df) + 1))
    weights = weights / weights.sum()
    return (df["signal"] * weights).sum()

# =========================
# MAIN PIPELINE
# =========================
def run_pipeline():
    articles = []
    articles += fetch_rss_news(COINDESK_RSS, "CoinDesk")
    articles += fetch_rss_news(COINTELEGRAPH_RSS, "Cointelegraph")
    articles += fetch_rss_news(REUTERS_RSS, "Reuters")

    df = analyze_news_batch(articles)

    if not df.empty:
        print(df[["title", "sentiment", "confidence", "importance", "signal"]].head(10))

    import numpy as np
    market_signal = compute_momentum(df)
    market_signal += np.random.normal(0, 0.02)

    prices = fetch_prices()
    btc_price = prices["btc"]
    eth_price = prices["eth"]

    print(f"\nBTC Price: ${btc_price}")
    print(f"ETH Price: ${eth_price}\n")

    if not df.empty:
        print("DEBUG SIGNALS:")
        print(df[["sentiment", "confidence", "importance", "signal"]].head(10))

    save_cache()

    return df, btc_price, eth_price, market_signal
