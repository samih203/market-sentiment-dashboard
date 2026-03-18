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
        body = get_article_content(a)
        headline = a.get("title", "")

        full_text = headline + ". " + body
        short_text = shorten_text(full_text)

        if not short_text:
            continue

        texts.append(short_text)
        metadata.append(a)

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
            "title": a["title"],
            "sentiment": sentiment,
            "confidence": confidence,
            "published_at": a["published_at"],
            "source": a["source"]
        })

    return pd.DataFrame(rows)

# =========================
# SIGNAL LOGIC
# =========================
SOURCE_WEIGHTS = {
    "Reuters": 0.4,
    "Bloomberg": 0.4,
    "CoinDesk": 0.2,
    "Cointelegraph": 0.2
}

def compute_importance(row):
    base = SOURCE_WEIGHTS.get(row["source"], 0.1)
    confidence_bonus = row["confidence"] * 0.3
    return min(base + confidence_bonus, 1.0)

def signal_strength(row):
    sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
    return sentiment_map[row["sentiment"]] * row["confidence"] * row["importance"]

# =========================
# BTC PRICE
# =========================
cg = CoinGeckoAPI()

def fetch_btc_price():
    price_data = cg.get_price(ids='bitcoin', vs_currencies='usd')
    return price_data['bitcoin']['usd']

# =========================
# MAIN PIPELINE
# =========================
def run_pipeline():
    articles = []

    articles += fetch_rss_news(COINDESK_RSS, "CoinDesk")
    articles += fetch_rss_news(COINTELEGRAPH_RSS, "Cointelegraph")
    articles += fetch_rss_news(REUTERS_RSS, "Reuters")

    df = analyze_news_batch(articles)

    df["importance"] = df.apply(compute_importance, axis=1)
    df["signal"] = df.apply(signal_strength, axis=1)

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

    return df, btc_price

# =========================
# RUN
# =========================
df = run_pipeline()
