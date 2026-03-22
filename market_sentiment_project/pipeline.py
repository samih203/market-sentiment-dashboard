# =========================
# IMPORTS
# =========================
import pandas as pd
import feedparser
import torch
import json
import os
import time
import numpy as np
from transformers import pipeline as hf_pipeline
from pycoingecko import CoinGeckoAPI

torch.set_grad_enabled(False)

# =========================
# ARTICLE CACHE
# =========================
CACHE_FILE = "article_cache.json"

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        ARTICLE_CACHE = json.load(f)
else:
    ARTICLE_CACHE = {}

def save_cache():
    with open(CACHE_FILE, "w") as f:
        json.dump(ARTICLE_CACHE, f)

# =========================
# MODEL — module-level singleton
# app.py wraps get_sentiment_pipeline() in @st.cache_resource
# so the heavy model is only loaded once per Streamlit session.
# =========================
_sentiment_pipeline = None

def get_sentiment_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        _sentiment_pipeline = hf_pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            device=-1,
        )
    return _sentiment_pipeline

# =========================
# TIME DECAY
# =========================
def time_decay(published_at):
    if pd.isna(published_at):
        return 0.5
    try:
        published_at = pd.to_datetime(published_at, utc=True)
        now = pd.Timestamp.utcnow()
        hours_old = (now - published_at).total_seconds() / 3600
        return max(0.1, 1 / (1 + hours_old / 6))
    except Exception:
        return 0.5

# =========================
# BATCH SENTIMENT
# =========================
def batch_sentiment(texts, batch_size=16):
    nlp = get_sentiment_pipeline()
    return nlp(texts, batch_size=batch_size, truncation=True)

# =========================
# RSS SOURCES
# =========================
FEEDS = {
    "CoinDesk":      "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "Cointelegraph": "https://cointelegraph.com/rss",
    "Reuters":       "https://feeds.reuters.com/reuters/businessNews",
}

def fetch_rss_news(url, source_name):
    try:
        feed = feedparser.parse(url)
        articles = []
        for entry in feed.entries:
            articles.append({
                "title":        entry.get("title", ""),
                "url":          entry.get("link", ""),
                "published_at": pd.to_datetime(
                    entry.get("published", None), utc=True, errors="coerce"
                ),
                "source": source_name,
            })
        return articles
    except Exception as e:
        print(f"RSS fetch error ({source_name}): {e}")
        return []

def fetch_all_news():
    articles = []
    for name, url in FEEDS.items():
        articles += fetch_rss_news(url, name)
    return articles

# =========================
# IMPORTANCE & SIGNAL
# =========================
IMPORTANT_KEYWORDS = {
    "bitcoin":   1.0,
    "ethereum":  1.0,
    "etf":       1.2,
    "sec":       1.2,
    "crash":     2.0,
    "surge":     2.0,
    "inflation": 1.5,
    "fed":       1.5,
    "adoption":  1.0,
    "blackrock": 1.3,
    "halving":   1.3,
}

SOURCE_WEIGHTS = {
    "Reuters":               0.4,
    "Bloomberg":             0.4,
    "CoinDesk":              0.2,
    "Cointelegraph":         0.2,
    "Reddit/CryptoCurrency": 0.15,
}

BEARISH_KEYWORDS = ["crash", "collapse", "lawsuit", "ban", "hack",
                    "seized", "fraud", "scam", "plunge", "warning"]
BULLISH_KEYWORDS = ["surge", "rally", "approval", "etf", "adoption",
                    "record", "breakout", "bullish", "all-time high", "milestone"]

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
    ml_signal = row["sentiment"] * row["confidence"]

    keyword_boost = 0.0
    if any(k in text for k in BEARISH_KEYWORDS):
        keyword_boost -= 0.7
    if any(k in text for k in BULLISH_KEYWORDS):
        keyword_boost += 0.7

    raw   = (ml_signal * 0.6) + (keyword_boost * 0.4)
    final = raw * row["importance"]
    return final

# =========================
# ANALYZE ARTICLES
# =========================
def analyze_news_batch(articles):
    texts, metadata = [], []
    for a in articles[:50]:
        text = a.get("title", "").strip()
        if text:
            texts.append(text)
            metadata.append(a)

    if not texts:
        return pd.DataFrame()

    results = batch_sentiment(texts)
    rows = []
    for r, a in zip(results, metadata):
        try:
            sentiment = {"positive": 1, "negative": -1, "neutral": 0}.get(
                r["label"].lower(), 0
            )
            rows.append({
                "title":        a["title"],
                "url":          a.get("url", ""),
                "sentiment":    sentiment,
                "confidence":   float(r["score"]),
                "published_at": a["published_at"],
                "source":       a["source"],
            })
        except Exception as e:
            print("Row error:", e)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["importance"]      = df.apply(compute_importance, axis=1)
    df["signal"]          = df.apply(signal_strength, axis=1)
    df["time_weight"]     = df["published_at"].apply(time_decay)
    df["weighted_signal"] = df["signal"] * df["time_weight"]
    return df

# =========================
# PRICE DATA
# =========================
cg = CoinGeckoAPI()
_price_cache = {"data": None, "ts": 0}

def fetch_prices(ttl=60):
    now = time.time()
    if _price_cache["data"] and now - _price_cache["ts"] < ttl:
        return _price_cache["data"]
    try:
        raw = cg.get_price(
            ids="bitcoin,ethereum",
            vs_currencies="usd",
            include_24hr_change="true",
        )
        data = {
            "btc":        raw["bitcoin"]["usd"],
            "eth":        raw["ethereum"]["usd"],
            "btc_change": raw["bitcoin"].get("usd_24h_change", 0.0),
            "eth_change": raw["ethereum"].get("usd_24h_change", 0.0),
        }
        _price_cache["data"] = data
        _price_cache["ts"]   = now
        return data
    except Exception as e:
        print("Price fetch error:", e)
        return _price_cache["data"] or {"btc": 0, "eth": 0,
                                        "btc_change": 0.0, "eth_change": 0.0}

def fetch_ohlc(coin="bitcoin", days=1):
    """Return a DataFrame with columns [time, open, high, low, close]."""
    try:
        raw = cg.get_coin_ohlc_by_id(id=coin, vs_currency="usd", days=days)
        df  = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close"])
        df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df[["time", "open", "high", "low", "close"]]
    except Exception as e:
        print(f"OHLC fetch error ({coin}):", e)
        return pd.DataFrame()

# =========================
# MOMENTUM & FEAR/GREED
# =========================
def compute_momentum(df):
    if df.empty:
        return 0.0
    weights = pd.Series(range(1, len(df) + 1), dtype=float)
    weights /= weights.sum()
    return float((df["signal"] * weights).sum())

def compute_fear_greed(market_signal, rolling_corr=0.0, signal_volatility=0.0):
    """
    Returns a 0-100 Fear & Greed index.
    50 = neutral | 0 = extreme fear | 100 = extreme greed
    """
    sig_norm  = np.clip(market_signal,      -1, 1)
    corr_norm = np.clip(rolling_corr,       -1, 1)
    vol_norm  = np.clip(1 - signal_volatility * 10, -1, 1)

    composite = (sig_norm * 0.5) + (corr_norm * 0.3) + (vol_norm * 0.2)
    return round(float(np.clip((composite + 1) / 2 * 100, 0, 100)), 1)

def fear_greed_label(score):
    if score >= 75: return "Extreme Greed", "#00d4a8"
    if score >= 55: return "Greed",         "#4ade80"
    if score >= 45: return "Neutral",        "#f5a623"
    if score >= 25: return "Fear",           "#fb923c"
    return                 "Extreme Fear",   "#ff4d6a"

# =========================
# MAIN PIPELINE
# =========================
def run_pipeline():
    articles      = fetch_all_news()
    df            = analyze_news_batch(articles)
    prices        = fetch_prices()
    market_signal = compute_momentum(df) if not df.empty else 0.0
    market_signal += np.random.normal(0, 0.02)
    save_cache()
    return df, prices, market_signal
