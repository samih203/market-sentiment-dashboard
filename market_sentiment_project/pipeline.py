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
# COIN REGISTRY
# Every coin has: CoinGecko id, display name, ticker, and
# a set of keywords that route articles to its signal.
# =========================
COINS = {
    "BTC":  {"id": "bitcoin",          "name": "Bitcoin",   "keywords": ["bitcoin", "btc", "satoshi", "halving"]},
    "ETH":  {"id": "ethereum",         "name": "Ethereum",  "keywords": ["ethereum", "eth", "vitalik", "eip", "gas fee", "defi"]},
    "SOL":  {"id": "solana",           "name": "Solana",    "keywords": ["solana", "sol", "solana network", "saga"]},
    "BNB":  {"id": "binancecoin",      "name": "BNB",       "keywords": ["binance", "bnb", "bsc", "cz"]},
    "XRP":  {"id": "ripple",           "name": "XRP",       "keywords": ["ripple", "xrp", "garlinghouse", "sec ripple"]},
    "AVAX": {"id": "avalanche-2",      "name": "Avalanche", "keywords": ["avalanche", "avax", "subnet"]},
    "DOGE": {"id": "dogecoin",         "name": "Dogecoin",  "keywords": ["dogecoin", "doge", "meme coin", "elon musk"]},
    "ADA":  {"id": "cardano",          "name": "Cardano",   "keywords": ["cardano", "ada", "hoskinson", "vasil"]},
}

COIN_IDS = [v["id"] for v in COINS.values()]

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
# MODEL SINGLETON
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
# RSS FEEDS
# =========================
FEEDS = {
    "CoinDesk":      "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "Cointelegraph": "https://cointelegraph.com/rss",
    "Reuters":       "https://feeds.reuters.com/reuters/businessNews",
    "The Block":     "https://www.theblock.co/rss.xml",
    "Decrypt":       "https://decrypt.co/feed",
}

SOURCE_WEIGHTS = {
    "Reuters":       0.40,
    "Bloomberg":     0.40,
    "The Block":     0.30,
    "CoinDesk":      0.20,
    "Cointelegraph": 0.20,
    "Decrypt":       0.15,
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
# COIN ROUTING
# Tag each article with which coins it mentions.
# An article can be relevant to multiple coins.
# =========================
def tag_coins(title: str) -> list[str]:
    """Return list of coin tickers mentioned in the title."""
    text = title.lower()
    matched = []
    for ticker, meta in COINS.items():
        if any(kw in text for kw in meta["keywords"]):
            matched.append(ticker)
    # Macro/market-wide articles (no specific coin) → tag ALL coins with reduced weight
    if not matched:
        matched = ["MACRO"]
    return matched

# =========================
# IMPORTANCE & SIGNAL
# =========================
GLOBAL_IMPORTANT_KEYWORDS = {
    "etf":       1.2,
    "sec":       1.2,
    "crash":     2.0,
    "surge":     2.0,
    "inflation": 1.5,
    "fed":       1.5,
    "adoption":  1.0,
    "blackrock": 1.3,
    "halving":   1.3,
    "regulation":1.1,
    "lawsuit":   1.2,
    "hack":      1.5,
    "ban":       1.4,
}

BEARISH_KEYWORDS = ["crash", "collapse", "lawsuit", "ban", "hack",
                    "seized", "fraud", "scam", "plunge", "warning",
                    "exploit", "breach", "insolvent", "bankrupt"]
BULLISH_KEYWORDS = ["surge", "rally", "approval", "etf", "adoption",
                    "record", "breakout", "bullish", "all-time high",
                    "milestone", "partnership", "launch", "upgrade"]

def compute_importance(row):
    text = str(row["title"]).lower()
    score = 0.4
    for keyword, weight in GLOBAL_IMPORTANT_KEYWORDS.items():
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
# Returns a flat DataFrame where MACRO articles appear once,
# coin-specific articles appear once per tagged coin.
# =========================
def analyze_news_batch(articles):
    texts, metadata = [], []
    for a in articles[:80]:
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
            base = {
                "title":        a["title"],
                "url":          a.get("url", ""),
                "sentiment":    sentiment,
                "confidence":   float(r["score"]),
                "published_at": a["published_at"],
                "source":       a["source"],
            }
            tagged = tag_coins(a["title"])
            for coin in tagged:
                row = dict(base)
                row["coin"]         = coin
                row["macro_article"]= (coin == "MACRO")
                rows.append(row)
        except Exception as e:
            print("Row error:", e)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["importance"]      = df.apply(compute_importance, axis=1)
    df["signal"]          = df.apply(signal_strength, axis=1)
    # Macro articles get a dampened signal (broad market, not specific)
    df.loc[df["macro_article"], "signal"] *= 0.4
    df["time_weight"]     = df["published_at"].apply(time_decay)
    df["weighted_signal"] = df["signal"] * df["time_weight"]
    return df

# =========================
# PRICE DATA
# =========================
cg = CoinGeckoAPI()
_price_cache = {"data": None, "ts": 0}

def fetch_prices(ttl=30):
    now = time.time()
    if _price_cache["data"] and now - _price_cache["ts"] < ttl:
        return _price_cache["data"]
    try:
        ids_str = ",".join(COIN_IDS)
        raw = cg.get_price(
            ids=ids_str,
            vs_currencies="usd",
            include_24hr_change="true",
            include_market_cap="true",
        )
        data = {}
        for ticker, meta in COINS.items():
            cid = meta["id"]
            if cid in raw:
                data[ticker] = {
                    "price":     raw[cid].get("usd", 0),
                    "change_24h":raw[cid].get("usd_24h_change", 0.0),
                    "mcap":      raw[cid].get("usd_market_cap", 0),
                }
            else:
                data[ticker] = {"price": 0, "change_24h": 0.0, "mcap": 0}
        _price_cache["data"] = data
        _price_cache["ts"]   = now
        return data
    except Exception as e:
        print("Price fetch error:", e)
        return _price_cache["data"] or {
            t: {"price": 0, "change_24h": 0.0, "mcap": 0} for t in COINS
        }

def fetch_ohlc(coin_ticker="BTC", days=1):
    """Return OHLC DataFrame for a given ticker symbol."""
    coin_id = COINS.get(coin_ticker, {}).get("id", "bitcoin")
    try:
        raw = cg.get_coin_ohlc_by_id(id=coin_id, vs_currency="usd", days=days)
        df  = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close"])
        df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df[["time", "open", "high", "low", "close"]]
    except Exception as e:
        print(f"OHLC fetch error ({coin_ticker}):", e)
        return pd.DataFrame()

# =========================
# PER-COIN MOMENTUM
# =========================
def compute_momentum(df, coin=None):
    """
    Compute momentum for a specific coin, or overall if coin=None.
    Uses exponential-weighted sum so recent articles matter more.
    """
    if df.empty:
        return 0.0
    if coin and "coin" in df.columns:
        subset = df[df["coin"].isin([coin, "MACRO"])]
    else:
        subset = df
    if subset.empty:
        return 0.0
    weights = pd.Series(range(1, len(subset) + 1), dtype=float)
    weights /= weights.sum()
    return float((subset["signal"].values * weights.values).sum())

def compute_all_momentum(df):
    """Return dict of {ticker: momentum_score} for all coins."""
    scores = {}
    for ticker in COINS:
        scores[ticker] = compute_momentum(df, coin=ticker)
    return scores

# =========================
# FEAR & GREED
# =========================
def compute_fear_greed(market_signal, rolling_corr=0.0, signal_volatility=0.0):
    sig_norm  = np.clip(market_signal,           -1, 1)
    corr_norm = np.clip(rolling_corr,            -1, 1)
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
    articles        = fetch_all_news()
    df              = analyze_news_batch(articles)
    prices          = fetch_prices()
    coin_momentum   = compute_all_momentum(df)
    # Overall market signal = weighted average of all coin momentums
    market_signal   = float(np.mean(list(coin_momentum.values()))) if coin_momentum else 0.0
    market_signal  += np.random.normal(0, 0.01)
    save_cache()
    return df, prices, market_signal, coin_momentum
