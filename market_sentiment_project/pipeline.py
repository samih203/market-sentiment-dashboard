# =========================
# IMPORTS
# =========================
import pandas as pd
import feedparser
import torch
import json
import os
import time
import tempfile
import numpy as np
from transformers import pipeline as hf_pipeline
from pycoingecko import CoinGeckoAPI

torch.set_grad_enabled(False)

# =========================
# COIN REGISTRY
# =========================
COINS = {
    "BTC":  {"id": "bitcoin",     "name": "Bitcoin",   "keywords": ["bitcoin", "btc", "satoshi", "halving"]},
    "ETH":  {"id": "ethereum",    "name": "Ethereum",  "keywords": ["ethereum", "eth", "vitalik", "eip", "gas fee", "defi"]},
    "SOL":  {"id": "solana",      "name": "Solana",    "keywords": ["solana", "sol", "solana network", "saga"]},
    "BNB":  {"id": "binancecoin", "name": "BNB",       "keywords": ["binance", "bnb", "bsc", "cz"]},
    "XRP":  {"id": "ripple",      "name": "XRP",       "keywords": ["ripple", "xrp", "garlinghouse", "sec ripple"]},
    "AVAX": {"id": "avalanche-2", "name": "Avalanche", "keywords": ["avalanche", "avax", "subnet"]},
    "DOGE": {"id": "dogecoin",    "name": "Dogecoin",  "keywords": ["dogecoin", "doge", "meme coin", "elon musk"]},
    "ADA":  {"id": "cardano",     "name": "Cardano",   "keywords": ["cardano", "ada", "hoskinson", "vasil"]},
}
COIN_IDS = [v["id"] for v in COINS.values()]

# =========================
# ARTICLE CACHE  (stale-data fallback)
# Stores the last successful pipeline result so a transient
# RSS / NLP failure serves yesterday's data instead of crashing.
# =========================
CACHE_FILE  = "article_cache.json"
STALE_CACHE = "pipeline_stale.json"   # persisted last-good pipeline output

def _load_json(path: str, default):
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return default

def _save_json_atomic(path: str, data) -> bool:
    """Write to a temp file then rename — prevents corrupt JSON on crash."""
    try:
        dir_ = os.path.dirname(os.path.abspath(path)) or "."
        fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)
        return True
    except Exception as e:
        print(f"Cache write error ({path}): {e}")
        return False

ARTICLE_CACHE: dict = _load_json(CACHE_FILE, {})

def save_cache():
    _save_json_atomic(CACHE_FILE, ARTICLE_CACHE)

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
# RSS FEEDS  (with timeout + retry)
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

# Track per-feed health so the UI can show which feeds are down
feed_status: dict[str, dict] = {
    name: {"ok": True, "last_ok": None, "error": None}
    for name in FEEDS
}

def _fetch_feed_with_retry(url: str, source_name: str,
                            timeout: int = 10, max_retries: int = 2) -> list[dict]:
    """
    Fetch a single RSS feed with:
      - connect/read timeout
      - exponential backoff retry (2 attempts)
      - per-feed status tracking
    Returns list of article dicts, empty on total failure.
    """
    import socket
    # feedparser respects socket default timeout
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            old_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(timeout)
            try:
                feed = feedparser.parse(url)
            finally:
                socket.setdefaulttimeout(old_timeout)

            # feedparser doesn't raise on HTTP errors — check bozo flag
            if feed.get("bozo") and not feed.entries:
                raise ValueError(f"Bozo feed: {feed.get('bozo_exception', 'unknown')}")

            articles = []
            for entry in feed.entries:
                title = entry.get("title", "").strip()
                if not title:
                    continue
                articles.append({
                    "title":        title,
                    "url":          entry.get("link", ""),
                    "published_at": pd.to_datetime(
                        entry.get("published", None), utc=True, errors="coerce"
                    ),
                    "source": source_name,
                })

            feed_status[source_name]["ok"]      = True
            feed_status[source_name]["last_ok"] = time.time()
            feed_status[source_name]["error"]   = None
            return articles

        except Exception as e:
            last_exc = e
            if attempt < max_retries:
                wait = 2 ** attempt   # 1s, 2s
                print(f"RSS retry {attempt+1}/{max_retries} ({source_name}): {e} — waiting {wait}s")
                time.sleep(wait)

    # All retries exhausted
    feed_status[source_name]["ok"]    = False
    feed_status[source_name]["error"] = str(last_exc)
    print(f"RSS fetch failed ({source_name}) after {max_retries+1} attempts: {last_exc}")
    return []


def fetch_all_news() -> tuple[list[dict], int]:
    """
    Fetch all feeds concurrently.
    Returns (articles, successful_feed_count).
    """
    articles = []
    ok_count = 0
    for name, url in FEEDS.items():
        result = _fetch_feed_with_retry(url, name)
        if result:
            ok_count += 1
        articles += result
    return articles, ok_count

# =========================
# COIN ROUTING
# =========================
def tag_coins(title: str) -> list[str]:
    text = title.lower()
    matched = [t for t, m in COINS.items() if any(kw in text for kw in m["keywords"])]
    return matched if matched else ["MACRO"]

# =========================
# IMPORTANCE & SIGNAL
# =========================
GLOBAL_IMPORTANT_KEYWORDS = {
    "etf": 1.2, "sec": 1.2, "crash": 2.0, "surge": 2.0,
    "inflation": 1.5, "fed": 1.5, "adoption": 1.0,
    "blackrock": 1.3, "halving": 1.3, "regulation": 1.1,
    "lawsuit": 1.2, "hack": 1.5, "ban": 1.4,
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
    text     = str(row["title"]).lower()
    ml_signal = row["sentiment"] * row["confidence"]
    boost    = 0.0
    if any(k in text for k in BEARISH_KEYWORDS): boost -= 0.7
    if any(k in text for k in BULLISH_KEYWORDS): boost += 0.7
    return (ml_signal * 0.6 + boost * 0.4) * row["importance"]

# =========================
# ANALYZE ARTICLES
# =========================
def analyze_news_batch(articles: list[dict]) -> pd.DataFrame:
    texts, metadata = [], []
    for a in articles[:80]:
        text = a.get("title", "").strip()
        if text:
            texts.append(text)
            metadata.append(a)

    if not texts:
        return pd.DataFrame()

    try:
        results = batch_sentiment(texts)
    except Exception as e:
        print(f"FinBERT inference error: {e}")
        return pd.DataFrame()

    rows = []
    for r, a in zip(results, metadata):
        try:
            sentiment = {"positive": 1, "negative": -1, "neutral": 0}.get(
                r["label"].lower(), 0
            )
            for coin in tag_coins(a["title"]):
                rows.append({
                    "title":        a["title"],
                    "url":          a.get("url", ""),
                    "sentiment":    sentiment,
                    "confidence":   float(r["score"]),
                    "published_at": a["published_at"],
                    "source":       a["source"],
                    "coin":         coin,
                    "macro_article": coin == "MACRO",
                })
        except Exception as e:
            print(f"Row error: {e}")

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["importance"]      = df.apply(compute_importance, axis=1)
    df["signal"]          = df.apply(signal_strength, axis=1)
    df.loc[df["macro_article"], "signal"] *= 0.4
    df["time_weight"]     = df["published_at"].apply(time_decay)
    df["weighted_signal"] = df["signal"] * df["time_weight"]
    return df

# =========================
# PRICE DATA  (retry + backoff)
# =========================
cg = CoinGeckoAPI()
_price_cache: dict = {"data": None, "ts": 0}

def _fetch_prices_with_retry(max_retries: int = 3, base_wait: float = 2.0) -> dict | None:
    """
    Fetch prices from CoinGecko with exponential backoff.
    Returns data dict on success, None on total failure.
    Handles 429 rate-limit responses explicitly.
    """
    ids_str = ",".join(COIN_IDS)
    for attempt in range(max_retries):
        try:
            raw = cg.get_price(
                ids=ids_str,
                vs_currencies="usd",
                include_24hr_change="true",
                include_market_cap="true",
            )
            data = {}
            for ticker, meta in COINS.items():
                cid = meta["id"]
                entry = raw.get(cid, {})
                data[ticker] = {
                    "price":      entry.get("usd", 0),
                    "change_24h": entry.get("usd_24h_change", 0.0),
                    "mcap":       entry.get("usd_market_cap", 0),
                }
            return data
        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = "429" in err_str or "rate limit" in err_str or "too many" in err_str
            wait = (base_wait ** attempt) * (3 if is_rate_limit else 1)
            print(f"CoinGecko attempt {attempt+1}/{max_retries}: {e} — wait {wait:.1f}s")
            if attempt < max_retries - 1:
                time.sleep(wait)
    return None

def fetch_prices(ttl: int = 30) -> dict:
    """
    Return cached prices if fresh, else fetch with retry.
    Falls back to last-known-good data if all retries fail —
    never returns zeros if we ever had valid prices.
    """
    now = time.time()
    if _price_cache["data"] and now - _price_cache["ts"] < ttl:
        return _price_cache["data"]

    fresh = _fetch_prices_with_retry()
    if fresh:
        _price_cache["data"] = fresh
        _price_cache["ts"]   = now
        return fresh

    # Retry failed — serve stale data if available
    if _price_cache["data"]:
        print("CoinGecko failed — serving stale price data")
        return _price_cache["data"]

    # Absolute fallback: zeros (first ever run with no network)
    return {t: {"price": 0, "change_24h": 0.0, "mcap": 0} for t in COINS}

def fetch_ohlc(coin_ticker: str = "BTC", days: int = 1) -> pd.DataFrame:
    coin_id = COINS.get(coin_ticker, {}).get("id", "bitcoin")
    for attempt in range(3):
        try:
            raw = cg.get_coin_ohlc_by_id(id=coin_id, vs_currency="usd", days=days)
            df  = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close"])
            df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            return df[["time", "open", "high", "low", "close"]]
        except Exception as e:
            wait = 2 ** attempt
            print(f"OHLC attempt {attempt+1}/3 ({coin_ticker}): {e} — wait {wait}s")
            if attempt < 2:
                time.sleep(wait)
    return pd.DataFrame()

# =========================
# PER-COIN MOMENTUM
# =========================
def compute_momentum(df: pd.DataFrame, coin: str | None = None) -> float:
    if df.empty:
        return 0.0
    subset = df[df["coin"].isin([coin, "MACRO"])] if coin and "coin" in df.columns else df
    if subset.empty:
        return 0.0
    weights = pd.Series(range(1, len(subset) + 1), dtype=float)
    weights /= weights.sum()
    return float((subset["signal"].values * weights.values).sum())

def compute_all_momentum(df: pd.DataFrame) -> dict[str, float]:
    return {ticker: compute_momentum(df, coin=ticker) for ticker in COINS}

# =========================
# FEAR & GREED
# =========================
def compute_fear_greed(market_signal: float, rolling_corr: float = 0.0,
                        signal_volatility: float = 0.0) -> float:
    sig  = np.clip(market_signal,           -1, 1)
    corr = np.clip(rolling_corr,            -1, 1)
    vol  = np.clip(1 - signal_volatility * 10, -1, 1)
    return round(float(np.clip((sig * 0.5 + corr * 0.3 + vol * 0.2 + 1) / 2 * 100, 0, 100)), 1)

def fear_greed_label(score: float) -> tuple[str, str]:
    if score >= 75: return "Extreme Greed", "#00d4a8"
    if score >= 55: return "Greed",         "#4ade80"
    if score >= 45: return "Neutral",        "#f5a623"
    if score >= 25: return "Fear",           "#fb923c"
    return                 "Extreme Fear",   "#ff4d6a"

# =========================
# STALE DATA PERSISTENCE
# Saves and loads the last successful pipeline output so a
# cold start with network issues can still show real data.
# =========================
def _df_to_records(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    d = df.copy()
    # Convert Timestamps to ISO strings for JSON serialisation
    for col in d.select_dtypes(include=["datetimetz", "datetime64"]).columns:
        d[col] = d[col].astype(str)
    return d.to_dict(orient="records")

def _records_to_df(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    return df

def save_stale_cache(df: pd.DataFrame, prices: dict,
                     market_signal: float, coin_momentum: dict):
    payload = {
        "ts":            time.time(),
        "market_signal": market_signal,
        "coin_momentum": coin_momentum,
        "prices":        prices,
        "articles":      _df_to_records(df),
    }
    _save_json_atomic(STALE_CACHE, payload)

def load_stale_cache() -> tuple[pd.DataFrame, dict, float, dict] | None:
    payload = _load_json(STALE_CACHE, None)
    if not payload:
        return None
    try:
        df            = _records_to_df(payload.get("articles", []))
        prices        = payload.get("prices", {t: {"price": 0, "change_24h": 0.0, "mcap": 0} for t in COINS})
        market_signal = float(payload.get("market_signal", 0.0))
        coin_momentum = payload.get("coin_momentum", {t: 0.0 for t in COINS})
        return df, prices, market_signal, coin_momentum
    except Exception as e:
        print(f"Stale cache load error: {e}")
        return None

# =========================
# MAIN PIPELINE
# =========================
def run_pipeline() -> tuple[pd.DataFrame, dict, float, dict, dict, bool]:
    """
    Returns:
        df, prices, market_signal, coin_momentum, feed_health, is_stale
    feed_health  — per-feed status dict (for UI display)
    is_stale     — True if we're serving cached data due to failures
    """
    articles, ok_feeds = fetch_all_news()
    is_stale = False

    if articles:
        df = analyze_news_batch(articles)
    else:
        df = pd.DataFrame()

    # If NLP produced nothing, try stale cache
    if df.empty:
        stale = load_stale_cache()
        if stale:
            print("Pipeline: no fresh articles — serving stale cache")
            df, prices, market_signal, coin_momentum = stale
            is_stale = True
            return df, prices, market_signal, coin_momentum, dict(feed_status), is_stale

    prices        = fetch_prices()
    coin_momentum = compute_all_momentum(df) if not df.empty else {t: 0.0 for t in COINS}
    market_signal = float(np.mean(list(coin_momentum.values()))) if coin_momentum else 0.0
    market_signal += np.random.normal(0, 0.01)

    save_cache()
    # Only persist stale cache when we have real data
    if not df.empty:
        save_stale_cache(df, prices, market_signal, coin_momentum)

    return df, prices, market_signal, coin_momentum, dict(feed_status), is_stale
