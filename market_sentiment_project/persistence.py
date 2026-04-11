"""
persistence.py — SQLite-backed signal history and price alert storage.

Tables:
  signal_history  — per-coin price + signal snapshots (one row per rerun)
  price_alerts    — user-defined alert rules (coin, type, threshold, active)
  alert_log       — fired alert history (so we don't re-fire every rerun)
"""

import sqlite3
import os
import pandas as pd
import time
from contextlib import contextmanager

DB_PATH = "signaldesk.db"

# ─────────────────────────────────
# CONNECTION HELPER
# ─────────────────────────────────
@contextmanager
def _db():
    """Thread-safe SQLite connection with WAL mode for concurrent reads."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

# ─────────────────────────────────
# SCHEMA INIT
# ─────────────────────────────────
def init_db():
    """Create tables if they don't exist. Safe to call on every startup."""
    with _db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS signal_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          REAL    NOT NULL,          -- unix timestamp
            coin        TEXT    NOT NULL,
            price       REAL    NOT NULL,
            change_24h  REAL    DEFAULT 0,
            signal      REAL    NOT NULL,
            pred_score  REAL    DEFAULT 0,
            momentum    REAL    DEFAULT 0,
            volatility  REAL    DEFAULT 0,
            fear_greed  REAL    DEFAULT 50
        );

        CREATE INDEX IF NOT EXISTS idx_sh_coin_ts
            ON signal_history(coin, ts DESC);

        CREATE TABLE IF NOT EXISTS price_alerts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            coin        TEXT    NOT NULL,
            alert_type  TEXT    NOT NULL,  -- 'price_above','price_below','signal_above','signal_below'
            threshold   REAL    NOT NULL,
            label       TEXT    DEFAULT '',
            active      INTEGER DEFAULT 1,
            created_at  REAL    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS alert_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_id    INTEGER REFERENCES price_alerts(id),
            fired_at    REAL    NOT NULL,
            value       REAL    NOT NULL,
            message     TEXT    NOT NULL
        );
        """)

# ─────────────────────────────────
# SIGNAL HISTORY  (write + read)
# ─────────────────────────────────
def append_history(coin: str, price: float, change_24h: float,
                   signal: float, pred_score: float = 0.0,
                   momentum: float = 0.0, volatility: float = 0.0,
                   fear_greed: float = 50.0):
    """Insert one snapshot row. Called on every rerun for every coin."""
    with _db() as conn:
        conn.execute(
            """INSERT INTO signal_history
               (ts, coin, price, change_24h, signal, pred_score,
                momentum, volatility, fear_greed)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (time.time(), coin, price, change_24h, signal,
             pred_score, momentum, volatility, fear_greed),
        )

def bulk_append_history(rows: list[dict]):
    """Batch insert — pass list of dicts with keys matching append_history args."""
    if not rows:
        return
    with _db() as conn:
        conn.executemany(
            """INSERT INTO signal_history
               (ts, coin, price, change_24h, signal, pred_score,
                momentum, volatility, fear_greed)
               VALUES (:ts, :coin, :price, :change_24h, :signal,
                       :pred_score, :momentum, :volatility, :fear_greed)""",
            rows,
        )

def load_history(coin: str, hours: int = 168) -> pd.DataFrame:
    """
    Load up to `hours` of history for a coin (default 7 days = 168h).
    Returns DataFrame with columns matching the table + a 'time' Timestamp col.
    """
    cutoff = time.time() - hours * 3600
    with _db() as conn:
        rows = conn.execute(
            """SELECT ts, coin, price, change_24h, signal, pred_score,
                      momentum, volatility, fear_greed
               FROM signal_history
               WHERE coin = ? AND ts >= ?
               ORDER BY ts ASC""",
            (coin, cutoff),
        ).fetchall()

    if not rows:
        return pd.DataFrame(columns=[
            "time", "coin", "price", "change_24h", "signal",
            "pred_score", "momentum", "volatility", "fear_greed",
        ])

    df = pd.DataFrame([dict(r) for r in rows])
    df["time"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_localize(None)
    return df

def load_all_history(hours: int = 168) -> pd.DataFrame:
    """Load history for all coins over `hours`."""
    cutoff = time.time() - hours * 3600
    with _db() as conn:
        rows = conn.execute(
            """SELECT ts, coin, price, change_24h, signal, pred_score,
                      momentum, volatility, fear_greed
               FROM signal_history
               WHERE ts >= ?
               ORDER BY coin, ts ASC""",
            (cutoff,),
        ).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df["time"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_localize(None)
    return df

def prune_history(keep_days: int = 30):
    """Delete rows older than keep_days. Call occasionally to bound DB size."""
    cutoff = time.time() - keep_days * 86400
    with _db() as conn:
        conn.execute("DELETE FROM signal_history WHERE ts < ?", (cutoff,))

def history_stats() -> dict:
    """Return row counts and oldest timestamp for the UI."""
    with _db() as conn:
        total = conn.execute("SELECT COUNT(*) FROM signal_history").fetchone()[0]
        oldest = conn.execute("SELECT MIN(ts) FROM signal_history").fetchone()[0]
    return {
        "total_rows": total,
        "oldest_ts":  oldest,
        "oldest_str": pd.Timestamp(oldest, unit="s").strftime("%Y-%m-%d %H:%M") if oldest else "—",
    }

# ─────────────────────────────────
# PRICE ALERTS  (CRUD)
# ─────────────────────────────────
ALERT_TYPES = {
    "price_above":  "Price rises above",
    "price_below":  "Price falls below",
    "signal_above": "Signal rises above",
    "signal_below": "Signal falls below",
}

def add_alert(coin: str, alert_type: str, threshold: float, label: str = "") -> int:
    """Create a new alert rule. Returns the new alert id."""
    assert alert_type in ALERT_TYPES, f"Unknown alert type: {alert_type}"
    with _db() as conn:
        cur = conn.execute(
            """INSERT INTO price_alerts (coin, alert_type, threshold, label, active, created_at)
               VALUES (?, ?, ?, ?, 1, ?)""",
            (coin, alert_type, threshold, label, time.time()),
        )
        return cur.lastrowid

def delete_alert(alert_id: int):
    with _db() as conn:
        conn.execute("DELETE FROM price_alerts WHERE id = ?", (alert_id,))

def toggle_alert(alert_id: int, active: bool):
    with _db() as conn:
        conn.execute("UPDATE price_alerts SET active = ? WHERE id = ?",
                     (1 if active else 0, alert_id))

def list_alerts() -> pd.DataFrame:
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM price_alerts ORDER BY coin, alert_type"
        ).fetchall()
    if not rows:
        return pd.DataFrame(columns=["id","coin","alert_type","threshold",
                                      "label","active","created_at"])
    return pd.DataFrame([dict(r) for r in rows])

def _recently_fired(alert_id: int, cooldown_minutes: int = 30) -> bool:
    """True if this alert fired within the last `cooldown_minutes`."""
    cutoff = time.time() - cooldown_minutes * 60
    with _db() as conn:
        row = conn.execute(
            "SELECT 1 FROM alert_log WHERE alert_id = ? AND fired_at > ? LIMIT 1",
            (alert_id, cutoff),
        ).fetchone()
    return row is not None

def _log_alert(alert_id: int, value: float, message: str):
    with _db() as conn:
        conn.execute(
            "INSERT INTO alert_log (alert_id, fired_at, value, message) VALUES (?,?,?,?)",
            (alert_id, time.time(), value, message),
        )

def check_alerts(prices: dict, coin_momentum: dict,
                 cooldown_minutes: int = 30) -> list[dict]:
    """
    Evaluate all active alerts against current prices and signals.
    Returns list of fired alert dicts: {alert_id, coin, message, value}.
    Respects cooldown so the same alert doesn't fire every 15 seconds.
    """
    alerts_df = list_alerts()
    if alerts_df.empty:
        return []

    fired = []
    active = alerts_df[alerts_df["active"] == 1]

    for _, row in active.iterrows():
        aid       = int(row["id"])
        coin      = row["coin"]
        atype     = row["alert_type"]
        threshold = float(row["threshold"])
        label     = row["label"] or ALERT_TYPES.get(atype, atype)

        price  = prices.get(coin, {}).get("price", 0)
        signal = coin_momentum.get(coin, 0.0)

        triggered = False
        value     = 0.0

        if atype == "price_above"  and price  > threshold: triggered, value = True, price
        elif atype == "price_below"  and price  < threshold: triggered, value = True, price
        elif atype == "signal_above" and signal > threshold: triggered, value = True, signal
        elif atype == "signal_below" and signal < threshold: triggered, value = True, signal

        if triggered and not _recently_fired(aid, cooldown_minutes):
            fmt_val = f"${value:,.2f}" if "price" in atype else f"{value:+.3f}"
            msg = f"[{coin}] {label} {threshold} — current: {fmt_val}"
            _log_alert(aid, value, msg)
            fired.append({"alert_id": aid, "coin": coin,
                          "message": msg, "value": value, "type": atype})

    return fired

def recent_alert_log(limit: int = 20) -> pd.DataFrame:
    """Return most recent fired alerts for the UI."""
    with _db() as conn:
        rows = conn.execute(
            """SELECT al.fired_at, al.message, al.value,
                      pa.coin, pa.alert_type, pa.threshold
               FROM alert_log al
               JOIN price_alerts pa ON al.alert_id = pa.id
               ORDER BY al.fired_at DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df["time"] = pd.to_datetime(df["fired_at"], unit="s").dt.strftime("%m/%d %H:%M")
    return df[["time", "coin", "alert_type", "threshold", "value", "message"]]
