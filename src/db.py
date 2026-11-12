"""SQLite storage. Swap for PostgreSQL/TimescaleDB later if needed."""
import sqlite3
import pandas as pd
from .config import load, path


def connect():
    cfg = load()
    p = path(cfg["database"])
    p.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(p)


def init():
    cfg = load()
    cols = ",\n".join(f"{c} REAL" for c in cfg["parameters"])
    c = connect()
    c.executescript(f"""
    CREATE TABLE IF NOT EXISTS readings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        {cols},
        quality_result TEXT,
        source TEXT NOT NULL DEFAULT 'real'
    );
    CREATE TABLE IF NOT EXISTS predictions (
        reading_id INTEGER PRIMARY KEY,
        ts TEXT,
        anomaly_score REAL,
        is_anomaly INTEGER,
        defect_prob REAL,
        model_version TEXT
    );
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT, level TEXT, key TEXT, message TEXT
    );
    CREATE INDEX IF NOT EXISTS ix_readings_ts ON readings(ts);
    """)
    c.commit()
    c.close()


def insert_readings(df):
    """Insert a DataFrame of readings; returns it with the new 'id' column."""
    cfg = load()
    cols = ["ts"] + cfg["parameters"] + ["quality_result", "source"]
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    ids = []
    c = connect()
    sql = f"INSERT INTO readings ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})"
    for row in df[cols].itertuples(index=False):
        cur = c.execute(sql, [None if pd.isna(v) else v for v in row])
        ids.append(cur.lastrowid)
    c.commit()
    c.close()
    df.insert(0, "id", ids)
    return df


def insert_predictions(pred_df):
    c = connect()
    pred_df.to_sql("predictions", c, if_exists="append", index=False)
    c.commit()
    c.close()


def insert_alert(ts, level, key, message):
    c = connect()
    c.execute("INSERT INTO alerts (ts, level, key, message) VALUES (?,?,?,?)",
              (ts, level, key, message))
    c.commit()
    c.close()


def recent_alert_exists(key, since_ts):
    c = connect()
    r = c.execute("SELECT 1 FROM alerts WHERE key=? AND ts>=? LIMIT 1",
                  (key, since_ts)).fetchone()
    c.close()
    return r is not None


def read_sql(sql, params=()):
    c = connect()
    df = pd.read_sql_query(sql, c, params=params)
    c.close()
    return df


def purge_dummy():
    """Delete every simulated row (and its predictions)."""
    c = connect()
    c.execute("DELETE FROM predictions WHERE reading_id IN (SELECT id FROM readings WHERE source='dummy')")
    n = c.execute("DELETE FROM readings WHERE source='dummy'").rowcount
    c.execute("DELETE FROM alerts WHERE key LIKE 'dummy:%'")
    c.commit()
    c.close()
    return n
