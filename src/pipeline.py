"""The hourly job: collect -> store -> predict -> alert -> refresh prediction file."""
import logging
import pandas as pd
from . import alerts, db, model
from .collector import get_collector
from .config import load, path

log = logging.getLogger("pipeline")


def export_predictions_csv():
    cfg = load()
    df = db.read_sql("""
        SELECT r.id, r.ts, r.*, p.anomaly_score, p.is_anomaly, p.defect_prob, p.model_version
        FROM readings r JOIN predictions p ON p.reading_id = r.id ORDER BY r.ts""")
    df = df.loc[:, ~df.columns.duplicated()]
    out = path(cfg["predictions_csv"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


def run_hour(ts_end=None, fault=None, notify=True):
    cfg = load()
    db.init()
    ts_end = pd.Timestamp(ts_end) if ts_end is not None else pd.Timestamp.now().floor("s")
    col = get_collector(cfg)

    df = col.read_cycles(ts_end, fault=fault)                 # 1. collect
    df = db.insert_readings(df)                               # 2. store
    preds = model.predict(df)                                 # 3. predict
    db.insert_predictions(preds)
    found = alerts.evaluate(df, preds)                        # 4. alert rules
    sent = alerts.dispatch(found, ts_end, prefix=f"{col.source}:") if notify else []
    export_predictions_csv()                                  # 5. refresh prediction file
    log.info("Hour done: %d cycles, %d anomalies, %d new alerts",
             len(df), int(preds.is_anomaly.sum()), len(sent))
    return df, preds, sent


def backfill(hours, fault_last_hours=0):
    """Generate `hours` of past dummy data (hourly steps) so the dashboard has history."""
    now = pd.Timestamp.now().floor("h")
    for h in range(hours, 0, -1):
        fault = None
        run_hour(now - pd.Timedelta(hours=h - 1), fault=fault, notify=False)
