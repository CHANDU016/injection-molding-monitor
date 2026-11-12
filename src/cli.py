"""Command line:  python -m src.cli <command>"""
import argparse
import logging
import pandas as pd
from . import db, model, pipeline
from .config import load

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def serve():
    from apscheduler.schedulers.blocking import BlockingScheduler
    cfg = load()
    s = BlockingScheduler()
    s.add_job(pipeline.run_hour, "cron", minute=0, id="hourly", max_instances=1, coalesce=True)
    s.add_job(model.train, "cron", day_of_week=cfg["retrain_day"], hour=cfg["retrain_hour"], id="retrain")
    logging.info("Scheduler started: pipeline every hour at :00, retrain %s %02d:00",
                 cfg["retrain_day"], cfg["retrain_hour"])
    s.start()


def add_labels(csv_path):
    """CSV with columns: id,quality_result  (Good / Defective) -> updates stored readings."""
    df = pd.read_csv(csv_path)
    c = db.connect()
    c.executemany("UPDATE readings SET quality_result=? WHERE id=?",
                  list(zip(df["quality_result"], df["id"])))
    c.commit()
    c.close()
    print(f"Updated {len(df)} labels")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init", help="create database + train first model")
    b = sub.add_parser("backfill", help="generate N hours of past dummy data")
    b.add_argument("--hours", type=int, default=72)
    r = sub.add_parser("run-once", help="run one hourly cycle now")
    r.add_argument("--fault", choices=["oil_overheat", "pressure_drop", "heater_fail"])
    sub.add_parser("retrain", help="retrain models on stored data")
    sub.add_parser("purge-dummy", help="DELETE all dummy rows, then switch mode: real")
    sub.add_parser("serve", help="start the hourly scheduler (leave running)")
    l = sub.add_parser("add-labels", help="update quality labels from CSV (id,quality_result)")
    l.add_argument("csv")
    a = ap.parse_args()

    if a.cmd == "init":
        db.init()
        print(model.train())
    elif a.cmd == "backfill":
        db.init()
        model.load_bundle()
        pipeline.backfill(a.hours)
    elif a.cmd == "run-once":
        pipeline.run_hour(fault=a.fault)
    elif a.cmd == "retrain":
        print(model.train())
    elif a.cmd == "purge-dummy":
        n = db.purge_dummy()
        pipeline.export_predictions_csv()
        print(f"Deleted {n} dummy rows. Now set `mode: real` in config.yaml and run `retrain` when real data exists.")
    elif a.cmd == "serve":
        serve()
    elif a.cmd == "add-labels":
        add_labels(a.csv)


if __name__ == "__main__":
    main()
