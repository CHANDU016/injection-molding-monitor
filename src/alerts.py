"""Rule checks + notifications (console, Telegram, email)."""
import logging
import os
import smtplib
from email.message import EmailMessage
import numpy as np
import pandas as pd
import requests
from . import db
from .config import load
from .model import load_bundle

log = logging.getLogger("alerts")


def evaluate(readings, preds):
    """Return list of (level, key, message) for the latest hour of data."""
    cfg = load()
    b = load_bundle()
    out = []
    n = len(readings)

    for p, t in b["thresholds"].items():
        s = readings[p]
        n_out = int(((s < t["min"]) | (s > t["max"])).sum())
        mean = s.mean()
        if mean < t["min"] or mean > t["max"]:
            out.append(("CRITICAL", f"limit:{p}",
                        f"{p}: hourly mean {mean:.2f} is outside limits [{t['min']:.1f}, {t['max']:.1f}]"))
        elif n_out >= 3:
            out.append(("WARNING", f"limit:{p}",
                        f"{p}: {n_out}/{n} cycles outside limits [{t['min']:.1f}, {t['max']:.1f}]"))

        base = b["baseline"][p]
        z = (mean - base["mean"]) / (base["std"] / np.sqrt(n) + 1e-9)
        if abs(z) > cfg["anomaly"]["drift_sigma"]:
            out.append(("WARNING", f"drift:{p}",
                        f"{p}: hourly mean {mean:.2f} drifted from baseline {base['mean']:.2f} (z={z:.1f})"))

    rate = preds["is_anomaly"].mean()
    if rate > cfg["anomaly"]["hour_alert_rate"]:
        out.append(("WARNING", "anomaly_rate",
                    f"{rate:.0%} of cycles this hour look abnormal (limit {cfg['anomaly']['hour_alert_rate']:.0%})"))

    if b["metrics"]["defect_model_trusted"]:
        mp = preds["defect_prob"].mean()
        if mp > cfg["defect"]["alert_threshold"]:
            out.append(("CRITICAL", "defect_prob", f"Mean predicted defect probability {mp:.0%}"))
    return out


def dispatch(alerts, ts, prefix=""):
    """Store alerts (respecting cooldown) and send notifications. Returns alerts actually sent."""
    cfg = load()
    cool = pd.Timedelta(minutes=cfg["alerts"]["cooldown_minutes"])
    since = (pd.Timestamp(ts) - cool).strftime("%Y-%m-%d %H:%M:%S")
    sent = []
    for level, key, msg in alerts:
        key = prefix + key
        if db.recent_alert_exists(key, since):
            continue
        db.insert_alert(pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S"), level, key, msg)
        sent.append((level, msg))
        log.warning("[%s] %s", level, msg)
    if sent:
        notify("\n".join(f"[{l}] {m}" for l, m in sent), cfg)
    return sent


def notify(text, cfg):
    text = "Injection Molding Machine ALERT\n" + text
    tg = cfg["alerts"]["telegram"]
    if tg.get("enabled"):
        try:
            token, chat = os.environ["TELEGRAM_BOT_TOKEN"], os.environ["TELEGRAM_CHAT_ID"]
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": chat, "text": text}, timeout=10)
        except Exception as e:
            log.error("Telegram failed: %s", e)
    em = cfg["alerts"]["email"]
    if em.get("enabled"):
        try:
            msg = EmailMessage()
            msg["Subject"], msg["From"], msg["To"] = "Molding machine alert", os.environ["SMTP_USER"], ", ".join(em["to"])
            msg.set_content(text)
            with smtplib.SMTP(em["smtp_host"], em["smtp_port"]) as s:
                s.starttls()
                s.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
                s.send_message(msg)
        except Exception as e:
            log.error("Email failed: %s", e)
