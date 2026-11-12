# Injection Molding Monitor (dummy-data phase)

Hourly pipeline: **collect -> store -> predict -> alert -> update prediction file**, plus a live dashboard.
Everything simulated is tagged `source = dummy`, so it can be deleted with one command.

## 1. Setup (once)
```
pip install -r requirements.txt
python -m src.cli init                  # creates DB, trains first model from your CSV
python -m src.cli backfill --hours 72   # 3 days of dummy history for the dashboard
```

## 2. Run (two terminals, leave both running)
```
python -m src.cli serve                 # every hour at :00 -> new data, predictions, alerts
streamlit run app/dashboard.py          # dashboard at http://localhost:8501 (auto-refresh 60 s)
```
Output prediction file (rewritten every hour): `data/predictions.csv`

## 3. Test the alerts
```
python -m src.cli run-once --fault oil_overheat     # or pressure_drop / heater_fail
```
Alerts appear in the console, the dashboard Alerts tab and (if enabled) Telegram / email.

## 4. Turn on notifications
Edit `config.yaml` -> `alerts.telegram.enabled: true` and set environment variables
`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` (create the bot with @BotFather).
Email: `alerts.email.enabled: true`, set `SMTP_USER`, `SMTP_PASSWORD`.

## 5. Switch to real data (after a few days)
1. `python -m src.cli purge-dummy`            # deletes ALL dummy rows + their predictions
2. In `config.yaml` set `mode: real`
3. Implement `RealCollector.read_cycles()` in `src/collector.py` (OPC UA / Modbus / MQTT / SQL).
   It must return: `ts`, the 10 parameter columns, `quality_result` (or None), `source="real"`.
4. Add quality labels as inspection results arrive:
   `python -m src.cli add-labels labels.csv`   (columns: `id,quality_result`)
5. After >= 500 real rows the weekly retrain automatically trains on real data only
   (`python -m src.cli retrain` to do it manually).

## How the prediction works
| Part | Needs labels? | Used for |
|---|---|---|
| Limit alerts (min/max per parameter) | No | Immediate out-of-range warning |
| Drift alert (hourly mean vs baseline) | No | Slow shifts, e.g. oil temp creeping up |
| Anomaly detection (IsolationForest) | No | "Cycle looks unlike normal operation" |
| Defect classifier (RandomForest) | Yes | Defect probability. Only used for alerts when CV AUC >= 0.65 |

With the current dummy data the defect model scores AUC ~0.57 (no real signal), so it is
automatically marked "not trusted" and does not trigger alerts. This will change with real labelled data.

## Files
```
config.yaml          all settings (limits, schedule, alerts)
src/collector.py     dummy / real data source
src/db.py            SQLite tables: readings, predictions, alerts
src/model.py         training + prediction
src/alerts.py        rules + Telegram/email
src/pipeline.py      the hourly job
src/cli.py           commands + scheduler
app/dashboard.py     Streamlit live dashboard
```
