"""Online monitoring dashboard.  Run:  streamlit run app/dashboard.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src import db, model
from src.config import load

st.set_page_config(page_title="Injection Molding Monitor", layout="wide")
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=60_000, key="auto")        # refresh every 60 s
except Exception:
    pass

cfg = load()
params = cfg["parameters"]
db.init()
bundle = model.load_bundle()
thr, metrics = bundle["thresholds"], bundle["metrics"]

st.title("Injection Molding Machine - Live Monitor")
if cfg["mode"] == "dummy":
    st.warning("DUMMY MODE: data is simulated. Predictions are for testing the system only.")

hours = st.sidebar.slider("Show last N hours", 6, 168, 48)
since = (pd.Timestamp.now() - pd.Timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
df = db.read_sql("""SELECT r.*, p.anomaly_score, p.is_anomaly, p.defect_prob
                    FROM readings r JOIN predictions p ON p.reading_id=r.id
                    WHERE r.ts >= ? ORDER BY r.ts""", (since,))
alerts = db.read_sql("SELECT ts, level, message FROM alerts ORDER BY ts DESC LIMIT 100")

if df.empty:
    st.info("No data yet. Run: python -m src.cli backfill --hours 72")
    st.stop()
df["ts"] = pd.to_datetime(df["ts"])

last_ts = df["ts"].max()
last_hour = df[df["ts"] > last_ts - pd.Timedelta(hours=1)]
recent_alerts = alerts[pd.to_datetime(alerts["ts"]) > last_ts - pd.Timedelta(hours=3)] if len(alerts) else alerts
crit = (recent_alerts["level"] == "CRITICAL").any() if len(recent_alerts) else False
warn = len(recent_alerts) > 0

status = "CRITICAL" if crit else "WARNING" if warn else "NORMAL"
color = {"CRITICAL": "#d62728", "WARNING": "#ff9f0e", "NORMAL": "#2ca02c"}[status]
st.markdown(f"<div style='padding:10px 16px;border-radius:8px;background:{color};color:white;"
            f"font-size:20px;font-weight:600'>Machine status: {status}</div>", unsafe_allow_html=True)
st.write("")

c = st.columns(5)
c[0].metric("Last data", last_ts.strftime("%d %b %H:%M"))
c[1].metric("Cycles (last hour)", len(last_hour))
c[2].metric("Abnormal cycles", f"{last_hour['is_anomaly'].mean():.1%}")
c[3].metric("Avg defect probability", f"{last_hour['defect_prob'].mean():.0%}" if last_hour["defect_prob"].notna().any() else "n/a",
            help="Only reliable if the defect model is 'trusted' (see Model tab)")
c[4].metric("Alerts (last 3 h)", len(recent_alerts))

tab1, tab2, tab3, tab4 = st.tabs(["Parameters", "Predictions", "Alerts", "Model & data"])

with tab1:
    sel = st.multiselect("Parameters", params, default=["Hydraulic_Oil_Temperature_C", "Injection_Pressure_bar", "Melt_Temperature_C"])
    hourly = df.set_index("ts")[params].resample("1h").mean()
    for p in sel:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["ts"], y=df[p], mode="lines", name="cycle", line=dict(width=1), opacity=0.4))
        fig.add_trace(go.Scatter(x=hourly.index, y=hourly[p], mode="lines+markers", name="hourly mean"))
        fig.add_hline(y=thr[p]["max"], line_dash="dash", line_color="red")
        fig.add_hline(y=thr[p]["min"], line_dash="dash", line_color="red")
        fig.update_layout(title=p, height=300, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")

with tab2:
    h = df.set_index("ts").resample("1h").agg({"anomaly_score": "mean", "is_anomaly": "mean", "defect_prob": "mean"})
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h.index, y=h["is_anomaly"], name="Abnormal cycle rate", mode="lines+markers"))
    fig.add_hline(y=cfg["anomaly"]["hour_alert_rate"], line_dash="dash", line_color="red")
    fig.update_layout(title="Share of abnormal cycles per hour", height=300, yaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch")
    if metrics["defect_model_trusted"]:
        st.plotly_chart(go.Figure(go.Scatter(x=h.index, y=h["defect_prob"], mode="lines+markers"))
                        .update_layout(title="Predicted defect probability (hourly mean)", height=300), width="stretch")
    else:
        st.info("Defect-probability chart hidden: the defect model is not reliable yet (see Model & data tab).")
    st.subheader("Latest predictions")
    st.dataframe(df.tail(100).sort_values("ts", ascending=False)[["ts", "anomaly_score", "is_anomaly", "defect_prob", "source"]],
                 width="stretch", hide_index=True)

with tab3:
    st.dataframe(alerts, width="stretch", hide_index=True) if len(alerts) else st.success("No alerts so far.")

with tab4:
    st.json(metrics)
    if not metrics["defect_model_trusted"]:
        st.warning("Defect model CV AUC is below the trust level, so it is NOT used for alerts. "
                   "Anomaly detection + limit alerts are active.")
    counts = db.read_sql("SELECT source, COUNT(*) AS rows FROM readings GROUP BY source")
    st.write("Rows in database by source"); st.dataframe(counts, hide_index=True)
    st.write("Alarm limits in use"); st.dataframe(pd.DataFrame(thr).T.round(2))
