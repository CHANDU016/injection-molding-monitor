"""Two models:
 1. Anomaly detector (IsolationForest) -> works WITHOUT labels, main early-warning tool.
 2. Defect classifier (RandomForest)   -> needs good labels. Its cross-validated AUC is stored,
    and it is only 'trusted' for alerts when AUC >= defect.min_cv_auc.
"""
import json
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from . import db
from .config import load, path


def _mdir():
    d = path(load()["model_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def model_exists():
    return (_mdir() / "bundle.joblib").exists()


def training_data(cfg):
    """Real data once enough exists, otherwise everything in DB, otherwise the seed CSV."""
    params = cfg["parameters"]
    cols = ["quality_result", "source"] + params
    df = db.read_sql(f"SELECT {','.join(cols)} FROM readings")
    if len(df):
        real = df[df.source == "real"]
        if len(real) >= cfg["min_real_rows_to_train"]:
            return real, "real"
        if len(df) >= 300:
            return df, "all"
    seed = pd.read_csv(path(cfg["seed_csv"]))
    seed = seed.rename(columns={"Quality_Result": "quality_result"})
    return seed, "seed_csv"


def train():
    cfg = load()
    params = cfg["parameters"]
    df, used = training_data(cfg)
    X = df[params].astype(float)

    anomaly = make_pipeline(
        StandardScaler(),
        IsolationForest(n_estimators=300, contamination=cfg["anomaly"]["contamination"], random_state=0),
    ).fit(X)

    metrics = {"trained_at": datetime.now().isoformat(timespec="seconds"),
               "training_rows": int(len(X)), "training_source": used,
               "defect_model_trusted": False, "cv_auc": None}

    defect = None
    lab = df.dropna(subset=["quality_result"])
    y = (lab["quality_result"] == "Defective").astype(int)
    if len(lab) >= 100 and y.nunique() == 2:
        rf = RandomForestClassifier(300, min_samples_leaf=5, class_weight="balanced", random_state=0, n_jobs=-1)
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        auc = float(cross_val_score(rf, lab[params], y, cv=cv, scoring="roc_auc").mean())
        defect = rf.fit(lab[params], y)
        metrics.update(cv_auc=round(auc, 3), defect_model_trusted=auc >= cfg["defect"]["min_cv_auc"])

    # baseline statistics for drift detection
    baseline = {p: {"mean": float(X[p].mean()), "std": float(X[p].std())} for p in params}

    # alarm thresholds (auto from data, then overrides)
    k = cfg["thresholds"]["auto_sigma"]
    thr = {p: {"min": baseline[p]["mean"] - k * baseline[p]["std"],
               "max": baseline[p]["mean"] + k * baseline[p]["std"]} for p in params}
    for p, o in (cfg["thresholds"].get("overrides") or {}).items():
        thr[p].update(o)

    metrics["version"] = datetime.now().strftime("%Y%m%d%H%M%S")
    joblib.dump({"anomaly": anomaly, "defect": defect, "baseline": baseline,
                 "thresholds": thr, "metrics": metrics}, _mdir() / "bundle.joblib")
    (_mdir() / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def load_bundle():
    if not model_exists():
        train()
    return joblib.load(_mdir() / "bundle.joblib")


def predict(df):
    """df must contain 'id', 'ts' and parameter columns. Returns the predictions table rows."""
    cfg = load()
    b = load_bundle()
    X = df[cfg["parameters"]].astype(float)
    score = -b["anomaly"].decision_function(X)           # higher = more abnormal
    is_anom = (b["anomaly"].predict(X) == -1).astype(int)
    prob = b["defect"].predict_proba(X)[:, 1] if b["defect"] is not None else np.full(len(X), np.nan)
    return pd.DataFrame({"reading_id": df["id"].values, "ts": df["ts"].values,
                         "anomaly_score": score.round(4), "is_anomaly": is_anom,
                         "defect_prob": np.round(prob, 4),
                         "model_version": b["metrics"]["version"]})
