"""Predict a lane change 3 s before it happens, and watch the model drift as traffic changes.

Samples are taken every 0.5 s for cars in the five mainline lanes. The label is whether a
clean lane change starts within the next 3 s. Three evaluations:
  * leaky: a random split of samples, so frames from the same car sit on both sides
  * grouped: whole cars held out, within the first period
  * over time: trained on 7:50 to 8:05 and tested on the later, more congested periods,
    then retrained with the newer period added, the way an automated retraining job would
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, train_test_split

from .ingest import MAINLINE, PERIODS

HORIZON = 30      # frames, 3 s
STEP = 5          # sample every 0.5 s
COOLDOWN = 20     # skip 2 s after a lane change, when the car is still settling
FEATURES = ["offset", "lat_v", "v", "a", "gap", "closing", "time_headway", "adv_left", "adv_right",
            "lane", "y", "vclass"]
SEED = 3
P1, P2, P3 = list(PERIODS)


def samples(df: pd.DataFrame, lc: pd.DataFrame) -> pd.DataFrame:
    d = df[df["lane"].isin(MAINLINE)].copy()
    centre = d.groupby("lane")["x"].median()
    d["offset"] = d["x"] - d["lane"].map(centre)
    ls = d.groupby(["period", "frame", "lane"])["v"].mean()
    own = ls.reindex(list(zip(d["period"], d["frame"], d["lane"]))).to_numpy()
    left = ls.reindex(list(zip(d["period"], d["frame"], d["lane"] - 1))).to_numpy()
    right = ls.reindex(list(zip(d["period"], d["frame"], d["lane"] + 1))).to_numpy()
    d["adv_left"] = left - own
    d["adv_right"] = right - own
    d = d[d["frame"] % STEP == 0]
    # label: a clean lane change starts in (t, t + 3 s]; drop samples just after a change
    ev = lc[["period", "vid", "frame"]].rename(columns={"frame": "lc"})
    m = d[["period", "vid", "frame"]].reset_index().merge(ev, on=["period", "vid"], how="left")
    m["ahead"] = (m["lc"] > m["frame"]) & (m["lc"] <= m["frame"] + HORIZON)
    m["after"] = (m["lc"] <= m["frame"]) & (m["lc"] > m["frame"] - COOLDOWN)
    lab = m.groupby("index").agg(target=("ahead", "any"), after=("after", "any"))
    d = d.join(lab)
    d = d[~d["after"]].drop(columns="after")
    d["gap"] = d["gap"].fillna(150.0)
    d["closing"] = d["closing"].fillna(0.0)
    d["vid_key"] = d["period"] + ":" + d["vid"].astype(str)
    return d.dropna(subset=["lat_v"])


def _fit(tr: pd.DataFrame):
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1)
    return m.fit(tr[FEATURES], tr["target"])


def _score(m, te: pd.DataFrame) -> dict:
    p = m.predict_proba(te[FEATURES])[:, 1]
    return {"auc": float(roc_auc_score(te["target"], p)), "ap": float(average_precision_score(te["target"], p)),
            "base_rate": float(te["target"].mean()), "samples": int(len(te))}


def psi(a: pd.Series, b: pd.Series, bins: int = 10) -> float:
    q = np.unique(np.nanquantile(a, np.linspace(0, 1, bins + 1)))
    pa = np.histogram(np.clip(a, q[0], q[-1]), q)[0] / len(a) + 1e-4
    pb = np.histogram(np.clip(b, q[0], q[-1]), q)[0] / len(b) + 1e-4
    return float(((pb - pa) * np.log(pb / pa)).sum())


def evaluate(s: pd.DataFrame) -> dict:
    p1, p2, p3 = (s[s["period"] == p] for p in (P1, P2, P3))
    tr, te = train_test_split(p1, test_size=0.25, random_state=SEED, stratify=p1["target"])
    leaky = _score(_fit(tr), te)
    shared = te["vid_key"].isin(set(tr["vid_key"])).mean()
    grouped = []
    for a, b in GroupKFold(n_splits=4).split(p1, p1["target"], p1["vid_key"]):
        grouped.append(_score(_fit(p1.iloc[a]), p1.iloc[b]))
    g = {k: float(np.mean([x[k] for x in grouped])) for k in ("auc", "ap", "base_rate")}
    m1 = _fit(p1)
    m12 = _fit(pd.concat([p1, p2]))
    imp = pd.Series(m1.booster_.feature_importance("gain"), index=FEATURES)
    return {
        "samples": {p: int((s["period"] == p).sum()) for p in PERIODS},
        "positive_rate": {p: float(s.loc[s["period"] == p, "target"].mean()) for p in PERIODS},
        "leaky_random_split": leaky, "leaky_test_cars_seen_in_training": float(shared),
        "grouped_by_car": g,
        "trained_on_first": {P2: _score(m1, p2), P3: _score(m1, p3)},
        "retrained_with_second": {P3: _score(m12, p3)},
        "psi_first_vs_last": {f: psi(p1[f].astype(float), p3[f].astype(float)) for f in
                              ["v", "gap", "time_headway", "closing", "adv_left", "adv_right", "lat_v", "offset"]},
        "importance_share": (imp / imp.sum()).sort_values(ascending=False).round(4).to_dict(),
    }
