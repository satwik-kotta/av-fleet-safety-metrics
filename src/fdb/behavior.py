"""Lane change behavior clusters and a similar manoeuvre search.

Each clean lane change becomes one row of features measured from 3 s before to 3 s after
the moment the car crosses into the new lane. A Gaussian mixture, with the number of
clusters picked by BIC, groups them into behaviors. The same manoeuvres are also embedded
as short trajectory snippets and indexed in FAISS, so an engineer can pull up past lane
changes that looked like a new one.
"""
from __future__ import annotations

import faiss
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

WIN = 30                 # frames, 3 s each side
RISKY_TTC = 2.0          # a lane change that leaves a TTC under 2 s to the new leader
FEATURES = ["speed", "speed_change", "peak_lateral_speed", "duration", "gap_new_leader",
            "lane_speed_advantage", "min_ttc_after"]
SEED = 11


def lane_speeds(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["period", "frame", "lane"])["v"].mean().rename("lane_v").reset_index()


def features(df: pd.DataFrame, lc: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """One row per lane change with a full 6 s window, plus a snippet matrix for search."""
    key = df["period"] + ":" + df["vid"].astype(str)
    starts = key.ne(key.shift()).to_numpy().nonzero()[0]
    bounds = dict(zip(key.iloc[starts], zip(starts, np.r_[starts[1:], len(df)])))
    ls = lane_speeds(df).set_index(["period", "frame", "lane"])["lane_v"]
    X, x, v, a, ttc, lat = (df[c].to_numpy() for c in ["frame", "x", "v", "a", "ttc", "lat_v"])
    rows, snips = [], []
    for r in lc.itertuples():
        s, e = bounds[f"{r.period}:{r.vid}"]
        i = s + int(r.frame - X[s])
        if i - WIN < s or i + WIN >= e:
            continue
        w = slice(i - WIN, i + WIN + 1)
        lat_w = np.abs(lat[w])
        moving = lat_w > 0.3
        after = ttc[i:i + WIN + 1]
        after = after[~np.isnan(after)]
        old = ls.get((r.period, r.frame, int(r.prev_lane)), np.nan)
        new = ls.get((r.period, r.frame, int(r.lane)), np.nan)
        gap = df["gap"].iloc[i + 5] if not pd.isna(df["gap"].iloc[i + 5]) else np.nan
        rows.append({
            "period": r.period, "vid": r.vid, "frame": r.frame, "direction": r.direction,
            "from_lane": int(r.prev_lane), "to_lane": int(r.lane),
            "speed": v[i], "speed_change": v[i + WIN] - v[i - WIN],
            "peak_lateral_speed": np.nanmax(lat_w), "duration": moving.sum() * 0.1,
            "gap_new_leader": gap, "lane_speed_advantage": new - old,
            "min_ttc_after": min(after.min(), 10.0) if len(after) else 10.0,
            "min_accel": a[w].min(),
        })
        idx = np.arange(i - WIN, i + WIN + 1, 5)
        snips.append(np.r_[x[idx] - x[i], v[idx] - v[i]])
    f = pd.DataFrame(rows)
    # No leader in the new lane within view means an open lane: use a long gap. Negative gaps are
    # overlapping tracking boxes, set to zero.
    f["gap_new_leader"] = f["gap_new_leader"].fillna(f["gap_new_leader"].quantile(0.95)).clip(0, 100)
    f["lane_speed_advantage"] = f["lane_speed_advantage"].fillna(0.0)
    f["risky"] = f["min_ttc_after"] < RISKY_TTC
    return f, np.asarray(snips, dtype="float32")


def clusters(f: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    Z = StandardScaler().fit_transform(f[FEATURES])
    fits = {k: GaussianMixture(k, covariance_type="full", n_init=5, random_state=SEED).fit(Z) for k in range(2, 7)}
    bic = {k: float(m.bic(Z)) for k, m in fits.items()}
    best = min(bic, key=bic.get)
    f = f.assign(cluster=fits[best].predict(Z))
    prof = f.groupby("cluster").agg(
        lane_changes=("speed", "size"), speed=("speed", "median"), speed_change=("speed_change", "median"),
        peak_lateral_speed=("peak_lateral_speed", "median"), duration=("duration", "median"),
        gap_new_leader=("gap_new_leader", "median"), lane_speed_advantage=("lane_speed_advantage", "median"),
        risky_share=("risky", "mean"), left_share=("direction", lambda s: (s == "left").mean()))
    prof["share"] = prof["lane_changes"] / len(f)
    prof["name"] = [name_cluster(r, prof) for _, r in prof.iterrows()]
    return f, {"bic": bic, "k": best, "profiles": prof.reset_index().to_dict(orient="records")}


def name_cluster(r, prof) -> str:
    """Plain names from how a cluster compares with the others."""
    parts = []
    parts.append("fast traffic" if r["speed"] >= prof["speed"].median() else "slow traffic")
    if r["gap_new_leader"] <= prof["gap_new_leader"].min() * 1.15:
        parts.append("tight gap")
    elif r["gap_new_leader"] >= prof["gap_new_leader"].max() * 0.85:
        parts.append("open gap")
    if r["peak_lateral_speed"] >= prof["peak_lateral_speed"].max() * 0.9:
        parts.append("sharp")
    elif r["peak_lateral_speed"] <= prof["peak_lateral_speed"].min() * 1.15:
        parts.append("slow drift across the line")
    if r["lane_speed_advantage"] >= prof["lane_speed_advantage"].max() * 0.85 and r["lane_speed_advantage"] > 0.3:
        parts.append("into a faster lane")
    return ", ".join(parts)


def search(f: pd.DataFrame, snips: np.ndarray, k: int = 10) -> dict:
    """Index snippets in FAISS. For each risky lane change, how many of its k nearest
    lane changes by other vehicles were also risky, against the overall rate."""
    Z = StandardScaler().fit_transform(snips).astype("float32")
    index = faiss.IndexFlatL2(Z.shape[1])
    index.add(Z)
    _, nn = index.search(Z, 50)
    vid = (f["period"] + ":" + f["vid"].astype(str)).to_numpy()
    risky = f["risky"].to_numpy()
    prec = []
    for i in np.where(risky)[0]:
        nb = [j for j in nn[i] if j != i and vid[j] != vid[i]][:k]
        prec.append(risky[nb].mean())
    return {"lane_changes_indexed": int(len(Z)), "dims": int(Z.shape[1]), "k": k,
            "risky_queries": int(risky.sum()), "precision_at_k_for_risky": float(np.mean(prec)),
            "base_rate": float(risky.mean())}
