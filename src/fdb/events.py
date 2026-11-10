"""Driving events from the trajectories: lane changes, hard braking and near misses,
and their rates per 1,000 vehicle km with exact Poisson intervals."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2

from .ingest import MAINLINE, OUT, PERIODS

DT = 0.1                 # seconds per frame
JITTER_FRAMES = 10       # a lane change undone within 1 s is tracking noise, not a manoeuvre
HARD_BRAKE = -3.0        # m/s^2
HARD_BRAKE_FRAMES = 5    # sustained for 0.5 s
PLAUSIBLE_ACCEL = 8.0    # m/s^2; beyond this the tracking, not the driver, is the likely cause
NEAR_MISS_TTC = 1.5      # seconds
NEAR_MISS_FRAMES = 3     # sustained for 0.3 s


def load(path=OUT) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["period"] = df["period"].astype(str)
    return df.sort_values(["period", "vid", "frame"]).reset_index(drop=True)


def _runs(mask: np.ndarray, groups: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """Start and end index of runs where mask is true for at least min_len rows within a group."""
    out, start = [], None
    for i in range(len(mask) + 1):
        on = i < len(mask) and mask[i] and (start is None or groups[i] == groups[start])
        if on and start is None:
            start = i
        elif not on and start is not None:
            if i - start >= min_len:
                out.append((start, i - 1))
            start = i if (i < len(mask) and mask[i]) else None
    return out


def lane_changes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Every switch between mainline lanes, with switches undone within 1 s removed as jitter."""
    ch = df[(df["prev_lane"].notna()) & (df["lane"] != df["prev_lane"])
            & df["lane"].isin(MAINLINE) & df["prev_lane"].isin(MAINLINE)].copy()
    ch["prev_lane"] = ch["prev_lane"].astype(int)
    ch = ch.sort_values(["period", "vid", "frame"])
    nxt = ch.groupby(["period", "vid"]).shift(-1)
    prv = ch.groupby(["period", "vid"]).shift(1)
    undone_next = (nxt["frame"] - ch["frame"] <= JITTER_FRAMES) & (nxt["lane"] == ch["prev_lane"])
    undone_prev = (ch["frame"] - prv["frame"] <= JITTER_FRAMES) & (prv["prev_lane"] == ch["lane"])
    jitter = undone_next | undone_prev
    clean = ch[~jitter].copy()
    clean["direction"] = np.where(clean["lane"] < clean["prev_lane"], "left", "right")
    return clean, {"raw_lane_switches": int(len(ch)), "jitter_switches": int(jitter.sum()),
                   "jitter_share": float(jitter.mean())}


def hard_brakes(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    g = (df["period"] + ":" + df["vid"].astype(str)).to_numpy()
    a = df["a"].to_numpy()
    runs = _runs(a < HARD_BRAKE, g, HARD_BRAKE_FRAMES)
    keep, implausible = [], 0
    for s, e in runs:
        if np.abs(a[s:e + 1]).max() > PLAUSIBLE_ACCEL:
            implausible += 1
            continue
        keep.append(s)
    return df.iloc[keep][["period", "vid", "frame", "v", "lane", "y"]].assign(min_a=[a[s:e + 1].min() for s, e in runs
                                                                                       if np.abs(a[s:e + 1]).max() <= PLAUSIBLE_ACCEL]), implausible


def near_misses(df: pd.DataFrame) -> pd.DataFrame:
    ok = (~df["leader_missing"]) & df["ttc"].notna() & (df["lane"] == df["lane_lead"])
    m = (ok & (df["ttc"] < NEAR_MISS_TTC)).to_numpy()
    g = (df["period"] + ":" + df["vid"].astype(str)).to_numpy()
    runs = _runs(m, g, NEAR_MISS_FRAMES)
    ttc = df["ttc"].to_numpy()
    idx = [s for s, _ in runs]
    return df.iloc[idx][["period", "vid", "frame", "v", "lane", "y"]].assign(min_ttc=[ttc[s:e + 1].min() for s, e in runs])


def poisson_ci(k: int, alpha: float = 0.05) -> tuple[float, float]:
    lo = 0.0 if k == 0 else chi2.ppf(alpha / 2, 2 * k) / 2
    return float(lo), float(chi2.ppf(1 - alpha / 2, 2 * k + 2) / 2)


def exposure(df: pd.DataFrame) -> pd.DataFrame:
    main = df[df["lane"].isin(MAINLINE)]
    return main.groupby("period").agg(vehicle_km=("v", lambda s: s.sum() * DT / 1000),
                                      vehicles=("vid", "nunique"), mean_speed=("v", "mean")).reset_index()


def rates(df: pd.DataFrame, events: dict[str, pd.DataFrame]) -> list[dict]:
    exp = exposure(df).set_index("period")
    rows = []
    for name, ev in events.items():
        ev = ev[ev["lane"].isin(MAINLINE)]
        for p in PERIODS:
            k = int((ev["period"] == p).sum())
            km = float(exp.loc[p, "vehicle_km"])
            lo, hi = poisson_ci(k)
            rows.append({"event": name, "period": p, "label": PERIODS[p], "count": k, "vehicle_km": km,
                         "per_1000_km": 1000 * k / km, "low": 1000 * lo / km, "high": 1000 * hi / km,
                         "mean_speed": float(exp.loc[p, "mean_speed"])})
    return rows


def compare(rows: list[dict], event: str, a: str, b: str) -> dict:
    ra = next(r for r in rows if r["event"] == event and r["period"] == a)
    rb = next(r for r in rows if r["event"] == event and r["period"] == b)
    n = ra["count"] + rb["count"]
    p = rb["vehicle_km"] / (ra["vehicle_km"] + rb["vehicle_km"])
    return {"event": event, "from": a, "to": b, "rate_ratio": rb["per_1000_km"] / ra["per_1000_km"],
            "p_value": float(binomtest(rb["count"], n, p).pvalue)}
