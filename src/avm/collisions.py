"""Collision reports (DMV form OL 316, 2014 to mid 2024).

Vehicle 1 is the autonomous test vehicle. The question tested here is whether collisions
in autonomous mode look different from collisions with the same vehicles in conventional
(human) mode, using a chi-square test on collision type and on the test vehicle's movement.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from .clean import RAW


def load(path: Path = RAW / "dmv_collisions.csv") -> pd.DataFrame:
    c = pd.read_csv(path, encoding="utf-8-sig")
    t = c["Collision Type"].astype(str).str.strip().str.lower().str.replace("-", " ")
    m = c["Mov Proceed Col Veh 1"].astype(str).str.strip().str.lower()
    return pd.DataFrame({
        "mode": c["Mode"].astype(str).str.strip().str.title(),
        "type": t.map({"rear end": "Rear end", "side swipe": "Side swipe", "broadside": "Broadside",
                       "hit object": "Hit object", "head on": "Head on"}).fillna("Other or unknown"),
        "av_movement": np.select(
            [m.str.contains("stopped"), m.str.contains("slowing"), m.str.contains("straight"),
             m.str.contains("turn"), m.str.contains("lane")],
            ["Stopped", "Slowing or stopping", "Going straight", "Turning", "Changing lanes"], "Other or unknown"),
        "av_at_fault": c["Fault"].astype(str).str.strip().str.lower().eq("vehicle 1"),
        "date": pd.to_datetime(c["Date"], format="%m/%d/%y", errors="coerce"),
    })


def analyse(c: pd.DataFrame) -> dict:
    c = c[c["mode"].isin(["Autonomous", "Conventional"])]
    out = {"reports": int(len(c)), "by_mode": c["mode"].value_counts().to_dict(),
           "first": str(c["date"].min().date()), "last": str(c["date"].max().date())}
    for col in ["type", "av_movement"]:
        tab = pd.crosstab(c[col], c["mode"])
        chi, p, dof, _ = chi2_contingency(tab)
        n = tab.to_numpy().sum()
        out[col] = {
            "table": {k: {m: int(v) for m, v in r.items()} for k, r in tab.iterrows()},
            "share": {m: (tab[m] / tab[m].sum()).round(4).to_dict() for m in tab.columns},
            "chi2": float(chi), "dof": int(dof), "p_value": float(p),
            "cramers_v": float(np.sqrt(chi / (n * (min(tab.shape) - 1)))),
        }
    auto = c[c["mode"] == "Autonomous"]
    out["autonomous_rear_end_share"] = float((auto["type"] == "Rear end").mean())
    out["autonomous_stopped_or_slowing_share"] = float(auto["av_movement"].isin(["Stopped", "Slowing or stopping"]).mean())
    out["autonomous_av_at_fault_share"] = float(auto["av_at_fault"].mean())
    out["conventional_av_at_fault_share"] = float(c[c["mode"] == "Conventional"]["av_at_fault"].mean())
    return out
