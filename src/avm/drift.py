"""Drift monitoring on the filings themselves.

Two signals per company and year, each against the company's previous year:
  * PSI on the mix of cause categories (did what goes wrong change?)
  * template novelty, the share of events whose cause sentence the company never used the
    year before (did the way they write it up change?)
A big PSI with high novelty usually means a change in reporting, not in driving, which is
exactly the case a metrics dashboard should flag before anyone reads a trend into it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PSI_ALERT = 0.25
NOVELTY_ALERT = 0.5
MIN_EVENTS = 30


def psi(p: pd.Series, q: pd.Series, eps: float = 1e-3) -> float:
    idx = p.index.union(q.index)
    p = p.reindex(idx, fill_value=0) + eps
    q = q.reindex(idx, fill_value=0) + eps
    p, q = p / p.sum(), q / q.sum()
    return float(((p - q) * np.log(p / q)).sum())


def drift_table(ev: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c, g in ev.groupby("company"):
        years = sorted(g["year"].unique())
        for a, b in zip(years, years[1:]):
            A, B = g[g["year"] == a], g[g["year"] == b]
            if len(A) < MIN_EVENTS or len(B) < MIN_EVENTS:
                continue
            p = B["cat"].value_counts(normalize=True)
            q = A["cat"].value_counts(normalize=True)
            novelty = float((~B["template"].isin(set(A["template"]))).mean())
            score = psi(p, q)
            shift = (p.reindex(p.index.union(q.index), fill_value=0)
                     - q.reindex(p.index.union(q.index), fill_value=0))
            rows.append({
                "company": c, "from": int(a), "to": int(b), "events_from": int(len(A)), "events_to": int(len(B)),
                "psi": score, "template_novelty": novelty,
                "biggest_shift": shift.abs().idxmax(), "biggest_shift_points": float(shift[shift.abs().idxmax()]),
                "alert": bool(score > PSI_ALERT),
                "likely_reporting_change": bool(score > PSI_ALERT and novelty > NOVELTY_ALERT),
            })
    return pd.DataFrame(rows).sort_values("psi", ascending=False)
