"""Fleet metrics: miles per disengagement with exact Poisson intervals, year over year
tests, and how concentrated disengagements are across vehicles."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2

MIN_MILES = 100_000  # company years with fewer autonomous miles are too small to compare


def poisson_ci(k: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact (Garwood) interval for a Poisson count."""
    lo = 0.0 if k == 0 else chi2.ppf(alpha / 2, 2 * k) / 2
    hi = chi2.ppf(1 - alpha / 2, 2 * k + 2) / 2
    return float(lo), float(hi)


def mpd_table(mi: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    """Miles per disengagement for each company year, overall and for disengagements the
    AV system started itself, with 95% intervals. With zero events the upper bound is open."""
    g = mi.groupby(["company", "year"]).agg(vehicles=("vin", "nunique"), miles=("miles", "sum"),
                                            disengagements=("disengagements", "sum")).reset_index()
    av = ev[ev["initiated_by"] != "Test driver"].groupby(["company", "year"]).size().rename("av_initiated")
    g = g.merge(av, on=["company", "year"], how="left").fillna({"av_initiated": 0})
    rows = []
    for r in g.itertuples():
        k = int(r.disengagements)
        lo, hi = poisson_ci(k)
        rows.append({
            "company": r.company, "year": int(r.year), "vehicles": int(r.vehicles), "miles": float(r.miles),
            "disengagements": k, "av_initiated": int(r.av_initiated),
            "mpd": r.miles / k if k else None,
            "mpd_low": r.miles / hi,
            "mpd_high": r.miles / lo if lo > 0 else None,
            "per_million_miles": 1e6 * k / r.miles if r.miles > 0 else None,
        })
    t = pd.DataFrame(rows)
    t["comparable"] = t["miles"] >= MIN_MILES
    return t


def year_over_year(t: pd.DataFrame, company: str) -> list[dict]:
    """Exact conditional test that the disengagement rate per mile changed between consecutive
    years. Given the total count, the count in the later year is binomial with p = its share of miles."""
    s = t[t["company"] == company].sort_values("year")
    out = []
    for a, b in zip(s.itertuples(), s.iloc[1:].itertuples()):
        n = a.disengagements + b.disengagements
        if n == 0:
            continue
        p = b.miles / (a.miles + b.miles)
        test = binomtest(b.disengagements, n, p)
        rr = (b.disengagements / b.miles) / (a.disengagements / a.miles) if a.disengagements else None
        out.append({"company": company, "from": int(a.year), "to": int(b.year), "rate_ratio": rr,
                    "p_value": float(test.pvalue)})
    return out


def concentration(mi: pd.DataFrame, company: str, year: int) -> dict:
    """How many vehicles account for the disengagements in one company year."""
    g = mi[(mi["company"] == company) & (mi["year"] == year)].groupby("vin").agg(
        miles=("miles", "sum"), dis=("disengagements", "sum"))
    g = g[g["miles"] > 0]
    total = g["dis"].sum()
    top = g["dis"].sort_values(ascending=False)
    k = max(1, int(np.ceil(len(g) * 0.10)))
    return {"company": company, "year": year, "vehicles": int(len(g)),
            "share_with_zero": float((g["dis"] == 0).mean()),
            "top10pct_vehicle_share": float(top.head(k).sum() / total) if total else None}
