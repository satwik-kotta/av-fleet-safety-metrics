"""Run everything after ingest and write docs/data/freeway.json and outputs/freeway/*.csv.

    python -m fdb.ingest   # PySpark: Parquet table with leader joins and data checks
    python -m fdb.run      # events, rates, behavior clusters, search, lane change model
"""
from __future__ import annotations

import json

import pandas as pd

from . import behavior, events, predict
from .ingest import PERIODS, ROOT

OUT = ROOT / "outputs" / "freeway"
DOCS = ROOT / "docs" / "data"


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    ingest = json.loads((OUT / "ingest.json").read_text())
    df = events.load()
    lc, lc_info = events.lane_changes(df)
    hb, implausible = events.hard_brakes(df)
    nm = events.near_misses(df)
    rate_rows = events.rates(df, {"Lane change": lc, "Hard brake": hb, "Near miss": nm})
    first, last = list(PERIODS)[0], list(PERIODS)[-1]
    comp = [events.compare(rate_rows, e, first, last) for e in ["Lane change", "Hard brake", "Near miss"]]
    exposure = events.exposure(df)

    f, snips = behavior.features(df, lc)
    f, cl = behavior.clusters(f)
    search = behavior.search(f, snips)

    s = predict.samples(df, lc)
    model = predict.evaluate(s)

    f.to_csv(OUT / "lane_changes.csv", index=False)
    pts = [[round(g, 2), round(v, 3), int(c)] for g, v, c in zip(f["gap_new_leader"], f["peak_lateral_speed"], f["cluster"])]
    (DOCS / "freeway_lane_changes.json").write_text(json.dumps(pts))
    pd.DataFrame(rate_rows).to_csv(OUT / "event_rates.csv", index=False)

    speed_profile = (df[df["lane"].isin([1, 2, 3, 4, 5])].assign(minute=lambda d: (d["frame"] // 600).astype(int))
                     .groupby(["period", "minute"])["v"].mean().reset_index())
    payload = {
        "ingest": ingest,
        "rows_after_cleaning": int(len(df)),
        "vehicles_after_cleaning": int(df[["period", "vid"]].drop_duplicates().shape[0]),
        "periods": PERIODS,
        "exposure": exposure.to_dict(orient="records"),
        "lane_change_cleaning": lc_info,
        "hard_brake_runs_rejected_as_tracking_noise": int(implausible),
        "rates": rate_rows,
        "first_vs_last": comp,
        "lane_changes": {"analysed": int(len(f)), "risky_share": float(f["risky"].mean()),
                         "left_share": float((f["direction"] == "left").mean())},
        "clusters": cl,
        "search": search,
        "model": model,
        "speed_by_minute": speed_profile.to_dict(orient="records"),
    }
    (DOCS / "freeway.json").write_text(json.dumps(payload, indent=2, default=float))
    print(json.dumps({k: payload[k] for k in ["rows_after_cleaning", "vehicles_after_cleaning", "lane_change_cleaning",
                                               "hard_brake_runs_rejected_as_tracking_noise", "first_vs_last",
                                               "lane_changes", "search"]}, indent=1, default=float))
    print(pd.DataFrame(cl["profiles"]).round(2).to_string())


if __name__ == "__main__":
    main()
