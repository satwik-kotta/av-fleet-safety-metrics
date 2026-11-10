"""Run the full analysis and write docs/data/results.json and outputs/*.csv.

    python -m avm.run   (from src/, after scripts/get_data.sh)
"""
from __future__ import annotations

import json

import pandas as pd

from . import collisions, drift, labels, metrics, textmodel
from .clean import ROOT, checks, load_events, load_mileage, reconcile

OUT = ROOT / "outputs"
DOCS = ROOT / "docs" / "data"
FOCUS = ["Zoox", "Waymo", "Cruise"]


def main():
    OUT.mkdir(exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    raw = pd.read_excel(ROOT / "data/raw/dmv_disengagements_mileage_2021_2025.xlsx", sheet_name=0)
    ev, mi = load_events(), load_mileage()
    integrity = checks(ev, mi)
    integrity.update({"raw_company_spellings": int(raw["manufacturer"].nunique()),
                      "raw_initiator_spellings": int(raw["disengagement_initiated_by"].nunique()),
                      "raw_location_spellings": int(raw["disengagement_location"].nunique()),
                      "distinct_cause_sentences": int(ev["template"].nunique())})
    recon = reconcile(ev, mi)

    # labels, one rule evaluation per distinct sentence
    u = ev.drop_duplicates("template")
    lab = {t: labels.cause(x) for t, x in zip(u["template"], u["cause_text"])}
    ev["cat"] = ev["template"].map(lambda t: lab[t][0])
    ev["amb"] = ev["template"].map(lambda t: lab[t][1])
    ev["behaviors"] = ev["cause_text"].map(labels.behaviors)
    top10 = ev["template"].value_counts().head(10).sum() / len(ev)

    # metrics
    mpd = metrics.mpd_table(mi, ev)
    yoy = sum((metrics.year_over_year(mpd, c) for c in FOCUS), [])
    conc = [metrics.concentration(mi, c, y) for c, y in
            mpd[mpd["comparable"]][["company", "year"]].itertuples(index=False)]

    # causes and behaviors
    cat_by_company = (ev.groupby("company")["cat"].value_counts(normalize=True).unstack(fill_value=0))
    behav = ev.explode("behaviors").dropna(subset=["behaviors"]).reset_index(drop=True)
    behav_share = behav["behaviors"].value_counts().div(len(ev)).to_dict()
    behav_cause = pd.crosstab(behav["behaviors"], behav["cat"], normalize="index")
    zoox = ev[ev["company"] == "Zoox"]
    zb = zoox.explode("behaviors").dropna(subset=["behaviors"]).reset_index(drop=True)

    model = textmodel.evaluate(ev)
    sim, sim_examples = textmodel.similarity_search(ev)
    dr = drift.drift_table(ev)
    col = collisions.analyse(collisions.load())

    mpd.to_csv(OUT / "miles_per_disengagement.csv", index=False)
    dr.to_csv(OUT / "drift.csv", index=False)
    ev.drop(columns=["behaviors"]).assign(behaviors=ev["behaviors"].str.join("; ")).to_csv(
        OUT / "events_labelled.csv", index=False)

    payload = {
        "integrity": integrity, "reconciliation": recon,
        "labels": {"event_share": ev["cat"].value_counts(normalize=True).to_dict(),
                   "sentence_share_unclear": float((u["template"].map(lambda t: lab[t][0]) == "Unclear").mean()),
                   "event_share_mixed": float(ev["amb"].mean()),
                   "top10_sentences_share_of_events": float(top10),
                   "by_company": {c: r.round(4).to_dict() for c, r in cat_by_company.loc[
                       cat_by_company.index.isin(mpd[mpd["comparable"]]["company"])].iterrows()}},
        "behaviors": {"share_of_events": behav_share,
                      "cause_mix": {b: r.round(4).to_dict() for b, r in behav_cause.iterrows()},
                      "zoox_events": int(len(zoox)),
                      "zoox_share": zb["behaviors"].value_counts().div(len(zoox)).round(4).to_dict(),
                      "zoox_causes": zoox["cat"].value_counts().to_dict()},
        "mpd": json.loads(mpd.to_json(orient="records")),
        "year_over_year": yoy,
        "concentration": conc,
        "model": model,
        "similarity": sim, "similarity_examples": sim_examples,
        "drift": json.loads(dr.to_json(orient="records")),
        "collisions": col,
    }
    (DOCS / "results.json").write_text(json.dumps(payload, indent=2, default=str))

    pd.set_option("display.width", 220)
    print(json.dumps(integrity))
    print("recon", recon["company_years"], recon["mismatched_company_years"])
    print("top10 sentences share", round(top10, 3), "mixed", round(ev["amb"].mean(), 3))
    print(mpd[mpd["comparable"]].sort_values(["company", "year"]).round(0).to_string())
    print(pd.DataFrame(yoy).round(4).to_string())
    print(pd.DataFrame(conc).round(3).to_string())
    print(json.dumps(model, indent=1))
    print(sim)
    print(dr.head(12).round(3).to_string())
    print({k: v for k, v in col.items() if k not in ("type", "av_movement")})
    print("type chi2", col["type"]["chi2"], col["type"]["p_value"], col["type"]["cramers_v"], col["type"]["share"])
    print("move chi2", col["av_movement"]["p_value"], col["av_movement"]["cramers_v"], col["av_movement"]["share"])
    print("behav", {k: round(v, 3) for k, v in behav_share.items()})
    print("zoox", payload["behaviors"]["zoox_share"], payload["behaviors"]["zoox_causes"])


if __name__ == "__main__":
    main()
