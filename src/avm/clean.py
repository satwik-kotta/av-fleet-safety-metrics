"""Load and clean the California DMV autonomous vehicle filings (2021 to 2025).

The filings are typed in by each company, so the same company appears under several
spellings, the same field holds a dozen versions of one value, some text carries broken
characters and a few mileage cells hold typos. Everything below is normalized before
any metric is computed, and the event file is reconciled against the mileage file.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
XLSX = RAW / "dmv_disengagements_mileage_2021_2025.xlsx"

# Normalized key -> display name. Keys come from _key(); variants of one company share a name.
NAMES = {
    "aimotive": "aiMotive", "apollo": "Apollo (Baidu)", "apolloautonomousdrivingusa": "Apollo (Baidu)",
    "apple": "Apple", "argoai": "Argo AI", "auroraoperations": "Aurora", "autox": "AutoX",
    "autoxtechnologies": "AutoX", "beep": "Beep", "bosch": "Bosch", "cruise": "Cruise",
    "deeprouteai": "DeepRoute.ai", "didiresearchamerica": "DiDi", "easymile": "EasyMile",
    "gatikai": "Gatik", "ghostautonomy": "Ghost Autonomy", "imagry": "Imagry", "intel": "Intel (Mobileye)",
    "lyft": "Lyft", "maymobility": "May Mobility", "mercedes": "Mercedes-Benz",
    "mercedesbenzresearchdevelopmentnorthamerica": "Mercedes-Benz", "motionalad": "Motional",
    "nissan": "Nissan", "nissannorthamerica": "Nissan", "nissannorthamericadbaallianceinnovationlab": "Nissan",
    "nissannorthamericadbanatcsv": "Nissan", "nullmax": "Nullmax", "nuro": "Nuro", "nvidia": "NVIDIA",
    "ponyai": "Pony.ai", "qcraft": "QCraft", "qualcommtechnologies": "Qualcomm", "tensorauto": "Tensor",
    "toyotaresearchinstitute": "Toyota Research Institute", "udelv": "Udelv", "valeo": "Valeo",
    "valeonorthamerica": "Valeo", "vuerontechnology": "Vueron", "vuerontechnologyusa": "Vueron",
    "waymo": "Waymo", "weride": "WeRide", "wovenbytoyotaus": "Woven by Toyota", "zoox": "Zoox",
}

INITIATOR = {"test driver": "Test driver", "test drive": "Test driver", "driver": "Test driver",
             "operator": "Test driver", "test driver soft stop": "Test driver",
             "av system": "AV system", "ads": "AV system", "software": "AV system",
             "av system emergency stop": "AV system", "test driver av system": "Both"}

LOCATION = {"street": "Street", "street driving": "Street", "rural road": "Street",
            "freeway": "Freeway or highway", "highway": "Freeway or highway", "interstate": "Freeway or highway",
            "parking facility": "Parking facility", "parking lot": "Parking facility", "urban": "Street",
            "road": "Street", "express way": "Freeway or highway"}


def _key(s) -> str:
    s = str(s).lower()
    for w in ["inc", "llc", "corporation", "corp"]:
        s = s.replace(w, "")
    return re.sub(r"[^a-z0-9]", "", s)


def company(s) -> str:
    k = _key(s)
    if k not in NAMES:
        raise KeyError(f"unmapped manufacturer: {s!r} (key {k!r})")
    return NAMES[k]


def _simple(s) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", str(s).lower())).strip()


def fix_text(s) -> str:
    """Repair the broken characters the filings carry (a dagger where a space was, a
    stray accent where an apostrophe was) and collapse whitespace."""
    s = str(s).replace("†", " ").replace("í", "'").replace("Õ", "'").replace("Ò", '"')
    return re.sub(r"\s+", " ", s).strip()


def to_number(x) -> float:
    """Mileage cells sometimes hold typos such as '830 .04'. Strip spaces and commas."""
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = re.sub(r"[\s,]", "", str(x))
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_date(x):
    """Most dates arrive as dates. One company filed them as numbers like 20230308."""
    if isinstance(x, (int, float, np.integer, np.floating)) and not pd.isna(x) and 19000101 < x < 21000101:
        return pd.to_datetime(str(int(x)), format="%Y%m%d")
    v = pd.to_datetime(x, errors="coerce")
    if pd.notna(v) and v.year == 1970:  # a number stored as nanoseconds since 1970
        return pd.to_datetime(str(int(v.value)), format="%Y%m%d", errors="coerce")
    return v


def load_events(path: Path = XLSX) -> pd.DataFrame:
    d = pd.read_excel(path, sheet_name="2021-2025 AV Disengagements")
    out = pd.DataFrame({
        "company": d["manufacturer"].map(company),
        "year": d["reporting_year"].astype(int),
        "date": d["date"].map(parse_date),
        "vin": d["vin"].astype(str).str.strip().str.upper(),
        "initiated_by": d["disengagement_initiated_by"].map(lambda s: INITIATOR.get(_simple(s), "Other")),
        "location": d["disengagement_location"].map(lambda s: LOCATION.get(_simple(s), "Mixed or other")),
        "driverless": d["driver_present"].map(lambda s: _simple(s) != "yes"),
        "cause_text": d["disengagement_cause"].map(fix_text),
    })
    out["template"] = out["cause_text"].map(_simple)
    return out


def load_mileage(path: Path = XLSX) -> pd.DataFrame:
    m = pd.read_excel(path, sheet_name="2021-2025 Mileage")
    # The twelve monthly mileage columns sit between the disengagement total and the annual total.
    cols = list(m.columns)
    months = cols[cols.index("Annual Total of Disengagements") + 1: cols.index("ANNUAL TOTAL")]
    assert len(months) == 12, months
    monthly = m[months].apply(lambda c: c.map(to_number)).fillna(0).sum(axis=1)
    annual = m["ANNUAL TOTAL"].map(to_number)
    return pd.DataFrame({
        "company": m["Manufacturer"].map(company),
        "year": m["Year"].astype(int),
        "vin": m["VIN NUMBER"].astype(str).str.strip().str.upper(),
        # A blank annual total is filled from the monthly columns (47 Zoox 2023 vehicles, all zero).
        "miles": annual.fillna(monthly),
        "annual_was_blank": annual.isna(),
        "monthly_sum": monthly,
        "disengagements": m["Annual Total of Disengagements"].map(to_number),
    })


def reconcile(ev: pd.DataFrame, mi: pd.DataFrame) -> dict:
    """Compare event rows with the disengagement totals in the mileage file."""
    a = ev.groupby(["company", "year"]).size().rename("events")
    b = mi.groupby(["company", "year"])["disengagements"].sum().rename("mileage_file")
    j = pd.concat([a, b], axis=1).fillna(0)
    bad = j[j["events"] != j["mileage_file"]]
    return {
        "company_years": int(len(j)),
        "mismatched_company_years": int(len(bad)),
        "mismatches": [{"company": c, "year": int(y), "events": int(r.events), "mileage_file": int(r.mileage_file)}
                       for (c, y), r in bad.iterrows()],
        "raw_spellings": None,
    }


def checks(ev: pd.DataFrame, mi: pd.DataFrame) -> dict:
    out = {
        "events": int(len(ev)),
        "vehicle_years": int(len(mi)),
        "companies": int(ev["company"].nunique()),
        "missing_miles": int(mi["miles"].isna().sum()),
        "negative_miles": int((mi["miles"] < 0).sum()),
        "events_missing_date": int(ev["date"].isna().sum()),
        "events_outside_year": int(((ev["date"].dt.year < 2020) | (ev["date"].dt.year > 2025)).sum()),
        "initiator_other": int((ev["initiated_by"] == "Other").sum()),
        "location_mixed_or_other": int((ev["location"] == "Mixed or other").sum()),
        "duplicate_vehicle_years": int(mi.duplicated(["company", "year", "vin"]).sum()),
        "annual_totals_filled_from_months": int(mi["annual_was_blank"].sum()),
        "annual_total_off_from_months_by_over_1_mile": int(((mi["miles"] - mi["monthly_sum"]).abs() > 1).sum()),
    }
    for k in ["missing_miles", "negative_miles", "events_missing_date", "events_outside_year", "initiator_other"]:
        if out[k] != 0:
            raise AssertionError(f"check failed: {k} = {out[k]}")
    return out
