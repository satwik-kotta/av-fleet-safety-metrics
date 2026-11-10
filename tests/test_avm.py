import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from avm import clean, drift, labels, metrics  # noqa: E402


@pytest.mark.parametrize("raw,name", [("ZOOX, INC", "Zoox"), ("Zoox, Inc.", "Zoox"), ("WAYMO LLC", "Waymo"),
                                      ("Nissan North America dba Alliance Innovation Lab", "Nissan"),
                                      ("Woven by Toyota, U.S., Inc.", "Woven by Toyota")])
def test_company_spellings_collapse(raw, name):
    assert clean.company(raw) == name


def test_unknown_company_fails_loudly():
    with pytest.raises(KeyError):
        clean.company("Totally New Robotaxi Co")


def test_numbers_and_dates():
    assert clean.to_number("830 .04") == pytest.approx(830.04)
    assert clean.to_number("1,204.5") == pytest.approx(1204.5)
    assert pd.isna(clean.to_number("n/a"))
    assert clean.parse_date(20230308) == pd.Timestamp("2023-03-08")
    assert clean.parse_date(pd.Timestamp(20230308)) == pd.Timestamp("2023-03-08")  # number read as nanoseconds
    assert clean.parse_date("2024-05-01") == pd.Timestamp("2024-05-01")


def test_broken_characters_repaired():
    assert clean.fix_text("The ADS†initiated  a stop") == "The ADS initiated a stop"
    assert clean.fix_text("doesnít") == "doesn't"


@pytest.mark.parametrize("text,cat", [
    ("The ADS incorrectly predicted the behavior of another road user, which resulted in a motion plan requiring disengagement",
     "Prediction"),
    ("A discrepancy in the onboard map resulted in a motion plan requiring disengagement", "Map and localization"),
    ("A vehicle impeded the planned path of the AV. Root cause: vehicle impeding planned path. Conditions: Sunny",
     "Other road users and conditions"),
    ("Safety Driver disengaged to ensure stopping at stop line. Root cause: inaccurate traffic light or stop line detection. "
     "Conditions: Non-inclement weather", "Perception"),
    ("Test driver took control due to a hardware fault. Root Cause: hardware fault.", "Hardware"),
    ("Perception discrepancy. Incorrect trajectory estimation based on a traffic island.", "Perception"),
    ("Prediction discrepancy. Incorrect trajectory estimation of a pedestrian crossing the road.", "Prediction"),
    ("The weather was nice today", "Unclear"),
])
def test_cause_rules(text, cat):
    assert labels.cause(text)[0] == cat


def test_behavior_tags():
    tags = labels.behaviors("Incorrect trajectory estimation of a cyclist during a lane change near the curb")
    assert "Pedestrian or cyclist" in tags and "Lane change or merge" in tags and "Pull over, stop or parking" in tags


def test_poisson_interval_known_values():
    lo, hi = metrics.poisson_ci(0)
    assert lo == 0 and hi == pytest.approx(3.689, abs=1e-3)
    lo, hi = metrics.poisson_ci(4)
    assert lo == pytest.approx(1.090, abs=1e-3) and hi == pytest.approx(10.242, abs=1e-3)


def test_mpd_table_and_zero_events():
    mi = pd.DataFrame({"company": ["A", "A", "B"], "year": [2024, 2024, 2024], "vin": ["1", "2", "3"],
                       "miles": [60_000.0, 40_000.0, 50_000.0], "disengagements": [3, 1, 0]})
    ev = pd.DataFrame({"company": ["A"] * 4, "year": [2024] * 4,
                       "initiated_by": ["AV system", "Test driver", "Test driver", "Test driver"]})
    t = metrics.mpd_table(mi, ev).set_index("company")
    assert t.loc["A", "mpd"] == pytest.approx(25_000)
    assert t.loc["A", "av_initiated"] == 1
    assert t.loc["B", "mpd"] is None or pd.isna(t.loc["B", "mpd"])
    assert t.loc["B", "mpd_low"] == pytest.approx(50_000 / 3.689, rel=1e-3)


def test_psi_is_zero_for_identical_mix_and_grows_with_shift():
    p = pd.Series({"a": 0.5, "b": 0.5})
    assert drift.psi(p, p) == pytest.approx(0.0, abs=1e-12)
    assert drift.psi(pd.Series({"a": 0.9, "b": 0.1}), p) > drift.psi(pd.Series({"a": 0.6, "b": 0.4}), p) > 0
