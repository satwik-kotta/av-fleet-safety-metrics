import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdb import events, predict  # noqa: E402


def frames(vid, lanes, a=None, ttc=None, period="0750_0805"):
    n = len(lanes)
    return pd.DataFrame({
        "period": period, "vid": vid, "frame": np.arange(n), "lane": lanes,
        "prev_lane": [np.nan] + list(lanes[:-1]), "a": a if a is not None else np.zeros(n),
        "v": np.full(n, 10.0), "y": np.arange(n, dtype=float), "ttc": ttc if ttc is not None else np.full(n, np.nan),
        "leader_missing": False, "lane_lead": lanes,
    })


def test_runs_respect_minimum_length_and_groups():
    m = np.array([1, 1, 1, 0, 1, 1, 1, 1], bool)
    g = np.array(["a", "a", "a", "a", "a", "b", "b", "b"])
    # the run 4..7 is split by the group change into a (4) and b (5..7)
    assert events._runs(m, g, 3) == [(0, 2), (5, 7)]


def test_lane_change_jitter_removed():
    real = frames(1, [2] * 20 + [1] * 30)
    jitter = frames(2, [3] * 20 + [4] * 5 + [3] * 25)   # back within 0.5 s
    lc, info = events.lane_changes(pd.concat([real, jitter], ignore_index=True))
    assert list(lc["vid"]) == [1]
    assert lc.iloc[0]["direction"] == "left"
    assert info["jitter_switches"] == 2


def test_hard_brake_needs_duration_and_plausibility():
    a = np.zeros(40)
    a[5:11] = -4.0          # 0.6 s of hard braking: counts
    a[20:23] = -4.0         # 0.3 s: too short
    a[30:37] = -4.0
    a[33] = -12.0           # contains an impossible value: tracking noise
    hb, rejected = events.hard_brakes(frames(1, [2] * 40, a=a))
    assert len(hb) == 1 and hb.iloc[0]["frame"] == 5
    assert rejected == 1


def test_near_miss_needs_same_lane_and_duration():
    ttc = np.full(20, 5.0)
    ttc[3:7] = 1.0
    ttc[12:14] = 1.0        # only 0.2 s
    nm = events.near_misses(frames(1, [2] * 20, ttc=ttc))
    assert list(nm["frame"]) == [3]


def test_poisson_interval():
    lo, hi = events.poisson_ci(0)
    assert lo == 0 and hi == pytest.approx(3.689, abs=1e-3)


def test_psi_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(size=5000))
    assert predict.psi(a, a) == pytest.approx(0, abs=1e-9)
    assert predict.psi(a, a + 1.0) > 0.25
