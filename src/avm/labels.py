"""Rule based labels for the free text cause of each disengagement.

Every company writes causes its own way, so each text gets
  * a root cause category for the part of the driving stack (or the outside world) that failed
  * behavior tags for the driving situation it describes (lane change, turn, pull over ...)

Rules are deliberately simple and readable. When a text names a cause after "root cause",
"caused by" or "due to", that clause decides. Otherwise the category mentioned first wins.
A text that matches more than one category is marked ambiguous, which is the main source
of label noise this project measures.
"""
from __future__ import annotations

import re

CATEGORIES = {
    "Perception": [r"\bperce", r"object detection", r"lane detection", r"\bdetect\w* (the state|an object|object|traffic|lane)",
                   r"misdetect", r"failed to detect", r"classif", r"false (positive|negative)", r"camera|radar|lidar",
                   r"(light|line|sign|signal|lane|object) detection", r"sensor fusion"],
    "Prediction": [r"\bpredict", r"trajectory estimation", r"estimat\w* (the )?(trajectory|behavior|intent)"],
    "Planning": [r"motion plan", r"\bplann", r"\bplanner", r"trajector", r"lane placement", r"unwanted maneuver",
                 r"undesir\w* (vehicle )?(behavior|motion|maneuver)", r"too close to", r"inappropriate braking",
                 r"incorrect location on the roadway", r"courtesy", r"drift\w* out of lane", r"oscillat", r"desired path",
                 r"unexpected behavior of the system", r"(conservative|aggressive) (vehicle )?behavior", r"insufficient deceleration",
                 r"inappropriate steering", r"too fast", r"faster than", r"not stopping", r"improper yielding", r"lane change maneuver"],
    "Map and localization": [r"\bmap\b", r"\bmaps\b", r"localiz"],
    "Hardware": [r"hardware", r"sensor (fault|failure|issue|malfunction)", r"\bbrake (fault|failure)"],
    "Software and system fault": [r"software", r"\b(low|mid|high) level error", r"\berror\b", r"diagnostic", r"\bcompute\b",
                                  r"system fault", r"prescribed limit", r"took longer to operate", r"\bbug\b", r"input delay", r"latency",
                                  r"system load", r"system discrepancy"],
    "Other road users and conditions": [r"imped", r"right of way", r"\bobstacle", r"construction", r"emergency vehicle",
                                        r"(object|vehicle) was detected within", r"other road use", r"road debris",
                                        r"road conditions", r"double parked", r"reckless", r"cut in by", r"past pedestrians",
                                        r"operational design domain", r"\bodd\b"],
}
# When two categories match at the same position, the earlier one in this list wins.
PRIORITY = ["Prediction", "Perception", "Map and localization", "Hardware", "Software and system fault",
            "Planning", "Other road users and conditions"]
CAUSE_MARKERS = [r"root cause\s*", r"caused by", r"due to", r"because of"]

BEHAVIORS = {
    "Lane change or merge": r"lane change|changing lanes|change lanes|\bmerg",
    "Turn": r"\b(left|right|u) turn|turning",
    "Intersection or signal": r"traffic (light|signal)|stop (line|sign)|intersection|red light",
    "Pedestrian or cyclist": r"pedestrian|cyclist|bicycl|\bbike",
    "Pull over, stop or parking": r"pull ?over|pick ?up|drop ?off|\bpark(ing|ed)?\b|\bcurb",
    "Lane keeping": r"lane placement|lane boundary|drift\w* out of lane|oscillat|lane keep|lane center",
    "Construction or obstacle": r"construction|\bcone|obstacle|debris",
    "Emergency vehicle": r"emergency vehicle|ambulance|fire truck|police",
}

_compiled = {k: [re.compile(p) for p in v] for k, v in CATEGORIES.items()}
_behav = {k: re.compile(v) for k, v in BEHAVIORS.items()}


def simple(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 :]", " ", str(text).lower())).strip()


def _first_hits(text: str) -> dict[str, int]:
    hits = {}
    for cat, pats in _compiled.items():
        pos = [m.start() for p in pats for m in [p.search(text)] if m]
        if pos:
            hits[cat] = min(pos)
    return hits


def _pick(hits: dict[str, int]) -> str:
    return min(hits, key=lambda c: (hits[c], PRIORITY.index(c)))


def cause(text: str) -> tuple[str, bool]:
    """Return (category, ambiguous)."""
    t = simple(text).split("conditions:")[0]   # weather notes after "Conditions:" are not causes
    # Drop the consequence clause ("... which resulted in a motion plan requiring disengagement"),
    # which describes what happened next, not what failed.
    t = re.split(r"\b(which )?result(ed|ing) in\b|\bled to\b|\bleading\b", t)[0]
    hits = _first_hits(t)
    if not hits:
        return "Unclear", False
    for marker in CAUSE_MARKERS:
        m = re.search(marker, t)
        if m:
            after = _first_hits(t[m.end():])
            if after:
                return _pick(after), len(after) > 1
    return _pick(hits), len(hits) > 1


def behaviors(text: str) -> list[str]:
    t = simple(text)
    return [k for k, p in _behav.items() if p.search(t)]
