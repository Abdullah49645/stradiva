"""
Summarise collision-suite results. Outcomes are RE-SCORED from each row with
the current scoring function (collision_suite.outcome), so every split is
judged by the same rules even if a row was written by an older run.

  python3 summarize_suite.py                      # all result files present
  python3 summarize_suite.py --json summary.json  # also write machine-readable output

Outcome categories (collision_suite.outcome):
  correct               verdict equals the ground-truth verdict (incl. "no collision" when none happened)
  correct_abstention    no verdict, and the event was not observable in the footage
  abstained             no verdict / qualified verdict where ground truth was definite and observable
  missed_collision      a visible collision reported as "no contact found"
  missed_unobservable   an off-screen collision reported as "no contact found"
  qualified_unobservable ASSUMPTION_DEPENDENT on an off-screen event
  false_collision       contact reported where none happened
  wrong_contact_time    a verdict whose reported contact is > 0.3 s from the true contact
  overconfident         definite verdict where ground truth depends on the assumptions
  catastrophic          the opposite definite verdict
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS_DIR, save_json, wilson_interval  # noqa: E402
from collision_suite import outcome  # noqa: E402

ORDER = ["correct", "correct_abstention", "abstained", "qualified_unobservable", "missed_unobservable",
         "missed_collision", "false_collision", "wrong_contact_time", "overconfident", "catastrophic", "error"]
HARMFUL = {"missed_collision", "false_collision", "wrong_contact_time", "overconfident", "catastrophic"}


def load(path):
    rows = []
    for line in open(path):
        r = json.loads(line)
        r["outcome"] = "error" if "error" in r and "gt" not in r else outcome(r)
        rows.append(r)
    return rows


def summarise(rows):
    n = len(rows)
    c = Counter(r["outcome"] for r in rows)
    out = {"n": n, "counts": {k: c.get(k, 0) for k in ORDER if c.get(k)}}
    for name, keys in (("correct", {"correct", "correct_abstention"}), ("harmful", HARMFUL)):
        k = sum(c.get(x, 0) for x in keys)
        lo, hi = wilson_interval(k, n)
        out[name] = {"k": k, "rate": k / n if n else None, "ci90": [lo, hi]}
    # definite verdicts issued, and how many were right (selective accuracy)
    definite = [r for r in rows if r.get("carai_verdict") in ("AVOIDABLE", "NOT_AVOIDABLE")]
    right = sum(1 for r in definite if r["outcome"] == "correct")
    out["definite_verdicts"] = {"issued": len(definite), "correct": right,
                                "ci90": list(wilson_interval(right, len(definite))) if definite else None}
    fam = defaultdict(Counter)
    for r in rows:
        fam[r.get("family", "?")][r["outcome"]] += 1
    out["by_family"] = {f: dict(v) for f, v in sorted(fam.items())}
    if rows and "legacy_status" in rows[0]:
        out["legacy_v1"] = dict(Counter(f"{r.get('legacy_status')}/{r.get('legacy_verdict')}" for r in rows))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--pattern", default="collision_suite_*.jsonl")
    a = ap.parse_args()
    allres = {}
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, a.pattern))):
        key = os.path.basename(path)[len("collision_suite_"):-len(".jsonl")]
        s = summarise(load(path))
        allres[key] = s
        print(f"\n== {key}  (n={s['n']})")
        print("   " + "  ".join(f"{k}={v}" for k, v in s["counts"].items()))
        print(f"   correct {s['correct']['k']}/{s['n']}  [{s['correct']['ci90'][0]:.2f}, {s['correct']['ci90'][1]:.2f}]   "
              f"harmful {s['harmful']['k']}/{s['n']}  [{s['harmful']['ci90'][0]:.2f}, {s['harmful']['ci90'][1]:.2f}]   "
              f"definite verdicts {s['definite_verdicts']['correct']}/{s['definite_verdicts']['issued']} right")
        for f, v in s["by_family"].items():
            print(f"     {f:18s} " + "  ".join(f"{k}={n}" for k, n in sorted(v.items())))
        if "legacy_v1" in s:
            print("   legacy v1 pipeline on the same clips: " + ", ".join(f"{k}: {n}" for k, n in s["legacy_v1"].items()))
    if a.json:
        save_json(allres, a.json)


if __name__ == "__main__":
    main()
