"""
Builds every evaluation number the UI and the docs show, from the result
files in evaluation/results/ (nothing is typed in by hand):

  frontend/public/eval/testing.json        "How it was tested" tab
  frontend/public/eval/real_footage.json   "Real dashcam footage" tab
  docs/EVALUATION_RESULTS.md               the same tables, for the report

  python3 evaluation/build_report.py
"""
import datetime
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation", "experiments"))
from summarize_suite import HARMFUL, ORDER, load, summarise  # noqa: E402

RES = os.path.join(ROOT, "evaluation", "results")
OUT_UI = os.path.join(ROOT, "frontend", "public", "eval")
OUT_MD = os.path.join(ROOT, "docs", "EVALUATION_RESULTS.md")
os.makedirs(OUT_UI, exist_ok=True)

LABEL = {"correct": "Correct", "correct_abstention": "Correct refusal (event off-screen)",
         "abstained": "No verdict / qualified", "qualified_unobservable": "Qualified, event off-screen",
         "missed_unobservable": "Missed, event off-screen", "missed_collision": "Missed a visible collision",
         "false_collision": "Reported a collision that did not happen", "wrong_contact_time": "Verdict on the wrong contact time",
         "overconfident": "Overconfident", "catastrophic": "Opposite verdict", "error": "Crashed"}
MODE = {"oracle_clean": "Reasoning only (true tracks)", "oracle_occluded": "Reasoning only (true tracks, occluded)",
        "e2e": "Full pipeline (from pixels)"}


def jload(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def pct(x, d=0):
    return "–" if x is None else f"{100 * x:.{d}f} %"


POST = os.path.join(RES, "post_freeze_v21")


def suite(split, mode, post=False):
    if post:
        p = os.path.join(POST, f"collision_suite_{split}_{mode}.jsonl")
    else:
        p = os.path.join(RES, f"collision_suite_{split}" + ("" if mode == "e2e" else f"_{mode}") + ".jsonl")
    if not os.path.exists(p):
        return None, None
    rows = load(p)
    return rows, summarise(rows)


def suite_table(split, post=False, label_split=False):
    cols = (["Split"] if label_split else []) + ["Layer", "n"] + [LABEL[k] for k in ORDER if k != "error"]
    rows = []
    for sp in (split if isinstance(split, (list, tuple)) else [split]):
        for mode in ("oracle_clean", "oracle_occluded", "e2e"):
            _, s = suite(sp, mode, post)
            if s is None:
                continue
            rows.append(([sp] if label_split else []) + [MODE[mode], s["n"]] + [s["counts"].get(k, 0) for k in ORDER if k != "error"])
    return {"columns": cols, "rows": rows} if rows else None


def kpis(s):
    return [{"value": f"{s['correct']['k']}/{s['n']}", "label": "full pipeline: correct, incl. correct refusals"},
            {"value": f"{s['harmful']['k']}/{s['n']}", "label": "full pipeline: harmful errors (missed, false or wrong-time collision, overconfident, opposite verdict)"},
            {"value": f"{s['definite_verdicts']['correct']}/{s['definite_verdicts']['issued']}", "label": "definite verdicts (avoidable / not avoidable) that were right"}]


def resolution_table(split):
    rows, _ = suite(split, "e2e")
    if not rows:
        return None
    bins = [(0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 99)]
    out = []
    for lo, hi in bins:
        rs = [r for r in rows if (r.get("event_resolution") or {}).get("m_per_px_along_road") is not None
              and lo <= r["event_resolution"]["m_per_px_along_road"] < hi]
        if not rs:
            continue
        ok = sum(r["outcome"] in ("correct", "correct_abstention") for r in rs)
        bad = sum(r["outcome"] in HARMFUL for r in rs)
        out.append([f"{lo:g}–{hi:g} m" if hi < 99 else f"> {lo:g} m", len(rs), ok, bad])
    return {"columns": ["Road length per image row at the event", "n", "correct", "harmful"], "rows": out}


def family_table(split):
    rows, _ = suite(split, "e2e")
    if not rows:
        return None
    fams = sorted({r.get("family") for r in rows})
    out = []
    for f in fams:
        rs = [r for r in rows if r.get("family") == f]
        out.append([f.replace("_", " "), len(rs), sum(r["outcome"] in ("correct", "correct_abstention") for r in rs),
                    sum(r["outcome"] == "abstained" for r in rs), sum(r["outcome"] in HARMFUL for r in rs)])
    return {"columns": ["Camera family", "n", "correct", "no verdict", "harmful"], "rows": out}


def legacy_table(split, label="v2.0 (frozen)"):
    rows, _ = suite(split, "e2e")
    if not rows or "legacy_status" not in rows[0]:
        return None
    n = len(rows)
    v1_verdicts = sum(r.get("legacy_verdict") in ("AVOIDABLE", "NOT_AVOIDABLE", "ASSUMPTION_DEPENDENT") for r in rows)
    v1_ok = sum(1 for r in rows if r.get("legacy_verdict") == r["gt"]["verdict"] and r.get("gt_collision_t_s") is not None)
    v2_ok = sum(r["outcome"] in ("correct", "correct_abstention") for r in rows)
    v2_bad = sum(r["outcome"] in HARMFUL for r in rows)
    return {"columns": ["Pipeline", "n", "issued a collision verdict", "correct", "harmful"],
            "rows": [["v1 as received (process_video)", n, v1_verdicts, v1_ok, "–"],
                     [label, n, sum(r.get("carai_verdict") in ("AVOIDABLE", "NOT_AVOIDABLE", "ASSUMPTION_DEPENDENT") for r in rows),
                      v2_ok, v2_bad]]}


def geometry_tables():
    base, v2 = jload("baseline_legacy_geometry_sweep.json"), jload("geometry_sweep_v2.json")
    if not base or not v2:
        return None
    fams = []
    for r in base["rows"]:
        if r["family"] not in fams:
            fams.append(r["family"])
    rows = []
    for f in fams:
        b = [r for r in base["rows"] if r["family"] == f]
        n = [r for r in v2["rows"] if r["family"] == f]
        be = [abs(r["disp_scale_error"]) for r in b if r.get("accepted") and r.get("disp_scale_error") is not None]
        ne = [abs(r["disp_scale_error"] - (r.get("expected_scale_error_from_prior") or 0.0)) if f == "prior_mismatch"
              else abs(r["disp_scale_error"]) for r in n if r.get("accepted") and r.get("disp_scale_error") is not None]
        rows.append([f.replace("_", " "), len(b),
                     f"{sum(r['accepted'] for r in b)}", pct(max(be), 1) if be else "–",
                     f"{sum(r['accepted'] for r in n)}", pct(max(ne), 2) if ne else "–"])
    note = ("For 'prior mismatch' scenes the true markings differ from the assumed standard. v2's error there is "
            "measured against the error the wrong prior implies (it cannot be identified from one view); v1's is raw.")
    mism = [r for r in v2["rows"] if r["family"] == "prior_mismatch" and r.get("accepted")]
    shape = [r for r in mism if abs(r["dash_len"] / (r["dash_len"] + r["dash_gap"]) - 0.25) > 0.02]
    caught = sum("marking_standard" in (r.get("skeptic_disputed") or []) for r in shape)
    return {"columns": ["Scenes", "n", "v1 accepted", "v1 worst abs. error", "v2 accepted", "v2 worst abs. error"], "rows": rows}, note, \
        (caught, len(shape))


def fault_table():
    d = jload("skeptic_fault_injection.json")
    if not d:
        return None
    s = d["summary"]
    names = {"clean": "No fault (a flag here with speeds within 10 % would be a false alarm)", "camera_bump": "Camera knocked mid-clip",
             "wrong_standard_shape": "Wrong marking standard (6 m + 12 m)", "misindexed_dash": "Dash mis-indexed (algorithm fault)",
             "wrong_lateral_prior": "Wrong lane-width prior", "same_duty_longer": "Longer markings, same proportions (undetectable by design)",
             "wrong_fps_metadata": "Wrong frame rate in file (undetectable by design)"}
    rows = []
    for k, v in s.items():
        rs = [r for r in d["rows"] if r["fault"] == k]
        false_alarm = sum(1 for r in rs if (r["disputed"] or r["blocking_dispute"]) and r["worst_speed_rel_error"] is not None
                          and r["worst_speed_rel_error"] <= 0.10)
        rows.append([names.get(k, k), v["n"], v.get("blocked_or_rejected", "–"), v["flagged_or_rejected"], false_alarm,
                     v["silent_failures"], pct(v["median_worst_speed_err"], 1) if v["median_worst_speed_err"] is not None else "–"])
    return {"columns": ["Injected fault", "n", "verdict withheld or refused", "any check flagged",
                        "flagged runs whose reported speeds were all within 10 %", "silent failures (> 10 % error, no flag)",
                        "median worst speed error"], "rows": rows}


def dev_history_table():
    d = os.path.join(RES, "dev_history")
    if not os.path.isdir(d):
        return None
    rows = []
    for it in ("iter1", "iter2"):
        for mode in ("oracle_clean", "oracle_occluded", "e2e"):
            name = f"collision_suite_dev" + ("" if mode == "e2e" else f"_{mode}") + f"_{it}.jsonl"
            p = os.path.join(d, name)
            if os.path.exists(p):
                s = summarise(load(p))
                rows.append([it + (" (stopped early)" if s["n"] < 30 else ""), MODE[mode], s["n"], s["correct"]["k"], s["harmful"]["k"]])
    for v, label in (("legacy", "whole-clip median + first-half fallback (kept)"), ("resolve3", "region-resolved, 3 segments"),
                     ("resolve5", "region-resolved, 5 segments")):
        p = os.path.join(d, f"ablation_bg_{v}.jsonl")
        if os.path.exists(p):
            s = summarise(load(p))
            rows.append([f"background ablation: {label}", MODE["e2e"], s["n"], s["correct"]["k"], s["harmful"]["k"]])
    for mode in ("oracle_clean", "oracle_occluded", "e2e"):
        _, s = suite("dev", mode)
        if s:
            rows.append(["final (frozen)", MODE[mode], s["n"], s["correct"]["k"], s["harmful"]["k"]])
    return {"columns": ["Dev iteration", "Layer", "n", "correct", "harmful"], "rows": rows}


def testing_json():
    blocks = []
    blocks.append({"title": "What was tested, and what was not",
                   "text": ("Collision verdicts are tested on a simulator whose ground truth CARAI never sees: 3D vehicles, "
                            "true markings that differ from the assumed standard, camera roll, yaw and noise. Seeds for the "
                            "dev, test and shift splits were fixed in advance; the code was developed on dev only, frozen, "
                            "and then run once on test and shift. Each layer is tested separately: the reasoning layer with "
                            "true tracks, then the full pipeline from pixels."),
                   "callout": {"tone": "stop", "text": ("No real collision footage with ground truth was available to us under a "
                                                         "licence that permits this use, so the collision verdicts are "
                                                         "validated on simulation only. Real footage was used for the dashcam "
                                                         "speed mode (see the Real dashcam footage tab).")}})
    for split, title in (("test", "Held-out test split, frozen code (v2.0)"),
                         ("shift", "Distribution-shift split, frozen code (v2.0; harder camera families over-sampled)")):
        t = suite_table(split)
        if not t:
            continue
        _, s = suite(split, "e2e")
        blocks.append({"title": title, "kpis": kpis(s) if s else [], "table": t,
                       "text": "Counts per outcome. 'No verdict / qualified' is CARAI declining, or answering 'depends on the assumptions', where the truth was definite."})
    t2 = suite_table("test2")
    if t2:
        _, s = suite("test2", "e2e")
        blocks.append({"title": "After the freeze: v2.1 on a new held-out split (test2)", "kpis": kpis(s) if s else [], "table": t2,
                       "text": ("The frozen run showed that every harmful end-to-end error on test happened 70-130 m from the camera, where "
                                "one image row spans 0.65-2.5 m of road. v2.1 adds two rules derived from the stated priors, not fitted: "
                                "neither contact nor 'no contact' is claimed when one image row at the vehicles spans more road than "
                                "twice the vehicle-length uncertainty (0.7 m); and 'no contact' must also hold if a hidden lead braked "
                                "at the envelope's 8 m/s^2 (or a hidden follower did not brake). It also fixes a crash on an empty "
                                "Monte Carlo set. Because test motivated these rules, v2.1 is scored on test2, a split defined and "
                                "committed before v2.1 was run on it (same distribution as test).")})
    tp = suite_table(["test", "shift", "dev"], post=True, label_split=True)
    if tp:
        blocks.append({"title": "v2.1 re-run on splits it has already seen (post hoc, not independent)", "table": tp,
                       "text": "Shown so the effect of the change on the original splits is visible; these numbers are not a held-out result."})
    for split in ("test",):
        rt = resolution_table(split)
        if rt:
            rt = {**rt, "columns": ["Road length per image row at the event (frozen v2.0, test)"] + rt["columns"][1:]}
        if rt:
            blocks.append({"title": "Where the full pipeline fails: distance",
                           "text": ("Grouped by how much road one image row covers where the event happens (from ground truth). "
                                    "Far from the camera the gap between two vehicles cannot be measured to within a vehicle "
                                    "length's uncertainty, and CARAI says so."), "table": rt})
        ft = family_table(split)
        if ft:
            blocks.append({"title": "Where the full pipeline fails: camera family", "table": ft})
        lt = legacy_table(split)
        if lt:
            lt2 = legacy_table("test2", "v2.1")
            if lt2:
                lt = {"columns": ["Split"] + lt["columns"],
                      "rows": [["test"] + r for r in lt["rows"]] + [["test2"] + r for r in lt2["rows"]]}
            blocks.append({"title": "Before and after: the pipeline as received vs this one, same clips", "table": lt})
    g = geometry_tables()
    if g:
        t, note, (caught, n_shape) = g
        blocks.append({"title": "Road geometry before and after, same scenes", "table": t,
                       "text": note + f" The duty-cycle check flagged {caught} of {n_shape} accepted scenes whose dash proportions differ from the standard."})
    ft = fault_table()
    if ft:
        blocks.append({"title": "The Skeptic under fault injection", "table": ft,
                       "text": ("Each fault is injected into otherwise clean clips (two vehicles, no collision) analysed by the "
                                "unchanged pipeline (v2.1). 'Speed error' is the worst error among the speeds CARAI reports, so it "
                                "includes tracking errors: on the clean clips, every dispute was raised on a reconstruction that "
                                "really was 20-40 % wrong (a tracking failure), and two tracking errors (16 % and 34 %) went "
                                "unflagged. The last two rows are faults no check can see from a single static camera; they are "
                                "listed so the blind spots are explicit.")})
    dh = dev_history_table()
    if dh:
        blocks.append({"title": "Development history on the dev split", "table": dh,
                       "text": ("Every scored dev run, including earlier code versions. Scoring was made stricter between "
                                "iterations (missed and wrong-time collisions became harmful); all rows are re-scored with the final rules.")})
    blocks.append({"title": "Reproduce", "text": ("python3 evaluation/experiments/collision_suite.py --split test --n 30 --mode e2e   "
                                                  "(also --mode oracle_clean / oracle_occluded, --split shift); "
                                                  "python3 evaluation/experiments/summarize_suite.py; python3 evaluation/build_report.py")})
    return {"title": "How it was tested", "generated": "evaluation/build_report.py on " + datetime.date.today().isoformat(),
            "intro": "Every number on this page is computed from the result files committed in evaluation/results.",
            "blocks": blocks}


def real_json():
    ho = jload("comma2k19_holdout.json")
    hp = jload("comma2k19_holdout_posthoc_scale_check.json")
    sc = jload("comma_eval_california_freeway.json")
    sp = jload("comma_eval_california_freeway_posthoc_v2.json")
    dv = jload("dashcam_dev.json")
    blocks = [{"title": "Two real datasets, tested under written pre-registrations",
               "text": ("Own-vehicle speed from lane markings on real dashcam footage, compared with the speed the car itself "
                        "recorded. The protocol, metrics and marking prior were committed before any real frame was analysed "
                        "(evaluation/real/PREREGISTRATION*.md). The second dataset was an independent holdout. It failed, and the "
                        "failure is shown first.")}]
    series = None
    if ho:
        pre = ho["fps_from_sensor_timestamps"]
        fps = pre["summary"]["fps_used"]
        w = pre["windows"]
        truth = [{"t": x["frame0"] / fps, "v": x["true"]} for x in w]
        est_pre = [{"t": x["frame0"] / fps, "est": x["est"], "lo": x["ci"][0], "hi": x["ci"][1]} for x in w if x["status"] == "OK"]
        ests = [{"name": "Pre-registered (freeway 48 ft prior): wrong standard, 2x too fast", "color": "#9e2a2b", "points": est_pre}]
        if hp:
            u = hp["urban_24ft"]["windows"]
            ests.append({"name": "Post hoc, with the city 24 ft standard the scale check points to", "color": "#2f5d8c",
                         "points": [{"t": x["frame0"] / fps, "est": x["est"], "lo": x["ci"][0], "hi": x["ci"][1]} for x in u if x["status"] == "OK"]})
        series = {"title": "Holdout: comma2k19 segment (city street, MIT licence)", "x_label": "seconds into the segment",
                  "note": ("Red: the pre-registered run. The freeway marking standard was wrong for this street, so every speed is "
                           "about twice the truth with tight intervals. Blue: after the post-hoc fix, a second ruler (lane width "
                           "x focal length) rejects the freeway standard; rerun with the 24 ft city standard it implies. Choosing "
                           "that standard after seeing the data makes the blue points a demonstration, not a validation."),
                  "truth": truth, "estimates": ests}
        rows = [["Pre-registered (container fps 25, unverified)", ho["fps_from_container_unverified"]["summary"]["n_windows"],
                 ho["fps_from_container_unverified"]["summary"]["n_answered"], pct(ho["fps_from_container_unverified"]["summary"]["median_ape"]),
                 pct(ho["fps_from_container_unverified"]["summary"]["ci90_coverage"])],
                ["Pre-registered (sensor-timestamp fps 20)", pre["summary"]["n_windows"], pre["summary"]["n_answered"],
                 pct(pre["summary"]["median_ape"]), pct(pre["summary"]["ci90_coverage"])]]
        if hp:
            c, u = hp["caltrans_freeway_48ft"]["summary"], hp["urban_24ft"]["summary"]
            rows.append(["Post hoc: + lane-width scale check, freeway prior", c["n"], c["answered"], "–", "–"])
            rows.append(["Post hoc: + scale check, city 24 ft prior", u["n"], u["answered"], pct(u["median_ape"], 1), pct(u["coverage"])])
        blocks.append({"title": "Holdout numbers", "table": {"columns": ["Run", "2 s windows", "answered", "median abs. error", "90 % interval coverage"], "rows": rows},
                       "callout": {"tone": "stop", "text": ("The pre-registered holdout failed: 16 confident answers, all about 2x too high. "
                                                             "Cause: the marking standard is the speed scale and was wrong for a city street. "
                                                             "The scale cross-check that now refuses this case was added afterwards.")}})
    if sc:
        rows = []
        for tag, d in (("Pre-registered", sc), ("Post hoc v2 (duty + continuity checks)", sp)):
            if not d:
                continue
            for seg in ("freeway", "city"):
                s = d["summary"][seg]
                rows.append([tag, s["segment"], s["n_windows"], s["n_answered"], pct(s["median_ape"], 1), pct(s["p90_ape"], 1),
                             pct(s["ci90_coverage"])])
        blocks.append({"title": "comma.ai speed challenge (17 minutes, freeway then city)",
                       "text": ("Used for evaluation only; the video has no licence file, so it is not redistributed and not plotted here. "
                                "Freeway prior (Caltrans 12 ft + 36 ft). CARAI answers only a minority of windows. On the freeway the "
                                "answers are accurate, with interval coverage close to the nominal 90 %. In the city the prior is "
                                "wrong and the few answers are wrong."),
                       "table": {"columns": ["Run", "Segment", "windows", "answered", "median abs. error", "90th pct abs. error", "90 % coverage"], "rows": rows}})
    if dv:
        rws = dv["rows"]
        ok = [r for r in rws if r["status"] == "OK"]
        blocks.append({"title": "Synthetic development set (before any real footage)",
                       "text": (f"{len(ok)} of {len(rws)} windows answered; median abs. error "
                                f"{np.median([abs(r['err_pct']) for r in ok]):.2f} %; {sum(r['covered'] for r in ok)} of {len(ok)} "
                                "inside their 90 % interval. The estimator was frozen after this (commit 2bf75b9).")})
    return {"title": "Real dashcam footage", "intro": "", "blocks": blocks, "series": series,
            "generated": "evaluation/build_report.py on " + datetime.date.today().isoformat()}


def to_md(rep):
    out = [f"# {rep['title']}", "", f"_Generated by evaluation/build_report.py on {datetime.date.today().isoformat()}. Do not edit by hand._", ""]
    if rep.get("intro"):
        out += [rep["intro"], ""]
    for b in rep["blocks"]:
        out += [f"## {b['title']}", ""]
        if b.get("text"):
            out += [b["text"], ""]
        if b.get("callout"):
            out += [f"> {b['callout']['text']}", ""]
        for k in b.get("kpis") or []:
            out.append(f"* **{k['value']}** {k['label']}")
        if b.get("kpis"):
            out.append("")
        if b.get("table"):
            t = b["table"]
            out.append("| " + " | ".join(str(c) for c in t["columns"]) + " |")
            out.append("|" + "---|" * len(t["columns"]))
            for r in t["rows"]:
                out.append("| " + " | ".join(str(c) for c in r) + " |")
            out.append("")
    return "\n".join(out)


def main():
    t, r = testing_json(), real_json()
    json.dump(t, open(os.path.join(OUT_UI, "testing.json"), "w"), indent=1)
    json.dump(r, open(os.path.join(OUT_UI, "real_footage.json"), "w"), indent=1)
    md = to_md(t) + "\n\n" + to_md({**r, "title": r["title"]}).replace("# Real dashcam footage", "# Real dashcam footage", 1)
    open(OUT_MD, "w").write(md)
    print("wrote", os.path.join(OUT_UI, "testing.json"), os.path.join(OUT_UI, "real_footage.json"), OUT_MD)


if __name__ == "__main__":
    main()
