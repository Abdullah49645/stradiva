"""Render GT boxes (green) and pipeline track boxes (red measured, magenta
other) for a cached simulator case, as a 2x3 contact sheet."""
import json, sys, os
import cv2, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "evaluation", "experiments"))
from backend.pipeline.analyze import analyze_video
from common import read_frames

def main(split, name, out, frames_at):
    p = os.path.join(ROOT, "evaluation/_sim_cache/collision_suite", split, name)
    gt = json.load(open(p + "_ground_truth.json"))
    case, intern = analyze_video(p + ".mp4", seed=0, return_internals=True)
    fps, fr = read_frames(p + ".mp4")
    tiles = []
    for k in frames_at:
        k = min(k, len(fr) - 1)
        im = fr[k].copy()
        for vid, g in gt["frames"][k]["vehicles"].items():
            if g["bbox"]:
                x, y, w, h = map(int, g["bbox"]); cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(im, vid, (x, y - 4), 0, 0.8, (0, 255, 0), 2)
        for tr in intern["tracks"]:
            for r in tr["records"]:
                if r["frame"] == k and r.get("bbox"):
                    x, y, w, h = map(int, r["bbox"]); col = (0, 0, 255) if r["status"].startswith("MEAS") else (255, 0, 255)
                    cv2.rectangle(im, (x, y), (x + w, y + h), col, 2)
                    cv2.putText(im, f"{tr['id']} {r['status'][:5]}", (x, y + h + 20), 0, 0.7, col, 2)
        cv2.putText(im, f"f{k} t={k / fps:.2f}", (10, 34), 0, 1.1, (255, 255, 0), 2)
        tiles.append(cv2.resize(im, (640, 360)))
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(out, np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]))
    print([(s["stage"], s["status"], s["detail"]) for s in case["stages"]])
    print("collision", {k: case.get("collision", {}) and case["collision"].get(k) for k in ("t_contact_s", "source", "gap_when_lead_last_observed_m", "lead_last_observed_s")})

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], [int(v) for v in sys.argv[4:]] or [20, 50, 80, 110, 140, 170])
