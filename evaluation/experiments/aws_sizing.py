"""
Sizing evidence for the processing Lambda (docs/aws.md): wall time and peak
resident memory of what the processing Lambda runs (the investigator: the
pipeline plus its remedies) on a clip at the cap the deployment uses (15 s,
1280x720, 30 fps), each in a fresh child process
(resource.getrusage(RUSAGE_CHILDREN).ru_maxrss). Two clips: an ordinary one,
and the same clip with the camera knocked half-way (the costliest remedy:
both stable parts are analysed again). Measured on the machine it runs on,
NOT on Lambda.

  python3 aws_sizing.py      -> evaluation/results/aws_sizing.json
"""
import json
import os
import platform
import resource
import subprocess
import sys
import time

from common import CACHE_DIR, ROOT, save_json
from collision_suite import sample_scenario

from evaluation.sim.world import render_scenario


def main():
    sc, _ = sample_scenario("dev", 0)
    sc.name, sc.duration_s = "sizing_15s", 15.0
    out_dir = os.path.join(CACHE_DIR, "sizing")
    vp = os.path.join(out_dir, sc.name + ".mp4")
    if not os.path.exists(vp):
        vp, _, _ = render_scenario(sc, out_dir)
    bumped = vp.replace(".mp4", "_bumped.mp4")
    if not os.path.exists(bumped):
        from skeptic_faults import bump_video
        bump_video(vp, bumped, sc.fps)
    runs = {}
    for name, clip in (("ordinary", vp), ("camera_bumped", bumped)):
        code = ("import sys, json; sys.path.insert(0, %r); from backend.investigator.investigator import investigate; "
                "c = investigate(%r, max_seconds=15); print(json.dumps({'status': c['status'], 'timings': c.get('timings_s'), "
                "'investigation_s': c['investigation']['duration_s'], 'remedies': [s.get('step') for s in c['investigation']['trace']]}))"
                % (ROOT, clip))
        t0 = time.time()
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        wall = time.time() - t0
        rss_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss   # max over children so far
        last = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
        runs[name] = {"wall_s": round(wall, 1), "peak_rss_mb_so_far": round(rss_kb / 1024, 0),
                      "result": json.loads(last[-1]) if last else {"stderr": p.stderr[-800:]}}
    res = {"clip": {"seconds": 15.0, "width": 1280, "height": 720, "fps": 30.0}, "runs": runs,
           "wall_s": max(r["wall_s"] for r in runs.values()),
           "peak_rss_mb": max(r["peak_rss_mb_so_far"] for r in runs.values()),
           "machine": {"cpus": os.cpu_count(), "platform": platform.platform(), "python": platform.python_version()},
           "note": ("development machine, not Lambda; Lambda at 4096 MB has about 2.3 vCPUs. peak_rss_mb is the maximum "
                    "over the child processes (getrusage reports the largest child so far)")}
    print(json.dumps(res, indent=1))
    save_json(res, "aws_sizing.json")


if __name__ == "__main__":
    main()
