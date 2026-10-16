"""
Sizing evidence for the processing Lambda (docs/aws.md): wall time and peak
resident memory of one full analyze_video run on a clip at the cap the
deployment uses (15 s, 1280x720, 30 fps), measured in a fresh child process
(resource.getrusage(RUSAGE_CHILDREN).ru_maxrss). Measured on the machine it
runs on, NOT on Lambda.

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
    code = ("import sys, json; sys.path.insert(0, %r); from backend.pipeline.analyze import analyze_video; "
            "c = analyze_video(%r, max_seconds=15); print(json.dumps({'status': c['status'], 'timings': c.get('timings_s')}))"
            % (ROOT, vp))
    t0 = time.time()
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    wall = time.time() - t0
    rss_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    last = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
    res = {"clip": {"seconds": 15.0, "width": 1280, "height": 720, "fps": 30.0},
           "wall_s": round(wall, 1), "peak_rss_mb": round(rss_kb / 1024, 0),
           "pipeline": json.loads(last[-1]) if last else {"stderr": p.stderr[-800:]},
           "machine": {"cpus": os.cpu_count(), "platform": platform.platform(), "python": platform.python_version()},
           "note": "development machine, not Lambda; Lambda at 4096 MB has about 2.3 vCPUs"}
    print(json.dumps(res, indent=1))
    save_json(res, "aws_sizing.json")


if __name__ == "__main__":
    main()
