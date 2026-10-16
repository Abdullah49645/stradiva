# COOL / Graviton Assessment

Lowest priority in the brief's own ranking (P2) and treated that way here
— this is a profiling + benchmark-readiness pass, not a claim that CARAI
should enter the COOL award. The brief is explicit: "do NOT artificially
force COOL into the project... only [pursue it if] benchmarking
demonstrates meaningful benefit." This doc follows that instruction
literally, including being willing to conclude "not yet, here's why."

## What this sandbox can and can't do

This sandbox is x86_64 (`Intel(R) Xeon(R)`), not ARM — there is no
Graviton/arm64 hardware available here, so **no real x86-vs-Graviton
comparison exists yet**. What follows is (1) profiling to find a
legitimate benchmark target, and (2) an architecture-portable benchmark
script anyone with EC2 access can run on both instance types in minutes.
Nothing below should be read as "COOL will win" — it's "here's the one
part of this pipeline where the question is even worth asking."

## Profiling result

Full single-vehicle pipeline run (`cProfile` on `run_feasibility.py`,
one ~5s synthetic clip, 147 frames): 3.30s wall time total. Breakdown:

| component | cumulative time | share |
|---|---|---|
| `cv2.VideoCapture.read()` (video decode) | 2.42s | 73% |
| per-frame tracking (cvtColor+inRange+morphologyEx+findContours, ×147) | 0.56s | 17% |
| one-time geometry/agent search (line detection, homography) | 0.10s | 3% |
| everything else (trajectory smoothing, scipy fitting, etc.) | ~0.22s | 7% |

## Why the tracking loop, not video decode, is the right benchmark target

Video decode dominates wall time, but it's mostly libavcodec software
decoding — a different optimization question (codec-level, often
hardware-accelerated separately) than the "vectorized OpenCV image
processing" story the COOL award is actually about. The one-time
geometry search is real OpenCV work but runs once per video regardless
of length, so it doesn't scale with data volume — a weak case for
"COOL saves you money at scale."

The per-frame tracking loop (`cv2.cvtColor` → `cv2.inRange` →
`cv2.morphologyEx` → `cv2.findContours`, called once per frame) is the
only candidate that's both genuinely vectorized image processing *and*
scales linearly with video length / fleet volume — exactly the
"image preprocessing, feature operations" category the brief names as a
legitimate COOL candidate, and the only one where a real per-dollar
throughput difference would actually matter at scale.

## Benchmark script

`evaluation/scripts/benchmark_tracking_loop.py` isolates exactly that
loop with zero project dependencies beyond opencv-python/numpy, so it
can be dropped onto a bare EC2 instance (x86 or Graviton) and run
as-is — no ARM-specific code, architecture-portable by construction.

**x86_64 baseline measured in this sandbox** (Intel Xeon @ 2.10GHz, 300
synthetic frames at 1280×720, OpenCV 5.0.0):

```
frames_per_second: 218.0
ms_per_frame: 4.588
```

**Not yet measured: the same script on a Graviton (arm64) instance.**
That's the number that actually decides this — run
`python evaluation/scripts/benchmark_tracking_loop.py` on a comparable
Graviton instance (e.g. `c7g.large` vs `c7i.large` for a fair per-vCPU
comparison) and compare `frames_per_second` directly, then factor in the
two instance types' hourly cost for a genuine per-dollar-throughput
number — that ratio, not raw speed, is what the award actually rewards.

## Recommendation

Don't commit to the COOL award track until that real comparison exists.
If Graviton comes back meaningfully cheaper per frame-second (a common
outcome for this class of vectorized OpenCV workload, but not
guaranteed — untested here), it's a clean, honest, non-forced entry:
migrate the tracking-loop's container to run on Graviton, keep the
benchmark script as the submission's evidence, and be upfront that only
this one stage was targeted deliberately rather than the whole pipeline.
If it doesn't come back meaningfully cheaper, the right move per the
brief's own principle is to skip the award entirely rather than submit
a forced, unconvincing entry.
