# Real Footage

Every other scenario in `evaluation/dataset/` is synthetic
(`scene_generator.py`). This folder is for real video, which the brief
(sections 24-27) treats as necessary evidence, not a nice-to-have: an
all-synthetic evaluation invites the exact "only works on one cherry-picked
video" criticism the brief explicitly warns against.

## How to use

Drop any `.mp4`/`.mov`/`.avi` file in this folder, then run:

```
python evaluation/scripts/run_on_real_footage.py
```

with no arguments -- it runs every video here through the real production
pipeline (agentic geometry search, roll correction, tracking,
reconstructability gate, speed + uncertainty) and writes
`docs/real_footage_results.json`. No ground truth is required. If you
happen to know a clip's true speed independently (a GPS-logged test
drive, a radar-gun reading), pass it:

```
python evaluation/scripts/run_on_real_footage.py path/to/clip.mp4 --true-speed-mps 15.6
```

## What to put here

Two kinds of clips are both useful, and for different reasons:

1. **Ordinary dashcam/CCTV driving footage with visible lane markings.**
   Tests the reconstruction path end-to-end on video the pipeline was
   never tuned against. A correct, well-calibrated speed estimate here is
   a much stronger result than another clean synthetic win.
2. **Deliberately hard footage** — night driving, tunnels, heavy rain,
   extreme angles, no visible lane markings. Tests the *rejection* path.
   The brief is explicit that a system correctly saying "insufficient
   evidence" is a result, not a failure — this project's whole premise is
   that CARAI must know when it can't answer, so a folder of only "nice"
   footage undersells that.

## What NOT to put here (a deliberate choice, not an oversight)

Actual crash/collision footage of real, identifiable people was not
sourced for this folder, even though it's the project's literal use
case. Real crash videos can show real injury or distress, involve people
who never consented to their worst moment becoming a training/demo
asset, and raise licensing questions that need a human judgment call
about a specific clip, not a blanket policy. If you want to add real
collision footage, that selection should be made deliberately by a
person, clip by clip -- not auto-populated.

Free, no-attribution-required sources for the two categories above (as
of this writing):
- Pixabay's "Car Dashcam" and "Road, Highway, Lanes" categories
  (pixabay.com/videos/search/car%20dashcam/, content license — free to
  use, no attribution required)
- Internet Archive's Moving Image Archive / Stock Footage collections,
  filterable by license (archive.org)

These weren't downloaded automatically -- picking specific clips and
checking each one's actual usage terms is worth doing by hand, and this
sandbox's network access doesn't reach stock-footage sites anyway (see
git history / session notes for the allowlist).
