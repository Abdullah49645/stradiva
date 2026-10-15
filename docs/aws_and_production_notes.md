# Production Pipeline & AWS Notes

## The production entry point (`backend/api/process_video.py`)

Unlike `evaluation/scripts/run_causality_feasibility.py` (which uses the
synthetic scene's known camera model to render a perfectly clean
vehicle-free background reference), `process_video()` has **no privileged
knowledge of the scene** — this is what actually runs in production
(Lambda, see below). Background is estimated via temporal median over
~20 sampled frames, geometry via the agentic frame-search loop, tracking
via background subtraction seeded from that estimate.

## What was validated, honestly

Ran `process_video()` against `synthetic_collision_production_demo.mp4`
— a new scenario built specifically to be more representative of real
rear-end collisions than the earlier two (lead vehicle A **drives and
brakes**, rather than being parked for the entire clip, both because
that's the more common real scenario and because it gives the background
estimator clean road samples across a range of frames).

Result: geometry/scale recovery succeeded (agent found a good frame,
reconstructability 71.8/100), both vehicles tracked for the first ~2.4s
of the clip (73/137 frames) — but tracking of vehicle A degraded and
was lost well before the collision point, because A is still stationary
for roughly the back half of the clip, which is enough for a 20-sample
temporal median to partially absorb it. The pipeline's collision-
detection heuristic correctly found no reliable late-window gap
measurement and returned `INSUFFICIENT_EVIDENCE` rather than guessing —
**the right failure mode**, but it means the production tracker's
effective ceiling is still below what the two hand-validated collision
scenarios (which use privileged synthetic-scene background rendering)
demonstrate.

## The actual open problem

Passive, motion-only vehicle detection (background subtraction, optical
flow, frame differencing — anything that infers "vehicle" purely from
*movement*) cannot, in principle, reliably detect an object that stops
moving for an extended period, because it becomes visually
indistinguishable from background by the same signal the detector relies
on. This isn't a bug to patch — it's a real limit of the technique
class, and it matters a lot here because "a vehicle stopped in the lane
when it was struck" is one of the most common real rear-end collision
setups, not an edge case.

**Path forward (not yet built):** replace or supplement the background-
subtraction detector with an appearance-based detector that doesn't
depend on motion — section 13 of the master brief already anticipates
this ("Use a suitable object detector/tracker, but do NOT make generic
object detection the core contribution... OpenCV must remain central").
A lightweight pretrained vehicle detector (even a classical HOG+SVM or a
small learned model, run per-frame rather than relying on temporal
background) would detect a stationary vehicle exactly as well as a
moving one, at the cost of being the "generic detector" the brief warns
against over-relying on. The reconstruction/scale/causality math
downstream is unaffected either way — this is purely a detection-stage
swap.

## AWS architecture actually shipped

See `infrastructure/aws/`:
- `lambda_handler.py` — S3-triggered Lambda that runs `process_video()`
  on an uploaded clip and writes JSON + the markdown report back to S3,
  updating a DynamoDB job-status record.
- `api_handler.py` — API Gateway Lambda: `POST /jobs` issues a presigned
  S3 upload URL and creates a `PENDING` job record; `GET /jobs/{id}`
  returns status and, once done, the result.
- `template.yaml` — AWS SAM template: S3 bucket (uploads/ + results/
  prefixes), the two Lambda functions, a DynamoDB table, API Gateway,
  least-privilege IAM roles, and the S3 → processing-Lambda event
  trigger.
- `Dockerfile.lambda` — container image (OpenCV 5 doesn't fit
  comfortably in a zip-based Lambda layer at reasonable size/cold-start,
  so this uses a container-image Lambda instead).

### Why this architecture, not the full Step Functions/ECS version

Per section 20's own instruction ("do not use every AWS service just
because it exists... choose the architecture based on actual workload
requirements"): each clip's processing is single-digit seconds of
compute on a short video. That fits comfortably inside one Lambda
invocation (container image, generous memory for OpenCV, a few-minute
timeout) with no orchestration step needed. Step Functions/ECS would
earn their complexity if clips got long enough to risk Lambda's 15-
minute ceiling, or if the pipeline needed to fan out (e.g. parallel
per-vehicle processing) — neither is true yet. This is a stated,
revisitable choice, not a permanent one.

### Honesty about what's been tested

**This infrastructure has not been deployed against live AWS in this
environment** — the sandbox has no AWS network egress (see the network
allowlist; AWS domains aren't in it), so `sam deploy` and any live
Lambda invocation are unverified against a real account here.

What *has* been verified without live AWS, and is worth taking
seriously rather than dismissing as "untested": `cfn-lint` validates the
SAM template cleanly, and `tests/test_aws_handlers.py` runs both actual
Lambda handler functions against **moto** (in-memory mocked AWS) —
exercising the real code, not a rewritten stand-in — through the full
upload → S3-trigger → `process_video()` → S3-write → DynamoDB-update →
API-read round trip, plus two distinct failure paths. This caught and
fixed a real bug: the error-handling code path itself threw an
exception (`error` is a DynamoDB reserved keyword), which would have
silently stuck any real production failure at `PROCESSING` forever with
no recorded error message. See `docs/aws_deployment_runbook.md` for
exactly what this does and doesn't cover, and the exact commands to run
the actual `sam build && sam deploy` once someone has AWS credentials.
