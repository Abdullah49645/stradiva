# Definition of Done

(Supersedes brief section 34, amended to make Collision Causality &
Avoidability Analysis a first-class output — not optional — per the
project owner's addendum. Inserted conceptually right after Motion
Reconstruction in the pipeline; produced for every video where two or
more vehicles' trajectories are reconstructed.)

After full development, CARAI (working name) should be able to:

## Input
An ordinary accident-style video.

## Analyze
- video timing
- scene geometry
- camera geometry
- physical scale
- vehicle motion (all vehicles present, not just one)

## Decide
Is this video quantitatively reconstructable? (section 16 — unchanged, a
core gate, not a fallback path.)

## If yes, produce:
- trajectory (per vehicle)
- estimated speed (per vehicle)
- uncertainty interval
- evidence
- reconstruction quality
- **Collision Causality & Avoidability Analysis** (new, mandatory
  wherever ≥2 vehicle trajectories were reconstructed and a collision
  point is identifiable):
  - estimated pre-impact speed and direction of each involved vehicle
  - time-to-collision / closing-rate over the observed window
  - whether evasive action (braking/steering) is detectable, and
    approximately when it began
  - whether each vehicle appears to have had a physically plausible
    opportunity to avoid the collision, under explicitly stated
    physical assumptions (max deceleration, reaction time — never
    hidden constants)
  - an **AVOIDABLE / NOT_AVOIDABLE / INSUFFICIENT_EVIDENCE** verdict
  - the visual/kinematic evidence supporting that verdict
  - **never a legal-fault claim** — physical, evidence-based language
    only ("appears to have had approximately N seconds of potential
    avoidance time"), gated by the same uncertainty system as speed
    and trajectory (if upstream confidence is too low, the module
    must output "insufficient evidence to determine avoidability"
    rather than guess)

## If no (not reconstructable):
- explicit rejection
- reason
- missing information
- confidence
- suggested human-review requirement

## Agentic case
Perceive → assess → identify missing evidence → select next visual
analysis → run tool → reassess. The agent's tool interface must expose
causality/avoidability as a queryable stage, not just geometry/tracking
tools (e.g. `assess_avoidability()` alongside `reconstruct_trajectory()`)
so the agent can, for example, decide to search for more frames
specifically because an avoidability verdict came back
INSUFFICIENT_EVIDENCE.

## Report / UI / demo — primary framing (changed)
The primary result surfaced to the user is **not** "what speed was the
car traveling" in isolation. It is, in this order:

**What happened → how the vehicles moved → what each vehicle was likely
doing → whether the collision appears physically avoidable → how
confident CARAI is in that conclusion.**

Speed and trajectory numbers remain visible (and are the evidence the
avoidability conclusion rests on) but are no longer the top-line
deliverable on their own. This applies to:
- the final written report (structure to be updated: an "Avoidability
  Assessment" section follows "Vehicle Motion" and precedes "Evidence")
- the web UI (a dedicated avoidability panel, not buried in a details tab)
- the evaluation dataset (needs collision scenarios with independently
  computable ground-truth avoidability answers, not just speed-accuracy
  scenarios — see `evaluation/scripts/generate_collision_scene.py`)
- the 5-minute competition demo (the "physical result" and "agentic
  moment" beats in the brief's demo script should show an avoidability
  verdict forming, not just a speed number)

## Deployment
A judge should be able to interact with the system through a working
endpoint/demo.
