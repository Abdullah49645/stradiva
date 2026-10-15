# Priority Order

(Supersedes brief section 33, amended per project owner's addendum: the
Collision Causality & Avoidability module is inserted into P0 — it is a
required core capability, not an optional add-on stacked on top of a
"done" P0 list.)

## P0 — MUST WORK

1. OpenCV 5 geometry
2. Ground-plane reconstruction
3. Vehicle tracking (**multi-vehicle** — causality analysis needs ≥2
   tracked trajectories, so single-vehicle tracking alone no longer
   satisfies this item)
4. Physical trajectory
5. Speed estimation
6. Uncertainty
7. Reconstructability/rejection
8. **Collision Causality & Avoidability Analysis** — pre-impact
   speed/direction per vehicle, time-to-collision, evasive-action
   detection, avoidable/not-avoidable/insufficient-evidence verdict,
   inheriting the uncertainty system from item 6/7. Inserted here
   because the brief's addendum makes it "a central CARAI output,
   alongside speed and trajectory — not an optional feature."
9. Quantitative evaluation (now must include avoidability-verdict
   accuracy against ground-truth-computable collision scenarios, not
   just speed MAE/RMSE)

## P1 — HIGH VALUE

10. Agentic active analysis (tool interface must expose the
    avoidability assessment as a callable stage, not just geometry/
    tracking tools)
11. Polished visualization (must include an avoidability panel, not
    just speed/trajectory charts)
12. AWS deployment
13. Report generation (report structure updated — see
    docs/definition_of_done.md — avoidability assessment is a top-level
    section, not an appendix)

## P2 — OPTIONAL

14. COOL optimization
15. Additional vehicle interactions
16. Advanced scene understanding
17. Additional reconstruction quantities

Never sacrifice P0 to add P2 features. This still holds, and now
explicitly includes item 8 under that protection — the causality module
does not get deprioritized under time pressure any more than speed
estimation would.
