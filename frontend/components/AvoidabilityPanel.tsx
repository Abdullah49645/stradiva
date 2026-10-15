import { CausalityAssessment, UncertaintyInfo } from "@/lib/types";
import Eyebrow from "./Eyebrow";
import Panel from "./Panel";

const VERDICT_META: Record<string, { color: string; textColor: string; label: string }> = {
  AVOIDABLE: { color: "#E8A33D", textColor: "text-signal", label: "Appears Avoidable" },
  NOT_AVOIDABLE: { color: "#C1443C", textColor: "text-hazard", label: "Appears Unavoidable" },
  INSUFFICIENT_EVIDENCE: { color: "#6B7280", textColor: "text-insufficient", label: "Insufficient Evidence" },
};

export default function AvoidabilityPanel({
  causality,
  uncertainty,
}: {
  causality?: CausalityAssessment;
  uncertainty?: UncertaintyInfo;
}) {
  if (!causality) {
    return (
      <Panel>
        <p className="text-muted">No avoidability assessment available for this result.</p>
      </Panel>
    );
  }

  const meta = VERDICT_META[causality.verdict] ?? VERDICT_META.INSUFFICIENT_EVIDENCE;

  return (
    <Panel reticleColor={meta.color} className="relative overflow-hidden">
      <div
        className="absolute inset-0 opacity-[0.06] pointer-events-none"
        style={{
          backgroundImage: `repeating-linear-gradient(0deg, ${meta.color} 0, ${meta.color} 1px, transparent 1px, transparent 28px)`,
        }}
      />
      <div className="relative">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
          <Eyebrow color={meta.textColor}>Avoidability Assessment</Eyebrow>
          <span className="font-mono text-[11px] uppercase tracking-wider text-muted border border-hairline rounded px-2 py-1 whitespace-nowrap">
            confidence:&nbsp;<span className={meta.textColor}>{causality.confidence ?? "n/a"}</span>
          </span>
        </div>

        <h2 key={causality.verdict} className={`scan-reveal verdict-glow font-display text-4xl sm:text-6xl font-semibold mb-4 ${meta.textColor}`}>
          {meta.label}
        </h2>

        <p className="text-paper/90 leading-relaxed mb-6 max-w-3xl">{causality.narrative}</p>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-px bg-hairline border border-hairline rounded overflow-hidden mb-5">
          <Stat label="Pre-impact speed"
                value={causality.striking_vehicle_speed_at_earliest_observed_mps != null
                  ? `${causality.striking_vehicle_speed_at_earliest_observed_mps.toFixed(1)}` : "—"}
                unit="m/s"
                sub={
                  uncertainty?.striking_vehicle_speed_ci90_mps
                    ? `90% CI ${uncertainty.striking_vehicle_speed_ci90_mps[0].toFixed(1)}–${uncertainty.striking_vehicle_speed_ci90_mps[1].toFixed(1)} m/s`
                    : undefined
                } />
          <Stat label="Time-to-collision"
                value={causality.time_to_collision_at_earliest_observed_s != null
                  ? `${causality.time_to_collision_at_earliest_observed_s.toFixed(1)}` : "—"}
                unit="s (earliest obs.)" />
          <Stat label="Evasive action"
                value={
                  causality.braking_detected && causality.steering_detected ? "BRAKE + STEER" :
                  causality.braking_detected ? "BRAKING" :
                  causality.steering_detected ? `STEERING (${causality.steering_direction})` : "NONE DETECTED"
                } unit="" />
          <Stat label="Avoidance window"
                value={causality.last_possible_avoidance_time_before_impact_s != null
                  ? `${causality.last_possible_avoidance_time_before_impact_s.toFixed(1)}` : "—"}
                unit="s before impact" />
        </div>

        {causality.assumptions && (
          <p className="font-mono text-[11px] text-muted mb-4">
            assumptions // max_decel={causality.assumptions.max_decel_mps2} m/s² &nbsp; reaction_time={causality.assumptions.reaction_time_s}s
            <span className="text-muted/70"> — stated priors, not measured properties of this vehicle or road</span>
          </p>
        )}

        {uncertainty?.dominant_sources && uncertainty.dominant_sources.length > 0 && (
          <p className="font-mono text-[11px] text-muted mb-4">
            dominant uncertainty // {uncertainty.dominant_sources.join(", ")}
            <span className="text-muted/70"> — see docs/uncertainty_calibration.md for how the interval is calibrated</span>
          </p>
        )}

        {causality.evidence && causality.evidence.length > 0 && (
          <details className="text-sm">
            <summary className="cursor-pointer font-mono text-[11px] uppercase tracking-wider text-muted hover:text-hud transition-colors">
              ▸ Evidence Log
            </summary>
            <ul className="mt-2 space-y-1.5 text-paper/80 pl-4 border-l border-hairline">
              {causality.evidence.map((e, i) => <li key={i}>{e}</li>)}
            </ul>
          </details>
        )}

        {causality.contribution_factors && (
          <div className="mt-4 pt-4 border-t border-hairline">
            <Eyebrow color="text-muted">Physical Contribution Factors</Eyebrow>
            <p className="text-xs text-muted/80 mt-2 mb-3 italic">
              {causality.contribution_factors.note}
            </p>
            <div className="grid sm:grid-cols-2 gap-3">
              {causality.contribution_factors.following_vehicle_safe_speed_for_observed_gap_mps != null && (
                <div className="bg-graphite2 p-3 rounded">
                  <div className="font-mono text-[10px] uppercase tracking-wide text-muted mb-1">Following vehicle vs. modeled safe speed</div>
                  <div className="font-mono text-lg">
                    {causality.contribution_factors.following_vehicle_speed_vs_safe_speed_pct != null
                      ? `${causality.contribution_factors.following_vehicle_speed_vs_safe_speed_pct > 0 ? "+" : ""}${causality.contribution_factors.following_vehicle_speed_vs_safe_speed_pct.toFixed(0)}%`
                      : "—"}
                    <span className="text-xs text-muted ml-2">
                      (safe: {causality.contribution_factors.following_vehicle_safe_speed_for_observed_gap_mps} m/s)
                    </span>
                  </div>
                </div>
              )}
              {causality.contribution_factors.lead_vehicle_deceleration_classification && (
                <div className="bg-graphite2 p-3 rounded">
                  <div className="font-mono text-[10px] uppercase tracking-wide text-muted mb-1">Lead vehicle braking profile</div>
                  <div className="font-mono text-lg capitalize">
                    {causality.contribution_factors.lead_vehicle_deceleration_classification}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        <p className="text-xs text-muted/70 mt-5 italic border-t border-hairline pt-4">
          Physical, evidence-based assessment only. Does not determine legal fault.
        </p>
      </div>
    </Panel>
  );
}

function Stat({ label, value, unit, sub }: { label: string; value: string; unit: string; sub?: string }) {
  return (
    <div className="bg-graphite2 p-3">
      <div className="font-mono text-[10px] uppercase tracking-wide text-muted mb-1">{label}</div>
      <div className="font-mono text-xl font-medium text-paper">
        {value}
        {unit && value !== "—" && <span className="text-xs text-muted ml-1">{unit}</span>}
      </div>
      {sub && <div className="font-mono text-[10px] text-muted mt-0.5">{sub}</div>}
    </div>
  );
}
