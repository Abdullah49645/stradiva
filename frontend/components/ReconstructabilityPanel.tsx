import { Geometry, Reconstructability } from "@/lib/types";
import Eyebrow from "./Eyebrow";
import Panel from "./Panel";

export default function ReconstructabilityPanel({
  recon,
  geometry,
}: {
  recon?: Reconstructability;
  geometry?: Geometry;
}) {
  if (!recon) return null;

  const verdictColor =
    recon.verdict === "RECONSTRUCTABLE" ? "text-hud" :
    recon.verdict === "NOT_RECONSTRUCTABLE" ? "text-hazard" : "text-signal";

  const rollDeg = geometry?.roll_correction_applied_deg;

  return (
    <Panel>
      <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
        <Eyebrow>Reconstruction Quality</Eyebrow>
        <span className={`font-mono text-[11px] uppercase tracking-wider whitespace-nowrap ${verdictColor}`}>{recon.verdict}</span>
      </div>
      <div className="mb-5 flex items-baseline gap-2">
        <span className="font-display text-4xl font-semibold">{recon.overall}</span>
        <span className="font-mono text-sm text-muted">/ 100</span>
      </div>
      {rollDeg !== undefined && rollDeg !== null && rollDeg !== 0 && (
        <div className="mb-4 flex items-center gap-2 rounded border border-hairline bg-graphite2 px-3 py-2 font-mono text-[11px] text-muted">
          <span className="text-hud">camera stabilization</span>
          <span>·</span>
          <span className="text-paper">
            {rollDeg > 0 ? "+" : ""}{rollDeg.toFixed(1)}° roll auto-corrected before geometry fit
          </span>
        </div>
      )}
      {geometry?.skeptic_findings && geometry.skeptic_findings.length > 0 && (
        <div className={`relative mb-4 rounded border px-3 py-2 font-mono text-[11px] ${
          geometry.skeptic_disputed ? "hud-sweep border-hazard/40 bg-hazard/10 text-hazard" : "border-hairline bg-graphite2 text-muted"
        }`}>
          <div className="mb-1 uppercase tracking-wider flex items-center gap-1.5">
            {geometry.skeptic_disputed && <span className="inline-block w-1.5 h-1.5 rounded-full bg-hazard pulse-dot" />}
            {geometry.skeptic_disputed ? "⚠ disputed by the skeptic" : "cross-examined · no dispute"}
          </div>
          <ul className="space-y-1">
            {geometry.skeptic_findings.map((f, i) => (
              <li key={i} className={f.disputed ? "text-hazard" : "text-muted"}>
                — {f.check.replace(/_/g, " ")}: {f.reason}
              </li>
            ))}
          </ul>
        </div>
      )}
      <div className="space-y-3">
        {Object.entries(recon.components).map(([key, value]) => (
          <div key={key}>
            <div className="flex justify-between font-mono text-[11px] text-muted mb-1 uppercase tracking-wide">
              <span>{key}</span><span>{value}</span>
            </div>
            <div className="h-1 bg-graphite2 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full ${value >= 60 ? "bg-hud" : value >= 30 ? "bg-signal" : "bg-hazard"}`}
                style={{ width: `${Math.min(100, value)}%` }}
              />
            </div>
          </div>
        ))}
      </div>
      {recon.reasons && recon.reasons.length > 0 && (
        <ul className="mt-4 font-mono text-[11px] text-muted space-y-1 border-t border-hairline pt-3">
          {recon.reasons.map((r, i) => <li key={i}>— {r}</li>)}
        </ul>
      )}
    </Panel>
  );
}
