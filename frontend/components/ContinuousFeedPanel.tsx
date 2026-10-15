import { ErraticDrivingResult } from "@/lib/types";
import Eyebrow from "./Eyebrow";
import Panel from "./Panel";

const TYPE_COLOR: Record<string, string> = {
  WEAVING: "#E8A33D",
  HARSH_BRAKING: "#C1443C",
  HARSH_BRAKE_RELEASE: "#C1443C",
};

export default function ContinuousFeedPanel({ result }: { result?: ErraticDrivingResult }) {
  if (!result) return null;
  const gtSegments = result.ground_truth_erratic_segments;
  const detected = result.detected;
  const totalEnd = Math.max(
    ...gtSegments.map((s) => s.t_end_s),
    ...detected.map((d) => d.end_s),
    1
  );

  return (
    <Panel>
      <Eyebrow>Continuous-Feed Erratic-Driving Detection</Eyebrow>
      <p className="text-[13px] text-muted mt-3 mb-5 leading-relaxed">
        The same tracking and uncertainty-aware thresholding, applied to a continuous feed instead
        of one triggered clip: is this vehicle, right now, driving in a way worth flagging.
        Measured against real labeled ground truth — recall and precision below are honest, not
        rounded up. See <code className="text-hud">docs/erratic_driving_detection.md</code>.
      </p>

      <div className="flex gap-6 mb-5 font-mono text-[12px]">
        <div>
          <span className="text-muted">recall </span>
          <span className="text-paper">{result.recall !== null ? `${Math.round(result.recall * 100)}%` : "n/a"}</span>
        </div>
        <div>
          <span className="text-muted">precision </span>
          <span className="text-paper">{result.precision !== null ? `${Math.round(result.precision * 100)}%` : "n/a"}</span>
        </div>
        <div>
          <span className="text-muted">tracking coverage </span>
          <span className="text-paper">{Math.round(result.tracking_coverage * 100)}%</span>
        </div>
      </div>

      {/* Timeline: ground truth track on top, detected intervals below */}
      <div className="mb-2 font-mono text-[10px] uppercase tracking-wider text-muted">ground truth</div>
      <div className="relative h-6 bg-graphite2 rounded mb-3 overflow-hidden">
        {gtSegments.map((s, i) => (
          <div
            key={i}
            title={`${s.type} ${s.t_start_s.toFixed(1)}s-${s.t_end_s.toFixed(1)}s`}
            className="absolute top-0 bottom-0 opacity-70"
            style={{
              left: `${(s.t_start_s / totalEnd) * 100}%`,
              width: `${((s.t_end_s - s.t_start_s) / totalEnd) * 100}%`,
              backgroundColor: TYPE_COLOR[s.type] ?? "#8B9296",
            }}
          />
        ))}
      </div>

      <div className="mb-2 font-mono text-[10px] uppercase tracking-wider text-muted">detected</div>
      <div className="relative h-6 bg-graphite2 rounded mb-5 overflow-hidden">
        {detected.map((d, i) => (
          <div
            key={i}
            title={`${d.type} ${d.start_s.toFixed(1)}s-${d.end_s.toFixed(1)}s`}
            className="absolute top-0 bottom-0"
            style={{
              left: `${(d.start_s / totalEnd) * 100}%`,
              width: `${Math.max(0.5, ((d.end_s - d.start_s) / totalEnd) * 100)}%`,
              backgroundColor: TYPE_COLOR[d.type] ?? "#7FD1C6",
            }}
          />
        ))}
      </div>

      <div className="space-y-2">
        {detected.map((d, i) => (
          <div key={i} className="font-mono text-[11px] text-muted flex gap-2">
            <span style={{ color: TYPE_COLOR[d.type] ?? "#7FD1C6" }} className="shrink-0">
              {d.start_s.toFixed(1)}–{d.end_s.toFixed(1)}s {d.type}
            </span>
            <span className="text-muted/80">{d.reason}</span>
          </div>
        ))}
      </div>
    </Panel>
  );
}
