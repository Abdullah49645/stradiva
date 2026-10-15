import { AgentTrace } from "@/lib/types";
import Eyebrow from "./Eyebrow";
import Panel from "./Panel";

export default function AgentTracePanel({ trace }: { trace?: AgentTrace }) {
  if (!trace) return null;

  return (
    <Panel>
      <div className="flex flex-wrap items-center justify-between gap-2 mb-1">
        <Eyebrow>Agentic Geometry Recovery</Eyebrow>
        <span className={`font-mono text-[11px] uppercase tracking-wider whitespace-nowrap ${trace.success ? "text-hud" : "text-hazard"}`}>
          {trace.success ? "resolved" : "exhausted"}
        </span>
      </div>
      <p className="text-sm text-muted mb-4 mt-2">
        {trace.n_iterations} iteration{trace.n_iterations === 1 ? "" : "s"}, settled on frame{" "}
        <span className="font-mono text-paper">{trace.chosen_frame_idx ?? "—"}</span>.{" "}
        {trace.stop_reason}
      </p>
      <ol className="space-y-0">
        {trace.calls.map((call, i) => (
          <li key={i} className="relative pl-5 py-2.5 border-l border-hairline last:border-transparent">
            <span className="absolute left-[-3.5px] top-3.5 w-[7px] h-[7px] rounded-full bg-hud" />
            <div className="font-mono text-[13px] text-hud">{call.tool}<span className="text-muted">()</span></div>
            <div className="font-mono text-[11px] text-muted mt-0.5 break-all">
              {JSON.stringify(call.args)} <span className="text-hairline">→</span> {JSON.stringify(call.result)}
            </div>
            {call.reasoning && (
              <div className="text-[13px] text-paper/80 mt-1">{call.reasoning}</div>
            )}
          </li>
        ))}
      </ol>
    </Panel>
  );
}
