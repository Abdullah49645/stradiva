"use client";

import { CaseFile } from "@/lib/types";
import { Units } from "@/lib/format";
import { CartesianGrid, ComposedChart, ErrorBar, Legend, Line, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis } from "recharts";

export default function DashcamPanel({ c, units }: { c: CaseFile; units: Units }) {
  const d = c.dashcam;
  if (!d) return null;
  const f = units === "metric" ? 3.6 : 2.23694;
  const u = units === "metric" ? "km/h" : "mph";
  const truth = d.windows.filter((w) => w.truth_mps != null).map((w) => ({ t: (w.t_start_s + w.t_end_s) / 2, v: w.truth_mps! * f }));
  const est = d.windows.filter((w) => w.status === "OK").map((w) => ({
    t: (w.t_start_s + w.t_end_s) / 2, v: w.speed_mps! * f, e: [(w.speed_mps! - w.ci90_mps![0]) * f, (w.ci90_mps![1] - w.speed_mps!) * f] }));
  const counts: Record<string, number> = {};
  d.windows.filter((w) => w.status !== "OK").forEach((w) => {
    const r = (w.reasons[0] || "no reason recorded").split(":")[0].split("(")[0].trim();
    counts[r] = (counts[r] || 0) + 1;
  });
  return (
    <section className="section">
      <div className="section-head"><h2>Own-vehicle speed, window by window</h2>
        <span className="small muted">assumed marking: {d.prior.name} · frame rate: {d.fps} fps ({d.fps_source})</span></div>
      <div style={{ width: "100%", height: 300 }}>
        <ResponsiveContainer>
          <ComposedChart margin={{ top: 8, right: 16, bottom: 26, left: 0 }}>
            <CartesianGrid stroke="#d6dce1" />
            <XAxis dataKey="t" type="number" domain={[0, "dataMax"]} fontSize={13} label={{ value: "seconds", position: "insideBottom", offset: -14, fontSize: 13 }} />
            <YAxis fontSize={13} label={{ value: u, angle: -90, position: "insideLeft", fontSize: 13 }} />
            <Tooltip formatter={(v: number | number[]) => (Array.isArray(v) ? "" : v.toFixed(0))} />
            <Legend verticalAlign="top" height={28} />
            {truth.length > 0 && <Line data={truth} dataKey="v" name="Car's own speed sensor (not given to CARAI)" stroke="#7a8691" dot={false} strokeWidth={1.5} isAnimationActive={false} />}
            {est.length > 0 && <Scatter data={est} dataKey="v" name="CARAI estimate (90 % interval)" fill="#2f5d8c" isAnimationActive={false}><ErrorBar dataKey="e" direction="y" stroke="#2f5d8c" width={0} /></Scatter>}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      {est.length === 0 && <p className="callout">CARAI answered none of the {d.windows.length} windows under this assumption.</p>}
      <h3 style={{ marginTop: 18 }}>Why windows were refused</h3>
      <table className="plain small"><tbody>
        {Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([k, n]) => <tr key={k}><td>{k}</td><td className="num">{n}</td></tr>)}
      </tbody></table>
    </section>
  );
}
