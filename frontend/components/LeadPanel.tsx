"use client";

import { CaseFile } from "@/lib/types";
import { CartesianGrid, ComposedChart, ErrorBar, Legend, Line, ReferenceLine, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis } from "recharts";

export default function LeadPanel({ c }: { c: CaseFile }) {
  const d = c.dashcam_lead;
  if (!d) return null;
  if (d.status !== "OK" || !d.windows?.length)
    return (
      <section className="section">
        <div className="section-head"><h2>Vehicle ahead</h2></div>
        <p className="small muted">{d.reason ?? "Not measured."}</p>
      </section>
    );
  const off = d.range_offset_m ?? 0;
  const ws = d.windows;
  const mid = (w: { t_start_s: number; t_end_s: number }) => (w.t_start_s + w.t_end_s) / 2;
  const radarR = ws.filter((w) => w.radar_range_m != null).map((w) => ({ t: mid(w), r: w.radar_range_m!, v: w.radar_rel_speed_mps! }));
  const ok = ws.filter((w) => w.status === "OK" && w.range_m != null);
  const estR = ok.map((w) => ({ t: mid(w), r: w.range_m! + off, e: [w.range_m! - w.range_ci90_m![0], w.range_ci90_m![1] - w.range_m!] }));
  const estV = ok.map((w) => ({ t: mid(w), v: w.rel_speed_mps!, e: [w.rel_speed_mps! - w.rel_speed_ci90_mps![0], w.rel_speed_ci90_mps![1] - w.rel_speed_mps!] }));
  // rows where either side reports a time to collision short enough to matter
  const ttc = ws.filter((w) => w.ttc_s != null || (w.radar_ttc_s != null && w.radar_ttc_s < 20));
  const truthName = "Car's own radar (not given to CARAI)";
  return (
    <section className="section">
      <div className="section-head">
        <h2>Vehicle ahead: distance and closing speed</h2>
        <span className="small muted">{ok.length} of {ws.length} two-second windows answered</span>
      </div>
      <p className="small">
        A YOLOX-S detector on OpenCV&apos;s DNN module finds the vehicle in the camera&apos;s lane. Its distance comes from two
        independent rulers: where its tyres meet the road (camera height from the lane width) and its width in pixels
        (a car is about 1.8 m wide). When they disagree beyond their uncertainty, the window is refused.
        Pre-registered test: the first 20 s were used for development, including a constant {off.toFixed(2)} m
        camera-to-radar offset; the last 40 s were run once, unchanged.
      </p>
      <div className="grid-even">
        <div style={{ width: "100%", height: 260 }}>
          <ResponsiveContainer>
            <ComposedChart margin={{ top: 8, right: 16, bottom: 26, left: 0 }}>
              <CartesianGrid stroke="#d6dce1" />
              <XAxis dataKey="t" type="number" domain={[0, 60]} fontSize={13} label={{ value: "seconds", position: "insideBottom", offset: -14, fontSize: 13 }} />
              <YAxis fontSize={13} label={{ value: "distance (m)", angle: -90, position: "insideLeft", fontSize: 13 }} />
              <Tooltip formatter={(v: number | number[]) => (Array.isArray(v) ? "" : v.toFixed(1))} />
              <Legend verticalAlign="top" height={28} />
              <ReferenceLine x={20} stroke="#7a8691" strokeDasharray="4 4" label={{ value: "test part", fontSize: 12, position: "insideTopRight" }} />
              <Line data={radarR} dataKey="r" name={truthName} stroke="#7a8691" dot={false} strokeWidth={1.5} isAnimationActive={false} />
              <Scatter data={estR} dataKey="r" name="CARAI (90 % interval)" fill="#2f5d8c" isAnimationActive={false}><ErrorBar dataKey="e" direction="y" stroke="#2f5d8c" width={0} /></Scatter>
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <div style={{ width: "100%", height: 260 }}>
          <ResponsiveContainer>
            <ComposedChart margin={{ top: 8, right: 16, bottom: 26, left: 0 }}>
              <CartesianGrid stroke="#d6dce1" />
              <XAxis dataKey="t" type="number" domain={[0, 60]} fontSize={13} label={{ value: "seconds", position: "insideBottom", offset: -14, fontSize: 13 }} />
              <YAxis fontSize={13} label={{ value: "relative speed (m/s)", angle: -90, position: "insideLeft", fontSize: 13 }} />
              <Tooltip formatter={(v: number | number[]) => (Array.isArray(v) ? "" : v.toFixed(2))} />
              <Legend verticalAlign="top" height={28} />
              <ReferenceLine x={20} stroke="#7a8691" strokeDasharray="4 4" />
              <ReferenceLine y={0} stroke="#c9d0d6" />
              <Line data={radarR} dataKey="v" name={truthName} stroke="#7a8691" dot={false} strokeWidth={1.5} isAnimationActive={false} />
              <Scatter data={estV} dataKey="v" name="CARAI (90 % interval)" fill="#b07f00" isAnimationActive={false}><ErrorBar dataKey="e" direction="y" stroke="#b07f00" width={0} /></Scatter>
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>
      <p className="small muted">Negative relative speed: the gap is closing. The vertical line at 20 s separates development from test.</p>
      {ttc.length > 0 && (
        <div className="table-scroll">
          <table className="plain small">
            <thead><tr><th>Window</th><th>Time to collision, CARAI</th><th>Radar</th><th>Part</th></tr></thead>
            <tbody>
              {ttc.map((w) => (
                <tr key={w.t_start_s}>
                  <td className="num">{w.t_start_s.toFixed(0)}–{w.t_end_s.toFixed(0)} s</td>
                  <td className="num">{w.ttc_s != null ? `${w.ttc_s.toFixed(1)} s (${w.ttc_ci90_s![0].toFixed(1)}–${w.ttc_ci90_s![1].toFixed(1)})`
                    : w.status === "OK" ? "not reported: closing not significant" : "window refused"}</td>
                  <td className="num">{w.radar_ttc_s != null ? `${w.radar_ttc_s.toFixed(1)} s` : "not closing"}</td>
                  <td>{w.part}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
