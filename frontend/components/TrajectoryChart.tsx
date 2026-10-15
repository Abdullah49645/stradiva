"use client";

import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import Eyebrow from "./Eyebrow";

interface Trajectory {
  t_s: number[];
  x_m: number[];
  speed_mps: number[];
}

export default function TrajectoryChart({
  vehicleA,
  vehicleB,
  brakingOnsetBeforeImpact,
  collisionTimeS,
}: {
  vehicleA?: Trajectory;
  vehicleB?: Trajectory;
  brakingOnsetBeforeImpact?: number | null;
  collisionTimeS?: number | null;
}) {
  if (!vehicleB || vehicleB.t_s.length === 0) return null;

  const rows = vehicleB.t_s.map((t, i) => ({
    t: Number(t.toFixed(2)),
    speedB: Number((vehicleB.speed_mps[i] ?? 0).toFixed(1)),
    speedA: vehicleA && vehicleA.t_s[i] != null ? Number((vehicleA.speed_mps[i] ?? 0).toFixed(1)) : undefined,
  }));

  const brakingT = collisionTimeS != null && brakingOnsetBeforeImpact != null
    ? Number((collisionTimeS - brakingOnsetBeforeImpact).toFixed(2))
    : null;

  return (
    <div>
      <Eyebrow>Reconstructed Speed — Pre-Impact Window</Eyebrow>
      <div className="h-56 mt-3 -ml-4">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={rows} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="#2A2F33" strokeDasharray="2 4" vertical={false} />
            <XAxis
              dataKey="t"
              type="number"
              domain={["dataMin", "dataMax"]}
              stroke="#8B9296"
              tick={{ fontFamily: "var(--font-mono)", fontSize: 11, fill: "#8B9296" }}
              tickFormatter={(v) => `${v}s`}
              axisLine={{ stroke: "#2A2F33" }}
              tickLine={false}
              tickCount={6}
            />
            <YAxis
              stroke="#8B9296"
              tick={{ fontFamily: "var(--font-mono)", fontSize: 11, fill: "#8B9296" }}
              tickFormatter={(v) => `${v}`}
              axisLine={false}
              tickLine={false}
              label={{ value: "m/s", angle: -90, position: "insideLeft", fill: "#8B9296", fontSize: 11 }}
            />
            <Tooltip
              contentStyle={{
                background: "#15181A", border: "1px solid #2A2F33", borderRadius: 4,
                fontFamily: "var(--font-mono)", fontSize: 12,
              }}
              labelFormatter={(v) => `t = ${v}s`}
            />
            {brakingT != null && (
              <ReferenceLine x={brakingT} stroke="#E8A33D" strokeDasharray="3 3"
                label={{ value: "braking onset", fill: "#E8A33D", fontSize: 10, position: "top" }} />
            )}
            <Line type="monotone" dataKey="speedB" name="Following vehicle" stroke="#7FD1C6" strokeWidth={2} dot={false} />
            {vehicleA && vehicleA.t_s.length > 0 && (
              <Line type="monotone" dataKey="speedA" name="Lead vehicle" stroke="#8B9296" strokeWidth={1.5} dot={false} strokeDasharray="4 2" />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="flex gap-4 mt-1 font-mono text-[11px] text-muted">
        <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-hud inline-block" /> following vehicle</span>
        {vehicleA && vehicleA.t_s.length > 0 && (
          <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-muted inline-block" /> lead vehicle</span>
        )}
        {brakingT != null && (
          <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-signal inline-block" /> braking onset</span>
        )}
      </div>
    </div>
  );
}
