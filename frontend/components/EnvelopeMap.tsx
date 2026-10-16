import { Counterfactual } from "@/lib/types";

// Reaction time (x) x braking deceleration (y). Each cell: share of the
// reconstructions consistent with the video in which braking after the hazard
// would have kept the vehicles apart. The outlined box is the range of
// assumptions Odolite requires a verdict to hold across. Ticks: the latest
// reaction that still avoids contact, with its 90 % interval.

const W = 640, Hh = 300, L = 64, R = 16, T = 16, B = 46;

function fill(p: number) {
  if (p >= 0.95) return "#9fcbb3";
  if (p <= 0.05) return "#e3aaa4";
  return "#efd27a";
}

export default function EnvelopeMap({ cf }: { cf: Counterfactual }) {
  const rs = cf.reaction_grid_s;
  const as = cf.decel_grid_mps2;
  const xMin = 0.5, xMax = 2.75;
  const x = (r: number) => L + ((r - xMin) / (xMax - xMin)) * (W - L - R);
  const rowH = (Hh - T - B) / as.length;
  const y = (i: number) => T + (as.length - 1 - i) * rowH; // strongest braking on top
  const edges = rs.map((r, j) => [j === 0 ? xMin : (rs[j - 1] + r) / 2, j === rs.length - 1 ? xMax : (r + rs[j + 1]) / 2]);
  const env = cf.envelope;
  const envRows = as.map((a, i) => (a >= env.decel_mps2[0] && a <= env.decel_mps2[1] ? i : -1)).filter((i) => i >= 0);
  const envTop = y(Math.max(...envRows));
  const envBot = y(Math.min(...envRows)) + rowH;

  return (
    <figure className="envelope" style={{ margin: 0 }}>
      <svg viewBox={`0 0 ${W} ${Hh}`} role="img"
           aria-label="Map of reaction time against braking deceleration showing where the collision would have been avoided">
        {as.map((a, i) =>
          rs.map((r, j) => {
            const p = cf.p_avoid[i][j];
            return (
              <rect key={`${i}-${j}`} x={x(edges[j][0])} y={y(i)} width={x(edges[j][1]) - x(edges[j][0]) - 1}
                    height={rowH - 1} fill={fill(p)}>
                <title>{`${a} m/s², reaction ${r} s: avoided in ${(p * 100).toFixed(0)} % of reconstructions`}</title>
              </rect>
            );
          })
        )}
        <rect x={x(env.reaction_s[0])} y={envTop} width={x(env.reaction_s[1]) - x(env.reaction_s[0])}
              height={envBot - envTop} fill="none" stroke="#1e2a33" strokeWidth={2.5} />
        {as.map((a, i) => {
          const c = cf.critical_reaction_s[`${a}`] ?? cf.critical_reaction_s[a.toFixed(0)];
          if (!c) return null;
          const cy = y(i) + rowH / 2;
          const lo = Math.max(xMin, c.ci90_s[0]), hi = Math.min(xMax, c.ci90_s[1]), m = Math.min(xMax, Math.max(xMin, c.median_s));
          return (
            <g key={`c${i}`}>
              <line x1={x(lo)} x2={x(hi)} y1={cy} y2={cy} stroke="#1e2a33" strokeWidth={2} />
              <line x1={x(m)} x2={x(m)} y1={cy - rowH * 0.32} y2={cy + rowH * 0.32} stroke="#1e2a33" strokeWidth={3} />
            </g>
          );
        })}
        {as.map((a, i) => (
          <text key={`ya${i}`} x={L - 10} y={y(i) + rowH / 2 + 5} textAnchor="end" fontSize={14} fontFamily="Overpass" fill="#4a5763">
            {a}
          </text>
        ))}
        {[0.5, 1.0, 1.5, 2.0, 2.5].map((r) => (
          <text key={`xa${r}`} x={x(r)} y={Hh - B + 22} textAnchor="middle" fontSize={14} fontFamily="Overpass" fill="#4a5763">
            {r.toFixed(1)}
          </text>
        ))}
        <text x={(L + W - R) / 2} y={Hh - 6} textAnchor="middle" fontSize={14} fontFamily="IBM Plex Sans" fill="#4a5763">
          Driver reaction time after the hazard (s)
        </text>
        <text x={14} y={(T + Hh - B) / 2} textAnchor="middle" fontSize={14} fontFamily="IBM Plex Sans" fill="#4a5763"
              transform={`rotate(-90 14 ${(T + Hh - B) / 2})`}>
          Braking (m/s²)
        </text>
      </svg>
      <figcaption className="legend">
        <span><i style={{ background: "#9fcbb3" }} />Avoided in ≥ 95 % of reconstructions</span>
        <span><i style={{ background: "#efd27a" }} />Uncertain</span>
        <span><i style={{ background: "#e3aaa4" }} />Contact in ≥ 95 %</span>
        <span>Outlined box: the range a verdict must hold across</span>
        <span>Tick and bar: latest reaction that still avoids contact, with its 90 % interval</span>
      </figcaption>
    </figure>
  );
}
