import { CaseFile, Check } from "@/lib/types";
import { CHECK_NAMES, STATUS_TEXT, Units, secs, speedRange } from "@/lib/format";
import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

function statusClass(s: string) {
  if (s === "MEASURED" || s === "MEASURED_GIVEN_PRIOR" || s === "MEASURED_BY_EXTRAPOLATION") return "s-measured";
  if (s === "ASSUMED") return "s-assumed";
  return "s-unknown";
}

export function Establishes({ c }: { c: CaseFile }) {
  if (!c.identifiability?.length) return null;
  return (
    <div className="table-scroll">
      <table className="plain">
        <thead><tr><th>Quantity</th><th>Status</th><th>Basis</th></tr></thead>
        <tbody>
          {c.identifiability.map((r) => (
            <tr key={r.quantity}>
              <td><strong>{r.quantity}</strong></td>
              <td><span className={`status ${statusClass(r.status)}`}>{STATUS_TEXT[r.status] ?? r.status}</span></td>
              <td className="small">{r.basis}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function checkClass(s: Check["status"]) {
  return s === "PASS" ? "s-pass" : s === "DISPUTE" ? "s-dispute" : s === "ADVISORY" ? "s-advisory" : "s-na";
}
const CHECK_STATUS: Record<string, string> = {
  PASS: "Holds",
  DISPUTE: "Disputed",
  ADVISORY: "Holds for the two vehicles; a bystander's speeds are withheld",
  NOT_APPLICABLE: "Not testable here",
};

export function CrossExamination({ c }: { c: CaseFile }) {
  const sk = c.skeptic;
  if (!sk) return null;
  return (
    <div>
      <ul className="checks">
        {sk.checks.map((k) => (
          <li key={k.check}>
            <div><span className={`status ${checkClass(k.status)}`}>{CHECK_STATUS[k.status]}</span></div>
            <div>
              <strong>{CHECK_NAMES[k.check] ?? k.check}</strong>
              <div className="small">{k.evidence}</div>
              <div className="small muted">Would catch: {k.catches}</div>
            </div>
          </li>
        ))}
      </ul>
      {sk.untestable?.length ? (
        <p className="callout small" style={{ marginTop: 14 }}>
          Not testable from this footage: {sk.untestable.join("; ")}.
        </p>
      ) : null}
    </div>
  );
}

export function MotionChart({ c, units }: { c: CaseFile; units: Units }) {
  const withheld = new Set((c.speeds ?? []).filter((s) => s.withheld).map((s) => s.vehicle_id));
  const vs = (c.vehicles ?? []).filter((v) => !withheld.has(v.id));
  if (!vs.length) return null;
  const f = units === "metric" ? 3.6 : 2.23694;
  const names: Record<number, string> = {};
  if (c.collision?.follower_id != null) names[c.collision.follower_id] = "Following vehicle";
  if (c.collision?.lead_id != null) names[c.collision.lead_id] = "Lead vehicle";
  const series = vs.map((v, i) => ({
    name: names[v.id] ?? `Vehicle ${i + 1}`,
    data: v.t_s.map((t, k) => ({ t: +t.toFixed(3), v: Math.abs(v.v_mps[k]) * f })),
  }));
  const colors = ["#2f5d8c", "#b07f00", "#9e2a2b", "#2f6b4f"];
  return (
    <>
    {withheld.size > 0 && <p className="small muted">{withheld.size} other vehicle{withheld.size > 1 ? "s are" : " is"} not shown: its reconstructed motion is physically impossible, which means it was mistracked.</p>}
    <div style={{ width: "100%", height: 280 }}>
      <ResponsiveContainer>
        <LineChart margin={{ top: 8, right: 16, bottom: 24, left: 0 }}>
          <CartesianGrid stroke="#d6dce1" />
          <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} tickFormatter={(x) => `${x.toFixed(1)}`}
                 label={{ value: "time (s)", position: "insideBottom", offset: -12, fontSize: 13 }} fontSize={13} />
          <YAxis fontSize={13} label={{ value: units === "metric" ? "km/h" : "mph", angle: -90, position: "insideLeft", fontSize: 13 }} />
          <Tooltip formatter={(v: number) => v.toFixed(1)} labelFormatter={(l) => `t = ${Number(l).toFixed(2)} s`} />
          <Legend verticalAlign="top" height={28} />
          {series.map((s, i) => (
            <Line key={s.name} data={s.data} dataKey="v" name={s.name} stroke={colors[i % 4]} dot={false} strokeWidth={2} isAnimationActive={false} />
          ))}
          {c.hazard_onset?.source === "lead_brake_onset" && (
            <ReferenceLine x={c.hazard_onset.t_s} stroke="#d9a400" strokeWidth={2} label={{ value: "lead brakes", fontSize: 12, position: "top" }} />
          )}
          {c.collision?.t_contact_s != null && (
            <ReferenceLine x={c.collision.t_contact_s} stroke="#9e2a2b" strokeWidth={2} label={{ value: "contact", fontSize: 12, position: "top" }} />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
    </>
  );
}

export function Validation({ c }: { c: CaseFile }) {
  if (!c.validation) return null;
  return (
    <div>
      <p className="small muted">{c.validation.note}</p>
      <div className="table-scroll">
        <table className="plain">
          <thead><tr><th>Quantity</th><th>Ground truth</th><th>CARAI</th><th></th></tr></thead>
          <tbody>
            {c.validation.rows.map((r) => (
              <tr key={r.label}>
                <td>{r.label}</td><td className="num">{r.truth}</td><td className="num">{r.carai}</td>
                <td>{r.ok == null ? "" : r.ok ? <span className="status s-pass">consistent</span> : <span className="status s-dispute">differs</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function KeyFacts({ c, units }: { c: CaseFile; units: Units }) {
  const facts: { k: string; v: string; sub?: string }[] = [];
  if (c.follower_speed_at_hazard_onset)
    facts.push({ k: "Following vehicle when the hazard began", v: speedRange(c.follower_speed_at_hazard_onset.ci90_mps, units), sub: "90 % interval" });
  else {
    const sp = (c.speeds ?? []).find((s) => s.ci90_mps && !s.withheld);
    if (sp) facts.push({ k: "Vehicle speed", v: speedRange(sp.ci90_mps, units), sub: "90 % interval" });
  }
  const cr = c.counterfactual?.critical_reaction_s?.["7"];
  if (cr) facts.push({ k: "Latest reaction that avoids contact, at 7 m/s²", v: `${cr.median_s.toFixed(2)} s`,
                       sub: `90 % interval ${cr.ci90_s[0].toFixed(2)}–${cr.ci90_s[1].toFixed(2)} s` });
  if (c.hazard_onset)
    facts.push({ k: "Hazard onset", v: c.hazard_onset.source === "lead_brake_onset" ? secs(c.hazard_onset.t_s) : "not observed",
                 sub: c.hazard_onset.source === "lead_brake_onset" ? "lead vehicle brake onset, measured" : "reaction budgets are relative to the first tracked frame" });
  if (c.collision?.t_contact_s != null)
    facts.push({ k: "Contact", v: secs(c.collision.t_contact_s), sub: c.collision.source });
  if (!facts.length) return null;
  return (
    <div className="facts">
      {facts.map((f) => (
        <div className="fact" key={f.k}><div className="k">{f.k}</div><div className="v">{f.v}<small>{f.sub}</small></div></div>
      ))}
    </div>
  );
}
