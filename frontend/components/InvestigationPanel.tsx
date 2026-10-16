import { CaseFile, InvestigationStep } from "@/lib/types";
import { CHECK_NAMES, INPUT_NAMES, STANDARD_NAMES, VERDICT_TEXT } from "@/lib/format";

function plain(s: string | undefined): string {
  if (!s) return "";
  let out = s;
  for (const [id, name] of Object.entries(CHECK_NAMES)) {
    out = out.split(`${id}:`).join(`${name}:`);
    out = out.split(id).join(`“${name.toLowerCase()}”`);
  }
  out = out.split("carai_synthetic_centerline").join("CARAI synthetic 3 m + 9 m");
  return out.charAt(0).toUpperCase() + out.slice(1);
}

function outcomeOf(st: InvestigationStep): { text: string; cls: string } | null {
  if (!st.decision) return null;
  if (st.decision.startsWith("adopted") || st.decision.startsWith("the verdict (")) return { text: "Adopted", cls: "s-pass" };
  if (st.decision.startsWith("the standards disagree")) return { text: "No verdict", cls: "s-advisory" };
  return { text: "Refusal stands", cls: "s-na" };
}

function statusLabel(status?: string, verdict?: string | null) {
  if (!status) return "–";
  if (status === "DISPUTED") return "disputed";
  return (VERDICT_TEXT[verdict ?? ""]?.title ?? status.toLowerCase().replace(/_/g, " "));
}

function Step({ st }: { st: InvestigationStep }) {
  const o = outcomeOf(st);
  const obs = typeof st.observation === "string" ? st.observation : "";
  return (
    <li className="inv-step">
      <div className="inv-head">
        <strong>{plain(st.diagnosis)}</strong>
        {o && <span className={`status ${o.cls}`}>{o.text}</span>}
      </div>
      {st.action && <p className="small"><span className="muted">Tried: </span>{st.action}.</p>}
      {obs && <p className="small"><span className="muted">Found: </span>{obs}.</p>}
      {st.candidates?.length ? (
        <div className="table-scroll">
          <table className="plain small">
            <thead><tr><th>Standard</th><th>Cycle</th><th>Result under it</th></tr></thead>
            <tbody>
              {st.candidates.map((c) => (
                <tr key={c.standard}>
                  <td>{STANDARD_NAMES[c.standard] ?? c.name}</td>
                  <td className="num">{c.cycle_m.toFixed(2)} m</td>
                  <td>{statusLabel(c.status, c.verdict)}{c.sub_step?.decision ? <div className="muted">{c.sub_step.decision}</div> : null}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
      {st.segments?.length ? (
        <div className="table-scroll">
          <table className="plain small">
            <thead><tr><th>Part of the clip</th><th>Result</th></tr></thead>
            <tbody>
              {st.segments.map((g, i) => (
                <tr key={i}>
                  <td className="num">{g.t_s ? `${g.t_s[0].toFixed(2)}–${g.t_s[1].toFixed(2)} s` : `frames ${g.frames[0]}–${g.frames[1]}`}</td>
                  <td>{g.skipped ? `not analysed: ${g.skipped}` : statusLabel(g.status, g.verdict)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
      {st.decision && <p className="small"><span className="muted">Decision: </span>{plain(st.decision)}.</p>}
    </li>
  );
}

export default function InvestigationPanel({ c }: { c: CaseFile }) {
  const inv = c.investigation;
  if (!inv) return null;
  const steps = inv.trace.filter((s) => s.step.startsWith("remedy") || s.step === "diagnose");
  const requests = inv.requests ?? [];
  if (!steps.length && !requests.length) return null;
  return (
    <section className="section">
      <div className="section-head">
        <h2>{steps.length ? "What CARAI tried after the dispute" : "What would let CARAI say more"}</h2>
        {steps.length > 0 && <span className="small muted">first result: {statusLabel(inv.base.status, inv.base.verdict)}</span>}
      </div>
      <div className="grid-even">
        {steps.length > 0 && (
          <div>
            <p className="small">
              Each dispute names a specific failure. CARAI runs the one remedy that failure allows, then the
              cross-examination runs again from scratch. A remedy is adopted only if every check then holds;
              thresholds are never relaxed.
            </p>
            <ol className="inv">{steps.map((s, i) => <Step key={i} st={s} />)}</ol>
          </div>
        )}
        {requests.length > 0 && (
          <div>
            {steps.length > 0 && <h3 style={{ marginBottom: 10 }}>What would let CARAI say more</h3>}
            <p className="small">These cannot be inferred from the footage. Supply them with the upload and the analysis uses them instead of its defaults.</p>
            <table className="plain small">
              <tbody>
                {requests.map((r, i) => (
                  <tr key={i}>
                    <td><strong>{INPUT_NAMES[r.input] ?? r.input}</strong></td>
                    <td>{r.why.charAt(0).toUpperCase() + r.why.slice(1)}.
                      {r.options?.length ? <div className="muted">Options: {r.options.map((o) => STANDARD_NAMES[o] ?? o).join("; ")}</div> : null}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </section>
  );
}
