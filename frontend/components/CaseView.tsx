"use client";

import { CaseFile } from "@/lib/types";
import { CHECK_NAMES, Units, VERDICT_TEXT, verdictKey } from "@/lib/format";
import EvidencePlayer from "./EvidencePlayer";
import EnvelopeMap from "./EnvelopeMap";
import DashcamPanel from "./DashcamPanel";
import { CrossExamination, Establishes, KeyFacts, MotionChart, Validation } from "./CasePanels";
import InvestigationPanel from "./InvestigationPanel";
import NearMissPanel from "./NearMissPanel";
import LeadPanel from "./LeadPanel";

export default function CaseView({ c, live, units }: { c: CaseFile; live: boolean; units: Units }) {
  const key = verdictKey(c);
  const v = VERDICT_TEXT[key] ?? { title: key, tone: "neutral" };
  let reason =
    c.status === "NOT_RECONSTRUCTABLE" || c.status === "INVALID_INPUT"
      ? (c.reasons ?? []).join(" ")
      : c.verdict?.reason ?? "";
  for (const [id, name] of Object.entries(CHECK_NAMES)) reason = reason.split(id).join(`“${name.toLowerCase()}”`);
  reason = reason.charAt(0).toUpperCase() + reason.slice(1);
  const prov = live ? "live" : c.provenance?.kind === "real" ? "real" : "";
  const provText = live
    ? "Analysed just now by the CARAI service"
    : c.provenance?.kind === "real"
      ? `Real footage, precomputed: ${c.provenance.source}`
      : "Synthetic footage with known ground truth, precomputed by the same pipeline";

  return (
    <article>
      <section className="section grid-2">
        <div>
          <span className={`provenance ${prov}`}><span className="dot" />{provText}</span>
          {c.provenance && !live && <p className="small muted" style={{ marginBottom: 0 }}>{c.provenance.summary}</p>}
          <h1 className={`verdict-title tone-${v.tone}`}><span className="mark" aria-hidden="true" />{v.title}</h1>
          <p className="reason">{reason}</p>
          <KeyFacts c={c} units={units} />
        </div>
        <div>
          {c.video ? <EvidencePlayer c={c} units={units} /> : (
            <div className="sheet small muted">This upload was analysed without keeping the video; the overlay is only available for bundled cases.</div>
          )}
        </div>
      </section>

      <InvestigationPanel c={c} />

      {c.dashcam && <DashcamPanel c={c} units={units} />}
      {c.dashcam_lead && <LeadPanel c={c} />}

      {c.counterfactual && (
        <section className="section">
          <div className="section-head">
            <h2>Which assumption decides the answer</h2>
            <span className="small muted">{c.counterfactual.n_samples} reconstructions consistent with the video</span>
          </div>
          <div className="grid-2">
            <p className="small">
              The video cannot show how fast the following driver could react or how hard the tyres could brake. CARAI
              therefore replays the approach for every combination below. The follower moves as reconstructed until it
              reacts, then brakes. A verdict is stated only when it holds inside the whole outlined box.
            </p>
            <EnvelopeMap cf={c.counterfactual} />
          </div>
        </section>
      )}

      {!c.dashcam && <section className="section grid-even">
        <div>
          <div className="section-head"><h2>What this video establishes</h2></div>
          <Establishes c={c} />
        </div>
        <div>
          <div className="section-head"><h2>Cross-examination</h2></div>
          {c.skeptic ? (
            <>
              <p className="small">
                After the reconstruction is built, a separate set of checks tries to contradict it, using evidence the
                reconstruction was not fitted to. A dispute on any check that affects speed or timing withholds the verdict.
              </p>
              <CrossExamination c={c} />
            </>
          ) : <p className="small muted">Not reached: the footage was rejected before reconstruction.</p>}
        </div>
      </section>}

      {c.status !== "DISPUTED" && <NearMissPanel c={c} />}

      {c.vehicles?.length ? (
        <section className="section">
          <div className="section-head"><h2>Reconstructed speeds</h2>
            {c.time_base && <span className="small muted">{c.time_base}</span>}</div>
          <MotionChart c={c} units={units} />
        </section>
      ) : null}

      {c.validation && !live && (
        <section className="section">
          <div className="section-head"><h2>Checked against ground truth</h2></div>
          <Validation c={c} />
        </section>
      )}

      {c.stages?.length ? (
        <section className="section">
          <div className="section-head"><h2>Processing record</h2>
            <span className="small muted">pipeline {c.pipeline_version} · input {c.input?.file}</span></div>
          <div className="table-scroll">
            <table className="plain small">
              <tbody>
                {c.stages.map((s, i) => (
                  <tr key={i}><td>{s.stage}</td><td>{s.status}</td><td>{s.detail}</td><td className="num">{s.t_s.toFixed(1)} s</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}
    </article>
  );
}
