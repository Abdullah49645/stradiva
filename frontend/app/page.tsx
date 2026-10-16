"use client";

import { useEffect, useState } from "react";
import CaseView from "@/components/CaseView";
import UploadPanel from "@/components/UploadPanel";
import { RealFootageView, TestingView } from "@/components/EvidenceViews";
import { CaseFile } from "@/lib/types";
import { Units } from "@/lib/format";

type View = "cases" | "real" | "testing";
type IndexEntry = { id: string; title: string; summary: string };

export default function Home() {
  const [view, setView] = useState<View>("cases");
  const [index, setIndex] = useState<IndexEntry[] | null>(null);
  const [indexError, setIndexError] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);
  const [caseFile, setCaseFile] = useState<CaseFile | null>(null);
  const [live, setLive] = useState(false);
  const [units, setUnits] = useState<Units>("metric");

  useEffect(() => {
    fetch("/cases/index.json")
      .then((r) => r.json())
      .then((d: IndexEntry[]) => { setIndex(d); if (d.length) setSelected(d[0].id); })
      .catch(() => setIndexError(true));
  }, []);

  useEffect(() => {
    if (!selected || live) return;
    setCaseFile(null);
    fetch(`/cases/${selected}.json`).then((r) => r.json()).then(setCaseFile).catch(() => setCaseFile(null));
  }, [selected, live]);

  return (
    <>
      <header className="mast">
        <div className="wrap mast-row">
          <div className="brand">Odolite<small>evidence from traffic video</small></div>
          <nav className="tabs" role="tablist" aria-label="Views">
            {([["cases", "Case files"], ["real", "Real dashcam footage"], ["testing", "How it was tested"]] as [View, string][]).map(([k, label]) => (
              <button key={k} role="tab" aria-selected={view === k} className="tab" onClick={() => setView(k)}>{label}</button>
            ))}
          </nav>
          <div className="seg" role="group" aria-label="Units">
            <button aria-pressed={units === "metric"} onClick={() => setUnits("metric")}>km/h</button>
            <button aria-pressed={units === "imperial"} onClick={() => setUnits("imperial")}>mph</button>
          </div>
        </div>
      </header>

      <main className="wrap">
        {view === "cases" && (
          <>
            <div className="picker" role="group" aria-label="Case files">
              {indexError && <p className="err">The bundled case files are missing. Run <code>python3 evaluation/demo/build_cases.py</code>.</p>}
              {index?.map((e) => (
                <button key={e.id} className="pick" aria-pressed={!live && selected === e.id}
                        onClick={() => { setLive(false); setSelected(e.id); }}>
                  <strong>{e.title}</strong>{e.summary}
                </button>
              ))}
              {live && <button className="pick" aria-pressed={true}><strong>Your upload</strong>analysed just now</button>}
            </div>
            {caseFile ? <CaseView c={caseFile} live={live} units={units} /> : !indexError && <p className="section muted">Loading case…</p>}
            <section className="section">
              <UploadPanel onResult={(c) => { setLive(true); setCaseFile(c); window.scrollTo({ top: 0, behavior: "smooth" }); }} />
            </section>
          </>
        )}
        {view === "real" && <RealFootageView />}
        {view === "testing" && <TestingView />}
        <footer className="foot">
          Odolite states physical evidence and the assumptions it rests on. It does not determine fault, and its output is a
          starting point for expert review, not a substitute for it.
        </footer>
      </main>
    </>
  );
}
