"use client";

import { useEffect, useState } from "react";
import { apiBase, apiHealth, createJob, JobInputs, pollJob, uploadFile } from "@/lib/api";
import { CaseFile } from "@/lib/types";
import { STANDARD_NAMES } from "@/lib/format";

const MAX_MB = 200;

export default function UploadPanel({ onResult }: { onResult: (c: CaseFile) => void }) {
  const [health, setHealth] = useState<{ ok: boolean; mode?: string } | null | undefined>(undefined);
  const [file, setFile] = useState<File | null>(null);
  const [state, setState] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [fps, setFps] = useState("");
  const [focal, setFocal] = useState("");
  const [standard, setStandard] = useState("");

  useEffect(() => { apiHealth().then(setHealth); }, []);

  async function analyse() {
    if (!file) return;
    setError(null);
    if (file.size > MAX_MB * 1024 * 1024) { setError(`The file is larger than ${MAX_MB} MB. Trim the clip to the event and try again.`); return; }
    setBusy(true);
    try {
      setState("Creating job");
      const inputs: JobInputs = {};
      if (fps.trim()) inputs.fps = Number(fps);
      if (focal.trim()) inputs.focal_px = Number(focal);
      if (standard) inputs.marking_standard = standard;
      const { job_id, upload_url } = await createJob(file.name, inputs);
      setState("Uploading");
      await uploadFile(upload_url, file);
      setState("Waiting for the analysis to start");
      const final = await pollJob(job_id, (s) => setState(s.status === "PROCESSING" ? "Analysing (typically 20–60 s)" : s.stage ?? s.status));
      if (final.status === "ERROR") throw new Error(final.error || "the analysis failed");
      if (!final.result) throw new Error("the service finished without returning a result");
      setState(null);
      onResult(final.result);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setState(null);
    } finally {
      setBusy(false);
    }
  }

  if (!apiBase()) {
    return (
      <div className="drop small">
        <strong>Analyse your own clip</strong>
        <p style={{ margin: "6px 0 0" }}>No analysis service is configured for this page, so it shows the bundled cases only.
          Run <code>uvicorn backend.api.server:app</code> and start the page with <code>NEXT_PUBLIC_STRADIVA_API_BASE=http://localhost:8000</code> to analyse uploads.</p>
      </div>
    );
  }
  return (
    <div className="drop">
      <strong>Analyse your own clip</strong>
      <p className="small" style={{ margin: "6px 0 10px" }}>
        A fixed camera looking along a straight road with a dashed centre line works best. Clips are cut to the first 15 s.
        {health === null && <span className="err"> The analysis service at {apiBase()} is not responding.</span>}
        {health?.mode && <span className="muted"> Service: {health.mode}.</span>}
      </p>
      <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
        <input type="file" accept="video/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} disabled={busy} aria-label="Video file" />
        <button className="btn" onClick={analyse} disabled={!file || busy || health === null}>{busy ? "Analysing…" : "Analyse clip"}</button>
      </div>
      <details style={{ marginTop: 12 }}>
        <summary className="small">What you know about the camera (optional)</summary>
        <div className="inputs">
          <label>Frame rate (fps)
            <input inputMode="decimal" value={fps} onChange={(e) => setFps(e.target.value)} placeholder="from the file" disabled={busy} />
          </label>
          <label>Focal length (pixels)
            <input inputMode="decimal" value={focal} onChange={(e) => setFocal(e.target.value)} placeholder="unknown" disabled={busy} />
          </label>
          <label>Marking standard
            <select value={standard} onChange={(e) => setStandard(e.target.value)} disabled={busy}>
              <option value="">default (Stradiva test roads)</option>
              {Object.entries(STANDARD_NAMES).map(([k, n]) => <option key={k} value={k}>{n}</option>)}
            </select>
          </label>
        </div>
        <p className="small muted" style={{ margin: 0 }}>Every speed scales with the frame rate and the marking standard; the focal length lets a
          dashcam clip be checked against the lane width and the vehicle ahead be measured.</p>
      </details>
      {state && <p className="small" aria-live="polite">{state}</p>}
      {error && <p className="small err" role="alert">{error}</p>}
    </div>
  );
}
