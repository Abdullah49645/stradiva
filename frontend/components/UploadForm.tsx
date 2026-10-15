"use client";

import { useState } from "react";
import { createJob, uploadToPresignedUrl, isLiveApiConfigured } from "@/lib/api";
import Eyebrow from "./Eyebrow";
import Panel from "./Panel";

export default function UploadForm({ onJobCreated }: { onJobCreated: (jobId: string) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const liveApi = isLiveApiConfigured();

  async function handleSubmit() {
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      const { job_id, upload_url } = await createJob(file.name);
      await uploadToPresignedUrl(upload_url, file);
      onJobCreated(job_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setUploading(false);
    }
  }

  return (
    <Panel>
      <Eyebrow>Submit Footage</Eyebrow>
      {!liveApi && (
        <p className="text-[13px] text-signal/90 mt-3 mb-3 leading-relaxed">
          No live endpoint configured — nothing is deployed to AWS in this environment.
          Use a case file from the log on the right instead.
        </p>
      )}
      <input
        type="file"
        accept="video/*"
        disabled={!liveApi}
        onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        className="block w-full text-sm text-muted mt-4 mb-4 disabled:opacity-30
                   file:mr-3 file:py-1.5 file:px-3 file:rounded file:border file:border-hairline
                   file:bg-graphite2 file:text-paper file:text-xs file:font-mono file:uppercase file:tracking-wide
                   file:cursor-pointer"
      />
      <button
        onClick={handleSubmit}
        disabled={!liveApi || !file || uploading}
        className="w-full py-2.5 rounded font-mono text-xs uppercase tracking-wider
                   bg-hud text-asphalt hover:brightness-110 disabled:bg-graphite2 disabled:text-muted
                   disabled:cursor-not-allowed transition-all"
      >
        {uploading ? "Uploading…" : "Analyze Footage"}
      </button>
      {error && <p className="text-hazard text-sm mt-2">{error}</p>}
    </Panel>
  );
}
