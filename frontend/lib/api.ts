import { JobStatusResponse } from "./types";

// One contract, two back ends: the local FastAPI server (backend/api/server.py)
// or the AWS API Gateway stage (infrastructure/aws). Unset = no analysis
// service; the UI then only shows the bundled, precomputed case files.
const API_BASE = (process.env.NEXT_PUBLIC_STRADIVA_API_BASE || "").replace(/\/$/, "");

export function apiBase(): string {
  return API_BASE;
}

export async function apiHealth(): Promise<{ ok: boolean; mode?: string; pipeline_version?: string } | null> {
  if (!API_BASE) return null;
  try {
    const res = await fetch(`${API_BASE}/health`, { cache: "no-store" });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

export type JobInputs = { fps?: number; focal_px?: number; marking_standard?: string };

export async function createJob(filename: string, inputs: JobInputs = {}): Promise<{ job_id: string; upload_url: string }> {
  const res = await fetch(`${API_BASE}/jobs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename, inputs }),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || detail.error || `the service refused the job (HTTP ${res.status})`);
  }
  return res.json();
}

export async function uploadFile(uploadUrl: string, file: File): Promise<void> {
  const res = await fetch(uploadUrl, { method: "PUT", body: file, headers: { "Content-Type": file.type || "video/mp4" } });
  if (!res.ok) throw new Error(`upload failed (HTTP ${res.status})`);
}

export async function getJob(jobId: string): Promise<JobStatusResponse> {
  const res = await fetch(`${API_BASE}/jobs/${jobId}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`could not read job status (HTTP ${res.status})`);
  return res.json();
}

export async function pollJob(jobId: string, onUpdate: (s: JobStatusResponse) => void, intervalMs = 1500,
                              timeoutMs = 15 * 60 * 1000): Promise<JobStatusResponse> {
  const t0 = Date.now();
  for (;;) {
    const s = await getJob(jobId);
    onUpdate(s);
    if (!["PENDING", "PROCESSING"].includes(s.status)) return s;
    if (Date.now() - t0 > timeoutMs) throw new Error("analysis did not finish within 15 minutes");
    await new Promise((r) => setTimeout(r, intervalMs));
  }
}
