import { JobStatusResponse } from "./types";

// Points at the deployed API Gateway stage (see infrastructure/aws/template.yaml
// Outputs.ApiEndpoint). Unset in this environment since nothing has been
// deployed yet -- see docs/aws_and_production_notes.md. The UI falls back to
// bundled demo results (public/demo-results.json) when this isn't configured.
const API_BASE = process.env.NEXT_PUBLIC_CARAI_API_BASE || "";

export function isLiveApiConfigured(): boolean {
  return API_BASE.length > 0;
}

export async function createJob(filename: string): Promise<{ job_id: string; upload_url: string }> {
  const res = await fetch(`${API_BASE}/jobs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename }),
  });
  if (!res.ok) throw new Error(`createJob failed: ${res.status}`);
  return res.json();
}

export async function uploadToPresignedUrl(uploadUrl: string, file: File): Promise<void> {
  const res = await fetch(uploadUrl, { method: "PUT", body: file });
  if (!res.ok) throw new Error(`upload failed: ${res.status}`);
}

export async function getJobStatus(jobId: string): Promise<JobStatusResponse> {
  const res = await fetch(`${API_BASE}/jobs/${jobId}`);
  if (!res.ok) throw new Error(`getJobStatus failed: ${res.status}`);
  return res.json();
}

export async function pollJobUntilDone(
  jobId: string,
  onUpdate: (status: JobStatusResponse) => void,
  intervalMs = 2000,
  maxAttempts = 150
): Promise<JobStatusResponse> {
  for (let i = 0; i < maxAttempts; i++) {
    const status = await getJobStatus(jobId);
    onUpdate(status);
    if (!["PENDING", "PROCESSING"].includes(status.status)) {
      return status;
    }
    await new Promise((r) => setTimeout(r, intervalMs));
  }
  throw new Error("polling timed out");
}
