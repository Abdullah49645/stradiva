"""
Local CARAI API -- the same job contract as the AWS deployment
(infrastructure/aws/api_handler.py + lambda_handler.py), so the web UI talks
to either without code changes:

    POST /jobs                 {"filename": "..."}  ->  {"job_id", "upload_url", "expires_in_s"}
    PUT  <upload_url>          raw video bytes       ->  204   (processing starts)
    GET  /jobs/{job_id}                              ->  {"job_id", "status", "stage", "result"?, "error"?}
    GET  /health                                     ->  {"ok": true, "pipeline_version": ...}

Status: PENDING -> PROCESSING -> <case status> (e.g. RECONSTRUCTED_COLLISION,
NOT_RECONSTRUCTABLE, DISPUTED) or ERROR. Exactly what the Lambda writes to
DynamoDB.

Run:  uvicorn backend.api.server:app --port 8000      (from the repo root)
      then start the UI with NEXT_PUBLIC_CARAI_API_BASE=http://localhost:8000
"""
from __future__ import annotations

import os
import tempfile
import threading
import time
import traceback
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from backend.pipeline.analyze import analyze_video, PIPELINE_VERSION

MAX_UPLOAD_BYTES = int(os.environ.get("CARAI_MAX_UPLOAD_MB", "200")) * 1024 * 1024
ALLOWED_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
WORK_DIR = os.environ.get("CARAI_WORK_DIR", os.path.join(tempfile.gettempdir(), "carai_jobs"))
os.makedirs(WORK_DIR, exist_ok=True)

app = FastAPI(title="CARAI local API", version=PIPELINE_VERSION)
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("CARAI_CORS_ORIGINS", "http://localhost:3000").split(","),
                   allow_methods=["GET", "POST", "PUT", "OPTIONS"], allow_headers=["*"])

JOBS: dict = {}
LOCK = threading.Lock()
WORKER_SLOTS = threading.Semaphore(int(os.environ.get("CARAI_WORKERS", "1")))


def _set(job_id, **kw):
    with LOCK:
        JOBS[job_id].update(kw, updated_at=time.time())


def _process(job_id, path):
    with WORKER_SLOTS:
        _set(job_id, status="PROCESSING", stage="analysing")
        try:
            case = analyze_video(path, seed=0)
            _set(job_id, status=case.get("status", "DONE"), stage="done", result=case)
        except Exception as exc:  # noqa: BLE001 -- report, never hang at PROCESSING
            _set(job_id, status="ERROR", stage="failed", error=f"{type(exc).__name__}: {exc}",
                 trace=traceback.format_exc()[-2000:])
        finally:
            try:
                os.remove(path)
            except OSError:
                pass


@app.get("/health")
def health():
    return {"ok": True, "pipeline_version": PIPELINE_VERSION, "mode": "local"}


@app.post("/jobs")
async def create_job(request: Request):
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    filename = os.path.basename(str(body.get("filename") or "upload.mp4"))
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"unsupported file type '{ext}' (accepted: {', '.join(sorted(ALLOWED_EXT))})")
    job_id = str(uuid.uuid4())
    with LOCK:
        JOBS[job_id] = {"job_id": job_id, "status": "PENDING", "stage": "awaiting upload", "filename": filename,
                        "created_at": time.time()}
    base = str(request.base_url).rstrip("/")
    return {"job_id": job_id, "upload_url": f"{base}/jobs/{job_id}/upload", "expires_in_s": 900}


@app.put("/jobs/{job_id}/upload")
async def upload(job_id: str, request: Request):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job["status"] != "PENDING":
        raise HTTPException(409, f"job already {job['status']} (uploads are accepted once)")
    data = await request.body()
    if not data:
        raise HTTPException(400, "empty upload")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"file larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    path = os.path.join(WORK_DIR, f"{job_id}{os.path.splitext(job['filename'])[1].lower()}")
    with open(path, "wb") as f:
        f.write(data)
    _set(job_id, status="PENDING", stage="queued")
    threading.Thread(target=_process, args=(job_id, path), daemon=True).start()
    return Response(status_code=204)


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return {k: v for k, v in job.items() if k != "trace"}
