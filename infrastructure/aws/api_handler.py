"""
API Gateway Lambda (proxy integration). Same contract as the local server
(backend/api/server.py), so the web UI works against either:

  POST /jobs          {"filename"}  -> job record (PENDING) + presigned S3 PUT URL for uploads/{job_id}/{filename}
  GET  /jobs/{job_id}               -> {"job_id", "status", "result"?, "error"?}

Every response carries CORS headers for the configured web origin.
"""
import json
import os
import time
import uuid

import boto3

S3 = boto3.client("s3")
DDB = boto3.resource("dynamodb")
BUCKET = os.environ.get("CARAI_BUCKET", "carai-videos")
JOBS_TABLE = os.environ.get("JOBS_TABLE", "carai-jobs")
ALLOWED_ORIGIN = os.environ.get("CARAI_ALLOWED_ORIGIN", "*")
UPLOAD_URL_EXPIRY_S = 900
ALLOWED_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
RETENTION_S = 30 * 24 * 3600


def _response(code, body):
    return {"statusCode": code,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": ALLOWED_ORIGIN,
                        "Access-Control-Allow-Headers": "Content-Type", "Access-Control-Allow-Methods": "GET,POST,OPTIONS"},
            "body": json.dumps(body, default=str)}


def _create_job(event):
    try:
        body = json.loads(event.get("body") or "{}")
    except json.JSONDecodeError:
        return _response(400, {"error": "request body must be JSON"})
    filename = os.path.basename(str(body.get("filename") or "upload.mp4")).replace(" ", "_")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return _response(400, {"error": f"unsupported file type '{ext}' (accepted: {', '.join(sorted(ALLOWED_EXT))})"})
    job_id = str(uuid.uuid4())
    key = f"uploads/{job_id}/{filename}"
    url = S3.generate_presigned_url("put_object", Params={"Bucket": BUCKET, "Key": key}, ExpiresIn=UPLOAD_URL_EXPIRY_S)
    now = int(time.time())
    DDB.Table(JOBS_TABLE).put_item(Item={"job_id": job_id, "status": "PENDING", "created_at": now,
                                         "upload_key": key, "expires_at": now + RETENTION_S})
    return _response(200, {"job_id": job_id, "upload_url": url, "expires_in_s": UPLOAD_URL_EXPIRY_S})


def _get_job(job_id):
    item = DDB.Table(JOBS_TABLE).get_item(Key={"job_id": job_id}).get("Item")
    if item is None:
        return _response(404, {"error": "job not found"})
    out = {"job_id": job_id, "status": item.get("status")}
    if item.get("result_key") and item.get("status") not in ("PENDING", "PROCESSING"):
        out["result"] = json.loads(S3.get_object(Bucket=BUCKET, Key=item["result_key"])["Body"].read())
    if item.get("error"):
        out["error"] = item["error"]
    return _response(200, out)


def handler(event, context):
    method = event.get("httpMethod", "GET")
    params = event.get("pathParameters") or {}
    try:
        if method == "OPTIONS":
            return _response(204, {})
        if method == "POST" and event.get("resource", "").endswith("/jobs"):
            return _create_job(event)
        if method == "GET" and "job_id" in params:
            return _get_job(params["job_id"])
        if method == "GET" and event.get("resource", "").endswith("/health"):
            return _response(200, {"ok": True, "mode": "aws"})
        return _response(404, {"error": "route not found"})
    except Exception as exc:  # noqa: BLE001
        return _response(500, {"error": f"{type(exc).__name__}: {exc}"})
