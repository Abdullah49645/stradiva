"""
API Gateway Lambda (proxy integration): the small, user-facing API.

  POST /jobs          -> create a job record (PENDING), return a
                          presigned S3 upload URL scoped to this job's key
  GET  /jobs/{job_id}  -> return job status, and once DONE, the result
                          JSON + report markdown (fetched from S3)

Kept deliberately thin -- all real work happens in lambda_handler.py,
triggered by the S3 upload this API hands out a URL for. Not
deployed/tested against live AWS in this sandbox; see
docs/aws_and_production_notes.md.
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
UPLOAD_URL_EXPIRY_S = 900


def _response(status_code, body):
    return {"statusCode": status_code, "headers": {"Content-Type": "application/json"},
            "body": json.dumps(body, default=str)}


def _create_job(event):
    body = json.loads(event.get("body") or "{}")
    filename = body.get("filename", "upload.mp4")
    job_id = str(uuid.uuid4())
    key = f"uploads/{job_id}/{filename}"

    upload_url = S3.generate_presigned_url(
        "put_object", Params={"Bucket": BUCKET, "Key": key},
        ExpiresIn=UPLOAD_URL_EXPIRY_S,
    )

    table = DDB.Table(JOBS_TABLE)
    table.put_item(Item={
        "job_id": job_id, "status": "PENDING", "created_at": int(time.time()),
        "upload_key": key,
    })

    return _response(200, {"job_id": job_id, "upload_url": upload_url, "expires_in_s": UPLOAD_URL_EXPIRY_S})


def _get_job(event, job_id):
    table = DDB.Table(JOBS_TABLE)
    item = table.get_item(Key={"job_id": job_id}).get("Item")
    if item is None:
        return _response(404, {"error": "job not found"})

    out = {"job_id": job_id, "status": item.get("status")}
    if item.get("status") not in ("PENDING", "PROCESSING") and item.get("result_key"):
        result_obj = S3.get_object(Bucket=BUCKET, Key=item["result_key"])
        out["result"] = json.loads(result_obj["Body"].read())
    if item.get("report_key"):
        report_obj = S3.get_object(Bucket=BUCKET, Key=item["report_key"])
        out["report_markdown"] = report_obj["Body"].read().decode("utf-8")
    if item.get("error"):
        out["error"] = item["error"]
    return _response(200, out)


def handler(event, context):
    method = event.get("httpMethod", "GET")
    path_params = event.get("pathParameters") or {}

    try:
        if method == "POST" and event.get("resource", "").endswith("/jobs"):
            return _create_job(event)
        if method == "GET" and "job_id" in path_params:
            return _get_job(event, path_params["job_id"])
        return _response(404, {"error": "route not found"})
    except Exception as exc:  # noqa: BLE001
        return _response(500, {"error": str(exc)})
