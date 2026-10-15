"""
S3-triggered Lambda: fires when a video lands under uploads/{job_id}/...,
runs the full CARAI pipeline, writes result JSON + markdown report to
results/{job_id}/, and updates the job's DynamoDB status record.

Packaged as a container-image Lambda (see Dockerfile.lambda) since
OpenCV 5 doesn't fit comfortably in a zip Lambda layer at a reasonable
size. Not deployed/tested against live AWS in this sandbox -- see
docs/aws_and_production_notes.md for exactly what has and hasn't been
verified.
"""
import json
import os
import sys
import time
import urllib.parse

import boto3

sys.path.insert(0, "/opt/carai")  # repo root, baked into the container image
from backend.api.process_video import process_video  # noqa: E402

S3 = boto3.client("s3")
DDB = boto3.resource("dynamodb")
JOBS_TABLE = os.environ.get("JOBS_TABLE", "carai-jobs")
RESULTS_PREFIX = "results"
TMP_DIR = "/tmp"


def _update_job_status(job_id, status, extra=None):
    """NOTE: every key coming through `extra` MUST go through
    ExpressionAttributeNames (#alias), not straight into the expression
    string. DynamoDB reserves a long list of words as keywords -- "error"
    among them -- so `extra={"error": ...}` (exactly the call this
    function's own except-block makes) previously threw a
    ValidationException from *inside* the error handler itself, leaving
    the job silently stuck at PROCESSING forever with no recorded error.
    Caught by tests/test_aws_handlers.py's missing-object test, not by
    inspection -- moto actually enforces DynamoDB's reserved-word rules,
    a real service does too, but a hand-read of this code doesn't."""
    table = DDB.Table(JOBS_TABLE)
    expr = "SET #s = :status, updated_at = :ts"
    names = {"#s": "status"}
    values = {":status": status, ":ts": int(time.time())}
    if extra:
        for k, v in extra.items():
            alias = f"#{k}"
            expr += f", {alias} = :{k}"
            names[alias] = k
            values[f":{k}"] = v
    table.update_item(Key={"job_id": job_id}, UpdateExpression=expr,
                       ExpressionAttributeNames=names, ExpressionAttributeValues=values)


def handler(event, context):
    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])
        # expected key shape: uploads/{job_id}/{filename}
        parts = key.split("/")
        if len(parts) < 3 or parts[0] != "uploads":
            continue
        job_id = parts[1]
        filename = parts[-1]
        local_path = os.path.join(TMP_DIR, filename)

        try:
            _update_job_status(job_id, "PROCESSING")
            S3.download_file(bucket, key, local_path)

            result, report_md = process_video(local_path, n_vehicles_hint=2, scenario_name=job_id)

            result_key = f"{RESULTS_PREFIX}/{job_id}/result.json"
            report_key = f"{RESULTS_PREFIX}/{job_id}/report.md"
            S3.put_object(Bucket=bucket, Key=result_key,
                           Body=json.dumps(result, indent=2, default=str).encode("utf-8"),
                           ContentType="application/json")
            S3.put_object(Bucket=bucket, Key=report_key,
                           Body=report_md.encode("utf-8"), ContentType="text/markdown")

            _update_job_status(job_id, result.get("status", "DONE"), extra={
                "result_key": result_key, "report_key": report_key,
            })
        except Exception as exc:  # noqa: BLE001 -- top-level handler, must not crash silently
            _update_job_status(job_id, "ERROR", extra={"error": str(exc)})
            raise
        finally:
            if os.path.exists(local_path):
                os.remove(local_path)

    return {"statusCode": 200}
