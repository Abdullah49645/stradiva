"""
S3-triggered processing Lambda (container image, see infrastructure/docker/Dockerfile.lambda).

uploads/{job_id}/{filename}  ->  backend.investigator.investigate (the pipeline + its remedies)  ->  results/{job_id}/case.json
and the job record in DynamoDB moves PENDING -> PROCESSING -> <case status> | ERROR.

Robustness:
  * idempotent: S3 can deliver the same event more than once. The PENDING ->
    PROCESSING transition is a DynamoDB conditional write, so a duplicate event
    for a job that is already processing or finished is ignored.
  * never stuck: any exception lands the job in ERROR with the message
    (status attribute names go through ExpressionAttributeNames -- "error" and
    "status" are DynamoDB reserved words).
  * bounded: the pipeline analyses at most MAX_SECONDS of video, downscaled to
    <= 1280 px wide, so memory stays within the function's allocation.
"""
import json
import os
import sys
import time
import urllib.parse

import boto3
from botocore.exceptions import ClientError

sys.path.insert(0, "/opt/carai")  # repo root inside the container image
from backend.pipeline.analyze import analyze_video  # noqa: E402
from backend.investigator.investigator import investigate  # noqa: E402

S3 = boto3.client("s3")
DDB = boto3.resource("dynamodb")
JOBS_TABLE = os.environ.get("JOBS_TABLE", "carai-jobs")
MAX_SECONDS = float(os.environ.get("CARAI_MAX_SECONDS", "15"))
USE_INVESTIGATOR = os.environ.get("CARAI_INVESTIGATOR", "1") != "0"
RESULTS_PREFIX = "results"
TMP_DIR = "/tmp"


def _update(job_id, status, extra=None, condition_pending=False):
    table = DDB.Table(JOBS_TABLE)
    expr = "SET #s = :status, updated_at = :ts"
    names = {"#s": "status"}
    values = {":status": status, ":ts": int(time.time())}
    for k, v in (extra or {}).items():
        names[f"#{k}"] = k
        values[f":{k}"] = v
        expr += f", #{k} = :{k}"
    kw = {}
    if condition_pending:
        kw["ConditionExpression"] = "attribute_not_exists(#s) OR #s = :pending"
        values[":pending"] = "PENDING"
    table.update_item(Key={"job_id": job_id}, UpdateExpression=expr, ExpressionAttributeNames=names,
                      ExpressionAttributeValues=values, **kw)


def handler(event, context):
    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])
        parts = key.split("/")
        if len(parts) < 3 or parts[0] != "uploads":
            continue
        job_id, filename = parts[1], os.path.basename(parts[-1])
        try:
            _update(job_id, "PROCESSING", condition_pending=True)
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                continue  # duplicate delivery: this job is already being processed or is finished
            raise
        local = os.path.join(TMP_DIR, f"{job_id}_{filename}")
        try:
            S3.download_file(bucket, key, local)
            item = DDB.Table(JOBS_TABLE).get_item(Key={"job_id": job_id}).get("Item") or {}
            inputs = json.loads(item.get("inputs") or "{}")
            if USE_INVESTIGATOR:
                case = investigate(local, seed=0, user_inputs=inputs, max_seconds=MAX_SECONDS)
            else:
                case = analyze_video(local, max_seconds=MAX_SECONDS, fps_override=inputs.get("fps"),
                                     focal_px=inputs.get("focal_px"))
            case["job_id"] = job_id
            result_key = f"{RESULTS_PREFIX}/{job_id}/case.json"
            S3.put_object(Bucket=bucket, Key=result_key, Body=json.dumps(case).encode("utf-8"),
                          ContentType="application/json")
            _update(job_id, case.get("status", "DONE"), {"result_key": result_key,
                                                         "pipeline_s": str(case.get("timings_s", {}).get("total", ""))})
        except Exception as exc:  # noqa: BLE001 -- the job must never be left at PROCESSING
            _update(job_id, "ERROR", {"error": f"{type(exc).__name__}: {exc}"[:1000]})
        finally:
            if os.path.exists(local):
                os.remove(local)
    return {"statusCode": 200}
