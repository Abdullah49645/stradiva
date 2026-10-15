"""
Integration test for infrastructure/aws/*.py against MOCKED AWS (moto),
not live AWS -- see docs/aws_and_production_notes.md for why this
sandbox can't do a real `sam deploy` (no AWS network egress in the
allowlist). This is the strongest verification available without real
credentials: it exercises the actual handler code (not a rewritten
stand-in) against an in-memory S3/DynamoDB that behaves like the real
services for the calls these handlers make, catching exactly the class
of bug that pure static/template review can't -- wrong parameter names,
wrong response shapes, key-format mismatches between the two handlers.

What this DOES verify:
  - api_handler.handler's POST /jobs creates a real DynamoDB item and a
    presigned S3 URL with the exact bucket/key/table names the deployed
    template would use.
  - Uploading a real video to that S3 key, then invoking
    lambda_handler.handler with a realistic S3 event, actually runs
    process_video() against it and writes result.json + report.md back
    to S3 in the exact key shape api_handler expects to read.
  - api_handler.handler's GET /jobs/{id} correctly reads that result
    back and returns it in the API response.
  - The error path: a genuinely missing upload (S3 event fired but the
    object isn't there) correctly lands the job in ERROR status with a
    message, distinctly from a present-but-corrupt file, which
    process_video()'s own `len(frames) < 5` guard already resolves to a
    clean NOT_RECONSTRUCTABLE -- this test suite checks both paths stay
    distinct rather than collapsing into one.

What this does NOT verify (needs real AWS, see the runbook):
  - IAM policy correctness (moto doesn't enforce permissions)
  - The container image actually building/running under python3.12 +
    Lambda's real runtime environment
  - Real S3 event trigger wiring, API Gateway request mapping, cold
    starts, timeouts, concurrency
"""
import importlib
import json
import os
import sys
import time

import boto3
import pytest
from moto import mock_aws

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "infrastructure", "aws"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

BUCKET = "carai-videos-123456789012"
TABLE = "carai-jobs"
REGION = "us-east-1"
SAMPLE_VIDEO = os.path.join(os.path.dirname(__file__), "..", "evaluation", "dataset", "synthetic_scene_A.mp4")


@pytest.fixture
def aws_env(monkeypatch):
    monkeypatch.setenv("CARAI_BUCKET", BUCKET)
    monkeypatch.setenv("JOBS_TABLE", TABLE)
    monkeypatch.setenv("AWS_DEFAULT_REGION", REGION)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    with mock_aws():
        s3 = boto3.client("s3", region_name=REGION)
        s3.create_bucket(Bucket=BUCKET)
        ddb = boto3.resource("dynamodb", region_name=REGION)
        ddb.create_table(
            TableName=TABLE,
            AttributeDefinitions=[{"AttributeName": "job_id", "AttributeType": "S"}],
            KeySchema=[{"AttributeName": "job_id", "KeyType": "HASH"}],
            BillingMode="PAY_PER_REQUEST",
        )
        # handlers create their boto3 clients/resources at import time, so
        # (re)import them *after* the mock and env vars are live, exactly
        # like a fresh Lambda cold start would.
        import api_handler
        import lambda_handler
        importlib.reload(api_handler)
        importlib.reload(lambda_handler)
        yield {"s3": s3, "ddb": ddb, "api_handler": api_handler, "lambda_handler": lambda_handler}


def _s3_event(bucket, key):
    return {"Records": [{"s3": {"bucket": {"name": bucket}, "object": {"key": key}}}]}


def test_create_job_writes_ddb_and_presigned_url(aws_env):
    event = {"httpMethod": "POST", "resource": "/jobs", "body": json.dumps({"filename": "clip.mp4"})}
    resp = aws_env["api_handler"].handler(event, None)
    assert resp["statusCode"] == 200
    body = json.loads(resp["body"])
    assert "job_id" in body and "upload_url" in body
    assert body["upload_url"].startswith("https://")

    item = aws_env["ddb"].Table(TABLE).get_item(Key={"job_id": body["job_id"]}).get("Item")
    assert item is not None
    assert item["status"] == "PENDING"
    assert item["upload_key"] == f"uploads/{body['job_id']}/clip.mp4"


def test_end_to_end_upload_process_retrieve(aws_env):
    if not os.path.exists(SAMPLE_VIDEO):
        pytest.skip(f"sample video not found at {SAMPLE_VIDEO} (run evaluation/scripts/run_stress_tests.py first)")

    create_resp = aws_env["api_handler"].handler(
        {"httpMethod": "POST", "resource": "/jobs", "body": json.dumps({"filename": "clip.mp4"})}, None)
    job_id = json.loads(create_resp["body"])["job_id"]
    upload_key = f"uploads/{job_id}/clip.mp4"

    # simulate the presigned-URL upload the real client would do
    with open(SAMPLE_VIDEO, "rb") as f:
        aws_env["s3"].put_object(Bucket=BUCKET, Key=upload_key, Body=f.read())

    # simulate the S3->Lambda trigger event that would fire on that PutObject
    aws_env["lambda_handler"].handler(_s3_event(BUCKET, upload_key), None)

    get_resp = aws_env["api_handler"].handler(
        {"httpMethod": "GET", "pathParameters": {"job_id": job_id}}, None)
    assert get_resp["statusCode"] == 200
    body = json.loads(get_resp["body"])
    assert body["status"] in ("RECONSTRUCTABLE", "NOT_RECONSTRUCTABLE", "INSUFFICIENT_EVIDENCE")
    assert "result" in body
    assert "geometry" in body["result"]
    assert "report_markdown" in body
    assert body["report_markdown"].startswith("#") or len(body["report_markdown"]) > 0


def test_job_not_found_returns_404(aws_env):
    resp = aws_env["api_handler"].handler(
        {"httpMethod": "GET", "pathParameters": {"job_id": "does-not-exist"}}, None)
    assert resp["statusCode"] == 404


def test_corrupt_video_lands_in_graceful_rejection_not_a_crash(aws_env, tmp_path):
    """A totally unreadable file should not crash the Lambda -- it should
    resolve to a clean NOT_RECONSTRUCTABLE status with a stated reason.
    process_video.py already guards this (`len(frames) < 5`); this test
    exists to make sure that guard keeps working, not to newly assert it."""
    create_resp = aws_env["api_handler"].handler(
        {"httpMethod": "POST", "resource": "/jobs", "body": json.dumps({"filename": "bad.mp4"})}, None)
    job_id = json.loads(create_resp["body"])["job_id"]
    upload_key = f"uploads/{job_id}/bad.mp4"
    aws_env["s3"].put_object(Bucket=BUCKET, Key=upload_key, Body=b"not a real video file")

    aws_env["lambda_handler"].handler(_s3_event(BUCKET, upload_key), None)  # must not raise

    item = aws_env["ddb"].Table(TABLE).get_item(Key={"job_id": job_id}).get("Item")
    assert item["status"] == "NOT_RECONSTRUCTABLE"
    assert "result_key" in item and "report_key" in item

    get_resp = aws_env["api_handler"].handler(
        {"httpMethod": "GET", "pathParameters": {"job_id": job_id}}, None)
    body = json.loads(get_resp["body"])
    assert body["status"] == "NOT_RECONSTRUCTABLE"
    assert "fewer than 5 frames" in str(body["result"]["reconstructability"]["reasons"])


def test_missing_video_at_key_lands_in_error_status(aws_env):
    """Unlike a corrupt-but-present file, a key that was never actually
    uploaded (S3 event fired but the object is gone/inaccessible) is a
    real infrastructure error, not a content judgment -- this SHOULD
    surface as ERROR, distinctly from NOT_RECONSTRUCTABLE."""
    create_resp = aws_env["api_handler"].handler(
        {"httpMethod": "POST", "resource": "/jobs", "body": json.dumps({"filename": "ghost.mp4"})}, None)
    job_id = json.loads(create_resp["body"])["job_id"]
    upload_key = f"uploads/{job_id}/ghost.mp4"
    # deliberately never put_object'd -- simulates a stale/incorrect event

    with pytest.raises(Exception):
        aws_env["lambda_handler"].handler(_s3_event(BUCKET, upload_key), None)

    item = aws_env["ddb"].Table(TABLE).get_item(Key={"job_id": job_id}).get("Item")
    assert item["status"] == "ERROR"
    assert "error" in item
