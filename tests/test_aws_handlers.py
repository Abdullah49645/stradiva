"""
AWS handlers (infrastructure/aws/*.py) against MOCKED AWS (moto). This tests
the real handler code and the real pipeline end to end, in-process. It does
NOT test IAM, the container image, API Gateway mapping or S3 event wiring --
those need a live deployment (docs/aws.md).
"""
import importlib
import json
import os
import sys

import boto3
import pytest
from moto import mock_aws

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "infrastructure", "aws"))
sys.path.insert(0, ROOT)

BUCKET, TABLE, REGION = "carai-videos-test", "carai-jobs", "us-east-1"


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory):
    from evaluation.sim.world import Scenario, VehicleSpec, render_scenario
    d = tmp_path_factory.mktemp("aws")
    sc = Scenario(name="aws_sample", seed=5, duration_s=2.0, vehicles=[VehicleSpec("v", 14.0, 18.0, -0.9)])
    vp, _, _ = render_scenario(sc, str(d))
    return vp


@pytest.fixture
def aws(monkeypatch):
    for k, v in {"CARAI_BUCKET": BUCKET, "JOBS_TABLE": TABLE, "AWS_DEFAULT_REGION": REGION,
                 "AWS_ACCESS_KEY_ID": "testing", "AWS_SECRET_ACCESS_KEY": "testing", "CARAI_ALLOWED_ORIGIN": "https://ui.example"}.items():
        monkeypatch.setenv(k, v)
    with mock_aws():
        s3 = boto3.client("s3", region_name=REGION)
        s3.create_bucket(Bucket=BUCKET)
        ddb = boto3.resource("dynamodb", region_name=REGION)
        ddb.create_table(TableName=TABLE, AttributeDefinitions=[{"AttributeName": "job_id", "AttributeType": "S"}],
                         KeySchema=[{"AttributeName": "job_id", "KeyType": "HASH"}], BillingMode="PAY_PER_REQUEST")
        import api_handler
        import lambda_handler
        importlib.reload(api_handler)
        importlib.reload(lambda_handler)
        yield {"s3": s3, "ddb": ddb, "api": api_handler, "fn": lambda_handler}


def _event(key):
    return {"Records": [{"s3": {"bucket": {"name": BUCKET}, "object": {"key": key}}}]}


def _create(aws, name="clip.mp4"):
    r = aws["api"].handler({"httpMethod": "POST", "resource": "/jobs", "body": json.dumps({"filename": name})}, None)
    return r, json.loads(r["body"])


def test_create_job_record_url_and_cors(aws):
    r, body = _create(aws)
    assert r["statusCode"] == 200 and body["upload_url"].startswith("https://")
    assert r["headers"]["Access-Control-Allow-Origin"] == "https://ui.example"
    item = aws["ddb"].Table(TABLE).get_item(Key={"job_id": body["job_id"]})["Item"]
    assert item["status"] == "PENDING" and item["upload_key"] == f"uploads/{body['job_id']}/clip.mp4"
    assert int(item["expires_at"]) > int(item["created_at"])


def test_rejects_unsupported_file_type(aws):
    r, body = _create(aws, "notes.pdf")
    assert r["statusCode"] == 400 and "unsupported" in body["error"]


def test_upload_process_retrieve_and_duplicate_event(aws, sample_video):
    _, body = _create(aws)
    job, key = body["job_id"], f"uploads/{body['job_id']}/clip.mp4"
    with open(sample_video, "rb") as f:
        aws["s3"].put_object(Bucket=BUCKET, Key=key, Body=f.read())
    aws["fn"].handler(_event(key), None)
    r = aws["api"].handler({"httpMethod": "GET", "pathParameters": {"job_id": job}}, None)
    out = json.loads(r["body"])
    assert out["status"] not in ("PENDING", "PROCESSING", "ERROR"), out
    assert out["result"]["pipeline_version"].startswith("carai-")
    assert out["result"]["geometry"]["ok"] is True          # the synthetic road has measurable markings
    before = aws["ddb"].Table(TABLE).get_item(Key={"job_id": job})["Item"]["updated_at"]
    aws["fn"].handler(_event(key), None)                      # S3 may deliver the same event twice
    after = aws["ddb"].Table(TABLE).get_item(Key={"job_id": job})["Item"]
    assert after["updated_at"] == before and after["status"] == out["status"]


def test_missing_object_lands_in_error_not_stuck(aws):
    _, body = _create(aws)
    aws["fn"].handler(_event(f"uploads/{body['job_id']}/clip.mp4"), None)
    item = aws["ddb"].Table(TABLE).get_item(Key={"job_id": body["job_id"]})["Item"]
    assert item["status"] == "ERROR" and item["error"]


def test_corrupt_upload_is_invalid_input_not_error(aws):
    _, body = _create(aws)
    key = f"uploads/{body['job_id']}/clip.mp4"
    aws["s3"].put_object(Bucket=BUCKET, Key=key, Body=b"not a video")
    aws["fn"].handler(_event(key), None)
    item = aws["ddb"].Table(TABLE).get_item(Key={"job_id": body["job_id"]})["Item"]
    assert item["status"] == "INVALID_INPUT"


def test_unknown_job_404(aws):
    r = aws["api"].handler({"httpMethod": "GET", "pathParameters": {"job_id": "nope"}}, None)
    assert r["statusCode"] == 404
