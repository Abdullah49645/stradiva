# AWS Deployment Runbook

Exact commands for whoever has AWS credentials and can actually deploy —
this sandbox cannot (see `docs/aws_and_production_notes.md`: no AWS
network egress in this environment). Everything up to "what this
sandbox verified" below has been checked as far as possible without
live AWS; everything after needs a real account.

## Prerequisites

- AWS CLI configured with credentials that can create S3/DynamoDB/Lambda/
  API Gateway/IAM resources.
- AWS SAM CLI (`sam --version`).
- Docker running locally (SAM builds the container image with it).

## Deploy

```bash
cd infrastructure/aws
sam build
sam deploy --guided --resolve-image-repos
```

`sam build` builds `infrastructure/docker/Dockerfile.lambda` using the
repo root as build context (see the `Metadata` block on
`ProcessingFunction` in `template.yaml` — this is what lets `sam build`
do this automatically instead of requiring a manual
`docker build && docker push` to a hand-created ECR repo first).

`--guided` walks through stack name/region/confirmation prompts and
saves them to `samconfig.toml` for future `sam deploy` runs.
`--resolve-image-repos` tells SAM to create and manage the ECR
repository for the container image itself, rather than requiring one to
already exist.

On success, the `ApiEndpoint` output is the base URL for the two routes:

```bash
# create a job, get a presigned upload URL
curl -X POST "$API_ENDPOINT/jobs" -d '{"filename": "clip.mp4"}'
# -> {"job_id": "...", "upload_url": "https://...", "expires_in_s": 900}

# upload the actual video to that presigned URL
curl -X PUT "$UPLOAD_URL" --data-binary @clip.mp4

# poll for the result (the S3 upload triggers ProcessingFunction automatically)
curl "$API_ENDPOINT/jobs/$JOB_ID"
# -> {"status": "PENDING"} then {"status": "PROCESSING"} then
#    {"status": "RECONSTRUCTABLE", "result": {...}, "report_markdown": "..."}
```

## What this sandbox verified (no live AWS needed)

- `cfn-lint` on `template.yaml`: clean, zero findings.
- `tests/test_aws_handlers.py`: both Lambda handlers run against
  **moto** (in-memory mocked AWS) exercising the actual handler code —
  job creation, presigned URLs, the full S3-upload → Lambda-trigger →
  process_video() → S3-write → DynamoDB-update → API-read round trip,
  and two distinct failure paths (corrupt-but-present file →
  `NOT_RECONSTRUCTABLE`; missing S3 object → `ERROR`). **Found and fixed
  a real bug this way**: `error` is a DynamoDB reserved keyword, so the
  error-handling code path itself threw a `ValidationException` when it
  tried to record an error — meaning any real production failure would
  have silently stuck the job at `PROCESSING` forever with no error
  message. Fixed in `lambda_handler.py`'s `_update_job_status` (every
  dynamic attribute name now goes through `ExpressionAttributeNames`,
  not just the hardcoded `status`).

## What still needs a real account

- The container image actually building and running under Lambda's real
  `python3.12` image runtime (dependency resolution, `mesa-libGL`
  availability, cold-start time/memory in practice).
- Real IAM permission boundaries — moto doesn't enforce IAM, so
  `S3CrudPolicy`/`DynamoDBCrudPolicy` scoping is only verified by
  `cfn-lint`'s template validity, not by an actual permission-denied
  test.
- The real S3 event trigger and API Gateway request/response mapping
  end to end (moto approximates these; live AWS is authoritative).
- Cost/latency at realistic video sizes — everything tested here uses
  the ~5-10s synthetic clips already in the repo.
