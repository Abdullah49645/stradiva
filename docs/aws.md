# AWS delivery

**Status, stated plainly: the AWS stack has NOT been deployed from this
repository's development environment** (no AWS credentials were available
there). What has been verified is listed under "What is verified". A live
endpoint for judging needs the runbook below to be executed with an AWS
account; until then, the demo runs against the local server, which
implements the identical job contract.

## What the template deploys (`infrastructure/aws/template.yaml`)

| Resource | Purpose | Settings and why |
|---|---|---|
| `CaraiBucket` (S3) | uploads and results | Public access blocked, SSE-S3, CORS PUT for the UI origin only, lifecycle: `uploads/` deleted after 7 days, `results/` after 30 (accident footage is sensitive; keep it no longer than needed) |
| `JobsTable` (DynamoDB) | job status | On-demand billing; TTL on `expires_at` (30 days) |
| `ApiFunction` (Lambda, zip, Python 3.12, 512 MB) | `POST /jobs`, `GET /jobs/{id}`, `GET /health` | Issues a presigned PUT URL (15 min) for `uploads/{job}/{file}`; validates the file extension and the optional user inputs (`fps`, `focal_px`, `marking_standard`), stored with the job |
| `CaraiApi` (API Gateway) | HTTPS front door | CORS for the UI origin; throttled to 5 req/s, burst 10 |
| `ProcessingFunction` (Lambda, container image) | runs the investigator (`investigate`: the pipeline, then a remedy if the cross-examination blocks the result) with the job's user inputs; the image includes the YOLOX-S detector, fetched and SHA-256-checked at build time (`models/fetch_models.sh`) | 4096 MB, 900 s timeout, 2 GB `/tmp`, reserved concurrency 3 (cost ceiling for a public demo); S3 `ObjectCreated` on `uploads/` triggers it |

Why one Lambda and not a queue, Step Functions or ECS: one clip is one
invocation that fits comfortably in Lambda's limits (measured below), the
stages are sequential in one process, and the only fan-out is the Monte
Carlo loop, which is cheap. The job record's conditional
`PENDING -> PROCESSING` write makes duplicate S3 event deliveries harmless,
and any exception lands the job in `ERROR` with the message, so nothing is
left at `PROCESSING`.

## Sizing (measured)

Measured by `evaluation/experiments/aws_sizing.py` (wall time and peak RSS
of a fresh child process, from `rusage`; result in
`evaluation/results/aws_sizing.json`) on the development machine, 2 vCPU
x86_64, OpenCV 5.0.0.93 headless:

| Input | Wall time | Peak RSS |
|---|---|---|
| 15 s, 1280x720, 30 fps synthetic clip, pipeline + investigator (v2.2), no remedy needed | 27.5 s | 1.99 GB |
| Same clip with the camera knocked half-way: the costliest remedy (both stable parts re-analysed, frames decoded once and shared) | 66 s | 2.67 GB |

Lambda allocates CPU in proportion to memory; at 4096 MB a function gets
about 2.3 vCPUs, comparable to the machine measured. 4096 MB leaves about 1.5x headroom over
the worst measured peak, and the 15 s cap (`MaxClipSeconds`) keeps a clip well
inside the 900 s timeout.
These are development-machine measurements, not Lambda measurements.

## What is verified, and how

| Claim | Evidence | Verified against real AWS? |
|---|---|---|
| Handlers implement the job contract (create, upload trigger, idempotent processing, error paths, 404, user inputs validated and stored, API validator identical to the pipeline's) | `tests/test_aws_handlers.py`, 8 tests, run against `moto` mocks of S3 and DynamoDB, with the real pipeline | **No**: mocked AWS |
| Template is valid CloudFormation/SAM | `cfn-lint infrastructure/aws/template.yaml` passes | No (static check) |
| Dependencies install on both Lambda architectures | pinned wheels resolved for `manylinux x86_64` and `aarch64` | No image built: container registries were unreachable from the development environment |
| Pipeline memory and time fit the function | `evaluation/results/aws_sizing.json` | No: measured locally |
| Live endpoint | none | **No** |

## Runbook (to deploy)

Prerequisites: AWS account and credentials, AWS SAM CLI, Docker. The image
build downloads the detector model from media.githubusercontent.com, so the
build machine needs internet access.

```bash
cd infrastructure/aws
sam build --use-container            # builds the processing image from infrastructure/docker/Dockerfile.lambda
sam deploy --guided \
  --stack-name carai \
  --parameter-overrides AllowedOrigin=https://<your-ui-host> MaxClipSeconds=15 \
  --capabilities CAPABILITY_IAM \
  --resolve-image-repos
# note the ApiEndpoint output, then:
curl -s <ApiEndpoint>/health          # expect {"ok": true, "mode": "aws"}
```

Smoke test end to end (any short static-camera clip):

```bash
JOB=$(curl -s -X POST <ApiEndpoint>/jobs -H 'Content-Type: application/json' \
      -d '{"filename":"clip.mp4", "inputs": {"fps": 30}}')     # inputs optional: fps, focal_px, marking_standard
URL=$(echo "$JOB" | python3 -c 'import json,sys; print(json.load(sys.stdin)["upload_url"])')
ID=$(echo "$JOB" | python3 -c 'import json,sys; print(json.load(sys.stdin)["job_id"])')
curl -s -X PUT --upload-file clip.mp4 "$URL"
watch -n 5 curl -s <ApiEndpoint>/jobs/$ID
```

UI: build with `NEXT_PUBLIC_CARAI_API_BASE=<ApiEndpoint> npm run build`
and host the static export (S3 + CloudFront, or Amplify Hosting). Set
`AllowedOrigin` to that host.

Cost guardrails: reserved concurrency 3, API throttling, lifecycle
expiry. At the measured run time a clip costs well under one cent of
Lambda compute; check current pricing before relying on that.

Tear down: `sam delete --stack-name carai` (empty the bucket first).

## Responsible use built into the deployment

* Footage expires automatically (7 days uploads, 30 days results, job
  records by TTL).
* The bucket is private; uploads use short-lived presigned URLs.
* CARAI's outputs never assign fault; every result states the
  assumptions it depends on (docs/assumptions.md).
