#!/usr/bin/env bash
# One-command AWS deployment of the Stradiva backend (API Gateway + API Lambda, S3, DynamoDB,
# container-image processing Lambda). Run from the repository root:
#
#   bash deploy/deploy_aws.sh                                   # first deploy, any web origin allowed
#   ORIGIN=https://stradiva.vercel.app bash deploy/deploy_aws.sh   # lock the API to the website
#
# Needs: AWS CLI logged in (aws configure / aws sso login), AWS SAM CLI, Docker running.
# Optional: AWS_REGION (default us-east-1), STACK (default stradiva), ARCH (default: arm64 on an
# Apple-silicon Mac, else x86_64), MEMORY_MB (default 4096), RESERVED (default 3; 0 = none).
set -euo pipefail
cd "$(dirname "$0")/.."

REGION="${AWS_REGION:-us-east-1}"
STACK="${STACK:-stradiva}"
ORIGIN="${ORIGIN:-*}"
MEMORY_MB="${MEMORY_MB:-4096}"
RESERVED="${RESERVED:-3}"
if [ -z "${ARCH:-}" ]; then
  if [ "$(uname -m)" = "arm64" ] || [ "$(uname -m)" = "aarch64" ]; then ARCH=arm64; else ARCH=x86_64; fi
fi

say() { printf '\n== %s\n' "$*"; }
need() { command -v "$1" >/dev/null 2>&1 || { echo "Missing: $1 ($2)"; exit 1; }; }
need aws  "install: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
need sam  "install: brew install aws-sam-cli"
need docker "install Docker Desktop and start it"
docker info >/dev/null 2>&1 || { echo "Docker is installed but not running: start Docker Desktop and retry."; exit 1; }
aws sts get-caller-identity --query Account --output text >/dev/null 2>&1 || {
  echo "AWS CLI is not logged in: run 'aws configure' (or 'aws sso login') and retry."; exit 1; }
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)

say "Account $ACCOUNT, region $REGION, stack $STACK, processing arch $ARCH, memory ${MEMORY_MB} MB, allowed origin $ORIGIN"

say "Building (the processing image includes OpenCV 5 and downloads the YOLOX-S detector; first build takes a few minutes)"
( cd infrastructure/aws && sam build --parameter-overrides ProcessingArchitecture="$ARCH" )

say "Deploying"
( cd infrastructure/aws && sam deploy \
    --stack-name "$STACK" --region "$REGION" \
    --resolve-s3 --resolve-image-repos \
    --capabilities CAPABILITY_IAM \
    --no-confirm-changeset --no-fail-on-empty-changeset \
    --parameter-overrides AllowedOrigin="$ORIGIN" ProcessingArchitecture="$ARCH" \
                          ProcessingMemoryMB="$MEMORY_MB" ReservedConcurrency="$RESERVED" MaxClipSeconds=15 )

API=$(aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" \
      --query "Stacks[0].Outputs[?OutputKey=='ApiEndpoint'].OutputValue" --output text)

say "Smoke test"
curl -s "$API/health"; echo

say "Done"
echo "ApiEndpoint: $API"
echo "Set this in Vercel as NEXT_PUBLIC_STRADIVA_API_BASE, redeploy the site, then lock the API to the site:"
echo "  ORIGIN=https://<your-site>.vercel.app bash deploy/deploy_aws.sh"
