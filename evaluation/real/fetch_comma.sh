#!/usr/bin/env bash
# Fetches the comma.ai speed-challenge training video + labels for EVALUATION ONLY.
# The repository has no license file; CARAI does not redistribute the data.
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)/data/comma"
mkdir -p "$D"
curl -L -o "$D/train.mp4" https://media.githubusercontent.com/media/commaai/speedchallenge/master/data/train.mp4
curl -L -o "$D/train.txt" https://raw.githubusercontent.com/commaai/speedchallenge/master/data/train.txt
echo "a9f16228de735cbe65946093287bc32e317bea52d0738b81340bcd28f72ae06f  $D/train.mp4" | sha256sum -c -
