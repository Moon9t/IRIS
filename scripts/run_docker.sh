#!/usr/bin/env bash
# Build and run the reproducible test Docker image (Linux native)
set -euo pipefail

IMAGE=iris-test:1.0
docker build -t "$IMAGE" .

docker run --rm -v "$PWD":/work -w /work --platform linux/amd64 "$IMAGE"
