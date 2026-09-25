#!/usr/bin/env bash
# CPU-only oracle, deliberately outside the broker's device phase.
set -euo pipefail
if [[ $# != 2 ]]; then
  echo "usage: $0 <isolated-build-dir> <completed-output-dir>" >&2
  exit 2
fi
if [[ -n ${GPUQ_JOB_ID:-} ]]; then
  echo "CPU oracle must run after the broker lease ends" >&2
  exit 1
fi
build_root=$(realpath "$1")
output_root=$(realpath "$2")
adapter_root=$(cd "$(dirname "$0")" && pwd)
if [[ ! -f $output_root/device-observation.json ]]; then
  echo "Device observation absent" >&2
  exit 1
fi
docker run --rm --network none --user "$(id -u):$(id -g)" \
  -e HOME=/tmp/weave-td-home -e NVIDIA_VISIBLE_DEVICES=void \
  -v "$build_root:/build:ro" -v "$adapter_root:/adapter:ro" \
  -v "$output_root:/out" -w /adapter \
  lmsysorg/sglang:latest-cu130-runtime \
  /build/venv/bin/python /adapter/runner.py check --output /out
