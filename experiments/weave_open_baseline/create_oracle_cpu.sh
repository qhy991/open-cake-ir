#!/usr/bin/env bash
# Build every expected EP4 output before requesting the broker lease.
set -euo pipefail
if [[ $# != 2 ]]; then
  echo "usage: $0 <isolated-build-dir> <new-oracle-output-dir>" >&2
  exit 2
fi
if [[ -n ${GPUQ_JOB_ID:-} ]]; then
  echo "CPU oracle must run outside a GPU lease" >&2
  exit 1
fi
build_root=$(realpath "$1")
output_root=$(realpath -m "$2")
adapter_root=$(cd "$(dirname "$0")" && pwd)
if [[ -e $output_root ]]; then
  echo "Oracle output already exists: $output_root" >&2
  exit 1
fi
output_parent=$(dirname "$output_root")
mkdir -p "$output_parent"
docker run --rm --network none --user "$(id -u):$(id -g)" \
  -e HOME=/tmp/weave-td-home -e NVIDIA_VISIBLE_DEVICES=void \
  -e OPENBLAS_NUM_THREADS=8 -e OMP_NUM_THREADS=8 -e MKL_NUM_THREADS=8 \
  -v "$build_root:/build:ro" -v "$adapter_root:/adapter:ro" \
  -v "$output_parent:/outputs" -w /adapter \
  lmsysorg/sglang:latest-cu130-runtime \
  /build/venv/bin/python /adapter/runner.py oracle \
  --output "/outputs/$(basename "$output_root")"
