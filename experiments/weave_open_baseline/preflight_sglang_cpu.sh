#!/usr/bin/env bash
# No-GPU source/package/input gate for the SGLang + DeepEP fallback.
set -euo pipefail
if [[ $# != 1 ]]; then
  echo "usage: $0 <retained-cpu-input-dir>" >&2
  exit 2
fi
input_root=$(realpath "$1")
adapter_root=$(cd "$(dirname "$0")" && pwd)
image_id=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["source"]["container_image_id"])' "$adapter_root/contract_sglang_deepep.json")
docker run --rm --network none --user "$(id -u):$(id -g)" \
  -e HOME=/tmp/weave-sglang-home -e NVIDIA_VISIBLE_DEVICES=void \
  -v "$adapter_root:/adapter:ro" -v "$input_root:/inputs:ro" \
  -w /adapter \
  "$image_id" \
  python3 /adapter/runner_sglang_deepep.py preflight --inputs /inputs
