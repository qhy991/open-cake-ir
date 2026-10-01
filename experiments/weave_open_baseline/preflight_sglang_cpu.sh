#!/usr/bin/env bash
# No-GPU source/package/input gate for the SGLang + DeepEP fallback.
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <retained-cpu-input-dir> [contract_sglang_deepep_fanin_v2.json]" >&2
  exit 2
fi
input_root=$(realpath "$1")
adapter_root=$(cd "$(dirname "$0")" && pwd)
contract_name=${2:-contract_sglang_deepep.json}
if [[ $contract_name != contract_sglang_deepep.json && $contract_name != contract_sglang_deepep_fanin_v2.json ]]; then
  echo "Unreviewed SGLang/DeepEP contract name" >&2
  exit 2
fi
image_id=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["source"]["container_image_id"])' "$adapter_root/$contract_name")
docker run --rm --network none --user "$(id -u):$(id -g)" \
  -e HOME=/tmp/weave-sglang-home -e USER="$(id -un)" -e LOGNAME="$(id -un)" \
  -e NVIDIA_VISIBLE_DEVICES=void \
  -v "$adapter_root:/adapter:ro" -v "$input_root:/inputs:ro" \
  -w /adapter \
  "$image_id" \
  python3 /adapter/runner_sglang_deepep.py preflight --inputs /inputs \
  --experiment-contract "/adapter/$contract_name"
