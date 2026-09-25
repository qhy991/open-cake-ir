#!/usr/bin/env bash
# Invoke only as the child command of gpu-run --mode exclusive --gpu-count 4.
set -euo pipefail
if [[ $# != 2 ]]; then
  echo "usage: $0 <retained-cpu-input-dir> <new-output-dir>" >&2
  exit 2
fi
input_root=$(realpath "$1")
output_root=$(realpath "$2")
adapter_root=$(cd "$(dirname "$0")" && pwd)
image_id=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["source"]["container_image_id"])' "$adapter_root/contract_sglang_deepep.json")
if [[ ! -f $output_root/admission.json ]]; then
  echo "This adapter requires a broker-issued exclusive NVIDIA lease" >&2
  exit 1
fi
lease_info=$(python3 "$adapter_root/verify_broker_lease.py" "$output_root/admission.json")
read -r broker_job_id broker_device_ids <<< "$lease_info"
if [[ ! -f $input_root/cpu-oracle-observation.json || -e $output_root/device-observation.json ]]; then
  echo "Expected retained CPU input and new broker-admitted output" >&2
  exit 1
fi
cache_root=$output_root/runtime-cache
if [[ ! -d $cache_root ]]; then
  echo "Runtime cache absent; prepare it before the GPU lease" >&2
  exit 1
fi
for rank in 0 1 2 3; do
  if [[ -e $output_root/rank$rank-output.npy ]]; then
    echo "Output already exists for rank $rank" >&2
    exit 1
  fi
done

docker run --rm --network host --ipc host \
  --gpus "\"device=$broker_device_ids\"" \
  --user "$(id -u):$(id -g)" \
  -e HOME=/cache -e USER="$(id -un)" -e LOGNAME="$(id -un)" \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -e WEAVE_BROKER_JOB_ID="$broker_job_id" -e WEAVE_BROKER_DEVICE_IDS="$broker_device_ids" \
  -v "$adapter_root:/adapter:ro" -v "$input_root:/inputs:ro" \
  -v "$output_root:/out" -v "$cache_root:/cache" -w /adapter \
  "$image_id" \
  torchrun --standalone --nproc_per_node=4 \
  /adapter/runner_sglang_deepep.py run --inputs /inputs --output /out
