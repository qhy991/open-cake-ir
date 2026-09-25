#!/usr/bin/env bash
# Invoke only as the child command of gpu-run --mode exclusive --gpu-count 4.
set -euo pipefail
if [[ $# != 4 ]]; then
  echo "usage: $0 <upstream-checkout> <isolated-build-dir> <cpu-input-dir> <output-dir>" >&2
  exit 2
fi
source_root=$(realpath "$1")
build_root=$(realpath "$2")
input_root=$(realpath "$3")
output_root=$(realpath "$4")
adapter_root=$(cd "$(dirname "$0")" && pwd)
if [[ -z ${GPUQ_JOB_ID:-} || ${GPUQ_MODE:-} != exclusive || ${GPUQ_BACKEND:-} != nvidia ]]; then
  echo "This adapter requires a broker-issued exclusive NVIDIA lease" >&2
  exit 1
fi
if [[ ${CUDA_VISIBLE_DEVICES:-} != "${GPUQ_DEVICE_IDS:-}" ]]; then
  echo "Broker visibility and device allocation disagree" >&2
  exit 1
fi
IFS=, read -ra device_ids <<< "$GPUQ_DEVICE_IDS"
if [[ ${#device_ids[@]} != 4 ]]; then
  echo "This EP4 run requires exactly four broker-allocated GPUs" >&2
  exit 1
fi
if [[ ! -f $input_root/cpu-oracle-observation.json || ! -f $output_root/admission.json || -e $output_root/device-observation.json ]]; then
  echo "Expected new output with broker admission receipt" >&2
  exit 1
fi
for rank in 0 1 2 3; do
  if [[ -e $output_root/rank$rank-output.npy ]]; then
    echo "Output already exists for rank $rank" >&2
    exit 1
  fi
done

docker run --rm --network host --ipc host \
  --gpus "\"device=$GPUQ_DEVICE_IDS\"" \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp/weave-td-home -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -e GPUQ_JOB_ID -e GPUQ_MODE -e GPUQ_DEVICE_IDS \
  -v "$source_root:/src:ro" -v "$build_root:/build:ro" \
  -v "$adapter_root:/adapter:ro" -v "$input_root:/inputs:ro" \
  -v "$output_root:/out" -w /adapter \
  lmsysorg/sglang:latest-cu130-runtime \
  /build/venv/bin/torchrun --standalone --nproc_per_node=4 \
  /adapter/runner.py run --upstream /src --inputs /inputs --output /out
