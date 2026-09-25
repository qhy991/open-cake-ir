#!/usr/bin/env bash
# CPU-only source build. The container deliberately has no --gpus option.
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 <init|deps|cmake|build|probe> <upstream-checkout> <isolated-build-dir> [image]" >&2
  exit 2
fi
stage=$1
source_root=$(realpath "$2")
build_root=$(realpath -m "$3")
image=${4:-lmsysorg/sglang:latest-cu130-runtime}
expected=63de69e48dde17f32b0ee80ba83901c6950404cd
if [[ $(git -C "$source_root" rev-parse HEAD) != "$expected" ]]; then
  echo "upstream source commit mismatch" >&2
  exit 1
fi
if [[ -n $(git -C "$source_root" diff --name-only HEAD) ]]; then
  echo "upstream tracked files modified" >&2
  exit 1
fi
if [[ $stage == build ]] && [[ $(git -C "$source_root/3rdparty/triton" rev-parse HEAD) != f53694a72a1e4f464fa245df2c7305ccda7cb2a9 ]]; then
  echo "upstream Triton submodule mismatch" >&2
  exit 1
fi
if [[ $stage == init ]]; then
  if [[ -e $build_root ]]; then
    echo "build output already exists: $build_root" >&2
    exit 1
  fi
  mkdir -p "$build_root"
else
  if [[ ! -d $build_root ]]; then
    echo "build output absent: $build_root" >&2
    exit 1
  fi
fi

case "$stage" in
  init) command='python3 -m venv --system-site-packages /build/venv && /build/venv/bin/python -m pip --version' ;;
  deps) command='/build/venv/bin/python -m pip install --no-cache-dir numpy==1.26.4 cuda.core==0.2.0 cuda-python==12.4 nvidia-nvshmem-cu12==3.3.9 Cython==0.29.24 nvshmem4py-cu12==0.1.2 setuptools==69.0.0 cmake==3.31.10 wheel pybind11' ;;
  cmake) command='/build/venv/bin/python -m pip install --no-cache-dir cmake==3.31.10' ;;
  build) command='cd /src && USE_TRITON_DISTRIBUTED_AOT=0 MAX_JOBS=8 /build/venv/bin/python -m pip install --no-cache-dir -e python --verbose --no-build-isolation --use-pep517' ;;
  probe) command='/build/venv/bin/python -c "import torch, triton, triton_dist, numpy; print(torch.__version__, triton.__version__, numpy.__version__)"' ;;
  *) echo "unknown stage: $stage" >&2; exit 2 ;;
esac

docker run --rm --network host --user "$(id -u):$(id -g)" \
  -e HOME=/tmp/weave-td-home -e NVIDIA_VISIBLE_DEVICES=void \
  -e HTTP_PROXY -e HTTPS_PROXY -e NO_PROXY \
  -e http_proxy -e https_proxy -e no_proxy \
  -v "$source_root:/src" -v "$build_root:/build" -w /src \
  "$image" bash -lc "$command"
