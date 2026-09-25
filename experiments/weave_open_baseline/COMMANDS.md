# Retained B300-M4 reproduction commands

All paths below are on B300-M4. Source and experiment commit `7a179dc8` ran
the three successful device jobs. The broker generated each `admission.json`;
none was copied between jobs. The independent CPU input/oracle existed before
the first GPU request. The paths are the **actual retained outputs**; use fresh
paths for a new reproduction because all outputs are create-only.

```bash
ADAPTER=/home/qinhaiyan/weave-sglang-deepep-adapter-7a179dc8-20260925
INPUT=/home/qinhaiyan/weave-fanin-cpu-oracle-7a179dc8-20260925
GPU_RUN=/home/qinhaiyan/agent-gpu-broker/bin/gpu-run
PYTHON=/home/qinhaiyan/opt/python312/bin/python3.12

OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 "$PYTHON" \
  "$ADAPTER/create_fanin_oracle.py" --output "$INPUT"
"$ADAPTER/preflight_sglang_cpu.sh" "$INPUT" \
  contract_sglang_deepep_fanin_v2.json
```

For each row, create the new output and `runtime-cache` directory **before**
submitting. These three commands then run sequentially, never concurrently:

```bash
OUT=/home/qinhaiyan/weave-sglang-deepep-fanin-ep4-run-7a179dc8-20260925
"$GPU_RUN" --label weave-sglang-deepep-fanin-ep4-7a179dc8 \
  --mode exclusive --gpu-count 4 --estimate unknown \
  --queue-timeout 2h --run-timeout 10m \
  --receipt-out "$OUT/admission.json" --cwd /home/qinhaiyan -- \
  "$ADAPTER/run_sglang_under_broker.sh" "$INPUT" "$OUT" \
  contract_sglang_deepep_fanin_v2.json > "$OUT/broker.log" 2>&1

OUT=/home/qinhaiyan/weave-sglang-deepep-fanin-ep4-repeat1-7a179dc8-20260925
"$GPU_RUN" --label weave-sglang-deepep-fanin-repeat1-7a179dc8 \
  --mode exclusive --gpu-count 4 --estimate unknown \
  --queue-timeout 2h --run-timeout 10m \
  --receipt-out "$OUT/admission.json" --cwd /home/qinhaiyan -- \
  "$ADAPTER/run_sglang_under_broker.sh" "$INPUT" "$OUT" \
  contract_sglang_deepep_fanin_v2.json > "$OUT/broker.log" 2>&1

OUT=/home/qinhaiyan/weave-sglang-deepep-fanin-ep4-profile-7a179dc8-20260925
"$GPU_RUN" --label weave-sglang-deepep-fanin-profile-7a179dc8 \
  --mode exclusive --gpu-count 4 --estimate unknown \
  --queue-timeout 2h --run-timeout 10m \
  --receipt-out "$OUT/admission.json" --cwd /home/qinhaiyan -- \
  "$ADAPTER/run_sglang_under_broker.sh" "$INPUT" "$OUT" \
  contract_sglang_deepep_fanin_v2.json profile > "$OUT/broker.log" 2>&1
```

The repeat's JIT cache was fresh. Before the profile request, the previous
run's `runtime-cache/` was copied into the profile output directory to keep
compilation outside the observed trace; the copy was CPU-only:

```bash
rsync -a \
  /home/qinhaiyan/weave-sglang-deepep-fanin-ep4-repeat1-7a179dc8-20260925/runtime-cache/ \
  /home/qinhaiyan/weave-sglang-deepep-fanin-ep4-profile-7a179dc8-20260925/runtime-cache/
```

After each
broker job reached a terminal state and released its GPUs, its outputs were
checked with:

```bash
"$PYTHON" "$ADAPTER/runner_sglang_deepep.py" check \
  --experiment-contract "$ADAPTER/contract_sglang_deepep_fanin_v2.json" \
  --inputs "$INPUT" --output "$OUT" > "$OUT/check.log" 2>&1
```

The bitwise repeat audit is `repeat-comparison.json` in the repeat directory.
The profile directory contains four Chrome traces and `trace-audit.json`.
These are correctness and development trace records, not a qualified timing
protocol or a Cake speedup.

The derived audits were run on the CPU after lease release:

```bash
"$PYTHON" /home/qinhaiyan/weave-audit-tools-2f5b1b27-20260925/audit_routing.py \
  --inputs "$INPUT" --out "$INPUT/routing-volume.json"
"$PYTHON" /home/qinhaiyan/weave-sglang-deepep-adapter-d16a83b6-20260925/compare_repeats.py \
  --first /home/qinhaiyan/weave-sglang-deepep-fanin-ep4-run-7a179dc8-20260925 \
  --second /home/qinhaiyan/weave-sglang-deepep-fanin-ep4-repeat1-7a179dc8-20260925 \
  --out /home/qinhaiyan/weave-sglang-deepep-fanin-ep4-repeat1-7a179dc8-20260925/repeat-comparison.json
"$PYTHON" /home/qinhaiyan/weave-sglang-deepep-adapter-ecbf4977-20260925/audit_trace.py \
  --dir /home/qinhaiyan/weave-sglang-deepep-fanin-ep4-profile-7a179dc8-20260925 \
  --out /home/qinhaiyan/weave-sglang-deepep-fanin-ep4-profile-7a179dc8-20260925/trace-audit.json
"$PYTHON" /home/qinhaiyan/weave-sglang-deepep-adapter-d16a83b6-20260925/compare_repeats.py \
  --first /home/qinhaiyan/weave-sglang-deepep-fanin-ep4-run-7a179dc8-20260925 \
  --second /home/qinhaiyan/weave-sglang-deepep-fanin-ep4-profile-7a179dc8-20260925 \
  --out /home/qinhaiyan/weave-sglang-deepep-fanin-ep4-profile-7a179dc8-20260925/profile-vs-unprofiled.json
```
