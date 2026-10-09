# Frozen C1 candidate for c550-bench L1/069

This independent engineering evaluation uses Compiler commit
`5bb474c6df2948d1cc50b2d46f9ab828525be3c2` and Bench commit
`ababa4c0656b89bdf0a9c60ef0d2e90e0aebb425`.

Run the preparation script from its clean adapter checkout. Its commit is recorded
as `adapter_commit`, separately from the frozen `compiler_commit`. The adapter
commit adds evaluation tools and is not a new Compiler version or an optimization.

The complete task contains 16 original workloads. Inputs are contiguous BF16
`hidden_states` and `residual` with shape `[batch_size, seq_len, 8192]`, BF16
`weight[8192]`, and scalar `eps=1e-5`. The return value is a BF16 tensor of the
original input shape. Flattening the first two dimensions changes indexing only.

The Schedule retains both numerical boundaries in the Bench contract:

1. Add the two BF16 inputs in FP32, then round the sum to BF16.
2. Widen that rounded sum and perform RMS normalization in FP32.
3. Round the normalized value to BF16 before multiplication by the BF16 weight.
4. Round the weighted output to BF16.

`prepare_l1_069.py` reads the raw definition and workload metadata in place. It
does not copy or publish raw Bench data. It generates nine distinct row-count
variants, calls the frozen `Compiler.assess` and `Compiler.lower`, and compiles
their generated Triton through `IsolatedTritonCompiler`. Output creation is
exclusive. A failed attempt remains at its original output path.

CPU preparation, inside the root agent's accepted CPU container:

```sh
PYTHONPATH="$COMPILER_ROOT/src" /opt/conda/bin/python3 \
  "$ADAPTER_ROOT/tools/benchmarks/c550/prepare_l1_069.py" \
  --compiler-root "$COMPILER_ROOT" --bench-root "$BENCH_ROOT" --output "$OUTPUT_ROOT/compiled"
```

The runtime adapter `l1_069_cake_candidate.py` can be copied from that committed
adapter checkout into the external output directory. It requires
`CAKE_BENCH_ARTIFACTS=$OUTPUT_ROOT/compiled` and the same Compiler `src` on
`PYTHONPATH`. It uses the existing MACA admission and therefore must run as a
child of the existing local broker, under a physical device lease. The broker
must expose exactly one runtime device and retain its expected PCI identity.
Torch only allocates the output and provides tensor metadata and the current
stream. All arithmetic runs in the precompiled native ELF. No evaluation-stage
compilation, reference invocation, or Torch arithmetic exists in this adapter.

The root agent owns the leased launch of the unchanged Bench command:

```sh
/opt/conda/bin/python3 "$BENCH_ROOT/c550bench.py" check \
  --task L1/069_rms_norm --device cuda:0 \
  --candidate "$OUTPUT_ROOT/l1_069_cake_candidate.py" \
  --output "$OUTPUT_ROOT/result"
```

This evaluation must cover all 16 original workloads and all ten fresh-input
rounds per workload. Only the actual Bench receipt can establish full task
correctness. No performance or speedup claim follows from this check.

Local CPU verification completed: all nine original row-count variants were
accepted, lowered, and passed the isolated-kernel source boundary at the fixed
Compiler commit. The generated source retains both BF16 cast boundaries.
