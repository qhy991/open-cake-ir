# Python artifact optimization — Metal scaffold v4

Use the supplied Workload Contract and Python starter to propose structurally distinct
Schedules for the declared exact Apple target and shape. Preserve
argument order, tensor shapes, FP32 numerical semantics and output ownership. Author
Python only; submit each source as a `python_source` member of the candidate-set envelope
named by the StateCard. Keep the starter's ABI, target binding and lowering route intact.

The Lab already holds the frozen Workload. Do not write or copy a
`workload_contract_sha256` into the decorator; the Lab binds it after checking
the exact target, route and public tensor ABI. An explicit conflicting identity
is rejected.

The current Compiler Metal route enforces these authoring limits:

- Declare exactly one role with the current frontend spelling
  `execution_groups=[0, ..., n-1]`. Groups must be consecutive from zero, with at least
  one group and no more than the exact Target's derived `maximum_warps_per_cta`.
- A multi-group role widens the emitted stripe. Its elementwise operations may combine
  matching-shape operands and scalars. A narrower non-scalar operand is refused with
  `METAL_BROADCAST_WIDTH_UNSUPPORTED` because this route has no cross-group exchange for
  that broadcast.
- Use a non-persistent scalar ProgramMap: each `lm.program` has `tile=1`. Do not add
  `lm.range` tile loops, explicit allocations, pipelines or barriers; this route does not
  emit them.
- Access tensors through scalar program indices and finite contiguous dimension slices
  (`METAL_ACCESS_UNSUPPORTED`). Preserve every varying program index in output stores,
  and set stores to `coalesced=False` (`METAL_COALESCING_UNSUPPORTED`).
- Keep global inputs, outputs, private intermediates and finite scalar literals FP32.
  When fused multiply-add is part of the Workload semantics, spell it explicitly as
  `lm.fma(..., instruction={"contract": "metal.fma.f32"})`. Do not use CUDA/PTX
  contracts.
- Put scalar literals on the right of frontend binary expressions, as in `x * 2.0`;
  preserve operand order and the Workload's numerical contract.

This example exercises the current frontend spelling, two consecutive execution groups,
scalar row mapping and the admitted Metal FP32 FMA contract. It is illustrative only:
retain the supplied starter's target, entry point, shapes and ABI.

```python
from open_cake_ir.compiler import frontend as cake

@cake.schedule(name="metal-fma-template", target="apple_gpu_family8", backend="metal",
               entry_point="cake_metal_fma_template")
def candidate(lm, a: cake.Tensor((3, 37), "fp32"),
              b: cake.Tensor((3, 37), "fp32"),
              c: cake.Tensor((3, 37), "fp32"),
              out: cake.Tensor((3, 37), "fp32", mode="output")):
    compute = lm.role(execution_groups=[0, 1])
    row = lm.program(a, axis=0, dimension=0, tile=1)
    with compute:
        a_values = lm.load(a[row, :], id="load_a")
        b_values = lm.load(b[row, :], id="load_b")
        c_values = lm.load(c[row, :], id="load_c")
        result = lm.fma(a_values, b_values, c_values,
                        instruction={"contract": "metal.fma.f32"}, id="fma")
        lm.store(out[row, :], result, coalesced=False, id="store_out")
```

Generate candidates that differ in dataflow or arithmetic structure, load reuse, live
private intermediates, reduction structure, or work partition. An admitted execution-group
count may be tuned alongside one of those structural changes, but changing the group count
alone does not satisfy the required structural alternative. A renaming or formatting change
is also not structurally distinct. Keep a fixed baseline and do not hide an incorrect or
slow seed behind a dispatcher predicate.

Compiler diagnostics and static resource analysis filter candidates before device time.
The emitted `Peak live lane-owned Buffer values` number is a logical storage-slot estimate,
not a physical register count, spill measurement, occupancy result or performance result.
The current Metal assay's timestamp-only profiling does not measure physical registers,
spills or occupancy; claims about those resources require appropriate target counter
evidence. Only qualified timing supports a performance claim. Read localized Compiler and
Evaluation feedback before the next turn.

The common Lab owns compilation, checks across every input case, timing, profiling and
stopping. Do not invoke those tools, a GPU, the network or another task. Change only the
named candidate envelope; keep TASK.md and AGENTS.md unchanged. Do not author serialized
Schedule JSON or insert a low-level baseline.

This Metal v4 scaffold is a separate authoring treatment from v3. Keep its Runs separate;
do not pool, relabel or extend earlier evidence. A locally confirmed artifact establishes
no arm-comparison or cross-shape claim.
