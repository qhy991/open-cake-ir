# Python ProgramMap options

`@cake.schedule(..., program_map={...})` exposes the existing map-level
`persistent` and `traversal` fields. `lm.program` continues to declare each axis.
Both spellings elaborate into the same canonical JSON `ProgramMap`; its parser,
Verifier and backend remain the owners of admission and emission.

```python
from open_cake_ir.compiler import frontend as cake

@cake.schedule(target="gfx938", backend="triton",
               residency={"ctas_per_multiprocessor": 2},
               program_map={"persistent": True,
                            "traversal": ("column", "row")})
def copy_tiles(lm, x: cake.Tensor((257, 130), "fp32"),
               y: cake.Tensor((257, 130), "fp32", mode="output")):
    compute = lm.role(execution_groups=[0, 1, 2, 3])
    row = lm.program(x, axis=0, dimension=0, tile=1)
    column = lm.program(x, axis=1, dimension=1, tile=64)
    with compute:
        values = lm.load(x[row, column], reuse="streamed")
        lm.store(y[row, column], values)
```

The map is serialized as:

```json
{
  "axes": [
    {"name": "row", "axis": 0, "buffer": "x", "dimension": 0, "tile": 1},
    {"name": "column", "axis": 1, "buffer": "x", "dimension": 1, "tile": 64}
  ],
  "persistent": true,
  "traversal": ["column", "row"]
}
```

`traversal` names every declared axis exactly once, fastest-varying first. Names
are strings because the decorator precedes the body. With persistence enabled,
omitting traversal uses ascending axis numbers. Omitting `program_map` preserves
existing Python documents and lowering; the default remains one CTA per tile.

Persistence uses the exact Target's declared multiprocessor count and the
Schedule's `residency.ctas_per_multiprocessor`, capped at the number of work tiles.
There is no separate launch-count option. The example has 771 work tiles; its
explicit residency and gfx938 Target facts give 128 launched CTAs. These are
lowering choices, not measured occupancy or a performance result.

The frontend refuses `axes` in the decorator mapping: axis names, owners and
tiles have one source in `lm.program`. It also refuses duplicate declarations,
options on individual axes, and a simultaneous explicit `grid`. Existing
ProgramMap rules reject traversal without persistence and malformed values;
the Verifier rejects missing/repeated/unknown axis names and missing CTA
residency. Triton refuses a persistent map when the exact Target has no occupancy
facts. Frontend access does not expand any backend's supported domain.

The CPU contract tests compare independent Python and JSON spellings, their
Assessments and emitted sources on gfx938 and sm_100a, with both traversal orders
and a masked tail. They also check the original refusal paths. No device
correctness or speedup is established by this authoring change; those require
separate target-bound evaluation.

中文要点：映射级选项只在 `@cake.schedule(program_map={...})` 中声明，轴只在
`lm.program` 中声明。`traversal` 从变化最快的轴开始排列，必须恰好包含全部轴名。
持久化所需的 CTA 数由已有 residency 与精确 Target 的 occupancy 声明推导；
缺失事实仍按原路径拒绝。本改动补齐 Python 作者接口，不新增 IR 字段或性能结论。

See [Python frontend](PYTHON_FRONTEND.md) for the shared authoring and diagnostic
contract. The implementation is `compiler/frontend.py`; the canonical map is
`compiler/ir/mapping.py`; executable coverage is
`tests/contracts/test_program_map_frontend.py`.
