"""Two carried MMA phases separated by a typed row solve (fixed C32 witness)."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

from open_cake_ir.compiler import Schedule, Target
from open_cake_ir.compiler.backends import native_cuda
from open_cake_ir.compiler.performance.work import work_bound
from open_cake_ir.compiler.verifier import verify

ROOT = Path(__file__).resolve().parents[2]
MMA = "tcgen05.mma.cta_group::1.kind::f16"


def document() -> dict:
    def buf(name, space, dtype, shape, mode="scratch", **extra):
        return dict(name=name, space=space, dtype=dtype, shape=shape, mode=mode, **extra)
    def access(op, name, indices):
        return dict(operation=op, buffer=name, indices=indices,
                    boundary="mask_tiled_axes")
    dim0={"source":"dimension","dimension":0}
    dim1={"source":"dimension","dimension":1}
    chunk={"source":"loop_tile","name":"chunk_index"}
    def mma(name, reads, out, ready, done, deps, pipe):
        return dict(id=name,kind="mma",role="mma",reads=reads,writes=[out],
                    waits=[ready,"state_ready" if pipe=="base" else "u_ready"],
                    signals=[done],depends_on=deps,pipeline=pipe,
                    parameters={"accumulator":"fp32","tile_shape":[128,32,32],
                                "instruction":{"contract":MMA,"shape":[128,32,16],
                                               "cta_group":1,"operand_source":"tensor",
                                               "operand_major":["k","k"]}})
    buffers=[
        buf("initial_state","global","bf16",[128,32],"input"),
        buf("b_base","global","bf16",[64,32],"input"),
        buf("p","global","bf16",[64,32],"input"),
        buf("b_correction","global","bf16",[64,32],"input"),
        buf("rhs","global","fp32",[128,64],"input"),
        buf("final_state","global","bf16",[128,32],"output"),
        buf("initial_reg","register","bf16",[128,32]),
        buf("state_tmem","tensor","bf16",[128,32],allocation="tensor"),
        buf("base_stage","shared","bf16",[32,32],allocation="base_smem",swizzle="swizzle_64b"),
        buf("base_acc","tensor","fp32",[128,32],allocation="tensor",byte_offset=16384),
        buf("base_pred","register","fp32",[128,32]),
        buf("rhs_reg","register","fp32",[128,32]),
        buf("rhs_adjusted","register","fp32",[128,32]),
        buf("p_stage","shared","bf16",[32,32],allocation="p_smem",swizzle="swizzle_64b"),
        buf("updates","register","fp32",[128,32]),
        buf("updates_bf","register","bf16",[128,32]),
        buf("updates_tmem","tensor","bf16",[128,32],allocation="tensor",byte_offset=8192),
        buf("correction_stage","shared","bf16",[32,32],allocation="correction_smem",swizzle="swizzle_64b"),
        buf("correction_acc","tensor","fp32",[128,32],allocation="tensor",byte_offset=32768),
        buf("correction","register","fp32",[128,32]),
        buf("next_state","register","bf16",[128,32]),
    ]
    operations=[
        dict(id="load_initial",kind="load",role="compute",reads=["initial_state"],
             writes=["initial_reg"],parameters={"movement":"global"}),
        dict(id="store_initial",kind="tmem_store",role="compute",reads=["initial_reg"],
             writes=["state_tmem"],signals=["state_ready"],depends_on=["load_initial"],
             parameters={"destination_atom":{"op":"tcgen05.St32x32b","repetition":8}}),
        dict(id="load_base",kind="load",role="copy",reads=["b_base"],
             writes=["base_stage"],signals=["base_ready"],pipeline="base",
             parameters={"movement":"tma","descriptor_box":[32,32]}),
        mma("mma_base",["state_tmem","base_stage"],"base_acc","base_ready",
            "base_done",["store_initial","load_base"],"base"),
        dict(id="read_base",kind="load",role="compute",reads=["base_acc"],
             writes=["base_pred"],waits=["base_done"],depends_on=["mma_base"],
             parameters={"movement":"tmem","source_atom":{"op":"tcgen05.Ld32x32b","repetition":16}}),
        dict(id="load_rhs",kind="load",role="compute",reads=["rhs"],
             writes=["rhs_reg"],parameters={"movement":"global"}),
        dict(id="subtract_base",kind="elementwise",role="compute",reads=["rhs_reg","base_pred"],
             writes=["rhs_adjusted"],depends_on=["load_rhs","read_base"],parameters={"op":"sub"}),
        dict(id="load_p",kind="load",role="copy",reads=["p"],
             writes=["p_stage"],signals=["p_ready"],parameters={"movement":"global"}),
        dict(id="solve",kind="forward_substitute",role="compute",
             reads=["p_stage","rhs_adjusted"],writes=["updates"],waits=["p_ready"],
             depends_on=["load_p","subtract_base"],parameters={}),
        dict(id="round_updates",kind="cast",role="compute",reads=["updates"],
             writes=["updates_bf"],depends_on=["solve"],parameters={"to":"bf16"}),
        dict(id="publish_updates",kind="tmem_store",role="compute",reads=["updates_bf"],
             writes=["updates_tmem"],signals=["u_ready"],depends_on=["round_updates"],
             parameters={"destination_atom":{"op":"tcgen05.St32x32b","repetition":8}}),
        dict(id="load_correction",kind="load",role="copy",reads=["b_correction"],
             writes=["correction_stage"],signals=["correction_ready"],pipeline="correction",
             parameters={"movement":"tma","descriptor_box":[32,32]}),
        mma("mma_correction",["updates_tmem","correction_stage"],"correction_acc",
            "correction_ready","correction_done",["publish_updates","load_correction"],"correction"),
        dict(id="read_correction",kind="load",role="compute",reads=["correction_acc"],
             writes=["correction"],waits=["correction_done"],depends_on=["mma_correction"],
             parameters={"movement":"tmem","source_atom":{"op":"tcgen05.Ld32x32b","repetition":16}}),
        dict(id="round_state",kind="cast",role="compute",reads=["correction"],
             writes=["next_state"],depends_on=["read_correction"],parameters={"to":"bf16"}),
        dict(id="publish_state",kind="tmem_store",role="compute",reads=["next_state"],
             writes=["state_tmem"],signals=["state_ready"],depends_on=["round_state"],
             parameters={"destination_atom":{"op":"tcgen05.St32x32b","repetition":8}}),
        dict(id="store_final",kind="store",role="compute",reads=["next_state"],
             writes=["final_state"],depends_on=["publish_state"],
             parameters={"coalesced":False}),
    ]
    body=[op["id"] for op in operations[2:]]
    return dict(
        schema_version=2,schedule_id="native-two-phase-carried-c32",target="sm_103a",
        lowering={"backend":"native_cuda","entry_point":"cake_two_phase_carried_c32"},
        grid=[1,1,1],metadata={},
        roles=[{"name":"compute","execution_groups":[0,1,2,3]},
               {"name":"mma","execution_groups":[4]},
               {"name":"copy","execution_groups":[5]}],
        allocations=[{"name":name,"space":"shared","size_bytes":2048}
                     for name in ("base_smem","p_smem","correction_smem")]+[
                     {"name":"tensor","space":"tensor","size_bytes":65536,
                      "tensor_columns":128,"allocating_role":"compute"}],
        buffers=buffers,
        pipelines=[{"name":"base","stages":1},{"name":"correction","stages":1}],
        barriers=[
            {"name":"state_ready","count":4,"producers":["compute"],"consumers":["mma"],"mechanism":"mbarrier"},
            {"name":"base_ready","count":1,"producers":["copy"],"consumers":["mma"],"mechanism":"mbarrier","pipeline":"base"},
            {"name":"base_done","count":1,"producers":["mma"],"consumers":["compute"],"mechanism":"mbarrier"},
            {"name":"p_ready","count":1,"producers":["copy"],"consumers":["compute"],"mechanism":"mbarrier"},
            {"name":"u_ready","count":4,"producers":["compute"],"consumers":["mma"],"mechanism":"mbarrier"},
            {"name":"correction_ready","count":1,"producers":["copy"],"consumers":["mma"],"mechanism":"mbarrier","pipeline":"correction"},
            {"name":"correction_done","count":1,"producers":["mma"],"consumers":["compute"],"mechanism":"mbarrier"},
        ],
        operations=operations,
        tile_loops=[{"name":"chunks","iterator":"chunk_index","buffer":"b_base",
                     "dimension":0,"tile":32,"body":body,"carried_buffers":["state_tmem"],
                     "range_options":{"num_stages":1,"loop_unroll_factor":1,"flatten":False,
                                      "warp_specialize":True,"disallow_acc_multi_buffer":True,
                                      "disable_licm":False}}],
        access_maps=[
            access("load_initial","initial_state",[dim0,dim1]),
            access("load_base","b_base",[chunk,dim1]),
            access("load_rhs","rhs",[dim0,chunk]),
            access("load_p","p",[chunk,dim1]),
            access("load_correction","b_correction",[chunk,dim1]),
            access("store_final","final_state",[dim0,dim1]),
        ],
        outputs=["final_state"],
    )


def target() -> Target:
    value=json.loads((ROOT/"compiler/targets/sm_103a.json").read_text())
    for kind in ("forward_substitute","tmem_store"):
        if kind not in value["operation_kinds"]:
            value["operation_kinds"].append(kind)
    return Target.from_dict(value)


class NativeTwoPhaseCarried(unittest.TestCase):
    def test_typed_two_phase_witness(self):
        schedule=Schedule.from_dict(document())
        self.assertEqual([f for f in verify(schedule,target()) if f.blocks_lowering],[])
        self.assertEqual(native_cuda.preflight(schedule,target()),())
        source=native_cuda.emit(schedule,target()).source
        self.assertLess(source.index("// CAKE_OP: mma_base"),source.index("// CAKE_OP: solve"))
        self.assertLess(source.index("// CAKE_OP: solve"),source.index("// CAKE_OP: mma_correction"))
        work=work_bound(schedule)
        self.assertIsNotNone(work)
        self.assertEqual(next(row.whole_grid for row in work.operation_repetitions
                              if row.operation=="solve"),2)
        self.assertEqual(work.flops-work.mma_flops,
                         2*128*32*31 + 2*128*32)  # solve FMAs plus RHS subtraction

    def test_correction_cannot_run_before_solve_and_u_publication(self):
        value=document()
        body=value["tile_loops"][0]["body"]
        body.remove("solve")
        body.insert(body.index("read_correction"),"solve")
        self.assertIn("NATIVE_TWO_PHASE_ORDER",
                      {f.code for f in native_cuda.preflight(Schedule.from_dict(value),target())})
        value=document()
        body=value["tile_loops"][0]["body"]
        body.remove("publish_updates")
        body.insert(body.index("read_correction"),"publish_updates")
        self.assertIn("NATIVE_TWO_PHASE_ORDER",
                      {f.code for f in native_cuda.preflight(Schedule.from_dict(value),target())})


if __name__=="__main__":
    unittest.main()
