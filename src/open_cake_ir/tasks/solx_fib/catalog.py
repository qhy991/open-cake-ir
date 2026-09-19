"""The 26-task pack inventory, distinct from tasks with working authoring routes.

Names are the pack join keys in the retained FlashInfer round-one inventory.
Listing a task here does not register a Workload, an oracle or a device qualification.
"""
from . import gemm, workload

TASK_IDS = (
    '001_fused_add_rmsnorm_h2048',
    '002_fused_add_rmsnorm_h4096',
    '003_fused_add_rmsnorm_h7168',
    '004_gemm_n128_k2048',
    '005_gemm_n256_k7168',
    '006_gemm_n2048_k4096',
    '007_gemm_n4096_k4096',
    '008_gemm_n4096_k14336',
    '009_gemm_n5120_k2048',
    '010_gemm_n6144_k4096',
    '011_gemm_n28672_k4096',
    '012_gqa_paged_decode_h32_kv4_d128_ps1',
    '013_gqa_paged_decode_h32_kv8_d128_ps1',
    '014_gqa_paged_prefill_causal_h32_kv4_d128_ps1',
    '015_gqa_paged_prefill_causal_h32_kv8_d128_ps1',
    '016_gqa_ragged_prefill_causal_h32_kv4_d128',
    '017_gqa_ragged_prefill_causal_h32_kv8_d128',
    '018_mla_paged_decode_h16_ckv512_kpe64_ps1',
    '019_mla_paged_prefill_causal_h16_ckv512_kpe64_ps1',
    '020_moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048',
    '021_rmsnorm_h128',
    '022_rmsnorm_h512',
    '023_rmsnorm_h1536',
    '024_rmsnorm_h2048',
    '025_rmsnorm_h4096',
    '026_rmsnorm_h7168',
)


def task_owner(task_id: str):
    if task_id not in TASK_IDS:
        raise ValueError("unknown FlashInfer pack task id")
    name = "fib_" + task_id.split("_", 1)[1]
    if name in workload.TASKS:
        return name, workload
    if name in gemm.TASKS:
        return name, gemm
    return name, None
