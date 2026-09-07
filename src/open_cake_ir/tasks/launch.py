from __future__ import annotations
from typing import Mapping
from open_cake_ir.evaluation.core import TensorLaunchManifest

def parse_launch_manifest(document: object):
    """Historical fixed ABI and explicit Workload ABI meet at one replay boundary."""
    from .flash_kmeans.cuda_manifest import CudaLaunchManifest
    if isinstance(document, Mapping) and document.get('abi') == 'workload_tensors_v1':
        return TensorLaunchManifest.from_dict(document)
    return CudaLaunchManifest.from_dict(document)
