"""Task policy around the common authoring environment."""
from __future__ import annotations

from collections.abc import Mapping

from open_cake_ir.lab.environments import OpenCakeEnvironment
from .flash_kmeans.authoring import LOWERING_ROUTE


class TaskOpenCakeEnvironment(OpenCakeEnvironment):
    def __init__(self, compiler, toolchain, *, authority_document, workload, case_id, executor=None):
        explicit = isinstance(workload.document["semantics"].get("candidate_abi"), Mapping)
        if explicit and "candidate_selection" in authority_document:
            raise ValueError("empirical selection requires the Flash/direct-CUDA assay")
        if not explicit and authority_document.get("lowering_route") != LOWERING_ROUTE:
            raise ValueError("Open Cake Authoring Environment lowering route differs")
        super().__init__(compiler, toolchain, authority_document=authority_document,
                         workload=workload, case_id=case_id, executor=executor)
