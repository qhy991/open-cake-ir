"""The Workload tensor ABI every sealed launch manifest carries, whichever object it launches.

Two manifest spellings exist -- `workload_tensors_v1` for a module the host launches
itself (a cubin or an hsaco) and `metal_workload_tensors_v1` for a binary archive the
native observer dispatches -- and the execution platform row says which one a candidate
seals. What they say about the Workload is the same: which contract, which case, which
tensors in which order, on which target, under which kernel name, over which grid and
block. That half lived twice, once per spelling, with the same six methods; it lives here
once. Each spelling keeps its own document fields, its own limits and its own refusal
text, so a retained manifest parses exactly as it always did.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from hashlib import sha256
from typing import ClassVar, Mapping

from open_cake_ir.serialization import canonical_json_bytes

from .workload import WorkloadContract

TensorRows = tuple[tuple[str, tuple[int, ...], str, str], ...]


def check_pointer_alignments(pointers, alignments) -> None:
    """Check actual launch addresses, including outputs, never an allocation base."""
    for name, alignment in alignments.items():
        pointer = pointers.get(name)
        if type(pointer) is not int or pointer <= 0 or pointer % alignment:
            raise ValueError(f'tensor {name!r} does not meet its {alignment}-byte alignment contract')


def tensor_abi_rows(rows: object, *, dtypes: frozenset[str], ascii_names: bool,
                    row_error: str, order_error: str) -> TensorRows:
    """Parse the ordered tensor rows one spelling admits, with that spelling's words.

    Every row names one tensor with a positive shape, an admitted dtype and a mode; names
    are unique, inputs precede outputs and both modes occur. The caller has already
    checked how many rows its spelling allows.
    """
    abi = []
    for row in rows:
        if (not isinstance(row, Mapping) or set(row) != {"name", "shape", "dtype", "mode"}
                or not isinstance(row["name"], str) or not row["name"].isidentifier()
                or (ascii_names and not row["name"].isascii())
                or not isinstance(row["dtype"], str) or row["dtype"] not in dtypes
                or not isinstance(row["mode"], str) or row["mode"] not in {"input", "output"}
                or not isinstance(row["shape"], list) or not row["shape"]
                or any(type(v) is not int or v <= 0 for v in row["shape"])):
            raise ValueError(row_error)
        abi.append((row["name"], tuple(row["shape"]), row["dtype"], row["mode"]))
    modes = [row[3] for row in abi]
    if (len({row[0] for row in abi}) != len(abi) or "input" not in modes or "output" not in modes
            or modes != sorted(modes)):
        raise ValueError(order_error)
    return tuple(abi)


@dataclass(frozen=True)
class WorkloadTensorManifest:
    """The Workload half of a sealed launch manifest; a spelling adds its own launch half."""

    workload_sha256: str
    case_id: str
    tensor_abi: TensorRows
    target: str
    kernel_name: str
    grid: tuple[int, int, int]
    block: tuple[int, int, int]

    # The `abi` this spelling seals, and the words it refuses a Workload mismatch with.
    abi: ClassVar[str]
    workload_mismatch: ClassVar[str]

    def as_dict(self) -> dict:
        raise NotImplementedError

    def check_workload(self, workload: WorkloadContract, case_id: str) -> None:
        expected = tuple((t.name, t.shape, t.dtype, t.mode) for t in workload.tensor_abi(case_id))
        if (self.workload_sha256 != workload.canonical_sha256 or self.case_id != case_id
                or self.tensor_abi != expected
                or self.target != workload.target):
            raise ValueError(self.workload_mismatch)

    @property
    def block_threads(self) -> int:
        return math.prod(self.block)

    @property
    def canonical_sha256(self) -> str:
        return sha256(canonical_json_bytes(self.as_dict())).hexdigest()
