"""CPU reference for Weave's scheduling decisions and work conservation.

This model owns no Cake IR admission or GPU timing. It makes the runtime plan
inputs and the queue/DAG invariants explicit before adding native CUDA control
flow or cross-CTA synchronization.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from collections.abc import Mapping, Sequence
from types import MappingProxyType


@dataclass(frozen=True)
class RoutedVolume:
    local_pairs: int
    incoming_pairs: int
    incoming_unique: int
    hidden: int
    intermediate: int

    def __post_init__(self) -> None:
        if (any(type(value) is not int for value in (
                    self.local_pairs, self.incoming_pairs, self.incoming_unique,
                    self.hidden, self.intermediate))
                or min(self.local_pairs, self.incoming_pairs, self.incoming_unique) < 0
                or self.incoming_unique > self.incoming_pairs
                or self.hidden <= 0 or self.intermediate <= 0):
            raise ValueError("routing counts and MoE dimensions differ")

    @property
    def compute_flops(self) -> int:
        return (self.local_pairs + self.incoming_pairs) * 6 * self.hidden * self.intermediate

    @property
    def dispatch_bytes(self) -> int:
        return self.incoming_unique * 2 * self.hidden

    @property
    def combine_bytes(self) -> int:
        return self.incoming_pairs * 2 * self.hidden


@dataclass(frozen=True)
class Calibration:
    """Exact-target measured curves, supplied rather than inherited."""

    bandwidth_by_comm_sms: Mapping[int, float]
    flops_by_compute_sms: Mapping[int, float]
    chunk_efficiency: Mapping[int, float]
    overlap_ratio: float

    def __post_init__(self) -> None:
        if not 0 <= self.overlap_ratio < 1:
            raise ValueError("overlap ratio must be in [0, 1)")
        for curve in (self.bandwidth_by_comm_sms, self.flops_by_compute_sms):
            if not curve or any(type(k) is not int or k <= 0
                                or isinstance(v, bool) or not isinstance(v, (int, float))
                                or not math.isfinite(v) or v <= 0
                                for k, v in curve.items()):
                raise ValueError("throughput curves require positive finite observations")
        if (not self.chunk_efficiency
                or any(type(k) is not int or k <= 0
                       or isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) or not 0 < v <= 1
                       for k, v in self.chunk_efficiency.items())):
            raise ValueError("chunk efficiency requires positive finite observations at or below one")
        for name in ("bandwidth_by_comm_sms", "flops_by_compute_sms", "chunk_efficiency"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))


@dataclass(frozen=True)
class Plan:
    comm_sms: int
    chunks: int
    predicted_seconds: float
    compute_seconds: float
    communication_seconds: float
    exposed_tail_seconds: float
    steal_tiles_per_sm_estimate: float


def choose_plan(n_sms: int, routed: RoutedVolume, measured: Calibration,
                *, tile_flops: int) -> Plan:
    """Evaluate paper equations 1–16 over the *declared* calibration grid.

    The steal estimate is continuous, as in the paper. It is not a work-claim
    count or permission to publish a payload; the queue model below treats the
    integer steal limit as an explicit separate input.
    """
    if type(n_sms) is not int or n_sms < 2 or type(tile_flops) is not int or tile_flops <= 0:
        raise ValueError("SM count and tile FLOPs must be positive integers")
    candidates = []
    for comm_sms, bandwidth in measured.bandwidth_by_comm_sms.items():
        compute_sms = n_sms - comm_sms
        if not 1 <= comm_sms < n_sms or compute_sms not in measured.flops_by_compute_sms:
            raise ValueError("calibration does not cover a legal communication/compute split")
        throughput = measured.flops_by_compute_sms[compute_sms]
        communication = (routed.dispatch_bytes + routed.combine_bytes) / bandwidth
        for chunks, efficiency in measured.chunk_efficiency.items():
            computation = routed.compute_flops / (throughput * efficiency)
            tail = (1 - measured.overlap_ratio) * routed.combine_bytes / (bandwidth * chunks)
            total = max(computation + tail, communication)
            remaining = max(0.0, routed.compute_flops - communication * throughput)
            steal = remaining / (n_sms * tile_flops)
            candidates.append(Plan(comm_sms, chunks, total, computation, communication, tail, steal))
    return min(candidates, key=lambda plan: (plan.predicted_seconds, plan.comm_sms, plan.chunks))


@dataclass(frozen=True, order=True)
class Tile:
    chunk: int
    phase: str
    index: int


@dataclass(frozen=True)
class Event:
    worker: str
    tile: Tile
    stolen: bool


class ChunkQueue:
    """Immediate-completion DAG model, without timing or GPU memory visibility."""

    PHASES = ("dispatch", "gemm0", "gemm1", "combine")

    def __init__(self, chunks: Sequence[tuple[int, int, int, int]], *, max_stolen: int):
        if not chunks or type(max_stolen) is not int or max_stolen < 0:
            raise ValueError("active chunks and a nonnegative steal budget are required")
        if any(len(row) != 4 or any(type(count) is not int or count < 0 for count in row)
               for row in chunks):
            raise ValueError("each chunk names nonnegative dispatch/GEMM/combine tile counts")
        self.pending = {Tile(chunk, phase, index)
                        for chunk, row in enumerate(chunks)
                        for phase, count in zip(self.PHASES, row, strict=True)
                        for index in range(count)}
        self.completed: set[Tile] = set()
        self.events: list[Event] = []
        self.chunks = tuple(tuple(row) for row in chunks)
        self.max_stolen = max_stolen
        self.stolen = 0
        self.combine_started = False

    def _phase_complete(self, chunk: int, phase: str) -> bool:
        count = self.chunks[chunk][self.PHASES.index(phase)]
        return all(Tile(chunk, phase, index) in self.completed for index in range(count))

    def _ready(self, tile: Tile) -> bool:
        if tile.phase == "dispatch":
            return True
        predecessor = self.PHASES[self.PHASES.index(tile.phase) - 1]
        return self._phase_complete(tile.chunk, predecessor)

    def _first(self, phases: tuple[str, ...], *, final_only: bool = False) -> Tile | None:
        for tile in sorted(self.pending):
            if (tile.phase in phases
                    and (not final_only or tile.chunk == len(self.chunks) - 1)
                    and self._ready(tile)):
                return tile
        return None

    def claim(self, worker: str) -> Event | None:
        """Claim one ready tile; a comm worker steals only in one pre-combine window."""
        if worker not in {"comm", "compute"}:
            raise ValueError("worker must be comm or compute")
        all_dispatch_done = all(self._phase_complete(chunk, "dispatch")
                                for chunk in range(len(self.chunks)))
        all_gemm_done = all(self._phase_complete(chunk, phase)
                            for chunk in range(len(self.chunks)) for phase in ("gemm0", "gemm1"))
        earlier_combine_done = all(self._phase_complete(chunk, "combine")
                                   for chunk in range(len(self.chunks) - 1))
        final_ready = all_gemm_done and earlier_combine_done
        tile = None
        stolen = False
        if worker == "comm":
            tile = self._first(("dispatch",))
            if tile is None and all_dispatch_done:
                tile = self._first(("combine",))
                if tile is not None and tile.chunk == len(self.chunks) - 1 and not final_ready:
                    tile = None
                if tile is None and not self.combine_started and self.stolen < self.max_stolen:
                    tile = self._first(("gemm0", "gemm1"))
                    stolen = tile is not None
        else:
            tile = self._first(("gemm0", "gemm1"))
            if tile is None and final_ready:
                tile = self._first(("combine",), final_only=True)
        if tile is None:
            return None
        self.pending.remove(tile)
        self.completed.add(tile)
        if stolen:
            self.stolen += 1
        if tile.phase == "combine":
            self.combine_started = True
        event = Event(worker, tile, stolen)
        self.events.append(event)
        return event


def run_chunks(n_sms: int, comm_sms: int, chunks: Sequence[tuple[int, int, int, int]],
               *, max_stolen: int, seed: int) -> tuple[Event, ...]:
    """Explore one fair worker order; no progress is an actual DAG deadlock."""
    if type(n_sms) is not int or type(comm_sms) is not int or not 1 <= comm_sms < n_sms:
        raise ValueError("both worker groups need at least one SM")
    queue = ChunkQueue(chunks, max_stolen=max_stolen)
    workers = ["comm"] * comm_sms + ["compute"] * (n_sms - comm_sms)
    rng = random.Random(seed)
    while queue.pending:
        rng.shuffle(workers)
        progress = False
        for worker in workers:
            progress |= queue.claim(worker) is not None
        if not progress:
            raise RuntimeError("chunk DAG made no progress")
    return tuple(queue.events)
