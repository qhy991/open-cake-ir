from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class CudaTensorContract:
    """Exact Workload-derived tensor shapes for one launchable candidate."""

    batch: int
    tokens: int
    centroids: int
    features: int
    target: str = "sm_100a"

    def __post_init__(self) -> None:
        if self.target != "sm_100a" or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in (self.batch, self.tokens, self.centroids, self.features)
        ):
            raise ValueError("CUDA tensor contract differs")

    @property
    def tensors(self) -> tuple[tuple[str, tuple[int, ...], str], ...]:
        return (
            ("tokens", (self.batch, self.tokens, self.features), "torch.bfloat16"),
            (
                "centroids",
                (self.batch, self.centroids, self.features),
                "torch.bfloat16",
            ),
            ("centroid_sq", (self.batch, self.centroids), "torch.float32"),
            ("assignments", (self.batch, self.tokens), "torch.int32"),
        )
