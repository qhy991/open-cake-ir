"""Independent token recurrence for the BF16 recurrent-KDA prefill boundary.

This is a correctness oracle, not a candidate kernel or a timing baseline. Inputs
represent BF16 tensor values as FP32 arrays; state and output round to BF16 after
each token, matching the public recurrent state effect.
"""

from __future__ import annotations

import math

import numpy as np


def _bf16_round(values: np.ndarray) -> np.ndarray:
    """Round finite FP32 values to BF16 with ties to even, retaining FP32 storage."""

    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def reference_prefill(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    g: np.ndarray,
    beta: np.ndarray,
    A_log: np.ndarray,
    dt_bias: np.ndarray,
    initial_state: np.ndarray | None,
    cu_seqlens: np.ndarray,
    *,
    scale: float | None = None,
    lower_bound: float = -5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return complete output and final V-first state for fixed or packed sequences.

    ``cu_seqlens`` always gives logical sequence boundaries. A fixed-layout batch
    uses ``[0, T, 2T, ...]``; packed storage uses one leading batch axis and the
    packed offsets. The function does not mutate any caller-owned array.
    """

    q = np.asarray(q, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    g = np.asarray(g, dtype=np.float32)
    beta = np.asarray(beta, dtype=np.float32)
    A_log = np.asarray(A_log, dtype=np.float32)
    dt_bias = np.asarray(dt_bias, dtype=np.float32)
    offsets = np.asarray(cu_seqlens)
    if q.ndim != 4 or any(t.shape != q.shape for t in (k, v, g)):
        raise ValueError("q/k/v/g must share [batch, tokens, heads, width]")
    batch, tokens_per_batch, heads, width = q.shape
    total_tokens = batch * tokens_per_batch
    if beta.shape != (batch, tokens_per_batch, heads):
        raise ValueError("beta must share the q token/head axes")
    if A_log.shape != (heads,) or dt_bias.shape != (heads, width):
        raise ValueError("A_log and dt_bias must cover every head and key coordinate")
    if (offsets.ndim != 1 or offsets.dtype.kind not in "iu" or len(offsets) < 2
            or offsets[0] != 0 or offsets[-1] != total_tokens
            or np.any(np.diff(offsets) <= 0)):
        raise ValueError("cu_seqlens must partition all tokens into nonempty sequences")
    sequences = len(offsets) - 1
    state_shape = (sequences, heads, width, width)
    if initial_state is None:
        state = np.zeros(state_shape, dtype=np.float32)
    else:
        initial = np.asarray(initial_state, dtype=np.float32)
        if initial.shape != state_shape:
            raise ValueError("initial_state must have V-first [sequences, heads, V, K] shape")
        state = initial.copy()
    if (not math.isfinite(lower_bound) or lower_bound >= 0
            or scale is not None and not math.isfinite(scale)):
        raise ValueError("lower_bound must be finite and negative; scale finite")
    for name, value in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta),
                        ("A_log", A_log), ("dt_bias", dt_bias), ("state", state)):
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be finite")

    q_flat = q.reshape(total_tokens, heads, width)
    k_flat = k.reshape(total_tokens, heads, width)
    v_flat = v.reshape(total_tokens, heads, width)
    g_flat = g.reshape(total_tokens, heads, width)
    beta_flat = beta.reshape(total_tokens, heads)
    q_norm = q_flat / np.maximum(
        np.sqrt(np.sum(q_flat * q_flat, axis=-1, dtype=np.float32)),
        np.float32(1e-12),
    )[..., None]
    k_norm = k_flat / np.maximum(
        np.sqrt(np.sum(k_flat * k_flat, axis=-1, dtype=np.float32)),
        np.float32(1e-12),
    )[..., None]
    beta_gate = _sigmoid(beta_flat)
    gate_rate = np.exp(A_log)[None, :, None]
    decay = np.exp(lower_bound * _sigmoid(
        gate_rate * (g_flat + dt_bias[None, :, :])
    ))
    resolved_scale = np.float32(width ** -0.5 if scale is None else scale)
    output = np.empty_like(q_flat)
    for sequence, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        current = state[sequence]
        for token in range(int(start), int(end)):
            decayed = current * decay[token, :, None, :]
            prediction = np.einsum("hvk,hk->hv", decayed, k_norm[token], optimize=True)
            residual = beta_gate[token, :, None] * (v_flat[token] - prediction)
            current = _bf16_round(
                decayed + residual[:, :, None] * k_norm[token, :, None, :]
            )
            projected = np.einsum("hvk,hk->hv", current, q_norm[token], optimize=True)
            output[token] = _bf16_round(resolved_scale * projected)
        state[sequence] = current
    return output.reshape(q.shape), state
