"""CPU-only chunk algebra for a future B300 KDA native lowering.

This is a candidate derivation, not the Workload oracle or an executable Cake
Schedule. It keeps the V-first state matrix fixed during one chunk and solves
the token updates as a lower-triangular system. The large state matrix is read
and written once per chunk instead of once per token. The direct scalar CUDA
prototype showed that the latter is far too slow on B300-M4.
"""

from __future__ import annotations

import math

import numpy as np

from .oracle import _bf16_round, _sigmoid


def chunked_prefill(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    g: np.ndarray,
    beta: np.ndarray,
    A_log: np.ndarray,
    dt_bias: np.ndarray,
    initial_state: np.ndarray,
    cu_seqlens: np.ndarray,
    *,
    chunk: int = 32,
    scale: float | None = None,
    lower_bound: float = -5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate one candidate chunk decomposition on CPU for numerical screening.

    The Workload compares this result to :func:`oracle.reference_prefill`.
    Each token's output is rounded to BF16, while state is rounded at chunk
    boundaries. Direct products of intervening decays avoid dividing by a
    cumulative product that may underflow over 32 tokens.
    """

    if (type(chunk) is not int or chunk <= 0 or chunk > 32
            or chunk & (chunk - 1)):
        raise ValueError("chunk must be a power of two in [1, 32]")
    if not math.isfinite(lower_bound) or lower_bound >= 0:
        raise ValueError("lower_bound must be finite and negative")
    q = np.asarray(q, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    g = np.asarray(g, dtype=np.float32)
    beta = np.asarray(beta, dtype=np.float32)
    A_log = np.asarray(A_log, dtype=np.float32)
    dt_bias = np.asarray(dt_bias, dtype=np.float32)
    initial_state = np.asarray(initial_state, dtype=np.float32)
    offsets = np.asarray(cu_seqlens)
    if q.ndim != 4 or any(value.shape != q.shape for value in (k, v, g)):
        raise ValueError("q/k/v/g must share [batch, tokens, heads, width]")
    batch, tokens_per_batch, heads, width = q.shape
    total_tokens = batch * tokens_per_batch
    if (beta.shape != (batch, tokens_per_batch, heads)
            or A_log.shape != (heads,) or dt_bias.shape != (heads, width)
            or offsets.ndim != 1 or offsets.dtype.kind not in "iu"
            or len(offsets) < 2 or offsets[0] != 0
            or offsets[-1] != total_tokens or np.any(np.diff(offsets) <= 0)
            or initial_state.shape != (len(offsets) - 1, heads, width, width)):
        raise ValueError("candidate inputs must match the complete KDA state boundary")
    if scale is not None and not math.isfinite(scale):
        raise ValueError("scale must be finite")

    q = q.reshape(total_tokens, heads, width)
    k = k.reshape(total_tokens, heads, width)
    v = v.reshape(total_tokens, heads, width)
    g = g.reshape(total_tokens, heads, width)
    beta = beta.reshape(total_tokens, heads)
    q_norm = q / np.maximum(
        np.sqrt(np.sum(q * q, axis=-1, dtype=np.float32)), np.float32(1e-12)
    )[..., None]
    k_norm = k / np.maximum(
        np.sqrt(np.sum(k * k, axis=-1, dtype=np.float32)), np.float32(1e-12)
    )[..., None]
    decay = np.exp(lower_bound * _sigmoid(
        np.exp(A_log)[None, :, None] * (g + dt_bias[None, :, :])
    ))
    beta_gate = _sigmoid(beta)
    resolved_scale = np.float32(width ** -0.5 if scale is None else scale)
    output = np.empty_like(q)
    state = initial_state.copy()

    for sequence, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        current = state[sequence]
        for origin in range(int(start), int(end), chunk):
            stop = min(origin + chunk, int(end))
            keys = k_norm[origin:stop]
            queries = q_norm[origin:stop]
            decays = decay[origin:stop]
            values = v[origin:stop]
            betas = beta_gate[origin:stop]
            count = stop - origin
            prefix = np.cumprod(decays, axis=0, dtype=np.float32)

            # S0 times gated keys/queries: two VxK by KxC matrix products.
            base_prediction = np.einsum(
                "hvk,thk->thv", current, prefix * keys, optimize=True
            )
            base_output = np.einsum(
                "hvk,thk->thv", current, prefix * queries, optimize=True
            )

            # Couplings from prior token i to token t. Intervening decay is
            # accumulated directly so no inverse cumulative gate is needed.
            prediction_coupling = np.zeros((count, count, heads), dtype=np.float32)
            output_coupling = np.zeros((count, count, heads), dtype=np.float32)
            for source in range(count):
                between = np.ones((heads, width), dtype=np.float32)
                for target in range(source, count):
                    if target > source:
                        between *= decays[target]
                    propagated_key = keys[source] * between
                    output_coupling[target, source] = np.einsum(
                        "hk,hk->h", propagated_key, queries[target], optimize=True
                    )
                    if target > source:
                        prediction_coupling[target, source] = np.einsum(
                            "hk,hk->h", propagated_key, keys[target], optimize=True
                        )

            # U_t = beta_t * (V_t - S0 C_t - sum_{i<t} A[t,i] U_i).
            updates = np.empty((count, heads, width), dtype=np.float32)
            for token in range(count):
                prior = (
                    np.einsum(
                        "ih,ihv->hv", prediction_coupling[token, :token],
                        updates[:token], optimize=True,
                    ) if token else 0.0
                )
                updates[token] = betas[token, :, None] * (
                    values[token] - base_prediction[token] - prior
                )
                corrected = np.einsum(
                    "ih,ihv->hv", output_coupling[token, :token + 1],
                    updates[:token + 1], optimize=True,
                )
                output[origin + token] = _bf16_round(
                    resolved_scale * (base_output[token] + corrected)
                )

            # S_end = S0 diag(prefix_end) + U^T K_propagated.
            suffix = np.ones((heads, width), dtype=np.float32)
            propagated = np.empty_like(keys)
            for token in range(count - 1, -1, -1):
                propagated[token] = keys[token] * suffix
                suffix *= decays[token]
            current = _bf16_round(
                current * prefix[-1, :, None, :]
                + np.einsum("ihv,ihk->hvk", updates, propagated, optimize=True)
            )
        state[sequence] = current
    return output.reshape(batch, tokens_per_batch, heads, width), state
