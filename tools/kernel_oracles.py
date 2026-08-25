"""Current correctness oracles for emitted-kernel observations.

`kernel_cases.ORACLES` predates the ranking calibration and its exact file bytes are now
part of frozen calibration authority. Rewriting those authorities would invalidate
historical replay. This module is the current extension point: it projects the retained
base registry and owns only oracles added after that freeze.
"""

from __future__ import annotations

from kernel_cases import ORACLES as _RETAINED_ORACLES


def _swiglu_oracle(inputs, torch):
    """The tanh identity used by the KDA v12 SwiGLU delta.

    Written from the operator definition rather than from the generated decomposition.
    It proves the standalone arithmetic slice, not grouped GEMMs or routed scatter.
    """

    up, gate, _ = inputs
    return up * gate * (0.5 * (torch.tanh(0.5 * gate) + 1.0)), None


ORACLES = {**_RETAINED_ORACLES, "swiglu_b8_smoke": _swiglu_oracle}
