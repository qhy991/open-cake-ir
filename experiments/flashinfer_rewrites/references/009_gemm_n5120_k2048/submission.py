from solution import run as _impl
import torch


def run(A, B):
    _out = torch.empty((A.shape[0], B.shape[0]), dtype=A.dtype, device=A.device)
    _impl(A, B, _out)
    return _out
