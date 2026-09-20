from rmsnorm_h512 import run as _impl
import torch


def run(hidden_states, weight):
    _out = torch.empty_like(hidden_states)
    _impl(hidden_states, weight, _out)
    return _out
