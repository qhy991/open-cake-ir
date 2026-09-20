from kernel import run as _impl
import torch

def run(hidden_states, weight):
    output = torch.empty_like(hidden_states)
    _impl(hidden_states, weight, output)
    return output
