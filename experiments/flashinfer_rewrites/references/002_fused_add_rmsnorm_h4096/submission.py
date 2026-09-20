from kernel import run as _impl

def run(hidden_states, residual, weight):
    return _impl(hidden_states, residual, weight)
