from kernel import run as _impl

def run(hidden_states, weight):
    return _impl(hidden_states, weight)
