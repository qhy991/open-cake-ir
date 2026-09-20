from best_kernel import fused_add_rmsnorm as _impl

def run(hidden_states, residual, weight):
    return _impl(hidden_states, residual, weight)
