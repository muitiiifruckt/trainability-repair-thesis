"""A-cycle 5: DESIGN of split Adam-state reset for the RL runner (H-split). Not wired into experiments/ (frozen).
Usage by a future runner action:  head_weights_and_state_head / state_body etc. via apply_split_state_reset(state.q, state.optimizer, scope).
scope in {'head','body','all'}. Deletes (exp_avg, exp_avg_sq, step) per parameter -> torch Adam re-initialises lazily with its own per-parameter step,
so head t restarts at 1 while body t continues (bias correction is per-parameter in torch.optim.Adam)."""
import sys; sys.dont_write_bytecode = True
import torch
from torch import nn
def apply_split_state_reset(model, optimizer, scope):
    head_ids = {id(p) for p in model.head.parameters()}
    n = 0
    for p in list(optimizer.state.keys()):
        is_head = id(p) in head_ids
        if scope == "all" or (scope == "head" and is_head) or (scope == "body" and not is_head):
            del optimizer.state[p]; n += 1
    return n
def _test():
    torch.manual_seed(0)
    class M(nn.Module):
        def __init__(s): super().__init__(); s.body = nn.Linear(4, 8); s.head = nn.Linear(8, 2)
        def forward(s, x): return s.head(torch.relu(s.body(x)))
    m = M(); o = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(5):
        o.zero_grad(); m(torch.randn(16, 4)).pow(2).mean().backward(); o.step()
    assert apply_split_state_reset(m, o, "head") == 2
    assert all(p not in o.state for p in m.head.parameters()) and all(p in o.state for p in m.body.parameters())
    o.zero_grad(); m(torch.randn(16, 4)).pow(2).mean().backward(); o.step()
    assert int(o.state[m.head.weight]["step"]) == 1 and int(o.state[m.body.weight]["step"]) == 6
    assert apply_split_state_reset(m, o, "body") == 2 and apply_split_state_reset(m, o, "all") == 2
    print("ok: split reset keeps per-parameter clocks (head t=1, body t=6 after head-only reset)")
if __name__ == "__main__": _test()
