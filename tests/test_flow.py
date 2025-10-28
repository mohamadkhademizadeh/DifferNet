import torch
from utils.flow_blocks import RealNVP

def test_forward_shapes():
    d=32
    f = RealNVP(d, n_blocks=2, hidden=64)
    x = torch.randn(4, d)
    z, log_px = f(x)
    assert z.shape == x.shape
    assert log_px.shape[0] == x.shape[0]
