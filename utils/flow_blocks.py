import torch, torch.nn as nn, torch.nn.functional as F
import math

class ActNorm(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        super().__init__()
        self.initialized = False
        self.bias = nn.Parameter(torch.zeros(1, num_features))
        self.log_scale = nn.Parameter(torch.zeros(1, num_features))
        self.eps = eps

    def initialize(self, x):
        with torch.no_grad():
            mean = x.mean(dim=0, keepdim=True)
            std = x.std(dim=0, keepdim=True) + self.eps
            self.bias.data = -mean
            self.log_scale.data = torch.log(1.0 / std)

    def forward(self, x):
        if not self.initialized:
            self.initialize(x)
            self.initialized = True
        z = (x + self.bias) * torch.exp(self.log_scale)
        logdet = self.log_scale.sum(dim=1)  # per-sample logdet for 1D features
        return z, logdet

class AffineCoupling(nn.Module):
    def __init__(self, d, hidden=512, mask='checkerboard'):
        super().__init__()
        self.d = d
        self.h = hidden
        # binary mask (alternate features)
        m = torch.zeros(d)
        m[::2] = 1.0
        self.register_buffer('mask', m)
        self.nn = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, d*2)
        )

    def forward(self, x):
        # x split by mask: x1 passes, x2 transformed
        x1 = x * self.mask
        h = self.nn(x1)
        s, t = h.chunk(2, dim=1)
        s = torch.tanh(s)  # stabilize
        y = x1 + (1 - self.mask) * (x * torch.exp(s) + t)
        logdet = ((1 - self.mask) * s).sum(dim=1)
        return y, logdet

class FlowStep(nn.Module):
    def __init__(self, d, hidden=512):
        super().__init__()
        self.actnorm = ActNorm(d)
        self.coupling = AffineCoupling(d, hidden)
        self.perm = torch.randperm(d)

    def forward(self, x):
        z, ld1 = self.actnorm(x)
        z = z[:, self.perm]  # permutation
        z, ld2 = self.coupling(z)
        return z, ld1 + ld2

class RealNVP(nn.Module):
    def __init__(self, d, n_blocks=8, hidden=512):
        super().__init__()
        self.blocks = nn.ModuleList([FlowStep(d, hidden) for _ in range(n_blocks)])

    def forward(self, x):
        logdet = 0.0
        z = x
        for b in self.blocks:
            z, ld = b(z)
            logdet = logdet + ld
        # log-likelihood under standard normal
        log_pz = -0.5 * (z*z + math.log(2*math.pi)).sum(dim=1)
        log_px = log_pz + logdet
        return z, log_px
