"""Shared temporal encoder + per-user linear heads with a noisy-AND trajectory likelihood.

    r_k(t) = w_k^T phi(o_{t-5:t+5}) + b_k
    P(y=1 | tau, k) = prod_t sigmoid(r_k(t))

phi sees a short centered window because jerk/wobble aren't defined from a single step.

With shared=True each head is a population head plus a per-user correction,
w_k = w_0 + delta_k, b_k = b_0 + c_k, and train(prior=lam) adds lam * sum_k |delta_k, c_k|^2
(a Gaussian prior around the population head): users with few or uninformative labels
fall back to the population, and only move away where their labels consistently disagree.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def log1mexp(x):
    """log(1 - exp(x)) for x < 0, numerically stable."""
    x = x.clamp(max=-1e-7)
    return torch.where(x > -0.6931, torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


class RewardModel(nn.Module):
    def __init__(self, obs_dim, n_heads, horizon, d=32, hidden=64, window=11, shared=False):
        super().__init__()
        self.shared = shared
        self.enc = nn.Sequential(
            nn.Conv1d(obs_dim, hidden, window, padding=window // 2), nn.GELU(),
            nn.Conv1d(hidden, hidden, 1), nn.GELU(),
            nn.Conv1d(hidden, d, 1),
        )
        # bias so that sigmoid(b)^horizon = 0.5: trajectories start near P(y=1)=0.5
        p = 0.5 ** (1.0 / horizon)
        b_init = math.log(p / (1 - p))
        if shared:
            self.w0 = nn.Parameter(torch.randn(d) * 0.1)
            self.b0 = nn.Parameter(torch.tensor(b_init))
            self.w = nn.Parameter(torch.zeros(n_heads, d))     # per-user corrections
            self.b = nn.Parameter(torch.zeros(n_heads))
        else:
            self.w = nn.Parameter(torch.randn(n_heads, d) * 0.1)
            self.b = nn.Parameter(torch.full((n_heads,), b_init))

    def heads(self):
        if self.shared:
            return self.w0 + self.w, self.b0 + self.b
        return self.w, self.b

    def prior_penalty(self):
        if not self.shared:
            return torch.zeros((), device=self.w.device)
        return (self.w ** 2).sum() + (self.b ** 2).sum()

    def forward(self, obs):
        """obs [B, T, D] -> step rewards [B, T, H]."""
        phi = self.enc(obs.transpose(1, 2)).transpose(1, 2)
        w, b = self.heads()
        return phi @ w.T + b

    def log_p_good(self, obs, k):
        r = self(obs)
        rk = r.gather(2, k.view(-1, 1, 1).expand(-1, r.shape[1], 1)).squeeze(-1)
        return F.logsigmoid(rk).sum(1)


def nll(model, obs, k, y):
    lp1 = model.log_p_good(obs, k)
    return -(y * lp1 + (1 - y) * log1mexp(lp1)).mean()


def train(model, obs, rows, epochs=40, bs=512, lr=1e-3, single_head=False, min_steps=2000, prior=0.0,
          log=print):
    """rows: [M, 3] array of (traj_idx, user, y). obs: tensor [N, T, D] on device.
    Runs at least min_steps gradient steps so small label budgets aren't undertrained.
    prior: weight of the per-user correction penalty, on the scale of the summed NLL."""
    dev = obs.device
    idx = torch.as_tensor(rows[:, 0], dtype=torch.long, device=dev)
    k = torch.as_tensor(rows[:, 1], dtype=torch.long, device=dev)
    y = torch.as_tensor(rows[:, 2], dtype=torch.float32, device=dev)
    if single_head:
        k = torch.zeros_like(k)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    epochs = max(epochs, math.ceil(min_steps / math.ceil(len(y) / bs)))
    for ep in range(epochs):
        perm = torch.randperm(len(y), device=dev)
        tot = 0.0
        for s in range(0, len(y), bs):
            b = perm[s:s + bs]
            loss = nll(model, obs[idx[b]], k[b], y[b]) + prior * model.prior_penalty() / len(y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        if ep % max(1, epochs // 4) == 0 or ep == epochs - 1:
            log(f"  epoch {ep:3d}  nll {tot / len(y):.4f}")
    return model
