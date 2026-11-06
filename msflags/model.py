"""Fully-convolutional residual network with three heads.

- mine head:   per-cell logit of P(mine | public board). Trained supervised
               (exact solver marginals / true layout), no RL needed.
- policy head: per-cell move logits. Built as
                   policy = beta * mine_logit + residual(features)
               with a zero-initialised residual. Bomb actions score 5x5 expected
               mine totals with a learned scale and residual; both action types
               are normalised jointly.
- value head:  global-pooled -> tanh, expected outcome for the player to move.

Being fully convolutional with global pooling, the same weights run on any
board size.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .game import NUM_PLANES, Config, encode


@dataclass
class NetConfig:
    channels: int = 64
    blocks: int = 6
    policy_beta: float = 3.0
    planes: int = NUM_PLANES
    policy_gamma: float = 3.0
    bomb_threshold: float = 3.0

    def __post_init__(self):
        if self.planes != NUM_PLANES:
            raise ValueError(f"networks require {NUM_PLANES} input planes")


class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(c)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class Net(nn.Module):
    def __init__(self, cfg: NetConfig = NetConfig()):
        super().__init__()
        self.cfg = cfg
        c = cfg.channels
        self.stem = nn.Sequential(nn.Conv2d(cfg.planes, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU())
        self.body = nn.Sequential(*[ResBlock(c) for _ in range(cfg.blocks)])
        self.mine = nn.Sequential(nn.Conv2d(c, 32, 1), nn.ReLU(), nn.Conv2d(32, 1, 1))
        self.policy_res = nn.Sequential(nn.Conv2d(c, 32, 1), nn.ReLU(), nn.Conv2d(32, 1, 1))
        nn.init.zeros_(self.policy_res[-1].weight)
        nn.init.zeros_(self.policy_res[-1].bias)
        self.beta = nn.Parameter(torch.tensor(cfg.policy_beta))
        self.bomb_res = nn.Sequential(nn.Conv2d(c, 32, 1), nn.ReLU(), nn.Conv2d(32, 1, 1))
        nn.init.zeros_(self.bomb_res[-1].weight)
        nn.init.zeros_(self.bomb_res[-1].bias)
        self.gamma = nn.Parameter(torch.tensor(cfg.policy_gamma))
        self.value = nn.Sequential(nn.Linear(2 * c, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor):
        """Return joint action logits [B,2N], value [B], and mine logits [B,N]."""
        f = self.body(self.stem(x))
        mine = self.mine(f).flatten(1)
        policy = self.beta * mine + self.policy_res(f).flatten(1)
        unknown = x[:, 0].flatten(1) > 0.5
        policy = policy.masked_fill(~unknown, -1e9)
        # Sum only still-hidden mines; zero padding clips windows at edges.
        expected = F.avg_pool2d(
            torch.sigmoid(mine).view_as(x[:, :1]) * x[:, :1],
            5, stride=1, padding=2, divisor_override=1,
        ).flatten(1)
        bomb = self.gamma * (expected - self.cfg.bomb_threshold) + self.bomb_res(f).flatten(1)
        available = x[:, 18].flatten(1) > 0.5
        bomb_centres = F.max_pool2d(x[:, :1], 5, stride=1, padding=2).flatten(1) > 0.5
        bomb = bomb.masked_fill(~(bomb_centres & available), -1e9)
        policy = torch.cat((policy, bomb), dim=1)
        board = x[:, 17:18]  # ones on the board
        pooled = torch.cat([(f * board).sum((2, 3)) / board.sum((2, 3)), f.amax((2, 3))], 1)
        value = torch.tanh(self.value(pooled)).squeeze(1)
        return policy, value, mine


def save(net: Net, path: str, **extra) -> None:
    torch.save({"net_cfg": asdict(net.cfg), "state": net.state_dict(), **extra}, path)


def load(path: str, device: str | torch.device = "cpu") -> tuple[Net, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    config = {key: value for key, value in ck["net_cfg"].items() if key != "rules"}
    if ck["state"]["stem.0.weight"].shape[1] != NUM_PLANES:
        raise ValueError(f"checkpoint requires {NUM_PLANES} input planes")
    net = Net(NetConfig(**config)).to(device)
    net.load_state_dict(ck["state"])
    net.eval()
    return net, ck


class Evaluator:
    """Batched numpy-in/numpy-out inference."""

    def __init__(self, net: Net, device: str | torch.device):
        self.net = net.to(device).eval()
        self.device = torch.device(device)

    @torch.inference_mode()
    def __call__(self, states: list[tuple], cfg: Config):
        """Evaluate states containing (board, scores, to_move, bombs)."""
        encoded = [encode(board, scores, to_move, cfg, bombs)
                   for board, scores, to_move, bombs in states]
        x = torch.from_numpy(np.stack(encoded)).to(self.device)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                            enabled=self.device.type == "cuda"):
            logits, value, mine = self.net(x)
        policy = torch.softmax(logits.float(), 1)
        return policy.cpu().numpy(), value.float().cpu().numpy(), torch.sigmoid(mine.float()).cpu().numpy()
