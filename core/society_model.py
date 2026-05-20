import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from collections import deque


EVENT_TYPES = [
    "TASK_POSTED",
    "BID_WON",
    "COALITION_FORMED",
    "TASK_SUCCESS",
    "TASK_FAIL",
    "AGENT_DIED",
    "AGENT_BORN",
]

EVENT_TO_IDX = {e: i for i, e in enumerate(EVENT_TYPES)}
N_EVENTS     = len(EVENT_TYPES)
WINDOW_SIZE  = 1000
D_MODEL      = 64
N_HEADS      = 4
N_LAYERS     = 2
CTX          = 32
CURIOSITY_BONUS = 5


@dataclass
class SocietyEvent:
    tick        : int
    event_type  : str
    agent_id    : int  = -1
    quality     : float = 0.0
    tokens      : int   = 0
    success     : bool  = False

    def to_tensor(self) -> torch.Tensor:
        type_oh  = F.one_hot(
            torch.tensor(EVENT_TO_IDX[self.event_type]), N_EVENTS
        ).float()
        scalars  = torch.tensor([
            self.agent_id / 10.0,
            self.quality,
            self.tokens  / 200.0,
            float(self.success),
        ])
        return torch.cat([type_oh, scalars])


INPUT_DIM = N_EVENTS + 4


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.scale * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-8)


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.norm2 = RMSNorm(d)
        self.attn  = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ffn   = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.GELU(),
            nn.Linear(d * 4, d),
        )

    def forward(self, x):
        n = self.norm1(x)
        a, _ = self.attn(n, n, n, need_weights=False)
        x = x + a
        x = x + self.ffn(self.norm2(x))
        return x


class SocietyModel(nn.Module):
    def __init__(self,
                 d_model    : int = D_MODEL,
                 n_heads    : int = N_HEADS,
                 n_layers   : int = N_LAYERS,
                 ctx        : int = CTX,
                 window_size: int = WINDOW_SIZE,
                 lr         : float = 1e-3):
        super().__init__()
        self.ctx         = ctx
        self.window_size = window_size

        self.input_proj  = nn.Linear(INPUT_DIM, d_model)
        self.blocks      = nn.ModuleList([Block(d_model, n_heads)
                                          for _ in range(n_layers)])
        self.norm        = RMSNorm(d_model)
        self.event_head  = nn.Linear(d_model, N_EVENTS)
        self.outcome_head= nn.Linear(d_model, 1)

        self._window     : deque[SocietyEvent] = deque(maxlen=window_size)
        self._optimizer  = torch.optim.Adam(self.parameters(), lr=lr)
        self._train_steps= 0
        self._total_loss = 0.0
        self.curiosity_bonuses: dict[int, int] = {}

    def observe(self, event: SocietyEvent):
        self._window.append(event)
        if len(self._window) >= self.ctx + 1:
            self._train_step()

    def _train_step(self):
        window = list(self._window)
        start  = random.randint(0, len(window) - self.ctx - 1)
        seq    = window[start: start + self.ctx]
        target = window[start + self.ctx]

        x = torch.stack([e.to_tensor() for e in seq]).unsqueeze(0)
        t_event   = torch.tensor([EVENT_TO_IDX[target.event_type]])
        t_outcome = torch.tensor([[target.quality]])

        pred_event, pred_outcome = self._forward(x)

        loss = (
            F.cross_entropy(pred_event, t_event)
            + F.mse_loss(pred_outcome, t_outcome)
        )

        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self._optimizer.step()

        self._train_steps += 1
        self._total_loss  += loss.item()

    def _forward(self, x: torch.Tensor):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        last = x[:, -1, :]
        return self.event_head(last), torch.sigmoid(self.outcome_head(last))

    def predict(self, recent_events: list[SocietyEvent]):
        if len(recent_events) < self.ctx:
            return None, None
        seq = recent_events[-self.ctx:]
        x   = torch.stack([e.to_tensor() for e in seq]).unsqueeze(0)
        with torch.no_grad():
            pred_event, pred_outcome = self._forward(x)
        event_idx    = pred_event.argmax(dim=-1).item()
        event_type   = EVENT_TYPES[event_idx]
        outcome_prob = pred_outcome.squeeze().item()
        return event_type, outcome_prob

    def curiosity_check(self, actual_event: SocietyEvent,
                        balances: dict[int, int]) -> dict[int, int]:
        if len(self._window) < self.ctx + 1:
            return balances
        recent = list(self._window)[-(self.ctx + 1):-1]
        pred_type, pred_outcome = self.predict(recent)
        if pred_type is None:
            return balances
        surprised = (pred_type != actual_event.event_type)
        if surprised and actual_event.agent_id >= 0:
            aid = actual_event.agent_id
            balances[aid] = balances.get(aid, 0) + CURIOSITY_BONUS
            self.curiosity_bonuses[aid] = (
                self.curiosity_bonuses.get(aid, 0) + CURIOSITY_BONUS
            )
        return balances

    def snapshot(self) -> dict:
        avg_loss = (self._total_loss / self._train_steps
                    if self._train_steps > 0 else 0.0)
        return {
            "train_steps"      : self._train_steps,
            "avg_loss"         : round(avg_loss, 4),
            "window_size"      : len(self._window),
            "total_curiosity"  : sum(self.curiosity_bonuses.values()),
        }


if __name__ == "__main__":
    print("=== SOCIETY MODEL SMOKE TEST ===")

    model = SocietyModel(d_model=64, n_heads=4, n_layers=2, ctx=32)

    print("  feeding 200 synthetic events...")
    event_types = EVENT_TYPES
    for i in range(200):
        ev = SocietyEvent(
            tick       = i,
            event_type = random.choice(event_types),
            agent_id   = random.randint(0, 9),
            quality    = random.uniform(0.2, 1.0),
            tokens     = random.randint(0, 200),
            success    = random.random() > 0.5,
        )
        model.observe(ev)

    snap = model.snapshot()
    print(f"  train steps : {snap['train_steps']}")
    print(f"  avg loss    : {snap['avg_loss']}")
    print(f"  window size : {snap['window_size']}")
    assert snap["train_steps"] > 0, "No training steps taken"
    assert snap["avg_loss"] < 10.0, "Loss too high"

    print("\n  testing prediction...")
    recent = list(model._window)[-32:]
    pred_type, pred_outcome = model.predict(recent)
    assert pred_type in EVENT_TYPES, f"Invalid predicted event: {pred_type}"
    assert 0.0 <= pred_outcome <= 1.0, f"Outcome out of range: {pred_outcome}"
    print(f"  predicted next event : {pred_type}")
    print(f"  predicted outcome    : {pred_outcome:.3f}")

    print("\n  testing curiosity bonus...")
    balances = {i: 100 for i in range(10)}
    surprise_event = SocietyEvent(
        tick=201, event_type="AGENT_DIED",
        agent_id=3, quality=0.0, tokens=0, success=False,
    )
    balances = model.curiosity_check(surprise_event, balances)
    print(f"  balances after curiosity check: {balances}")
    print(f"  curiosity bonuses: {model.curiosity_bonuses}")

    print(f"\n  snapshot: {model.snapshot()}")
    print("\n  RESULT: PASS")