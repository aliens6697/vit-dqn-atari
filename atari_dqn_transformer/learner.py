"""
learner.py — DQN agent with transformer network
=================================================
Uses Double DQN + Dueling architecture for stable learning.
Double DQN: policy net selects action, target net evaluates it.
            Eliminates Q-value overestimation that causes divergence in vanilla DQN.
"""

import random
from contextlib import nullcontext as _nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import (REPLAY_CAPACITY, BATCH_SIZE, GAMMA, LR,
                    EPS_START, EPS_END, EPS_DECAY_STEPS, TARGET_SYNC, GRAD_STEPS)
from network import TransformerQNetwork


# ── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Pre-allocated numpy ring buffer — O(1) insert and O(1) sample."""

    def __init__(self, capacity=REPLAY_CAPACITY):
        self.capacity = capacity
        self.pos      = 0
        self.size     = 0

        from config import FRAME_STACK, FRAME_SIZE
        self.states      = np.empty((capacity, FRAME_STACK, FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)
        self.next_states = np.empty((capacity, FRAME_STACK, FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)
        self.actions     = np.empty((capacity,), dtype=np.int32)
        self.rewards     = np.empty((capacity,), dtype=np.float32)
        self.dones       = np.empty((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        self.states[self.pos]      = (state * 255).astype(np.uint8)
        self.next_states[self.pos] = (next_state * 255).astype(np.uint8)
        self.actions[self.pos]     = int(action)
        self.rewards[self.pos]     = float(reward)
        self.dones[self.pos]       = float(done)
        self.pos  = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size=BATCH_SIZE):
        """Sample a batch as pinned uint8/int tensors. Caller moves to GPU
        non-blocking and converts to float there — much faster than a CPU-side
        float32 cast plus a 4× larger H2D copy."""
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.states[idx]).pin_memory(),       # uint8
            torch.from_numpy(self.actions[idx]).long().pin_memory(),
            torch.from_numpy(self.rewards[idx]).pin_memory(),
            torch.from_numpy(self.next_states[idx]).pin_memory(),  # uint8
            torch.from_numpy(self.dones[idx]).pin_memory(),
        )

    def __len__(self):
        return self.size


# ── Agent ─────────────────────────────────────────────────────────────────────

class TransformerAgent:
    def __init__(self, n_actions: int, device: torch.device):
        self.n_actions = n_actions
        self.device    = device
        self.steps     = 0
        self.epsilon   = EPS_START

        self.policy_net = TransformerQNetwork(n_actions).to(device)
        self.target_net = TransformerQNetwork(n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self._use_amp = device.type == "cuda"

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=LR,
            eps=1.5e-4,
            weight_decay=0.0,
        )

        self.replay = ReplayBuffer()

    def compile_networks(self):
        # torch.compile needs Triton, which is Linux-only by default. On
        # Windows we skip it and rely on bf16 + cudnn.benchmark for speed.
        # If you install triton-windows, flip USE_COMPILE to True.
        USE_COMPILE = False
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            if USE_COMPILE:
                self.policy_net = torch.compile(self.policy_net, mode="reduce-overhead")
                self.target_net = torch.compile(self.target_net, mode="reduce-overhead")

    # ── Action selection ──────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        self.policy_net.eval()
        state_t = torch.tensor(state, dtype=torch.float32) \
                       .unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.policy_net(state_t).argmax(dim=1).item()
        self.policy_net.train()
        return action

    def update_epsilon(self):
        progress     = min(1.0, self.steps / EPS_DECAY_STEPS)
        self.epsilon = EPS_START + (EPS_END - EPS_START) * progress

    # ── Training step (Double DQN) ────────────────────────────────────────────

    def train_step(self) -> float | None:
        if len(self.replay) < BATCH_SIZE:
            return None

        s_u8, actions, rewards, ns_u8, dones = self.replay.sample()
        # H2D as uint8, then dequantize on GPU — ¼ the bandwidth of float32.
        s_u8    = s_u8.to(self.device, non_blocking=True)
        ns_u8   = ns_u8.to(self.device, non_blocking=True)
        actions = actions.to(self.device, non_blocking=True)
        rewards = rewards.to(self.device, non_blocking=True)
        dones   = dones.to(self.device, non_blocking=True)
        states      = s_u8.float().mul_(1.0 / 255.0)
        next_states = ns_u8.float().mul_(1.0 / 255.0)

        self.policy_net.train()

        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) \
            if self._use_amp else _nullcontext()

        with amp_ctx:
            q_values = self.policy_net(states) \
                           .gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
                next_q       = self.target_net(next_states).gather(1, next_actions).squeeze(1)
                targets      = rewards + GAMMA * next_q * (1.0 - dones)

            loss = nn.functional.huber_loss(q_values, targets)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        return loss.item()

    def sync_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # ── Checkpoint ────────────────────────────────────────────────────────────

    @staticmethod
    def _strip_compiled(state_dict: dict) -> dict:
        return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "steps":      self.steps,
            "epsilon":    self.epsilon,
            "policy_net": self._strip_compiled(self.policy_net.state_dict()),
            "target_net": self._strip_compiled(self.target_net.state_dict()),
            "optimizer":  self.optimizer.state_dict(),
        }, path)
        print(f"  [saved] {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.steps   = ckpt["steps"]
        self.epsilon = ckpt["epsilon"]
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        print(f"  [loaded] step={self.steps:,}  eps={self.epsilon:.3f}")
