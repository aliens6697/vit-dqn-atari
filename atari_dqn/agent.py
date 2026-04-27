"""
agent.py — DQN agent
=====================
Owns everything about learning:
    - Replay buffer      (stores experience, samples random batches)
    - Policy network     (the model we're training)
    - Target network     (frozen copy, updated every N steps)
    - Epsilon schedule   (exploration: starts random, becomes greedy)
    - Training step      (Bellman update via Huber loss)
    - Save / load        (checkpoints)
"""

import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import DQN

# ── Hyperparameters (DeepMind 2015) ──────────────────────────────────────────
REPLAY_CAPACITY  = 100_000   # paper: 1M — reduced to fit typical RAM
BATCH_SIZE       = 32
GAMMA            = 0.99      # discount factor
LR               = 2.5e-4
EPS_START        = 1.0
EPS_END          = 0.1
EPS_DECAY_STEPS  = 1_000_000 # anneal epsilon over 1M steps
TARGET_SYNC      = 10_000    # copy policy → target every N steps


# ── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Stores (state, action, reward, next_state, done) tuples.
    States saved as uint8 to save RAM — a float32 buffer of 1M transitions
    would need ~28 GB; uint8 needs ~7 GB, 100k needs ~700 MB.
    """

    def __init__(self, capacity=REPLAY_CAPACITY):
        self.buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buf.append((
            (state * 255).astype(np.uint8),
            int(action),
            float(reward),
            (next_state * 255).astype(np.uint8),
            bool(done),
        ))

    def sample(self, batch_size=BATCH_SIZE):
        batch = random.sample(self.buf, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.stack(states),      dtype=torch.float32) / 255.0,
            torch.tensor(actions,               dtype=torch.long),
            torch.tensor(rewards,               dtype=torch.float32),
            torch.tensor(np.stack(next_states), dtype=torch.float32) / 255.0,
            torch.tensor(dones,                 dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buf)


# ── Agent ─────────────────────────────────────────────────────────────────────

class DQNAgent:
    def __init__(self, n_actions: int, device: torch.device):
        self.n_actions = n_actions
        self.device    = device
        self.steps     = 0
        self.epsilon   = EPS_START

        # Two identical networks — policy is trained, target is frozen snapshot
        self.policy_net = DQN(n_actions).to(device)
        self.target_net = DQN(n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.replay    = ReplayBuffer()

    # ── Action selection ──────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray) -> int:
        """
        Epsilon-greedy:
            with prob epsilon  → random action  (explore)
            with prob 1-epsilon → argmax Q       (exploit)
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        state_t = torch.tensor(state, dtype=torch.float32) \
                       .unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.policy_net(state_t).argmax(dim=1).item()

    def update_epsilon(self):
        """Linear decay from EPS_START → EPS_END over EPS_DECAY_STEPS."""
        progress     = min(1.0, self.steps / EPS_DECAY_STEPS)
        self.epsilon = EPS_START + (EPS_END - EPS_START) * progress

    # ── Training step ─────────────────────────────────────────────────────────

    def train_step(self) -> float | None:
        """
        One gradient update using a random batch from replay buffer.
        Returns loss value, or None if buffer not ready yet.

        Bellman target:
            Q_target(s,a) = r + γ * max_a' Q_target(s', a')   if not done
            Q_target(s,a) = r                                  if done
        """
        if len(self.replay) < BATCH_SIZE:
            return None

        states, actions, rewards, next_states, dones = \
            [t.to(self.device) for t in self.replay.sample()]

        # Q(s, a) — what our policy network predicts for the actions we took
        q_values = self.policy_net(states) \
                       .gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target: r + γ * max Q'(s', a')  using frozen target network
        with torch.no_grad():
            next_q   = self.target_net(next_states).max(1)[0]
            targets  = rewards + GAMMA * next_q * (1.0 - dones)

        # Huber loss (less sensitive to outliers than MSE)
        loss = nn.functional.huber_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        return loss.item()

    def sync_target(self):
        """Copy policy weights → target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "steps":       self.steps,
            "epsilon":     self.epsilon,
            "policy_net":  self.policy_net.state_dict(),
            "target_net":  self.target_net.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
        }, path)
        print(f"  [saved] {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.steps      = ckpt["steps"]
        self.epsilon    = ckpt["epsilon"]
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        print(f"  [loaded] step={self.steps:,}  eps={self.epsilon:.3f}")
