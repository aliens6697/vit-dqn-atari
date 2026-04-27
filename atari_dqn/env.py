"""
env.py — Atari emulator wrapper
================================
Two classes:

  AtariEnv    — single env, used by play.py
  VecAtariEnv — N envs in separate processes, used by train.py

The key fix over the previous version:
  Before: Python for-loop over N envs (still sequential, GIL blocks threads)
  Now:    gymnasium AsyncVectorEnv — each env runs in its own OS process,
          all stepping truly in parallel, results gathered via pipes

Frame skip and frame stacking are handled via gymnasium wrappers
applied inside each worker process — cleaner and faster than doing
it manually in Python after the fact.
"""

import random
from collections import deque

import ale_py
import cv2
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

FRAME_SIZE  = 84
FRAME_STACK = 4
FRAME_SKIP  = 4


# ── Preprocessing helper (used by single env only) ────────────────────────────

def _preprocess(frame: np.ndarray) -> np.ndarray:
    """RGB (210,160,3) → grayscale float32 (84,84) in [0,1]."""
    gray    = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (FRAME_SIZE, FRAME_SIZE),
                         interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


# ── Single environment (for play.py) ─────────────────────────────────────────

class AtariEnv:
    def __init__(self, game="ALE/Breakout-v5", render_mode=None):
        gym.register_envs(ale_py)
        self.env = gym.make(
            game,
            frameskip=1,
            repeat_action_probability=0.0,
            render_mode=render_mode,
        )
        self.frame_buffer = deque(maxlen=2)
        self.stack        = deque(maxlen=FRAME_STACK)
        self.n_actions    = self.env.action_space.n
        self.game         = game

    def reset(self):
        obs, _ = self.env.reset()
        for _ in range(random.randint(1, 30)):
            obs, _, terminated, truncated, _ = self.env.step(0)
            if terminated or truncated:
                obs, _ = self.env.reset()
        frame = _preprocess(obs)
        for _ in range(FRAME_STACK):
            self.stack.append(frame)
        return self._state()

    def step(self, action):
        total_reward = 0.0
        self.frame_buffer.clear()
        for _ in range(FRAME_SKIP):
            obs, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward
            self.frame_buffer.append(obs)
            if terminated or truncated:
                break
        if len(self.frame_buffer) == 2:
            max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])
        else:
            max_frame = self.frame_buffer[-1]
        self.stack.append(_preprocess(max_frame))
        return self._state(), float(np.clip(total_reward, -1.0, 1.0)), \
               terminated or truncated, total_reward

    def action_meanings(self):
        return self.env.unwrapped.get_action_meanings()

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    def _state(self):
        return np.stack(self.stack, axis=0)


# ── Vectorized environment (for train.py) ────────────────────────────────────

def _make_env(game: str):
    """
    Factory function — called inside each worker process.
    Wraps the raw env with gymnasium's built-in Atari preprocessing:
      AtariPreprocessing  — grayscale, resize 84x84, frame skip 4, max-pool
      FrameStackObservation — stack 4 frames → (4, 84, 84)
    """
    def _init():
        gym.register_envs(ale_py)
        env = gym.make(game, frameskip=1, repeat_action_probability=0.0)
        env = AtariPreprocessing(
            env,
            noop_max=30,
            frame_skip=FRAME_SKIP,
            screen_size=FRAME_SIZE,
            grayscale_obs=True,
            grayscale_newaxis=False,   # shape: (84,84) not (84,84,1)
            scale_obs=True,            # uint8 → float32 /255
        )
        env = FrameStackObservation(env, FRAME_STACK)  # → (4,84,84)
        return env
    return _init


class VecAtariEnv:
    """
    N Atari environments running in separate OS processes via AsyncVectorEnv.

    All preprocessing (grayscale, resize, frame skip, frame stack) happens
    inside the worker processes — no Python loops in the main process.

    Usage:
        env    = VecAtariEnv("ALE/Breakout-v5", num_envs=16)
        states = env.reset()                              # (16, 4, 84, 84)
        states, rewards, dones, raw = env.step(actions)  # actions: (16,)
    """

    def __init__(self, game="ALE/Breakout-v5", num_envs=16):
        self.num_envs  = num_envs
        self.game      = game

        # Each env runs in its own process — true parallelism, bypasses GIL
        self._vec = gym.vector.AsyncVectorEnv(
            [_make_env(game) for _ in range(num_envs)]
        )
        self.n_actions = self._vec.single_action_space.n

        # Track per-env running rewards for episode scoring
        self._ep_rewards = np.zeros(num_envs, dtype=np.float32)

    def reset(self):
        """Returns (N, 4, 84, 84) float32."""
        obs, _ = self._vec.reset()
        self._ep_rewards[:] = 0.0
        return obs.astype(np.float32)

    def step(self, actions: np.ndarray):
        """
        All N envs step simultaneously in their worker processes.
        gymnasium handles auto-reset on episode end.

        Returns:
            states      (N, 4, 84, 84)  float32
            rewards     (N,)            float32  clipped [-1,+1]
            dones       (N,)            bool
            raw_rewards (N,)            float32  unclipped episode delta
        """
        obs, rewards, terminated, truncated, infos = self._vec.step(actions)

        dones = terminated | truncated

        # gymnasium auto-resets done envs — obs already contains the new
        # episode's first state for those envs

        raw_rewards = rewards.copy().astype(np.float32)
        self._ep_rewards += raw_rewards

        # Collect finished episode scores
        ep_scores = []
        for i in range(self.num_envs):
            if dones[i]:
                ep_scores.append(float(self._ep_rewards[i]))
                self._ep_rewards[i] = 0.0

        clipped = np.clip(rewards, -1.0, 1.0).astype(np.float32)
        return obs.astype(np.float32), clipped, dones, raw_rewards, ep_scores

    def action_meanings(self):
        # Make a temporary env just to read action meanings
        gym.register_envs(ale_py)
        tmp = gym.make(self.game)
        meanings = tmp.unwrapped.get_action_meanings()
        tmp.close()
        return meanings

    def close(self):
        self._vec.close()