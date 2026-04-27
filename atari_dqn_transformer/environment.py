"""
environment.py — Atari emulator wrapper
========================================
Identical structure to the DQN project but imports everything
from config.py so frame stack / frame size stay in sync with the model.

Two classes:
  AtariEnv    — single env  (used by viewer.py)
  VecAtariEnv — N async envs (used by trainer.py)
"""

import random
from collections import deque

import ale_py
import cv2
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

from config import FRAME_SIZE, FRAME_STACK, FRAME_SKIP


# ── Preprocessing (single env only) ──────────────────────────────────────────

def _preprocess(frame: np.ndarray) -> np.ndarray:
    """RGB (210,160,3) → grayscale float32 (84,84) in [0,1]."""
    gray    = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (FRAME_SIZE, FRAME_SIZE),
                         interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


# ── Single environment ────────────────────────────────────────────────────────

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
        return np.stack(self.stack, axis=0)   # (FRAME_STACK, 84, 84)


# ── Vectorized environment ────────────────────────────────────────────────────

class EpisodicLifeAndFireReset(gym.Wrapper):
    """Standard Atari training wrapper, applied OUTSIDE AtariPreprocessing.

    Two jobs:
      1. EpisodicLife: report `terminated=True` on every life loss so the
         agent gets a strong negative signal for dying, but DON'T actually
         reset the ALE — the next reset() just FIREs to launch the next ball,
         preserving remaining lives. Only when the real game ends (game_over)
         do we fully reset the underlying env.
      2. FireReset: for games like Breakout the ball won't launch unless
         FIRE (action 1) is pressed. Press it after every reset so the
         agent isn't stuck in a frozen state.
    """

    def __init__(self, env):
        super().__init__(env)
        meanings = env.unwrapped.get_action_meanings()
        self._fire_action = meanings.index("FIRE") if "FIRE" in meanings else None
        self._lives = 0
        self._real_done = True

    def _ale_lives(self):
        return self.env.unwrapped.ale.lives()

    def _fire(self):
        if self._fire_action is None:
            return None, False, False
        obs, _, terminated, truncated, _ = self.env.step(self._fire_action)
        return obs, terminated, truncated

    def reset(self, **kwargs):
        if self._real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # No-op step to advance past the life-loss frame without
            # resetting the ALE — preserves remaining lives.
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)

        if self._fire_action is not None:
            fire_obs, term, trunc = self._fire()
            if fire_obs is not None:
                obs = fire_obs
            if term or trunc:
                obs, info = self.env.reset(**kwargs)
                fire_obs, _, _ = self._fire()
                if fire_obs is not None:
                    obs = fire_obs

        self._lives = self._ale_lives()
        self._real_done = False
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._real_done = terminated or truncated
        lives = self._ale_lives()
        if 0 < lives < self._lives:
            terminated = True
        self._lives = lives
        info = dict(info)
        info["real_done"] = self._real_done
        return obs, reward, terminated, truncated, info


def _make_env(game: str, startup_delay: float = 0.0):
    """Worker process factory — preprocessing runs inside each process.
    startup_delay staggers initialization so all envs don't spike the CPU at once."""
    def _init():
        if startup_delay > 0:
            import time
            time.sleep(startup_delay)
        gym.register_envs(ale_py)
        env = gym.make(game, frameskip=1, repeat_action_probability=0.0)
        env = AtariPreprocessing(
            env,
            noop_max=30,
            frame_skip=FRAME_SKIP,
            screen_size=FRAME_SIZE,
            grayscale_obs=True,
            grayscale_newaxis=False,
            scale_obs=True,
        )
        env = EpisodicLifeAndFireReset(env)
        env = FrameStackObservation(env, FRAME_STACK)
        return env
    return _init


class VecAtariEnv:
    """N Atari envs running in separate OS processes (true parallelism)."""

    def __init__(self, game="ALE/Breakout-v5", num_envs=16):
        self.num_envs    = num_envs
        self.game        = game
        # Stagger startup by 0.1s per env so they don't all spike the CPU at once
        self._vec        = gym.vector.AsyncVectorEnv(
            [_make_env(game, startup_delay=i * 0.1) for i in range(num_envs)]
        )
        self.n_actions   = self._vec.single_action_space.n
        self._ep_rewards = np.zeros(num_envs, dtype=np.float32)

    def reset(self):
        obs, _ = self._vec.reset()
        self._ep_rewards[:] = 0.0
        return obs.astype(np.float32)

    def step(self, actions: np.ndarray):
        obs, rewards, terminated, truncated, infos = self._vec.step(actions)
        dones       = terminated | truncated
        raw_rewards = rewards.copy().astype(np.float32)
        self._ep_rewards += raw_rewards

        # `real_done` (full game over) comes from EpisodicLifeAndFireReset.
        # Vec env autoreset on `done` (which here is life-loss-or-game-over) is
        # fine for training; we only flush the score on a real game over.
        real_done = infos.get("real_done", dones) if isinstance(infos, dict) else dones
        real_done = np.asarray(real_done, dtype=bool)

        ep_scores = []
        for i in range(self.num_envs):
            if real_done[i]:
                ep_scores.append(float(self._ep_rewards[i]))
                self._ep_rewards[i] = 0.0
            elif dones[i]:
                # Life lost but game continues — keep accumulating across the auto-reset.
                pass

        clipped = np.sign(rewards).astype(np.float32)
        return obs.astype(np.float32), clipped, dones, raw_rewards, ep_scores

    def action_meanings(self):
        gym.register_envs(ale_py)
        tmp      = gym.make(self.game, frameskip=1, repeat_action_probability=0.0)
        meanings = tmp.unwrapped.get_action_meanings()
        tmp.close()
        return meanings

    def close(self):
        self._vec.close()
