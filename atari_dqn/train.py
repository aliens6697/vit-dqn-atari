"""
train.py — training loop (vectorized, async)
=============================================
Runs N Atari environments in separate OS processes to saturate the GPU.

USAGE:
    python train.py                        # auto-resume if checkpoint exists
    python train.py --no-resume            # force start from scratch
    python train.py --envs 32             # more parallel envs
    python train.py --game ALE/Pong-v5    # different game

CHECKPOINTS:
    checkpoints/latest.pt                  # saved every 50k steps (overwritten)
    checkpoints/step_XXXXXXXX.pt           # kept every 500k steps

TUNING --envs:
    Watch GPU utilization with: nvidia-smi dmon -s ut
    Increase --envs until SM% stops climbing (typically 32–64 for a 3080)

WHAT TO EXPECT:
    0  – 10k    warming up replay buffer
    10k – 200k  loss noisy, score near 0, epsilon high
    200k – 1M   agent starts surviving longer
    1M – 5M     steady score improvement
    5M+         strong agent, score 100–400+
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from agent import DQNAgent, TARGET_SYNC
from env import VecAtariEnv

# ── Config ────────────────────────────────────────────────────────────────────
NUM_ENVS      = 40
WARMUP_STEPS  = 10_000
TRAIN_FREQ    = 4
MAX_STEPS     = 20_000_000
SAVE_LATEST   = 50_000
SAVE_SNAPSHOT = 500_000
LOG_EVERY     = 10_000


def train(game: str, resume: bool, num_envs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device   : {device}")
    print(f"Game     : {game}")
    print(f"Num envs : {num_envs}")
    print()

    env   = VecAtariEnv(game, num_envs=num_envs)
    agent = DQNAgent(env.n_actions, device)

    print(f"Actions  : {env.action_meanings()}")
    print()

    if resume:
        ckpt = Path("checkpoints/latest.pt")
        if ckpt.exists():
            agent.load(str(ckpt))
        else:
            print("No checkpoint found, starting fresh.")

    # ── Warm-up ───────────────────────────────────────────────────────────────
    print(f"Warming up — collecting {WARMUP_STEPS:,} transitions...")
    states    = env.reset()
    collected = 0

    while collected < WARMUP_STEPS:
        actions = np.random.randint(0, env.n_actions, size=num_envs)
        next_states, rewards, dones, raw_rewards, _ = env.step(actions)

        for i in range(num_envs):
            agent.replay.push(states[i], actions[i], rewards[i],
                              next_states[i], dones[i])
        states     = next_states
        collected += num_envs

    print(f"Replay buffer ready: {len(agent.replay):,} transitions\n")

    # ── Training loop ─────────────────────────────────────────────────────────
    states         = env.reset()
    episode_scores = []
    episode_count  = 0
    total_loss     = 0.0
    loss_steps     = 0
    t0             = time.time()

    print(f"Training for {MAX_STEPS:,} steps...\n")

    step = agent.steps

    while step < MAX_STEPS:

        # 1. Batched action selection — one GPU forward pass for all N envs
        state_t = torch.tensor(states, dtype=torch.float32).to(device)
        with torch.no_grad():
            q_values = agent.policy_net(state_t)   # (N, n_actions)

        # Epsilon-greedy across all envs
        random_mask = np.random.rand(num_envs) < agent.epsilon
        greedy      = q_values.argmax(dim=1).cpu().numpy()
        random_acts = np.random.randint(0, env.n_actions, size=num_envs)
        actions     = np.where(random_mask, random_acts, greedy)

        # 2. Step all envs in parallel (async worker processes)
        next_states, rewards, dones, raw_rewards, ep_scores = env.step(actions)

        # 3. Store N transitions
        for i in range(num_envs):
            agent.replay.push(states[i], actions[i], rewards[i],
                              next_states[i], dones[i])

        episode_scores.extend(ep_scores)
        episode_count += len(ep_scores)
        states  = next_states
        step   += num_envs
        agent.steps = step

        # 4. Train
        if step % TRAIN_FREQ == 0:
            loss = agent.train_step()
            if loss is not None:
                total_loss += loss
                loss_steps += 1

        # 5. Sync target network
        if step % TARGET_SYNC == 0:
            agent.sync_target()

        # 6. Decay epsilon
        agent.update_epsilon()

        # 7. Logging
        if step % LOG_EVERY == 0 and episode_count > 0:
            recent   = episode_scores[-20:]
            avg      = np.mean(recent)
            best     = np.max(recent)
            avg_loss = total_loss / max(loss_steps, 1)
            fps      = step / max(time.time() - t0, 1)
            print(
                f"step {step:>9,} | "
                f"eps {agent.epsilon:.3f} | "
                f"ep {episode_count:>5} | "
                f"avg(20) {avg:>6.1f} | "
                f"best(20) {best:>5.0f} | "
                f"loss {avg_loss:.4f} | "
                f"fps {fps:.0f}"
            )
            total_loss = 0.0
            loss_steps = 0

        # 8. Checkpoints
        if step % SAVE_LATEST == 0 and step > 0:
            agent.save("checkpoints/latest.pt")

        if step % SAVE_SNAPSHOT == 0 and step > 0:
            agent.save(f"checkpoints/step_{step:08d}.pt")

    agent.save("checkpoints/latest.pt")
    env.close()
    print("\nTraining complete.")


if __name__ == "__main__":
    # Required on Windows for multiprocessing
    import multiprocessing
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument("--game",      default="ALE/Breakout-v5")
    parser.add_argument("--envs",      default=NUM_ENVS, type=int,
                        help=f"Parallel envs (default {NUM_ENVS})")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Start from scratch")
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    train(args.game, args.resume, args.envs)