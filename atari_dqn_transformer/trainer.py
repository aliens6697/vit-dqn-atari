"""
trainer.py — training loop
============================
Double DQN + Dueling transformer agent.

USAGE:
    python trainer.py                   # auto-resume if checkpoint exists
    python trainer.py --no-resume       # start from scratch
    python trainer.py --envs 16
    python trainer.py --game ALE/Pong-v5
"""

import multiprocessing
multiprocessing.freeze_support()

if __name__ == "__main__":
    import argparse
    import time
    from pathlib import Path

    import numpy as np
    import torch

    from config import (NUM_ENVS, WARMUP_STEPS, TRAIN_FREQ, GRAD_STEPS, MAX_STEPS,
                        SAVE_LATEST, SAVE_SNAPSHOT, LOG_EVERY, TARGET_SYNC, GAME)
    from environment import VecAtariEnv
    from learner import TransformerAgent


    def train(game: str, resume: bool, num_envs: int):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device   : {device}")
        if device.type == "cpu":
            print("WARNING  : CUDA not available — running on CPU (will be very slow).")
        else:
            print(f"GPU      : {torch.cuda.get_device_name(0)}")
            print(f"VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"Game     : {game}")
        print(f"Num envs : {num_envs}")
        print()

        env   = VecAtariEnv(game, num_envs=num_envs)
        agent = TransformerAgent(env.n_actions, device)

        print(f"Actions  : {env.action_meanings()}")
        n_params = sum(p.numel() for p in agent.policy_net.parameters())
        print(f"Params   : {n_params:,}  ({n_params/1e6:.1f}M)")
        print()

        if resume:
            ckpt = Path("checkpoints/latest.pt")
            if ckpt.exists():
                agent.load(str(ckpt))
            else:
                print("No checkpoint found, starting fresh.")

        agent.compile_networks()

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
        last_log_step  = agent.steps
        last_log_time  = t0

        print(f"Training for {MAX_STEPS:,} steps...\n")

        step = agent.steps
        # Counter-based triggers — robust to num_envs > threshold.
        next_train  = step + TRAIN_FREQ
        next_sync   = step + TARGET_SYNC
        next_log    = step + LOG_EVERY
        next_save   = max(SAVE_LATEST, ((step // SAVE_LATEST) + 1) * SAVE_LATEST)
        next_snap   = max(SAVE_SNAPSHOT, ((step // SAVE_SNAPSHOT) + 1) * SAVE_SNAPSHOT)

        while step < MAX_STEPS:

            # 1. Batched action selection — eval mode so dropout is off during rollout
            agent.policy_net.eval()
            state_t = torch.from_numpy(states).to(device, non_blocking=True)
            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")
            ):
                q_values = agent.policy_net(state_t)   # (N, n_actions)
            agent.policy_net.train()

            random_mask = np.random.rand(num_envs) < agent.epsilon
            greedy      = q_values.argmax(dim=1).cpu().numpy()
            random_acts = np.random.randint(0, env.n_actions, size=num_envs)
            actions     = np.where(random_mask, random_acts, greedy)

            # 2. Step all envs
            next_states, rewards, dones, raw_rewards, ep_scores = env.step(actions)

            # 3. Store transitions
            for i in range(num_envs):
                agent.replay.push(states[i], actions[i], rewards[i],
                                  next_states[i], dones[i])

            episode_scores.extend(ep_scores)
            episode_count += len(ep_scores)
            states   = next_states
            step    += num_envs
            agent.steps = step

            # 4. Train — one round of GRAD_STEPS updates each TRAIN_FREQ env steps.
            #    Loop in case num_envs >= TRAIN_FREQ, so we don't skip rounds.
            while step >= next_train:
                for _ in range(GRAD_STEPS):
                    loss = agent.train_step()
                    if loss is not None:
                        total_loss += loss
                        loss_steps += 1
                next_train += TRAIN_FREQ

            # 5. Sync target
            if step >= next_sync:
                agent.sync_target()
                next_sync += TARGET_SYNC

            # 6. Decay epsilon
            agent.update_epsilon()

            # 7. Logging
            if step >= next_log:
                next_log += LOG_EVERY
                now      = time.time()
                avg_loss = total_loss / max(loss_steps, 1)
                fps      = (step - last_log_step) / max(now - last_log_time, 1e-6)
                last_log_step = step
                last_log_time = now
                if episode_count > 0:
                    recent = episode_scores[-20:]
                    avg    = np.mean(recent)
                    best   = np.max(recent)
                    print(
                        f"step {step:>9,} | "
                        f"eps {agent.epsilon:.3f} | "
                        f"ep {episode_count:>5} | "
                        f"avg(20) {avg:>6.1f} | "
                        f"best(20) {best:>5.0f} | "
                        f"loss {avg_loss:.4f} | "
                        f"fps {fps:.0f}"
                    )
                else:
                    print(
                        f"step {step:>9,} | "
                        f"eps {agent.epsilon:.3f} | "
                        f"ep     0 | "
                        f"avg(20)    --- | "
                        f"best(20)   --- | "
                        f"loss {avg_loss:.4f} | "
                        f"fps {fps:.0f}"
                    )
                total_loss = 0.0
                loss_steps = 0

            # 8. Checkpoints
            if step >= next_save:
                agent.save("checkpoints/latest.pt")
                next_save += SAVE_LATEST

            if step >= next_snap:
                agent.save(f"checkpoints/step_{step:08d}.pt")
                next_snap += SAVE_SNAPSHOT

        agent.save("checkpoints/latest.pt")
        env.close()
        print("\nTraining complete.")


    parser = argparse.ArgumentParser()
    parser.add_argument("--game",      default=GAME)
    parser.add_argument("--envs",      default=NUM_ENVS, type=int)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    train(args.game, args.resume, args.envs)
