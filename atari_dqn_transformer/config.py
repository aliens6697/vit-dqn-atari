"""
config.py — single source of truth
====================================
All hyperparameters in one place.
Both the environment and model import from here — no more mismatches.
"""

# ── Environment ───────────────────────────────────────────────────────────────
GAME        = "ALE/Breakout-v5"
FRAME_SIZE  = 84       # resize frames to 84×84
FRAME_STACK = 4        # number of consecutive frames stacked as state
FRAME_SKIP  = 4        # act every N emulator frames

# ── Transformer model ─────────────────────────────────────────────────────────
# patch_size 14 → 6×6=36 patches/frame, 144 tokens total over 4 frames.
# 576 tokens (patch_size 7) was too many for a small DQN to learn from — ViTs
# need orders of magnitude more samples than CNNs on Atari, and token count
# scales the data hunger. Fewer, larger patches = stronger inductive bias.
PATCH_SIZE  = 14       # was 7 → 36 patches/frame (was 144), 144 total tokens (was 576)
D_MODEL     = 128      # transformer embedding dimension
N_HEADS     = 4        # 4 heads × 32 dims each
N_LAYERS    = 4        # transformer encoder layers
D_FF        = 512      # feedforward hidden dim inside transformer
DROPOUT     = 0.0      # dropout off — hurts Q-learning by adding noise to value estimates

# ── Training ──────────────────────────────────────────────────────────────────
# Tuned to match cleanrl / SB3 / Rainbow proven Atari hyperparameters.
# Previous values had Q-values oscillating: target_sync=1000 with 16 envs and
# 2 grad steps every 4 env steps → only ~8 grad updates between target syncs
# (canonical: ~250–2500). Targets non-stationary → no learning.
NUM_ENVS        = 16
REPLAY_CAPACITY = 500_000      # 500K fits ~14GB RAM (uint8 storage)
BATCH_SIZE      = 256          # was 64 — saturates GPU; transformer kernels need bulk
GAMMA           = 0.99
LR              = 1e-4         # was 2.5e-4 — canonical Adam-DQN value (cleanrl/SB3)
EPS_START       = 1.0
EPS_END         = 0.01
EPS_DECAY_STEPS = 1_000_000    # env steps to reach EPS_END
TARGET_SYNC     = 8_000        # was 1000 — gives ~2000 grad updates between syncs
WARMUP_STEPS    = 80_000       # was 10000 — cleanrl uses 80k, SB3 uses 100k
TRAIN_FREQ      = 4            # train every 4 env steps
GRAD_STEPS      = 1            # was 2 — cleanrl/SB3/Mnih all use 1 grad per train
MAX_STEPS       = 50_000_000
SAVE_LATEST     = 50_000
SAVE_SNAPSHOT   = 1_000_000
LOG_EVERY       = 10_000

# ── Viewer ────────────────────────────────────────────────────────────────────
VIEWER_PORT  = 5000
VIEWER_FPS   = 30
VIEWER_SCALE = 6
