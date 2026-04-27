"""
play.py — live viewer
======================
Streams the game to your browser in real time.
Works with any checkpoint — or with no model at all (random agent).

USAGE:
    python play.py                                    # random agent
    python play.py --model checkpoints/latest.pt      # trained agent
    python play.py --model checkpoints/latest.pt --eps 0.05  # 5% random

THEN OPEN:
    http://localhost:5000

The game runs in a background thread.
The browser receives a live MJPEG stream via the /stream endpoint.
Score and step count update every 500ms via /stats.
"""

import argparse
import threading
import time

import cv2
import numpy as np
import torch
from flask import Flask, Response, jsonify, render_template_string

from agent import DQNAgent
from env import AtariEnv

# ── Config ────────────────────────────────────────────────────────────────────
GAME   = "ALE/Breakout-v5"
FPS    = 30
SCALE  = 6    # upscale 160×210 → 480×630 so it's visible in browser

# ── Shared state (game thread writes, Flask thread reads) ─────────────────────
_latest_frame: bytes | None = None
_stats = {"episode": 1, "score": 0, "step": 0, "epsilon": 1.0, "mode": "random"}
_lock  = threading.Lock()


# ── Game loop ─────────────────────────────────────────────────────────────────

def game_loop(model_path: str | None, epsilon: float):
    global _latest_frame, _stats

    env   = AtariEnv(GAME, render_mode="rgb_array")
    agent = DQNAgent(env.n_actions, torch.device("cpu"))

    if model_path:
        agent.load(model_path)
        agent.epsilon = epsilon
        mode = f"model  eps={epsilon:.2f}"
    else:
        agent.epsilon = 1.0    # fully random
        mode = "random"

    episode = 1

    while True:
        state        = env.reset()
        total_reward = 0.0
        step         = 0

        while True:
            # ── Agent picks action ────────────────────────────────────────────
            action = agent.select_action(state)

            # ── Step emulator ─────────────────────────────────────────────────
            state, _, done, raw_reward = env.step(action)
            total_reward += raw_reward
            step         += 1

            # ── Grab raw RGB frame for streaming ──────────────────────────────
            raw_frame = env.render()   # (210, 160, 3) RGB

            # Upscale with nearest-neighbour (keeps pixel art sharp)
            frame = cv2.resize(
                raw_frame,
                (raw_frame.shape[1] * SCALE, raw_frame.shape[0] * SCALE),
                interpolation=cv2.INTER_NEAREST,
            )
            frame_bgr      = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            _, jpeg_bytes  = cv2.imencode(".jpg", frame_bgr,
                                          [cv2.IMWRITE_JPEG_QUALITY, 85])

            with _lock:
                _latest_frame = jpeg_bytes.tobytes()
                _stats.update({
                    "episode": episode,
                    "score":   int(total_reward),
                    "step":    step,
                    "epsilon": round(agent.epsilon, 3),
                    "mode":    mode,
                })

            time.sleep(1.0 / FPS)

            if done:
                break

        episode += 1


# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__)

PAGE = """
<!DOCTYPE html>
<html>
<head>
  <title>Atari DQN</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #111;
      color: #ccc;
      font-family: monospace;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 24px;
      gap: 16px;
    }
    h2 { color: #00ff99; letter-spacing: 3px; font-size: 14px; }
    #game {
      image-rendering: pixelated;
      border: 1px solid #333;
      display: block;
    }
    #hud {
      display: flex;
      gap: 24px;
      font-size: 13px;
      color: #888;
    }
    #hud span { color: #00ff99; }
    #mode-badge {
      font-size: 11px;
      padding: 2px 8px;
      border-radius: 4px;
      background: #1a1a1a;
      border: 1px solid #333;
      color: #aaa;
    }
  </style>
  <script>
    setInterval(() => {
      fetch('/stats').then(r => r.json()).then(d => {
        document.getElementById('ep').textContent    = d.episode;
        document.getElementById('score').textContent = d.score;
        document.getElementById('step').textContent  = d.step;
        document.getElementById('eps').textContent   = d.epsilon;
        document.getElementById('mode').textContent  = d.mode;
      });
    }, 500);
  </script>
</head>
<body>
  <h2>ATARI DQN</h2>
  <img id="game" src="/stream">
  <div id="hud">
    <div>Episode <span id="ep">-</span></div>
    <div>Score <span id="score">-</span></div>
    <div>Step <span id="step">-</span></div>
    <div>Epsilon <span id="eps">-</span></div>
  </div>
  <div id="mode-badge">agent: <span id="mode">-</span></div>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(PAGE)

@app.route("/stream")
def stream():
    """MJPEG stream — browser renders this as live video via <img src="/stream">."""
    def generate():
        while True:
            with _lock:
                frame = _latest_frame
            if frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame +
                    b"\r\n"
                )
            time.sleep(1.0 / FPS)

    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stats")
def stats():
    with _lock:
        return jsonify(_stats)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── PyCharm / no-args defaults — edit these ───────────────────────────────
    DEFAULT_MODEL = "checkpoints/latest.pt"   # set to None for random agent
    DEFAULT_EPS   = 0.05
    DEFAULT_GAME  = "ALE/Breakout-v5"
    # ─────────────────────────────────────────────────────────────────────────

    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--eps",   default=DEFAULT_EPS, type=float)
    parser.add_argument("--game",  default=DEFAULT_GAME)
    parser.add_argument("--random", dest="model", action="store_const", const=None,
                        help="Force random agent")
    args = parser.parse_args()

    GAME = args.game

    # If checkpoint doesn't exist yet, fall back to random gracefully
    model_path = args.model
    if model_path and not Path(model_path).exists():
        print(f"  Checkpoint not found: {model_path}")
        print("  Falling back to random agent.")
        model_path = None

    t = threading.Thread(
        target=game_loop,
        args=(model_path, args.eps),
        daemon=True,
    )
    t.start()

    print("=" * 40)
    print("  Atari DQN Live Viewer")
    print("  http://localhost:5000")
    print("=" * 40)
    if model_path:
        print(f"  Model  : {model_path}")
        print(f"  Epsilon: {args.eps}")
    else:
        print("  Mode   : random agent (no model)")
    print()

    app.run(host="0.0.0.0", port=5000, threaded=True)
