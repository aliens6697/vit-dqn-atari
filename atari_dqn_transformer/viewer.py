"""
viewer.py — live game viewer
==============================
Streams the game to your browser in real time.
Identical UX to play.py in the DQN project.

USAGE:
    python viewer.py                                     # random agent
    python viewer.py --model checkpoints/latest.pt       # trained transformer
    python viewer.py --model checkpoints/latest.pt --eps 0.05

THEN OPEN:
    http://localhost:5000
"""

import argparse
import threading
import time
from pathlib import Path

import cv2
import torch
from flask import Flask, Response, jsonify, render_template_string

from config import GAME, VIEWER_FPS, VIEWER_SCALE, VIEWER_PORT
from environment import AtariEnv
from learner import TransformerAgent

# ── Shared state ──────────────────────────────────────────────────────────────
_latest_frame = None
_stats        = {"episode": 1, "score": 0, "step": 0,
                 "epsilon": 1.0, "mode": "random", "model": "transformer"}
_lock         = threading.Lock()


# ── Game loop ─────────────────────────────────────────────────────────────────

def game_loop(model_path: str | None, epsilon: float, game: str):
    global _latest_frame, _stats

    env   = AtariEnv(game, render_mode="rgb_array")
    agent = TransformerAgent(env.n_actions, torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    if model_path:
        agent.load(model_path)
        agent.epsilon = epsilon
        mode = f"transformer  eps={epsilon:.2f}"
    else:
        agent.epsilon = 1.0
        mode = "random"

    episode = 1

    while True:
        state        = env.reset()
        total_reward = 0.0
        step         = 0

        while True:
            action = agent.select_action(state)
            state, _, done, raw_reward = env.step(action)
            total_reward += raw_reward
            step         += 1

            raw_frame = env.render()
            frame     = cv2.resize(
                raw_frame,
                (raw_frame.shape[1] * VIEWER_SCALE,
                 raw_frame.shape[0] * VIEWER_SCALE),
                interpolation=cv2.INTER_NEAREST,
            )
            frame_bgr     = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            _, jpeg_bytes = cv2.imencode(".jpg", frame_bgr,
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

            time.sleep(1.0 / VIEWER_FPS)
            if done:
                break

        episode += 1


# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__)

PAGE = """
<!DOCTYPE html>
<html>
<head>
  <title>Atari Transformer</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #0a0a0f;
      color: #ccc;
      font-family: monospace;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 24px;
      gap: 16px;
    }
    h2 { color: #7c6af7; letter-spacing: 3px; font-size: 14px; }
    #game {
      image-rendering: pixelated;
      border: 1px solid #222;
      display: block;
    }
    #hud {
      display: flex;
      gap: 24px;
      font-size: 13px;
      color: #666;
    }
    #hud span { color: #7c6af7; }
    #badge {
      font-size: 11px;
      padding: 3px 10px;
      border-radius: 4px;
      background: #111;
      border: 1px solid #222;
      color: #7c6af7;
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
  <h2>ATARI TRANSFORMER</h2>
  <img id="game" src="/stream">
  <div id="hud">
    <div>Episode <span id="ep">-</span></div>
    <div>Score <span id="score">-</span></div>
    <div>Step <span id="step">-</span></div>
    <div>Epsilon <span id="eps">-</span></div>
  </div>
  <div id="badge">agent: <span id="mode">-</span></div>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(PAGE)

@app.route("/stream")
def stream():
    def generate():
        while True:
            with _lock:
                frame = _latest_frame
            if frame:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + frame + b"\r\n")
            time.sleep(1.0 / VIEWER_FPS)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stats")
def stats():
    with _lock:
        return jsonify(_stats)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── PyCharm defaults — edit these ─────────────────────────────────────────
    DEFAULT_MODEL = "checkpoints/latest.pt"
    DEFAULT_EPS   = 0.05
    DEFAULT_GAME  = GAME
    # ─────────────────────────────────────────────────────────────────────────

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default=DEFAULT_MODEL)
    parser.add_argument("--eps",    default=DEFAULT_EPS, type=float)
    parser.add_argument("--game",   default=DEFAULT_GAME)
    parser.add_argument("--random", dest="model", action="store_const",
                        const=None, help="Force random agent")
    args = parser.parse_args()

    model_path = args.model
    if model_path and not Path(model_path).exists():
        print(f"  Checkpoint not found: {model_path}")
        print("  Falling back to random agent.")
        model_path = None

    t = threading.Thread(
        target=game_loop,
        args=(model_path, args.eps, args.game),
        daemon=True,
    )
    t.start()

    print("=" * 40)
    print("  Atari Transformer Viewer")
    print(f"  http://localhost:{VIEWER_PORT}")
    print("=" * 40)
    print(f"  Model : {model_path or 'random'}")
    print()

    app.run(host="0.0.0.0", port=VIEWER_PORT, threaded=True)
