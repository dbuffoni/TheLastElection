#!/usr/bin/env bash
# Prepares the Fedora desktop for the S1 server (Kev) and installs it as a user service.
# S2 runs in LM Studio (see README.md). Usage: ./setup-desktop.sh [checkpoint] [port]
set -euo pipefail

CKPT="${1:-jaredpalmer/kev-4b}"
PORT="${2:-8009}"
DIR="${TLE_LOCAL_DIR:-$HOME/tle-local}"

command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: install the driver first (README, step 1)"; exit 1; }
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

command -v git >/dev/null || sudo dnf install -y git
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }

mkdir -p "$DIR"
[ -d "$DIR/kev" ] || git clone https://github.com/jaredpalmer/kev.git "$DIR/kev"
(cd "$DIR/kev" && uv sync --extra serve)

mkdir -p "$HOME/.config/systemd/user"
cat > "$HOME/.config/systemd/user/tle-s1.service" <<UNIT
[Unit]
Description=The Last Election S1 decider ($CKPT)
After=network-online.target

[Service]
WorkingDirectory=$DIR/kev
ExecStart=$(command -v uv) run --extra serve python -m kev.serve --run $CKPT --port $PORT
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
UNIT
systemctl --user daemon-reload
systemctl --user enable --now tle-s1.service
loginctl enable-linger "$USER" || true

echo "S1 starting on 127.0.0.1:$PORT (first start downloads the checkpoint). Follow: journalctl --user -u tle-s1 -f"
echo "Then: python3 bench.py check"
