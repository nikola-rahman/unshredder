#!/usr/bin/env bash
# Prepare a fresh GPU pod for training. Run on the pod from the synced repo root:
#   WANDB_API_KEY=... bash scripts/pod_setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq unzip fonts-dejavu-core fonts-liberation fonts-urw-base35 fonts-freefont-ttf \
    fonts-crosextra-carlito fonts-crosextra-caladea >/dev/null
command -v uv >/dev/null || pip install -q uv
uv sync

# Real shredded benchmarks + intact pages (Paixão et al.), ~850 MB.
if [ ! -d /root/data/datasets ]; then
    mkdir -p /root/data
    uv run --with gdown gdown 1hWpdOemB4ERppMBYRgKItK5zoGgy_qc3 -O /root/data/datasets.zip
    unzip -q /root/data/datasets.zip -d /root/data && rm /root/data/datasets.zip
fi
ln -sfn /root/data/datasets datasets

if [ -n "${WANDB_API_KEY:-}" ]; then
    uv run wandb login --relogin "$WANDB_API_KEY" >/dev/null
fi
mkdir -p models
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "pod ready"
