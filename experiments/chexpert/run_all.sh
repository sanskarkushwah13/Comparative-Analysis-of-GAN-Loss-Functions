#!/usr/bin/env bash
# Run all 6 GAN loss functions on CheXpert (300 epochs each)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

LOSSES=(standard lsgan wgan wgangp hinge hybrid)

for loss in "${LOSSES[@]}"; do
  echo "=== CheXpert | ${loss} ==="
  python src/train.py --loss "$loss" --dataset chexpert --epochs 300 --seed 42
done

echo "All CheXpert experiments complete. Results in ./runs/"
