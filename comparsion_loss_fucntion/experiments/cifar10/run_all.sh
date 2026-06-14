#!/usr/bin/env bash
# Run all 6 GAN loss functions on CIFAR-10 (200 epochs each)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

LOSSES=(standard lsgan wgan wgangp hinge hybrid)

for loss in "${LOSSES[@]}"; do
  echo "=== CIFAR-10 | ${loss} ==="
  python src/train.py --loss "$loss" --dataset cifar10 --epochs 200 --seeds 42
done

echo "All CIFAR-10 experiments complete. Results in ./runs/"
