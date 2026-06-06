#!/usr/bin/env bash
# Run all 6 GAN loss functions on EuroSAT (150 epochs each)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

LOSSES=(standard lsgan wgan wgangp hinge hybrid)

for loss in "${LOSSES[@]}"; do
  echo "=== EuroSAT | ${loss} ==="
  python src/train.py --loss "$loss" --dataset eurosat --epochs 150 --seed 42
done

echo "All EuroSAT experiments complete. Results in ./runs/"
