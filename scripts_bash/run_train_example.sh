#!/usr/bin/env bash
set -e
python scripts/train.py --config configs/train/ldm_aleatoric.yaml --dry-run
