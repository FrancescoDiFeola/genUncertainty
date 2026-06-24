#!/usr/bin/env python3
"""Image-level training backend placeholder.

The public API and configuration schema already support image-level frameworks.
This backend is intentionally minimal because image-level training loops are
project-specific in the current codebase. Use `scripts/infer.py` for image-level
inference, or replace this file with your project-specific training loop while
keeping the same command-line arguments.
"""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, required=False)
    parser.add_argument('--task', type=str, required=False)
    parser.add_argument('--output_dir', type=str, required=False)
    parser.add_argument('--diff_ckpt', type=str, required=False)
    parser.add_argument('--num_workers', type=int, required=False)
    parser.add_argument('--n_epochs', type=int, required=False)
    parser.add_argument('--batch_size', type=int, required=False)
    parser.add_argument('--lr', type=float, required=False)
    parser.add_argument('--in_ch', type=int, required=False)
    parser.add_argument('--out_ch', type=int, required=False)
    parser.add_argument('--spatial_enc_channels', type=int, required=False)
    args, _ = parser.parse_known_args()
    raise NotImplementedError(
        'Image-level training backend is declared in the public API but must be '
        'implemented for the target project. Image-level inference is supported. '
        f'Received arguments: {args}'
    )


if __name__ == '__main__':
    main()
