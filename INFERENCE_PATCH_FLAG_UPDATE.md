# Inference patch-mode flag update

The inference command-line behavior is now explicit:

- no patch flag: full-image inference;
- `--patch-based`: MONAI sliding-window inference;
- `--no-patch-based`: explicit full-image inference.

The parser default is `False`, so a YAML `patch_based: true` value cannot silently enable sliding-window inference when no command-line flag is supplied.

Examples:

```bash
python3 scripts/infer.py --config configs/inference/fm_aleatoric.yaml
```

runs full-image inference.

```bash
python3 scripts/infer.py \
  --config configs/inference/fm_aleatoric.yaml \
  --patch-based \
  --patch-size 128 \
  --patch-overlap 0.25
```

runs MONAI sliding-window inference.
