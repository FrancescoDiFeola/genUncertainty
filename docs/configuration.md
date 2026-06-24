# Configuration

See `CONFIG_REFERENCE.md` for the complete schema. The most important fields are:

```yaml
run:
  framework: dm        # dm, fm, ldm, lfm
  mode: aleatoric      # base, aleatoric, selfcond

model:
  in_ch: 2
  out_ch: 1
  diff_ckpt: /path/to/checkpoint.pth
  vae_ckpt: null       # required only for ldm/lfm
  context_ckpt: null   # required only for selfcond

data:
  dataset_class: my_project.datasets.MyDataset
  dataset_kwargs:
    root: /data
    split: test

analysis:
  metrics: true
  sparsification: false
  spatial_error_correlation: false
  calibration_bins: false
```

Datasets must return `condition`, `target`, and optionally `case_id`.
