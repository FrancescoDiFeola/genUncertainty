# `latent_uq.audit`: step 1, the autoencoder ceiling

Status: skeleton. Every function documents its contract and raises `NotImplementedError`.
Plan, hypotheses and decision criteria: `docs/idea_traduzione_senza_vae.md`, section 6.

## Data flow for one subject and one representation

```text
cache (raw int16) ──io.load_volume──▶ Volume (original grid, raw intensities, labels, brain)
     │
     ├─ io.maisi_normalize ──▶ normalized volume (0th/99.5th percentiles → 0/1, no clipping)
     │        └─ synthetic.insert (analysis C only, after normalization)
     ├─ io.prepare ──▶ RAS orientation + padding to a multiple of 16
     ├─ representation(volume) ──▶ reconstruction, same shape
     ├─ io.restore ──▶ back to the original grid and orientation
     └─ compare / lesions / synthetic.measure / frequency ──▶ table rows (CSV)
```

## Conventions

- Arrays are `float32` `X, Y, Z` on the original BraTS grid, 1 mm isotropic, unless stated.
- All representations are compared on the same normalized scale, with a fixed data range
  of 1; metrics are computed inside region masks, the background separately.
- No resampling: BraTS is already 1 mm isotropic.
- Decoding must not introduce seams: whole volume, or MAISI's exact tensor splitting
  (`num_splits`), never a sliding window unless declared.
- Labels (BraTS 2023 GLI): 1 necrotic core, 2 edema, 3 enhancing tumor.

## Modules

| Module | Role |
|---|---|
| `io.py` | subjects of a split, raw volumes from the cache, MAISI normalization, RAS and padding |
| `representations.py` | `Identity`, `HaarRoundTrip`, `Downsample`, `MaisiVAE`, built from config specs |
| `compare.py` | region masks and global metrics (analysis A); entry point that assembles all tables |
| `lesions.py` | lesion extraction, size and thickness, contrast retention, detection (analysis B) |
| `synthetic.py` | lesion placement, rendering, insertion and measurement (analysis C) |
| `frequency.py` | apodization, Fourier shell correlation, power ratio, effective resolution (analysis D) |
| `segmenter.py` | fixed BraTS segmenter used for detection; choice still open |
| `stats.py` | cluster bootstrap, binned summaries, d50/d90 critical sizes |

Entry points: `scripts/make_patient_splits.py`, `scripts/audit_vae.py`,
`configs/audit_maisi.yaml`, `hpc/audit_vae.sh`; tests in `tests/test_audit.py`.
