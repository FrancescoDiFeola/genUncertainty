"""Step 1 of the VAE study: how much a pretrained 3D autoencoder loses on brain tumor MRI.

Inference only. Every representation (MAISI VAE, 4x downsampling, identity, Haar) maps a
normalized volume to a reconstruction of the same shape, and the analyses compare the two
on a common scale. Plan and hypotheses: docs/idea_traduzione_senza_vae.md, section 6,
"Step 1 in dettaglio"; module map and conventions: README.md in this folder.

Modules:
    io               raw volumes from the BraTS cache, RAS orientation, MAISI normalization
    representations  the encode-decode round trips under audit
    compare          generic original-vs-candidate comparison (analysis A), reused in step 2
    lesions          analysis B: real enhancing lesions by size and thickness
    synthetic        analysis C: inserted lesions, contrast-size curves
    frequency        analysis D: Fourier shell correlation and power ratio
    segmenter        fixed BraTS segmenter for lesion detection in analysis B
    stats            cluster bootstrap and critical-size estimation
"""
