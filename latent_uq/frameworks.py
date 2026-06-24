from __future__ import annotations

LATENT_FRAMEWORKS = {"ldm", "lfm"}
IMAGE_FRAMEWORKS = {"dm", "fm"}
ALL_FRAMEWORKS = LATENT_FRAMEWORKS | IMAGE_FRAMEWORKS

FRAMEWORK_ALIASES = {
    "ldm": "ldm",
    "lfm": "lfm",
    "dm": "dm",
    "ddpm": "dm",
    "diffusion": "dm",
    "fm": "fm",
    "rf": "fm",
    "flow": "fm",
    "flow_matching": "fm",
}


def normalize_framework(name: str | None, default: str = "ldm") -> str:
    if name is None:
        return default
    key = str(name).lower().replace("-", "_")
    if key not in FRAMEWORK_ALIASES:
        valid = ", ".join(sorted(FRAMEWORK_ALIASES))
        raise ValueError(f"Unsupported framework '{name}'. Valid values/aliases: {valid}")
    return FRAMEWORK_ALIASES[key]


def is_latent_framework(name: str) -> bool:
    return normalize_framework(name) in LATENT_FRAMEWORKS


def is_image_framework(name: str) -> bool:
    return normalize_framework(name) in IMAGE_FRAMEWORKS
