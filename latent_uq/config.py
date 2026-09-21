"""One strict configuration shared by training, inference, and saved checkpoints."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, get_type_hints
import math

import yaml


@dataclass
class Component:
    class_path: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    checkpoint: str | None = None


@dataclass
class CustomAnalysisSpec:
    """A pluggable inference analysis, not one of the four built-in ones.

    class_path must resolve to a latent_uq.analysis.CustomAnalysis subclass.
    sampling_analysis names which built-in analysis's sampling parameters
    (steps, last_k, decode_samples, ...) this one reuses; "metrics" is the only
    analysis available for every framework/mode/uncertainty combination, so it is
    the safe default, but any other built-in name can be selected explicitly.
    """
    class_path: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    sampling_analysis: str = "metrics"


@dataclass
class DataConfig:
    class_path: str = "latent_uq.data.PairedDataset"
    kwargs: dict[str, Any] = field(default_factory=lambda: {"csv_path": "examples/data/pairs.csv"})
    batch_size: int = 2
    num_workers: int = 0
    drop_last: bool = True  # Applies only during training; inference uses every sample.


@dataclass
class ModelConfig:
    backbone: Component = field(default_factory=lambda: Component("latent_uq.models.UNet"))
    context_encoder: Component | None = None
    autoencoder: Component | None = None
    condition_channels: int = 1
    target_channels: int = 1
    latent_channels: int = 3
    scaling_factor: float = 1.0
    context_dim: int = 128
    context_tokens: int = 1
    context_input: str = "variance"
    uncertainty_decoder: Component | None = None
    vae_use_forward: bool = True  # forward() must return (reconstruction, mean, sigma).


@dataclass
class ProcessConfig:
    num_train_timesteps: int = 1000
    beta_start: float = 0.0015
    beta_end: float = 0.0205
    beta_schedule: str = "scaled_linear_beta"
    flow_base_size: int | None = None
    flow_inference_base_size: int | None = None
    flow_inference_size: int | None = None


@dataclass
class TrainingConfig:
    epochs: int = 1
    lr: float = 0.000015
    patch_size: int | None = None
    pad_value: float = -1.0
    min_logvar: float = -7.0
    regularization: float = 0.001
    loss_weight: float = 1.0
    calibration_weight: float = 0.01
    weight_decay: float = 0.01
    amp: bool = True
    grad_clip: float | None = None
    tensorboard: bool = False
    preview_steps: int = 0


@dataclass
class InferenceConfig:
    steps: int | None = None  # 50 for diffusion, 30 for flow matching.
    uncertainty: str = "auto"  # auto, none, propagated, posthoc.
    last_k: int | None = None  # Left null, the resolved Reference supplies a model-specific default.
    samples: int | None = None  # 10 for metrics; 4 for the other posthoc analyses.
    decode_samples: int | None = None  # 10 aleatoric; 20 selfcond.
    self_conditioning: bool = True  # False implements the test-time ablation.
    patch_size: int | None = None
    overlap: float = 0.25
    window_batch_size: int = 1
    blend_mode: str = "gaussian"
    pad_value: float = -1.0
    analyses: list[str] = field(default_factory=lambda: ["metrics"])
    data_range: float | None = None  # Defaults to max(target) - min(target), per image.
    save_predictions: bool = True
    custom_analyses: dict[str, CustomAnalysisSpec] = field(default_factory=dict)


@dataclass
class Config:
    framework: str = "dm"
    mode: str = "selfcond"
    seed: int = 0
    device: str = "cpu"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    process: ProcessConfig = field(default_factory=ProcessConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    @property
    def latent(self) -> bool:
        return self.framework in {"ldm", "lfm"}

    @property
    def diffusion(self) -> bool:
        return self.framework in {"dm", "ldm"}

    @property
    def steps(self) -> int:
        return self.inference.steps or (50 if self.diffusion else 30)

    @property
    def uncertainty(self) -> str:
        value = self.inference.uncertainty
        return ("none" if self.mode == "base" else "propagated") if value == "auto" else value

    def validate(self) -> None:
        if self.framework not in {"dm", "fm", "ldm", "lfm"}:
            raise ValueError("framework must be dm, fm, ldm, or lfm")
        if self.mode not in {"base", "aleatoric", "selfcond"}:
            raise ValueError("mode must be base, aleatoric, or selfcond")
        positive = {
            "batch_size": self.data.batch_size,
            "epochs": self.training.epochs,
            "lr": self.training.lr,
            "scaling_factor": self.model.scaling_factor,
            "condition_channels": self.model.condition_channels,
            "target_channels": self.model.target_channels,
            "latent_channels": self.model.latent_channels,
            "context_dim": self.model.context_dim,
            "context_tokens": self.model.context_tokens,
            "num_train_timesteps": self.process.num_train_timesteps,
            "window_batch_size": self.inference.window_batch_size,
        }
        for name, value in dict(last_k=self.inference.last_k,
                                data_range=self.inference.data_range,
                                flow_base_size=self.process.flow_base_size,
                                flow_inference_base_size=self.process.flow_inference_base_size,
                                flow_inference_size=self.process.flow_inference_size).items():
            if value is not None:
                positive[name] = value
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.data.num_workers < 0 or self.seed < 0 or self.training.preview_steps < 0:
            raise ValueError("num_workers, seed, and preview_steps must be nonnegative")
        if self.inference.steps is not None and self.inference.steps < 1:
            raise ValueError("inference.steps must be positive")
        if self.training.preview_steps > self.process.num_train_timesteps:
            raise ValueError("preview_steps cannot exceed process.num_train_timesteps")
        if self.steps > self.process.num_train_timesteps:
            raise ValueError("inference.steps cannot exceed process.num_train_timesteps")
        if not 0 < self.process.beta_start < self.process.beta_end < 1:
            raise ValueError("Require 0 < beta_start < beta_end < 1")
        if not 0 <= self.inference.overlap < 1:
            raise ValueError("overlap must be in [0, 1)")
        if self.inference.blend_mode not in {"constant", "gaussian"}:
            raise ValueError("blend_mode must be constant or gaussian")
        for size in (self.training.patch_size, self.inference.patch_size):
            if size is not None and size < 1:
                raise ValueError("patch_size must be positive or null")
        if self.training.grad_clip is not None and (not math.isfinite(self.training.grad_clip)
                                                    or self.training.grad_clip <= 0):
            raise ValueError("grad_clip must be positive and finite or null")
        for value in (self.training.min_logvar, self.training.regularization,
                      self.training.pad_value, self.inference.pad_value, self.training.loss_weight,
                      self.training.calibration_weight, self.training.weight_decay):
            if not math.isfinite(value):
                raise ValueError("Loss bounds, regularization, and padding must be finite")
        if not -80 <= self.training.min_logvar <= 80:
            raise ValueError("min_logvar must be in [-80,80]")
        if min(self.training.regularization, self.training.loss_weight,
               self.training.calibration_weight, self.training.weight_decay) < 0:
            raise ValueError("Loss weights, regularization and weight_decay must be nonnegative")
        if self.model.context_input not in {"variance", "prediction_variance"}:
            raise ValueError("context_input must be variance or prediction_variance")
        if self.model.uncertainty_decoder and (self.framework, self.mode) != ("ldm", "aleatoric"):
            raise ValueError("model.uncertainty_decoder is only supported for ldm/aleatoric")
        if self.inference.uncertainty not in {"auto", "none", "propagated", "posthoc"}:
            raise ValueError("uncertainty must be auto, none, propagated, or posthoc")
        if self.uncertainty == "propagated" and self.mode == "base":
            raise ValueError("base has no variance head; use posthoc or none")
        if self.inference.samples is not None and self.inference.samples < 2:
            raise ValueError("posthoc requires at least two independent samples")
        if self.latent and self.model.autoencoder is None:
            raise ValueError("ldm/lfm require model.autoencoder with an encode/decode interface")
        if not self.latent and self.model.autoencoder is not None:
            raise ValueError("model.autoencoder is only used by ldm/lfm")
        if self.inference.decode_samples is not None and self.inference.decode_samples < 2:
            raise ValueError("Latent propagation requires decode_samples >= 2")
        builtin = {"metrics", "sparsification", "calibration", "uncertainty_summary"}
        if set(self.inference.custom_analyses) & builtin:
            raise ValueError(f"custom_analyses names cannot reuse built-in names {sorted(builtin)}")
        for name, spec in self.inference.custom_analyses.items():
            if spec.sampling_analysis not in builtin:
                raise ValueError(
                    f"custom_analyses.{name}.sampling_analysis must be one of {sorted(builtin)}")
        allowed = builtin | set(self.inference.custom_analyses)
        if set(self.inference.analyses) - allowed or len(set(self.inference.analyses)) != len(
                self.inference.analyses):
            raise ValueError(f"analyses must contain unique names from {sorted(allowed)}")
        # A custom analysis's own uncertainty requirement is checked at run time
        # (CustomAnalysis.requires_uncertainty), since importing its class here would
        # break the deferred-instantiation pattern used for every other class_path.
        if self.uncertainty == "none" and set(
                self.inference.analyses) - {"metrics"} - set(self.inference.custom_analyses):
            raise ValueError("Uncertainty analyses require propagated or posthoc uncertainty")


def _construct(cls, values: dict, path="config"):
    """Reject unknown fields and wrong scalar types instead of silently ignoring them."""
    if not isinstance(values, dict):
        raise ValueError(f"{path} must be a mapping")
    definitions = {f.name: f for f in fields(cls)}
    unknown = values.keys() - definitions.keys()
    if unknown:
        raise ValueError(f"Unknown fields in {path}: {sorted(unknown)}")
    hints = get_type_hints(cls)
    nested = {
        "data": DataConfig,
        "model": ModelConfig,
        "process": ProcessConfig,
        "training": TrainingConfig,
        "inference": InferenceConfig,
        "backbone": Component,
        "context_encoder": Component,
        "autoencoder": Component,
        "uncertainty_decoder": Component
    }
    nested_dicts = {"custom_analyses": CustomAnalysisSpec}
    result = {}
    for key, value in values.items():
        kind = hints[key]
        if key in nested and value is not None:
            value = _construct(nested[key], value, f"{path}.{key}")
        elif key in nested_dicts:
            value = {
                name: _construct(nested_dicts[key], spec, f"{path}.{key}.{name}")
                for name, spec in value.items()
            }
        elif value is None:
            if type(None) not in getattr(kind, "__args__", ()):
                raise ValueError(f"{path}.{key} cannot be null")
        else:
            arguments = getattr(kind, "__args__", ())
            if type(None) in arguments:
                kind = next(t for t in arguments if t is not type(None))
            origin = getattr(kind, "__origin__", kind)
            valid = isinstance(value,
                               (int, float)) if origin is float else isinstance(value, origin)
            if origin is float and isinstance(value, bool):
                valid = False
            if origin is int and isinstance(value, bool):
                valid = False
            if not valid:
                raise ValueError(f"Wrong type for {path}.{key}: expected {kind}")
            if origin is list and not all(isinstance(item, str) for item in value):
                raise ValueError(f"{path}.{key} must be a list of strings")
        result[key] = value
    return cls(**result)


def from_dict(values: dict) -> Config:
    config = _construct(Config, values)
    config.validate()
    return config


def load_config(path: str | Path) -> Config:
    return from_dict(yaml.safe_load(Path(path).read_text()) or {})


def save_config(config: Config, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(asdict(config), sort_keys=False))
