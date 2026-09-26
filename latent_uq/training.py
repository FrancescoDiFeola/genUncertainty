"""One training loop for DM/FM/LDM/LFM and base/aleatoric/selfcond objectives."""
from dataclasses import asdict, replace
from pathlib import Path
import json
import random
import time
import numpy as np
import torch
from .config import Config, _construct, save_config
from .data import make_loader, prepare_batch, random_crop_pair
from .models import build_models, encode, decode, encode_context, predict, zero_context
from .monitor import Monitor
from .process import Process, heteroscedastic_loss
from .utils import finite, seed_everything

CHECKPOINT_VERSION = 2  # Version 1 used different self-conditioning and flow algorithms.
MODEL_PARTS = ("backbone", "context", "vae", "uncertainty_decoder")


def train_step(condition, target, models, process, config, stats=None):
    """Loss of one batch. A stats dictionary, if given, receives detached components of
    the loss for monitoring; it never changes the loss."""
    pixel_target = target
    condition, target = encode(models, condition, config), encode(models, target, config)
    if condition.shape[2:] != target.shape[2:]:
        raise ValueError("Encoded condition and target must have matching spatial dimensions")
    noisy, times, objective = process.training_pair(target)
    model_input = torch.cat([noisy, condition], dim=1)
    context = None
    if config.mode == "selfcond":
        with torch.no_grad():
            initial_prediction, initial_logvar = predict(models, model_input, times, config,
                                                         zero_context(len(noisy), config, noisy))
    with torch.autocast(device_type=condition.device.type,
                        enabled=condition.is_cuda and config.training.amp):
        if config.mode == "selfcond":
            context = encode_context(models, initial_logvar.detach(), config,
                                     initial_prediction.detach())
        prediction, logvar = predict(models, model_input, times, config, context)
        if config.mode == "base":
            loss = (prediction.float() - objective.float()).square().mean()
        else:
            loss = config.training.loss_weight * heteroscedastic_loss(
                prediction, logvar, objective, config.training.regularization,
                config.training.min_logvar)
            if stats is not None:
                # Base mode's loss, comparable across modes and free of the variance terms.
                stats["mse"] = (prediction.detach().float() - objective.float()).square().mean()
                stats["logvar"] = logvar.detach().float().mean()
        if models.uncertainty_decoder is not None:
            alpha = process.train_scheduler.alphas_cumprod.to(noisy.device)[times].reshape(
                -1, 1, 1, 1)
            x0 = ((noisy - (1 - alpha).sqrt() * prediction) / (alpha.sqrt() + 1e-8)).detach()
            with torch.no_grad():
                error = (pixel_target - decode(models, x0, config)).abs().mean(1, keepdim=True)
            estimate = models.uncertainty_decoder(logvar.clamp(-10, 10).float())
            if estimate.shape != error.shape:
                raise ValueError("Uncertainty decoder output must match the image-space error map")
            calibration = calibration_loss(estimate, error)
            loss = loss + config.training.calibration_weight * calibration
            if stats is not None:
                stats["calibration"] = calibration.detach()
    return finite(loss, "training loss")


def _preview(image):
    """First channel of the first image; the central slice of the last axis for volumes."""
    image = image[:1, :1].cpu()
    return image[..., image.shape[-1] // 2] if image.ndim == 5 else image


def calibration_loss(estimate, error):
    """Reference LDM calibration: MSE of spatially standardized uncertainty/error maps."""

    def standardize(value):
        centered = value - value.mean(dim=(2, 3), keepdim=True)
        return centered / (centered.std(dim=(2, 3), keepdim=True) + 1e-8)

    return (standardize(estimate) - standardize(error)).square().mean()


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError(
            "Expected a versioned latent_uq run checkpoint; a raw weight file can only be used "
            "for model initialization or direct inference, not resume"
        )
    return checkpoint


def restore_models(models, checkpoint):
    for name in MODEL_PARTS:
        module, weights = getattr(models, name), checkpoint["models"].get(name)
        if (module is None) != (weights is None):
            raise ValueError(f"Checkpoint/config mismatch for {name}")
        if module is not None:
            module.load_state_dict(weights, strict=True)


def _save_checkpoint(path, config, models, optimizer, scaler, epoch):
    checkpoint = dict(
        format_version=CHECKPOINT_VERSION,
        config=asdict(config),
        epoch=epoch,
        models={
            name: getattr(models, name).state_dict() if getattr(models, name) is not None else None
            for name in MODEL_PARTS
        },
        optimizer=optimizer.state_dict(),
        scaler=scaler.state_dict(),
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        python_rng=random.getstate(),
        numpy_rng=[
            *np.random.get_state()[:1],
            np.random.get_state()[1].tolist(), *np.random.get_state()[2:]
        ])
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def validate_checkpoint_config(config, checkpoint, *, resume=False):
    """Protect the trained algorithm in both Python and CLI entry points."""
    # Filling in defaults keeps checkpoints saved before a field was added comparable.
    current, previous = asdict(config), asdict(_construct(Config, checkpoint["config"]))
    protected = ["framework", "mode", "model", "process"]
    if resume:
        protected += ["seed", "data"]
    else:
        protected += ["training"]
    for key in protected:
        if current[key] != previous[key]:
            message = "Resume configuration mismatch" if resume else "Inference cannot override checkpoint"
            raise ValueError(f"{message}: {key}")
    if resume:
        for key in ("lr", "patch_size", "pad_value", "min_logvar", "regularization", "loss_weight",
                    "calibration_weight", "weight_decay", "amp", "grad_clip"):
            if current["training"][key] != previous["training"][key]:
                raise ValueError(f"Resume configuration mismatch: training.{key}")


def train(config, output_dir, *, resume=None):
    config.validate()
    seed_everything(config.seed)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()) and resume is None:
        raise FileExistsError(
            f"Training directory is not empty: {output}. Choose a new directory or use --resume.")
    checkpoint = load_checkpoint(resume) if resume else None
    if checkpoint:
        validate_checkpoint_config(config, checkpoint, resume=True)
    loader = make_loader(config, training=True)
    # Built before any output exists, so a monitoring setup error does not leave behind
    # a directory that only --resume accepts. It never moves the random state.
    monitor = Monitor(config, output) if (config.training.plot_every
                                          or config.training.sample_every) else None
    models = build_models(config, initialize=checkpoint is None)
    if checkpoint:
        restore_models(models, checkpoint)
    parameters = list(models.backbone.parameters())
    models.backbone.train()
    if models.context is not None:
        models.context.train()
        parameters += [p for p in models.context.parameters() if p.requires_grad]
    if models.uncertainty_decoder is not None:
        models.uncertainty_decoder.train()
        parameters += list(models.uncertainty_decoder.parameters())
    optimizer = torch.optim.AdamW(parameters,
                                  lr=config.training.lr,
                                  weight_decay=config.training.weight_decay)
    scaler = torch.amp.GradScaler("cuda",
                                  enabled=config.training.amp
                                  and torch.device(config.device).type == "cuda")
    start = 0
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start = checkpoint["epoch"]
        torch.set_rng_state(checkpoint["torch_rng"])
        if torch.cuda.is_available() and checkpoint["cuda_rng"]:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        random.setstate(checkpoint["python_rng"])
        np_state = checkpoint["numpy_rng"]
        np.random.set_state((np_state[0], np.asarray(np_state[1], dtype=np.uint32), *np_state[2:]))
    if config.training.epochs <= start:
        raise ValueError(
            "training.epochs is the total epoch budget and must exceed the saved epoch")
    output.mkdir(parents=True, exist_ok=True)
    save_config(config, output / "config.yaml")
    writer = None
    if config.training.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(output / "tensorboard"))
    process = Process(config)
    history = []
    try:
        for epoch in range(start, config.training.epochs):
            started = time.perf_counter()
            total_loss, items, components = 0.0, 0, {}
            for batch in loader:
                condition, target = prepare_batch(batch, config, require_target=True)
                if config.training.patch_size is not None:
                    condition, target = random_crop_pair(condition, target,
                                                         config.training.patch_size,
                                                         config.training.pad_value)
                optimizer.zero_grad(set_to_none=True)
                stats = {}
                loss = train_step(condition, target, models, process, config, stats)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if config.training.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(parameters,
                                                   config.training.grad_clip,
                                                   error_if_nonfinite=True)
                elif not scaler.is_enabled():
                    for parameter in parameters:
                        if parameter.grad is not None:
                            finite(parameter.grad, "gradient")
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(loss.detach()) * len(condition)
                for key, value in stats.items():
                    components[key] = components.get(key, 0.0) + float(value) * len(condition)
                items += len(condition)
            mean_loss = total_loss / items
            record = dict(epoch=epoch + 1, loss=mean_loss)
            record.update({key: value / items for key, value in components.items()})
            record["seconds"] = time.perf_counter() - started
            if torch.device(config.device).type == "cuda":
                # Peak since the run started, to size batch_size and patch_size.
                record["max_memory_gb"] = torch.cuda.max_memory_allocated(config.device) / 2**30
            history.append(record)
            details = "".join(f", {key}={record[key]:.6f}" for key in components)
            print(f"Epoch {epoch+1}/{config.training.epochs}: loss={mean_loss:.6f}{details}, "
                  f"{record['seconds'] / 60:.1f} min" +
                  (f", peak GPU memory={record['max_memory_gb']:.1f} GB"
                   if "max_memory_gb" in record else ""),
                  flush=True)
            if writer:
                writer.add_scalar("train/loss", mean_loss, epoch + 1)
                for key in components:
                    writer.add_scalar(f"train/{key}", record[key], epoch + 1)
                if config.training.preview_steps:
                    from .sampling import sample
                    preview = replace(config,
                                      inference=replace(config.inference,
                                                        steps=config.training.preview_steps,
                                                        uncertainty="none",
                                                        analyses=[],
                                                        patch_size=None))
                    # Preview must not change the random sequence of later training epochs.
                    devices = [torch.device(config.device).index or 0] if torch.device(
                        config.device).type == "cuda" else []
                    with torch.random.fork_rng(devices=devices):
                        generated = sample(condition[:1], models, preview)
                    writer.add_images("train/prediction", _preview(generated.image), epoch + 1)
                    writer.add_images("train/target", _preview(target), epoch + 1)
            _save_checkpoint(output / "checkpoint.pt", config, models, optimizer, scaler, epoch + 1)
            with (output / "training.jsonl").open("a") as handle:
                handle.write(json.dumps(history[-1]) + "\n")
            # After the checkpoint: a preview interrupted by the time limit loses nothing.
            if monitor is not None:
                monitor.after_epoch(epoch + 1, models)
    finally:
        if writer:
            writer.close()
    return output / "checkpoint.pt"
