from dataclasses import asdict, replace
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from latent_uq.analysis import analyze, sparsification, sparsification_scores
from latent_uq.cli import infer, main
from latent_uq.config import from_dict, save_config, Component, CustomAnalysisSpec
from latent_uq.data import PairedDataset, make_loader, prepare_batch, random_crop_pair
from latent_uq.models import build_models, load_weights, predict
from latent_uq.sampling import sample
from latent_uq.training import train, load_checkpoint, restore_models


@pytest.mark.parametrize("framework", ["dm", "fm", "ldm", "lfm"])
@pytest.mark.parametrize("mode", ["base", "aleatoric", "selfcond"])
def test_train_reload_infer_all_modes(config_factory, tmp_path, framework, mode):
    config = config_factory(framework, mode)
    checkpoint_path = train(config, tmp_path / "train")
    checkpoint = load_checkpoint(checkpoint_path)
    assert checkpoint["epoch"] == 1
    assert (checkpoint["models"]["context"] is not None) == (mode == "selfcond")
    assert (checkpoint["models"]["vae"] is not None) == config.latent
    if mode == "base":
        config.inference.uncertainty = "posthoc"
    config.inference.analyses = ["metrics", "sparsification", "calibration", "uncertainty_summary"]
    if mode == 'aleatoric':
        config.inference.analyses = ['metrics', 'sparsification']
    # The run checkpoint includes the VAE: inference must not depend on its original path.
    if config.latent:
        Path(config.model.autoencoder.checkpoint).unlink()
    infer(config, checkpoint, tmp_path / "infer")
    files = list((tmp_path / "infer/metrics/predictions").glob("*.npz"))
    assert len(files) == 3
    for file in files:
        arrays = np.load(file)
        assert arrays["prediction"].shape == arrays["target"].shape == arrays["variance"].shape
        assert np.isfinite(arrays["variance"]).all() and (arrays["variance"] >= 0).all()
    with (tmp_path / "infer/metrics/metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [r["sample"] for r in rows] == ["0", "1", "2"]
    assert [r["case_id"] for r in rows] == ["example_0", "example_1", "example_2"]
    metric = 'mae' if (framework, mode) == ('dm', 'selfcond') else 'mse'
    assert all(np.isfinite(float(r[metric])) for r in rows)
    for analysis in config.inference.analyses:
        metadata = json.loads((tmp_path / 'infer' / analysis / 'run.json').read_text())
        assert metadata['legacy_function']
    with pytest.raises(FileExistsError):
        infer(config, checkpoint, tmp_path / "infer")


def test_resume_matches_uninterrupted_training(config_factory, tmp_path):
    config = config_factory("fm")
    checkpoint = train(config, tmp_path / "resumed")
    config.training.epochs = 2
    train(config, tmp_path / "resumed", resume=checkpoint)
    train(config, tmp_path / "continuous")
    resumed = load_checkpoint(checkpoint)
    continuous = load_checkpoint(tmp_path / "continuous/checkpoint.pt")
    for name in ("backbone", "context"):
        for key, value in resumed["models"][name].items():
            torch.testing.assert_close(value, continuous["models"][name][key], rtol=0, atol=0)
    config.training.lr *= 2
    with pytest.raises(ValueError, match="training.lr"):
        train(config, tmp_path / "resumed", resume=checkpoint)


def test_monitor_pngs_leave_training_unchanged(config_factory, tmp_path):
    config = config_factory("fm")
    config.training.epochs = 2
    config.training.sample_every = 1
    monitored = load_checkpoint(train(config, tmp_path / "monitored"))
    for name in ("losses.png", "samples.png"):
        assert (tmp_path / "monitored/monitor" / name).read_bytes().startswith(b"\x89PNG")
    records = [
        json.loads(line)
        for line in (tmp_path / "monitored/training.jsonl").read_text().splitlines()
    ]
    assert [record["epoch"] for record in records] == [1, 2]
    assert all({"loss", "mse", "logvar", "seconds"} <= record.keys() for record in records)
    config.training.plot_every = config.training.sample_every = 0
    plain = load_checkpoint(train(config, tmp_path / "plain"))
    assert not (tmp_path / "plain/monitor").exists()
    for name in ("backbone", "context"):
        for key, value in plain["models"][name].items():
            torch.testing.assert_close(value, monitored["models"][name][key], rtol=0, atol=0)


@pytest.mark.parametrize("framework,mode", [("lfm", "selfcond"), ("ldm", "aleatoric")])
def test_real_monai_unet_vae_and_optional_calibration(config_factory, tmp_path, framework, mode):
    config = config_factory(framework, mode)
    config.model.context_dim = 8
    config.model.context_encoder = None
    config.model.backbone = Component(
        "latent_uq.models.UNet",
        dict(num_channels=[8, 16],
             num_res_blocks=1,
             attention_levels=[False, True],
             norm_num_groups=4,
             num_head_channels=4))
    from monai.networks.nets import AutoencoderKL
    kwargs = dict(spatial_dims=2,
                  in_channels=1,
                  out_channels=1,
                  channels=[8, 8],
                  num_res_blocks=1,
                  latent_channels=2,
                  attention_levels=[False, False],
                  norm_num_groups=4,
                  with_encoder_nonlocal_attn=False,
                  with_decoder_nonlocal_attn=False)
    vae = AutoencoderKL(**kwargs)
    weights = tmp_path / "real_vae.pt"
    torch.save(vae.state_dict(), weights)
    config.model.autoencoder = Component("monai.networks.nets.AutoencoderKL", kwargs, str(weights))
    if mode == "aleatoric":
        config.model.uncertainty_decoder = Component(
            "latent_uq.models.LatentUncertaintyDecoder",
            dict(latent_channels=2, base_channels=32, out_channels=1, upsample_factor=2))
    checkpoint_path = train(config, tmp_path / "real")
    checkpoint = load_checkpoint(checkpoint_path)
    assert (checkpoint["models"]["uncertainty_decoder"] is not None) == (mode == "aleatoric")
    infer(config, checkpoint, tmp_path / "real_infer")
    assert len(list((tmp_path / "real_infer/predictions").glob("*.npz"))) == 3


@pytest.mark.parametrize("framework", ["dm", "fm", "ldm", "lfm"])
def test_model_module_direct_inference_defaults(config_factory, framework):
    from latent_uq.inference import module_for
    from latent_uq.models import encode

    config = config_factory(framework, "base")
    models = build_models(config)
    condition = torch.ones(2, 1, 8, 8)
    result = module_for(framework).infer(encode(models, condition, config), models, config)
    assert result.image.shape == condition.shape
    assert not result.image.requires_grad
    assert result.variance is None
    assert models.backbone.training  # The inference context restores the caller's mode.


def test_cli_checkpoint_settings_and_typed_overrides(config_factory, tmp_path, capsys):
    config = config_factory('dm', 'base')
    path = tmp_path / "config.yaml"
    save_config(config, path)
    main([
        "train", "--config",
        str(path), "--output-dir",
        str(tmp_path / "cli"), "--set", "framework=fm"
    ])
    checkpoint = str(tmp_path / "cli/checkpoint.pt")
    main([
        "infer", "--checkpoint", checkpoint, "--output-dir",
        str(tmp_path / "cli-infer"), "--set", "inference.patch_size=4", "--set",
        "inference.uncertainty=posthoc"
    ])
    assert json.loads((tmp_path / "cli-infer/run.json").read_text())["uncertainty"] == "posthoc"
    with pytest.raises(ValueError, match="cannot override"):
        main([
            "infer", "--checkpoint", checkpoint, "--output-dir",
            str(tmp_path / 'invalid'), "--set", "mode=aleatoric"
        ])
    with pytest.raises(ValueError, match="Unknown"):
        main([
            "train", "--config",
            str(path), "--output-dir", "unused", "--set", "training.learing_rate=0.1"
        ])


def test_unlabeled_inference(config_factory, tmp_path):
    config = config_factory("fm", "base")
    checkpoint = load_checkpoint(train(config, tmp_path / "train"))
    path = tmp_path / "data/unlabeled.csv"
    path.write_text("condition,case_id\ncondition_0.npy,scan_a\n")
    config.data.kwargs["csv_path"] = str(path)
    config.inference.analyses = []
    infer(config, checkpoint, tmp_path / "unlabeled")
    arrays = np.load(tmp_path / "unlabeled/predictions/000000.npz")
    assert "target" not in arrays and "variance" not in arrays
    assert str(arrays["case_id"]) == "scan_a"


def test_custom_analysis_extension_point(config_factory, tmp_path):
    from latent_uq.inference import module_for
    config = config_factory("dm", "selfcond")
    checkpoint = load_checkpoint(train(config, tmp_path / "train"))
    config.inference.analyses = ["threshold_check"]
    config.inference.custom_analyses = {
        "threshold_check":
        CustomAnalysisSpec("tests.helpers.ThresholdedUncertainty",
                           kwargs=dict(threshold=0.0),
                           sampling_analysis="sparsification")
    }
    infer(config, checkpoint, tmp_path / "custom")
    with (tmp_path / "custom/threshold_check.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [r["sample"] for r in rows] == ["0", "1", "2"]
    assert all(0.0 <= float(r["fraction_above_threshold"]) <= 1.0 for r in rows)
    metadata = json.loads((tmp_path / "custom/run.json").read_text())
    assert metadata["custom_analysis"] is True
    # sampling_analysis="sparsification" drives which built-in reference produces
    # the image/variance a custom analysis receives, not "metrics" (the default).
    assert metadata["legacy_function"] == module_for("dm").reference(config, "sparsification").function


def test_cli_inference_from_explicit_legacy_weights(config_factory, tmp_path):
    config = config_factory("ldm", "selfcond")
    models = build_models(config)
    config_path = tmp_path / "legacy.yaml"
    save_config(config, config_path)
    destination = tmp_path / "legacy_infer"
    command = ["infer", "--config", str(config_path), "--output-dir", str(destination)]
    with pytest.raises(ValueError, match="backbone.checkpoint"):
        main(command)
    for name, spec in [("backbone", config.model.backbone),
                       ("context", config.model.context_encoder)]:
        path = tmp_path / f"{name}.pt"
        torch.save(
            {
                "state_dict": {
                    f"module.{k}": v
                    for k, v in getattr(models, name).state_dict().items()
                }
            }, path)
        spec.checkpoint = str(path)
    save_config(config, config_path)
    main(command)
    assert len(list((destination / "predictions").glob("*.npz"))) == 3
    metadata = json.loads((destination / "run.json").read_text())
    assert metadata["weights_source"] == "component_checkpoints"


@pytest.mark.parametrize("section,key,value", [
    ("training", "epochs", 0),
    ("training", "lr", float("nan")),
    ("data", "batch_size", True),
    ("inference", "steps", 0),
    ("inference", "overlap", 1.0),
    ("inference", "patch_size", -1),
    ("inference", "samples", 1),
    ("inference", "analyses", ["typo"]),
    ("training", "tensorboard", "false"),
    ("training", "min_logvar", -100),
    ("training", "plot_every", -1),
    ("training", "sample_count", 0),
    ("training", "sample_steps", 0),
    ("training", "sample_data", ["split"]),
])
def test_invalid_config_rejected(config_factory, section, key, value):
    config = config_factory()
    config.inference.uncertainty = "posthoc"
    values = asdict(config)
    values[section][key] = value
    with pytest.raises(ValueError):
        from_dict(values)


def test_custom_analysis_config_validation(config_factory):
    config = config_factory()
    config.inference.custom_analyses = {
        "metrics": CustomAnalysisSpec("tests.helpers.ThresholdedUncertainty")
    }
    with pytest.raises(ValueError, match="built-in names"):
        config.validate()
    config.inference.custom_analyses = {
        "my_check": CustomAnalysisSpec("tests.helpers.ThresholdedUncertainty", sampling_analysis="bogus")
    }
    with pytest.raises(ValueError, match="sampling_analysis"):
        config.validate()


def test_config_round_trip_and_unknown_fields(config_factory):
    config = config_factory()
    assert asdict(from_dict(asdict(config))) == asdict(config)
    with pytest.raises(ValueError, match="Unknown"):
        from_dict({"training": {"epoch_start": 1}})


def test_strict_weights_and_shape_checks(config_factory, tmp_path):
    config = config_factory()
    models = build_models(config)
    missing = tmp_path / "missing.pt"
    with pytest.raises(FileNotFoundError):
        load_weights(models.backbone, missing)
    incompatible = tmp_path / "wrong.pt"
    torch.save({"wrong": torch.ones(1)}, incompatible)
    with pytest.raises(RuntimeError):
        load_weights(models.backbone, incompatible)
    with pytest.raises(ValueError, match="matching batch"):
        prepare_batch(dict(condition=torch.ones(1, 1, 8, 8), target=torch.ones(1, 1, 7, 8)), config)
    with pytest.raises(ValueError, match="NaN"):
        prepare_batch(dict(condition=torch.full((1, 1, 8, 8), float("nan"))), config)


def test_paired_crop_and_loader_validation(config_factory, tmp_path):
    config = config_factory()
    dataset = PairedDataset(**config.data.kwargs)
    assert len(dataset) == 3
    c = torch.rand(2, 1, 3, 5)
    pc, pt = random_crop_pair(c, c.clone(), 8)
    torch.testing.assert_close(pc, pt)
    assert pc.shape[-2:] == (8, 8)
    with pytest.raises(ValueError):
        random_crop_pair(c, torch.rand(2, 1, 5, 3), 2)
    path = tmp_path / "bad.csv"
    path.write_text("condition,target\na.npy,b.npy\nc.npy,\n")
    with pytest.raises(ValueError, match="all rows"):
        PairedDataset(str(path))


def test_analysis_ideal_and_uninformative_rankings():
    target = np.zeros((2, 5, 5), dtype=np.float32)
    prediction = np.linspace(0, 1, 50).reshape(2, 5, 5)
    scores = sparsification_scores(*sparsification(prediction, prediction))
    assert scores['ause'] == pytest.approx(0)
    assert scores['aurg'] > 0
    result = analyze(target, prediction, prediction.copy(), analysis='calibration')
    assert sum(row["count"] for row in result["calibration"]) == 25  # Reference uses channel zero.
    constant = analyze(target, prediction, np.ones_like(prediction))
    assert constant["metrics"][0]["pearson"] is None
    perfect = analyze(target, target, np.zeros_like(target))
    assert sparsification_scores(
        *sparsification(np.zeros_like(target), np.zeros_like(target)))['ause'] == 0
    assert perfect["metrics"][0]["psnr"] == float("inf")
    assert analyze(target[:, :2, :2], target[:, :2, :2], None)["metrics"][0]["ssim"] is None


def test_python_inference_rejects_changed_algorithm(config_factory, tmp_path):
    config = config_factory("dm")
    checkpoint = load_checkpoint(train(config, tmp_path / "train"))
    config.framework = "fm"
    with pytest.raises(ValueError, match="cannot override checkpoint: framework"):
        infer(config, checkpoint, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_resume_dry_run_loads_checkpoint_without_external_vae(config_factory, tmp_path):
    config = config_factory("lfm")
    checkpoint = train(config, tmp_path / "train")
    config_path = tmp_path / "config.yaml"
    save_config(config, config_path)
    Path(config.model.autoencoder.checkpoint).unlink()
    main([
        "train", "--config",
        str(config_path), "--output-dir",
        str(tmp_path / "dry"), "--resume",
        str(checkpoint), "--dry-run"
    ])
    assert not (tmp_path / "dry").exists()


def test_checkpoint_saved_before_a_config_field_existed(config_factory, tmp_path):
    config = config_factory("fm", "base")
    checkpoint = load_checkpoint(train(config, tmp_path / "train"))
    del checkpoint["config"]["model"]["spatial_dims"]
    infer(config, checkpoint, tmp_path / "infer")
    assert len(list((tmp_path / "infer/predictions").glob("*.npz"))) == 3


def test_analyses_accept_volumes():
    rng = np.random.default_rng(0)
    target = rng.random((1, 8, 9, 10)).astype(np.float32)
    prediction = target + 0.1 * rng.random(target.shape).astype(np.float32)
    uncertainty = rng.random(target.shape).astype(np.float32)
    for analysis in ("metrics", "sparsification", "calibration", "uncertainty_summary"):
        assert analyze(target, prediction, uncertainty, analysis=analysis)[analysis]
    assert np.isfinite(analyze(target, prediction, None)["metrics"][0]["ssim"])


TILINGS = ("per_window", "per_window_shared_noise", "per_step")


def _sample(condition, models, config, **inference):
    torch.manual_seed(0)
    return sample(condition, models, replace(config, inference=replace(config.inference,
                                                                        **inference)))


@pytest.mark.parametrize("framework", ["dm", "fm"])
@pytest.mark.parametrize("mode,uncertainty", [("base", "none"), ("base", "posthoc"),
                                              ("aleatoric", "propagated"),
                                              ("selfcond", "propagated")])
def test_every_tiling_with_one_window_is_whole_image_sampling(config_factory, framework, mode,
                                                              uncertainty):
    config = config_factory(framework, mode)
    config.inference.uncertainty = uncertainty
    models = build_models(config)
    condition = torch.randn(1, 1, 8, 8)
    whole = _sample(condition, models, config)
    for tiling in TILINGS:
        tiled = _sample(condition, models, config, patch_size=8, tiling=tiling)
        torch.testing.assert_close(tiled.image, whole.image, rtol=1e-5, atol=1e-5)
        if whole.variance is not None:
            torch.testing.assert_close(tiled.variance, whole.variance, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("framework", ["dm", "fm"])
@pytest.mark.parametrize("mode", ["base", "aleatoric"])
def test_shared_noise_and_per_step_tiling_keep_one_sample(config_factory, framework, mode):
    # The test backbone is pointwise and both samplers are deterministic: each pixel's
    # trajectory then depends only on its own noise, so tilings that share the noise field
    # reproduce the whole-image sample exactly, while independent window noise does not.
    config = config_factory(framework, mode)
    models = build_models(config)
    condition = torch.randn(1, 1, 12, 12)
    whole = _sample(condition, models, config)
    windows = dict(patch_size=4, overlap=0.5)
    for tiling in ("per_window_shared_noise", "per_step"):
        tiled = _sample(condition, models, config, tiling=tiling, **windows)
        torch.testing.assert_close(tiled.image, whole.image, rtol=1e-5, atol=1e-5)
        if mode != "base":
            torch.testing.assert_close(tiled.variance, whole.variance, rtol=1e-5, atol=1e-5)
    independent = _sample(condition, models, config, tiling="per_window", **windows)
    assert not torch.allclose(independent.image, whole.image, atol=1e-3)


def test_tiling_and_loader_validation(config_factory):
    config = config_factory("ldm", "selfcond")
    for section, key, value in [("inference", "tiling", "per_pixel"),
                                ("inference", "tiling", "per_step"),
                                ("data", "persistent_workers", True),
                                ("data", "prefetch_factor", 2),
                                ("inference", "prediction_format", "png"),
                                ("inference", "window_profile", True)]:
        values = asdict(config)
        values[section][key] = value
        with pytest.raises(ValueError):
            from_dict(values)


def test_loader_worker_options(config_factory):
    config = config_factory()
    config.data.num_workers = 1
    config.data.persistent_workers = True
    config.data.prefetch_factor = 1
    config.data.pin_memory = True
    config.validate()
    loader = make_loader(config, training=False)
    assert loader.persistent_workers and loader.prefetch_factor == 1
    assert next(iter(loader))["condition"].shape == (2, 1, 8, 8)


def test_analysis_mask_restricts_every_analysis():
    from latent_uq.config import Config
    rng = np.random.default_rng(0)
    target = rng.random((1, 8, 9, 10)).astype(np.float32)
    prediction = target + 0.1 * rng.random(target.shape).astype(np.float32)
    uncertainty = rng.random(target.shape).astype(np.float32)
    mask = np.zeros(target.shape[1:], bool)
    mask[2:6, 2:7, 3:8] = True
    metrics = analyze(target, prediction, uncertainty, mask=mask)["metrics"][0]
    assert metrics["mse"] == pytest.approx(np.mean((target[0][mask] - prediction[0][mask])**2))
    outside = prediction.copy()
    outside[0][~mask] += 5
    config = Config(framework="fm", mode="selfcond")
    for analysis in ("metrics", "sparsification", "calibration", "uncertainty_summary"):
        results = []
        for candidate in (prediction, outside):
            np.random.seed(0)
            results.append(analyze(target, candidate, uncertainty, analysis=analysis,
                                   config=config, mask=mask))
        for rows in (results[0], results[1]):
            for row in rows.get("metrics", []):
                row.pop("ssim")  # SSIM near the mask border also sees pixels outside it.
        assert results[0] == results[1]
    with pytest.raises(ValueError, match="mask"):
        analyze(target, prediction, uncertainty, mask=np.zeros(target.shape[1:], bool))


def test_window_concentration_counts_contributing_windows(config_factory):
    from latent_uq.sampling import window_concentration
    config = config_factory()
    config.inference.blend_mode = "constant"
    config.inference.patch_size = 8
    np.testing.assert_allclose(window_concentration((8, 8), config), 1)
    config.inference.patch_size, config.inference.overlap = 4, 0.5
    # Windows start at 0, 2 and 4 on each axis, so pixels see 1, 2, 2, 1 windows per axis.
    concentration = window_concentration((8, 8), config)
    assert concentration[0, 0] == pytest.approx(1)
    assert concentration[0, 3] == pytest.approx(1 / 2)
    assert concentration[3, 4] == pytest.approx(1 / 4)
    # A window larger than the image is padded as sliding-window inference pads it.
    config.inference.patch_size = 10
    np.testing.assert_allclose(window_concentration((8, 7), config), 1)
