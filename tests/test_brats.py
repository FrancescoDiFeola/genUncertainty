from dataclasses import asdict
import json

import numpy as np
import pytest
import torch

from latent_uq.brats import BraTSVolumeDataset, prepare_brats
from latent_uq.config import Component, Config, from_dict
from latent_uq.data import random_crop_pair
from latent_uq.models import ContextEncoder

nib = pytest.importorskip("nibabel")

MODALITIES = {"t1": "t1n", "t1ce": "t1c", "t2": "t2w", "t2f": "t2f", "seg": "seg"}
SHAPE = (20, 18, 12)
BRAIN = (slice(4, 15), slice(3, 14), slice(2, 10))


def write_subject(root, name, rng, shape=SHAPE):
    entry = {}
    for key, suffix in MODALITIES.items():
        volume = np.zeros(shape, np.int16)
        if key == "seg":
            volume[8:11, 6:9, 4:7] = 2
        else:
            volume[BRAIN] = rng.integers(1, 1000, size=volume[BRAIN].shape)
        path = root / name / f"{name}-{suffix}.nii.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(volume, np.diag([-1.0, -1.0, 1.0, 1.0])), path)
        entry[key] = f"{name}/{path.name}"
    return entry


@pytest.fixture
def brats(tmp_path):
    rng = np.random.default_rng(0)
    root = tmp_path / "nifti"
    entries = [write_subject(root, f"BraTS-GLI-0000{i}-000", rng) for i in range(3)]
    split_file = tmp_path / "fold0.json"
    split_file.write_text(
        json.dumps(dict(data_dir=str(root), train=entries[:2], val=[], test=entries[2:])))
    cache = tmp_path / "cache"
    prepare_brats(split_file, cache, workers=2)
    kwargs = dict(cache_dir=str(cache), split_file=str(split_file), condition=["t1", "t2", "t2f"],
                  target=["t1ce"])
    return root, split_file, cache, kwargs


def test_prepare_crops_to_brain_and_resumes(brats, capsys):
    root, split_file, cache, _ = brats
    index = json.loads((cache / "index.json").read_text())
    assert index["modalities"] == ["t1", "t1ce", "t2", "t2f"] and index["label"] == "seg"
    meta = index["subjects"]["BraTS-GLI-00000-000"]
    assert meta["bounds"] == [[4, 15], [3, 14], [2, 10]] and meta["shape"] == list(SHAPE)
    stored = np.load(cache / "volumes/BraTS-GLI-00000-000.npy")
    assert stored.dtype == np.int16 and stored.shape == (4, 11, 11, 8)
    assert np.load(cache / "volumes/BraTS-GLI-00000-000_seg.npy").shape == (11, 11, 8)
    prepare_brats(split_file, cache)
    assert "3 already cached, 0 to prepare" in capsys.readouterr().out


def test_whole_volumes_match_direct_normalization(brats):
    root, _, _, kwargs = brats
    dataset = BraTSVolumeDataset(split="test", **kwargs)
    item = dataset[0]
    assert len(dataset) == 1 and item["case_id"] == "BraTS-GLI-00002-000"
    assert item["condition"].shape == (3, *SHAPE) and item["target"].shape == (1, *SHAPE)
    raw = nib.load(root / "BraTS-GLI-00002-000/BraTS-GLI-00002-000-t1c.nii.gz").get_fdata()
    low, high = np.percentile(raw[raw != 0], [0.5, 99.5])
    expected = np.full(SHAPE, -1.0)
    expected[BRAIN] = np.clip((raw[BRAIN] - low) / (high - low), 0, 1) * 2 - 1
    np.testing.assert_allclose(item["target"][0].numpy(), expected, atol=1e-5)
    unlabeled = BraTSVolumeDataset(**dict(kwargs, target=[]), split="test")[0]
    assert "target" not in unlabeled


def test_training_patches_cover_the_brain(brats):
    _, _, _, kwargs = brats
    dataset = BraTSVolumeDataset(split="train", patch_size=8, samples_per_volume=3, **kwargs)
    assert len(dataset) == 6
    for index in range(len(dataset)):
        item = dataset[index]
        assert item["condition"].shape == (3, 8, 8, 8) and item["target"].shape == (1, 8, 8, 8)
        assert (item["condition"] > -1).any()
    # A patch larger than the volume is padded with background.
    large = BraTSVolumeDataset(split="train", patch_size=24, **kwargs)[0]["condition"]
    assert large.shape == (3, 24, 24, 24) and (large[:, SHAPE[0]:] == -1).all()


def test_dataset_rejects_invalid_requests(brats, tmp_path):
    _, split_file, cache, kwargs = brats
    for override in (dict(target=["t1"]), dict(condition=["flair"]), dict(percentiles=[5, 95]),
                     dict(split="val"), dict(patch_size=0)):
        with pytest.raises(ValueError):
            BraTSVolumeDataset(**{**kwargs, "split": "train", **override})
    with pytest.raises(FileNotFoundError, match="prepare_brats"):
        BraTSVolumeDataset(**dict(kwargs, cache_dir=str(tmp_path / "missing"), split="train"))
    values = json.loads(split_file.read_text())
    values["test"].append({k: v.replace("00002", "00009") for k, v in values["test"][0].items()})
    other = tmp_path / "fold1.json"
    other.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="not in"):
        BraTSVolumeDataset(**dict(kwargs, split_file=str(other), split="test"))


def test_prepare_reports_subjects_on_a_different_grid(tmp_path):
    rng = np.random.default_rng(0)
    good = write_subject(tmp_path, "BraTS-GLI-00000-000", rng)
    bad = write_subject(tmp_path, "BraTS-GLI-00001-000", rng)
    nib.save(nib.Nifti1Image(np.ones((20, 18, 13), np.int16), np.eye(4)),
             tmp_path / bad["t2"])
    split_file = tmp_path / "fold.json"
    split_file.write_text(json.dumps(dict(train=[good, bad])))
    with pytest.raises(RuntimeError, match="BraTS-GLI-00001-000"):
        prepare_brats(split_file, tmp_path / "cache")
    index = json.loads((tmp_path / "cache/index.json").read_text())
    assert list(index["subjects"]) == ["BraTS-GLI-00000-000"]


def volume_config(kwargs, framework, mode):
    config = Config(framework=framework, mode=mode)
    config.data.class_path = "latent_uq.brats.BraTSVolumeDataset"
    config.data.kwargs = dict(kwargs, split="train", patch_size=8, samples_per_volume=2)
    config.model.spatial_dims = 3
    config.model.condition_channels = 3
    config.model.context_dim = 8
    config.model.backbone = Component(
        "latent_uq.models.UNet",
        dict(num_channels=[8, 8],
             num_res_blocks=1,
             attention_levels=[False, False],
             norm_num_groups=4,
             num_head_channels=4))
    config.process.flow_base_size = 8**3
    config.inference.steps = 3
    config.inference.last_k = 1
    config.inference.samples = 2
    config.inference.patch_size = 8
    config.inference.overlap = 0.5
    config.validate()
    return config


@pytest.mark.parametrize("framework,mode,tiling,window", [
    ("dm", "base", "per_window", 8),
    ("dm", "selfcond", "per_window", 8),
    ("fm", "base", "per_window", 8),
    ("fm", "aleatoric", "per_window", 8),
    ("fm", "selfcond", "per_window", 8),
    ("dm", "selfcond", "per_step", 8),
    # A 16-voxel window exceeds the 12-slice axis, so the volume is padded and cropped back.
    ("fm", "selfcond", "per_step", 16),
    ("dm", "base", "per_window_shared_noise", 16),
    ("fm", "aleatoric", "per_window_shared_noise", 8),
])
def test_3d_patch_training_and_whole_volume_inference(brats, tmp_path, framework, mode, tiling,
                                                      window):
    from latent_uq.cli import infer
    from latent_uq.training import load_checkpoint, train

    _, _, _, kwargs = brats
    config = volume_config(kwargs, framework, mode)
    checkpoint = load_checkpoint(train(config, tmp_path / "train"))
    config.data.kwargs["split"] = "test"
    config.inference.tiling = tiling
    config.inference.patch_size = window
    if mode == "base":
        config.inference.uncertainty = "posthoc"
    infer(config, checkpoint, tmp_path / "infer")
    arrays = np.load(tmp_path / "infer/predictions/000000.npz")
    assert str(arrays["case_id"]) == "BraTS-GLI-00002-000"
    assert arrays["prediction"].shape == arrays["target"].shape == (1, *SHAPE)
    assert arrays["variance"].shape == (1, *SHAPE) and (arrays["variance"] >= 0).all()
    assert (tmp_path / "infer/metrics_brain.csv").exists()  # The default evaluation region.


def test_3d_building_blocks_and_validation():
    condition = torch.rand(2, 1, 3, 5, 7)
    cropped, paired = random_crop_pair(condition, condition.clone(), 4)
    assert cropped.shape == (2, 1, 4, 4, 4)
    torch.testing.assert_close(cropped, paired)
    assert ContextEncoder(1, 8, spatial_dims=3)(torch.rand(2, 1, 6, 6, 6)).shape == (2, 1, 8)
    values = asdict(Config(framework="fm"))
    values["model"]["spatial_dims"] = 3
    with pytest.raises(ValueError, match="flow_base_size"):
        from_dict(values)
    values["process"]["flow_base_size"] = 512
    from_dict(values)
    for section, key, value in [("model", "spatial_dims", 4), ("framework", None, "lfm")]:
        invalid = json.loads(json.dumps(values))
        if key is None:
            invalid[section] = value
        else:
            invalid[section][key] = value
        with pytest.raises(ValueError):
            from_dict(invalid)
    values["model"]["backbone"]["kwargs"] = {"spatial_dims": 2}
    with pytest.raises(ValueError, match="backbone.kwargs.spatial_dims"):
        from_dict(values)


def test_region_metrics_nifti_export_and_window_profile(brats, tmp_path):
    import csv
    from latent_uq.cli import infer
    from latent_uq.training import load_checkpoint, train

    root, _, _, kwargs = brats
    config = volume_config(kwargs, "fm", "aleatoric")
    checkpoint = load_checkpoint(train(config, tmp_path / "train"))
    config.data.kwargs.update(split="test", regions={"brain": None, "tumor": [2], "enhancing": [3]})
    config.inference.tiling = "per_step"
    config.inference.prediction_format = "nifti"
    config.inference.window_profile = True
    infer(config, checkpoint, tmp_path / "infer")
    output = tmp_path / "infer"
    # The synthetic segmentation has a 3x3x3 label-2 region and no label 3.
    for region, voxels in (("brain", 11 * 11 * 8), ("tumor", 27)):
        with (output / f"metrics_{region}.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        assert [row["case_id"] for row in rows] == ["BraTS-GLI-00002-000"]
        assert np.isfinite(float(rows[0]["mse"])) and np.isfinite(float(rows[0]["pearson"]))
        with (output / f"window_profile_{region}.csv").open() as handle:
            assert sum(int(row["count"]) for row in csv.DictReader(handle)) == voxels
    assert not (output / "metrics_enhancing.csv").exists()
    assert json.loads((output / "run.json").read_text())["regions"] == ["brain", "tumor"]
    source = nib.load(root / "BraTS-GLI-00002-000/BraTS-GLI-00002-000-t1c.nii.gz")
    for name in ("prediction", "variance"):
        image = nib.load(output / f"predictions/BraTS-GLI-00002-000_{name}.nii.gz")
        assert image.shape == SHAPE and np.allclose(image.affine, source.affine)


def test_whole_volumes_carry_regions_and_subsets(brats):
    _, _, _, kwargs = brats
    item = BraTSVolumeDataset(**kwargs, split="test", regions={"brain": None, "tumor": [2]})[0]
    assert item["regions"]["brain"].shape == (1, *SHAPE) and item["regions"]["brain"].sum() == 968
    assert item["regions"]["tumor"].sum() == 27 and item["affine"].shape == (4, 4)
    assert "regions" not in BraTSVolumeDataset(**kwargs, split="test", regions={})[0]
    patches = BraTSVolumeDataset(**kwargs, split="train", patch_size=8, samples_per_volume=3,
                                 max_subjects=1)
    assert len(patches) == 3 and "regions" not in patches[0] and "affine" not in patches[0]
