"""Control tests for step 1 (latent_uq.audit).

They describe the contracts of the skeleton; until a function is implemented its tests
are expected failures (NotImplementedError), so the suite stays green. Any other failure
is a real error.
"""
import numpy as np
import pytest

from latent_uq.audit import compare, frequency, io, lesions, representations, synthetic

pytestmark = pytest.mark.xfail(raises=NotImplementedError, strict=False,
                               reason="step 1 skeleton: function not implemented yet")


def smooth_volume(shape=(40, 36, 30), seed=0):
    """A brain-like volume: smooth positive intensities inside an ellipsoid, zero outside."""
    rng = np.random.default_rng(seed)
    grid = np.meshgrid(*[np.linspace(-1, 1, n) for n in shape], indexing="ij")
    inside = sum(axis**2 for axis in grid) < 0.8
    image = np.where(inside, 100 + 20 * grid[0] + rng.normal(0, 1, shape), 0)
    return image.astype(np.float32), inside


def sphere(shape, center, radius):
    grid = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    return sum((axis - c)**2 for axis, c in zip(grid, center)) <= radius**2


def test_maisi_normalize_maps_percentiles_without_clipping():
    image, _ = smooth_volume()
    image[0, 0, 0] = 10 * image.max()  # A single bright voxel above the 99.5th percentile.
    normalized = io.maisi_normalize(image)
    assert normalized.min() == pytest.approx(0)
    assert np.percentile(normalized, 99.5) == pytest.approx(1, abs=1e-5)
    assert normalized.max() > 1  # No clipping.


def test_prepare_and_restore_are_inverse_and_pad_to_16():
    image, _ = smooth_volume()
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])  # LPS-like axes, as in BraTS files.
    prepared, layout = io.prepare(image, affine)
    assert all(n % io.MAISI_MULTIPLE == 0 for n in prepared.shape)
    np.testing.assert_array_equal(io.restore(prepared, layout), image)


@pytest.mark.parametrize("spec", [{"type": "identity"}, {"type": "haar", "levels": 2}])
def test_exact_representations_reproduce_the_input(spec):
    image, _ = smooth_volume(shape=(32, 32, 32))
    model = representations.build(spec)
    np.testing.assert_allclose(model(image), image, rtol=0, atol=1e-4)


def test_downsample_keeps_shape_and_smooth_content():
    image, _ = smooth_volume(shape=(32, 32, 32))
    model = representations.build({"type": "downsample", "factor": 4})
    reconstruction = model(image)
    assert reconstruction.shape == image.shape
    assert abs(float(reconstruction.mean() - image.mean())) < 1


def test_global_metrics_are_perfect_for_identical_volumes():
    image, inside = smooth_volume()
    labels = np.zeros(image.shape, np.uint8)
    labels[sphere(image.shape, (20, 18, 15), 4)] = 3
    masks = compare.region_masks(labels, inside)
    for row in compare.global_metrics(image, image.copy(), masks):
        assert row["mse"] == 0 and row["mae"] == 0


def test_extract_lesions_measures_a_sphere():
    labels = np.zeros((40, 40, 40), np.uint8)
    labels[sphere(labels.shape, (20, 20, 20), 5)] = 3
    labels[2, 2, 2] = 3  # A single voxel: below min_voxels, discarded.
    found, components, discarded = lesions.extract_lesions(labels)
    assert len(found) == 1 and discarded == 1
    assert found[0].diameter_mm == pytest.approx(10, rel=0.1)
    assert found[0].thickness_max_mm == pytest.approx(10, rel=0.2)
    assert components.max() == 1


def test_contrast_retention_is_one_for_identical_volumes():
    image, inside = smooth_volume()
    component = sphere(image.shape, (20, 18, 15), 3)
    image[component] += 50
    surround = lesions.shell(component, component, inside)
    result = lesions.contrast_retention(image, image.copy(), component, surround)
    assert result["retention"] == pytest.approx(1)


def test_synthetic_lesion_survives_the_identity():
    image, inside = smooth_volume()
    normalized = io.maisi_normalize(image)
    lesion = synthetic.SyntheticLesion(center=(0, 0, 0), diameter_mm=6, contrast=0.3)
    placed = synthetic.place(inside, np.zeros_like(inside), [lesion], np.random.default_rng(0))
    with_lesion, masks = synthetic.insert(normalized, placed)
    (row, ) = synthetic.measure(with_lesion, with_lesion.copy(), masks, placed)
    assert row["contrast_retention"] == pytest.approx(1)


def test_fourier_shell_correlation_of_a_volume_with_itself_is_one():
    image, _ = smooth_volume(shape=(32, 32, 32))
    _, fsc = frequency.fourier_shell_correlation(image, image.copy())
    np.testing.assert_allclose(fsc, 1, atol=1e-6)


def test_patient_splits_leave_no_shared_patient():
    from scripts.make_patient_splits import rewrite, shared_patients

    def entry(case):
        return {"t1": f"{case}/{case}-t1n.nii.gz", "t1ce": f"{case}/{case}-t1c.nii.gz"}

    values = dict(train=[entry("BraTS-GLI-00001-000"), entry("BraTS-GLI-00002-000")],
                  val=[entry("BraTS-GLI-00003-000")],
                  test=[entry("BraTS-GLI-00001-001"), entry("BraTS-GLI-00004-000")])
    assert set(shared_patients(values)) == {"BraTS-GLI-00001"}
    for policy in ("move", "drop"):
        new_values, _ = rewrite(values, policy)
        assert not shared_patients(new_values)
        assert "BraTS-GLI-00004-000/BraTS-GLI-00004-000-t1n.nii.gz" in [
            e["t1"] for e in new_values["test"]]
