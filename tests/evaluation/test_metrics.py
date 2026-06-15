import pytest

pytest.importorskip("itk")
pytest.importorskip("medpy")
import itk
import numpy as np
from medpy.metric.binary import hd95 as medpy_hd95
from medpy.metric.image import mutual_information as medpy_mi

from unigradicon.evaluation.metrics import (
    compute_chamfer_mm,
    compute_dice_score,
    compute_hd95_mm,
    compute_mutual_information,
    compute_tre_mm,
    generate_surface_points_from_segmentation,
    image_physical_bounds,
    load_points,
    mean_point_distance_mm,
    points_inside_physical_bounds,
)


def _make_image(array: np.ndarray):
    image = itk.GetImageFromArray(array.astype(np.float32))
    image.SetSpacing((1.0, 1.0, 1.0))
    image.SetOrigin((0.0, 0.0, 0.0))
    image.SetDirection(np.eye(3))
    return image


def test_mean_point_distance_mm():
    points_a = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float64)
    points_b = points_a + np.array([1.0, 0.0, 0.0], dtype=np.float64)
    assert np.isclose(mean_point_distance_mm(points_a, points_b), 1.0)


def test_image_physical_bounds_and_point_containment():
    array = np.zeros((5, 5, 5), dtype=np.float32)
    image = itk.GetImageFromArray(array)
    image.SetSpacing((2.0, 2.0, 2.0))
    image.SetOrigin((1.0, 1.0, 1.0))
    image.SetDirection(np.eye(3))
    bounds_min, bounds_max = image_physical_bounds(image)
    inside = np.array([[1.0, 1.0, 1.0], [9.0, 9.0, 9.0]], dtype=np.float64)
    outside = np.array([[100.0, 100.0, 100.0]], dtype=np.float64)
    assert points_inside_physical_bounds(inside, bounds_min, bounds_max)
    assert not points_inside_physical_bounds(outside, bounds_min, bounds_max)


def test_tre_zero_for_identity():
    points = np.array([[0.0, 0.0, 0.0], [10.0, 2.0, 1.0]], dtype=np.float64)
    identity = itk.IdentityTransform[itk.D, 3].New()
    tre = compute_tre_mm(
        moving_points=points,
        fixed_points=points.copy(),
        predicted_transform=identity,
    )
    assert tre == 0.0


def test_chamfer_zero_for_identical_clouds():
    points = np.array([[1.0, 0.0, 0.0], [5.0, 2.0, 3.0]], dtype=np.float64)
    chamfer, chamfer_half = compute_chamfer_mm(points, points.copy())
    assert chamfer == 0.0
    assert chamfer_half == 0.0


def test_chamfer_matches_expected_for_shifted_clouds():
    points_a = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
    points_b = points_a + np.array([1.0, 0.0, 0.0], dtype=np.float64)
    chamfer, chamfer_half = compute_chamfer_mm(points_a, points_b)
    assert np.isclose(chamfer, 2.0)
    assert np.isclose(chamfer_half, 1.0)


def test_chamfer_with_unequal_point_counts():
    points_a = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float64)
    points_b = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    chamfer, chamfer_half = compute_chamfer_mm(points_a, points_b)
    assert np.isclose(chamfer, 1.5)
    assert np.isclose(chamfer_half, 0.75)


def test_dice_and_hd95_for_identical_masks():
    array = np.zeros((8, 8, 8), dtype=np.uint8)
    array[2:6, 2:6, 2:6] = 1
    seg = _make_image(array)
    dice = compute_dice_score(seg, seg)
    hd95 = compute_hd95_mm(seg, seg)
    assert np.isclose(dice, 1.0)
    assert np.isclose(hd95, 0.0)


def test_hd95_matches_medpy_for_shifted_masks():
    fixed_arr = np.zeros((12, 12, 12), dtype=np.uint8)
    fixed_arr[2:8, 2:8, 2:8] = 1
    moving_arr = np.zeros((12, 12, 12), dtype=np.uint8)
    moving_arr[4:10, 3:9, 2:8] = 1

    spacing = (1.5, 2.0, 1.0)  # ITK order (x, y, z)
    fixed = itk.GetImageFromArray(fixed_arr.astype(np.float32))
    fixed.SetSpacing(spacing)
    moving = itk.GetImageFromArray(moving_arr.astype(np.float32))
    moving.SetSpacing(spacing)

    expected = medpy_hd95(
        moving_arr > 0,
        fixed_arr > 0,
        voxelspacing=tuple(spacing)[::-1],
    )
    assert np.isclose(compute_hd95_mm(fixed, moving), expected)


def test_hd95_per_label_averaged():
    fixed_arr = np.zeros((12, 12, 12), dtype=np.uint8)
    fixed_arr[2:6, 2:6, 2:6] = 1
    fixed_arr[7:11, 7:11, 7:11] = 2
    moving_arr = np.zeros((12, 12, 12), dtype=np.uint8)
    moving_arr[3:7, 2:6, 2:6] = 1
    moving_arr[7:11, 6:10, 7:11] = 2

    spacing = (1.0, 1.5, 2.0)  # ITK order (x, y, z)
    fixed = itk.GetImageFromArray(fixed_arr.astype(np.float32))
    fixed.SetSpacing(spacing)
    moving = itk.GetImageFromArray(moving_arr.astype(np.float32))
    moving.SetSpacing(spacing)

    array_spacing = tuple(spacing)[::-1]
    expected = np.mean(
        [
            medpy_hd95(moving_arr == label, fixed_arr == label, voxelspacing=array_spacing)
            for label in (1, 2)
        ]
    )
    assert np.isclose(compute_hd95_mm(fixed, moving), expected)


def test_mi_higher_for_aligned_than_shuffled():
    rng = np.random.default_rng(42)
    fixed = rng.random((12, 12, 12))
    moving_same = fixed.copy()
    moving_shuffled = fixed.ravel().copy()
    rng.shuffle(moving_shuffled)
    moving_shuffled = moving_shuffled.reshape(fixed.shape)

    fixed_img = _make_image(fixed)
    same_img = _make_image(moving_same)
    shuffled_img = _make_image(moving_shuffled)
    assert compute_mutual_information(fixed_img, same_img) > compute_mutual_information(
        fixed_img, shuffled_img
    )


def test_mi_matches_medpy_in_bits():
    rng = np.random.default_rng(7)
    fixed = rng.random((10, 10, 10))
    moving = fixed + 0.1 * rng.random((10, 10, 10))

    fixed_img = _make_image(fixed)
    moving_img = _make_image(moving)

    expected = medpy_mi(
        fixed.astype(np.float64).ravel(),
        moving.astype(np.float64).ravel(),
        bins=64,
    )
    assert np.isclose(compute_mutual_information(fixed_img, moving_img), expected)


def test_mi_ignores_nonfinite_voxels():
    rng = np.random.default_rng(11)
    fixed = rng.random((8, 8, 8))
    moving = fixed + 0.05 * rng.random((8, 8, 8))

    corrupted = moving.copy()
    corrupted[0, 0, 0] = np.nan
    corrupted[1, 1, 1] = np.inf
    corrupted[2, 2, 2] = -np.inf

    fixed_img = _make_image(fixed)
    corrupted_img = _make_image(corrupted)

    result = compute_mutual_information(fixed_img, corrupted_img)
    assert np.isfinite(result)

    finite = np.isfinite(fixed.ravel()) & np.isfinite(corrupted.ravel())
    expected = medpy_mi(
        fixed.ravel()[finite].astype(np.float64),
        corrupted.ravel()[finite].astype(np.float64),
        bins=64,
    )
    assert np.isclose(result, expected)


def test_load_points_from_whitespace_txt(tmp_path):
    points_path = tmp_path / "landmarks.txt"
    points_path.write_text("1.0 2.0 3.0\n4.0 5.0 6.0\n", encoding="utf-8")
    points = load_points(str(points_path))
    assert points.shape == (2, 3)
    assert np.allclose(points[1], np.array([4.0, 5.0, 6.0]))


def test_generate_surface_points_deterministic_with_seed():
    seg_arr = np.zeros((10, 10, 10), dtype=np.uint8)
    seg_arr[2:8, 2:8, 2:8] = 1
    seg = _make_image(seg_arr)
    points_a = generate_surface_points_from_segmentation(
        segmentation=seg,
        label_value=1,
        max_points=100,
        seed=123,
    )
    points_b = generate_surface_points_from_segmentation(
        segmentation=seg,
        label_value=1,
        max_points=100,
        seed=123,
    )
    assert points_a.shape == (100, 3)
    assert np.allclose(points_a, points_b)
