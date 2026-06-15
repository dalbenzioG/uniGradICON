from typing import Optional

import itk
import numpy as np
from medpy.metric.binary import hd95 as medpy_hd95
from medpy.metric.image import mutual_information as medpy_mi
from scipy.spatial import cKDTree


def load_points(path: str) -> np.ndarray:
    points = np.loadtxt(path, ndmin=2)
    if points.shape[1] != 3:
        raise ValueError(f"Point file '{path}' must contain Nx3 coordinates.")
    return points.astype(np.float64)


def generate_surface_points_from_segmentation(
    segmentation: itk.Image,
    label_value: int = 1,
    max_points: Optional[int] = 5000,
    seed: int = 0,
) -> np.ndarray:
    seg_arr = itk.GetArrayFromImage(segmentation)
    mask_arr = (seg_arr == label_value).astype(np.uint8)
    if np.count_nonzero(mask_arr) == 0:
        raise ValueError(f"Label {label_value} was not found in the segmentation.")

    mask = itk.GetImageFromArray(mask_arr)
    mask.CopyInformation(segmentation)
    contour = itk.binary_contour_image_filter(mask)
    contour_arr = itk.GetArrayFromImage(contour)
    zyxs = np.argwhere(contour_arr > 0)
    if zyxs.shape[0] == 0:
        raise ValueError("No boundary voxels were found in the segmentation.")

    if max_points is not None and zyxs.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(zyxs.shape[0], size=max_points, replace=False)
        zyxs = zyxs[keep]

    # ITK array order is z,y,x; TransformIndexToPhysicalPoint expects x,y,z.
    points = np.asarray(
        [
            contour.TransformIndexToPhysicalPoint((int(x), int(y), int(z)))
            for z, y, x in zyxs
        ],
        dtype=np.float64,
    )
    return points


def save_points(path: str, points: np.ndarray) -> None:
    np.savetxt(path, points, fmt="%.8f")


def image_physical_bounds(image: itk.Image) -> tuple[np.ndarray, np.ndarray]:
    size = image.GetLargestPossibleRegion().GetSize()
    corners = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                index = (
                    0 if i == 0 else int(size[0]) - 1,
                    0 if j == 0 else int(size[1]) - 1,
                    0 if k == 0 else int(size[2]) - 1,
                )
                corners.append(np.asarray(image.TransformIndexToPhysicalPoint(index), dtype=np.float64))
    corners = np.asarray(corners, dtype=np.float64)
    return corners.min(axis=0), corners.max(axis=0)


def points_inside_physical_bounds(
    points: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    margin_mm: float = 0.0,
) -> bool:
    if points.size == 0:
        return False
    lower = bounds_min - margin_mm
    upper = bounds_max + margin_mm
    return bool(np.all(points >= lower) and np.all(points <= upper))


def mean_point_distance_mm(points_a: np.ndarray, points_b: np.ndarray) -> float:
    if points_a.shape != points_b.shape:
        raise ValueError("Point arrays must have identical shape.")
    return float(np.linalg.norm(points_a - points_b, axis=1).mean())


def compute_tre_mm(
    moving_points: np.ndarray,
    fixed_points: np.ndarray,
    predicted_transform: itk.Transform,
    initial_transform: Optional[itk.Transform] = None,
) -> float:
    if moving_points.shape != fixed_points.shape:
        raise ValueError("TRE requires moving and fixed points with identical shape.")
    transformed = []
    for point in moving_points:
        coords = tuple(float(x) for x in point)
        if initial_transform is not None:
            coords = initial_transform.TransformPoint(coords)
        coords = predicted_transform.TransformPoint(coords)
        transformed.append(np.array(coords, dtype=np.float64))
    transformed = np.asarray(transformed)
    return float(np.linalg.norm(transformed - fixed_points, axis=1).mean())


def _mean_nearest_neighbor_distance_mm(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> float:
    if source_points.size == 0 or target_points.size == 0:
        raise ValueError("Chamfer distance requires both point sets to be non-empty.")
    source_points = np.asarray(source_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)
    if source_points.ndim != 2 or target_points.ndim != 2:
        raise ValueError("Chamfer distance expects point arrays with shape Nx3.")
    if source_points.shape[1] != 3 or target_points.shape[1] != 3:
        raise ValueError("Chamfer distance expects 3D points with shape Nx3.")

    nn_tree = cKDTree(target_points)
    nearest_distances, _ = nn_tree.query(source_points, k=1)
    return float(np.mean(nearest_distances))


def compute_chamfer_mm(points_a: np.ndarray, points_b: np.ndarray) -> tuple[float, float]:
    a_to_b = _mean_nearest_neighbor_distance_mm(points_a, points_b)
    b_to_a = _mean_nearest_neighbor_distance_mm(points_b, points_a)
    chamfer = float(a_to_b + b_to_a)
    return chamfer, float(0.5 * chamfer)


def _binary_mask(image: itk.Image) -> itk.Image:
    array = itk.GetArrayFromImage(image)
    binary = (array > 0).astype(np.uint8)
    mask = itk.GetImageFromArray(binary)
    mask.CopyInformation(image)
    return mask


def compute_dice_score(fixed_seg: itk.Image, warped_moving_seg: itk.Image) -> float:
    fixed = itk.GetArrayFromImage(fixed_seg).astype(np.int64)
    moving = itk.GetArrayFromImage(warped_moving_seg).astype(np.int64)
    labels = sorted((set(np.unique(fixed)) & set(np.unique(moving))) - {0})
    if not labels:
        return 0.0
    scores = []
    for label in labels:
        fixed_mask = fixed == label
        moving_mask = moving == label
        denom = fixed_mask.sum() + moving_mask.sum()
        if denom == 0:
            continue
        scores.append((2.0 * np.logical_and(fixed_mask, moving_mask).sum()) / denom)
    return float(np.mean(scores)) if scores else 0.0


def compute_hd95_mm(fixed_seg: itk.Image, warped_moving_seg: itk.Image) -> float:
    fixed = itk.GetArrayFromImage(fixed_seg).astype(np.int64)
    moving = itk.GetArrayFromImage(warped_moving_seg).astype(np.int64)
    labels = sorted((set(np.unique(fixed)) & set(np.unique(moving))) - {0})
    if not labels:
        return float("nan")

    # ITK spacing is (x, y, z); the numpy array axes are (z, y, x), so reverse.
    spacing = tuple(float(s) for s in fixed_seg.GetSpacing())[::-1]
    scores = [
        float(medpy_hd95(moving == label, fixed == label, voxelspacing=spacing))
        for label in labels
    ]
    return float(np.mean(scores))


def compute_mutual_information(
    fixed_image: itk.Image,
    warped_moving_image: itk.Image,
    bins: int = 64,
    fixed_mask: Optional[itk.Image] = None,
) -> float:
    fixed = itk.GetArrayFromImage(fixed_image).astype(np.float64)
    moving = itk.GetArrayFromImage(warped_moving_image).astype(np.float64)
    if fixed_mask is not None:
        mask = itk.GetArrayFromImage(_binary_mask(fixed_mask)).astype(bool)
        fixed = fixed[mask]
        moving = moving[mask]
    else:
        fixed = fixed.ravel()
        moving = moving.ravel()

    finite = np.isfinite(fixed) & np.isfinite(moving)
    fixed = fixed[finite]
    moving = moving[finite]

    if fixed.size == 0 or moving.size == 0:
        return float("nan")

    return float(medpy_mi(fixed, moving, bins=bins))
