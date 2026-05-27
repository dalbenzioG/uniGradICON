"""Landmark transforms for TRUSTED PCA+ICP rigid alignment."""

from __future__ import annotations

import json
import os
from typing import Callable, Dict, Optional

import itk
import numpy as np

from .metrics import mean_point_distance_mm

RAS_TO_LPS_SCALE = np.array([-1.0, -1.0, 1.0], dtype=np.float64)

LANDMARK_POINT_CONVENTIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {}


def _register_convention(name: str, fn: Callable[[np.ndarray], np.ndarray]) -> None:
    LANDMARK_POINT_CONVENTIONS[name] = fn


def convert_points_ras_to_lps(points: np.ndarray) -> np.ndarray:
    """Map Slicer-style RAS world coordinates to ITK/NIfTI LPS physical coordinates."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("convert_points_ras_to_lps expects points with shape Nx3.")
    return points * RAS_TO_LPS_SCALE


def convert_points_lps_to_ras(points: np.ndarray) -> np.ndarray:
    """Map ITK/NIfTI LPS physical coordinates to Slicer-style RAS world coordinates."""
    return convert_points_ras_to_lps(points)


def apply_landmark_point_convention(points: np.ndarray, convention: str) -> np.ndarray:
    try:
        transform = LANDMARK_POINT_CONVENTIONS[convention]
    except KeyError as exc:
        known = ", ".join(sorted(LANDMARK_POINT_CONVENTIONS))
        raise ValueError(f"Unknown landmark convention '{convention}'. Known: {known}.") from exc
    return transform(points)


_register_convention("identity", lambda points: np.asarray(points, dtype=np.float64))
_register_convention("ras_to_lps", convert_points_ras_to_lps)
_register_convention("lps_to_ras", convert_points_lps_to_ras)


def resolve_pca_icp_report_path(transform_path: str) -> Optional[str]:
    directory = os.path.dirname(os.path.abspath(transform_path))
    basename = os.path.basename(transform_path)
    if "_CT_to_US" in basename:
        case_id = basename.split("_CT_to_US", 1)[0]
    elif basename.endswith(".tfm"):
        case_id = os.path.splitext(basename)[0].split("_", 1)[0]
    else:
        return None
    return os.path.join(directory, f"{case_id}_pca_icp_report.json")


def _matrix_from_report_payload(payload: dict) -> Optional[np.ndarray]:
    for key in ("resample_matrix", "final_matrix"):
        raw = payload.get(key)
        if raw is None:
            continue
        matrix = np.asarray(raw, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(f"Expected 4x4 matrix for '{key}', got shape {matrix.shape}.")
        return matrix
    return None


def load_pca_icp_report_matrix(transform_path: str) -> Optional[np.ndarray]:
    report_path = resolve_pca_icp_report_path(transform_path)
    if report_path is None or not os.path.exists(report_path):
        return None
    with open(report_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"PCA+ICP report must be a JSON object: {report_path}")
    return _matrix_from_report_payload(payload)


def apply_affine_matrix(points: np.ndarray, matrix_4x4: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("apply_affine_matrix expects points with shape Nx3.")
    matrix_4x4 = np.asarray(matrix_4x4, dtype=np.float64)
    if matrix_4x4.shape != (4, 4):
        raise ValueError(f"apply_affine_matrix expects a 4x4 matrix, got {matrix_4x4.shape}.")
    homogeneous = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    transformed = (matrix_4x4 @ homogeneous.T).T
    return transformed[:, :3]


def _inverse_itk_transform(transform: itk.Transform) -> itk.Transform:
    if hasattr(transform, "GetInverseTransform"):
        return transform.GetInverseTransform()
    inverse = transform.GetInverse()
    if inverse is None:
        raise ValueError("ITK transform inverse could not be computed.")
    return inverse


def transform_points_with_itk(points: np.ndarray, transform: itk.Transform) -> np.ndarray:
    transformed = [
        np.asarray(transform.TransformPoint(tuple(float(x) for x in point)), dtype=np.float64)
        for point in points
    ]
    return np.asarray(transformed, dtype=np.float64)


def transform_moving_landmarks_initial(
    points: np.ndarray,
    transform_path: str,
    *,
    prefer_report_json: bool = True,
    invert: bool = False,
    invert_itk: bool = False,
) -> tuple[np.ndarray, str]:
    if prefer_report_json:
        matrix = load_pca_icp_report_matrix(transform_path)
        if matrix is not None:
            if invert:
                matrix = np.linalg.inv(matrix)
            return apply_affine_matrix(points, matrix), (
                "report_json_inverted" if invert else "report_json"
            )

    transform = itk.transformread(transform_path)[0]
    if invert or invert_itk:
        transform = _inverse_itk_transform(transform)
    transformed = transform_points_with_itk(points, transform)
    if invert or invert_itk:
        return transformed, "itk_tfm_inverted"
    return transformed, "itk_tfm"


def recommend_invert_from_probe(results: Dict[str, float]) -> bool:
    """Return True when inverted report_json gives lower baseline than direct."""
    direct = results.get("report_json", float("inf"))
    inverted = results.get("report_json_inverted", float("inf"))
    return inverted < direct


def probe_landmark_transforms(
    native_points: np.ndarray,
    fixed_points: np.ndarray,
    transform_path: str,
) -> Dict[str, float]:
    results: Dict[str, float] = {}
    results["native"] = mean_point_distance_mm(native_points, fixed_points)

    matrix = load_pca_icp_report_matrix(transform_path)
    if matrix is not None:
        aligned = apply_affine_matrix(native_points, matrix)
        results["report_json"] = mean_point_distance_mm(aligned, fixed_points)
        aligned_inv = apply_affine_matrix(native_points, np.linalg.inv(matrix))
        results["report_json_inverted"] = mean_point_distance_mm(aligned_inv, fixed_points)

    transform = itk.transformread(transform_path)[0]
    fwd = transform_points_with_itk(native_points, transform)
    results["itk_tfm"] = mean_point_distance_mm(fwd, fixed_points)
    inv = transform_points_with_itk(native_points, _inverse_itk_transform(transform))
    results["itk_tfm_inverted"] = mean_point_distance_mm(inv, fixed_points)
    return results
