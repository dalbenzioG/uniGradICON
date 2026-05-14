#!/usr/bin/env python3
"""
Pre-compute rigid, similarity, or affine transforms from CT/US kidney masks.

For rigid/similarity: uses SimpleITK CenteredTransformInitializer with MOMENTS
to align the centers of mass of CT (fixed) and US (moving) masks.

For affine: converts masks to signed distance maps, runs Mean Squares
registration to recover scaling and shear. Supports --rigid_init_dir to
initialize from pre-computed rigid transforms.

Expected naming (supported patterns):
  CT mask: {patient}{side}_seg.nii.gz  (e.g., 200L_seg.nii.gz)
  US mask: {patient}{side}_seg.nii.gz or {patient}{side}_maskUS.nii.gz

Usage example:
  python utils/precompute_rigid_from_masks.py \
    --ct_mask_dir C:/Users/gabridal/Documents/TRUSTED_dataset/CT_DATA/CT_masks_cropped \
    --us_mask_dir C:/Users/gabridal/Documents/TRUSTED_dataset/US_DATA/US_masks \
    --output_dir  C:/Users/gabridal/Documents/TRUSTED_dataset/rigid_mask_transforms

Note:
  This script saves BOTH directions for each pair:
    {key}_US_to_CT_{suffix}.tfm and {key}_CT_to_US_{suffix}.tfm
"""

from __future__ import annotations

import os
import glob
import argparse
import pickle
from typing import Dict, Optional, Tuple

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


def extract_patient_side(filename: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract (patient_id, side) from TRUSTED mask filename."""
    base = filename.replace(".nii.gz", "").replace(".nii", "")
    for suffix, side in (
        ("L_seg", "L"),
        ("R_seg", "R"),
        ("L_maskUS", "L"),
        ("R_maskUS", "R"),
        ("L_mask", "L"),
        ("R_mask", "R"),
    ):
        if base.endswith(suffix):
            patient_id = base.replace(suffix, "").strip("_")
            return patient_id, side
    return None, None


def load_binary_mask(path: str) -> sitk.Image:
    mask = sitk.ReadImage(path, sitk.sitkUInt8)
    mask = sitk.Cast(mask > 0, sitk.sitkUInt8)
    return mask


def mask_is_empty(mask: sitk.Image) -> bool:
    arr = sitk.GetArrayViewFromImage(mask)
    return bool(arr.sum() == 0)


def mask_centroid_mm(mask: sitk.Image) -> Tuple[float, float, float]:
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask)
    return tuple(stats.GetCentroid(1))


def centroid_distance_mm(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return float(np.linalg.norm(np.array(a, dtype=np.float64) - np.array(b, dtype=np.float64)))


def mask_to_signed_distance_map(mask: sitk.Image) -> sitk.Image:
    """Convert binary mask to signed distance map (Euclidean, in mm)."""
    # SignedMaurerDistanceMap: inside=negative, outside=positive by default
    sdf = sitk.SignedMaurerDistanceMap(
        mask,
        insideIsPositive=False,
        squaredDistance=False,
        useImageSpacing=True,
        backgroundValue=0.0,
    )
    return sitk.Cast(sdf, sitk.sitkFloat32)


def _rigid_to_affine(rigid_tx: sitk.Transform) -> sitk.AffineTransform:
    """Convert rigid/similarity transform to AffineTransform."""
    affine = sitk.AffineTransform(3)
    affine.SetMatrix(rigid_tx.GetMatrix())
    affine.SetTranslation(rigid_tx.GetTranslation())
    return affine


def compute_affine_from_distance_maps(
    fixed_sdf: sitk.Image,
    moving_sdf: sitk.Image,
    initial_transform: Optional[sitk.Transform] = None,
    num_iterations: int = 300,
    learning_rate: float = 1.0,
) -> Tuple[sitk.Transform, sitk.Transform]:
    """
    Run affine registration on signed distance maps with Mean Squares metric.

    Returns (us_to_ct, ct_to_us). Registration returns fixed->moving (CT->US);
    we invert to get moving->fixed (US->CT).
    """
    if initial_transform is not None:
        # Registration expects fixed->moving (CT->US). Our rigid US_to_CT maps moving->fixed.
        ct_to_us_rigid = initial_transform.GetInverse()
        initial_affine = _rigid_to_affine(ct_to_us_rigid)
    else:
        initial_affine = sitk.AffineTransform(3)
        initial_affine = sitk.CenteredTransformInitializer(
            fixed_sdf,
            moving_sdf,
            initial_affine,
            sitk.CenteredTransformInitializerFilter.MOMENTS,
        )

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMeanSquares()
    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=learning_rate,
        minStep=1e-4,
        numberOfIterations=num_iterations,
    )
    R.SetInterpolator(sitk.sitkLinear)
    R.SetInitialTransform(initial_affine, inPlace=False)
    R.SetFixedImage(fixed_sdf)
    R.SetMovingImage(moving_sdf)

    tx_ct_to_us = R.Execute(fixed_sdf, moving_sdf)
    tx_us_to_ct = tx_ct_to_us.GetInverse()
    return tx_us_to_ct, tx_ct_to_us


def compute_mask_initializer_transform(
    fixed_mask: sitk.Image,
    moving_mask: sitk.Image,
    transform_kind: str,
) -> sitk.Transform:
    if transform_kind == "similarity":
        t0 = sitk.Similarity3DTransform()
    else:
        t0 = sitk.Euler3DTransform()

    tx = sitk.CenteredTransformInitializer(
        fixed_mask,
        moving_mask,
        t0,
        sitk.CenteredTransformInitializerFilter.MOMENTS,
    )
    return tx


def precompute_all_transforms(
    ct_mask_dir: str,
    us_mask_dir: str,
    output_dir: str,
    transform_kind: str = "rigid",
    transform_suffix: str = "rigid_mask",
    force_recompute: bool = False,
    debug_centroids: bool = False,
    rigid_init_dir: Optional[str] = None,
    rigid_init_suffix: str = "rigid_mask",
    affine_iterations: int = 300,
    affine_learning_rate: float = 1.0,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    ct_files = sorted(glob.glob(os.path.join(ct_mask_dir, "*L_seg.nii*"))) + \
               sorted(glob.glob(os.path.join(ct_mask_dir, "*R_seg.nii*")))
    us_files = (
        sorted(glob.glob(os.path.join(us_mask_dir, "*L_seg.nii*")))
        + sorted(glob.glob(os.path.join(us_mask_dir, "*R_seg.nii*")))
        + sorted(glob.glob(os.path.join(us_mask_dir, "*L_maskUS.nii*")))
        + sorted(glob.glob(os.path.join(us_mask_dir, "*R_maskUS.nii*")))
        + sorted(glob.glob(os.path.join(us_mask_dir, "*L_mask.nii*")))
        + sorted(glob.glob(os.path.join(us_mask_dir, "*R_mask.nii*")))
    )

    ct_dict: Dict[str, str] = {}
    for ct_file in ct_files:
        filename = os.path.basename(ct_file)
        patient_id, side = extract_patient_side(filename)
        if patient_id and side:
            ct_dict[f"{patient_id}_{side}"] = ct_file

    us_dict: Dict[str, str] = {}
    for us_file in us_files:
        filename = os.path.basename(us_file)
        patient_id, side = extract_patient_side(filename)
        if patient_id and side:
            us_dict[f"{patient_id}_{side}"] = us_file

    pairs = []
    for key in sorted(ct_dict.keys()):
        if key in us_dict:
            pairs.append((key, ct_dict[key], us_dict[key]))

    print(f"Found {len(pairs)} CT-US mask pairs")
    print(f"Computing {transform_kind} transforms (suffix={transform_suffix})...")

    transforms_us_to_ct: Dict[str, str] = {}
    transforms_ct_to_us: Dict[str, str] = {}
    failed_pairs = []

    desc = "Computing mask initializers" if transform_kind != "affine" else "Affine registration"
    for key, ct_mask_path, us_mask_path in tqdm(pairs, desc=desc):
        us_to_ct_file = os.path.join(output_dir, f"{key}_US_to_CT_{transform_suffix}.tfm")
        ct_to_us_file = os.path.join(output_dir, f"{key}_CT_to_US_{transform_suffix}.tfm")
        if os.path.exists(us_to_ct_file) and os.path.exists(ct_to_us_file) and not force_recompute:
            transforms_us_to_ct[key] = us_to_ct_file
            transforms_ct_to_us[key] = ct_to_us_file
            continue

        try:
            fixed_mask = load_binary_mask(ct_mask_path)
            moving_mask = load_binary_mask(us_mask_path)

            if mask_is_empty(fixed_mask) or mask_is_empty(moving_mask):
                raise RuntimeError("Empty mask")

            if transform_kind == "affine":
                fixed_sdf = mask_to_signed_distance_map(fixed_mask)
                moving_sdf = mask_to_signed_distance_map(moving_mask)

                initial_rigid: Optional[sitk.Transform] = None
                if rigid_init_dir:
                    rigid_path = os.path.join(
                        rigid_init_dir, f"{key}_US_to_CT_{rigid_init_suffix}.tfm"
                    )
                    if os.path.exists(rigid_path):
                        initial_rigid = sitk.ReadTransform(rigid_path)
                    # else: fall back to moment initialization

                tx, tx_inverse = compute_affine_from_distance_maps(
                    fixed_sdf,
                    moving_sdf,
                    initial_transform=initial_rigid,
                    num_iterations=affine_iterations,
                    learning_rate=affine_learning_rate,
                )
            else:
                tx = compute_mask_initializer_transform(
                    fixed_mask=fixed_mask,
                    moving_mask=moving_mask,
                    transform_kind=transform_kind,
                )
                try:
                    tx_inverse = tx.GetInverse()
                except Exception:
                    tx_inverse = None

                if debug_centroids:
                    fixed_c = mask_centroid_mm(fixed_mask)
                    moving_c = mask_centroid_mm(moving_mask)
                    before = centroid_distance_mm(fixed_c, moving_c)
                    moved_c = tx.TransformPoint(moving_c)
                    after = centroid_distance_mm(fixed_c, moved_c)
                    try:
                        moved_c_inv = tx_inverse.TransformPoint(moving_c)
                        after_inv = centroid_distance_mm(fixed_c, moved_c_inv)
                    except Exception:
                        after_inv = float("inf")
                    print(
                        f"[{key}] COM fixed={fixed_c}, moving={moving_c}, "
                        f"dist_before={before:.3f} mm, dist_after={after:.3f} mm, "
                        f"dist_after_inv={after_inv:.3f} mm"
                    )

            # By construction, tx maps US (moving) -> CT (fixed).
            sitk.WriteTransform(tx, us_to_ct_file)
            transforms_us_to_ct[key] = us_to_ct_file

            if tx_inverse is not None:
                sitk.WriteTransform(tx_inverse, ct_to_us_file)
                transforms_ct_to_us[key] = ct_to_us_file

        except Exception as e:
            print(f"Warning: Failed to compute transform for {key}: {e}")
            failed_pairs.append(key)

    metadata = {
        "transform_kind": transform_kind,
        "transform_suffix": transform_suffix,
        "transforms_us_to_ct": transforms_us_to_ct,
        "transforms_ct_to_us": transforms_ct_to_us,
        "failed_pairs": failed_pairs,
        "total_pairs": len(pairs),
        "successful_pairs": len(transforms_us_to_ct),
    }

    metadata_file = os.path.join(output_dir, f"transforms_metadata_{transform_suffix}.pkl")
    with open(metadata_file, "wb") as f:
        pickle.dump(metadata, f)

    print(f"\nCompleted: {len(transforms_us_to_ct)}/{len(pairs)} transforms computed successfully")
    if failed_pairs:
        print(f"Failed pairs: {failed_pairs}")
    print(f"Transforms saved to: {output_dir}")
    print(f"Metadata saved to: {metadata_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute rigid/similarity/affine transforms from CT/US kidney masks"
    )
    parser.add_argument("--ct_mask_dir", type=str, required=True, help="Directory containing CT masks")
    parser.add_argument("--us_mask_dir", type=str, required=True, help="Directory containing US masks")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for transforms")
    parser.add_argument(
        "--transform_kind",
        type=str,
        default="rigid",
        choices=["rigid", "similarity", "affine"],
        help="Transform type: rigid, similarity, or affine (default: rigid)",
    )
    parser.add_argument(
        "--transform_suffix",
        type=str,
        default="rigid_mask",
        help="Filename/metadata suffix for transforms (default: rigid_mask)",
    )
    parser.add_argument(
        "--force_recompute",
        action="store_true",
        help="Force recomputation of existing transforms",
    )
    parser.add_argument(
        "--debug_centroids",
        action="store_true",
        help="Print centroid distances before/after initialization",
    )
    parser.add_argument(
        "--rigid_init_dir",
        type=str,
        default=None,
        help="Directory with pre-computed rigid transforms to initialize affine (only when transform_kind=affine)",
    )
    parser.add_argument(
        "--rigid_init_suffix",
        type=str,
        default="rigid_mask",
        help="Suffix for rigid transform filenames (default: rigid_mask)",
    )
    parser.add_argument(
        "--affine_iterations",
        type=int,
        default=300,
        help="Max iterations for affine registration (default: 300)",
    )
    parser.add_argument(
        "--affine_learning_rate",
        type=float,
        default=1.0,
        help="Learning rate for affine optimizer (default: 1.0)",
    )

    args = parser.parse_args()

    precompute_all_transforms(
        ct_mask_dir=args.ct_mask_dir,
        us_mask_dir=args.us_mask_dir,
        output_dir=args.output_dir,
        transform_kind=args.transform_kind,
        transform_suffix=args.transform_suffix,
        force_recompute=args.force_recompute,
        debug_centroids=args.debug_centroids,
        rigid_init_dir=args.rigid_init_dir,
        rigid_init_suffix=args.rigid_init_suffix,
        affine_iterations=args.affine_iterations,
        affine_learning_rate=args.affine_learning_rate,
    )


if __name__ == "__main__":
    main()
