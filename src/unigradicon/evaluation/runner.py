import csv
import hashlib
import json
import os
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import itk
import numpy as np

import icon_registration.itk_wrapper
import unigradicon
from .config import EvalCase, MethodEvalConfig, load_test_manifest
from .landmark_transforms import transform_moving_landmarks_initial
from .metrics import (
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
    save_points,
)


RESULT_COLUMNS = [
    "timestamp",
    "method_name",
    "case_id",
    "status",
    "elapsed_sec",
    "tre",
    "tre_baseline_raw_mm",
    "tre_phi_only_mm",
    "chamfer",
    "chamfer_half",
    "dice",
    "mi",
    "hd95",
    "transform_path",
    "chamfer_points_source",
    "fixed_surface_points_count",
    "moving_surface_points_count",
    "error",
]


_DEBUG_LOG_PATH = "C:/Users/gabridal/Documents/uniGradICON/debug-d67df2.log"
_DEBUG_SESSION_ID = "d67df2"


def _debug_log(
    hypothesis_id: str,
    location: str,
    message: str,
    data: Dict[str, object],
    run_id: str = "pre-fix",
) -> None:
    payload = {
        "sessionId": _DEBUG_SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "id": f"log_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}",
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass


def _image_meta(image: itk.Image) -> Dict[str, object]:
    size = tuple(int(v) for v in image.GetLargestPossibleRegion().GetSize())
    spacing = tuple(float(v) for v in image.GetSpacing())
    origin = tuple(float(v) for v in image.GetOrigin())
    direction = np.array(image.GetDirection(), dtype=np.float64).reshape(3, 3).tolist()
    return {
        "size": size,
        "spacing": spacing,
        "origin": origin,
        "direction": direction,
    }



def _log_progress(message: str) -> None:
    print(f"[unigradicon-eval] {message}", flush=True)


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _append_rows(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: str, payload: object) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _safe_float(value: object) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _summary_rows(method_name: str, rows: Iterable[Dict[str, str]]) -> List[Dict[str, object]]:
    rows = list(rows)
    success = [row for row in rows if row.get("status") == "ok"]
    summary = []
    for metric in ("tre", "chamfer", "chamfer_half", "dice", "mi", "hd95"):
        values = np.array([_safe_float(row.get(metric)) for row in success], dtype=np.float64)
        values = values[~np.isnan(values)]
        if values.size == 0:
            continue
        summary.append(
            {
                "method_name": method_name,
                "metric": metric,
                "n": int(values.size),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "median": float(np.median(values)),
                "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
            }
        )
    return summary


def _write_summary(path: str, method_name: str, rows: List[Dict[str, str]]) -> None:
    summary = _summary_rows(method_name, rows)
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["method_name", "metric", "n", "mean", "std", "median", "iqr"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)


def merge_method_summaries(summary_paths: List[str], output_path: str) -> None:
    rows: List[Dict[str, str]] = []
    for path in summary_paths:
        if not os.path.exists(path):
            continue
        rows.extend(_read_csv_rows(path))
    rows.sort(key=lambda row: (row.get("method_name", ""), row.get("metric", "")))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["method_name", "metric", "n", "mean", "std", "median", "iqr"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _resample_to_fixed(
    moving: itk.Image,
    fixed: itk.Image,
    transform: itk.Transform,
    nearest_neighbor: bool = False,
) -> itk.Image:
    moving, maybe_cast_back = unigradicon.maybe_cast(moving)
    interpolator = (
        itk.NearestNeighborInterpolateImageFunction.New(moving)
        if nearest_neighbor
        else itk.LinearInterpolateImageFunction.New(moving)
    )
    warped = itk.resample_image_filter(
        moving,
        transform=transform,
        interpolator=interpolator,
        use_reference_image=True,
        reference_image=fixed,
    )
    return maybe_cast_back(warped)


def _spacing_for_target_shape(
    image: itk.Image, target_shape: Tuple[int, int, int]
) -> Tuple[float, float, float]:
    size = tuple(int(v) for v in image.GetLargestPossibleRegion().GetSize())
    spacing = tuple(float(v) for v in image.GetSpacing())
    output_spacing = []
    for src_size, src_spacing, tgt_size in zip(size, spacing, target_shape):
        # Preserve physical extent similarly to align-corners style resampling.
        if src_size <= 1 or tgt_size <= 1:
            output_spacing.append(src_spacing)
        else:
            output_spacing.append(src_spacing * float(src_size - 1) / float(tgt_size - 1))
    return tuple(output_spacing)


def _resample_image_to_shape(
    image: itk.Image,
    target_shape: Tuple[int, int, int],
    nearest_neighbor: bool = False,
) -> itk.Image:
    # Keep label maps in their native integer type, but cast intensity images to float
    # before linear interpolation to avoid ITK default-pixel type mismatches.
    if nearest_neighbor:
        maybe_cast_back = lambda x: x
    else:
        maybe_cast_back = lambda x: x
        image = itk.CastImageFilter[type(image), itk.Image[itk.F, 3]].New()(image)
    interpolator = (
        itk.NearestNeighborInterpolateImageFunction.New(image)
        if nearest_neighbor
        else itk.LinearInterpolateImageFunction.New(image)
    )
    identity = itk.IdentityTransform[itk.D, 3].New()
    output_spacing = _spacing_for_target_shape(image, target_shape)
    resampled = itk.resample_image_filter(
        image,
        transform=identity,
        interpolator=interpolator,
        size=tuple(int(v) for v in target_shape),
        output_spacing=output_spacing,
        output_origin=image.GetOrigin(),
        output_direction=image.GetDirection(),
        default_pixel_value=0 if nearest_neighbor else 0.0,
    )
    return maybe_cast_back(resampled)


def _cache_path(cache_dir: str, source_path: str, cache_tag: str) -> str:
    stat = os.stat(source_path)
    signature = f"{source_path}|{stat.st_mtime_ns}|{stat.st_size}|{cache_tag}"
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"{digest}.nii.gz")


def _preprocess_image(
    image_path: str,
    modality: str,
    mask_path: Optional[str],
    ct_window: Optional[tuple],
    quantile_range: Optional[tuple],
    cache_dir: Optional[str],
    input_shape: Optional[Tuple[int, int, int]] = None,
) -> itk.Image:
    # Keep evaluation config modality values user-friendly while matching preprocess expectations.
    modality_for_preprocess = "mri" if modality == "us" else modality
    cache_tag = f"{modality_for_preprocess}|{ct_window}|{quantile_range}|{mask_path}|{input_shape}"
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = _cache_path(cache_dir, image_path, cache_tag)
        if os.path.exists(cache_path):
            return itk.imread(cache_path)

    image = itk.imread(image_path)
    mask = itk.imread(mask_path) if mask_path else None
    # Match MultiGradICON finetuning order: resize first, preprocess second.
    if input_shape is not None:
        image = _resample_image_to_shape(
            image,
            target_shape=input_shape,
            nearest_neighbor=False,
        )
        if mask is not None:
            mask = _resample_image_to_shape(
                mask,
                target_shape=input_shape,
                nearest_neighbor=True,
            )
    processed = unigradicon.preprocess(
        image=image,
        modality=modality_for_preprocess,
        mask=mask,
        ct_window=ct_window,
        quantile_range=quantile_range,
    )
    if cache_dir is not None:
        itk.imwrite(processed, cache_path)
    return processed


def _maybe_apply_initial_transform(
    case: EvalCase,
    apply_to_images: bool,
    fixed_image: itk.Image,
    moving_image: itk.Image,
    moving_seg: Optional[itk.Image],
    moving_mask: Optional[itk.Image],
) -> Dict[str, object]:
    initial_transform = None
    if case.initial_transform:
        initial_transform = itk.transformread(case.initial_transform)[0]
        if apply_to_images:
            moving_image = _resample_to_fixed(
                moving_image, fixed_image, initial_transform, nearest_neighbor=False
            )
            if moving_seg is not None:
                moving_seg = _resample_to_fixed(
                    moving_seg, fixed_image, initial_transform, nearest_neighbor=True
                )
            if moving_mask is not None:
                moving_mask = _resample_to_fixed(
                    moving_mask, fixed_image, initial_transform, nearest_neighbor=True
                )
    return {
        "moving_image": moving_image,
        "moving_seg": moving_seg,
        "moving_mask": moving_mask,
        "initial_transform": initial_transform,
    }


def _transform_points(points: np.ndarray, transform: itk.Transform) -> np.ndarray:
    transformed = [
        np.asarray(transform.TransformPoint(tuple(float(x) for x in point)), dtype=np.float64)
        for point in points
    ]
    return np.asarray(transformed, dtype=np.float64)


def _inverse_transform(transform: itk.Transform) -> itk.Transform:
    if hasattr(transform, "GetInverseTransform"):
        return transform.GetInverseTransform()
    inverse = transform.GetInverse()
    if inverse is None:
        raise ValueError("Initial transform inverse could not be computed for TRE.")
    return inverse


def _warn_tre_landmark_issues(
    case_id: str,
    fixed_points: np.ndarray,
    moving_points: np.ndarray,
    fixed_image: itk.Image,
    moving_image: itk.Image,
    baseline_raw_mm: float,
    threshold_mm: float = 50.0,
) -> None:
    fixed_bounds_min, fixed_bounds_max = image_physical_bounds(fixed_image)
    moving_bounds_min, moving_bounds_max = image_physical_bounds(moving_image)
    fixed_inside = points_inside_physical_bounds(
        fixed_points, fixed_bounds_min, fixed_bounds_max, margin_mm=1.0
    )
    moving_inside = points_inside_physical_bounds(
        moving_points, moving_bounds_min, moving_bounds_max, margin_mm=1.0
    )
    if not fixed_inside:
        _log_progress(
            f"WARNING {case_id}: fixed landmarks fall outside fixed image physical bounds."
        )
    if not moving_inside:
        _log_progress(
            f"WARNING {case_id}: moving landmarks fall outside moving image physical bounds."
        )
    if baseline_raw_mm > threshold_mm:
        _log_progress(
            f"WARNING {case_id}: baseline_raw={baseline_raw_mm:.3f} mm exceeds "
            f"{threshold_mm:.1f} mm. Run "
            "`python scripts/align_eval_landmarks.py validate --manifest <path>` "
            "and align CT landmarks if needed."
        )


def _run_case(
    net,
    settings: MethodEvalConfig,
    case: EvalCase,
    method_output_dir: str,
) -> Dict[str, object]:
    start = time.time()
    # #region agent log
    _debug_log(
        hypothesis_id="H5",
        location="runner.py:_run_case:start",
        message="case_start",
        data={
            "case_id": case.case_id,
            "input_shape": settings.input_shape,
            "apply_initial_transform_to_images": settings.apply_initial_transform_to_images,
            "apply_initial_transform_to_landmarks": settings.apply_initial_transform_to_landmarks,
            "invert_initial_transform_for_tre": settings.invert_initial_transform_for_tre,
        },
    )
    # #endregion
    _log_progress(f"{case.case_id}: loading inputs")
    metrics = {
        name: float("nan")
        for name in ("tre", "tre_baseline_raw_mm", "tre_phi_only_mm", "chamfer", "chamfer_half", "dice", "mi", "hd95")
    }
    transform_path = ""
    chamfer_points_source = ""
    fixed_surface_points_count = 0
    moving_surface_points_count = 0

    cache_dir = (
        os.path.join(method_output_dir, "preprocessed_cache") if settings.cache_preprocessed else None
    )

    fixed_mask = itk.imread(case.fixed.mask) if case.fixed.mask else None
    moving_mask = itk.imread(case.moving.mask) if case.moving.mask else None
    fixed_seg = itk.imread(case.fixed.segmentation) if case.fixed.segmentation else None
    moving_seg = itk.imread(case.moving.segmentation) if case.moving.segmentation else None
    if settings.input_shape is not None:
        _log_progress(f"{case.case_id}: resampling raw inputs to {settings.input_shape}")
        if fixed_mask is not None:
            fixed_mask = _resample_image_to_shape(
                fixed_mask,
                target_shape=settings.input_shape,
                nearest_neighbor=True,
            )
        if moving_mask is not None:
            moving_mask = _resample_image_to_shape(
                moving_mask,
                target_shape=settings.input_shape,
                nearest_neighbor=True,
            )
        if fixed_seg is not None:
            fixed_seg = _resample_image_to_shape(
                fixed_seg,
                target_shape=settings.input_shape,
                nearest_neighbor=True,
            )
        if moving_seg is not None:
            moving_seg = _resample_image_to_shape(
                moving_seg,
                target_shape=settings.input_shape,
                nearest_neighbor=True,
            )

    _log_progress(f"{case.case_id}: preprocessing fixed/moving images")
    fixed_pre = _preprocess_image(
        case.fixed.image,
        case.fixed.modality,
        case.fixed.mask if settings.input_masking else None,
        settings.ct_window,
        settings.quantile_range,
        cache_dir,
        input_shape=settings.input_shape,
    )
    moving_pre = _preprocess_image(
        case.moving.image,
        case.moving.modality,
        case.moving.mask if settings.input_masking else None,
        settings.ct_window,
        settings.quantile_range,
        cache_dir,
        input_shape=settings.input_shape,
    )
    moved = _maybe_apply_initial_transform(
        case=case,
        apply_to_images=settings.apply_initial_transform_to_images,
        fixed_image=fixed_pre,
        moving_image=moving_pre,
        moving_seg=moving_seg,
        moving_mask=moving_mask,
    )
    moving_pre = moved["moving_image"]
    moving_seg = moved["moving_seg"]
    moving_mask = moved["moving_mask"]
    initial_transform = moved["initial_transform"]
    if initial_transform is not None:
        mode = "images+landmarks" if settings.apply_initial_transform_to_images else "landmarks-only"
        _log_progress(f"{case.case_id}: initial transform loaded ({mode})")
    # #region agent log
    _debug_log(
        hypothesis_id="H5",
        location="runner.py:_run_case:post_preprocess",
        message="preprocessed_image_metadata",
        data={
            "case_id": case.case_id,
            "fixed_pre": _image_meta(fixed_pre),
            "moving_pre": _image_meta(moving_pre),
            "has_initial_transform": initial_transform is not None,
        },
    )
    # #endregion

    needs_seg = settings.dice_loss_weight > 0.0
    reg_mode = "register_pair_with_mask" if settings.loss_function_masking or needs_seg else "register_pair"
    _log_progress(
        f"{case.case_id}: registration started ({reg_mode}, io_iterations={settings.io_iterations}, io_lr={settings.io_lr})"
    )
    if settings.loss_function_masking or needs_seg:
        phi_ab, phi_ba = icon_registration.itk_wrapper.register_pair_with_mask(
            net,
            moving_pre,
            fixed_pre,
            mask_A=moving_mask if settings.loss_function_masking else None,
            mask_B=fixed_mask if settings.loss_function_masking else None,
            finetune_steps=settings.io_iterations,
            learning_rate=settings.io_lr,
            segmentation_A=moving_seg if needs_seg else None,
            segmentation_B=fixed_seg if needs_seg else None,
        )
    else:
        phi_ab, phi_ba = icon_registration.itk_wrapper.register_pair(
            net,
            moving_pre,
            fixed_pre,
            finetune_steps=settings.io_iterations,
            learning_rate=settings.io_lr,
        )
    _log_progress(f"{case.case_id}: registration finished")

    if settings.save_transforms:
        transform_dir = os.path.join(method_output_dir, "transforms")
        os.makedirs(transform_dir, exist_ok=True)
        transform_path = os.path.join(transform_dir, f"{case.case_id}.hdf5")
        itk.transformwrite([phi_ab], transform_path)
        _log_progress(f"{case.case_id}: transform saved -> {transform_path}")

    enabled = set(settings.metrics)
    _log_progress(f"{case.case_id}: computing metrics -> {sorted(enabled)}")
    if "mi" in enabled and case.metrics.mi:
        warped_moving = _resample_to_fixed(moving_pre, fixed_pre, phi_ab, nearest_neighbor=False)
        metrics["mi"] = compute_mutual_information(fixed_pre, warped_moving, fixed_mask=fixed_mask)
    if fixed_seg is not None and moving_seg is not None:
        warped_moving_seg = _resample_to_fixed(moving_seg, fixed_seg, phi_ab, nearest_neighbor=True)
        if "dice" in enabled and case.metrics.dice:
            metrics["dice"] = compute_dice_score(fixed_seg, warped_moving_seg)
        if "hd95" in enabled and case.metrics.hd95:
            metrics["hd95"] = compute_hd95_mm(fixed_seg, warped_moving_seg)
    if (
        "tre" in enabled
        and case.metrics.tre
        and case.fixed.landmarks is not None
        and case.moving.landmarks is not None
    ):
        fixed_points = load_points(case.fixed.landmarks)
        moving_points = load_points(case.moving.landmarks)
        moving_points_for_tre = moving_points
        diagnostics: Dict[str, float] = {}
        if settings.apply_initial_transform_to_landmarks and case.initial_transform is not None:
            moving_points_for_tre, transform_source = transform_moving_landmarks_initial(
                moving_points,
                case.initial_transform,
                prefer_report_json=True,
                invert=settings.invert_initial_transform_for_tre,
                invert_itk=settings.invert_initial_transform_for_tre,
            )
            _log_progress(
                f"{case.case_id}: TRE initial landmark transform source={transform_source}"
            )
            init_direct, _ = transform_moving_landmarks_initial(
                moving_points,
                case.initial_transform,
                prefer_report_json=True,
                invert=False,
            )
            diagnostics["initial_direct_only"] = mean_point_distance_mm(init_direct, fixed_points)
            init_inverse, _ = transform_moving_landmarks_initial(
                moving_points,
                case.initial_transform,
                prefer_report_json=True,
                invert=True,
            )
            diagnostics["initial_inverse_only"] = mean_point_distance_mm(init_inverse, fixed_points)
            # phi_ba maps moving-space points into fixed space (phi_ab is the
            # fixed->moving map used to resample the moving image onto the fixed grid).
            phi_after_init_direct = _transform_points(init_direct, phi_ba)
            diagnostics["phi_after_initial_direct"] = mean_point_distance_mm(
                phi_after_init_direct, fixed_points
            )
            phi_after_init_inverse = _transform_points(init_inverse, phi_ba)
            diagnostics["phi_after_initial_inverse"] = mean_point_distance_mm(
                phi_after_init_inverse, fixed_points
            )
        elif case.initial_transform is not None:
            _log_progress(f"{case.case_id}: TRE using pre-aligned landmarks (initial transform skipped)")

        tre_baseline_raw_mm = mean_point_distance_mm(moving_points_for_tre, fixed_points)
        diagnostics["baseline_raw"] = tre_baseline_raw_mm
        diagnostics["native_raw"] = mean_point_distance_mm(moving_points, fixed_points)
        _warn_tre_landmark_issues(
            case_id=case.case_id,
            fixed_points=fixed_points,
            moving_points=moving_points_for_tre,
            fixed_image=fixed_pre,
            moving_image=moving_pre,
            baseline_raw_mm=tre_baseline_raw_mm,
        )
        metrics["tre"] = compute_tre_mm(
            moving_points=moving_points_for_tre,
            fixed_points=fixed_points,
            predicted_transform=phi_ba,
            initial_transform=None,
        )
        metrics["tre_baseline_raw_mm"] = tre_baseline_raw_mm
        metrics["tre_phi_only_mm"] = metrics["tre"]
        # #region agent log
        _debug_log(
            hypothesis_id="H1_H2_H4",
            location="runner.py:_run_case:tre_diagnostics",
            message="tre_transform_composition_diagnostics",
            data={
                "case_id": case.case_id,
                "invert_initial_transform_for_tre": settings.invert_initial_transform_for_tre,
                "apply_initial_transform_to_landmarks": settings.apply_initial_transform_to_landmarks,
                "apply_initial_transform_to_images": settings.apply_initial_transform_to_images,
                "diagnostics_mm": diagnostics,
            },
        )
        _debug_log(
            hypothesis_id="H1_H2_H4",
            location="runner.py:_run_case:tre_result",
            message="tre_result",
            data={
                "case_id": case.case_id,
                "tre_mm": float(metrics["tre"]),
                "tre_baseline_raw_mm": float(tre_baseline_raw_mm),
            },
        )
        # #endregion
    if "chamfer" in enabled and case.metrics.chamfer:
        if fixed_seg is None or moving_seg is None:
            raise ValueError(
                "Chamfer requires both fixed and moving segmentations for boundary extraction."
            )
        fixed_points = _get_or_create_surface_points(
            case=case,
            segmentation=fixed_seg,
            surface_role="fixed",
            method_output_dir=method_output_dir,
            settings=settings,
        )
        moving_points = _get_or_create_surface_points(
            case=case,
            segmentation=moving_seg,
            surface_role="moving",
            method_output_dir=method_output_dir,
            settings=settings,
        )
        # phi_ba maps moving-space surface points into fixed space.
        moved_points = _transform_points(moving_points, phi_ba)
        fixed_surface_points_count = int(fixed_points.shape[0])
        moving_surface_points_count = int(moving_points.shape[0])
        metrics["chamfer"], metrics["chamfer_half"] = compute_chamfer_mm(moved_points, fixed_points)
        chamfer_points_source = "segmentation"

    elapsed = time.time() - start
    _log_progress(f"{case.case_id}: completed in {elapsed:.2f}s")
    return {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "status": "ok",
        "elapsed_sec": round(elapsed, 4),
        "transform_path": transform_path,
        "chamfer_points_source": chamfer_points_source,
        "fixed_surface_points_count": fixed_surface_points_count,
        "moving_surface_points_count": moving_surface_points_count,
        **metrics,
    }


def _get_or_create_surface_points(
    case: EvalCase,
    segmentation: itk.Image,
    surface_role: str,
    method_output_dir: str,
    settings: MethodEvalConfig,
) -> np.ndarray:
    cache_dir = os.path.join(method_output_dir, "cache", "surface_points")
    os.makedirs(cache_dir, exist_ok=True)
    cache_name = (
        f"{case.case_id}_{surface_role}_label{settings.chamfer_label_value}_"
        f"max{settings.chamfer_max_surface_points}_seed{settings.chamfer_random_seed}.txt"
    )
    cache_path = os.path.join(cache_dir, cache_name)
    if settings.chamfer_cache_surface_points and os.path.exists(cache_path):
        return load_points(cache_path)

    points = generate_surface_points_from_segmentation(
        segmentation=segmentation,
        label_value=settings.chamfer_label_value,
        max_points=settings.chamfer_max_surface_points,
        seed=settings.chamfer_random_seed,
    )
    if settings.chamfer_cache_surface_points:
        save_points(cache_path, points)
    return points


def run_evaluation(settings: MethodEvalConfig) -> str:
    cases = load_test_manifest(settings.test_manifest)
    method_output_dir = os.path.join(settings.output_root, settings.method_name)
    os.makedirs(method_output_dir, exist_ok=True)
    _log_progress(
        f"starting method={settings.method_name}, total_cases={len(cases)}, output_dir={method_output_dir}, input_shape={settings.input_shape}"
    )

    loss_fn = unigradicon.make_sim(settings.io_sim)
    net = unigradicon.get_model_from_model_zoo(
        model_name=settings.model,
        loss_fn=loss_fn,
        apply_intensity_conservation_loss=settings.intensity_conservation_loss,
        dice_loss_weight=settings.dice_loss_weight,
        loss_function_masking=settings.loss_function_masking,
        weights_location=settings.network_weights,
    )

    per_case_csv = os.path.join(method_output_dir, "per_case_metrics.csv")
    failures_json = os.path.join(method_output_dir, "failures.json")
    summary_csv = os.path.join(method_output_dir, "summary.csv")

    existing_rows = _read_csv_rows(per_case_csv) if settings.resume else []
    completed_case_ids = {
        row.get("case_id", "")
        for row in existing_rows
        if row.get("status") == "ok"
    }
    failures = []
    pending_rows: List[Dict[str, object]] = []

    sorted_cases = sorted(cases, key=lambda c: c.case_id)
    for case_idx, case in enumerate(sorted_cases, start=1):
        if settings.resume and case.case_id in completed_case_ids:
            _log_progress(f"[{case_idx}/{len(sorted_cases)}] {case.case_id}: skipped (already completed)")
            continue
        _log_progress(f"[{case_idx}/{len(sorted_cases)}] {case.case_id}: started")
        try:
            result = _run_case(net, settings, case, method_output_dir)
            row = {
                "method_name": settings.method_name,
                "case_id": case.case_id,
                "error": "",
                **result,
            }
        except Exception as exc:  # Keep long batch runs alive across case failures.
            _log_progress(f"[{case_idx}/{len(sorted_cases)}] {case.case_id}: failed -> {exc}")
            row = {
                "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
                "method_name": settings.method_name,
                "case_id": case.case_id,
                "status": "failed",
                "elapsed_sec": 0.0,
                "tre": float("nan"),
                "tre_baseline_raw_mm": float("nan"),
                "tre_phi_only_mm": float("nan"),
                "chamfer": float("nan"),
                "chamfer_half": float("nan"),
                "dice": float("nan"),
                "mi": float("nan"),
                "hd95": float("nan"),
                "transform_path": "",
                "chamfer_points_source": "",
                "fixed_surface_points_count": 0,
                "moving_surface_points_count": 0,
                "error": str(exc),
            }
            failures.append({"case_id": case.case_id, "error": str(exc), "case": asdict(case)})
        pending_rows.append(row)
        if len(pending_rows) >= settings.flush_every:
            _append_rows(per_case_csv, pending_rows)
            _log_progress(
                f"flushed {len(pending_rows)} row(s) to per_case_metrics.csv (flush_every={settings.flush_every})"
            )
            pending_rows = []
            _write_json(failures_json, failures)
            _log_progress("updated failures.json")

    if pending_rows:
        _append_rows(per_case_csv, pending_rows)
        _log_progress(f"flushed remaining {len(pending_rows)} row(s) to per_case_metrics.csv")
    _write_json(failures_json, failures)
    _log_progress("updated failures.json")

    final_rows = _read_csv_rows(per_case_csv)
    _write_summary(summary_csv, settings.method_name, final_rows)
    _log_progress(f"summary written -> {summary_csv}")
    _log_progress("evaluation finished")
    return method_output_dir
