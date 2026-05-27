import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml


VALID_METRICS = ("tre", "chamfer", "dice", "mi", "hd95")
VALID_MODELS = ("unigradicon", "multigradicon")
VALID_MODALITIES = ("ct", "mri", "us")
VALID_SIMILARITIES = ("lncc", "lncc2", "mind")


@dataclass
class ImageSpec:
    image: str
    modality: str
    segmentation: Optional[str] = None
    mask: Optional[str] = None
    landmarks: Optional[str] = None
    surface_points: Optional[str] = None


@dataclass
class CaseMetricsConfig:
    tre: bool = True
    chamfer: bool = True
    dice: bool = True
    mi: bool = True
    hd95: bool = True

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "CaseMetricsConfig":
        if raw is None:
            return cls()
        if "sre" in raw:
            raise ValueError(
                "Metric 'sre' is no longer supported. Use 'chamfer' in case metrics."
            )
        return cls(
            tre=bool(raw.get("tre", True)),
            chamfer=bool(raw.get("chamfer", True)),
            dice=bool(raw.get("dice", True)),
            mi=bool(raw.get("mi", True)),
            hd95=bool(raw.get("hd95", True)),
        )


@dataclass
class EvalCase:
    case_id: str
    fixed: ImageSpec
    moving: ImageSpec
    initial_transform: Optional[str] = None
    ground_truth_transform: Optional[str] = None
    metrics: CaseMetricsConfig = field(default_factory=CaseMetricsConfig)


@dataclass
class MethodEvalConfig:
    method_name: str
    test_manifest: str
    model: str = "multigradicon"
    network_weights: Optional[str] = None
    output_root: str = "results/eval"
    io_iterations: Optional[int] = 50
    io_lr: float = 0.00002
    io_sim: str = "lncc"
    dice_loss_weight: float = 0.0
    loss_function_masking: bool = False
    input_masking: bool = False
    intensity_conservation_loss: bool = False
    input_shape: Optional[Tuple[int, int, int]] = None
    ct_window: Optional[Tuple[float, float]] = None
    quantile_range: Optional[Tuple[float, float]] = None
    metrics: Tuple[str, ...] = VALID_METRICS
    resume: bool = True
    flush_every: int = 1
    save_transforms: bool = True
    cache_preprocessed: bool = False
    apply_initial_transform_to_images: bool = True
    apply_initial_transform_to_landmarks: bool = True
    invert_initial_transform_for_tre: bool = False
    chamfer_label_value: int = 1
    chamfer_max_surface_points: Optional[int] = 5000
    chamfer_random_seed: int = 0
    chamfer_cache_surface_points: bool = True


def _resolve_path(path: Optional[str], base_dir: str) -> Optional[str]:
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(base_dir, path))


def _parse_image_spec(raw: Dict[str, Any], base_dir: str, role: str) -> ImageSpec:
    if "image" not in raw:
        raise ValueError(f"Case entry '{role}' must define 'image'.")
    if "modality" not in raw:
        raise ValueError(f"Case entry '{role}' must define 'modality'.")
    modality = str(raw["modality"]).lower()
    if modality not in VALID_MODALITIES:
        raise ValueError(
            f"Unsupported modality '{raw['modality']}' for '{role}'. "
            f"Valid values: {list(VALID_MODALITIES)}."
        )
    spec = ImageSpec(
        image=_resolve_path(str(raw["image"]), base_dir),
        modality=modality,
        segmentation=_resolve_path(raw.get("segmentation"), base_dir),
        mask=_resolve_path(raw.get("mask"), base_dir),
        landmarks=_resolve_path(raw.get("landmarks"), base_dir),
        surface_points=_resolve_path(raw.get("surface_points"), base_dir),
    )
    if not os.path.exists(spec.image):
        raise FileNotFoundError(f"Image not found: {spec.image}")
    return spec


def load_method_eval_config(config_path: str) -> MethodEvalConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError("Evaluation config must be a YAML dictionary.")

    base_dir = os.path.dirname(os.path.abspath(config_path))
    if "method_name" not in raw:
        raise ValueError("Evaluation config missing required key 'method_name'.")
    if "test_manifest" not in raw:
        raise ValueError("Evaluation config missing required key 'test_manifest'.")

    model = str(raw.get("model", "multigradicon")).lower()
    if model not in VALID_MODELS:
        raise ValueError(f"'model' must be one of {list(VALID_MODELS)}.")
    io_sim = str(raw.get("io_sim", "lncc")).lower()
    if io_sim not in VALID_SIMILARITIES:
        raise ValueError(f"'io_sim' must be one of {list(VALID_SIMILARITIES)}.")

    legacy_sre_keys = {
        "sre_label_value",
        "sre_max_surface_points",
        "sre_random_seed",
        "sre_cache_surface_points",
    }
    used_legacy_keys = sorted(k for k in legacy_sre_keys if k in raw)
    if used_legacy_keys:
        raise ValueError(
            "SRE configuration keys are no longer supported: "
            f"{used_legacy_keys}. Use chamfer_* equivalents."
        )

    metrics = tuple(str(metric).lower() for metric in raw.get("metrics", list(VALID_METRICS)))
    if "sre" in metrics:
        raise ValueError(
            "Metric 'sre' is no longer supported. Replace it with 'chamfer' in 'metrics'."
        )
    invalid_metrics = [metric for metric in metrics if metric not in VALID_METRICS]
    if invalid_metrics:
        raise ValueError(
            f"Unsupported metrics {invalid_metrics}. Valid metrics: {list(VALID_METRICS)}."
        )

    io_iterations_raw = raw.get("io_iterations", 50)
    io_iterations = None if io_iterations_raw is None else int(io_iterations_raw)
    if io_iterations is not None and io_iterations < 0:
        raise ValueError("'io_iterations' must be null or a non-negative integer.")

    io_lr = float(raw.get("io_lr", 0.00002))
    if io_lr <= 0:
        raise ValueError("'io_lr' must be positive.")

    flush_every = int(raw.get("flush_every", 1))
    if flush_every <= 0:
        raise ValueError("'flush_every' must be positive.")

    ct_window_raw = raw.get("ct_window")
    ct_window = tuple(ct_window_raw) if ct_window_raw is not None else None
    input_shape_raw = raw.get("input_shape")
    input_shape = tuple(int(d) for d in input_shape_raw) if input_shape_raw is not None else None
    if input_shape is not None:
        if len(input_shape) != 3 or any(d <= 0 for d in input_shape):
            raise ValueError("'input_shape' must be null or a length-3 list of positive integers.")
    quantile_range_raw = raw.get("quantile_range")
    quantile_range = tuple(quantile_range_raw) if quantile_range_raw is not None else None
    chamfer_label_value = int(raw.get("chamfer_label_value", 1))
    if chamfer_label_value < 0:
        raise ValueError("'chamfer_label_value' must be a non-negative integer.")
    chamfer_max_surface_points_raw = raw.get("chamfer_max_surface_points", 5000)
    chamfer_max_surface_points = (
        None if chamfer_max_surface_points_raw is None else int(chamfer_max_surface_points_raw)
    )
    if chamfer_max_surface_points is not None and chamfer_max_surface_points <= 0:
        raise ValueError("'chamfer_max_surface_points' must be null or a positive integer.")
    chamfer_random_seed = int(raw.get("chamfer_random_seed", 0))

    network_weights = _resolve_path(raw.get("network_weights"), base_dir)
    if network_weights is not None and not os.path.exists(network_weights):
        raise FileNotFoundError(f"network_weights not found: {network_weights}")

    test_manifest = _resolve_path(str(raw["test_manifest"]), base_dir)
    if not os.path.exists(test_manifest):
        raise FileNotFoundError(f"test_manifest not found: {test_manifest}")

    return MethodEvalConfig(
        method_name=str(raw["method_name"]),
        test_manifest=test_manifest,
        model=model,
        network_weights=network_weights,
        output_root=_resolve_path(str(raw.get("output_root", "results/eval")), base_dir),
        io_iterations=io_iterations,
        io_lr=io_lr,
        io_sim=io_sim,
        dice_loss_weight=float(raw.get("dice_loss_weight", 0.0)),
        loss_function_masking=bool(raw.get("loss_function_masking", False)),
        input_masking=bool(raw.get("input_masking", False)),
        intensity_conservation_loss=bool(raw.get("intensity_conservation_loss", False)),
        input_shape=input_shape,
        ct_window=ct_window,
        quantile_range=quantile_range,
        metrics=metrics,
        resume=bool(raw.get("resume", True)),
        flush_every=flush_every,
        save_transforms=bool(raw.get("save_transforms", True)),
        cache_preprocessed=bool(raw.get("cache_preprocessed", False)),
        apply_initial_transform_to_images=bool(raw.get("apply_initial_transform_to_images", True)),
        apply_initial_transform_to_landmarks=bool(
            raw.get("apply_initial_transform_to_landmarks", True)
        ),
        invert_initial_transform_for_tre=bool(raw.get("invert_initial_transform_for_tre", False)),
        chamfer_label_value=chamfer_label_value,
        chamfer_max_surface_points=chamfer_max_surface_points,
        chamfer_random_seed=chamfer_random_seed,
        chamfer_cache_surface_points=bool(raw.get("chamfer_cache_surface_points", True)),
    )


def _assert_exists(path: Optional[str], field_name: str, case_id: str) -> None:
    if path is not None and not os.path.exists(path):
        raise FileNotFoundError(f"Case '{case_id}': {field_name} not found: {path}")


def load_test_manifest(manifest_path: str) -> List[EvalCase]:
    manifest_path = os.path.abspath(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("Test manifest must be a JSON dictionary.")
    if "data" not in raw or not isinstance(raw["data"], Sequence):
        raise ValueError("Test manifest must contain a top-level 'data' list.")

    base_dir = os.path.dirname(manifest_path)
    cases: List[EvalCase] = []
    seen_ids = set()
    for idx, item in enumerate(raw["data"]):
        if not isinstance(item, dict):
            raise ValueError(f"Manifest entry {idx} must be an object.")
        case_id = str(item.get("case_id", f"case_{idx:04d}"))
        if case_id in seen_ids:
            raise ValueError(f"Duplicate case_id '{case_id}' in test manifest.")
        seen_ids.add(case_id)
        if "fixed" not in item or "moving" not in item:
            raise ValueError(f"Case '{case_id}' must include 'fixed' and 'moving' objects.")
        fixed = _parse_image_spec(item["fixed"], base_dir=base_dir, role=f"{case_id}.fixed")
        moving = _parse_image_spec(item["moving"], base_dir=base_dir, role=f"{case_id}.moving")
        case = EvalCase(
            case_id=case_id,
            fixed=fixed,
            moving=moving,
            initial_transform=_resolve_path(item.get("initial_transform"), base_dir),
            ground_truth_transform=_resolve_path(item.get("ground_truth_transform"), base_dir),
            metrics=CaseMetricsConfig.from_dict(item.get("metrics")),
        )
        _assert_exists(case.initial_transform, "initial_transform", case_id)
        _assert_exists(case.ground_truth_transform, "ground_truth_transform", case_id)
        for field_name, path in (
            ("fixed.segmentation", case.fixed.segmentation),
            ("moving.segmentation", case.moving.segmentation),
            ("fixed.mask", case.fixed.mask),
            ("moving.mask", case.moving.mask),
            ("fixed.landmarks", case.fixed.landmarks),
            ("moving.landmarks", case.moving.landmarks),
            ("fixed.surface_points", case.fixed.surface_points),
            ("moving.surface_points", case.moving.surface_points),
        ):
            _assert_exists(path, field_name, case_id)
        if case.metrics.chamfer:
            if case.fixed.segmentation is None or case.moving.segmentation is None:
                raise ValueError(
                    f"Case '{case_id}': Chamfer is enabled but both "
                    "'fixed.segmentation' and 'moving.segmentation' are required."
                )
        cases.append(case)
    return cases
