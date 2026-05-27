#!/usr/bin/env python3
"""Validate and align CT landmarks for evaluation TRE.

TRUSTED workflow (default ``--landmark-space inimg``):
1. Use CT landmarks in **native CT NIfTI physical space** (official release: ``CT_landmarks_inimg/``; some local copies use ``CT_landmarks/`` for the same coordinates).
2. Run ``probe`` to pick ``report_json`` vs ``report_json_inverted`` direction.
3. Run ``unigradicon-eval`` with ``apply_initial_transform_to_landmarks: true``.

Usage:
    python scripts/align_eval_landmarks.py validate --manifest path/to/manifest.json
    python scripts/align_eval_landmarks.py probe --manifest path/to/manifest.json
    python scripts/align_eval_landmarks.py probe --manifest path/to/manifest.json --case-id 348L
    python scripts/align_eval_landmarks.py align \\
        --manifest path/to/manifest.json \\
        --landmark-space inimg \\
        --output-dir path/to/CT_landmarks_inimg_prealigned \\
        --write-manifest path/to/eval_test_manifest_prealigned.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import itk
import numpy as np

from unigradicon.evaluation.config import EvalCase, load_test_manifest
from unigradicon.evaluation.landmark_transforms import (
    probe_landmark_transforms,
    recommend_invert_from_probe,
    transform_moving_landmarks_initial,
)
from unigradicon.evaluation.metrics import (
    image_physical_bounds,
    load_points,
    mean_point_distance_mm,
    points_inside_physical_bounds,
    save_points,
)


@dataclass
class CaseValidationResult:
    case_id: str
    fixed_landmarks_path: str
    moving_landmarks_path: str
    fixed_landmarks_inside_image: bool
    moving_landmarks_inside_image: bool
    baseline_raw_mm: float
    fixed_image_bounds_min: List[float]
    fixed_image_bounds_max: List[float]
    moving_image_bounds_min: List[float]
    moving_image_bounds_max: List[float]
    fixed_landmarks_min: List[float]
    fixed_landmarks_max: List[float]
    moving_landmarks_min: List[float]
    moving_landmarks_max: List[float]
    passed: bool
    messages: List[str]


def resolve_moving_landmarks_path(
    case: EvalCase,
    aligned_dir: Optional[str] = None,
    suffix: str = "_aligned",
) -> Optional[str]:
    if case.moving.landmarks is None:
        return None
    if aligned_dir is None:
        return case.moving.landmarks
    stem = os.path.splitext(os.path.basename(case.moving.landmarks))[0]
    if stem.endswith(suffix):
        filename = f"{stem}.txt"
    else:
        filename = f"{stem}{suffix}.txt"
    return os.path.join(os.path.abspath(aligned_dir), filename)


def aligned_output_path(
    native_landmarks_path: str,
    output_dir: str,
    suffix: str = "_aligned",
) -> str:
    stem = os.path.splitext(os.path.basename(native_landmarks_path))[0]
    return os.path.join(os.path.abspath(output_dir), f"{stem}{suffix}.txt")


LANDMARK_SPACES = ("inimg", "annotation")


def _load_moving_points_for_validation(
    case: EvalCase,
    moving_landmarks_path: str,
    *,
    landmark_space: str,
    aligned_dir: Optional[str],
    invert_initial: bool,
) -> tuple[np.ndarray, bool]:
    if case.moving.landmarks is None:
        return np.empty((0, 3), dtype=np.float64), False

    if aligned_dir is not None:
        if os.path.exists(moving_landmarks_path):
            return load_points(moving_landmarks_path), True
        return np.empty((0, 3), dtype=np.float64), False

    if not os.path.exists(case.moving.landmarks):
        return np.empty((0, 3), dtype=np.float64), False

    native_points = load_points(case.moving.landmarks)
    if (
        landmark_space == "inimg"
        and case.initial_transform is not None
    ):
        transformed, _ = transform_moving_landmarks_initial(
            native_points,
            case.initial_transform,
            prefer_report_json=True,
            invert=invert_initial,
            invert_itk=invert_initial,
        )
        return transformed, True
    return native_points, True


def validate_case(
    case: EvalCase,
    *,
    aligned_dir: Optional[str] = None,
    suffix: str = "_aligned",
    threshold_mm: float = 50.0,
    margin_mm: float = 1.0,
    strict_bounds: bool = False,
    landmark_space: str = "inimg",
    invert_initial: bool = False,
) -> CaseValidationResult:
    messages: List[str] = []
    if case.fixed.landmarks is None or case.moving.landmarks is None:
        raise ValueError(f"Case '{case.case_id}' requires both fixed and moving landmarks.")

    moving_landmarks_path = resolve_moving_landmarks_path(case, aligned_dir=aligned_dir, suffix=suffix)
    assert moving_landmarks_path is not None

    fixed_image = itk.imread(case.fixed.image)
    moving_image = itk.imread(case.moving.image)
    fixed_bounds_min, fixed_bounds_max = image_physical_bounds(fixed_image)
    moving_bounds_min, moving_bounds_max = image_physical_bounds(moving_image)

    fixed_points = load_points(case.fixed.landmarks)
    moving_points, moving_found = _load_moving_points_for_validation(
        case,
        moving_landmarks_path,
        landmark_space=landmark_space,
        aligned_dir=aligned_dir,
        invert_initial=invert_initial,
    )
    if not moving_found:
        messages.append(f"Moving landmarks not found: {moving_landmarks_path}")
        moving_inside = False
        baseline_raw = float("inf")
    else:
        moving_inside = points_inside_physical_bounds(
            moving_points, moving_bounds_min, moving_bounds_max, margin_mm=margin_mm
        )
        baseline_raw = mean_point_distance_mm(moving_points, fixed_points)

    fixed_inside = points_inside_physical_bounds(
        fixed_points, fixed_bounds_min, fixed_bounds_max, margin_mm=margin_mm
    )

    if not fixed_inside:
        msg = "Fixed landmarks fall outside fixed image physical bounds."
        if strict_bounds:
            messages.append(msg)
        else:
            messages.append(f"WARNING: {msg}")
    if not moving_inside:
        msg = "Moving landmarks fall outside moving image physical bounds."
        if strict_bounds:
            messages.append(msg)
        else:
            messages.append(f"WARNING: {msg}")
    if baseline_raw > threshold_mm:
        messages.append(
            f"baseline_raw={baseline_raw:.3f} mm exceeds threshold {threshold_mm:.3f} mm."
        )

    passed = baseline_raw <= threshold_mm
    if strict_bounds:
        passed = passed and fixed_inside and moving_inside
    return CaseValidationResult(
        case_id=case.case_id,
        fixed_landmarks_path=case.fixed.landmarks,
        moving_landmarks_path=moving_landmarks_path,
        fixed_landmarks_inside_image=fixed_inside,
        moving_landmarks_inside_image=moving_inside,
        baseline_raw_mm=baseline_raw,
        fixed_image_bounds_min=fixed_bounds_min.tolist(),
        fixed_image_bounds_max=fixed_bounds_max.tolist(),
        moving_image_bounds_min=moving_bounds_min.tolist(),
        moving_image_bounds_max=moving_bounds_max.tolist(),
        fixed_landmarks_min=np.min(fixed_points, axis=0).tolist(),
        fixed_landmarks_max=np.max(fixed_points, axis=0).tolist(),
        moving_landmarks_min=(
            np.min(moving_points, axis=0).tolist() if moving_points.size else []
        ),
        moving_landmarks_max=(
            np.max(moving_points, axis=0).tolist() if moving_points.size else []
        ),
        passed=passed,
        messages=messages,
    )


def validate_manifest(
    manifest_path: str,
    *,
    aligned_dir: Optional[str] = None,
    suffix: str = "_aligned",
    threshold_mm: float = 50.0,
    margin_mm: float = 1.0,
    strict_bounds: bool = False,
    landmark_space: str = "inimg",
    invert_initial: bool = False,
) -> List[CaseValidationResult]:
    cases = load_test_manifest(manifest_path)
    results = []
    for case in cases:
        if case.fixed.landmarks is None or case.moving.landmarks is None:
            continue
        results.append(
            validate_case(
                case,
                aligned_dir=aligned_dir,
                suffix=suffix,
                threshold_mm=threshold_mm,
                margin_mm=margin_mm,
                strict_bounds=strict_bounds,
                landmark_space=landmark_space,
                invert_initial=invert_initial,
            )
        )
    return results


def align_case(
    case: EvalCase,
    *,
    output_dir: str,
    suffix: str = "_aligned",
    prefer_report_json: bool = True,
    invert_transform: bool = False,
    dry_run: bool = False,
    threshold_mm: float = 50.0,
    margin_mm: float = 1.0,
    strict_bounds: bool = False,
    landmark_space: str = "inimg",
) -> tuple[str, str, CaseValidationResult]:
    if case.moving.landmarks is None:
        raise ValueError(f"Case '{case.case_id}' is missing moving.landmarks.")
    if case.initial_transform is None:
        raise ValueError(
            f"Case '{case.case_id}' is missing initial_transform required for landmark alignment."
        )

    native_points = load_points(case.moving.landmarks)
    aligned_points, transform_source = transform_moving_landmarks_initial(
        native_points,
        case.initial_transform,
        prefer_report_json=prefer_report_json,
        invert=invert_transform,
        invert_itk=invert_transform,
    )
    output_path = aligned_output_path(case.moving.landmarks, output_dir, suffix=suffix)

    if not dry_run:
        os.makedirs(output_dir, exist_ok=True)
        save_points(output_path, aligned_points)

    validation = validate_case(
        case,
        aligned_dir=output_dir,
        suffix=suffix,
        threshold_mm=threshold_mm,
        margin_mm=margin_mm,
        strict_bounds=strict_bounds,
        landmark_space=landmark_space,
        invert_initial=invert_transform,
    )
    return output_path, transform_source, validation


def write_aligned_manifest(
    source_manifest_path: str,
    output_manifest_path: str,
    aligned_landmarks_by_case: Dict[str, str],
) -> None:
    with open(source_manifest_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict) or "data" not in raw:
        raise ValueError("Manifest must contain a top-level 'data' list.")

    out_dir = os.path.dirname(os.path.abspath(output_manifest_path))
    os.makedirs(out_dir, exist_ok=True)
    for item in raw["data"]:
        case_id = str(item.get("case_id"))
        if case_id not in aligned_landmarks_by_case:
            continue
        aligned_path = os.path.abspath(aligned_landmarks_by_case[case_id])
        rel_path = os.path.relpath(aligned_path, out_dir)
        item.setdefault("moving", {})["landmarks"] = rel_path.replace("\\", "/")

    with open(output_manifest_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2)
        f.write("\n")


def _print_validation_results(results: List[CaseValidationResult], as_json: bool) -> None:
    if as_json:
        print(json.dumps([asdict(result) for result in results], indent=2))
        return

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.case_id}")
        print(f"  fixed landmarks:  {result.fixed_landmarks_path}")
        print(f"  moving landmarks: {result.moving_landmarks_path}")
        print(f"  fixed inside image:  {result.fixed_landmarks_inside_image}")
        print(f"  moving inside image: {result.moving_landmarks_inside_image}")
        print(f"  baseline_raw_mm: {result.baseline_raw_mm:.4f}")
        for message in result.messages:
            print(f"  - {message}")
        print("")


def _cmd_validate(args: argparse.Namespace) -> int:
    results = validate_manifest(
        args.manifest,
        aligned_dir=args.aligned_dir,
        suffix=args.suffix,
        threshold_mm=args.threshold_mm,
        margin_mm=args.margin_mm,
        strict_bounds=args.strict_bounds,
        landmark_space=args.landmark_space,
        invert_initial=args.invert_transform,
    )
    if not results:
        print("No cases with both fixed and moving landmarks were found.", file=sys.stderr)
        return 1

    _print_validation_results(results, args.json)
    failed = [result for result in results if not result.passed]
    if failed:
        print(
            f"{len(failed)}/{len(results)} case(s) failed validation. "
            "Run `align` to create aligned CT landmark files, or check transform provenance.",
            file=sys.stderr,
        )
        return 1
    print(f"All {len(results)} case(s) passed validation.")
    return 0


def _cmd_align(args: argparse.Namespace) -> int:
    cases = load_test_manifest(args.manifest)
    aligned_by_case: Dict[str, str] = {}
    validations: List[CaseValidationResult] = []
    failures = 0

    for case in cases:
        if case.fixed.landmarks is None or case.moving.landmarks is None:
            continue
        output_path, transform_source, validation = align_case(
            case,
            output_dir=args.output_dir,
            suffix=args.suffix,
            prefer_report_json=not args.itk_transform,
            invert_transform=args.invert_transform,
            dry_run=args.dry_run,
            threshold_mm=args.threshold_mm,
            margin_mm=args.margin_mm,
            strict_bounds=args.strict_bounds,
            landmark_space=args.landmark_space,
        )
        aligned_by_case[case.case_id] = output_path
        validations.append(validation)
        status = "PASS" if validation.passed else "FAIL"
        print(f"[{status}] {case.case_id} -> {output_path} (transform={transform_source})")
        for message in validation.messages:
            print(f"  - {message}")
        if not validation.passed:
            failures += 1

    if not aligned_by_case:
        print("No cases with both fixed and moving landmarks were found.", file=sys.stderr)
        return 1

    if args.write_manifest and not args.dry_run:
        write_aligned_manifest(args.manifest, args.write_manifest, aligned_by_case)
        print(f"Wrote aligned manifest: {args.write_manifest}")

    if failures:
        print(
            f"{failures}/{len(validations)} aligned case(s) failed post-align validation.",
            file=sys.stderr,
        )
        return 1

    print(f"Aligned {len(validations)} case(s) successfully.")
    return 0


def _probe_case(case: EvalCase) -> Dict[str, float]:
    if case.fixed.landmarks is None or case.moving.landmarks is None:
        raise ValueError(f"Case '{case.case_id}' is missing landmark paths.")
    if case.initial_transform is None:
        raise ValueError(f"Case '{case.case_id}' is missing initial_transform.")

    native_points = load_points(case.moving.landmarks)
    fixed_points = load_points(case.fixed.landmarks)
    return probe_landmark_transforms(native_points, fixed_points, case.initial_transform)


def _print_probe_results(case: EvalCase, results: Dict[str, float]) -> None:
    print(f"Case: {case.case_id}")
    print(f"Transform: {case.initial_transform}")
    for name, value in sorted(results.items()):
        print(f"  {name:24s} baseline_raw_mm = {value:.4f}")
    if "report_json" in results and "report_json_inverted" in results:
        invert = recommend_invert_from_probe(results)
        winner = "report_json_inverted" if invert else "report_json"
        print(f"  Recommended eval YAML: invert_initial_transform_for_tre: {str(invert).lower()} ({winner})")


def _cmd_probe(args: argparse.Namespace) -> int:
    cases = load_test_manifest(args.manifest)
    if args.case_id is not None:
        selected = [case for case in cases if case.case_id == args.case_id]
        if not selected:
            print(f"Case '{args.case_id}' not found in manifest.", file=sys.stderr)
            return 1
    else:
        selected = [
            case
            for case in cases
            if case.fixed.landmarks is not None
            and case.moving.landmarks is not None
            and case.initial_transform is not None
        ]
        if not selected:
            print(
                "No cases with fixed/moving landmarks and initial_transform were found.",
                file=sys.stderr,
            )
            return 1

    failures = 0
    for index, case in enumerate(selected):
        if index > 0:
            print("")
        try:
            results = _probe_case(case)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            failures += 1
            continue
        _print_probe_results(case, results)

    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and align CT landmarks for unigradicon evaluation TRE."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--manifest", required=True, help="Path to eval test manifest JSON.")
    common.add_argument(
        "--suffix",
        default="_aligned",
        help="Suffix inserted before .txt for aligned landmark filenames (default: _aligned).",
    )
    common.add_argument(
        "--threshold-mm",
        type=float,
        default=50.0,
        help="Maximum allowed baseline_raw mean distance in mm (default: 50).",
    )
    common.add_argument(
        "--margin-mm",
        type=float,
        default=1.0,
        help="Extra physical margin when checking landmark/image bounds (default: 1).",
    )
    common.add_argument(
        "--strict-bounds",
        action="store_true",
        help="Fail validation when landmarks fall outside image physical bounds.",
    )
    common.add_argument(
        "--landmark-space",
        choices=LANDMARK_SPACES,
        default="inimg",
        help="Landmark coordinate space: inimg (CT NIfTI physical, default) or annotation.",
    )
    common.add_argument(
        "--invert-transform",
        action="store_true",
        help="Use inverse report_json / ITK transform when mapping CT landmarks.",
    )

    validate_parser = subparsers.add_parser(
        "validate",
        parents=[common],
        help="Check landmark/image consistency and baseline_raw distances.",
    )
    validate_parser.add_argument(
        "--aligned-dir",
        default=None,
        help="Directory containing aligned moving landmark files to validate.",
    )
    validate_parser.add_argument(
        "--json",
        action="store_true",
        help="Print validation results as JSON.",
    )

    align_parser = subparsers.add_parser(
        "align",
        parents=[common],
        help="Apply initial_transform to native CT landmarks and write aligned files.",
    )
    align_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where aligned moving landmark .txt files will be written.",
    )
    align_parser.add_argument(
        "--write-manifest",
        default=None,
        help="Optional output manifest with moving.landmarks paths updated.",
    )
    align_parser.add_argument(
        "--itk-transform",
        action="store_true",
        help="Use ITK .tfm TransformPoint instead of PCA+ICP report JSON matrix.",
    )
    align_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute aligned paths and validation without writing files.",
    )

    probe_parser = subparsers.add_parser(
        "probe",
        parents=[common],
        help="Compare baseline_raw for native/ITK/JSON transform variants.",
    )
    probe_parser.add_argument(
        "--case-id",
        default=None,
        help="Manifest case_id to probe. If omitted, all landmark cases are probed.",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "validate":
        return _cmd_validate(args)
    if args.command == "align":
        return _cmd_align(args)
    if args.command == "probe":
        return _cmd_probe(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
