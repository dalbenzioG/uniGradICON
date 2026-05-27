#!/usr/bin/env python3
"""Convert TRUSTED-style landmark .txt files to ITK/NIfTI LPS physical coordinates.

Slicer markups are often stored/exported in RAS world coordinates. Evaluation and
``phi_ab.TransformPoint`` expect ITK physical (LPS) coordinates for the exact NIfTI
referenced in the eval manifest.

Usage:
    python scripts/convert_landmarks_to_lps.py probe --manifest path/to/manifest.json
    python scripts/convert_landmarks_to_lps.py convert \\
        --manifest path/to/manifest.json \\
        --convention ras_to_lps \\
        --output-dir path/to/landmarks_lps \\
        --write-manifest path/to/manifest_lps.json
    python scripts/convert_landmarks_to_lps.py validate \\
        --manifest path/to/manifest_lps.json \\
        --convention identity
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import itk
import numpy as np

from unigradicon.evaluation.config import load_test_manifest
from unigradicon.evaluation.landmark_transforms import (
    LANDMARK_POINT_CONVENTIONS,
    apply_landmark_point_convention,
)
from unigradicon.evaluation.metrics import (
    image_physical_bounds,
    load_points,
    mean_point_distance_mm,
    points_inside_physical_bounds,
    save_points,
)

PROBE_CONVENTIONS = ("identity", "ras_to_lps")


@dataclass
class ConventionScore:
    convention: str
    fixed_inside_count: int
    moving_inside_count: int
    fixed_total: int
    moving_total: int
    baseline_raw_mm: Optional[float]

    @property
    def inside_count(self) -> int:
        return self.fixed_inside_count + self.moving_inside_count

    @property
    def total_count(self) -> int:
        return self.fixed_total + self.moving_total


def _landmark_output_path(input_path: str, output_dir: str, suffix: str) -> str:
    stem = os.path.splitext(os.path.basename(input_path))[0]
    if stem.endswith(suffix):
        filename = f"{stem}.txt"
    else:
        filename = f"{stem}{suffix}.txt"
    return os.path.join(os.path.abspath(output_dir), filename)


def _score_convention(
    case,
    convention: str,
    *,
    margin_mm: float,
) -> ConventionScore:
    fixed_total = 0
    moving_total = 0
    fixed_inside_count = 0
    moving_inside_count = 0
    baseline_raw_mm: Optional[float] = None
    fixed_points = None

    if case.fixed.landmarks is not None and case.fixed.image is not None:
        fixed_image = itk.imread(case.fixed.image)
        fixed_bounds_min, fixed_bounds_max = image_physical_bounds(fixed_image)
        fixed_points = apply_landmark_point_convention(
            load_points(case.fixed.landmarks),
            convention,
        )
        fixed_total = int(fixed_points.shape[0])
        fixed_inside_count = int(
            np.sum(
                np.all(fixed_points >= fixed_bounds_min - margin_mm, axis=1)
                & np.all(fixed_points <= fixed_bounds_max + margin_mm, axis=1)
            )
        )

    if case.moving.landmarks is not None and case.moving.image is not None:
        moving_image = itk.imread(case.moving.image)
        moving_bounds_min, moving_bounds_max = image_physical_bounds(moving_image)
        moving_points = apply_landmark_point_convention(
            load_points(case.moving.landmarks),
            convention,
        )
        moving_total = int(moving_points.shape[0])
        moving_inside_count = int(
            np.sum(
                np.all(moving_points >= moving_bounds_min - margin_mm, axis=1)
                & np.all(moving_points <= moving_bounds_max + margin_mm, axis=1)
            )
        )
        if fixed_points is not None and fixed_points.shape == moving_points.shape:
            baseline_raw_mm = mean_point_distance_mm(moving_points, fixed_points)

    return ConventionScore(
        convention=convention,
        fixed_inside_count=fixed_inside_count,
        moving_inside_count=moving_inside_count,
        fixed_total=fixed_total,
        moving_total=moving_total,
        baseline_raw_mm=baseline_raw_mm,
    )


def recommend_convention(scores: List[ConventionScore]) -> str:
    if not scores:
        raise ValueError("No convention scores were computed.")

    def sort_key(score: ConventionScore) -> tuple:
        baseline = score.baseline_raw_mm if score.baseline_raw_mm is not None else float("inf")
        return (-score.inside_count, baseline, score.convention)

    return sorted(scores, key=sort_key)[0].convention


def probe_case(case, *, margin_mm: float) -> List[ConventionScore]:
    return [_score_convention(case, name, margin_mm=margin_mm) for name in PROBE_CONVENTIONS]


def convert_landmark_file(
    input_path: str,
    output_path: str,
    convention: str,
    *,
    dry_run: bool = False,
) -> np.ndarray:
    converted = apply_landmark_point_convention(load_points(input_path), convention)
    if not dry_run:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        save_points(output_path, converted)
    return converted


def write_converted_manifest(
    source_manifest: str,
    output_manifest: str,
    converted_paths_by_case: Dict[str, Dict[str, str]],
) -> None:
    with open(source_manifest, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or "data" not in payload:
        raise ValueError(f"Manifest must contain a top-level 'data' list: {source_manifest}")

    out_dir = os.path.dirname(os.path.abspath(output_manifest))
    os.makedirs(out_dir, exist_ok=True)

    for entry in payload["data"]:
        case_id = entry.get("case_id")
        if case_id not in converted_paths_by_case:
            continue
        paths = converted_paths_by_case[case_id]
        if "fixed" in paths and entry.get("fixed") is not None:
            entry["fixed"]["landmarks"] = os.path.relpath(paths["fixed"], out_dir)
        if "moving" in paths and entry.get("moving") is not None:
            entry["moving"]["landmarks"] = os.path.relpath(paths["moving"], out_dir)

    with open(output_manifest, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _print_probe_case(case, scores: List[ConventionScore], recommended: str) -> None:
    print(f"[{case.case_id}]")
    for score in scores:
        baseline = (
            f"{score.baseline_raw_mm:.4f} mm"
            if score.baseline_raw_mm is not None
            else "n/a"
        )
        marker = " <-- recommended" if score.convention == recommended else ""
        print(
            f"  {score.convention:10s}  inside={score.inside_count}/{score.total_count}  "
            f"(fixed {score.fixed_inside_count}/{score.fixed_total}, "
            f"moving {score.moving_inside_count}/{score.moving_total})  "
            f"baseline_raw={baseline}{marker}"
        )


def _cmd_probe(args: argparse.Namespace) -> int:
    cases = load_test_manifest(args.manifest)
    if args.case_id is not None:
        cases = [case for case in cases if case.case_id == args.case_id]
        if not cases:
            print(f"No case with id '{args.case_id}' found.", file=sys.stderr)
            return 1

    recommended_global: Dict[str, int] = {name: 0 for name in PROBE_CONVENTIONS}
    probed = 0
    for case in cases:
        if case.fixed.landmarks is None:
            continue
        probed += 1
        scores = probe_case(case, margin_mm=args.margin_mm)
        recommended = recommend_convention(scores)
        recommended_global[recommended] += 1
        _print_probe_case(case, scores, recommended)
        print("")

    if probed == 0:
        print("No cases with fixed landmarks found.", file=sys.stderr)
        return 1

    winner = max(recommended_global.items(), key=lambda item: item[1])[0]
    print(f"Recommended convention across cases: {winner}")
    print("Use with: python scripts/convert_landmarks_to_lps.py convert --convention", winner)
    return 0


def _resolve_convention(args: argparse.Namespace, cases) -> str:
    if args.convention != "auto":
        return args.convention

    per_case_winners: List[str] = []
    for case in cases:
        if case.fixed.landmarks is None:
            continue
        scores = probe_case(case, margin_mm=args.margin_mm)
        per_case_winners.append(recommend_convention(scores))
    if not per_case_winners:
        raise ValueError("Could not auto-detect convention: no fixed landmarks in manifest.")
    return max(set(per_case_winners), key=per_case_winners.count)


def _cmd_convert(args: argparse.Namespace) -> int:
    cases = load_test_manifest(args.manifest)
    convention = _resolve_convention(args, cases)
    print(f"Using convention: {convention}")

    roles = {part.strip().lower() for part in args.roles.split(",") if part.strip()}
    unknown_roles = roles - {"fixed", "moving"}
    if unknown_roles:
        raise ValueError(f"Unknown --roles entries: {sorted(unknown_roles)}")
    if not roles:
        raise ValueError("--roles must include fixed and/or moving.")

    converted_by_case: Dict[str, Dict[str, str]] = {}
    for case in cases:
        converted_by_case[case.case_id] = {}
        if "fixed" in roles and case.fixed.landmarks is not None:
            out_path = _landmark_output_path(case.fixed.landmarks, args.output_dir, args.suffix)
            convert_landmark_file(
                case.fixed.landmarks,
                out_path,
                convention,
                dry_run=args.dry_run,
            )
            converted_by_case[case.case_id]["fixed"] = out_path
            print(f"{case.case_id}: fixed -> {out_path}")
        if "moving" in roles and case.moving.landmarks is not None:
            out_path = _landmark_output_path(case.moving.landmarks, args.output_dir, args.suffix)
            convert_landmark_file(
                case.moving.landmarks,
                out_path,
                convention,
                dry_run=args.dry_run,
            )
            converted_by_case[case.case_id]["moving"] = out_path
            print(f"{case.case_id}: moving -> {out_path}")

    if args.write_manifest is not None:
        if args.dry_run:
            print(f"Dry run: would write manifest -> {args.write_manifest}")
        else:
            write_converted_manifest(args.manifest, args.write_manifest, converted_by_case)
            print(f"Wrote manifest: {args.write_manifest}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    cases = load_test_manifest(args.manifest)
    convention = args.convention
    if convention == "auto":
        convention = _resolve_convention(args, cases)
        print(f"Using convention: {convention}")

    failures = 0
    checked = 0
    for case in cases:
        if case.fixed.landmarks is None:
            continue
        checked += 1
        fixed_image = itk.imread(case.fixed.image)
        fixed_bounds_min, fixed_bounds_max = image_physical_bounds(fixed_image)
        fixed_points = apply_landmark_point_convention(
            load_points(case.fixed.landmarks),
            convention,
        )
        fixed_inside = points_inside_physical_bounds(
            fixed_points,
            fixed_bounds_min,
            fixed_bounds_max,
            margin_mm=args.margin_mm,
        )

        moving_inside = True
        baseline_raw_mm: Optional[float] = None
        if case.moving.landmarks is not None and case.moving.image is not None:
            moving_image = itk.imread(case.moving.image)
            moving_bounds_min, moving_bounds_max = image_physical_bounds(moving_image)
            moving_points = apply_landmark_point_convention(
                load_points(case.moving.landmarks),
                convention,
            )
            moving_inside = points_inside_physical_bounds(
                moving_points,
                moving_bounds_min,
                moving_bounds_max,
                margin_mm=args.margin_mm,
            )
            if fixed_points.shape == moving_points.shape:
                baseline_raw_mm = mean_point_distance_mm(moving_points, fixed_points)

        passed = fixed_inside and moving_inside
        if not passed:
            failures += 1
        status = "PASS" if passed else "FAIL"
        baseline = (
            f"{baseline_raw_mm:.4f} mm" if baseline_raw_mm is not None else "n/a"
        )
        print(
            f"[{status}] {case.case_id}  fixed_inside={fixed_inside}  "
            f"moving_inside={moving_inside}  baseline_raw={baseline}"
        )

    if checked == 0:
        print("No cases with fixed landmarks found.", file=sys.stderr)
        return 1
    if failures:
        print(f"{failures}/{checked} case(s) failed validation.", file=sys.stderr)
        return 1
    print(f"All {checked} case(s) passed.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert landmark .txt files to ITK/NIfTI LPS physical coordinates."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--manifest", required=True, help="Path to eval test manifest JSON.")
    common.add_argument(
        "--margin-mm",
        type=float,
        default=1.0,
        help="Margin when checking landmark/image bounds (default: 1).",
    )
    common.add_argument(
        "--convention",
        choices=("auto", *sorted(LANDMARK_POINT_CONVENTIONS)),
        default="auto",
        help="Input landmark convention (default: auto). Output is always ITK LPS.",
    )

    probe_parser = subparsers.add_parser(
        "probe",
        parents=[common],
        help="Try identity vs ras_to_lps and recommend a convention per case.",
    )
    probe_parser.add_argument("--case-id", default=None, help="Optional single case id.")

    convert_parser = subparsers.add_parser(
        "convert",
        parents=[common],
        help="Convert landmark files referenced by the manifest.",
    )
    convert_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for converted .txt files.",
    )
    convert_parser.add_argument(
        "--suffix",
        default="_lps",
        help="Suffix before .txt for output files (default: _lps).",
    )
    convert_parser.add_argument(
        "--roles",
        default="fixed,moving",
        help="Comma-separated landmark roles to convert: fixed,moving (default: both).",
    )
    convert_parser.add_argument(
        "--write-manifest",
        default=None,
        help="Optional output manifest with updated landmark paths.",
    )
    convert_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned outputs without writing files.",
    )

    subparsers.add_parser(
        "validate",
        parents=[common],
        help="Check converted/current landmarks fall inside manifest image bounds.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "probe":
        return _cmd_probe(args)
    if args.command == "convert":
        return _cmd_convert(args)
    if args.command == "validate":
        return _cmd_validate(args)
    parser.error(f"Unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
