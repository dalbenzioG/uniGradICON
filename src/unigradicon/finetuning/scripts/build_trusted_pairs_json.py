#!/usr/bin/env python3
"""
Build a paired CT-US dataset JSON for uniGradICON finetuning (TRUSTED-style layout).

Scans CT and US directories, matches by (patient_id, side), and writes a JSON file
with two entries per pair (one CT, one US). Optionally adds segmentation paths when
files exist (e.g. *L_segCT.nii.gz, *L_segUS.nii.gz).

Usage:
    python -m unigradicon.finetuning.scripts.build_trusted_pairs_json \\
        --data_path /path/to/trusted_code_nsd \\
        --output trusted_pairs.json \\
        [--ct_dir ...] [--us_dir ...] [--seg_suffix_ct _segCT] [--seg_suffix_us _segUS]
"""

import argparse
import glob
import json
import os
from typing import Optional


def extract_patient_side(filename: str) -> tuple:
    """
    Extract (patient_id, side) from filename.
    e.g. '200L_imgCT.nii.gz' -> ('200', 'L'), '200R_imgUS_aligned.nii.gz' -> ('200', 'R')
    """
    base = os.path.basename(filename)
    base = base.replace(".nii.gz", "").replace(".nii", "")
    base = base.replace("_aligned", "")
    if base.endswith("L_imgCT") or base.endswith("L_imgUS"):
        patient_id = base.replace("L_imgCT", "").replace("L_imgUS", "").strip("_")
        return patient_id, "L"
    if base.endswith("R_imgCT") or base.endswith("R_imgUS"):
        patient_id = base.replace("R_imgCT", "").replace("R_imgUS", "").strip("_")
        return patient_id, "R"
    return None, None


def find_segmentation_path(image_path: str, seg_suffix: str, seg_ext: str = ".nii.gz"):
    """
    Given an image path like /dir/200L_imgCT.nii.gz and seg_suffix _segCT,
    return /dir/200L_segCT.nii.gz if it exists, else None.
    """
    base = image_path.replace(".nii.gz", "").replace(".nii", "")
    # Replace _imgCT / _imgUS with the seg suffix
    for pattern in ["_imgUS_aligned", "_imgCT", "_imgUS"]:
        if pattern in base:
            base = base.replace(pattern, seg_suffix)
            break
    seg_path = base + seg_ext
    return seg_path if os.path.exists(seg_path) else None


def find_segmentation_in_dir(
    image_path: str,
    seg_dir: str,
    basename_suffix: str,
    seg_ext: str = ".nii.gz",
) -> Optional[str]:
    """
    When segmentations live in a separate directory (e.g. CT_masks, US_masks),
    look for {patient_id}{side}{basename_suffix}.nii.gz in seg_dir.
    e.g. image_path=.../200R_imgCT.nii.gz, seg_dir=.../CT_masks, basename_suffix=_seg
    -> .../CT_masks/200R_seg.nii.gz
    """
    pid, side = extract_patient_side(image_path)
    if pid is None or side is None:
        return None
    seg_basename = f"{pid}{side}{basename_suffix}{seg_ext}"
    seg_path = os.path.join(seg_dir, seg_basename)
    return seg_path if os.path.exists(seg_path) else None


def main():
    parser = argparse.ArgumentParser(
        description="Build paired CT-US dataset JSON for TRUSTED-style directories."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Base data path (e.g. root of trusted_code_nsd).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON path (e.g. trusted_pairs.json).",
    )
    parser.add_argument(
        "--ct_dir",
        type=str,
        default=None,
        help="CT directory. Default: {data_path}/CT_DATA/CT_images_cropped",
    )
    parser.add_argument(
        "--us_dir",
        type=str,
        default=None,
        help="US directory. Default: {data_path}/US_DATA/US_images",
    )
    parser.add_argument(
        "--seg_suffix_ct",
        type=str,
        default="_segCT",
        help="Suffix for CT segmentation filenames (e.g. 200L_imgCT.nii.gz -> 200L_segCT.nii.gz).",
    )
    parser.add_argument(
        "--seg_suffix_us",
        type=str,
        default="_segUS",
        help="Suffix for US segmentation filenames (used only when seg is next to image).",
    )
    parser.add_argument(
        "--ct_seg_dir",
        type=str,
        default=None,
        help="Directory for CT segmentations (e.g. CT_masks). If set, seg filename is {id}{side}{ct_seg_basename_suffix}.nii.gz. Default: {data_path}/CT_masks if it exists.",
    )
    parser.add_argument(
        "--us_seg_dir",
        type=str,
        default=None,
        help="Directory for US segmentations (e.g. US_masks). Default: {data_path}/US_masks if it exists.",
    )
    parser.add_argument(
        "--ct_seg_basename_suffix",
        type=str,
        default="_seg",
        help="Basename suffix for CT seg when using --ct_seg_dir (e.g. 200R -> 200R_seg.nii.gz). Default: _seg.",
    )
    parser.add_argument(
        "--us_seg_basename_suffix",
        type=str,
        default="_maskUS",
        help="Basename suffix for US seg when using --us_seg_dir (e.g. 200R -> 200R_maskUS.nii.gz). Default: _maskUS.",
    )
    parser.add_argument(
        "--prefer_aligned_us",
        action="store_true",
        default=True,
        help="Prefer *_imgUS_aligned.nii.gz when present (default: True).",
    )
    parser.add_argument(
        "--no_prefer_aligned_us",
        action="store_false",
        dest="prefer_aligned_us",
        help="Do not prefer aligned US images.",
    )
    parser.add_argument(
        "--require_segmentation",
        action="store_true",
        help="Fail if any pair is missing CT or US segmentation files (for paired_ct_us_seg).",
    )
    args = parser.parse_args()

    data_path = os.path.abspath(args.data_path)
    if args.ct_dir is not None:
        ct_dir = os.path.abspath(args.ct_dir)
    else:
        default_ct = os.path.join(data_path, "CT_DATA", "CT_images_cropped")
        flat_ct = os.path.join(data_path, "CT_images")
        ct_dir = default_ct if os.path.isdir(default_ct) else flat_ct
    if args.us_dir is not None:
        us_dir = os.path.abspath(args.us_dir)
    else:
        default_us = os.path.join(data_path, "US_DATA", "US_images")
        flat_us = os.path.join(data_path, "US_images")
        us_dir = default_us if os.path.isdir(default_us) else flat_us

    if not os.path.isdir(ct_dir):
        raise FileNotFoundError(
            f"CT directory not found: {ct_dir}\n"
            f"Expected either {os.path.join(data_path, 'CT_DATA', 'CT_images_cropped')} "
            f"or {os.path.join(data_path, 'CT_images')}. Use --ct_dir to specify."
        )
    if not os.path.isdir(us_dir):
        raise FileNotFoundError(
            f"US directory not found: {us_dir}\n"
            f"Expected either {os.path.join(data_path, 'US_DATA', 'US_images')} "
            f"or {os.path.join(data_path, 'US_images')}. Use --us_dir to specify."
        )

    # Optional segmentation directories (e.g. CT_masks, US_masks)
    ct_seg_dir = args.ct_seg_dir
    if ct_seg_dir is not None:
        ct_seg_dir = os.path.abspath(ct_seg_dir)
        if not os.path.isdir(ct_seg_dir):
            raise FileNotFoundError(f"CT segmentation directory not found: {ct_seg_dir}")
    else:
        default_ct_masks = os.path.join(data_path, "CT_masks")
        if os.path.isdir(default_ct_masks):
            ct_seg_dir = default_ct_masks
    us_seg_dir = args.us_seg_dir
    if us_seg_dir is not None:
        us_seg_dir = os.path.abspath(us_seg_dir)
        if not os.path.isdir(us_seg_dir):
            raise FileNotFoundError(f"US segmentation directory not found: {us_seg_dir}")
    else:
        default_us_masks = os.path.join(data_path, "US_masks")
        if os.path.isdir(default_us_masks):
            us_seg_dir = default_us_masks

    # Discover CT files
    ct_files = (
        sorted(glob.glob(os.path.join(ct_dir, "*L_imgCT.nii.gz")))
        + sorted(glob.glob(os.path.join(ct_dir, "*R_imgCT.nii.gz")))
    )
    # Discover US files (aligned and non-aligned)
    us_files = (
        sorted(glob.glob(os.path.join(us_dir, "*L_imgUS*.nii.gz")))
        + sorted(glob.glob(os.path.join(us_dir, "*R_imgUS*.nii.gz")))
    )

    ct_dict = {}
    for p in ct_files:
        pid, side = extract_patient_side(p)
        if pid is not None and side is not None:
            ct_dict[f"{pid}_{side}"] = p

    us_dict = {}
    us_aligned = {}
    for p in us_files:
        pid, side = extract_patient_side(p)
        if pid is not None and side is not None:
            key = f"{pid}_{side}"
            if "_imgUS_aligned" in p or (p.endswith("_aligned.nii.gz") and "_imgUS" in p):
                us_aligned[key] = p
            else:
                us_dict[key] = p

    # Prefer aligned US when available
    for key in us_aligned:
        us_dict[key] = us_aligned[key]

    # Keep only keys that have both CT and US
    pair_keys = sorted(set(ct_dict.keys()) & set(us_dict.keys()))
    unmatched_ct = len(ct_dict) - len(pair_keys)
    unmatched_us = len(us_dict) - len(pair_keys)

    # Build output dir for relative paths (relative to output JSON's directory)
    output_path = os.path.abspath(args.output)
    output_dir = os.path.dirname(output_path)

    data = []
    for key in pair_keys:
        ct_path = ct_dict[key]
        us_path = us_dict[key]
        # Use relative paths from output JSON directory if possible
        try:
            ct_rel = os.path.relpath(ct_path, output_dir)
            us_rel = os.path.relpath(us_path, output_dir)
        except ValueError:
            ct_rel = ct_path
            us_rel = us_path

        ct_entry = {
            "image": ct_rel,
            "subject_id": key,
            "modality": "ct",
        }
        us_entry = {
            "image": us_rel,
            "subject_id": key,
            "modality": "us",
        }

        if ct_seg_dir is not None:
            seg_ct = find_segmentation_in_dir(ct_path, ct_seg_dir, args.ct_seg_basename_suffix)
        else:
            seg_ct = find_segmentation_path(ct_path, args.seg_suffix_ct)
        if us_seg_dir is not None:
            seg_us = find_segmentation_in_dir(us_path, us_seg_dir, args.us_seg_basename_suffix)
        else:
            seg_us = find_segmentation_path(us_path, args.seg_suffix_us)
        if seg_ct is not None:
            try:
                ct_entry["segmentation"] = os.path.relpath(seg_ct, output_dir)
            except ValueError:
                ct_entry["segmentation"] = seg_ct
        if seg_us is not None:
            try:
                us_entry["segmentation"] = os.path.relpath(seg_us, output_dir)
            except ValueError:
                us_entry["segmentation"] = seg_us

        if args.require_segmentation:
            if seg_ct is None:
                if ct_seg_dir is not None:
                    pid, side = extract_patient_side(ct_path)
                    expected = os.path.join(ct_seg_dir, f"{pid}{side}{args.ct_seg_basename_suffix}.nii.gz")
                    raise FileNotFoundError(
                        f"Segmentation required for CT but not found: expected {expected} "
                        f"(use --ct_seg_basename_suffix if your filenames differ)"
                    )
                raise FileNotFoundError(
                    f"Segmentation required for CT but not found: expected "
                    f"{os.path.basename(ct_path).replace('_imgCT', args.seg_suffix_ct)} "
                    f"next to {ct_path} (or use --seg_suffix_ct / --ct_seg_dir if segs are elsewhere)"
                )
            if seg_us is None:
                if us_seg_dir is not None:
                    pid, side = extract_patient_side(us_path)
                    expected = os.path.join(us_seg_dir, f"{pid}{side}{args.us_seg_basename_suffix}.nii.gz")
                    raise FileNotFoundError(
                        f"Segmentation required for US but not found: expected {expected} "
                        f"(use --us_seg_basename_suffix if your filenames differ)"
                    )
                raise FileNotFoundError(
                    f"Segmentation required for US but not found: expected "
                    f"{os.path.basename(us_path).replace('_imgUS', args.seg_suffix_us).replace('_aligned', '')} "
                    f"next to {us_path} (or use --seg_suffix_us / --us_seg_dir if segs are elsewhere)"
                )

        data.append(ct_entry)
        data.append(us_entry)

    out = {"data": data}
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {len(data)} entries ({len(pair_keys)} pairs) to {output_path}")
    if unmatched_ct > 0:
        print(f"  (Skipped {unmatched_ct} CT file(s) without matching US)")
    if unmatched_us > 0:
        print(f"  (Skipped {unmatched_us} US file(s) without matching CT)")
    seg_count = sum(1 for e in data if "segmentation" in e)
    if seg_count > 0:
        print(f"  Segmentation paths added: {seg_count} entries")


if __name__ == "__main__":
    main()
