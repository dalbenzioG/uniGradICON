"""Build finetuning-style encoder pretrain JSON manifests for TRUSTED data."""

import argparse
import json
import re
import sys
from pathlib import Path


def case_id_to_subject_id(case_id: str) -> str:
    match = re.fullmatch(r"(\d+)([LR])", case_id)
    if not match:
        return case_id
    return f"{match.group(1)}_{match.group(2)}"


def _path_for_json(path: Path, json_dir: Path) -> str:
    try:
        return path.relative_to(json_dir).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def build_from_eval_manifest(eval_manifest: Path, out_dir: Path) -> tuple[list, list, list]:
    payload = json.loads(eval_manifest.read_text(encoding="utf-8"))
    combined: list = []
    ct_only: list = []
    us_only: list = []

    for case in payload["data"]:
        case_id = case["case_id"]
        subject_id = case_id_to_subject_id(case_id)
        for side in ("moving", "fixed"):
            entry = case[side]
            image_path = Path(entry["image"])
            if not image_path.is_absolute():
                image_path = (eval_manifest.parent / image_path).resolve()
            if not image_path.exists():
                print(f"WARNING: missing image: {image_path}", file=sys.stderr)

            item = {
                "image": _path_for_json(image_path, out_dir),
                "subject_id": subject_id,
                "modality": entry["modality"],
            }
            if "segmentation" in entry:
                seg_path = Path(entry["segmentation"])
                if not seg_path.is_absolute():
                    seg_path = (eval_manifest.parent / seg_path).resolve()
                item["segmentation"] = _path_for_json(seg_path, out_dir)

            combined.append(item)
            if entry["modality"] == "ct":
                ct_only.append(item)
            elif entry["modality"] == "us":
                us_only.append(item)

    return combined, ct_only, us_only


def build_from_train_dir(train_root: Path, out_dir: Path) -> tuple[list, list, list]:
    ct_images_dir = train_root / "CT_images"
    us_images_dir = train_root / "US_images"
    ct_masks_dir = train_root / "CT_masks"
    us_masks_dir = train_root / "US_masks"

    for folder in (ct_images_dir, us_images_dir, ct_masks_dir, us_masks_dir):
        if not folder.is_dir():
            raise FileNotFoundError(f"Expected directory not found: {folder}")

    combined: list = []
    ct_only: list = []
    us_only: list = []

    ct_images = sorted(ct_images_dir.glob("*_imgCT.nii.gz"))
    for ct_image in ct_images:
        case_id = ct_image.name.replace("_imgCT.nii.gz", "")
        subject_id = case_id_to_subject_id(case_id)
        ct_mask = ct_masks_dir / f"{case_id}_maskCT.nii.gz"
        us_image = us_images_dir / f"{case_id}_imgUS.nii.gz"
        us_mask = us_masks_dir / f"{case_id}_maskUS.nii.gz"

        if not ct_mask.exists():
            print(f"WARNING: missing CT mask: {ct_mask}", file=sys.stderr)
        if not us_image.exists():
            print(f"WARNING: missing US image: {us_image}", file=sys.stderr)
            continue
        if not us_mask.exists():
            print(f"WARNING: missing US mask: {us_mask}", file=sys.stderr)

        ct_item = {
            "image": _path_for_json(ct_image, out_dir),
            "subject_id": subject_id,
            "modality": "ct",
        }
        if ct_mask.exists():
            ct_item["segmentation"] = _path_for_json(ct_mask, out_dir)

        us_item = {
            "image": _path_for_json(us_image, out_dir),
            "subject_id": subject_id,
            "modality": "us",
        }
        if us_mask.exists():
            us_item["segmentation"] = _path_for_json(us_mask, out_dir)

        ct_only.append(ct_item)
        us_only.append(us_item)
        combined.extend([ct_item, us_item])

    return combined, ct_only, us_only


def write_manifests(
    out_dir: Path,
    prefix: str,
    combined: list,
    ct_only: list,
    us_only: list,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix, data in (
        ("", combined),
        ("_ct", ct_only),
        ("_us", us_only),
    ):
        path = out_dir / f"encoder_pretrain_{prefix}{suffix}.json"
        path.write_text(json.dumps({"data": data}, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {path} ({len(data)} entries)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=("eval", "train", "both"),
        default="both",
        help="Which manifest(s) to build.",
    )
    parser.add_argument(
        "--train-root",
        type=Path,
        default=Path("C:/Users/gabridal/Documents/TRUSTED_reg/train"),
        help="TRUSTED train folder containing CT_images, US_images, CT_masks, US_masks.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "configs" / "trusted_kidney"
    eval_manifest = out_dir / "eval_test_manifest.json"

    if args.split in ("eval", "both"):
        combined, ct_only, us_only = build_from_eval_manifest(eval_manifest, out_dir)
        write_manifests(out_dir, "eval", combined, ct_only, us_only)

    if args.split in ("train", "both"):
        combined, ct_only, us_only = build_from_train_dir(args.train_root.resolve(), out_dir)
        write_manifests(out_dir, "train", combined, ct_only, us_only)


if __name__ == "__main__":
    main()
