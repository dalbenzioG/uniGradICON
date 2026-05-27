# Evaluation with uniGradICON / multiGradICON

This guide walks through test-set evaluation with `unigradicon-eval`: deformable registration plus metrics (TRE, Chamfer, Dice, MI, HD95).

## Table of Contents

- [Quick Start](#quick-start)
- [Step-by-Step Guide](#step-by-step-guide)
- [TRUSTED Kidney CT–US Pipeline](#trusted-kidney-ctus-pipeline)
- [Method Config Reference](#method-config-reference)
- [Initial-Transform and TRE Flags](#initial-transform-and-tre-flags)
- [Outputs](#outputs)
- [Troubleshooting](#troubleshooting)
- [Templates](#templates)

## Quick Start

**Recommended:** CUDA-capable GPU.

```bash
pip install -e .   # from repo root, if not already installed

unigradicon-eval --config configs/trusted_kidney/eval_method_trusted_mind_dice.yaml
```

Templates live under `configs/trusted_kidney/`.

---

## Step-by-Step Guide

### Step 1: Prepare volumes

Collect per case:

| Role | Typical TRUSTED paths | Notes |
|------|----------------------|--------|
| Fixed (US) | `US_images/`, `US_masks/` | Resampled + cropped to match your training pipeline |
| Moving (CT) | `CT_images/`, `CT_masks/` | Same preprocessing as used for finetuning |
| Landmarks | `US_landmarks/`, `CT_landmarks_aligned/` | See [TRUSTED pipeline](#trusted-kidney-ctus-pipeline) |
| Rigid init | `rigid_transforms/*_CT_to_US_pca_icp.tfm` | Provenance only if landmarks are pre-aligned |
| Weights | `network_weights_final.trch` | Same architecture as finetuning (`multigradicon`) |

**Important:** Images, masks, segmentations, and landmarks must describe the **same preprocessing stage** (same resample/crop). Eval does not rerun your offline resample/crop scripts.

Set `input_shape: [175, 175, 175]` in the eval YAML to match MultiGradICON finetuning and limit GPU memory.

### Step 2: Create the test manifest (JSON)

Top-level key `data`; one object per case. Use **absolute paths**.

```json
{
  "data": [
    {
      "case_id": "206L",
      "fixed": {
        "image": "/abs/path/US_images/206L_imgUS.nii.gz",
        "modality": "us",
        "segmentation": "/abs/path/US_masks/206L_maskUS.nii.gz",
        "mask": "/abs/path/US_masks/206L_maskUS.nii.gz",
        "landmarks": "/abs/path/landmarks_lps/206L_ldkUS_lps.txt"
      },
      "moving": {
        "image": "/abs/path/CT_images/206L_imgCT.nii.gz",
        "modality": "ct",
        "segmentation": "/abs/path/CT_masks/206L_maskCT.nii.gz",
        "mask": "/abs/path/CT_masks/206L_maskCT.nii.gz",
        "landmarks": "/abs/path/landmarks_lps/206L_ldkCT_aligned_lps.txt"
      },
      "initial_transform": "/abs/path/rigid_transforms/206L_CT_to_US_pca_icp.tfm",
      "metrics": {
        "tre": true,
        "chamfer": true,
        "dice": true,
        "mi": true,
        "hd95": true
      }
    }
  ]
}
```

Rules:

- **TRE** requires `fixed.landmarks` and `moving.landmarks`.
- **Chamfer / Dice / HD95** require both segmentations.
- `modality: "us"` is fine; eval routes US through the non-CT preprocessing branch internally.

### Step 3: Create the method config (YAML)

See [Method Config Reference](#method-config-reference). For preprocessed TRUSTED CT–US with offline-aligned LPS landmarks:

```yaml
method_name: "trusted_kidney_mind_dice"
test_manifest: "eval_test_manifest_lps_prealigned.json"

model: "multigradicon"
network_weights: "/abs/path/network_weights_final.trch"

io_iterations: 20
io_lr: 0.00005
io_sim: "mind"
lambda: 1.5
dice_loss_weight: 0.3

input_shape: [175, 175, 175]
ct_window: [-1000, 1000]
quantile_range: [0.0, 0.99]

output_root: "/abs/path/results"
metrics: ["tre", "chamfer", "dice", "mi", "hd95"]
resume: false
cache_preprocessed: false

apply_initial_transform_to_images: false
apply_initial_transform_to_landmarks: false
invert_initial_transform_for_tre: false
```

### Step 4: Run evaluation

```bash
unigradicon-eval --config /abs/path/eval_method.yaml
```

After changing landmarks, manifest paths, or LPS conversion, set `resume: false` and `cache_preprocessed: false` (or delete `preprocessed_cache/` and `per_case_metrics.csv`).

### Step 5: Inspect results

Under `{output_root}/{method_name}/`:

| File | Content |
|------|---------|
| `per_case_metrics.csv` | One row per case; includes `tre`, `tre_baseline_raw_mm`, `chamfer`, `dice`, … |
| `summary.csv` | Mean / std / median / IQR per metric |
| `failures.json` | Case-level errors |
| `transforms/<case_id>.hdf5` | Deformable `phi_ab` (moving → fixed), if `save_transforms: true` |

**Sanity check:** On good cases, `tre` should be close to `tre_baseline_raw_mm` and the same order of magnitude as `chamfer` (~5–15 mm). If `tre_baseline_raw_mm` is good but `tre` is ~50 mm, landmarks are likely out of ITK physical bounds (see [Troubleshooting](#troubleshooting)).

---

## TRUSTED Kidney CT–US Pipeline

End-to-end workflow when CT/US volumes are **already resampled and cropped offline**, and you want correct **TRE in mm**.

```text
Volumes (offline)          Landmarks (scripts)              Eval
─────────────────          ─────────────────────            ────
CT_images/                 US_landmarks/  (RAS)            unigradicon-eval
US_images/        ──►      CT_landmarks_aligned/ (RAS) ──►  + LPS manifest
CT_masks/                  landmarks_lps/ (LPS)             apply_initial_*: false
US_masks/
```

### Coordinate conventions

| Convention | Used by | TRUSTED `.txt` export |
|------------|---------|------------------------|
| **RAS** | 3D Slicer display / many exports | Typical if fiducials look correct in Slicer but eval says “outside bounds” |
| **LPS (ITK physical)** | `itk.imread`, `phi_ab`, Chamfer/Dice | Required for deformable TRE |

Crop/resample of volumes does **not** change landmark coordinates (ITK physical points are absolute). The manifest must still point at the **same resampled/cropped NIfTIs** used for registration.

The PCA+ICP `report_json` matrix (`resample_matrix`) matches **RAS** landmark files. Apply rigid alignment in RAS, then convert **both** US and aligned CT to LPS. Do **not** run `align` on LPS native CT with `report_json` (that mixes conventions).

### Pipeline commands

Run from the repo root with your eval conda env active.

**0. Manifest with aligned CT landmarks (RAS)**

Point `moving.landmarks` at aligned CT files and `fixed.landmarks` at native US files. Example: `configs/trusted_kidney/eval_test_manifest_ras_aligned.json`.

**1. Probe RAS → LPS**

```bash
python scripts/convert_landmarks_to_lps.py probe \
  --manifest configs/trusted_kidney/eval_test_manifest_ras_aligned.json
```

Expect `ras_to_lps` → `inside=14/14` and `baseline_raw` ~5–15 mm per case (same value under `identity` and `ras_to_lps` rows).

**Alternative: align with `report_json` (RAS native CT)**

If CT landmarks are still native (not manually aligned):

```bash
python scripts/align_eval_landmarks.py probe \
  --manifest configs/trusted_kidney/eval_test_manifest.json

python scripts/align_eval_landmarks.py align \
  --manifest configs/trusted_kidney/eval_test_manifest.json \
  --landmark-space inimg \
  --output-dir /abs/path/CT_landmarks_aligned \
  --write-manifest configs/trusted_kidney/eval_test_manifest_ras_aligned.json
```

Use `report_json` direct (`invert_initial_transform_for_tre: false`) unless probe shows inverted wins.

**2. Convert US + aligned CT to LPS**

```bash
python scripts/convert_landmarks_to_lps.py convert \
  --manifest configs/trusted_kidney/eval_test_manifest_ras_aligned.json \
  --convention ras_to_lps \
  --output-dir /abs/path/landmarks_lps \
  --write-manifest configs/trusted_kidney/eval_test_manifest_lps_prealigned.json
```

Converts **both** fixed (US) and moving (aligned CT). Do not run `align` after this step.

**3. Validate LPS**

```bash
python scripts/convert_landmarks_to_lps.py validate \
  --manifest configs/trusted_kidney/eval_test_manifest_lps_prealigned.json \
  --convention identity
```

Pass criteria: `fixed_inside=True`, `moving_inside=True`, `baseline_raw` ~5–15 mm.

**4. Eval**

```yaml
test_manifest: "eval_test_manifest_lps_prealigned.json"
apply_initial_transform_to_images: false
apply_initial_transform_to_landmarks: false
resume: false
cache_preprocessed: false
```

```bash
unigradicon-eval --config configs/trusted_kidney/eval_method_trusted_mind_dice.yaml
```

### Validate aligned landmarks in RAS (optional)

To check rigid pairing without double-applying `report_json` when files are already aligned:

```bash
python scripts/align_eval_landmarks.py validate \
  --manifest configs/trusted_kidney/eval_test_manifest.json \
  --aligned-dir /abs/path/CT_landmarks_aligned
```

Do **not** run `align_eval_landmarks.py validate` on a manifest whose `moving.landmarks` already points at `*_aligned.txt` without `--aligned-dir` (otherwise `report_json` is applied twice).

---

## Method Config Reference

| Field | Description |
|-------|-------------|
| `method_name` | Output subdirectory under `output_root` |
| `test_manifest` | Path to case JSON (relative to YAML dir or absolute) |
| `model` | `unigradicon` or `multigradicon` |
| `network_weights` | Finetuned checkpoint (`.trch`) |
| `io_iterations`, `io_lr`, `io_sim` | Deformable registration (match finetuning) |
| `lambda`, `dice_loss_weight` | Loss weights (match finetuning) |
| `input_shape` | `[175,175,175]` recommended for MultiGradICON kidney runs |
| `ct_window`, `quantile_range` | Preprocessing; match finetuning |
| `metrics` | Subset of: `tre`, `chamfer`, `dice`, `mi`, `hd95` |
| `resume` | Skip cases already in `per_case_metrics.csv` |
| `cache_preprocessed` | Cache 175³ preprocessed volumes |
| `save_transforms` | Write `phi_ab` per case to `transforms/` |
| `chamfer_*` | Surface point sampling for Chamfer |

---

## Initial-Transform and TRE Flags

| `apply_initial_transform_to_images` | `apply_initial_transform_to_landmarks` | When |
|-------------------------------------|----------------------------------------|------|
| `false` | `false` | **Recommended:** preprocessed CT/US + LPS pre-aligned landmarks |
| `false` | `true` | Native CT landmarks; apply `report_json` at TRE time only |
| `true` | `true` | Native volumes; rigid + deformable inside one eval run |

`invert_initial_transform_for_tre`: set `true` only if `align_eval_landmarks.py probe` shows `report_json_inverted` wins (default `false` for TRUSTED).

- `tre_baseline_raw_mm`: landmark distance after rigid (or from pre-aligned files).
- `tre` / `tre_phi_only_mm`: after deformable `phi_ab`. Should track Chamfer when landmarks are in ITK LPS and inside the registration grid.

---

## Outputs

Merge summaries from multiple methods (optional):

```bash
unigradicon-eval \
  --merge_glob "results/eval/*/summary.csv" \
  --merge_out "results/eval/final_summary.csv"
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `tre` ~50 mm, `tre_baseline_raw_mm` ~7 mm, Chamfer ~6 mm | Landmarks in RAS or outside ITK bounds | LPS convert + validate; use LPS prealigned manifest |
| `inside=0/14` on probe `identity`, `14/14` on `ras_to_lps` | Slicer RAS export | Convert US + aligned CT with `ras_to_lps` |
| `align` baseline ~600 mm after LPS convert | `report_json` on LPS points | Align in RAS first, then LPS convert only |
| `align` validate baseline ~900 mm on prealigned manifest | Double rigid transform | Use native manifest + `--aligned-dir` |
| `tre_baseline_raw_mm` ~50–70 mm on one case | Landmark pairing / L–R split | Fix in Slicer; re-export aligned CT |
| Stale metrics after pipeline change | `resume: true` or cached preprocess | `resume: false`; delete cache / `per_case_metrics.csv` |

General checks:

- Every manifest path exists.
- Segmentation paths match the same preprocessing as images.
- `metrics` keys are only: `tre`, `chamfer`, `dice`, `mi`, `hd95`.
- GPU OOM: lower `input_shape`.

---

## Templates

Under `configs/trusted_kidney/`:

| File | Purpose |
|------|---------|
| `eval_method_template.yaml` | Method config skeleton |
| `eval_test_manifest_template.json` | Manifest skeleton |
| `eval_test_manifest.json` | Native RAS landmarks |
| `eval_test_manifest_ras_aligned.json` | RAS aligned CT + native US |
| `eval_test_manifest_lps_prealigned.json` | LPS landmarks for eval (after convert) |
| `eval_method_trusted_mind_dice.yaml` | Example MultiGradICON kidney eval |

Helper scripts (repo `scripts/`):

| Script | Role |
|--------|------|
| `convert_landmarks_to_lps.py` | RAS → ITK LPS; `probe`, `convert`, `validate` |
| `align_eval_landmarks.py` | Rigid `report_json` align; `probe`, `align`, `validate` |
