# Finetuning uniGradICON on Your Data

This guide shows you how to finetune uniGradICON on your own datasets using configuration files. The finetuning system supports multiple datasets, weighted sampling, and segmentation-based training.

## 📋 Table of Contents
- [Quick Start](#-quick-start)
- [Step-by-Step Guide](#-step-by-step-guide)
- [Configuration Guide](#-configuration-guide)
- [Dataset Types](#-dataset-types)
- [Advanced Features](#-advanced-features)
- [Dice Loss and Masking](#-dice-loss-and-masking)

## 🚀 Quick Start

**Install (PyPI or source):**
- PyPI: `pip install unigradicon`
- Dev/source: `pip install -e .` from the repo root

```bash
# Run with your config
unigradicon-finetune --config /path/to/your_config.yaml

# Example config path (in repo checkout)
python -m unigradicon.finetuning.finetune --config src/unigradicon/finetuning/configs/config.yaml
```

## Step-by-Step Guide

### Step 1: Prepare Your Data

Organize your data and create a JSON file to define your datasets. All datasets use a consistent JSON format with a `data` list.

**Example `dataset.json` for unpaired data:**
```json
{
  "data": [
    {"image": "/path/to/img1.nii.gz"},
    {"image": "/path/to/img2.nii.gz"}
  ]
}
```

**Example `dataset.json` for paired data:**
```json
{
  "data": [
    {"image": "/path/p1_t0.nii.gz", "subject_id": "p1"},
    {"image": "/path/p1_t1.nii.gz", "subject_id": "p1"}
  ]
}
```

### Step 2: Create a Configuration File

Create a YAML file (e.g., `my_config.yaml`) in the `configs/` directory:

```yaml
experiment:
  name: "my_finetuning_experiment"
  model: "unigradicon"  # or "multigradicon"
  weights_path: "unigradicon"  # Auto-downloads if not found

training:
  batch_size: 4
  gpus: [0, 1]  # GPU device IDs
  epochs: 100
  eval_period: 10  # Validate every N epochs
  save_period: 50  # Save checkpoint every N epochs
  learning_rate: 0.00005
  input_shape: [175, 175, 175]  # Target image size
  
  # Loss configuration
  lambda: 1.5  # Regularization weight
  similarity: "lncc"  # Options: "lncc", "lncc2", "mind"
  lncc_sigma: 5  # For LNCC losses
  dice_loss_weight: 0.0  # >0 only for segmentation datasets
  loss_function_masking: false  # When true, Dice is disabled and must stay 0.0

datasets:
  - name: "my_dataset"
    weight: 1.0  # Relative sampling weight (any positive value)
    type: "unpaired"  # See Dataset Types below
    json_file: "my_dataset.json"
    maximum_images: null  # Optional: limit number of images
    shuffle: true
    is_ct: false  # Set to true for CT images
    quantile_range: [0.01, 0.99]  # For MRI normalization
```

### Step 3: Start Training

```bash
unigradicon-finetune --config configs/my_config.yaml
```

### Step 4: Monitor Training

Training progress is logged to TensorBoard. Validation writes scalar losses plus image panels
(moving/fixed/warped/difference), and segmentation panels when segmentation datasets are used:

```bash
# Footsteps stores runs in results/<experiment.name>/logs/<timestamp>
tensorboard --logdir="results/my_finetuning_experiment/logs"
# If you rerun with the same name, use the suffixed folder (e.g., results/my_finetuning_experiment-1/logs)
```

### Step 5: Use Your Finetuned Model

After training, use your model weights for inference:

```bash
# Your weights are saved in results/<experiment.name>/checkpoints/
ls results/my_finetuning_experiment/checkpoints/

# Use with uniGradICON CLI
unigradicon-register \
  --fixed fixed.nii.gz \
  --moving moving.nii.gz \
  --transform_out transform.hdf5 \
  --warped_moving_out warped.nii.gz \
  --network_weights results/my_finetuning_experiment/checkpoints/network_weights_100
```

## ⚙️ Configuration Guide

### Training Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `batch_size` | int | Images per GPU | 4 |
| `gpus` | list | GPU device IDs | [0] |
| `epochs` | int | Training epochs | 500 |
| `learning_rate` | float | Adam learning rate | 5e-5 |
| `input_shape` | list | Target image dimensions [D,H,W] used during finetuning | [175,175,175] |
| `eval_period` | int | Validate every N epochs | 15 |
| `save_period` | int | Save checkpoint every N epochs | 50 |
| `lambda` | float | Regularization weight | 1.5 |
| `similarity` | str | Loss function: "lncc", "lncc2", "mind" | "lncc" |
| `dice_loss_weight` | float | Dice loss term weight for segmentation mode | 0.0 |
| `loss_function_masking` | bool | Apply segmentation mask to similarity loss (segmentation mode only) | false |
| `samples_per_epoch` | int | Samples per epoch (optional) | null |

### Input Shape Guidance

- The released `unigradicon` / `multigradicon` weights were trained at `input_shape: [175, 175, 175]`.
- You can finetune with a different `input_shape`, but convergence may be slower and you may need more epochs.
- For the most stable transfer behavior, keep finetuning and inference preprocessing shapes consistent.
- If you finetune at a very different shape, inference quality can change because the model sees a different resampling distribution than pretraining.

### Dataset Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `name` | str | Dataset identifier | ✓ |
| `weight` | float | Sampling weight | ✓ |
| `type` | str | Dataset type (see below) | ✓ |
| `json_file` | str | Path to JSON dataset definition | ✓ |
| `maximum_images` | int | Limit number of images | Optional |
| `use_cache` | bool | Enable/disable caching | true |
| `is_ct` | bool | CT vs MRI preprocessing | false |

`json_file` paths are resolved relative to the YAML config file's directory, so you can usually reference just the filename.

## Dataset Types

### 1. Unpaired Dataset (`unpaired`)
Random pairs of images from different subjects.

```yaml
datasets:
  - name: "brain_mri"
    type: "unpaired"
    json_file: "brain_mri.json"
    weight: 1.0
```

**JSON Format:**
```json
{
  "data": [
    {"image": "/path/to/img1.nii.gz"},
    {"image": "/path/to/img2.nii.gz"}
  ]
}
```

### 2. Paired Dataset (`paired`)
Matched pairs of images from the same subject.

```yaml
datasets:
  - name: "lung_followup"
    type: "paired"
    json_file: "lung_pairs.json"
    weight: 1.0
```

**JSON Format:**
```json
{
  "data": [
    {"image": "/path/p1_t0.nii.gz", "subject_id": "p1"},
    {"image": "/path/p1_t1.nii.gz", "subject_id": "p1"}
  ]
}
```

### 3. Unpaired with Segmentation (`unpaired_with_seg`)
Random pairs with segmentation guidance using Dice loss.

```yaml
datasets:
  - name: "brain_structures"
    type: "unpaired_with_seg"
    json_file: "brain_seg.json"
    weight: 1.0
```

**JSON Format:**
```json
{
  "data": [
    {"image": "/path/img1.nii.gz", "segmentation": "/path/seg1.nii.gz"}
  ]
}
```

### 4. Paired with Segmentation (`paired_with_seg`)
Paired images with segmentation guidance.

```yaml
datasets:
  - name: "cardiac_phases"
    type: "paired_with_seg"
    json_file: "cardiac.json"
    weight: 1.0
```

**JSON Format:**
```json
{
  "data": [
    {"image": "/path/p1_t0.nii.gz", "segmentation": "/path/p1_t0_seg.nii.gz", "subject_id": "p1"},
    {"image": "/path/p1_t1.nii.gz", "segmentation": "/path/p1_t1_seg.nii.gz", "subject_id": "p1"}
  ]
}
```

### 5. Paired CT-US (`paired_ct_us` and `paired_ct_us_seg`)

For **cross-modality pairs** (one CT and one US per subject), use dedicated types so each image gets the correct preprocessing (CT: HU window; US: quantile/min-max). The repo’s usual CT/MRI setup uses **separate datasets** (e.g. one unpaired MRI, one unpaired CT); there is no built-in “one CT + one MRI in the same pair.” For CT–US pairs, use:

- **`paired_ct_us`**: returns `(moving_image, fixed_image)` with moving=US, fixed=CT (standard training loop).
- **`paired_ct_us_seg`**: returns `(moving_image, fixed_image, moving_seg, fixed_seg)` for Dice loss (segmentation training loop).

**JSON format:** One `subject_id` per physical pair, two entries per pair (one CT, one US). Only include pairs where both CT and US exist. Each entry must have `"modality": "ct"` or `"us"`. For `paired_ct_us_seg`, every entry must also have a `"segmentation"` path.

```json
{
  "data": [
    {"image": "path/to/200L_imgCT.nii.gz", "segmentation": "path/to/200L_segCT.nii.gz", "subject_id": "200_L", "modality": "ct"},
    {"image": "path/to/200L_imgUS.nii.gz", "segmentation": "path/to/200L_segUS.nii.gz", "subject_id": "200_L", "modality": "us"}
  ]
}
```

**Build the JSON from TRUSTED-style directories:** Use the provided script to scan CT/US dirs, match by (patient_id, side), and write the JSON (with optional segmentation paths when files exist):

```bash
python -m unigradicon.finetuning.scripts.build_trusted_pairs_json \
  --data_path /path/to/trusted_code_nsd \
  --output configs/trusted_pairs.json
```

Optional: `--ct_dir`, `--us_dir`, `--seg_suffix_ct`, `--seg_suffix_us`. For **paired_ct_us_seg**, every pair must have segmentation files. **If segmentations are in separate folders** (e.g. `CT_masks/`, `US_masks/` with files like `200R_seg.nii.gz` and `200R_maskUS.nii.gz`), the script auto-detects `{data_path}/CT_masks` and `{data_path}/US_masks` when present, and uses `--ct_seg_basename_suffix _seg` and `--us_seg_basename_suffix _maskUS` by default. Override with `--ct_seg_dir`, `--us_seg_dir`, `--ct_seg_basename_suffix`, `--us_seg_basename_suffix` if your layout or naming differs. If segs sit next to images, use `--seg_suffix_ct` / `--seg_suffix_us` (e.g. `200R_segCT.nii.gz`). Add `--require_segmentation` to fail if any pair is missing segmentations. See script docstring for details.

**Example config (paired CT-US with segmentation):**

```yaml
datasets:
  - name: "trusted_ct_us_seg"
    type: "paired_ct_us_seg"
    weight: 1.0
    json_file: "trusted_pairs.json"
    ct_window: [-1000, 1000]
    quantile_range: [0.0, 1.0]
```

Use `type: "paired_ct_us"` (and the standard training loop, no `dice_loss_weight`) when you do not have segmentations.

**Optional: pre-computed affine transforms**

You can resample one modality into the other’s space before training (e.g. US into CT space so the network sees roughly aligned pairs). Add these optional keys under the dataset entry:

- **`affine_transforms_dir`**: Directory containing `.tfm` files (e.g. `rigid_masks_transforms/`). Path is relative to the config file’s directory if not absolute. Omit or leave null to disable.
- **`affine_direction`**: `"us_to_ct"` (resample US into CT space; moving = US in CT space, fixed = CT) or `"ct_to_us"` (resample CT into US space; moving = US, fixed = CT in US space). Default: `"us_to_ct"`.
- **`affine_type`**: Suffix used in transform filenames (e.g. `"rigid_mask"`). Default: `"rigid_mask"`.

Transform filenames must match `subject_id` and direction: e.g. `200_R_US_to_CT_rigid_mask.tfm` for `us_to_ct` and `affine_type=rigid_mask`, or `200_R_CT_to_US_rigid_mask.tfm` for `ct_to_us`. When using `paired_ct_us_seg`, each image’s segmentation is resampled with the same transform and reference grid when that image is resampled, so segmentations stay aligned with the images.

## Advanced Features

### Multi-Dataset Training

Train on multiple datasets simultaneously with weighted sampling:

```yaml
datasets:
  - name: "brain_t1"
    weight: 0.4
    type: "unpaired"
    json_file: "brain_t1.json"
  
  - name: "brain_t2"
    weight: 0.3
    type: "unpaired"
    json_file: "brain_t2.json"
  
  - name: "lung_ct"
    weight: 0.3
    type: "unpaired"
    json_file: "lung_ct.json"
    is_ct: true
    ct_window: [-1000, 1000]
```

**Note:** Weights are relative; they do not need to sum to 1.0.

## 🎯 Dice Loss and Masking

When training with segmentation datasets (`unpaired_with_seg` or `paired_with_seg`), the total loss is:

`L_total = lambda * L_inverse_consistency + L_similarity + dice_loss_weight * L_dice`

- `dice_loss_weight` controls how strongly segmentation overlap is optimized.
- Set `dice_loss_weight: 0.0` to disable Dice.

### Important masking rule

If you enable:

```yaml
training:
  loss_function_masking: true
```

then Dice loss is not calculated in finetuning. In this mode, `dice_loss_weight` must be `0.0`.

### Auto-Download Pretrained Weights

Specify model name instead of path to auto-download:

```yaml
experiment:
  weights_path: "unigradicon"      # Auto-downloads uniGradICON weights
  # OR
  weights_path: "multigradicon"    # Auto-downloads multiGradICON weights
  # OR
  weights_path: "/path/to/my/weights.trch"  # Use custom weights
```

### Resume Training

The system automatically detects if you're resuming from a checkpoint:

```yaml
experiment:
  weights_path: "results/my_experiment/checkpoints/network_weights_50"
```

If `optimizer_weights_50` exists, training resumes with optimizer state. Otherwise, it starts fresh with the model weights.

### Control Samples Per Epoch

For large datasets or faster testing:

```yaml
training:
  samples_per_epoch: 4000  # Process 4000 samples per epoch
```

Without this parameter, all dataset samples are used each epoch.

### Disable Caching

For debugging or frequently changing data:

```yaml
datasets:
  - name: "test_dataset"
    type: "unpaired"
    json_file: "test.json"
    use_cache: false  # Reload images every time
```

**Default:** Caching is enabled. Cached data is stored in `results/[dataset_name]_cache/`.

### CT vs MRI Preprocessing

In multi-dataset configs, CT and MRI are used as **separate datasets** (e.g. one unpaired MRI dataset and one unpaired CT dataset). Each batch is then either two MRI or two CT images—there is no “one CT + one MRI in the same pair” in the standard types. For **CT–US paired data**, use the dedicated types `paired_ct_us` or `paired_ct_us_seg` (see [Paired CT-US](#5-paired-ct-us-paired_ct_us-and-paired_ct_us_seg)).

**MRI (default):**
```yaml
datasets:
  - name: "mri_dataset"
    is_ct: false
    quantile_range: [0.01, 0.99]  # Normalize using quantiles
```

**CT:**
```yaml
datasets:
  - name: "ct_dataset"
    is_ct: true
    ct_window: [-1000, 1000]  # HU windowing
```

## Troubleshooting

### "JSON file not found"
- Check that your `json_file` path is correct and accessible.
- Use absolute paths to avoid confusion.

### "Data must be provided"
- Ensure your JSON file contains the top-level `data` key.

### Weights are relative
- Sampler treats weights as relative multipliers; they do not need to sum to 1.0.
- Keep weights positive to avoid invalid sampler behavior.

### Cache takes too much disk space
- Set `use_cache: false`
- Delete old caches: `rm -rf results/*_cache`
- Use `maximum_images` to limit dataset size

### Transforms / augmentation
- Data augmentation (e.g. spatial transforms) is not included in this finetuning setup. A future option may add an optional `transforms` or `augmentation` section in the training YAML; if so, the same transform would be applied to both moving and fixed (and to segmentations when using segmentation types) to preserve spatial correspondence.
