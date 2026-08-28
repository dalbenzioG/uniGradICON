# Contrastive branches for CT–US finetuning: ContraReg patch NCE vs. dense InfoNCE

This repo implements **two mutually-exclusive contrastive auxiliary losses** for
multimodal (CT–US) registration finetuning. Both consume the same pair of frozen,
per-modality autoencoder checkpoints, but they differ in how features are compared and how
gradients reach the registration network.

| Branch | Config switch | Files |
|---|---|---|
| **A. ContraReg patch NCE** | `contrareg_enabled: true` | `src/unigradicon/contrareg.py`, `src/unigradicon/patch_contrastive.py` |
| **B. Dense voxel-wise InfoNCE** | `use_contrastive_loss: true` | `src/unigradicon/contrastive.py` |

Enabling both at once is a config validation error. Both plug into the same total loss:

```
L_total = lambda * L_inverse_consistency + L_similarity + dice_loss_weight * L_dice + w * L_contrastive_branch
```

Branch A is modeled on **ContraReg** (Dey et al., *ContraReg: Contrastive Learning of
Multi-modality Unsupervised Deformable Image Registration*, MICCAI 2022,
arXiv:2206.13434). Branch B is a simpler dense feature-matching InfoNCE — "contrastive
without patches".

---

## Shared prerequisite: pretraining the two autoencoders

Both branches need one frozen autoencoder per modality (preop = CT/MR, us = ultrasound),
pretrained with the ContraReg paper's reconstruction recipe (**L1 + α·local-NCC**;
defaults α = 1.0 for preop, α = 0.5 for US). One command trains both:

```bash
python training/train_encoders.py \
  --preop_json configs/trusted_kidney/encoder_pretrain_train.json \
  --output_dir results/encoder_pretrain
```

- One manifest can serve both encoders: entries are filtered by their `"modality"` field
  (`ct/mri/mr/preop` → preop encoder; `us/ultrasound` → US encoder). Separate
  `--us_json` is optional. Per-modality manifests also exist
  (`encoder_pretrain_train_ct.json`, `encoder_pretrain_train_us.json`).
- Output: `results/encoder_pretrain/preop_encoder.ckpt` and `us_encoder.ckpt`, each a
  self-describing bundle `{arch, arch_kwargs, state_dict, feature_level, metadata}`.
  `feature_level: -2` is recorded so the dense branch picks its feature scale
  automatically.
- Useful knobs: `--epochs` (50), `--batch_size` (2), `--learning_rate` (1e-4),
  `--alpha_ncc_preop` / `--alpha_ncc_us`, `--input_shape` (175 175 175).

The same two `.ckpt` files are then referenced by **both** finetuning branches — no
retraining needed to switch branches.

---

## Branch A — ContraReg patch NCE (`contrareg_enabled`)

### Mechanism

1. The **fixed image** and the **warped moving image** (`I_moving ∘ φ`, both on the fixed
   grid) are each encoded by their own modality's **frozen** autoencoder. Each image is
   routed by its `modality` string from the dataset JSON — the paired sampler may put
   either modality in the moving or fixed slot, so routing is per image, not per role.
2. Encoding runs **without** `torch.no_grad()`: the encoder weights are frozen
   (`requires_grad=False`) but gradients flow *through* the encoder into the warped image
   and hence into the registration network. This is the ContraReg formulation.
3. Multi-scale features (conv1–3: 32/64/128 channels at 1/2, 1/4, 1/8 resolution) are
   sampled at `contrareg_num_patches` shared spatial locations ("patches" = feature-map
   locations whose receptive fields cover image patches — no literal patch extraction),
   projected by **trainable** per-modality 2-layer MLPs (`PatchProjector3D`, 256-d,
   L2-normalized), and compared with a PatchNCE cross-entropy per level, averaged.
4. With `contrareg_roi_patches: true`, patch locations are restricted to the ROI `mask`
   (downsampled per feature level) so background voxels don't dominate the loss. Requires
   `mask` entries in the dataset JSON.
5. The projectors train jointly with the registration net (same Adam) and are saved as
   `contrareg_weights_<epoch>.trch` bundles (keys `projector_preop` / `projector_us`)
   alongside the network/optimizer checkpoints, and are resumed automatically.

### Config keys

| Key | Description | Default |
|---|---|---|
| `contrareg_enabled` | Master switch | false |
| `contrareg_weight` | Loss weight `w` | 0.01 |
| `contrareg_num_patches` | Sampled locations per feature level | 512 |
| `contrareg_temperature` | NCE temperature | 0.07 |
| `contrareg_embed_dim` | Projection MLP output dim | 256 |
| `contrareg_feature_channels` | Per-level channels, must match the AE arch | [32, 64, 128] |
| `contrareg_bidirectional` | Also compute the swapped (k→q) NCE term | false |
| `contrareg_roi_patches` | Restrict sampling to the ROI mask | false |
| `contrareg_preop_ae_checkpoint` | Frozen preop (CT/MR) encoder | — (required) |
| `contrareg_us_ae_checkpoint` | Frozen US encoder | — (required) |

> **Migration note:** the pre-2026-08 keys `contrareg_fixed_ae_checkpoint` /
> `contrareg_moving_ae_checkpoint` are rejected — encoders are now routed by modality, not
> by fixed/moving role. Projector bundles and results produced before that fix were
> affected by a modality mis-routing bug and should be retrained/re-run.

### Fidelity to the ContraReg paper (deliberate deviations)

| Aspect | Paper | This repo |
|---|---|---|
| Feature scales | 6 | 3 |
| Anchors / negatives | all positions, 1024 sampled negatives | 512 sampled anchors; negatives = the other 511 |
| Temperature | 0.007 | 0.07 |
| Projection MLP | 3-layer, 256-wide | 2-layer, 256-wide |
| Direction | bidirectional ½(d₁₂+d₂₁) | unidirectional by default (`contrareg_bidirectional` swaps q/k only) |
| AE pretraining loss | L1 + LNCC | same |

---

## Branch B — dense voxel-wise InfoNCE (`use_contrastive_loss`)

### Mechanism

1. Each *unwarped* input image is encoded by its modality's frozen autoencoder **under
   `torch.no_grad()`** — gradients never enter the encoder.
2. A **single feature level** is used (default: the checkpoint's `feature_level`, −2 =
   64 channels), L2-normalized per voxel, upsampled to full φ resolution.
3. The **feature maps themselves are warped** with φ_AB / φ_BA; gradients reach the
   registration network only through this warp.
4. A symmetric dense InfoNCE compares warped-A features against B features (and vice
   versa) at `contrastive_num_samples` sampled voxel locations; positives are the same
   location, negatives the other sampled locations. When an ROI `mask` is present in the
   batch it restricts sampling to valid voxels (wired automatically when `roi_masking`
   loads masks).
5. No projection heads, no extra trainable parameters, no extra checkpoint files.
   Optional linear warmup of the weight via `contrastive_warmup_epochs`.

### Config keys

| Key | Description | Default |
|---|---|---|
| `use_contrastive_loss` | Master switch | false |
| `contrastive_loss_weight` | Loss weight `w` (after warmup) | 0.0 |
| `contrastive_temperature` | InfoNCE temperature | 0.1 |
| `contrastive_num_samples` | Sampled voxels per direction | 2048 |
| `contrastive_warmup_epochs` | Linear weight warmup | 0 |
| `contrastive_feature_level` | Feature level override; null = from checkpoint (−2) | null |
| `contrastive_normalize_features` | L2-normalize features | true |
| `contrastive_preop_encoder_checkpoint` | Frozen preop encoder | — (required) |
| `contrastive_us_encoder_checkpoint` | Frozen US encoder | — (required) |

---

## Differences at a glance, and which to use

| | A. ContraReg patch NCE | B. Dense InfoNCE |
|---|---|---|
| Gradient path | through the frozen encoder (warped image encoded) | through the feature warp only (encoder no-grad) |
| Feature scales | 3 (multi-scale) | 1 |
| Trainable parts | 2 projection MLPs (per modality) | none |
| Cross-modal alignment | **learned** by the projectors during finetuning | relies on raw AE embeddings being comparable (AEs are trained independently — not guaranteed) |
| Direction | fixed vs warped-moving (optionally +swapped NCE) | symmetric (both warp directions) |
| Extra checkpoints | `contrareg_weights_*.trch` | none |
| Memory | moderate (features stay at native resolutions) | high: features are upsampled to full 175³ before warping (~1.4 GB per fp32 tensor at level −2; use `contrastive_feature_level: -3` (32 ch) if OOM) |
| Faithfulness to ContraReg paper | close (see deviations table) | different method — cite as "dense feature-space InfoNCE", not ContraReg |

**Guidance:** Branch A is the principled default for the CT–US setting — the trainable
projectors learn the cross-modal embedding alignment that Branch B simply assumes, and it
is the one backed by a published method. Branch B is the natural ablation: it isolates the
value of (multi-scale + learned projection + through-encoder gradients) by removing all
three. If B performs comparably, the simpler formulation suffices; if A wins, the
ContraReg machinery is earning its complexity.

---

## Step-by-step ablation recipe (TRUSTED kidney CT–US)

All three configs share the same data (ROI manifest), MIND (r2/d2) + Dice 0.3 +
`roi_masking`, lr 5e-5, batch 1, 200 epochs — they differ only in the contrastive branch:

1. **Baseline (no contrastive):**
   `configs/trusted_kidney/config_trusted_ct_us_seg-mind_dice_roi_fixed.yaml`
2. **+ ContraReg patch NCE:**
   `configs/trusted_kidney/config_trusted_ct_us_seg_contrastive_mind_dice.yaml`
3. **+ dense InfoNCE:**
   `configs/trusted_kidney/config_trusted_ct_us_seg_dense-infonce_mind_dice_roi_fixed.yaml`

```bash
# once: pretrain the encoders (feeds configs 2 and 3)
python training/train_encoders.py \
  --preop_json configs/trusted_kidney/encoder_pretrain_train.json \
  --output_dir results/encoder_pretrain

# then, per config:
unigradicon-finetune --config configs/trusted_kidney/<config>.yaml
```


