"""Two-stage from-scratch multigradicon training on Trusted CT-US pairs."""

import logging
import os
import random
from datetime import datetime
from typing import Any, Dict, FrozenSet, List, Optional

import footsteps
import torch
import unigradicon
from icon_registration import config as icon_config
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from unigradicon.finetuning.config import (
    ConfigSections,
    ExperimentKeys,
    FinetuningConfigSchema,
    TrainingConfig,
    TrainingKeys,
    create_scratch_data_loaders,
    load_config,
    set_reproducibility_seed,
    validate_scratch_training_config,
    DEFAULT_TRAIN_WANDB_PROJECT,
)
from unigradicon.finetuning.dataset import Fields, PairKeys
from unigradicon.finetuning.finetune import (
    CHECKPOINT_DIR,
    DEFAULT_LOG_PERIOD,
    _apply_roi_masking,
    _build_forward_kwargs,
    _initialize_footsteps,
    _log_loss_scalars_with_wandb,
    _save_checkpoint,
    _train_one_batch,
    loss_to_dict,
)
from unigradicon.finetuning.visualization import add_eval_composite_panel

logger = logging.getLogger(__name__)

STEP_1_FINAL = "Step_1_final.trch"
STEP_2_FINAL = "Step_2_final.trch"
SECOND_STEP_SUBDIR = "2nd_step/"


def _build_loss_fn(settings: TrainingConfig) -> Any:
    return unigradicon.make_sim(
        settings.similarity.lower(),
        sigma=settings.lncc_sigma,
        mind_radius=settings.mind_radius,
        mind_dilation=settings.mind_dilation,
    )


def _build_registration_network(
    settings: TrainingConfig,
    loss_fn: Any,
    include_last_step: bool,
) -> Any:
    return unigradicon.make_network(
        settings.network_input_shape,
        include_last_step=include_last_step,
        lmbda=settings.lmbda,
        loss_fn=loss_fn,
        use_label=settings.use_label,
        dice_loss_weight=settings.dice_loss_weight,
        loss_function_masking=settings.loss_function_masking,
    )


def _wrap_network(net: Any, settings: TrainingConfig, device: torch.device) -> Any:
    if device.type == "cuda" and len(settings.gpus) > 1:
        return torch.nn.DataParallel(
            net, device_ids=settings.gpus, output_device=settings.gpus[0]
        ).to(device)
    return net.to(device)


def _setup_cuda(settings: TrainingConfig) -> None:
    if not torch.cuda.is_available():
        return
    available = torch.cuda.device_count()
    invalid = [g for g in settings.gpus if g < 0 or g >= available]
    if invalid:
        raise ValueError(
            f"Requested GPU(s) {invalid} not available; "
            f"torch.cuda.device_count() reports {available} GPU(s)."
        )
    torch.cuda.set_device(settings.gpus[0])
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


def _run_validation(
    net: Any,
    net_par: Any,
    val_data_loaders_dict: Dict[str, Any],
    writer: SummaryWriter,
    iteration: int,
    settings: TrainingConfig,
    data_fields: FrozenSet[str],
    device: torch.device,
    use_wandb: bool,
    stage_prefix: str,
    current_epoch: int,
) -> None:
    if not val_data_loaders_dict:
        return

    ds_name, val_loader = random.choice(list(val_data_loaders_dict.items()))

    net_par.eval()
    net.eval()
    try:
        with torch.no_grad():
            val_batch = next(iter(val_loader))
            val_batch = {
                key: (value.to(device) if torch.is_tensor(value) else value)
                for key, value in val_batch.items()
            }
            if settings.roi_masking:
                _apply_roi_masking(val_batch)
            forward_kwargs = _build_forward_kwargs(
                val_batch,
                settings,
                data_fields,
                current_epoch=current_epoch,
            )
            val_loss = net(
                val_batch[PairKeys.IMAGE_A],
                val_batch[PairKeys.IMAGE_B],
                **forward_kwargs,
            )
            _log_loss_scalars_with_wandb(
                writer,
                f"{stage_prefix}/val/{ds_name}",
                loss_to_dict(val_loss),
                iteration,
                use_wandb=use_wandb,
            )

            has_seg = Fields.SEGMENTATION in data_fields
            has_mask = Fields.MASK in data_fields
            warped_seg = None
            if has_seg and settings.dice_loss_weight > 0.0 and hasattr(net, "warped_seg_A"):
                warped_seg = net.warped_seg_A

            add_eval_composite_panel(
                writer,
                iteration,
                moving=val_batch[PairKeys.IMAGE_A],
                fixed=val_batch[PairKeys.IMAGE_B],
                warped=net.warped_image_A,
                moving_seg=val_batch[PairKeys.SEGMENTATION_A] if has_seg else None,
                fixed_seg=val_batch[PairKeys.SEGMENTATION_B] if has_seg else None,
                warped_seg=warped_seg,
                moving_mask=val_batch[PairKeys.MASK_A] if has_mask else None,
                fixed_mask=val_batch[PairKeys.MASK_B] if has_mask else None,
                tag=f"{stage_prefix}/eval/{ds_name}",
                use_wandb=use_wandb,
            )
            net.clean()
    finally:
        net_par.train()


def _train_stage(
    net: Any,
    net_par: Any,
    optimizer: torch.optim.Optimizer,
    train_loader: Any,
    val_loaders: Dict[str, Any],
    settings: TrainingConfig,
    data_fields: FrozenSet[str],
    device: torch.device,
    writer: SummaryWriter,
    use_wandb: bool,
    stage_prefix: str,
    epochs: int,
    iteration: int,
    eval_period: int,
    save_period: int,
    output_dir: str,
) -> int:
    """Run one training stage; returns the next global iteration index."""
    for epoch in tqdm(range(epochs), desc=f"{stage_prefix} epochs"):
        for batch in train_loader:
            loss_dict = _train_one_batch(
                net,
                net_par,
                optimizer,
                batch,
                settings,
                data_fields,
                device,
                current_epoch=epoch,
            )
            _log_loss_scalars_with_wandb(
                writer,
                f"{stage_prefix}/train",
                loss_dict,
                iteration,
                use_wandb=use_wandb,
            )

            if iteration % DEFAULT_LOG_PERIOD == 0:
                loss_str = " | ".join([f"{k}={v:.4f}" for k, v in loss_dict.items()])
                logger.info(f"[{stage_prefix} epoch {epoch}, iter {iteration}] {loss_str}")

            iteration += 1

        is_last_epoch = epoch == epochs - 1
        is_periodic_save = epoch > 0 and epoch % save_period == 0
        if is_periodic_save and not is_last_epoch:
            _save_checkpoint(net, optimizer, output_dir, epoch, settings=settings)
            logger.info(f"[{stage_prefix}] Wrote checkpoint for epoch {epoch}.")

        if epoch % eval_period == 0:
            _run_validation(
                net,
                net_par,
                val_loaders,
                writer,
                iteration,
                settings,
                data_fields,
                device,
                use_wandb,
                stage_prefix,
                current_epoch=epoch,
            )

    return iteration


def train_trusted_two_stage(
    config: Dict[str, Any],
    train_loader: Any,
    val_loaders: Dict[str, Any],
    data_fields: FrozenSet[str],
    resume_from: str = "",
) -> Any:
    schema = FinetuningConfigSchema.from_dict(config)
    settings = schema.training
    exp_config = config[ConfigSections.EXPERIMENT]

    if settings.stage_epochs is None or len(settings.stage_epochs) != 2:
        raise ValueError("training.stage_epochs must contain exactly two positive integers")

    stage1_epochs, stage2_epochs = settings.stage_epochs

    use_wandb = exp_config.get(ExperimentKeys.USE_WANDB, False)
    wandb_project = exp_config.get(ExperimentKeys.WANDB_PROJECT, DEFAULT_TRAIN_WANDB_PROJECT)
    wandb_entity = exp_config.get(ExperimentKeys.WANDB_ENTITY)
    wandb_run_name = exp_config.get(ExperimentKeys.WANDB_RUN_NAME) or exp_config[ExperimentKeys.NAME]

    if use_wandb and wandb is None:
        raise ImportError(
            "use_wandb is true but wandb is not installed. "
            "Install with: pip install wandb. Or set use_wandb: false."
        )

    device = icon_config.device
    _setup_cuda(settings)

    loss_fn = _build_loss_fn(settings)

    # --- Stage 1 ---
    net = _build_registration_network(settings, loss_fn, include_last_step=False)
    if resume_from:
        if not os.path.exists(resume_from):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_from}")
        logger.info(f"Resuming stage 1 from {resume_from}")
        net.regis_net.load_state_dict(
            torch.load(resume_from, map_location="cpu", weights_only=True),
            strict=True,
        )

    net_par = _wrap_network(net, settings, device)
    optimizer = torch.optim.Adam(net_par.parameters(), lr=settings.learning_rate)
    net_par.train()

    os.makedirs(os.path.join(footsteps.output_dir, CHECKPOINT_DIR), exist_ok=True)
    writer = SummaryWriter(
        os.path.join(footsteps.output_dir, "logs", datetime.now().strftime("%Y%m%d-%H%M%S")),
        flush_secs=30,
    )

    wandb_initialized = False
    if use_wandb:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name,
            config={
                "experiment": dict(exp_config),
                "training": config.get(ConfigSections.TRAINING, {}),
                "datasets": config.get(ConfigSections.DATASETS, []),
            },
        )
        wandb_initialized = True

    logger.info(
        f"Scratch training: stage_epochs={settings.stage_epochs}, "
        f"similarity={settings.similarity}, dice_loss_weight={settings.dice_loss_weight}, "
        f"network_input_shape={settings.network_input_shape}."
    )

    iteration = 0

    try:
        iteration = _train_stage(
            net,
            net_par,
            optimizer,
            train_loader,
            val_loaders,
            settings,
            data_fields,
            device,
            writer,
            use_wandb,
            "stage1",
            stage1_epochs,
            iteration,
            settings.eval_period,
            settings.save_period,
            footsteps.output_dir,
        )

        step1_path = os.path.join(footsteps.output_dir, CHECKPOINT_DIR, STEP_1_FINAL)
        torch.save(net.regis_net.state_dict(), step1_path)
        logger.info(f"Stage 1 complete; saved {step1_path}")

        del net_par
        del optimizer

        # --- Stage 2 ---
        footsteps.output_dir_impl = footsteps.output_dir + SECOND_STEP_SUBDIR
        os.makedirs(os.path.join(footsteps.output_dir, CHECKPOINT_DIR), exist_ok=True)

        net_2 = _build_registration_network(settings, loss_fn, include_last_step=True)
        net_2.regis_net.netPhi.load_state_dict(
            torch.load(step1_path, map_location="cpu", weights_only=True),
            strict=True,
        )

        net_2_par = _wrap_network(net_2, settings, device)
        optimizer_2 = torch.optim.Adam(net_2_par.parameters(), lr=settings.learning_rate)
        net_2_par.train()

        iteration = _train_stage(
            net_2,
            net_2_par,
            optimizer_2,
            train_loader,
            val_loaders,
            settings,
            data_fields,
            device,
            writer,
            use_wandb,
            "stage2",
            stage2_epochs,
            iteration,
            settings.eval_period,
            settings.save_period,
            footsteps.output_dir,
        )

        step2_path = os.path.join(footsteps.output_dir, CHECKPOINT_DIR, STEP_2_FINAL)
        torch.save(net_2.regis_net.state_dict(), step2_path)
        _save_checkpoint(net_2, optimizer_2, footsteps.output_dir, "final", settings=settings)
        logger.info(f"Stage 2 complete; saved {step2_path}")

        return net_2
    finally:
        writer.close()
        if wandb_initialized:
            wandb.finish()


def main(argv: Optional[List[str]] = None) -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Two-stage from-scratch multigradicon training (Trusted CT-US)"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument(
        "--resume_from",
        type=str,
        default="",
        help="Optional stage-1 checkpoint to resume coarse training",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    validate_scratch_training_config(config)

    exp_config = config[ConfigSections.EXPERIMENT]
    use_wandb = exp_config.get(ExperimentKeys.USE_WANDB, False)
    if use_wandb and wandb is None:
        raise ImportError(
            "use_wandb is true but wandb is not installed. "
            "Install with: pip install wandb. Or set use_wandb: false."
        )

    seed = config.get(ConfigSections.TRAINING, {}).get(TrainingKeys.SEED)
    set_reproducibility_seed(seed)
    if seed is not None:
        logger.info(f"Reproducibility seed set: {seed}.")

    os.makedirs("results", exist_ok=True)
    _initialize_footsteps(run_name=exp_config[ExperimentKeys.NAME])

    loaders = create_scratch_data_loaders(args.config, config=config)
    config = loaders.config

    logger.info(
        f"Experiment '{exp_config[ExperimentKeys.NAME]}' ready: "
        f"{len(config[ConfigSections.DATASETS])} dataset(s), "
        f"auxiliary fields={sorted(loaders.data_fields) if loaders.data_fields else 'images only'}."
    )

    train_trusted_two_stage(
        config=config,
        train_loader=loaders.train_loader,
        val_loaders=loaders.val_loaders,
        data_fields=loaders.data_fields,
        resume_from=args.resume_from,
    )

    logger.info("Scratch training completed successfully.")


if __name__ == "__main__":
    main()
