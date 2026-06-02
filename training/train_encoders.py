import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import unigradicon
from unigradicon.encoder_losses import autoencoder_reconstruction_loss
from unigradicon.encoders import build_encoder_from_arch
from unigradicon.finetuning.dataset import ImagePreprocessor, ImageReader


@dataclass
class EncoderTrainConfig:
    name: str
    json_file: str
    modalities: Sequence[str]
    checkpoint_name: str
    alpha_ncc: float = 1.0


class JsonAutoEncoderDataset(Dataset):
    def __init__(
        self,
        json_file: str,
        modalities: Sequence[str],
        input_shape: Tuple[int, int, int],
        ct_window: Tuple[float, float],
        quantile_range: Tuple[float, float],
    ):
        self.json_file = os.path.abspath(json_file)
        self.modalities = {m.lower() for m in modalities}
        self.input_shape = input_shape
        self.ct_window = ct_window
        self.quantile_range = quantile_range

        with open(self.json_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if "data" not in payload or not isinstance(payload["data"], list):
            raise ValueError(f"JSON file {self.json_file} must contain a top-level 'data' list.")

        base_dir = os.path.dirname(self.json_file)
        self.items: List[Dict[str, str]] = []
        modality_map: Dict[str, bool] = {}
        for idx, entry in enumerate(payload["data"]):
            if "image" not in entry:
                continue
            modality = str(entry.get("modality", "")).lower()
            if modality not in self.modalities:
                continue
            image_path = entry["image"]
            image_path = image_path if os.path.isabs(image_path) else os.path.join(base_dir, image_path)
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image path does not exist (entry {idx}): {image_path}")
            self.items.append({"image": image_path, "modality": modality})
            modality_map[image_path] = modality == "ct"

        if not self.items:
            raise ValueError(
                f"No matching entries found in {self.json_file} for modalities {sorted(self.modalities)}."
            )

        self.preprocessor = ImagePreprocessor(
            reader=ImageReader(),
            input_shape=self.input_shape,
            is_ct=False,
            ct_window=self.ct_window,
            quantile_range=self.quantile_range,
            modality_map=modality_map,
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image_path = self.items[idx]["image"]
        # Returns [C, D, H, W] with C=1.
        return self.preprocessor.preprocess_image(image_path)


def train_single_encoder(
    cfg: EncoderTrainConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> str:
    dataset = JsonAutoEncoderDataset(
        json_file=cfg.json_file,
        modalities=cfg.modalities,
        input_shape=tuple(args.input_shape),
        ct_window=tuple(args.ct_window),
        quantile_range=tuple(args.quantile_range),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    arch_kwargs = {
        "in_channels": args.in_channels,
        "base_channels": args.base_channels,
        "feature_dim": args.feature_dim,
    }
    model = build_encoder_from_arch(args.arch, arch_kwargs).to(device)
    lncc_fn = unigradicon.make_sim("lncc", sigma=args.lncc_sigma)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    model.train()
    for epoch in range(args.epochs):
        running_total = 0.0
        running_recon = 0.0
        running_lncc = 0.0
        for batch in tqdm(loader, desc=f"{cfg.name} epoch {epoch + 1}/{args.epochs}"):
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, _ = model(batch)
            if reconstruction.shape[2:] != batch.shape[2:]:
                reconstruction = F.interpolate(
                    reconstruction,
                    size=batch.shape[2:],
                    mode="trilinear",
                    align_corners=True,
                )

            loss, recon_loss, lncc_loss = autoencoder_reconstruction_loss(
                batch,
                reconstruction,
                alpha_ncc=cfg.alpha_ncc,
                lncc_fn=lncc_fn,
                loss_type=args.loss,
                return_components=True,
            )

            loss.backward()
            optimizer.step()
            running_total += float(loss.detach().cpu())
            running_recon += float(recon_loss.detach().cpu())
            running_lncc += float(lncc_loss.detach().cpu())

        n_batches = max(len(loader), 1)
        print(
            f"[{cfg.name}] epoch={epoch + 1}/{args.epochs} "
            f"loss={running_total / n_batches:.6f} "
            f"recon={running_recon / n_batches:.6f} "
            f"lncc={running_lncc / n_batches:.6f}"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, cfg.checkpoint_name)
    checkpoint = {
        "arch": args.arch,
        "arch_kwargs": arch_kwargs,
        "state_dict": model.state_dict(),
        "feature_level": args.feature_level,
        "metadata": {
            "encoder_name": cfg.name,
            "json_file": os.path.abspath(cfg.json_file),
            "modalities": list(cfg.modalities),
            "input_shape": list(args.input_shape),
            "ct_window": list(args.ct_window),
            "quantile_range": list(args.quantile_range),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "loss": args.loss,
            "alpha_ncc": cfg.alpha_ncc,
            "lncc_sigma": args.lncc_sigma,
        },
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain preop/us AutoEncoder checkpoints for contrastive registration."
    )
    parser.add_argument("--preop_json", type=str, required=True, help="JSON file for preop encoder data.")
    parser.add_argument(
        "--us_json",
        type=str,
        default=None,
        help="JSON file for US encoder data. Defaults to --preop_json when omitted.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write checkpoints.")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--loss", type=str, choices=["l1", "mse"], default="l1")

    parser.add_argument("--arch", type=str, default="AutoEncoder3D")
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--feature_dim", type=int, default=128)
    parser.add_argument("--feature_level", type=int, default=-2)

    parser.add_argument("--alpha_ncc_preop", type=float, default=1.0)
    parser.add_argument("--alpha_ncc_us", type=float, default=0.5)
    parser.add_argument("--lncc_sigma", type=int, default=5)

    parser.add_argument("--input_shape", type=int, nargs=3, default=[175, 175, 175])
    parser.add_argument("--ct_window", type=float, nargs=2, default=[-1000.0, 1000.0])
    parser.add_argument("--quantile_range", type=float, nargs=2, default=[0.0, 0.99])

    parser.add_argument(
        "--preop_modalities",
        type=str,
        default="ct,mri,mr,preop",
        help="Comma-separated modalities for preop encoder training.",
    )
    parser.add_argument(
        "--us_modalities",
        type=str,
        default="us,ultrasound",
        help="Comma-separated modalities for US encoder training.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (e.g., cuda, cuda:0, cpu).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.us_json = args.us_json or args.preop_json

    device = torch.device(args.device)
    preop_cfg = EncoderTrainConfig(
        name="preop_encoder",
        json_file=args.preop_json,
        modalities=[m.strip().lower() for m in args.preop_modalities.split(",") if m.strip()],
        checkpoint_name="preop_encoder.ckpt",
        alpha_ncc=args.alpha_ncc_preop,
    )
    us_cfg = EncoderTrainConfig(
        name="us_encoder",
        json_file=args.us_json,
        modalities=[m.strip().lower() for m in args.us_modalities.split(",") if m.strip()],
        checkpoint_name="us_encoder.ckpt",
        alpha_ncc=args.alpha_ncc_us,
    )

    preop_path = train_single_encoder(preop_cfg, args, device)
    us_path = train_single_encoder(us_cfg, args, device)

    print("Saved checkpoints:")
    print(f"  preop: {preop_path}")
    print(f"  us:    {us_path}")


if __name__ == "__main__":
    main()
