from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn.functional as F


def autoencoder_reconstruction_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    alpha_ncc: float = 1.0,
    lncc_sigma: int = 5,
    lncc_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    loss_type: str = "l1",
    return_components: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """L1 (or MSE) plus weighted local NCC for autoencoder pretraining."""
    if loss_type == "mse":
        recon_loss = F.mse_loss(x_hat, x)
    else:
        recon_loss = F.l1_loss(x_hat, x)

    if alpha_ncc == 0:
        lncc_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        total = recon_loss
    else:
        if lncc_fn is None:
            import unigradicon

            lncc_fn = unigradicon.make_sim("lncc", sigma=lncc_sigma)
        lncc_loss = lncc_fn(x_hat, x)
        total = recon_loss + alpha_ncc * lncc_loss

    if return_components:
        return total, recon_loss, lncc_loss
    return total
