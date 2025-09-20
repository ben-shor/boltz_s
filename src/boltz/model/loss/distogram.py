import torch
from torch import Tensor


def distogram_loss(
    output: dict[str, Tensor],
    feats: dict[str, Tensor],
) -> tuple[Tensor, Tensor]:
    """Compute the  distogram loss.

    Parameters
    ----------
    output : Dict[str, Tensor]
        Output of the model
    feats : Dict[str, Tensor]
        Input features

    Returns
    -------
    Tensor
        The globally averaged loss.
    Tensor
        Per example loss.

    """
    # Get predicted distograms
    pred = output["pdistogram"]

    # Compute target distogram
    target = feats["disto_target"]

    # Combine target mask and padding mask
    mask = feats["token_disto_mask"]
    mask = mask[:, None, :] * mask[:, :, None]
    mask = mask * (1 - torch.eye(mask.shape[1])[None]).to(pred)

    # Compute the distogram loss
    errors = -1 * torch.sum(
        target * torch.nn.functional.log_softmax(pred, dim=-1),
        dim=-1,
    )
    denom = 1e-5 + torch.sum(mask, dim=(-1, -2))
    mean = errors * mask
    mean = torch.sum(mean, dim=-1)
    mean = mean / denom[..., None]
    batch_loss = torch.sum(mean, dim=-1)
    global_loss = torch.mean(batch_loss)
    return global_loss, batch_loss


def distogram_teacher_loss_kl(
    output: dict[str, Tensor],
    teacher_output: dict[str, Tensor],
    feats: dict[str, Tensor],
) -> tuple[Tensor, Tensor]:
    # Get predicted distograms
    pred = output["pdistogram"]

    # Compute target distogram
    target = teacher_output["pdistogram"]

    # Combine target mask and padding mask
    mask = feats["token_disto_mask"]
    mask = mask[:, None, :] * mask[:, :, None]
    mask = mask * (1 - torch.eye(mask.shape[1])[None]).to(pred)

    # Compute the distogram loss
    # errors = -1 * torch.sum(
    # #    target * torch.nn.functional.log_softmax(pred, dim=-1),
    #     -(torch.softmax(target, dim=-1) * torch.nn.functional.log_softmax(pred, dim=-1),
    #     dim=-1,
    # )
    errors = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(pred, dim=-1),
        torch.nn.functional.log_softmax(target, dim=-1),
        log_target=True,
        reduction="none"
    ).sum(dim=-1)

    denom = 1e-5 + torch.sum(mask, dim=(-1, -2))
    mean = errors * mask
    mean = torch.sum(mean, dim=-1)
    mean = mean / denom[..., None]
    batch_loss = torch.sum(mean, dim=-1)
    global_loss = torch.mean(batch_loss)
    return global_loss, batch_loss


def distogram_teacher_loss_ce(
    output: dict[str, Tensor],
    teacher_output: dict[str, Tensor],
    feats: dict[str, Tensor],
) -> tuple[Tensor, Tensor]:
    # Get predicted distograms
    pred = output["pdistogram"]

    # Compute target distogram
    target = teacher_output["pdistogram"]

    # Combine target mask and padding mask
    mask = feats["token_disto_mask"]
    mask = mask[:, None, :] * mask[:, :, None]
    mask = mask * (1 - torch.eye(mask.shape[1])[None]).to(pred)

    # Compute the distogram loss
    errors = -(torch.softmax(target, dim=-1) * torch.nn.functional.log_softmax(pred, dim=-1)).sum(dim=-1)

    denom = 1e-5 + torch.sum(mask, dim=(-1, -2))
    mean = errors * mask
    mean = torch.sum(mean, dim=-1)
    mean = mean / denom[..., None]
    batch_loss = torch.sum(mean, dim=-1)
    global_loss = torch.mean(batch_loss)
    return global_loss, batch_loss