import torch


def depth_logl1_loss(
    rendered_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    weight: torch.Tensor = None,
):
    """_summary_

    Args:
        rendered_depth (torch.Tensor): Predicted depth map. Shape: (H, W).
        gt_depth (torch.Tensor): Ground truth depth map. Shape: (H, W).
        weight (torch.Tensor): Weight map. Shape: (H, W).

    Returns:
        torch.Tensor: L1 loss between the predicted and ground truth depth maps.
    """
    loss = torch.log(1. + (rendered_depth - gt_depth).abs())
    if weight is not None:
        loss = loss * weight
    return loss.mean()


