import torch


def depth_gradient_l1_loss(
    rendered_depth: torch.Tensor,
    gt_depth: torch.Tensor,
):
    """

    Args:
        rendered_depth (torch.Tensor): Predicted depth map. Shape: (H, W).
        gt_depth (torch.Tensor): Ground truth depth map. Shape: (H, W).
        
    Returns:
        torch.Tensor: L1 loss between the image-gradients of the predicted and ground truth depth maps.
    """
    assert rendered_depth.dim() == 2
    assert rendered_depth.shape == gt_depth.shape
    
    x_grad_rendered = rendered_depth[1:, :] - rendered_depth[:-1, :]
    y_grad_rendered = rendered_depth[:, 1:] - rendered_depth[:, :-1]
    
    x_grad_gt = gt_depth[1:, :] - gt_depth[:-1, :]
    y_grad_gt = gt_depth[:, 1:] - gt_depth[:, :-1]
    
    return (x_grad_rendered - x_grad_gt).abs().mean() + (y_grad_rendered - y_grad_gt).abs().mean()


def depth_l1_loss(
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
    loss = (rendered_depth - gt_depth).abs()
    if weight is not None:
        loss = loss * weight
    return loss.mean()


def disp_l1_loss(
    rendered_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    weight: torch.Tensor = None,
):
    """_summary_

    Args:
        rendered_depth (torch.Tensor): Predicted disparity map. Shape: (H, W).
        gt_depth (torch.Tensor): Ground truth disparity map. Shape: (H, W).
        weight (torch.Tensor): Weight map. Shape: (H, W).

    Returns:
        torch.Tensor: L1 loss between the predicted and ground truth disparity maps.
    """
    rendered_disp = 1. / (rendered_depth + 1e-10)
    gt_disp = 1. / (gt_depth + 1e-10)
    loss = (rendered_disp - gt_disp).abs()
    if weight is not None:
        loss = loss * weight
    return loss.mean()


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


def depth_l1_invariant_loss(
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
        torch.Tensor: L1 invariant loss between the predicted and ground truth depth maps.
    """
    rendered_scale = (rendered_depth - rendered_depth.median()).abs().mean()
    gt_scale = (gt_depth - gt_depth.median()).abs().mean()

    reduced_rendered_depth = (rendered_depth - rendered_depth.median()) / rendered_scale
    reduced_gt_depth = (gt_depth - gt_depth.median()) / gt_scale
    
    loss = (reduced_rendered_depth - reduced_gt_depth).abs()
    if weight is not None:
        loss = loss * weight
    return loss.mean()


def disp_l1_invariant_loss(
    rendered_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    weight: torch.Tensor = None,
):
    """_summary_

    Args:
        rendered_depth (torch.Tensor): Predicted disparity map. Shape: (H, W).
        gt_depth (torch.Tensor): Ground truth disparity map. Shape: (H, W).
        weight (torch.Tensor): Weight map. Shape: (H, W).

    Returns:
        torch.Tensor: L1 invariant loss between the predicted and ground truth disparity maps.
    """
    rendered_disp = 1. / (rendered_depth + 1e-10)
    gt_disp = 1. / (gt_depth + 1e-10)
    
    return depth_l1_invariant_loss(rendered_disp, gt_disp, weight)


def compute_depth_order_loss(
    depth:torch.Tensor, 
    prior_depth:torch.Tensor, 
    scene_extent:float=1., 
    max_pixel_shift_ratio:float=0.05,
    normalize_loss:bool=True,
    log_space:bool=False,
    log_scale:float=20.,
    reduction:str="mean",
    debug:bool=False,
):
    """Compute a loss encouraging pixels in 'depth' to have the same relative depth order as in 'prior_depth'.
    This loss does not require prior depth maps to be multi-view consistent nor to have accurate relative scale.

    Args:
        depth (torch.Tensor): A tensor of shape (H, W), (H, W, 1) or (1, H, W) containing the depth values.
        prior_depth (torch.Tensor): A tensor of shape (H, W), (H, W, 1) or (1, H, W) containing the prior depth values.
        scene_extent (float): The extent of the scene used to normalize the loss and make the loss invariant to the scene scale.
        max_pixel_shift_ratio (float, optional): The maximum pixel shift ratio. Defaults to 0.05, i.e. 5% of the image size.
        normalize_loss (bool, optional): Whether to normalize the loss. Defaults to True.
        reduction (str, optional): The reduction to apply to the loss. Can be "mean", "sum" or "none". Defaults to "mean".
    
    Returns:
        torch.Tensor: A scalar tensor.
            If reduction is "none", returns a tensor with same shape as depth containing the pixel-wise depth order loss.
    """
    height, width = depth.squeeze().shape
    pixel_coords = torch.stack(torch.meshgrid(
        torch.linspace(0, height - 1, height, dtype=torch.long, device=depth.device),
        torch.linspace(0, width - 1, width, dtype=torch.long, device=depth.device),
        indexing='ij'
    ), dim=-1).view(-1, 2)

    # Get random pixel shifts
    # TODO: Change the sampling so that shifts of (0, 0) are not possible
    max_pixel_shift = round(max_pixel_shift_ratio * max(height, width))
    pixel_shifts = torch.randint(-max_pixel_shift, max_pixel_shift + 1, pixel_coords.shape, device=depth.device)

    # Apply pixel shifts to pixel coordinates and clamp to image boundaries
    shifted_pixel_coords = (pixel_coords + pixel_shifts).clamp(
        min=torch.tensor([0, 0], device=depth.device), 
        max=torch.tensor([height - 1, width - 1], device=depth.device)
    )

    # Get depth values at shifted pixel coordinates
    shifted_depth = depth.squeeze()[
        shifted_pixel_coords[:, 0], 
        shifted_pixel_coords[:, 1]
    ].reshape(depth.shape)
    shifted_prior_depth = prior_depth.squeeze()[
        shifted_pixel_coords[:, 0], 
        shifted_pixel_coords[:, 1]
    ].reshape(depth.shape)

    # Compute pixel-wise depth order loss
    diff = (depth - shifted_depth) / scene_extent
    prior_diff = (prior_depth - shifted_prior_depth) / scene_extent
    if normalize_loss:
        prior_diff = prior_diff / prior_diff.detach().abs().clamp(min=1e-8)
    depth_order_loss = - (diff * prior_diff).clamp(max=0)
    if log_space:
        depth_order_loss = torch.log(1. + log_scale * depth_order_loss)
    
    # Reduce the loss
    if reduction == "mean":
        depth_order_loss = depth_order_loss.mean()
    elif reduction == "sum":
        depth_order_loss = depth_order_loss.sum()
    elif reduction == "none":
        pass
    else:
        raise ValueError(f"Invalid reduction: {reduction}")
    
    if debug:
        return {
            "depth_order_loss": depth_order_loss,
            "diff": diff,
            "prior_diff": prior_diff,
            "shifted_depth": shifted_depth,
            "shifted_prior_depth": shifted_prior_depth,
        }
    return depth_order_loss
