import numpy as np
import torch

from matcha.dm_scene.cameras import GSCamera
from matcha.dm_utils.loss import ssim, l1_loss, l2_loss, cos_loss
from matcha.dm_utils.rendering import depth2normal_surfel, normal2curv


def depth_normal_consistency_loss(
    depth:torch.Tensor, 
    normals:torch.tensor,
    mask_vis:torch.Tensor, 
    camera:GSCamera, 
):
    """_summary_

    Args:
        depth (torch.Tensor): Has shape (1, height, width).
        normals (torch.Tensor): Has shape (3, height, width).
        mask_vis (torch.Tensor): Has shape (1, height, width).
        camera (GSCamera): _description_

    Returns:
        _type_: _description_
    """
    d2n = depth2normal_surfel(depth, mask_vis, camera)
    return cos_loss(normals, d2n, thrsh=np.pi*1/10000, weight=1)


def curvature_loss(
    normals:torch.Tensor, 
    mask_vis:torch.Tensor,
):
    """_summary_

    Args:
        normals (torch.Tensor): Has shape (3, height, width).
        mask_vis (torch.Tensor): Has shape (1, height, width).

    Returns:
        _type_: _description_
    """
    curv_n = normal2curv(normals, mask_vis)
    return l1_loss(curv_n, 0.)


def normal_prior_loss(
    normals:torch.Tensor,
    prior_normals:torch.Tensor,
):
    """_summary_

    Args:
        normals (torch.Tensor): Has shape (3, height, width).
        prior_normals (torch.Tensor): Has shape (3, height, width).

    Returns:
        _type_: _description_
    """
    height, width = normals.shape[1:]
    return (normals * prior_normals).sum() / (height * width)


def opacity_loss(opacity:torch.Tensor):
    """

    Args:
        opacity (torch.Tensor): Has shape (n_visible_gaussians, 1).

    Returns:
        _type_: _description_
    """
    return (torch.exp(- (opacity - 0.5).pow(2) / 0.05)).mean()


def gaussian_surfel_regularization(
    render_package:dict,
    camera:GSCamera,
    gaussian_opacities:torch.Tensor=None,
    prior_normals:torch.Tensor=None,
    depth_normal_consistency_weight:float=0.1,
    curvature_weight:float=0.005,
    opacity_weight:float=0.01,
    normal_prior_weight:float=0.04,
):  
    """_summary_

    Args:
        render_package (dict): Render package. Should contain the following keys:
            - 'opacity': Has shape (1, H, W).
            - 'depth': Has shape (H, W).
            - 'normal': Has shape (H, W, 3).
        camera (GSCamera): _description_
        gaussian_opacities (torch.Tensor, optional): Has shape (n_visible_gaussian, 1). Defaults to None.
        prior_normals (torch.Tensor, optional): Has shape (3, H, W). Defaults to None.
        depth_normal_consistency_weight (float, optional): Defaults to 0.1.
        curvature_weight (float, optional): Defaults to 0.005.
        opacity_weight (float, optional): Defaults to 0.01.
        normal_prior_weight (float, optional): Defaults to 0.04.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if gaussian_opacities is None and opacity_weight > 0:
        raise ValueError("gaussian_opacities must be provided if opacity_weight > 0.")
    if prior_normals is None and normal_prior_weight > 0:
        raise ValueError("prior_normals must be provided if normal_prior_weight > 0.")
    
    rendered_opac = render_package['opacity']  # Shape (1, H, W)
    mask_vis = rendered_opac > 1e-5  # Shape (1, H, W)
    rendered_depth = render_package['depth'][None]  # Shape (1, H, W)
    rendered_normals = render_package['normal'].permute(2, 0, 1) * mask_vis  # Shape (3, H, W)

    loss = 0.
    if depth_normal_consistency_weight > 0.:
        loss = loss + depth_normal_consistency_weight * depth_normal_consistency_loss(
            depth=rendered_depth, 
            normals=rendered_normals,
            mask_vis=mask_vis, 
            camera=camera,
        )

    if curvature_weight > 0.:
        loss = loss + curvature_loss(
            normals=rendered_normals, 
            mask_vis=mask_vis,
        )
        
    if opacity_weight > 0.:
        loss = loss + opacity_weight * opacity_loss(gaussian_opacities)
        
    if normal_prior_weight > 0.:
        loss = loss + normal_prior_weight * normal_prior_loss(
            normals=rendered_normals,
            prior_normals=prior_normals.permute(2, 0, 1),
        )
        
    return loss