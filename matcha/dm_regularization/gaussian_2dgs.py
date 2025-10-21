import numpy as np
import torch

from matcha.dm_scene.cameras import GSCamera
from matcha.dm_utils.loss import ssim, l1_loss, l2_loss, cos_loss
from matcha.dm_utils.rendering import depth2normal_2dgs


def depth_normal_consistency_loss(
    depth:torch.Tensor, 
    normal:torch.Tensor,
    opacity:torch.Tensor,
    camera:GSCamera,
    weight:torch.Tensor=None,
    use_l1:bool=False,
):
    """_summary_

    Args:
        depth (torch.Tensor): Has shape (1, height, width).
        normal (torch.tensor): Has shape (3, height, width). Should be in view space.
        opacity (torch.Tensor): Has shape (1, height, width).
        camera (GSCamera): _description_
        weight (torch.Tensor, optional): Has shape (height, width). Defaults to None.
    Returns:
        _type_: _description_
    """
    
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    normal_from_depth = depth2normal_2dgs(camera, depth).permute(2, 0, 1) * (opacity).detach()  # (3, h, w)
    
    # Transform normals from view space to world space (with COLMAP's convention, and not PyTorch3D's!)
    normal_world = (normal.permute(1, 2, 0) @ (camera.world_view_transform[:3,:3].T)).permute(2, 0, 1)  # (3, h, w)
    
    normal_error = (1 - (normal_world * normal_from_depth).sum(dim=0))  # (h, w)
    
    if use_l1:
        normal_error = normal_error + (normal_world - normal_from_depth).abs().sum(dim=0)
    
    if weight is not None:
        normal_error = normal_error * weight  # (h, w)
    
    return normal_error.mean()


def depth_distortion_loss(distortion):
    return distortion.mean()


def gaussian_2dgs_regularization(
    render_package:dict,
    camera:GSCamera,
    depth_normal_consistency_weight:float=0.05,
    depth_distortion_loss_weight:float=100.,
    use_median_depth:bool=False,
):
    rendered_depth = render_package['depth'][None] if not use_median_depth else render_package['median_depth'][None]
    rendered_normal = render_package['normal'].permute(2,0,1)
    rendered_opacity = render_package['opacity']
    distortion = render_package['distortion']
    device = rendered_depth.device
    
    loss = torch.tensor(0., device=device)
    
    if depth_normal_consistency_weight > 0.:
        loss = loss + depth_normal_consistency_weight * depth_normal_consistency_loss(
            rendered_depth, rendered_normal, rendered_opacity, camera
        )
        
    if depth_distortion_loss_weight > 0.:
        loss = loss + depth_distortion_loss_weight * depth_distortion_loss(distortion)
        
    return loss