import torch
import math
from matcha.dm_scene.cameras import GSCamera


def depths_to_points(view, depthmap):
    c2w = (view.world_view_transform.T).inverse()
    W, H = view.image_width, view.image_height
    fx = W / (2 * math.tan(view.FoVx / 2.))
    fy = H / (2 * math.tan(view.FoVy / 2.))
    intrins = torch.tensor(
        [[fx, 0., W/2.],
        [0., fy, H/2.],
        [0., 0., 1.0]]
    ).float().cuda()
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float() + 0.5, torch.arange(H, device='cuda').float() + 0.5, indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
    rays_o = c2w[:3,3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points


def depth_to_normal(view, depth):
    """
        view: view camera
        depth: depthmap Shape (1, H, W) or (H, W) or (H, W, 1)

        return: normal map Shape (H, W, 3)
        return: points Shape (H, W, 3)
    """
    _depth = depth.squeeze().unsqueeze(0)  # Shape (1, H, W)
    H, W = depth.shape[1:]
    points = depths_to_points(view, _depth).reshape(H, W, 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output, points


def depth_normal_consistency_loss_gof(
    depth:torch.Tensor, 
    normal:torch.Tensor,
    gs_camera:GSCamera,
    weight:torch.Tensor=None,
):
    """

    Args:
        depth (torch.Tensor): Has shape (H, W) or (H, W, 1)
        normal (torch.Tensor): Has shape (H, W, 3).
        gs_camera (GSCamera): _description_
        weight (torch.Tensor, optional): Has shape (H, W). Defaults to None.
    """
    
    H, W = depth.shape[:2]
    
    # Convert depth to normal
    d_normal, _ = depth_to_normal(gs_camera, depth.view(1, H, W))
    d_normal = d_normal.permute(2, 0, 1)
    
    # Transform normals from view space to world space
    c2w = (gs_camera.world_view_transform.T).inverse()
    normal2 = c2w[:3, :3] @ normal.permute(2, 0, 1).reshape(3, -1)
    normal_world = normal2.reshape(3, H, W)  # (3, h, w)
    
    # Compute normal consistency loss
    normal_error = 1 - (normal_world * d_normal).sum(dim=0)  # (h, w)
    
    if weight is not None:
        normal_error = normal_error * weight  # (h, w)
        
    depth_normal_loss = normal_error.mean()
    
    return depth_normal_loss


def depth_distortion_loss_gof(distortion):
    return distortion.mean()


def gaussian_gof_regularization(
    render_package:dict,
    camera:GSCamera,
    depth_normal_consistency_weight:float=0.05,
    depth_distortion_loss_weight:float=100.,
):
    rendered_depth = render_package['depth']  # Shape (H, W)
    rendered_normal = torch.nn.functional.normalize(render_package['normal'], p=2, dim=-1)  # Shape (H, W, 3)
    distortion = render_package['distortion']  # Shape (1, H, W)
    device = rendered_depth.device
    
    loss = torch.tensor(0., device=device)
    
    if depth_normal_consistency_weight > 0.:
        loss = loss + depth_normal_consistency_weight * depth_normal_consistency_loss_gof(
            rendered_depth, rendered_normal, camera
        )
        
    if depth_distortion_loss_weight > 0.:
        loss = loss + depth_distortion_loss_weight * depth_distortion_loss_gof(distortion)
        
    return loss