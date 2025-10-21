import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy
from .depth import depth_l1_loss
from matcha.pointmap.depthanythingv2 import get_points_depth_in_depthmap


def sfm_points_loss(
    camera_sfm_pts:torch.Tensor, 
    depth:torch.Tensor, 
    p3d_camera, 
    depth_loss=depth_l1_loss
):
    """Loss that encourages the manifold to be close to the sfm points visible in the camera.

    Args:
        camera_sfm_pts (torch.Tensor): Has shape (n_sfm_points, 3).
        depth (torch.Tensor): Has shape (h, w).
        p3d_camera (_type_): _
        depth_loss (_type_, optional): Defaults to depth_l1_loss.

    Returns:
        _type_: _description_
    """
    sfm_true_depth = p3d_camera.get_world_to_view_transform().transform_points(camera_sfm_pts)[..., 2]
    sfm_rendered_depth, fov_mask = get_points_depth_in_depthmap(pts=camera_sfm_pts, depthmap=depth, p3d_camera=p3d_camera)
    
    return depth_loss(sfm_rendered_depth, sfm_true_depth)


def opacity_entropy_loss(
    opacities:torch.Tensor,
):
    """Loss that encourages the opacities to be either 0 or 1, relying on the Shannon entropy of the opacity distribution.

    Args:
        opacities (torch.Tensor): Tensor with shape (..., 1).

    Returns:
        torch.Tensor: Loss value.
    """
    # loss = torch.mean(torch.log(opacities) + torch.log(1 - opacities))
    loss = - (
        opacities * torch.log(opacities + 1e-10) 
        + (1. - opacities) * torch.log(1. - opacities + 1e-10)
    ).mean()
    return loss


def intraconnectivity_loss(
    normal:torch.Tensor,
    opacity_maps:torch.Tensor,
):
    """Loss that encourages neighboring points in a pointmap to have similar opacities if they have similar normals.
    Points with similar normals are likely to be part of the same surface, and should have similar opacities.
    This loss is supposed to encourage UV maps to be smooth and continuous.
    
    Args:
        normal (torch.Tensor): Tensor with shape (n_charts, h_pointmap, w_pointmap, 3) containing reference normal vectors for each point in a pointmap.
        opacity_maps (torch.Tensor): Tensor with shape (n_charts, h_pointmap, w_pointmap, 1) containing the opacity of each point in a pointmap.
        
    Returns:
        torch.Tensor: Loss value.
    """
    opacity_grad_x = torch.abs(opacity_maps[..., 1:, 1:, :] - opacity_maps[..., :-1, 1:, :])[..., 0]
    opacity_grad_y = torch.abs(opacity_maps[..., 1:, 1:, :] - opacity_maps[..., 1:, :-1, :])[..., 0]
    
    normal_gradient_x = (normal[..., 1:, 1:, :] - normal[..., :-1, 1:, :]).norm(dim=-1)
    normal_gradient_y = (normal[..., 1:, 1:, :] - normal[..., 1:, :-1, :]).norm(dim=-1)

    if False:  # Combine directional gradients then compute penalty
        opacity_grad = torch.sqrt(opacity_grad_x**2 + opacity_grad_y**2)
        normal_gradient = torch.sqrt(normal_gradient_x**2 + normal_gradient_y**2)
        penalty = torch.exp(-normal_gradient) * opacity_grad
    else:  # Compute penalty for each direction then average (might be better)
        penalty_x = torch.exp(-normal_gradient_x) * opacity_grad_x
        penalty_y = torch.exp(-normal_gradient_y) * opacity_grad_y
        penalty = (penalty_x + penalty_y) / 2.
    
    return penalty.mean()


def _old_intraconnectivity_loss(
    faces_idx:torch.Tensor,
    face_neighbors:torch.Tensor,
    opacities:torch.Tensor,
    method='bce',  # 'bce', 'l1', 'l2'
):
    """_summary_

    Args:
        faces_idx (torch.Tensor): Tensor of shape (n_faces_with_neighbors, ) containing the indices of some faces.
        face_neighbors (torch.Tensor): Tensor of shape (n_faces_with_neighbors, 3) containing the indices of the neighboring faces of for some faces.
        opacities (torch.Tensor): Tensor of shape (n_all_faces, n_gaussians, 1) containing the opacity of each Gaussian in each face.
    """
    n_faces_with_neighbors = len(faces_idx)
    
    # Compute average opacity of a face by summing the opacity of all Gaussians in the face
    all_face_opacities = opacities.sum(dim=1)  # Shape: (n_all_faces, 1)
    face_opacities = all_face_opacities[faces_idx]  # Shape: (n_faces_with_neighbors, 1)
    neighbor_faces_opacities = all_face_opacities[face_neighbors]  # Shape: (n_faces_with_neighbors, 3, 1)
    
    if method == 'bce':
        loss = binary_cross_entropy(input=neighbor_faces_opacities, target=face_opacities[:, None].repeat(1, 3, 1))
    elif method == 'l1':
        loss = (face_opacities[:, None] - neighbor_faces_opacities).abs().mean()
    elif method == 'l2':
        loss = (face_opacities[:, None] - neighbor_faces_opacities).pow(2).mean()
    else:
        raise ValueError(f"Invalid method: {method}")
    
    return loss


def interconnectivity_loss(
    verts:torch.Tensor,
    manifold_idx:torch.Tensor,
):
    # TODO
    raise NotImplementedError("Interconnectivity loss is not implemented yet.")