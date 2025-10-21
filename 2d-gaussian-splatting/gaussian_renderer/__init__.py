#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.point_utils import depth_to_normal

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        # pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # currently don't support normal consistency loss if use precomputed covariance
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, far-near, near],
            [0, 0, 0, 1]]).float().cuda().T
        world2pix =  viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9) # column major
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    pipe.convert_SHs_python = False
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    
    rendered_image, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp
    )
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rets =  {"render": rendered_image,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
    }


    # additional regularizations
    render_alpha = allmap[1:2]

    # get normal map
    # transform normal from view space to world space
    render_normal = allmap[2:5]
    render_normal_cam = render_normal.clone()
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    
    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    
    # get depth distortion map
    render_dist = allmap[6:7]

    # psedo surface attributes
    # surf depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1; 
    # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
    surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
    
    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * (render_alpha).detach()

    # get surf normal in camera space
    surf_normal_world = surf_normal.clone()
    surf_normal_cam = (surf_normal_world.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3])).permute(2,0,1)


    rets.update({
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_normal_cam': render_normal_cam,
            'rend_dist': render_dist,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal,
            'surf_normal_cam': surf_normal_cam,
            'rend_depth': render_depth_expected,
    })

    return rets


def render_gslist(viewpoint_camera, pc_list, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render a scene with multiple Gaussian models.
    
    Args:
        viewpoint_camera: Camera viewpoint
        pc_list: List of GaussianModel objects
        pipe: Rendering pipeline
        bg_color: Background color tensor (must be on GPU)
        scaling_modifier: Scaling modifier coefficient
        override_color: Optional override color
        
    Returns:
        Dictionary containing rendering results
    """
    # Ensure input is a list
    if not isinstance(pc_list, list):
        raise ValueError("pc_list must be a list of GaussianModel objects")
        
    # Use active_sh_degree from the first model
    active_sh_degree = pc_list[0].active_sh_degree
    
    # Combine parameters from all models
    means3D_list = []
    opacity_list = []
    scales_list = []
    rotations_list = []
    features_list = []
    
    for pc in pc_list:
        means3D_list.append(pc.get_xyz)
        opacity_list.append(pc.get_opacity)
        scales_list.append(pc.get_scaling)
        rotations_list.append(pc.get_rotation)
        features_list.append(pc.get_features)
    
    # Concatenate parameters
    means3D = torch.cat(means3D_list, dim=0)
    opacity = torch.cat(opacity_list, dim=0)
    scales = torch.cat(scales_list, dim=0)
    rotations = torch.cat(rotations_list, dim=0)
    features = torch.cat(features_list, dim=0)
    
    # Create zero tensor (same as original render function)
    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points
    
    # If precomputed 3D covariance is provided, use it
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # Currently doesn't support normal consistency loss when using precomputed covariance
        # Special handling needed as we need to compute covariance for each point cloud
        cov3D_list = []
        for pc in pc_list:
            splat2world = pc.get_covariance(scaling_modifier)
            W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
            near, far = viewpoint_camera.znear, viewpoint_camera.zfar
            ndc2pix = torch.tensor([
                [W / 2, 0, 0, (W-1) / 2],
                [0, H / 2, 0, (H-1) / 2],
                [0, 0, far-near, near],
                [0, 0, 0, 1]]).float().cuda().T
            world2pix = viewpoint_camera.full_proj_transform @ ndc2pix
            pc_cov3D = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9)
            cov3D_list.append(pc_cov3D)
        cov3D_precomp = torch.cat(cov3D_list, dim=0)
    else:
        # Not using precomputed covariance, directly use concatenated scales and rotations
        scales = scales
        rotations = rotations

    # Handle colors or SH coefficients
    pipe.convert_SHs_python = False
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            # Calculate colors for each point cloud separately
            colors_list = []
            for i, pc in enumerate(pc_list):
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_list.append(torch.clamp_min(sh2rgb + 0.5, 0.0))
            colors_precomp = torch.cat(colors_list, dim=0)
        else:
            # Directly use the concatenated features
            shs = features
    else:
        # If override color is provided, ensure its size matches the concatenated point cloud
        if isinstance(override_color, list):
            colors_precomp = torch.cat(override_color, dim=0)
        else:
            # Assume override color is a single color, needs to be expanded to all points
            colors_precomp = override_color
    
    # Call rasterizer for rendering
    rendered_image, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp
    )
    
    # Create return dictionary
    rets = {
        "render": rendered_image,
        "viewspace_points": means2D,
        "visibility_filter": radii > 0,
        "radii": radii,
    }

    # Handle additional regularizations
    render_alpha = allmap[1:2]

    # Get normal map
    # Transform normal from view space to world space
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    
    # Get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # Get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    
    # Get depth distortion map
    render_dist = allmap[6:7]

    # Pseudo surface attributes
    # Surface depth can be either median or expected by setting depth_ratio to 1 or 0
    # For bounded scenes, use median depth (depth_ratio = 1)
    # For unbounded scenes, use expected depth (depth_ratio = 0) to reduce disk aliasing
    surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
    
    # Assume that depth points form the 'surface' and generate pseudo surface normal for regularizations
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    # Remember to multiply with accum_alpha since render_normal is unnormalized
    surf_normal = surf_normal * (render_alpha).detach()

    # Update return dictionary
    rets.update({
        'rend_alpha': render_alpha,
        'rend_normal': render_normal,
        'rend_dist': render_dist,
        'surf_depth': surf_depth,
        'surf_normal': surf_normal,
        'rend_depth': render_depth_expected,
    })
    
    # Add extra field to store starting index for each point cloud, for subsequent loss calculations
    model_start_indices = [0]
    total = 0
    for pc in pc_list:
        total += pc.get_xyz.shape[0]
        model_start_indices.append(total)
    
    rets['model_start_indices'] = model_start_indices

    return rets
