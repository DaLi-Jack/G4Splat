import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from scene import Scene
from os import makedirs
from gaussian_renderer import render
import random
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import trimesh
from tetranerf.utils.extension import cpp
from utils.tetmesh import marching_tetrahedra

def get_cameras_spatial_extent(views, device):
    from matcha.dm_scene.cameras import GSCamera, CamerasWrapper
    _gs_cameras = []
    for cam in views:
        _gs_cameras.append(
            GSCamera(
                colmap_id=cam.colmap_id, 
                R=torch.tensor(cam.R).float().to(device), 
                T=torch.tensor(cam.T).float().to(device), 
                FoVx=cam.FoVx, 
                FoVy=cam.FoVy, 
                image=cam.original_image, 
                gt_alpha_mask=cam.gt_alpha_mask,
                image_name=cam.image_name, 
                uid=cam.uid,
                data_device = device,
            )
        )
    nerf_cameras = CamerasWrapper(_gs_cameras)
    return nerf_cameras.get_spatial_extent()


@torch.no_grad()
def get_dilated_depth(view, depth, dilation_pixels, rgb=None, max_dilation=1e8):
    """
    Get dilated depth from a depth map.

    Args:
        view (GSCamera): Viewpoint camera.
        depth (torch.Tensor): Depth map. Should have shape (H, W) or (1, H, W), or (H, W, 1).
        dilation_factor (float): Dilation factor.
        rgb (torch.Tensor, optional): RGB image. Defaults to None. Has shape (3, H, W) or (H, W, 3).
    """
    from matcha.dm_scene.charts import depths_to_points_parallel
    from matcha.dm_scene.meshes import get_manifold_meshes_from_pointmaps, Meshes, TexturesVertex, render_mesh_with_pytorch3d
    from matcha.dm_scene.cameras import GSCamera, CamerasWrapper
    
    # Get camera object
    gs_cameras = []
    for cam in [view]:
        gs_cameras.append(
            GSCamera(
                colmap_id=cam.colmap_id, 
                R=torch.tensor(cam.R).float().to(depth.device), 
                T=torch.tensor(cam.T).float().to(depth.device), 
                FoVx=cam.FoVx, 
                FoVy=cam.FoVy, 
                image=cam.original_image, 
                gt_alpha_mask=cam.gt_alpha_mask,
                image_name=cam.image_name, 
                uid=cam.uid,
                data_device = "cuda",
            )
        )
    nerf_cameras = CamerasWrapper(gs_cameras)
    
    # Get 3D points from depth map
    depth_view = depth.squeeze()  # (H, W)
    if False:
        # Depth to points function from 2DGS leads to imprecision in the backprojection.
        pts = depths_to_points_parallel(depth_view[None], [view]).view(*depth_view.shape, 3)  # (H, W, 3)
    else:
        pts = nerf_cameras.backproject_depth(cam_idx=0, depth=depth_view).view(*depth_view.shape, 3)
    if rgb is not None:
        if rgb.shape[0] == 3:
            rgb_view = rgb.permute(1, 2, 0)  # (H, W, 3)
        else:
            rgb_view = rgb  # (H, W, 3)
    else:
        rgb_view = torch.ones_like(pts)
    
    # Build surface mesh
    meshes = get_manifold_meshes_from_pointmaps(
        points3d=pts[None],  # (1, H, W, 3)
        imgs=rgb_view[None],  # (1, H, W, 3)
        return_single_mesh_object=False
    )
    verts, faces = meshes[0].verts_packed(), meshes[0].faces_packed()
    verts_normals = meshes[0].verts_normals_packed()
    verts_features = meshes[0].textures.verts_features_packed()
    
    # Compute dilation factor, depending on the depth (dilate more in distant areas)
    focals = (nerf_cameras.fx[0] + nerf_cameras.fy[0]).item() / 2  # focal length in pixels
    dilation_factor = dilation_pixels / focals * depth.view(-1, 1)
    dilation_factor = dilation_factor.clamp(max=max_dilation)
    
    # Dilate surface mesh along normals
    dilated_verts = verts + dilation_factor * verts_normals
    dilated_mesh = Meshes(
        verts=[dilated_verts], 
        faces=[faces],
        textures=TexturesVertex(verts_features=[verts_features])
    )
    
    # Render dilated mesh to get dilated depth and rgb
    render_pkg = render_mesh_with_pytorch3d(
        dilated_mesh, 
        nerf_cameras=nerf_cameras, 
        camera_idx=0,
    )
    dilated_rgb = render_pkg["rgb"]
    dilated_depth = render_pkg["depth"]
    dilated_depth[dilated_depth == 0.] = depth_view[dilated_depth == 0.]
    dilated_rgb[dilated_depth == 0.] = rgb_view[dilated_depth == 0.]
    dilated_rgb = dilated_rgb.permute(2, 0, 1)
    return dilated_depth.view(*depth.shape), dilated_rgb


@torch.no_grad()
def integrate_with_depth(points, view, gaussians:GaussianModel, pipe, background, spatial_extent):
    from matcha.dm_scene.charts import (
        get_points_depth_in_depthmap_parallel, 
        get_patches_depth_in_depthmap_parallel,
        get_patches_points_in_depthmap_parallel,
        transform_points_world_to_view,
    )
    render_pkg = render(view, gaussians, pipe, background)
    depth = render_pkg['surf_depth']
    rgb = render_pkg["render"]
    
    # Get actual depth
    points_depths = transform_points_world_to_view([view], points)[0, ..., 2]  # (n_points)
    
    # Get depth in depth map
    if False:
        surf_depths, fov_mask = get_points_depth_in_depthmap_parallel(points, depth, [view])  # (n_charts, n_points)
        surf_depths, fov_mask = surf_depths[0], fov_mask[0]  # (n_points)
        # Compute signed distance
        # If the point is in front of the surface, the signed distance is positive.
        # If the point is behind the surface, the signed distance is negative.
        sgn_dists = surf_depths - points_depths
    elif False:
        surf_depths, fov_mask = get_patches_depth_in_depthmap_parallel(points, depth, [view], patch_size=3)
        surf_depths, fov_mask = surf_depths[0], fov_mask[0]  # (n_points, patch_size^2)
        # Compute signed distance
        # If the point is in front of the surface, the signed distance is positive.
        # If the point is behind the surface, the signed distance is negative.
        sgn_dists = (surf_depths - points_depths[..., None]).min(dim=-1).values
    elif True:
        # dilation_factor = 0.0025
        # 1. and 0.005 work pretty well.
        # 1.5 and 1e-3 work pretty well.
        dilation_pixels = 1.5  # 3., 2., 1.5
        max_dilation = 1e-3 * spatial_extent  # 0.01
        dilated_depth, dilated_rgb = get_dilated_depth(view, depth, dilation_pixels, rgb, max_dilation)
        surf_depths, fov_mask = get_points_depth_in_depthmap_parallel(
            points, dilated_depth, [view], 
            # interpolation_mode='nearest',
            interpolation_mode='bilinear',
        )  # (n_charts, n_points)
        surf_depths, fov_mask = surf_depths[0], fov_mask[0]  # (n_points)
        sgn_dists = surf_depths - points_depths
    else:
        # Compute depth
        surf_depths, fov_mask = get_points_depth_in_depthmap_parallel(points, depth, [view])  # (n_charts, n_points)
        surf_depths, fov_mask = surf_depths[0], fov_mask[0]  # (n_points)
        
        # Compute points
        surf_pts, _ = get_patches_points_in_depthmap_parallel(points, depth, [view], patch_size=7)
        surf_pts = surf_pts[0]  # (n_points, patch_size^2, 3)

        # Compute signed distance
        dists = (surf_pts - points[..., None, :]).norm(dim=-1).min(dim=-1).values
        # sgn_dists = dists * (surf_depths - points_depths).sign()
        sgn_dists = (dists - 0.01)
        
    # sgn_dists = (sgn_dists - 0.001)
          
    # Compute alpha: Points behind the surface are fully opaque.
    integrated_alpha = (sgn_dists < 0).float()
    # Points outside the field of view are considered fully opaque.
    integrated_alpha[~fov_mask] = 1.
    sgn_dists[~fov_mask] = -1.
    
    # TODO: Compute color: The color is the one of the surface.
    integrated_color = torch.ones(integrated_alpha.shape[0], 3, device=integrated_alpha.device)
    
    res_dict = {
        "sgn_dists": sgn_dists,
        "alpha_integrated": integrated_alpha,
        "color_integrated": integrated_color,
    }
    return res_dict


@torch.no_grad()
def evaluage_alpha(points, views, gaussians:GaussianModel, pipeline, background, return_color=False,):
    final_alpha = torch.ones((points.shape[0]), dtype=torch.float32, device="cuda")
    if return_color:
        final_color = torch.ones((points.shape[0], 3), dtype=torch.float32, device="cuda")
    
    final_sgns = -torch.ones((points.shape[0]), dtype=torch.float32, device="cuda")
    final_dists = 1e6 * torch.ones((points.shape[0]), dtype=torch.float32, device="cuda")
    
    spatial_extent = get_cameras_spatial_extent(views, device=points.device)
    
    with torch.no_grad():
        for _, view in enumerate(tqdm(views, desc="Rendering progress")):
            ret = integrate_with_depth(points, view, gaussians, pipeline, background, spatial_extent)
            alpha_integrated = ret["alpha_integrated"]
            sgn_dists = ret["sgn_dists"] # - 0.1
            if return_color:
                color_integrated = ret["color_integrated"]    
                final_color = torch.where((sgn_dists.abs() < final_dists).reshape(-1, 1), color_integrated, final_color)
                # final_color = torch.where((alpha_integrated < final_alpha).reshape(-1, 1), color_integrated, final_color)
            final_alpha = torch.min(final_alpha, alpha_integrated)
            final_sgns = torch.max(final_sgns, sgn_dists.sign())
            final_dists = torch.min(final_dists, sgn_dists.abs())
            
        # alpha = final_sgns * final_dists + 0.5 - 0.0025
        alpha = 1 - final_alpha

    if return_color:
        return alpha, final_color
    return alpha


@torch.no_grad()
def marching_tetrahedra_with_binary_search(
    model_path : str, 
    name : str, 
    iteration : int, 
    views, 
    gaussians : GaussianModel, 
    pipeline, 
    background : torch.Tensor, 
    filter_mesh : bool, 
    texture_mesh : bool, 
    downsample_ratio : float,
    gaussian_flatness : float,
    output_dir : str = None,
):
    if output_dir is None:
        render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "fusion")
    else:
        render_path = output_dir
    makedirs(render_path, exist_ok=True)
    
    # generate tetra points here
    points, points_scale = gaussians.get_tetra_points(downsample_ratio=downsample_ratio, gaussian_flatness=gaussian_flatness)
    # load cell if exists
    if os.path.exists(os.path.join(render_path, "cells.pt")):
        print("load existing cells")
        cells = torch.load(os.path.join(render_path, "cells.pt"))
    else:
        # create cell and save cells
        print("create cells and save")
        cells = cpp.triangulate(points)
        # we should filter the cell if it is larger than the gaussians
        torch.save(cells, os.path.join(render_path, "cells.pt"))
    
    # evaluate alpha
    alpha = evaluage_alpha(points, views, gaussians, pipeline, background)

    vertices = points.cuda()[None]
    tets = cells.cuda().long()

    print(vertices.shape, tets.shape, alpha.shape)
    def alpha_to_sdf(alpha):    
        sdf = alpha - 0.5
        sdf = sdf[None]
        return sdf
    
    sdf = alpha_to_sdf(alpha)
    
    torch.cuda.empty_cache()
    verts_list, scale_list, faces_list, _ = marching_tetrahedra(vertices, tets, sdf, points_scale[None])
    torch.cuda.empty_cache()
    
    end_points, end_sdf = verts_list[0]
    end_scales = scale_list[0]
    
    faces=faces_list[0].cpu().numpy()
    points = (end_points[:, 0, :] + end_points[:, 1, :]) / 2.
        
    left_points = end_points[:, 0, :]
    right_points = end_points[:, 1, :]
    left_sdf = end_sdf[:, 0, :]
    right_sdf = end_sdf[:, 1, :]
    left_scale = end_scales[:, 0, 0]
    right_scale = end_scales[:, 1, 0]
    distance = torch.norm(left_points - right_points, dim=-1)
    scale = left_scale + right_scale
    
    n_binary_steps = 8
    for step in range(n_binary_steps):
        print("binary search in step {}".format(step))
        mid_points = (left_points + right_points) / 2
        alpha = evaluage_alpha(mid_points, views, gaussians, pipeline, background)
        mid_sdf = alpha_to_sdf(alpha).squeeze().unsqueeze(-1)
        
        ind_low = ((mid_sdf < 0) & (left_sdf < 0)) | ((mid_sdf > 0) & (left_sdf > 0))

        left_sdf[ind_low] = mid_sdf[ind_low]
        right_sdf[~ind_low] = mid_sdf[~ind_low]
        left_points[ind_low.flatten()] = mid_points[ind_low.flatten()]
        right_points[~ind_low.flatten()] = mid_points[~ind_low.flatten()]
    
        points = (left_points + right_points) / 2
        if step not in [7]:
            continue
        
        if texture_mesh:
            active_sh_degree = gaussians.active_sh_degree
            gaussians.active_sh_degree = 0
            _, color = evaluage_alpha(points, views, gaussians, pipeline, background, return_color=True)
            vertex_colors=(color.cpu().numpy() * 255).astype(np.uint8)
            gaussians.active_sh_degree = active_sh_degree
        else:
            vertex_colors=None
        mesh = trimesh.Trimesh(vertices=points.cpu().numpy(), faces=faces, vertex_colors=vertex_colors, process=False)
        
        # filter
        if filter_mesh:
            mask = (distance <= scale).cpu().numpy()
            face_mask = mask[faces].all(axis=1)
            mesh.update_vertices(mask)
            mesh.update_faces(face_mask)
        
        mesh.export(os.path.join(render_path, f"tetra_mesh_binary_search_{step}.ply"))

    # linear interpolation
    # right_sdf *= -1
    # points = (left_points * left_sdf + right_points * right_sdf) / (left_sdf + right_sdf)
    # mesh = trimesh.Trimesh(vertices=points.cpu().numpy(), faces=faces)
    # mesh.export(os.path.join(render_path, f"mesh_binary_search_interp.ply"))
    

def extract_mesh(
    dataset : ModelParams, 
    iteration : int, 
    pipeline : PipelineParams, 
    filter_mesh : bool, 
    texture_mesh : bool, 
    downsample_ratio : float,
    gaussian_flatness : float,
    output_dir : str = None,
):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        print(("Dataset: ", dataset.source_path, dataset.model_path))
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        cams = scene.getTrainCameras()
        
        # TODO: remove
        # previous_depth_ratio = pipeline.depth_ratio
        # pipeline.depth_ratio = 1.0
        # print(f"[WARNING] pipeline.depth_ratio set from {previous_depth_ratio} to {pipeline.depth_ratio}")
        
        marching_tetrahedra_with_binary_search(
            model_path=dataset.model_path, 
            name="test", 
            iteration=iteration, 
            views=cams, 
            gaussians=gaussians, 
            pipeline=pipeline, 
            background=background, 
            filter_mesh=filter_mesh, 
            texture_mesh=texture_mesh, 
            downsample_ratio=downsample_ratio,
            gaussian_flatness=gaussian_flatness,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--filter_mesh", action="store_true")
    parser.add_argument("--texture_mesh", action="store_true")
    parser.add_argument("--downsample_ratio", type=float, default=None)
    parser.add_argument("--gaussian_flatness", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, default=None)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
    
    extract_mesh(
        dataset=model.extract(args), 
        iteration=args.iteration, 
        pipeline=pipeline.extract(args), 
        filter_mesh=args.filter_mesh, 
        texture_mesh=args.texture_mesh, 
        downsample_ratio=args.downsample_ratio,
        gaussian_flatness=args.gaussian_flatness,
        output_dir=args.output_dir
    )
