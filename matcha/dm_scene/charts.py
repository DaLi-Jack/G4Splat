import numpy as np
import torch
from pytorch3d.transforms import quaternion_apply
from matcha.dm_scene.meshes import get_manifold_meshes_from_pointmaps, remove_faces_from_single_mesh
from matcha.dm_scene.gaussians import get_gaussian_surfel_parameters_from_mesh
from matcha.dm_utils.rendering import depth2normal_parallel, normal2curv_parallel


def load_charts_data(path: str, device:str='cuda'):
    """Load charts data from a .npz file created by save_charts_data.
    
    Args:
        path (str): Path to the .npz file containing the charts data
        
    Returns:
        dict: Dictionary containing the loaded data with keys:
            - pts: 3D points (if saved)
            - cols: Point colors (if saved)
            - confs: Confidence values (if saved)
            - depths: Depth values (if saved)
            - normals: Normal vectors (if saved) 
            - curvatures: Curvature values (if saved)
    """
    if not path.endswith('.npz'):
        path = path + '.npz'
        
    data = np.load(path)
    
    # Convert all arrays to torch tensors
    output = {}
    for key in data.files:
        if data[key] is not None:
            output[key] = torch.from_numpy(data[key]).to(device)
            
    return output


def build_priors_from_charts_data(charts_data: dict, train_cameras):
    
    height, width = train_cameras[0].image_height, train_cameras[0].image_width
    
    charts_scale_factor = charts_data['scale_factor']
    
    charts_depths_height, charts_depths_width = charts_data['depths'][0].shape
    if charts_depths_height != height or charts_depths_width != width:
        print(f"[INFO] Interpolating charts depths from {charts_depths_height}x{charts_depths_width} to {height}x{width}...")
        charts_depths = torch.nn.functional.interpolate(
            charts_data['depths'][:, None] / charts_scale_factor, 
            size=(height, width), 
            mode="bilinear", 
            align_corners=False
        )[:, 0]  # Shape (n_charts, h, w)
    else:
        charts_depths = charts_data['depths'] / charts_scale_factor  # Shape (n_charts, h, w)
    
    charts_prior_depths = torch.nn.functional.interpolate(
        charts_data['prior_depths'][:, None] / charts_scale_factor, 
        size=(height, width), 
        mode="bilinear", 
        align_corners=False
    )[:, 0]  # Shape (n_charts, h, w)
    
    charts_confs = torch.nn.functional.interpolate(
        charts_data['confs'][:, None], 
        size=(height, width), 
        mode="bilinear", 
        align_corners=False
    )[:, 0]  # Shape (n_charts, h, w)
    
    """Build priors from charts data."""
    # Compute prior normals
    world_view_transforms = torch.stack([train_cameras[i].world_view_transform for i in range(len(train_cameras))])
    full_proj_transforms = torch.stack([train_cameras[i].full_proj_transform for i in range(len(train_cameras))])

    # We unscale the prior normals
    charts_prior_normals = depth2normal_parallel(
        charts_prior_depths, 
        world_view_transforms=world_view_transforms, 
        full_proj_transforms=full_proj_transforms
    ).permute(0, 3, 1, 2)  # Shape (n_charts, 3, h ,w)
    
    # Compute prior curvatures
    charts_prior_curvs = normal2curv_parallel(charts_prior_normals, torch.ones_like(charts_prior_normals[:, 0:1]))  # .permute(0, 2, 3, 1)  # Shape (n_charts, 1, h, w)
    
    output_pkg = {
        'scale_factor': charts_scale_factor,
        'depths': charts_depths[:, None],  # Shape (n_charts, 1, h, w)
        'prior_depths': charts_prior_depths[:, None],  # Shape (n_charts, 1, h, w)
        'confs': charts_confs[:, None],  # Shape (n_charts, 1, h, w)
        'normals': charts_prior_normals,  # Shape (n_charts, 3, h, w)
        'curvs': charts_prior_curvs  # Shape (n_charts, 1, h, w)
    }

    return output_pkg


def schedule_regularization_factor_1(iteration, initial_factor=0.5):
    regularization_factor = initial_factor
 
    if iteration > 3500:
        regularization_factor = initial_factor / 5.
 
    if iteration > 5000:
        regularization_factor = initial_factor / (5. ** 2)
    
    return regularization_factor


def schedule_regularization_factor_2(iteration, initial_factor=0.5): 
    n_thousands = iteration // 1000
    regularization_factor = initial_factor / (2 ** n_thousands)
    regularization_factor = max(regularization_factor, 0.015)
    return regularization_factor


def schedule_regularization_factor(iteration, initial_factor=0.5, time_interval=1000, downscale_factor=2, min_factor=0.015): 
    n_intervals = iteration // time_interval
    regularization_factor = initial_factor / (downscale_factor ** n_intervals)
    regularization_factor = max(regularization_factor, min_factor)
    return regularization_factor

def voxel_downsample_gaussians(gaussian_params, voxel_size=0.01):
    """
    Downsample Gaussian parameters using voxel-based method
    
    Args:
        gaussian_params: Dictionary containing Gaussian parameters, must include 'means' key
        voxel_size: Voxel size for downsampling, default 0.01
    
    Returns:
        sample_idx: Tensor containing the indices of the downsampled Gaussian parameters
    """
    import open3d as o3d
    
    # Get Gaussian center points
    means = gaussian_params['means']  # [N, 3]
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(means.cpu().numpy())
    
    # Add indices as color information for tracking original indices
    # Encode index information into colors
    original_indices = np.arange(len(means))
    # Use normalized indices as colors so we can recover original indices after downsampling
    normalized_indices = original_indices / len(means)
    colors = np.column_stack([normalized_indices, np.zeros(len(means)), np.zeros(len(means))])
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Perform voxel downsampling
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)
    
    # Recover original indices from downsampled point cloud
    downsampled_colors = np.asarray(downsampled_pcd.colors)
    recovered_indices = np.round(downsampled_colors[:, 0] * len(means)).astype(int)

    # Convert to torch tensor
    sample_idx = torch.from_numpy(recovered_indices).long()

    downsample_factor = len(means) / len(sample_idx)
    
    print(f"Voxel downsampling: {len(means)} -> {len(sample_idx)} gaussians (factor: {downsample_factor:.2f})")
    
    return sample_idx, downsample_factor

def get_gaussian_parameters_from_charts_data(
    charts_data: dict, 
    images, 
    conf_th=-1.,
    ratio_th=5.,
    normal_scale=1e-4,
    normalized_scales=0.5,
    visibility_masks=None,
):
    """Get gaussian parameters from charts data."""
    charts_pts = charts_data['pts'] / charts_data['scale_factor']
    charts_confs = charts_data['confs']

    if visibility_masks is not None:
        visibility_masks = torch.cat(visibility_masks, dim=0)
        charts_confs = charts_confs * visibility_masks
        conf_th = 0.1
    
    print("Conf Max/min: ", charts_confs.max(), charts_confs.min())
    
    # Get manifold mesh and remove faces with low confidence if needed
    manifold = get_manifold_meshes_from_pointmaps(
        points3d=charts_pts,
        imgs=images, 
        masks=charts_confs > conf_th,  
        return_single_mesh_object=True
    )
    
    # Remove elongated faces
    faces_verts = manifold.verts_packed()[manifold.faces_packed()]  # (n_faces, 3, 3)
    sides = (
        torch.roll(faces_verts, 1, dims=1)  # C, A, B
        - faces_verts  # A, B, C
    )  # (n_faces, 3, 3)  ;  AC, BA, CB
    normalized_sides = torch.nn.functional.normalize(sides, dim=-1)  # (n_faces, 3, 3)  ;  AC/||AC||, BA/||BA||, CB/||CB||
    alts = (
        sides  # AC
        - (sides * torch.roll(normalized_sides, -1, dims=1)).sum(dim=-1, keepdim=True) * normalized_sides # - (AC . BA) BA / ||BA||^2
    )  # (n_faces, 3, 3)
    alt_lengths = alts.norm(dim=-1)
    alt_ratios = alt_lengths.max(dim=1).values / alt_lengths.min(dim=1).values
    faces_mask = alt_ratios < ratio_th
    manifold = remove_faces_from_single_mesh(manifold, faces_to_keep_mask=faces_mask)
    
    # Get gaussian parameters
    gaussian_params = get_gaussian_surfel_parameters_from_mesh(
        barycentric_coords=1,
        mesh=manifold,
        normalized_scales=normalized_scales,
        get_colors_from_mesh=True,
        normal_scale=normal_scale / charts_data['scale_factor'],
    )
    
    return gaussian_params

def get_gaussian_parameters_from_pa_data(
    pa_points, 
    images, 
    conf_th=-1.,
    ratio_th=5.,
    normal_scale=1e-4,
    normalized_scales=0.5,
    visibility_masks=None,
):
    """Get gaussian parameters from pa data."""

    # NOTE: pa_conf temp
    pa_conf = torch.ones((pa_points.shape[0], pa_points.shape[1], 1), device=pa_points.device, dtype=torch.float32)

    if visibility_masks is not None:
        visibility_masks = torch.cat(visibility_masks, dim=0)
        pa_conf = pa_conf * visibility_masks
        conf_th = 0.1
    
    print("Conf Max/min: ", pa_conf.max(), pa_conf.min())
    
    # Get manifold mesh and remove faces with low confidence if needed
    manifold = get_manifold_meshes_from_pointmaps(
        points3d=pa_points,
        imgs=images, 
        # masks=pa_conf > conf_th,  
        masks=None,
        return_single_mesh_object=True
    )
    
    # Remove elongated faces
    faces_verts = manifold.verts_packed()[manifold.faces_packed()]  # (n_faces, 3, 3)
    sides = (
        torch.roll(faces_verts, 1, dims=1)  # C, A, B
        - faces_verts  # A, B, C
    )  # (n_faces, 3, 3)  ;  AC, BA, CB
    normalized_sides = torch.nn.functional.normalize(sides, dim=-1)  # (n_faces, 3, 3)  ;  AC/||AC||, BA/||BA||, CB/||CB||
    alts = (
        sides  # AC
        - (sides * torch.roll(normalized_sides, -1, dims=1)).sum(dim=-1, keepdim=True) * normalized_sides # - (AC . BA) BA / ||BA||^2
    )  # (n_faces, 3, 3)
    alt_lengths = alts.norm(dim=-1)
    alt_ratios = alt_lengths.max(dim=1).values / alt_lengths.min(dim=1).values
    faces_mask = alt_ratios < ratio_th
    manifold = remove_faces_from_single_mesh(manifold, faces_to_keep_mask=faces_mask)
    
    # Get gaussian parameters
    gaussian_params = get_gaussian_surfel_parameters_from_mesh(
        barycentric_coords=1,
        mesh=manifold,
        normalized_scales=normalized_scales,
        get_colors_from_mesh=True,
        normal_scale=normal_scale,
    )
    
    return gaussian_params

def transform_points_world_to_view(
    cameras,
    points:torch.Tensor,
    use_p3d_convention:bool=True,
):
    """_summary_

    Args:
        points (torch.Tensor): Should have shape (n_cameras, N, 3).
        use_p3d_convention (bool, optional): Defaults to True.
    """
    world_view_transforms = torch.stack([gs_camera.world_view_transform for gs_camera in cameras], dim=0)  # (n_cameras, 4, 4)
    
    points_h = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)  # (n_cameras, N, 4)
    view_points = (points_h @ world_view_transforms)[..., :3]  # (n_cameras, N, 3)
    if use_p3d_convention:
        factors = torch.tensor([[[-1, -1, 1]]], device=points.device)  # (1, 1, 3)
        view_points = factors * view_points  # (n_cameras, N, 3)
    return view_points
    
    
def project_points(
    cameras,
    points:torch.Tensor,
    points_are_already_in_view_space:bool=False,
    use_p3d_convention:bool=True,
    znear=1e-6,
):
    """_summary_

    Args:
        points (torch.Tensor): Should have shape (n_cameras, N, 3).
        use_p3d_convention (bool, optional): Defaults to True.

    Returns:
        _type_: _description_
    """
    if points_are_already_in_view_space:
        full_proj_transforms = torch.stack([gs_camera.projection_matrix for gs_camera in cameras])  # (n_depth, 4, 4)
    else:
        full_proj_transforms = torch.stack([gs_camera.full_proj_transform for gs_camera in cameras])  # (n_cameras, 4, 4)
    
    points_h = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)  # (n_cameras, N, 4)
    proj_points = points_h @ full_proj_transforms  # (n_cameras, N, 4)
    proj_points = proj_points[..., :2] / proj_points[..., 3:4].clamp_min(znear)  # (n_cameras, N, 2)
    if use_p3d_convention:
        height, width = cameras[0].image_height, cameras[0].image_width
        # TODO: Handle different image sizes for different cameras
        factors = torch.tensor([[[-width / min(height, width), -height / min(height, width)]]], device=points.device)  # (1, 1, 2)
        proj_points = factors * proj_points  # (n_cameras, N, 2)
        if points_are_already_in_view_space:
            proj_points = - proj_points
    return proj_points


def depths_to_points_parallel(
    depthmap,
    cameras=None,
    world_view_transforms=None, 
    full_proj_transforms=None,
):
    """Reworked. Originally comes from 2DGS.

    Args:
        world_view_transforms (_type_): (n_camera, 4, 4)
        full_proj_transforms (_type_): (n_camera, 4, 4)
        depthmap (_type_): (n_camera, H, W) or (n_camera, 1, H, W)

    Returns:
        _type_: _description_
    """
    no_matrix_provided = (world_view_transforms is None) or (full_proj_transforms is None)
    if no_matrix_provided and cameras is None:
        raise ValueError("Either provide the camera matrices or the camera objects.")
    if world_view_transforms is None:
        world_view_transforms = torch.stack([gs_camera.world_view_transform for gs_camera in cameras])
    if full_proj_transforms is None:
        full_proj_transforms = torch.stack([gs_camera.full_proj_transform for gs_camera in cameras])
    
    c2w = (world_view_transforms.transpose(-1, -2)).inverse()  # (n_camera, 4, 4)
    W, H = depthmap.shape[-1], depthmap.shape[-2]
    ndc2pix = torch.tensor([
        [W / 2, 0, 0, (W) / 2],
        [0, H / 2, 0, (H) / 2],
        [0, 0, 0, 1]]).float().cuda().T  # (4, 3)
    projection_matrix = c2w.transpose(-1, -2) @ full_proj_transforms  # (n_camera, 4, 4)
    intrins = (projection_matrix @ ndc2pix)[..., :3,:3].transpose(-1, -2)  # (n_camera, 3, 3)
    
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')  # (H, W)
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)  # (H * W, 3)
    rays_d = points[None] @ intrins.inverse().transpose(-1, -2) @ c2w[..., :3,:3].transpose(-1, -2)  # (n_camera, H * W, 3)
    rays_o = c2w[..., None, :3,3]  # (n_camera, 1, 3)
    points = depthmap.reshape(-1, H*W, 1) * rays_d + rays_o
    return points

def depths_to_sample_points_parallel(
    depthmap,
    sample_num,
    cameras=None,
    world_view_transforms=None, 
    full_proj_transforms=None,
):
    """Reworked. Originally comes from 2DGS.

    Args:
        world_view_transforms (_type_): (n_camera, 4, 4)
        full_proj_transforms (_type_): (n_camera, 4, 4)
        depthmap (_type_): (n_camera, H, W) or (n_camera, 1, H, W)

    Returns:
        _type_: _description_
    """
    no_matrix_provided = (world_view_transforms is None) or (full_proj_transforms is None)
    if no_matrix_provided and cameras is None:
        raise ValueError("Either provide the camera matrices or the camera objects.")
    if world_view_transforms is None:
        world_view_transforms = torch.stack([gs_camera.world_view_transform for gs_camera in cameras])
    if full_proj_transforms is None:
        full_proj_transforms = torch.stack([gs_camera.full_proj_transform for gs_camera in cameras])
    
    c2w = (world_view_transforms.transpose(-1, -2)).inverse()  # (n_camera, 4, 4)
    W, H = depthmap.shape[-1], depthmap.shape[-2]
    ndc2pix = torch.tensor([
        [W / 2, 0, 0, (W) / 2],
        [0, H / 2, 0, (H) / 2],
        [0, 0, 0, 1]]).float().cuda().T  # (4, 3)
    projection_matrix = c2w.transpose(-1, -2) @ full_proj_transforms  # (n_camera, 4, 4)
    intrins = (projection_matrix @ ndc2pix)[..., :3,:3].transpose(-1, -2)  # (n_camera, 3, 3)
    
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')  # (H, W)
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)  # (H * W, 3)
    rays_d = points[None] @ intrins.inverse().transpose(-1, -2) @ c2w[..., :3,:3].transpose(-1, -2)  # (n_camera, H * W, 3)
    rays_o = c2w[..., None, :3,3]  # (n_camera, 1, 3)
    t_vals = torch.linspace(0, 1, sample_num, device=depthmap.device)  # (sample_num,)
    t_vals = t_vals.reshape(1, 1, sample_num, 1) * depthmap.reshape(-1, H*W, 1, 1)  # (n_camera, H * W, sample_num, 1)
    points = rays_o.unsqueeze(-2) + t_vals * rays_d.unsqueeze(-2)  # (n_camera, H * W, sample_num, 3)
    return points

def get_points_depth_in_depthmap_parallel(
    pts:torch.Tensor, 
    depthmap:torch.Tensor, 
    cameras,
    padding_mode='zeros',  # 'reflection', 'border'
    interpolation_mode='bilinear',  # 'bilinear', 'nearest'
    znear=1e-6,
):
    """_summary_

    Args:
        pts (torch.Tensor): Has shape (n_depths, N, 3).
        depthmap (torch.Tensor): Has shape (n_depths, H, W) or (n_depths, H, W, 1).
        p3d_camera (P3DCameras): Should contain n_depths cameras.

    Returns:
        _type_: _description_
    """
    n_depths, image_height, image_width = depthmap.shape[:3]

    pts_projections = transform_points_world_to_view(cameras, pts)  # (n_depths, N, 3)
    fov_mask = pts_projections[..., 2] > 0.  # (n_depths, N)
    pts_projections.clamp(min=torch.tensor([[[-1e8, -1e8, znear]]]).to(pts_projections.device))
    
    pts_projections = project_points(cameras, pts_projections, points_are_already_in_view_space=True, znear=znear)  # (n_depths, N, 2)
    fov_mask = fov_mask & pts_projections.isfinite().all(dim=-1)  # (n_depths, N)
    pts_projections = pts_projections.nan_to_num(nan=0., posinf=0., neginf=0.)
        
    factor = -1 * min(image_height, image_width)
    factors = torch.tensor([[[factor / image_width, factor / image_height]]]).to(pts.device)  # (1, 1, 2)
    pts_projections = pts_projections[..., :2] * factors  # (n_depths, N, 2)
    pts_projections = pts_projections.view(n_depths, -1, 1, 2)

    depth_view = depthmap.reshape(n_depths, 1, image_height, image_width)  # (n_depths, 1, H, W)
    map_z = torch.nn.functional.grid_sample(
        input=depth_view,
        grid=pts_projections,
        mode=interpolation_mode,
        padding_mode=padding_mode,  # 'reflection', 'zeros'
        align_corners=False,
    )  # (n_depths, 1, N, 1)
    map_z = map_z[:, 0, :, 0]  # (n_depths, N)
    fov_mask = (map_z > 0.) & fov_mask
    map_z = map_z * fov_mask
    
    return map_z, fov_mask


def get_patches_depth_in_depthmap_parallel(
    pts:torch.Tensor, 
    depthmap:torch.Tensor, 
    cameras,
    padding_mode='zeros',  # 'reflection', 'border'
    znear=1e-6,
    patch_size:int=5,
    points_per_patch_side:int=None,
):
    """_summary_

    Args:
        pts (torch.Tensor): Has shape (n_depths, N, 3).
        depthmap (torch.Tensor): Has shape (n_depths, H, W) or (n_depths, H, W, 1).
        p3d_camera (P3DCameras): Should contain n_depths cameras.

    Returns:
        _type_: _description_
    """
    n_depths, image_height, image_width = depthmap.shape[:3]
    if points_per_patch_side is None:
        points_per_patch_side = patch_size

    pts_projections = transform_points_world_to_view(cameras, pts)  # (n_depths, N, 3)
    fov_mask = pts_projections[..., 2] > 0.  # (n_depths, N)
    pts_projections.clamp(min=torch.tensor([[[-1e8, -1e8, znear]]]).to(pts_projections.device))
    
    pts_projections = project_points(cameras, pts_projections, points_are_already_in_view_space=True, znear=znear)  # (n_depths, N, 2)
    fov_mask = fov_mask & pts_projections.isfinite().all(dim=-1)  # (n_depths, N)
    pts_projections = pts_projections.nan_to_num(nan=0., posinf=0., neginf=0.)  # Between -/+ max(image_height, image_width) / min(image_height, image_width)
        
    factor = -1 * min(image_height, image_width)
    factors = torch.tensor([[[factor / image_width, factor / image_height]]]).to(pts.device)  # (1, 1, 2)
    pts_projections = pts_projections[..., :2] * factors  # (n_depths, N, 2)
    pts_projections = pts_projections.view(n_depths, -1, 1, 2)  # Between -1 and 1
    
    x_shifts = torch.linspace(-(patch_size - 1) // 2, (patch_size - 1) // 2, points_per_patch_side, device=pts.device) / image_width * 2
    y_shifts = torch.linspace(-(patch_size - 1) // 2, (patch_size - 1) // 2, points_per_patch_side, device=pts.device) / image_height * 2
    shifts = torch.stack(torch.meshgrid(x_shifts, y_shifts, indexing='ij'), dim=-1)  # (points_per_patch_side, points_per_patch_side, 2)
    shifts = shifts.view(1, 1, -1, 2)  # (1, 1, points_per_patch_side^2, 2)
    
    patch_projections = pts_projections + shifts  # (n_depths, N, points_per_patch_side^2, 2)

    depth_view = depthmap.reshape(n_depths, 1, image_height, image_width)  # (n_depths, 1, H, W)
    map_z = torch.nn.functional.grid_sample(
        input=depth_view,
        grid=patch_projections,
        mode='bilinear',
        padding_mode='zeros',  # 'reflection', 'zeros'
        align_corners=False,
    )  # (n_depths, 1, N, points_per_patch_side^2)
    map_z = map_z[:, 0, :, :]  # (n_depths, N, points_per_patch_side^2)
    fov_mask = (map_z > 0.).all(dim=-1) & fov_mask
    map_z = map_z * fov_mask[..., None]
    
    return map_z, fov_mask


def get_patches_points_in_depthmap_parallel(
    pts:torch.Tensor, 
    depthmap:torch.Tensor, 
    cameras,
    padding_mode='zeros',  # 'reflection', 'border'
    znear=1e-6,
    patch_size:int=5,
    points_per_patch_side:int=None,
):
    """_summary_

    Args:
        pts (torch.Tensor): Has shape (n_depths, N, 3).
        depthmap (torch.Tensor): Has shape (n_depths, H, W) or (n_depths, H, W, 1).
        p3d_camera (P3DCameras): Should contain n_depths cameras.

    Returns:
        _type_: _description_
    """
    n_depths, image_height, image_width = depthmap.shape[:3]
    if points_per_patch_side is None:
        points_per_patch_side = patch_size

    pts_projections = transform_points_world_to_view(cameras, pts)  # (n_depths, N, 3)
    fov_mask = pts_projections[..., 2] > 0.  # (n_depths, N)
    pts_projections.clamp(min=torch.tensor([[[-1e8, -1e8, znear]]]).to(pts_projections.device))
    
    pts_projections = project_points(cameras, pts_projections, points_are_already_in_view_space=True, znear=znear)  # (n_depths, N, 2)
    fov_mask = fov_mask & pts_projections.isfinite().all(dim=-1)  # (n_depths, N)
    pts_projections = pts_projections.nan_to_num(nan=0., posinf=0., neginf=0.)  # Between -/+ max(image_height, image_width) / min(image_height, image_width)
        
    factor = -1 * min(image_height, image_width)
    factors = torch.tensor([[[factor / image_width, factor / image_height]]]).to(pts.device)  # (1, 1, 2)
    pts_projections = pts_projections[..., :2] * factors  # (n_depths, N, 2)
    pts_projections = pts_projections.view(n_depths, -1, 1, 2)  # Between -1 and 1
    
    x_shifts = torch.linspace(-(patch_size - 1) // 2, (patch_size - 1) // 2, points_per_patch_side, device=pts.device) / image_width * 2
    y_shifts = torch.linspace(-(patch_size - 1) // 2, (patch_size - 1) // 2, points_per_patch_side, device=pts.device) / image_height * 2
    shifts = torch.stack(torch.meshgrid(x_shifts, y_shifts, indexing='ij'), dim=-1)  # (points_per_patch_side, points_per_patch_side, 2)
    shifts = shifts.view(1, 1, -1, 2)  # (1, 1, points_per_patch_side^2, 2)
    
    patch_projections = pts_projections + shifts  # (n_depths, N, points_per_patch_side^2, 2)

    depth_view = depthmap.reshape(n_depths, 1, image_height, image_width)  # (n_depths, 1, H, W)
    pts_view = depths_to_points_parallel(depth_view, cameras).view(n_depths, image_height, image_width, 3)  # (n_depths, H, W, 3)
    pts_view = pts_view.permute(0, 3, 1, 2)  # (n_depths, 3, H, W)
    
    map_pts = torch.nn.functional.grid_sample(
        input=pts_view,
        grid=patch_projections,
        mode='bilinear',
        padding_mode='border',  # 'reflection', 'zeros'
        align_corners=False,
    )  # (n_depths, 3, N, points_per_patch_side^2)
    map_pts = map_pts.permute(0, 2, 3, 1)  # (n_depths, N, points_per_patch_side^2, 3)
    # fov_mask = (map_pts > 0.).all(dim=-1) & fov_mask
    # map_pts = map_pts * fov_mask[..., None]
    
    return map_pts, fov_mask


def sample_points_in_gaussians(means, scales, quaternions, n_points):
    """Sample points in Gaussians.

    Args:
        means (torch.Tensor): Means of the Gaussians. Shape (n_gaussians, 3).
        scales (torch.Tensor): Scales of the Gaussians. Shape (n_gaussians, 3).
        quaternions (torch.Tensor): Quaternions of the Gaussians. Shape (n_gaussians, 4).
        n_points (int): Number of points to sample.

    Returns:
        torch.Tensor: Points sampled from the Gaussians. Shape (n_points, 3).
    """
    gaussian_idx = torch.randint(0, means.shape[0], (n_points,))
    pts = torch.randn(n_points, 3, device=means.device)
    pts = means[gaussian_idx] + quaternion_apply(quaternions[gaussian_idx], scales[gaussian_idx] * pts)
    return pts


def get_distance_to_charts(pts, chart_depths, cameras, use_signed_distance=False, charts_confs=None):
    """Get the distance to the closest chart for each point.
    
    Args:
        pts (torch.Tensor): Points to compute the distance to. Shape (n_points, 3).
        chart_depths (torch.Tensor): Chart depths. Shape (n_charts, h, w).

    Returns:
        torch.Tensor: Distance to the closest chart for each point. Shape (n_points,).
    """
    
    n_charts = chart_depths.shape[0]
    n_points = pts.shape[0]
    
    # pts_depths = transform_points_world_to_view(cameras, pts[None].repeat(n_charts, 1, 1))[..., 2]  # (n_charts, n_points)
    # pts_chart_depths, fov_mask = get_points_depth_in_depthmap_parallel(pts[None].repeat(n_charts, 1, 1), chart_depths, cameras)  # (n_charts, n_points)
    pts_depths = transform_points_world_to_view(cameras, pts[None])[..., 2]  # (n_charts, n_points)
    pts_chart_depths, fov_mask = get_points_depth_in_depthmap_parallel(pts[None], chart_depths, cameras)  # (n_charts, n_points)

    distances = pts_chart_depths - pts_depths
    if not use_signed_distance:
        distances = distances.abs()
    distances[~fov_mask] = 1e8
    
    if charts_confs is not None:
        pts_chart_confs, _ = get_points_depth_in_depthmap_parallel(pts[None], charts_confs, cameras)  # (n_charts, n_points)
        distances = distances * pts_chart_confs
    
    return distances, fov_mask
