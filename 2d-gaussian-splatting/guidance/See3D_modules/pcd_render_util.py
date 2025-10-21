import os
import sys
sys.path.append(os.getcwd())
import torch
import numpy as np
from PIL import Image
from utils.sh_utils import eval_sh
from tqdm import tqdm
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    PerspectiveCameras,
)
from matcha.dm_scene.cameras import convert_camera_from_gs_to_pytorch3d
import matplotlib.pyplot as plt
import cv2
import open3d as o3d
import trimesh

def save_tensor_as_pcd(pcd, path, pcd_colors=None):

    if isinstance(pcd, torch.Tensor):
        pcd = pcd.detach().cpu().numpy()
    pcd = trimesh.PointCloud(pcd)
    if pcd_colors is not None:
        if isinstance(pcd_colors, torch.Tensor):
            pcd_colors = pcd_colors.detach().cpu().numpy()
        pcd.colors = pcd_colors
    pcd.export(path)

def vis_depth(depth, cmap='plasma', save_path=None, valid_min=None, valid_max=None):
    """
    Visualize a depth map.

    Args:
        depth (np.ndarray): Depth map, shape (H, W).
        cmap (str): Colormap to use (default: 'plasma').
        valid_min (float or None): Minimum depth value to clip (default: None).
        valid_max (float or None): Maximum depth value to clip (default: None).
    """
    # Handle invalid values (optional)
    depth_vis = depth.copy()

    if valid_min is not None:
        depth_vis = np.maximum(depth_vis, valid_min)
    if valid_max is not None:
        depth_vis = np.minimum(depth_vis, valid_max)

    # Normalize to 0-1
    depth_vis = depth_vis - np.min(depth_vis)
    if np.max(depth_vis) > 0:
        depth_vis = depth_vis / np.max(depth_vis)

    # Apply colormap
    depth_vis = (depth_vis * 255).astype(np.uint8)
    depth_vis_color = cv2.applyColorMap(depth_vis, getattr(cv2, f'COLORMAP_{cmap.upper()}'))

    # Convert BGR to RGB for matplotlib
    depth_vis_color = cv2.cvtColor(depth_vis_color, cv2.COLOR_BGR2RGB)

    # Show
    plt.figure(figsize=(10, 5))
    plt.imshow(depth_vis_color)
    plt.axis('off')
    plt.title('Depth Visualization')
    # plt.show()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()

def downsample_pcd(pcds, pcd_colors, method='voxel'):
    # support method = 'voxel', 'uniform'

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcds.cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(pcd_colors.cpu().numpy())

    if method == 'voxel':
        voxel_size = 0.02
        downsampled_pcd = pcd.voxel_down_sample(voxel_size)
    elif method == 'uniform':
        downsampled_pcd = pcd.uniform_down_sample(every_k_points=10)
    else:
        raise ValueError(f'Unsupported method: {method}')
    
    dpcds = torch.from_numpy(np.asarray(downsampled_pcd.points)).to(pcds.device).to(torch.float32)
    dpcd_colors = torch.from_numpy(np.asarray(downsampled_pcd.colors)).to(pcd_colors.device).to(torch.float32)
    
    return dpcds, dpcd_colors

def create_edge_mask_from_depth(depth_map, edge_thickness=5, low_threshold=10, high_threshold=50, 
                               min_depth=None, max_depth=None, blur_size=0):
    """
    Create an edge mask from a depth map, where edge regions are 1 and other regions are 0.
    
    Args:
        depth_map: Input depth map (any numeric type)
        edge_thickness: Edge dilation degree (controls edge thickness)
        low_threshold: Lower threshold for Canny edge detection
        high_threshold: Upper threshold for Canny edge detection
        min_depth: Minimum value for depth normalization (None to use image minimum)
        max_depth: Maximum value for depth normalization (None to use image maximum)
        blur_size: Gaussian blur kernel size (0 means no blur)
        
    Returns:
        edge_mask: Binary mask image, edge regions are 1, other regions are 0
    """
    # Convert depth map to uint8
    depth_uint8 = depth_to_uint8(depth_map, min_depth, max_depth)
    
    # Optionally apply Gaussian blur to reduce noise
    if blur_size > 0:
        depth_uint8 = cv2.GaussianBlur(depth_uint8, (blur_size, blur_size), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(depth_uint8, low_threshold, high_threshold)
    # cv2.imwrite('mycode/filter_pcd/edges_depth.png', edges)
    
    # Create structural element for dilation operation
    # Use elliptical structural element for more natural effect
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_thickness, edge_thickness))
    
    # Dilate edges (making them thicker)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    # cv2.imwrite('mycode/filter_pcd/dilated_edges.png', dilated_edges)

    # Create binary mask (edges are 1, others are 0)
    edge_mask = dilated_edges.astype(np.uint8) / 255
    
    return edge_mask

def depth_to_uint8(depth_map, min_depth=None, max_depth=None):
    """
    Convert depth map to uint8 type for visualization and edge detection.
    
    Args:
        depth_map: Input depth map (any numeric type)
        min_depth: Minimum value for depth normalization (None to use image minimum)
        max_depth: Maximum value for depth normalization (None to use image maximum)
        
    Returns:
        depth_uint8: Depth map converted to uint8 (0-255)
    """
    # Create a copy to avoid modifying the original data
    depth = depth_map.copy()
    
    # Replace invalid values (NaN and Inf)
    mask = np.isfinite(depth)
    if not np.all(mask):  # Check if there are any invalid values
        valid_min = np.min(depth[mask]) if np.any(mask) else 0
        valid_max = np.max(depth[mask]) if np.any(mask) else 0
        # Replace NaN and Inf with valid values
        depth = np.nan_to_num(depth, nan=valid_min, posinf=valid_max, neginf=valid_min)
    
    # Determine depth range for normalization
    d_min = min_depth if min_depth is not None else depth.min()
    d_max = max_depth if max_depth is not None else depth.max()
    
    # Avoid division by zero
    if d_max > d_min:
        # Clip values to the specified range
        depth = np.clip(depth, d_min, d_max)
        # Normalize to 0-255 range
        depth_normalized = ((depth - d_min) / (d_max - d_min) * 255)
    else:
        depth_normalized = np.zeros_like(depth)
    
    # Convert to uint8
    depth_uint8 = depth_normalized.astype(np.uint8)
    
    return depth_uint8

def filter_pcd_by_edge(charts_points, pcd_colors, training_depths):

    filtered_charts_points = []
    filtered_pcd_colors = []
    for idx in range(charts_points.shape[0]):
        # depth_path = f'{training_depth_path}/depth_{idx:05d}.tiff'
        # depth_map = np.array(Image.open(depth_path))
        depth_map = training_depths[idx]
        edge_mask = create_edge_mask_from_depth(depth_map, edge_thickness=20, low_threshold=10, high_threshold=50, blur_size=0)
        edge_mask = (edge_mask > 0.5).reshape(-1)

        filtered_charts_points.append(charts_points[idx][~edge_mask])
        filtered_pcd_colors.append(pcd_colors[idx][~edge_mask])

    filtered_charts_points = torch.cat(filtered_charts_points, dim=0)
    filtered_pcd_colors = torch.cat(filtered_pcd_colors, dim=0)

    return filtered_charts_points, filtered_pcd_colors

def setup_renderer(cameras, image_size, radius=0.01, points_per_pixel=10):
    """
    Set up point cloud renderer
    
    Args:
        cameras: PyTorch3D camera object
        image_size: Size of rendered image in (H, W) format
        radius: Point radius
        points_per_pixel: Maximum number of points to render per pixel
        
    Returns:
        renderer: Configured PyTorch3D point cloud renderer
    """
    # Set up rasterization parameters
    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius=radius,
        points_per_pixel=points_per_pixel,
        bin_size=0
    )
    
    # Create renderer
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
        compositor=AlphaCompositor()
    )
    
    return renderer

def fov_to_focal_length(fov, image_size):
    """
    Convert field of view to focal length
    
    Args:
        fov: Field of view in radians (a scalar)
        image_size: Size of the image dimension (width for fovx, height for fovy)
        
    Returns:
        focal_length: Focal length in pixels
    """
    # Calculate focal length: f = (image_size/2) / tan(fov/2)
    return (image_size / 2) / torch.tan(fov / 2)

def convert_camera_params(fovx_deg_list, fovy_deg_list, poses, image_size, device):
    """
    Convert camera intrinsic and pose matrices to PyTorch3D camera objects
    
    Args:
        fovx_list: List of horizontal field of view angles in degrees for each camera
        fovy_list: List of vertical field of view angles in degrees for each camera
        poses: [M, 4, 4] camera-to-world transformation matrices
        image_size: Rendered image size in (H, W) format
        device: Computation device
        
    Returns:
        cameras: PyTorch3D camera object
    """
    # Extract camera parameters
    M = poses.shape[0]  # Number of cameras
    H, W = image_size

    fovxs = torch.deg2rad(torch.tensor(fovx_deg_list, device=device))
    fovys = torch.deg2rad(torch.tensor(fovy_deg_list, device=device))

    focal_x = fov_to_focal_length(fovxs, W)
    focal_y = fov_to_focal_length(fovys, H)
    focal_lengths = torch.stack([focal_x, focal_y], dim=-1)

    principal_points = torch.tensor([[W/2, H/2]], device=device).expand(M, -1)
    
    # Extract rotation and translation from c2w matrices
    R = poses[:, :3, :3]  # [M, 3, 3]

    temp_pose = torch.linalg.inv(poses[0])
    T = (temp_pose[:3, 3]).unsqueeze(0).expand(M, -1)
    
    # PyTorch3D uses different coordinate system convention, need to convert
    # Convert from OpenCV/COLMAP coordinate system (right-down-forward) to PyTorch3D coordinate system (left-up-forward)
    R_convert = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], device=device).float()
    R_converted = R @ R_convert

    T_converted = T @ R_convert

    # Create PyTorch3D camera
    cameras = PerspectiveCameras(
        focal_length=focal_lengths,
        principal_point=principal_points,
        R=R_converted,
        T=T_converted,
        in_ndc=False,
        image_size=[image_size] * M,  # Use the same image size for each camera
        device=device
    )
    
    return cameras

def pcd_render_multiview(gaussians, fovx_deg_list, fovy_deg_list, poses, image_size=(512, 512), radius=0.01, points_per_pixel=10, device='cuda'):
    """
    Render point cloud from multiple viewpoints, processing one view at a time
    
    Args:
        gaussians: GaussianModel
        fovx_deg_list: List of horizontal field of view angles in degrees for each camera
        fovy_deg_list: List of vertical field of view angles in degrees for each camera
        poses: [M, 4, 4] Camera pose matrices (camera-to-world)
        image_size: Rendered image size in (H, W) format
        radius: Point radius
        points_per_pixel: Maximum number of points to render per pixel
        device: Computation device
        
    Returns:
        images: [M, H, W, 3] Rendered RGB images
        view_masks: [M, H, W, 1] Rendered mask images
    """
    pcds = gaussians.get_xyz

    # Ensure inputs are tensors
    if not isinstance(poses, torch.Tensor):
        poses = torch.tensor(poses, dtype=torch.float32, device=device)
    else:
        poses = poses.to(device)
    
    # Process one camera at a time to avoid batch dimension issues
    M = poses.shape[0]
    images_list = []
    masks_list = []
    
    for i in tqdm(range(M)):
        # Get single camera parameters
        fovx_deg_i = fovx_deg_list[i:i+1]
        fovy_deg_i = fovy_deg_list[i:i+1]
        pose_i = poses[i:i+1].to(device)

        # camera_center = pose_i[0, :3, 3]
        temp_pose = torch.linalg.inv(pose_i[0])
        camera_center = temp_pose[:3, 3]
        shs_view = gaussians.get_features.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree+1)**2)
        dir_pp = (gaussians.get_xyz - camera_center.repeat(gaussians.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized)
        pcd_colors = torch.clamp_min(sh2rgb + 0.5, 0.0)
        
        # Create camera
        cameras = convert_camera_params(fovx_deg_i, fovy_deg_i, pose_i, image_size, device)
        
        # Create renderer
        renderer = setup_renderer(cameras, image_size, radius, points_per_pixel)
        
        # Create point cloud (for a single view)
        point_cloud = Pointclouds(points=[pcds], features=[pcd_colors])
        
        # Render RGB image
        image = renderer(point_cloud)
        images_list.append(image)
        
        # Render mask
        white_colors = torch.ones_like(pcd_colors)
        point_cloud_mask = Pointclouds(points=[pcds], features=[white_colors])
        mask = renderer(point_cloud_mask)
        masks_list.append(mask)
    
    # Concatenate results
    images = torch.cat(images_list, dim=0)
    view_masks = torch.cat(masks_list, dim=0)
    
    return images, view_masks

# use init_pcd_render_multiview instead
def ori_init_pcd_render_multiview(pcds, pcd_colors, fovx_deg_list, fovy_deg_list, poses, image_size=(512, 512), radius=0.01, points_per_pixel=10, device='cuda'):
    """
    Render point cloud from multiple viewpoints, processing one view at a time
    
    Args:
        pcds: [N, 3] point cloud
        pcd_colors: [N, 3] point cloud color
        fovx_deg_list: List of horizontal field of view angles in degrees for each camera
        fovy_deg_list: List of vertical field of view angles in degrees for each camera
        poses: [M, 4, 4] Camera pose matrices (camera-to-world)
        image_size: Rendered image size in (H, W) format
        radius: Point radius
        points_per_pixel: Maximum number of points to render per pixel
        device: Computation device
        
    Returns:
        images: [M, H, W, 3] Rendered RGB images
        view_masks: [M, H, W, 1] Rendered mask images
    """

    # Ensure inputs are tensors
    if not isinstance(poses, torch.Tensor):
        poses = torch.tensor(poses, dtype=torch.float32, device=device)
    else:
        poses = poses.to(device)
    
    # Process one camera at a time to avoid batch dimension issues
    M = poses.shape[0]
    images_list = []
    masks_list = []
    
    for i in tqdm(range(M)):
        # Get single camera parameters
        fovx_deg_i = fovx_deg_list[i:i+1]
        fovy_deg_i = fovy_deg_list[i:i+1]
        pose_i = poses[i:i+1].to(device)
        
        # Create camera
        cameras = convert_camera_params(fovx_deg_i, fovy_deg_i, pose_i, image_size, device)
        
        # Create renderer
        renderer = setup_renderer(cameras, image_size, radius, points_per_pixel)
        
        # Create point cloud (for a single view)
        point_cloud = Pointclouds(points=[pcds], features=[pcd_colors])
        
        # Render RGB image
        image = renderer(point_cloud)
        images_list.append(image)
        
        # Render mask
        white_colors = torch.ones_like(pcd_colors)
        point_cloud_mask = Pointclouds(points=[pcds], features=[white_colors])
        mask = renderer(point_cloud_mask)
        masks_list.append(mask)
    
    # Concatenate results
    images = torch.cat(images_list, dim=0)
    view_masks = torch.cat(masks_list, dim=0)
    
    return images, view_masks

def init_pcd_render_multiview(pcds, pcd_colors, gs_cameras, gs_depths=None, depth_threshold=0.1, image_size=(512, 512), radius=0.01, points_per_pixel=10, device='cuda'):
    """
    Render point cloud from multiple viewpoints, processing one view at a time
    
    Args:
        pcds: [N, 3] point cloud
        pcd_colors: [N, 3] point cloud color
        gs_depths: [M, H, W] depth map (optional for filtering)
        gs_cameras: List of GSCamera
        image_size: Rendered image size in (H, W) format
        radius: Point radius
        points_per_pixel: Maximum number of points to render per pixel
        device: Computation device
        
    Returns:
        images: [M, H, W, 3] Rendered RGB images
        view_masks: [M, H, W, 1] Rendered mask images
    """
    # Convert GSCamera to PyTorch3D camera
    for gs_cam in gs_cameras:
        gs_cam.R = torch.from_numpy(gs_cam.R).to(device)
        gs_cam.T = torch.from_numpy(gs_cam.T).to(device)
    p3d_cameras = convert_camera_from_gs_to_pytorch3d(gs_cameras, device=device)

    # Process one camera at a time to avoid batch dimension issues
    M = len(p3d_cameras)
    images_list = []
    masks_list = []
    
    for i in tqdm(range(M)):
        # Get single camera parameters
        p3d_camera = p3d_cameras[i]
        
        # Create renderer
        renderer = setup_renderer(p3d_camera, image_size, radius, points_per_pixel)
        
        # Create point cloud (for a single view)
        point_cloud = Pointclouds(points=[pcds], features=[pcd_colors])
        
        # Render RGB image
        image = renderer(point_cloud)

        # Render mask
        white_colors = torch.ones_like(pcd_colors)
        point_cloud_mask = Pointclouds(points=[pcds], features=[white_colors])
        mask = renderer(point_cloud_mask)
        
        if gs_depths is not None:
            valid_mask = mask[0, :, :, 0] > 0

            reference_depth = gs_depths[i]
            if isinstance(reference_depth, np.ndarray):
                reference_depth = torch.from_numpy(reference_depth).to(device)

            # Get camera center and compute point distances for depth
            cam_center = p3d_camera.get_camera_center()
            points_cam = p3d_camera.get_world_to_view_transform().transform_points(pcds)
            
            # Use Z coordinate as depth
            depth_values = points_cam[..., 2:3]  # Get Z coordinates
            
            # Normalize depths to [0,1] for visualization - not needed for comparison
            depth_colors = depth_values.repeat(1, 3)  # Repeat to make RGB compatible
            
            # Create point cloud for depth rendering
            point_cloud_depth = Pointclouds(points=[pcds], features=[depth_colors])
            
            # Render depth image
            rendered_depth = renderer(point_cloud_depth)
            
            # Extract the depth channel (all channels should be the same)
            rendered_depth_map = rendered_depth[0, :, :, 0]  # [H, W]

            # # save rendered depth map
            # save_root_path = 'output/replica-5-views/scan6/free_gaussians/charts_pcd_render/rendered_depth_map/'
            # os.makedirs(save_root_path, exist_ok=True)
            # save_path = os.path.join(save_root_path, f'rendered_depth_map_{i:06d}.png')
            # vis_depth(rendered_depth_map.cpu().numpy(), cmap='viridis', save_path=save_path, valid_min=0, valid_max=10)

            depth_mask = torch.ones_like(rendered_depth_map, dtype=torch.bool)
            where_valid = valid_mask & (reference_depth > 0)

            depth_diff = torch.abs(rendered_depth_map[where_valid] - reference_depth[where_valid])
            relative_diff = depth_diff / (reference_depth[where_valid] + 1e-6)
            consistent_depths = relative_diff < depth_threshold

            depth_mask[where_valid] = consistent_depths
            rgb_depth_mask = depth_mask.unsqueeze(-1).repeat(1, 1, 3)

            image[0][~rgb_depth_mask] = 0.0  # Set to black
            mask[0][~rgb_depth_mask] = 0.0  # Set to transparent
    
        images_list.append(image)
        masks_list.append(mask)

    # Concatenate results
    images = torch.cat(images_list, dim=0)
    view_masks = torch.cat(masks_list, dim=0)
    
    return images, view_masks

def save_rendered_images(images, output_dir, prefix, image_format="png"):
    """
    Save rendered images
    
    Args:
        images: [M, H, W, 3] Rendered images
        output_dir: Output directory
        prefix: Filename prefix
        image_format: Image format
    """

    os.makedirs(output_dir, exist_ok=True)
    
    # Convert tensors to numpy arrays and save
    for i, img in enumerate(images):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        
        # Ensure values are in [0, 1] range
        img = np.clip(img, 0, 1)
        
        # Convert to [0, 255] range as uint8
        img_uint8 = (img * 255).astype(np.uint8)
        
        # Save image
        img_pil = Image.fromarray(img_uint8)
        img_pil.save(os.path.join(output_dir, f"{prefix}{i:06d}.{image_format}"))

