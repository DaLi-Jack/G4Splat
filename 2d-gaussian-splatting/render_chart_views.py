import torch
from scene.dataset_readers import load_cameras
import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import shutil
from argparse import ArgumentParser
from arguments import ModelParams

from utils.render_utils import save_img_f32, save_img_u8
from utils.general_utils import safe_state
from matcha.dm_scene.charts import load_charts_data, build_priors_from_charts_data, depths_to_points_parallel
from matcha.dm_utils.rendering import depth2normal_parallel
from guidance.cam_utils import build_visibility_masks

import cv2
import matplotlib.pyplot as plt
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

def get_surf_normal_parallel(views, depths):
    world_view_transforms = torch.stack([views[i].world_view_transform for i in range(len(views))])
    full_proj_transforms = torch.stack([views[i].full_proj_transform for i in range(len(views))])
    normals = depth2normal_parallel(depths, world_view_transforms=world_view_transforms, full_proj_transforms=full_proj_transforms)
    normals = normals.permute(0, 3, 1, 2)
    return normals

def create_vis_frequency_heatmap(rgb_image, match_mask, transparency = 0.8, colormap='viridis'):
    """
    Create a heatmap visualization overlaid on an RGB image based on match frequency.
    
    Args:
        rgb_image: The original RGB image (numpy array with shape [H, W, 3])
        match_mask: 2D array with same height and width as rgb_image, containing match frequency counts
        transparency: Transparency of the heatmap overlay (0.0 to 1.0)
        colormap: Matplotlib colormap name to use for the heatmap
    
    Returns:
        Combined visualization with heatmap overlaid on RGB image
    """
    # Ensure inputs have correct shapes
    assert rgb_image.shape[:2] == match_mask.shape, "RGB image and match mask must have same dimensions"

    match_mask = match_mask + 1e-4          # set base color
    
    # Normalize match frequency to 0-1 range for visualization
    if match_mask.max() > 0:
        normalized_mask = match_mask.astype(float) / match_mask.max()
    else:
        normalized_mask = match_mask.astype(float)
    
    # Get colormap from matplotlib
    colormap_func = plt.get_cmap(colormap)
    
    # Apply colormap to the normalized mask (returns RGBA)
    heatmap_colored = colormap_func(normalized_mask)
    
    # Convert to RGB with correct shape for blending
    heatmap_rgb = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
    
    # Ensure RGB image is in the right format
    if rgb_image.dtype != np.uint8:
        rgb_image = (rgb_image * 255).astype(np.uint8)
    
    # Blend the original image with the heatmap
    blended = rgb_image.copy()
    for i in range(3):
        blended[:, :, i] = rgb_image[:, :, i] * (1 - transparency) + heatmap_rgb[:, :, i] * transparency
    
    return blended.astype(np.uint8)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    parser.add_argument("--save_root_path", required=True, type=str)
    args = parser.parse_args()

    # Initialize system state (RNG)
    safe_state(False)

    train_viewpoints, _ = load_cameras(model.extract(args))

    # render train views
    train_save_root_path = os.path.join(args.source_path, 'render-charts-train-views')
    if os.path.exists(train_save_root_path):
        shutil.rmtree(train_save_root_path)
    os.makedirs(train_save_root_path, exist_ok=True)

    # vis charts data
    charts_data_path = os.path.join(args.source_path, 'charts_data.npz')
    charts_data = load_charts_data(charts_data_path)
    charts_priors = build_priors_from_charts_data(charts_data, train_viewpoints)
    charts_depths = charts_priors['depths']
    charts_depth_normals = get_surf_normal_parallel(train_viewpoints, charts_depths)            # normal from charts depth
    charts_prior_depths = charts_priors['prior_depths']                                         # depth-anything-v2 prior depth (mono depth + linear alignment)
    charts_confs = charts_priors['confs']
    charts_mono_normals = charts_priors['normals']                                              # normal from depth-anything-v2 (MAtCha use this as normal prior)
    charts_curvs = charts_priors['curvs']

    # # linear alignment prior pcds
    # charts_prior_pcds = depths_to_points_parallel(charts_prior_depths, train_viewpoints)
    # for idx in range(len(charts_prior_pcds)):
    #     charts_prior_pcd = charts_prior_pcds[idx]
    #     save_tensor_as_pcd(charts_prior_pcd, os.path.join(train_save_root_path, f'prior_pcd_frame{idx:06d}.ply'))

    charts_points = depths_to_points_parallel(charts_depths, train_viewpoints)

    # save charts pcd
    save_tensor_as_pcd(charts_points.reshape(-1, 3), os.path.join(args.source_path, 'chart_pcd.ply'))

    for idx in range(len(train_viewpoints)):
        vis_charts_conf = charts_confs[idx][0].detach().cpu().numpy()
        plt.imsave(os.path.join(train_save_root_path, f'charts_conf_frame{idx:06d}.png'), vis_charts_conf, cmap='viridis')

        vis_charts_depth = charts_depths[idx][0].detach().cpu().numpy()
        plt.imsave(os.path.join(train_save_root_path, f'depth_frame{idx:06d}.png'), vis_charts_depth, cmap='viridis')

        # save charts depth as .tiff
        save_img_f32(charts_depths[idx][0].detach().cpu().numpy(), os.path.join(train_save_root_path, f'depth_frame{idx:06d}.tiff'))

        # save prior depth as .tiff (depth anything v2 + linear alignment)
        save_img_f32(charts_prior_depths[idx][0].detach().cpu().numpy(), os.path.join(train_save_root_path, f'mono_depth_frame{idx:06d}.tiff'))
        plt.imsave(os.path.join(train_save_root_path, f'mono_depth_frame{idx:06d}.png'), charts_prior_depths[idx][0].detach().cpu().numpy(), cmap='viridis')

        # get normal in world coordinate
        vis_charts_mono_normal_world = charts_mono_normals[idx].permute(1,2,0).detach().cpu().numpy()
        # save world normal as npy
        np.save(os.path.join(train_save_root_path, f'mono_normal_world_frame{idx:06d}.npy'), vis_charts_mono_normal_world)
        # save world normal as png
        save_img_u8(vis_charts_mono_normal_world * 0.5 + 0.5, os.path.join(train_save_root_path, f'mono_normal_world_frame{idx:06d}.png'))

        # get normal in camera coordinate
        vis_charts_mono_normal = (charts_mono_normals[idx].permute(1,2,0) @ train_viewpoints[idx].world_view_transform[:3,:3]).permute(2,0,1)
        vis_charts_mono_normal = vis_charts_mono_normal.permute(1,2,0).detach().cpu().numpy()
        # save normal as npy
        np.save(os.path.join(train_save_root_path, f'mono_normal_frame{idx:06d}.npy'), vis_charts_mono_normal)
        # save normal as png
        save_img_u8(vis_charts_mono_normal * 0.5 + 0.5, os.path.join(train_save_root_path, f'mono_normal_frame{idx:06d}.png'))

        # get normal from charts depth in world coordinate
        vis_charts_depth_normal_world = charts_depth_normals[idx].permute(1,2,0).detach().cpu().numpy()
        np.save(os.path.join(train_save_root_path, f'depth_normal_world_frame{idx:06d}.npy'), vis_charts_depth_normal_world)
        save_img_u8(vis_charts_depth_normal_world * 0.5 + 0.5, os.path.join(train_save_root_path, f'depth_normal_world_frame{idx:06d}.png'))
        
        # get normal from charts depth in camera coordinate
        vis_charts_depth_normal = (charts_depth_normals[idx].permute(1,2,0) @ train_viewpoints[idx].world_view_transform[:3,:3]).permute(2,0,1)
        vis_charts_depth_normal = vis_charts_depth_normal.permute(1,2,0).detach().cpu().numpy()
        # save normal as npy
        np.save(os.path.join(train_save_root_path, f'depth_normal_frame{idx:06d}.npy'), vis_charts_depth_normal)
        # save normal as png
        save_img_u8(vis_charts_depth_normal * 0.5 + 0.5, os.path.join(train_save_root_path, f'depth_normal_frame{idx:06d}.png'))

        vis_charts_curv = charts_curvs[idx][0].detach().cpu().numpy()
        plt.imsave(os.path.join(train_save_root_path, f'curv_frame{idx:06d}.png'), vis_charts_curv, cmap='viridis')

    visibility_times_masks = build_visibility_masks(train_viewpoints, charts_depths, charts_points, mast3r_matching=None, return_origin_masks=True)
    vis_points = []
    for idx in range(len(visibility_times_masks)):
        rgb_name = train_viewpoints[idx].image_name
        rgb_path = os.path.join(args.source_path, f'images/{rgb_name}.png')
        dst_rgb_path = os.path.join(train_save_root_path, f'rgb_frame{idx:06d}.png')
        shutil.copy(rgb_path, dst_rgb_path)

        rgb_image = cv2.imread(rgb_path)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        match_mask = visibility_times_masks[idx][0].detach().cpu().numpy()
        np.save(os.path.join(train_save_root_path, f'visibility_frame{idx:06d}.npy'), match_mask)
        blended = create_vis_frequency_heatmap(rgb_image, match_mask)
        cv2.imwrite(os.path.join(train_save_root_path, f'visibility_frame{idx:06d}.png'), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

        ori_points = charts_points[idx].detach().cpu().numpy()
        vis_mask = (match_mask > 0.5).reshape(-1)
        vis_points.append(ori_points[vis_mask])
    vis_points = np.concatenate(vis_points, axis=0)
    save_tensor_as_pcd(vis_points, os.path.join(train_save_root_path, 'vis_points.ply'))

    # NOTE: copy train views render results to save_root_path
    os.makedirs(args.save_root_path, exist_ok=True)
    for idx in range(len(train_viewpoints)):
        # rgb
        shutil.copy(os.path.join(train_save_root_path, f'rgb_frame{idx:06d}.png'), os.path.join(args.save_root_path, f'rgb_frame{idx:06d}.png'))
        
        # depth
        shutil.copy(os.path.join(train_save_root_path, f'depth_frame{idx:06d}.tiff'), os.path.join(args.save_root_path, f'depth_frame{idx:06d}.tiff'))
        shutil.copy(os.path.join(train_save_root_path, f'depth_frame{idx:06d}.png'), os.path.join(args.save_root_path, f'depth_frame{idx:06d}.png'))
        
        shutil.copy(os.path.join(train_save_root_path, f'mono_depth_frame{idx:06d}.tiff'), os.path.join(args.save_root_path, f'mono_depth_frame{idx:06d}.tiff'))
        shutil.copy(os.path.join(train_save_root_path, f'mono_depth_frame{idx:06d}.png'), os.path.join(args.save_root_path, f'mono_depth_frame{idx:06d}.png'))

        # normal
        shutil.copy(os.path.join(train_save_root_path, f'depth_normal_world_frame{idx:06d}.npy'), os.path.join(args.save_root_path, f'depth_normal_world_frame{idx:06d}.npy'))
        shutil.copy(os.path.join(train_save_root_path, f'depth_normal_world_frame{idx:06d}.png'), os.path.join(args.save_root_path, f'depth_normal_world_frame{idx:06d}.png'))

        shutil.copy(os.path.join(train_save_root_path, f'mono_normal_world_frame{idx:06d}.npy'), os.path.join(args.save_root_path, f'mono_normal_world_frame{idx:06d}.npy'))
        shutil.copy(os.path.join(train_save_root_path, f'mono_normal_world_frame{idx:06d}.png'), os.path.join(args.save_root_path, f'mono_normal_world_frame{idx:06d}.png'))
        
        shutil.copy(os.path.join(train_save_root_path, f'mono_normal_frame{idx:06d}.npy'), os.path.join(args.save_root_path, f'mono_normal_frame{idx:06d}.npy'))
        shutil.copy(os.path.join(train_save_root_path, f'mono_normal_frame{idx:06d}.png'), os.path.join(args.save_root_path, f'mono_normal_frame{idx:06d}.png'))
        
        # visibility
        shutil.copy(os.path.join(train_save_root_path, f'visibility_frame{idx:06d}.npy'), os.path.join(args.save_root_path, f'visibility_frame{idx:06d}.npy'))
        shutil.copy(os.path.join(train_save_root_path, f'visibility_frame{idx:06d}.png'), os.path.join(args.save_root_path, f'visibility_frame{idx:06d}.png'))

    print(f'Train views render done!')

