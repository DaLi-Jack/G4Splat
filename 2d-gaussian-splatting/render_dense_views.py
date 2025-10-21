import torch
from scene import Scene, GaussianModel
from scene.dataset_readers import load_see3d_cameras
import os
import sys
sys.path.append(os.getcwd())
import json
import numpy as np
import shutil
from gaussian_renderer import render
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args

from utils.render_utils import save_img_f32, save_img_u8
from tqdm import tqdm
from PIL import Image
import trimesh

from utils.general_utils import safe_state
from matcha.dm_scene.charts import depths_to_points_parallel


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", required=True, type=str)
    args = get_combined_args(parser)

    # Initialize system state (RNG)
    safe_state(False)

    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)

    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    train_viewpoints = scene.getTrainCameras().copy()
    input_view_num = len(train_viewpoints)

    # render train views
    alpha_vis_thresh = 0.99
    save_root_path = os.path.join(args.source_path, 'render-dense-train-views')
    if os.path.exists(save_root_path):
        shutil.rmtree(save_root_path)
    os.makedirs(save_root_path, exist_ok=True)
    train_view_depths = []
    render_save_root_path = os.path.join(save_root_path, 'dense-render')
    dense_rgb_save_root_path = os.path.join(save_root_path, 'dense-gt-rgb')
    os.makedirs(render_save_root_path, exist_ok=True)
    os.makedirs(dense_rgb_save_root_path, exist_ok=True)
    for idx, viewpoint in enumerate(train_viewpoints):
        render_pkg = render(viewpoint, gaussians, pipe, background)
        rgb = render_pkg['render']
        alpha = render_pkg['rend_alpha']
        depth = render_pkg['surf_depth']
        train_view_depths.append(depth[0].detach())

        gt_rgb = viewpoint.original_image.permute(1,2,0).detach().cpu().numpy()
        save_img_u8(gt_rgb, os.path.join(dense_rgb_save_root_path, f'rgb_frame{idx:06d}.png'))

        save_img_u8(rgb.permute(1,2,0).detach().cpu().numpy(), os.path.join(render_save_root_path, f'ori_warp_frame{idx:06d}.png'))
        save_img_f32(depth[0].detach().cpu().numpy(), os.path.join(render_save_root_path, f'depth_frame{idx:06d}.tiff'))
        # save .npy
        np.save(os.path.join(render_save_root_path, f'alpha_{idx:06d}.npy'), alpha[0].detach().cpu().numpy())
        alpha_vis_mask = alpha[0].detach().cpu().numpy() > alpha_vis_thresh

        none_visible_rate = 1 - alpha_vis_mask.sum() / (alpha_vis_mask.shape[0] * alpha_vis_mask.shape[1])
        save_img_u8(alpha_vis_mask, os.path.join(render_save_root_path, f'alpha_mask_frame{idx:06d}.png'))

        # filter rgb use alpha
        rgb_filtered = rgb.permute(1,2,0).detach().cpu().numpy() * alpha_vis_mask[:,:,None]
        save_img_u8(rgb_filtered, os.path.join(render_save_root_path, f'alpha_warp_frame{idx:06d}.png'))

        print(f'Dense view {idx} save done!')

    print('All dense views render done!')

    # save dense view points for global 3D plane merge
    dense_visible_depths = torch.stack(train_view_depths, dim=0)
    invalid_depth_mask = dense_visible_depths <= 1e-6
    dense_visible_depths[invalid_depth_mask] = 1e-3
    dense_visible_points = depths_to_points_parallel(dense_visible_depths, train_viewpoints)

    dense_visible_points = dense_visible_points.reshape(-1, 3)
    invalid_depth_mask_flatten = invalid_depth_mask.reshape(-1)
    dense_visible_points = dense_visible_points[~invalid_depth_mask_flatten]
    
    # save dense view points
    dense_visible_points_path = os.path.join(save_root_path, 'dense-view-points.ply')
    trimesh.PointCloud(dense_visible_points.cpu().numpy()).export(dense_visible_points_path)
    print(f'Saved dense view points to {dense_visible_points_path}')


