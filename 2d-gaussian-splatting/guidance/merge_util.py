import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), '2d-gaussian-splatting'))
from utils.general_utils import seed_everything
from scene.dataset_readers import load_see3d_cameras
import numpy as np
from PIL import Image
from argparse import ArgumentParser
import shutil
import json


def run_command_safe(command):
    print(f"Running command: {command}")
    exit_code = os.system(command)
    if exit_code != 0:
        print("Command failed!")
        sys.exit(1)
    else:
        print("Command succeeded!")

def replace_inpaint_results(warp_root_dir, inpaint_root_dir, save_root_dir):
    os.makedirs(save_root_dir, exist_ok=True)
    inpaint_img_list = os.listdir(inpaint_root_dir)
    inpaint_img_list = [img for img in inpaint_img_list if '.png' in img]
    img_num = len(inpaint_img_list)
    for idx in range(img_num):
        gs_render_img_path = os.path.join(warp_root_dir, f'warp_frame{idx:06d}.png')
        mask_img_path = os.path.join(warp_root_dir, f'mask_frame{idx:06d}.png')
        inpaint_img_path = os.path.join(inpaint_root_dir, f'predict_warp_frame{idx:06d}.png')

        gs_render_img = Image.open(gs_render_img_path)
        mask_img = Image.open(mask_img_path)
        inpaint_img = Image.open(inpaint_img_path)

        mask_map = np.array(mask_img) / 255
        gs_render_img = np.array(gs_render_img)
        inpaint_img = np.array(inpaint_img)

        save_img = inpaint_img.copy()
        save_img[mask_map == 1] = gs_render_img[mask_map == 1]          # NOTE: visible part is gs_render_img, invisible part is inpaint_img
        save_img = Image.fromarray(save_img)
        save_img.save(os.path.join(save_root_dir, f'predict_warp_frame{idx:06d}.png'))

        print(f'Inpaint {idx} replace done!')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--source_path', type=str)
    parser.add_argument('--plane_root_dir', type=str)
    parser.add_argument("--see3d_stage", required=True, type=int)
    parser.add_argument("--none_replace", action='store_true')
    parser.add_argument("--anchor_view_id_json_path", type=str, required=True)
    args = parser.parse_args()

    seed_everything()

    see3d_root_dir = os.path.join(args.source_path, 'see3d_render')
    cur_see3d_root_dir = os.path.join(see3d_root_dir, f'stage{args.see3d_stage}')
    inpaint_root_dir = os.path.join(cur_see3d_root_dir, 'select-gs-inpainted')
    save_root_dir = os.path.join(cur_see3d_root_dir, 'select-gs-inpainted-merged')

    warp_root_dir = os.path.join(cur_see3d_root_dir, 'select-gs')
    # 1. replace inpaint results
    if not args.none_replace:
        replace_inpaint_results(warp_root_dir, inpaint_root_dir, save_root_dir)
        print(f'See3D stage {args.see3d_stage} replace inpaint results done!')

    # 2. copy inpaint results to all inpaint folder (NOTE: begin_idx is id in all inpaint images)
    all_inpaint_image_dir = os.path.join(see3d_root_dir, 'inpainted_images')
    if not os.path.exists(all_inpaint_image_dir):
        os.makedirs(all_inpaint_image_dir, exist_ok=True)
        begin_idx = 0
    else:
        begin_idx = len(os.listdir(all_inpaint_image_dir))

    cur_result_dir = save_root_dir if not args.none_replace else inpaint_root_dir
    inpaint_img_list = os.listdir(cur_result_dir)
    inpaint_img_list = sorted(inpaint_img_list)
    for result_img_name in inpaint_img_list:
        inpaint_img_path = os.path.join(cur_result_dir, result_img_name)
        shutil.copy(inpaint_img_path, os.path.join(all_inpaint_image_dir, f'predict_warp_frame{begin_idx:06d}.png'))
        begin_idx += 1
    print(f'See3D stage {args.see3d_stage} copy inpaint results to all inpaint folder done!')

    # 3. merge novel cameras
    see3d_cam_path = os.path.join(see3d_root_dir, 'see3d_cameras.npz')
    if os.path.exists(see3d_cam_path):
        pre_see3d_cameras = np.load(see3d_cam_path)
        pre_see3d_cameras = dict(pre_see3d_cameras)                 # NOTE: convert npz to dict
        pre_see3d_views = pre_see3d_cameras['n_views']

        os.remove(see3d_cam_path)
    else:
        pre_see3d_cameras = {}
        pre_see3d_views = 0

    cur_see3d_cam_path = os.path.join(cur_see3d_root_dir, f'stage{args.see3d_stage}_see3d_cameras.npz')
    cur_see3d_cameras = np.load(cur_see3d_cam_path)
    cur_see3d_views = cur_see3d_cameras['n_views']

    for i in range(cur_see3d_views):
        cur_id = i + pre_see3d_views
        pre_see3d_cameras[f'R_{cur_id:06d}'] = cur_see3d_cameras[f'R_{i:06d}']
        pre_see3d_cameras[f'T_{cur_id:06d}'] = cur_see3d_cameras[f'T_{i:06d}']
        pre_see3d_cameras[f'FoVx_{cur_id:06d}'] = cur_see3d_cameras[f'FoVx_{i:06d}']
        pre_see3d_cameras[f'FoVy_{cur_id:06d}'] = cur_see3d_cameras[f'FoVy_{i:06d}']
        pre_see3d_cameras[f'image_width_{cur_id:06d}'] = cur_see3d_cameras[f'image_width_{i:06d}']
        pre_see3d_cameras[f'image_height_{cur_id:06d}'] = cur_see3d_cameras[f'image_height_{i:06d}']

    pre_see3d_cameras['n_views'] = cur_see3d_views + pre_see3d_views
    if 'train_views' not in pre_see3d_cameras:
        pre_see3d_cameras['train_views'] = cur_see3d_cameras['train_views']
    np.savez(see3d_cam_path, **pre_see3d_cameras)
    print(f'See3D stage {args.see3d_stage} merge novel cameras done!')

    # 4. merge geometry cues (NOTE: begin_plane_idx is id in all plane images, input views + novel views)
    plane_root_dir = args.plane_root_dir
    cur_plane_root_dir = os.path.join(cur_see3d_root_dir, 'select-gs-planes')
    if os.path.exists(plane_root_dir):
        plane_file_list = os.listdir(plane_root_dir)
        plane_rgb_list = [file for file in plane_file_list if 'rgb_frame' in file]
        begin_plane_idx = len(plane_rgb_list)
    else:
        os.makedirs(plane_root_dir, exist_ok=True)
        begin_plane_idx = 0

    anchor_view_id_list = []
    for i in range(cur_see3d_views):
        # rgb
        shutil.copy(os.path.join(cur_plane_root_dir, f'rgb_frame{i:06d}.png'), os.path.join(plane_root_dir, f'rgb_frame{begin_plane_idx:06d}.png'))

        # depth
        shutil.copy(os.path.join(cur_plane_root_dir, f'depth_frame{i:06d}.tiff'), os.path.join(plane_root_dir, f'depth_frame{begin_plane_idx:06d}.tiff'))
        shutil.copy(os.path.join(cur_plane_root_dir, f'depth_frame{i:06d}.png'), os.path.join(plane_root_dir, f'depth_frame{begin_plane_idx:06d}.png'))

        shutil.copy(os.path.join(cur_plane_root_dir, f'mono_depth_frame{i:06d}.tiff'), os.path.join(plane_root_dir, f'mono_depth_frame{begin_plane_idx:06d}.tiff'))
        shutil.copy(os.path.join(cur_plane_root_dir, f'mono_depth_frame{i:06d}.png'), os.path.join(plane_root_dir, f'mono_depth_frame{begin_plane_idx:06d}.png'))

        # normal
        shutil.copy(os.path.join(cur_plane_root_dir, f'depth_normal_world_frame{i:06d}.npy'), os.path.join(plane_root_dir, f'depth_normal_world_frame{begin_plane_idx:06d}.npy'))
        shutil.copy(os.path.join(cur_plane_root_dir, f'depth_normal_world_frame{i:06d}.png'), os.path.join(plane_root_dir, f'depth_normal_world_frame{begin_plane_idx:06d}.png'))

        shutil.copy(os.path.join(cur_plane_root_dir, f'mono_normal_world_frame{i:06d}.npy'), os.path.join(plane_root_dir, f'mono_normal_world_frame{begin_plane_idx:06d}.npy'))
        shutil.copy(os.path.join(cur_plane_root_dir, f'mono_normal_world_frame{i:06d}.png'), os.path.join(plane_root_dir, f'mono_normal_world_frame{begin_plane_idx:06d}.png'))

        shutil.copy(os.path.join(cur_plane_root_dir, f'mono_normal_frame{i:06d}.npy'), os.path.join(plane_root_dir, f'mono_normal_frame{begin_plane_idx:06d}.npy'))
        shutil.copy(os.path.join(cur_plane_root_dir, f'mono_normal_frame{i:06d}.png'), os.path.join(plane_root_dir, f'mono_normal_frame{begin_plane_idx:06d}.png'))

        # visibility
        shutil.copy(os.path.join(cur_plane_root_dir, f'visibility_frame{i:06d}.npy'), os.path.join(plane_root_dir, f'visibility_frame{begin_plane_idx:06d}.npy'))
        shutil.copy(os.path.join(cur_plane_root_dir, f'visibility_frame{i:06d}.png'), os.path.join(plane_root_dir, f'visibility_frame{begin_plane_idx:06d}.png'))

        # 2D plane
        shutil.copy(os.path.join(cur_plane_root_dir, f'plane_mask_frame{i:06d}.npy'), os.path.join(plane_root_dir, f'plane_mask_frame{begin_plane_idx:06d}.npy'))
        shutil.copy(os.path.join(cur_plane_root_dir, f'plane_vis_frame{i:06d}.png'), os.path.join(plane_root_dir, f'plane_vis_frame{begin_plane_idx:06d}.png'))

        anchor_view_id_list.append(begin_plane_idx)

        begin_plane_idx += 1

    # copy need inpaint views points
    shutil.copy(os.path.join(cur_see3d_root_dir, f'stage{args.see3d_stage}_need_inpaint_views_points.ply'), os.path.join(plane_root_dir, f'stage{args.see3d_stage}_need_inpaint_views_points.ply'))

    with open(args.anchor_view_id_json_path, 'w') as f:
        json.dump(anchor_view_id_list, f)

    print(f'See3D stage {args.see3d_stage} merge geometry cues done!')


