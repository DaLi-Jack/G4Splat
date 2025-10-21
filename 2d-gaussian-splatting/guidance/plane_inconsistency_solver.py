import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), '2d-gaussian-splatting'))
from scene.dataset_readers import load_see3d_cameras
import torch
import numpy as np
from argparse import ArgumentParser
from arguments import ModelParams
from scene.dataset_readers import load_cameras
from utils.general_utils import safe_state, seed_everything
from guidance.cam_utils import project_points_to_image
import trimesh
from PIL import Image
import time
import json


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--plane_root_path", type=str, required=True)
    parser.add_argument("--see3d_root_path", type=str, default=None)
    parser.add_argument("--anchor_view_id_json_path", type=str, default=None)
    model = ModelParams(parser, sentinel=True)
    args = parser.parse_args()

    print('NOTE: Using training views from data_path')
    # Initialize system state (RNG)
    safe_state(False)

    t1 = time.time()

    input_viewpoints, _ = load_cameras(model.extract(args))

    if args.see3d_root_path is None:
        print('Not need to generate confident maps for see3d views!')
        input_view_num = len(input_viewpoints)
        view_id = 0
        temp_refine_depth_path = os.path.join(args.plane_root_path, f'refine_depth_frame{view_id:06d}.tiff')
        temp_refine_depth = Image.open(temp_refine_depth_path)
        temp_refine_depth = np.array(temp_refine_depth)
        H, W = temp_refine_depth.shape
        confident_map_np = np.ones((H, W), dtype=np.uint8)
        confident_map_vis = (confident_map_np * 255).astype(np.uint8)
        for view_id in range(input_view_num):
            confident_map_img_path = os.path.join(args.plane_root_path, f'confident_map_frame{view_id:06d}.png')
            Image.fromarray(confident_map_vis).save(confident_map_img_path)

        exit()

    see3d_root_path = args.see3d_root_path
    plane_root_path = args.plane_root_path
    depth_threshold = 0.1
    color_eps = 5  # Color matching tolerance
    with open(args.anchor_view_id_json_path, 'r') as f:
        anchor_view_id_list = json.load(f)
    anchor_view_id_list = [int(i) for i in anchor_view_id_list]         # id in all training views, include input views and see3d views

    # load see3d cameras
    camera_path = os.path.join(see3d_root_path, 'see3d_cameras.npz')
    inpaint_root_dir = os.path.join(see3d_root_path, 'inpainted_images')
    print(f'NOTE: Using training views from camera_path {camera_path}')
    # Load See3D cameras
    see3d_viewpoints, _ = load_see3d_cameras(camera_path, inpaint_root_dir)
    train_viewpoints = input_viewpoints + see3d_viewpoints

    input_view_num = len(input_viewpoints)
    see3d_view_num = len(see3d_viewpoints)

    # Load plane masks
    plane_masks = []
    for i in range(len(train_viewpoints)):
        plane_mask_path = os.path.join(plane_root_path, f"plane_mask_frame{i:06d}.npy")
        plane_mask = np.load(plane_mask_path)
        plane_masks.append(plane_mask)

    # Load rgb images
    rgb_images = []
    for i in range(len(train_viewpoints)):
        rgb_image_path = os.path.join(plane_root_path, f"rgb_frame{i:06d}.png")
        rgb_image = Image.open(rgb_image_path)
        rgb_image = np.array(rgb_image)
        rgb_images.append(rgb_image)

    # Load refine depths
    refine_depth_list = []
    file_list = os.listdir(plane_root_path)
    refine_depth_file_name = [file for file in file_list if 'refine_depth_frame' in file]
    refine_depth_file_name.sort()
    for refine_depth_file_name_i in refine_depth_file_name:
        refine_depth_path = os.path.join(plane_root_path, refine_depth_file_name_i)
        refine_depth_i = Image.open(refine_depth_path)
        refine_depth_i = np.array(refine_depth_i)
        refine_depth_i = torch.from_numpy(refine_depth_i).cuda()
        refine_depth_list.append(refine_depth_i)

    # Load local plane points
    local_plane_points = []
    for i in range(len(train_viewpoints)):
        local_plane_points_path = os.path.join(plane_root_path, f"refine_points_frame{i:06d}.ply")
        local_plane_points_mesh = trimesh.load(local_plane_points_path)
        local_plane_points.append(local_plane_points_mesh.vertices)

    # Load global 3D plane dict
    global_3Dplane_dict_path = os.path.join(plane_root_path, 'global_3Dplane_ID_dict.json')
    if not os.path.exists(global_3Dplane_dict_path):
        raise FileNotFoundError(f"Global 3D plane dict file does not exist: {global_3Dplane_dict_path}")
    
    with open(global_3Dplane_dict_path, 'r') as f:
        global_3Dplane_ID_dict = json.load(f)
    
    # Convert keys to int
    global_3Dplane_ID_dict = {int(k): v for k, v in global_3Dplane_ID_dict.items()}
    print(f"Loaded {len(global_3Dplane_ID_dict)} global 3D planes")

    # Get global 3D plane points
    global_3Dplane_points = {}
    global_3Dplane_points_anchor_view = {}              # build global plane ID - anchor view ID dict
    for item in global_3Dplane_ID_dict.items():
        k, v = item
        temp_global_3Dplane_points = []

        for (view_id, plane_id) in v:
            view_mask_map = plane_masks[view_id]
            H, W = view_mask_map.shape
            view_plane_points = local_plane_points[view_id].reshape(H, W, 3)
            view_global_plane_points = view_plane_points[view_mask_map == plane_id]
            temp_global_3Dplane_points.append(view_global_plane_points)
        
        temp_global_3Dplane_points = np.concatenate(temp_global_3Dplane_points, axis=0)
        global_3Dplane_points[k] = temp_global_3Dplane_points

        # project global plane points to anchor view
        max_inview_pnts_num = 0
        max_inview_pnts_view_id = -1
        temp_global_3Dplane_points = torch.from_numpy(temp_global_3Dplane_points).cuda().float()
        for anchor_view_id in anchor_view_id_list:
            anchor_viewpoint = train_viewpoints[anchor_view_id]
            anchor_refine_depth = refine_depth_list[anchor_view_id]
            anchor_view_points_depth, anchor_view_points_2d, in_image = project_points_to_image(anchor_viewpoint, temp_global_3Dplane_points)
            
            H, W = anchor_refine_depth.shape
            valid_points_2d = anchor_view_points_2d[in_image]
            u = torch.clamp(valid_points_2d[:, 0].long(), 0, W-1)
            v = torch.clamp(valid_points_2d[:, 1].long(), 0, H-1)
            valid_points_depth = anchor_view_points_depth[in_image]
            depth_at_pixels = anchor_refine_depth[v, u]

            # Check if depth difference is within threshold
            depth_diff = torch.abs(valid_points_depth - depth_at_pixels)
            relative_diff = depth_diff / (valid_points_depth + 1e-6)
            depth_valid = relative_diff < depth_threshold
            depth_valid = depth_valid & (valid_points_depth > 0)            # avoid negative depth

            inview_pnts_num = torch.sum(depth_valid).item()
            if inview_pnts_num > max_inview_pnts_num:
                max_inview_pnts_num = inview_pnts_num
                max_inview_pnts_view_id = anchor_view_id

        global_3Dplane_points_anchor_view[k] = max_inview_pnts_view_id

    # replace color
    for item in global_3Dplane_ID_dict.items():
        k, v = item
        max_inview_pnts_view_id = global_3Dplane_points_anchor_view[k]
        if max_inview_pnts_view_id == -1:       # no anchor view, not replace color
            continue

        anchor_viewpoint = train_viewpoints[max_inview_pnts_view_id]
        anchor_refine_depth = refine_depth_list[max_inview_pnts_view_id]
        anchor_rgb_map = rgb_images[max_inview_pnts_view_id]
        H, W = anchor_refine_depth.shape

        for (view_id, plane_id) in v:
            view_rgb_map = rgb_images[view_id]
            view_mask_map = plane_masks[view_id]
            view_mask_map_flat = view_mask_map.reshape(-1)
            view_plane_points = local_plane_points[view_id]
            view_global_plane_points = view_plane_points[view_mask_map_flat == plane_id]

            view_global_plane_points = torch.from_numpy(view_global_plane_points).cuda().float()
            view_global_plane_points_depth, view_global_plane_points_2d, in_image = project_points_to_image(anchor_viewpoint, view_global_plane_points)
            
            valid_points_2d = view_global_plane_points_2d[in_image]
            u = torch.clamp(valid_points_2d[:, 0].long(), 0, W-1)
            v = torch.clamp(valid_points_2d[:, 1].long(), 0, H-1)
            valid_points_depth = view_global_plane_points_depth[in_image]
            depth_at_pixels = anchor_refine_depth[v, u]

            # Check if depth difference is within threshold
            depth_diff = torch.abs(valid_points_depth - depth_at_pixels)
            relative_diff = depth_diff / (valid_points_depth + 1e-6)
            depth_valid = relative_diff < depth_threshold
            depth_valid = depth_valid & (valid_points_depth > 0)            # avoid negative depth
            valid_indices = torch.nonzero(in_image).squeeze(-1)[depth_valid].cpu().numpy()
            
            # replace color use anchor rgb map
            if torch.sum(depth_valid) > 0:
                # Get valid pixel coordinates in anchor view
                valid_u = u[depth_valid].cpu().numpy()
                valid_v = v[depth_valid].cpu().numpy()
                # Get anchor colors for valid pixels
                anchor_colors = anchor_rgb_map[valid_v, valid_u]  # Shape: (N_valid, 3)

                # Get corresponding pixel coordinates in current view
                current_pixel_coords = np.where(view_mask_map == plane_id)
                current_v_coords = current_pixel_coords[0]  # y coordinates
                current_u_coords = current_pixel_coords[1]  # x coordinates

                # Use these indices to get corresponding current view pixel coordinates
                current_v_valid = current_v_coords[valid_indices]
                current_u_valid = current_u_coords[valid_indices]
                
                # Replace colors in current view
                view_rgb_map[current_v_valid, current_u_valid] = anchor_colors
                
                # Update the rgb_images list
                rgb_images[view_id] = view_rgb_map

        print(f'********** global plane {k} replace color done **********')


    # save rgb images
    old_inpaint_root_dir = os.path.join(see3d_root_path, 'inpainted_images_ori')
    if os.path.exists(old_inpaint_root_dir):
        os.system(f'rm -rf {old_inpaint_root_dir}')
    os.rename(inpaint_root_dir, old_inpaint_root_dir)
    os.makedirs(inpaint_root_dir, exist_ok=True)
    for i in range(len(train_viewpoints)):

        # NOTE: save pseudo confident maps (all 1)
        H, W = rgb_images[i].shape[:2]
        confident_map_np = np.ones((H, W), dtype=np.uint8)
        confident_map_vis = (confident_map_np * 255).astype(np.uint8)
        confident_map_path = os.path.join(plane_root_path, f"confident_map_frame{i:06d}.png")
        Image.fromarray(confident_map_vis).save(confident_map_path)

        if i < input_view_num:
            continue

        save_id = i - input_view_num
        rgb_image_path = os.path.join(inpaint_root_dir, f"predict_warp_frame{save_id:06d}.png")
        Image.fromarray(rgb_images[i]).save(rgb_image_path)
        print(f'********** save rgb image {save_id} done **********')

    # cat images
    save_cat_image_path = os.path.join(see3d_root_path, 'cat_replace_color_images')
    os.makedirs(save_cat_image_path, exist_ok=True)
    for i in range(len(train_viewpoints)):
        cat_image_path = os.path.join(save_cat_image_path, f"cat_frame{i:06d}.png")
        
        # original image
        rgb_image_path = os.path.join(plane_root_path, f"rgb_frame{i:06d}.png")
        rgb_image_ori = Image.open(rgb_image_path)

        # replace color image
        rgb_image_replace = Image.fromarray(rgb_images[i])

        # cat image
        padding_size = 10
        cat_image = Image.new('RGB', (rgb_image_ori.width * 2 + padding_size, rgb_image_ori.height))
        cat_image.paste(rgb_image_ori, (0, 0))
        cat_image.paste(rgb_image_replace, (rgb_image_ori.width + padding_size, 0))
        cat_image.save(cat_image_path)
        print(f'********** save cat image {i} done **********')

    print('done')

