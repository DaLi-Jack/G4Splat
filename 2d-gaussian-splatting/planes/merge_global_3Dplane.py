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
from utils.general_utils import safe_state
from guidance.cam_utils import get_covisible_points, project_points_to_image
from guidance.cam_utils import get_pixel_to_points_tensor, get_visible_points_mask
import trimesh
from PIL import Image
import json
from matcha.dm_scene.cameras import GSCamera
import cv2

def save_tensor_as_pcd(pcd, path, pcd_colors=None):

    if isinstance(pcd, torch.Tensor):
        pcd = pcd.detach().cpu().numpy()
    pcd = trimesh.PointCloud(pcd)
    if pcd_colors is not None:
        if isinstance(pcd_colors, torch.Tensor):
            pcd_colors = pcd_colors.detach().cpu().numpy()
        pcd.colors = pcd_colors
    pcd.export(path)

def torch_intersect1d(tensor1, tensor2, assume_unique=True):
    """
    Find intersection of two 1D tensors (equivalent to torch.intersect1d)
    """
    mask = torch.isin(tensor1, tensor2, assume_unique=assume_unique)
    return tensor1[mask]

def torch_union1d(tensor1, tensor2):
    """
    Find union of two 1D tensors (equivalent to torch.union1d)
    """
    return torch.unique(torch.cat([tensor1, tensor2]))

def vis_plane_mask(plane_mask, plane_id, rgb_path, save_path):
    """
    Vis plane mask
    """
    obj_mask_map = (plane_mask == plane_id)
    rgb = Image.open(rgb_path)
    rgb = np.array(rgb)
    mask_rgb_map = np.zeros_like(rgb)
    obj_mask_map_temp = obj_mask_map.cpu().numpy()
    mask_rgb_map[obj_mask_map_temp] = rgb[obj_mask_map_temp]
    mask_rgb_map = Image.fromarray(mask_rgb_map)
    mask_rgb_map.save(save_path)

def vis_plane_mask_1(plane_mask, plane_id, rgb_path, save_path, transparency=0.6, color=[0, 0, 255]):
    """
    Vis plane mask with overlay visualization
    """
    # Create binary mask for the specific plane
    obj_mask_map = (plane_mask == plane_id)
    obj_mask_map = obj_mask_map.cpu().numpy()
    
    # Load RGB image
    rgb = Image.open(rgb_path)
    rgb_image = np.array(rgb)
    
    # Ensure correct input format
    if rgb_image.dtype != np.uint8:
        rgb_image = (rgb_image * 255).astype(np.uint8)
    
    # Create colored mask
    colored_mask = np.zeros_like(rgb_image)
    colored_mask[obj_mask_map] = color
    
    # Blend images
    blended = rgb_image.copy()
    mask_indices = obj_mask_map
    
    for i in range(3):
        blended[:, :, i] = np.where(
            mask_indices,
            rgb_image[:, :, i] * (1 - transparency) + colored_mask[:, :, i] * transparency,
            rgb_image[:, :, i]
        )
    
    # Save blended image
    blended_image = Image.fromarray(blended.astype(np.uint8))
    blended_image.save(save_path)

def get_plane_pnts_idx_from_mask(pixel_pnts_map, plane_mask, plane_id):
    """
    Get plane pnts from plane mask
    """
    obj_mask_map = (plane_mask == plane_id)
    plane_pnts_idx_temp = pixel_pnts_map[obj_mask_map]
    plane_pnts_idx = plane_pnts_idx_temp.unique()                   # get unique plane pnts index
    plane_pnts_idx = plane_pnts_idx[plane_pnts_idx != 0]            # delete 0 in plane_pnts_idx

    return plane_pnts_idx

def get_covisibility_rate(plane_pnts_idx_1, plane_pnts_idx_2):
    """
    Get covisibility rate between two plane pnts
    """
    covisible_idx = torch_intersect1d(plane_pnts_idx_1, plane_pnts_idx_2, assume_unique=True)
    
    ratio_1 = covisible_idx.shape[0] / plane_pnts_idx_1.shape[0]
    ratio_2 = covisible_idx.shape[0] / plane_pnts_idx_2.shape[0]

    max_ratio = max(ratio_1, ratio_2)
    return max_ratio

def get_init_global_3Dplane(init_pixel_pnts_map, plane_mask, view_id):
    """
    Get init global 3D plane from the first view pixel pnts map and plane mask
    """
    global_3Dplane_pnts_idx = []
    global_3Dplane_ID_dict = {}                     # {global_3Dplane_ID: [(view_id, plane_id)]}

    global_3Dplane_ID = 0
    plane_id_list = torch.unique(plane_mask)
    for plane_id in plane_id_list:
        if plane_id == 0:                           # 0 is default value, not a plane
            continue
        
        plane_pnts_idx = get_plane_pnts_idx_from_mask(init_pixel_pnts_map, plane_mask, plane_id)
        global_3Dplane_pnts_idx.append(plane_pnts_idx)
        global_3Dplane_ID_dict[global_3Dplane_ID] = [(view_id, plane_id.item())]
        global_3Dplane_ID += 1

    return global_3Dplane_pnts_idx, global_3Dplane_ID_dict

def update_global_3Dplane(global_3Dplane_pnts_idx, global_3Dplane_ID_dict, pixel_pnts_map, plane_mask, view_id, covisible_ratio_thresh=0.5):
    """
    Update global 3D plane
    """
    plane_id_list = torch.unique(plane_mask)
    for plane_id in plane_id_list:
        if plane_id == 0:                           # 0 is default value, not a plane
            continue

        plane_pnts_idx = get_plane_pnts_idx_from_mask(pixel_pnts_map, plane_mask, plane_id)

        # NOTE: if plane_pnts_idx is empty, skip (use chart pnts, not have points in See3D inpaint regions)
        if plane_pnts_idx.shape[0] == 0:
            continue

        global_3Dplane_num = len(global_3Dplane_pnts_idx)
        merge_flag = False
        for i in range(global_3Dplane_num):
            global_3Dplane_pnts_idx_i = global_3Dplane_pnts_idx[i]

            covisibility_rate = get_covisibility_rate(global_3Dplane_pnts_idx_i, plane_pnts_idx)
            if covisibility_rate > covisible_ratio_thresh:
                # merge two global 3D plane
                new_global_3Dplane_pnts_idx = torch_union1d(global_3Dplane_pnts_idx_i, plane_pnts_idx)
                global_3Dplane_pnts_idx[i] = new_global_3Dplane_pnts_idx
                global_3Dplane_ID_dict[i].append((view_id, plane_id.item()))

                merge_flag = True
                break

        if not merge_flag:                       # no merge, add new global 3D plane
            global_3Dplane_pnts_idx.append(plane_pnts_idx)
            global_3Dplane_ID_dict[global_3Dplane_num] = [(view_id, plane_id.item())]

    return global_3Dplane_pnts_idx, global_3Dplane_ID_dict

def final_merge_global_3Dplane(global_3Dplane_pnts_idx, global_3Dplane_ID_dict, covisible_ratio_thresh=0.5):
    """
    Final merge global 3D plane, check the covisibility rate between each global 3D plane
    """
    new_global_3Dplane_pnts_idx = []
    new_global_3Dplane_ID_dict = {}
    
    # Create a flag array to mark which planes have been merged
    merged = [False] * len(global_3Dplane_pnts_idx)
    new_plane_id = 0
    
    for i in range(len(global_3Dplane_pnts_idx)):
        if merged[i]:
            continue
            
        # Current plane's point indices and ID information
        current_plane_pnts_idx = global_3Dplane_pnts_idx[i]
        current_plane_ids = global_3Dplane_ID_dict[i].copy()
        
        # Check if other planes can be merged with current plane
        for j in range(i + 1, len(global_3Dplane_pnts_idx)):
            if merged[j]:
                continue
                
            # Calculate covisibility rate
            covisibility_rate = get_covisibility_rate(current_plane_pnts_idx, global_3Dplane_pnts_idx[j])
            
            if covisibility_rate > covisible_ratio_thresh:
                # Merge planes
                current_plane_pnts_idx = torch_union1d(current_plane_pnts_idx, global_3Dplane_pnts_idx[j])
                current_plane_ids.extend(global_3Dplane_ID_dict[j])
                merged[j] = True
        
        # Add merged plane to new lists
        new_global_3Dplane_pnts_idx.append(current_plane_pnts_idx)
        new_global_3Dplane_ID_dict[new_plane_id] = current_plane_ids
        new_plane_id += 1
        merged[i] = True
    
    return new_global_3Dplane_pnts_idx, new_global_3Dplane_ID_dict

def get_global_3Dplane(cameras, points, plane_masks, previous_global_3Dplane_ID_dict):
    """
    Get global 3D plane from multiple camera views
    """

    if previous_global_3Dplane_ID_dict is None:
        cur_view_id = 0
        pixel_pnts_map = get_pixel_to_points_tensor(cameras[cur_view_id], points)
        plane_mask = plane_masks[cur_view_id]
        global_3Dplane_pnts_idx, global_3Dplane_ID_dict = get_init_global_3Dplane(pixel_pnts_map, plane_mask, cur_view_id)
    else:
        cur_view_id = 0
        global_3Dplane_ID_dict = previous_global_3Dplane_ID_dict
        for global_3Dplane_id, local_3Dplane_id_list in global_3Dplane_ID_dict.items():
            for view_id, plane_id in local_3Dplane_id_list:
                cur_view_id = max(cur_view_id, view_id)

        pixel_pnts_map_list = []
        for view_id in range(cur_view_id+1):
            pixel_pnts_map = get_pixel_to_points_tensor(cameras[view_id], points)
            pixel_pnts_map_list.append(pixel_pnts_map)
        
        global_3Dplane_pnts_idx = []
        for global_3Dplane_id, local_3Dplane_id_list in global_3Dplane_ID_dict.items():
            global_3Dplane_pnts_idx_i = None
            for view_id, plane_id in local_3Dplane_id_list:
                plane_pnts_idx = get_plane_pnts_idx_from_mask(pixel_pnts_map_list[view_id], plane_masks[view_id], plane_id)
                if global_3Dplane_pnts_idx_i is None:
                    global_3Dplane_pnts_idx_i = plane_pnts_idx
                else:
                    global_3Dplane_pnts_idx_i = torch_union1d(global_3Dplane_pnts_idx_i, plane_pnts_idx)
            global_3Dplane_pnts_idx.append(global_3Dplane_pnts_idx_i)

    for view_id in range(cur_view_id+1, len(cameras)):

        pixel_pnts_map = get_pixel_to_points_tensor(cameras[view_id], points)
        plane_mask = plane_masks[view_id]
        global_3Dplane_pnts_idx, global_3Dplane_ID_dict = update_global_3Dplane(global_3Dplane_pnts_idx, global_3Dplane_ID_dict, pixel_pnts_map, plane_mask, view_id)
    
    global_3Dplane_pnts_idx, global_3Dplane_ID_dict = final_merge_global_3Dplane(global_3Dplane_pnts_idx, global_3Dplane_ID_dict)
    
    return global_3Dplane_pnts_idx, global_3Dplane_ID_dict


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--pnts_path", type=str, required=True)
    parser.add_argument("--plane_root_path", type=str, required=True)
    parser.add_argument("--see3d_root_path", type=str, default=None)
    parser.add_argument("--vis_plane_path", type=str, default=None)
    model = ModelParams(parser, sentinel=True)
    args = parser.parse_args()

    print('NOTE: Using training views from data_path')
    # Initialize system state (RNG)
    safe_state(False)

    train_viewpoints, _ = load_cameras(model.extract(args))

    if args.see3d_root_path is not None:
        camera_path = os.path.join(args.see3d_root_path, 'see3d_cameras.npz')
        inpaint_root_dir = os.path.join(args.see3d_root_path, 'inpainted_images')
        print(f'NOTE: Using training views from camera_path {camera_path}')
        # Load See3D cameras
        see3d_gs_cameras_list, _ = load_see3d_cameras(camera_path, inpaint_root_dir)

        train_viewpoints = train_viewpoints + see3d_gs_cameras_list

    # load plane mask and refine points
    plane_root_path = args.plane_root_path
    plane_masks = []
    for i in range(len(train_viewpoints)):
        plane_mask_path = os.path.join(plane_root_path, f"plane_mask_frame{i:06d}.npy")
        plane_mask = np.load(plane_mask_path)
        plane_mask = torch.tensor(plane_mask, dtype=torch.int).cuda()
        plane_masks.append(plane_mask)

    # load last refine points
    pnts_list = []
    file_list = os.listdir(plane_root_path)
    refine_points_file_name = [file for file in file_list if 'refine_points_frame' in file]
    refine_points_file_name.sort()
    for refine_points_file_name_i in refine_points_file_name:
        pnts_path = os.path.join(plane_root_path, refine_points_file_name_i)                # last refine points
        pnts_i = trimesh.load(pnts_path).vertices
        pnts_i = torch.tensor(pnts_i, dtype=torch.float32).cuda()
        pnts_list.append(pnts_i)

    # load need inpaint views points (this stage seen points)
    need_inpaint_points_file_name = [file for file in file_list if 'need_inpaint_views_points' in file]
    need_inpaint_points_file_name.sort()
    for need_inpaint_points_file_name_i in need_inpaint_points_file_name:
        pnts_path = os.path.join(plane_root_path, need_inpaint_points_file_name_i)          # this stage seen points
        pnts_i = trimesh.load(pnts_path).vertices
        pnts_i = torch.tensor(pnts_i, dtype=torch.float32).cuda()
        pnts_list.append(pnts_i)

    # load dense view points (when use dense view for training)
    dense_view_points_file_name = [file for file in file_list if 'dense-view-points.ply' in file]
    dense_view_points_file_name.sort()
    for dense_view_points_file_name_i in dense_view_points_file_name:
        pnts_path = os.path.join(plane_root_path, dense_view_points_file_name_i)          # dense view points
        pnts_i = trimesh.load(pnts_path).vertices
        pnts_i = torch.tensor(pnts_i, dtype=torch.float32).cuda()
        pnts_list.append(pnts_i)

    if len(pnts_list) == 0:
        # load default chart pnts
        pnts_path = args.pnts_path
        pnts = trimesh.load(pnts_path).vertices
        pnts = torch.tensor(pnts, dtype=torch.float32).cuda()
        print(f'********** load default chart pnts from {pnts_path} **********')
    else:
        pnts = torch.cat(pnts_list, dim=0)
        print(f'********** load refine points **********')

    save_path = os.path.join(plane_root_path, 'global_3Dplane_ID_dict.json')
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            global_3Dplane_ID_dict = json.load(f)
        previous_global_3Dplane_ID_dict = {int(k): v for k, v in global_3Dplane_ID_dict.items()}
    else:
        previous_global_3Dplane_ID_dict = None

    global_3Dplane_pnts_idx, global_3Dplane_ID_dict = get_global_3Dplane(train_viewpoints, pnts, plane_masks, previous_global_3Dplane_ID_dict)

    # save global 3D plane ID dict
    if os.path.exists(save_path):
        os.remove(save_path)

    json_dict = {str(k): v for k, v in global_3Dplane_ID_dict.items()}
    with open(save_path, 'w') as f:
        json.dump(json_dict, f)

    print(f'global 3Dplane ID dict saved to {save_path}')

    if args.vis_plane_path is not None:
        vis_root_path = args.vis_plane_path
        # check global 3Dplane
        global_save_root_path = os.path.join(vis_root_path, 'check_global_3Dplane')
        os.makedirs(global_save_root_path, exist_ok=True)
        for idx in range(len(global_3Dplane_pnts_idx)):

            save_plane_root_path = os.path.join(global_save_root_path, f'global_3Dplane_{idx}')
            os.makedirs(save_plane_root_path, exist_ok=True)

            global_3Dplane_pnts_idx_i = global_3Dplane_pnts_idx[idx]
            global_3Dplane_ID_dict_i = global_3Dplane_ID_dict[idx]
            plane_pnts = pnts[global_3Dplane_pnts_idx_i]
            save_tensor_as_pcd(plane_pnts, os.path.join(save_plane_root_path, f'global_3Dplane_{idx}.ply'))
            
            for view_id, plane_id in global_3Dplane_ID_dict_i:
                rgb_path = os.path.join(plane_root_path, f"rgb_frame{view_id:06d}.png")
                vis_plane_mask(plane_masks[view_id], plane_id, rgb_path, os.path.join(save_plane_root_path, f"frame{view_id:06d}_plane{plane_id:06d}_mask.png"))
                print(f'plane{plane_id} mask saved')

        print('global 3Dplane saved over')

        # check plane pnts
        local_save_root_path = os.path.join(vis_root_path, 'check_local_plane_pnts')
        os.makedirs(local_save_root_path, exist_ok=True)
        for idx in range(len(train_viewpoints)):
            pixel_pnts_map = get_pixel_to_points_tensor(train_viewpoints[idx], pnts)
            plane_mask = plane_masks[idx]
            plane_id_list = torch.unique(plane_mask)
            for plane_id in plane_id_list:
                if plane_id == 0:
                    continue
                plane_pnts_idx = get_plane_pnts_idx_from_mask(pixel_pnts_map, plane_mask, plane_id)
                plane_pnts = pnts[plane_pnts_idx]
                if plane_pnts.shape[0] == 0:
                    continue
                save_tensor_as_pcd(plane_pnts, os.path.join(local_save_root_path, f'frame{idx:06d}_plane{plane_id:06d}_pnts.ply'))
                print(f'plane{plane_id} pnts saved')

                rgb_path = os.path.join(plane_root_path, f"rgb_frame{idx:06d}.png")
                vis_plane_mask(plane_mask, plane_id, rgb_path, os.path.join(local_save_root_path, f'frame{idx:06d}_plane{plane_id:06d}_mask.png'))
                print(f'plane{plane_id} mask saved')

        print('local plane pnts saved over')


