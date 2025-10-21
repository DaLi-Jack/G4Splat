import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), '2d-gaussian-splatting'))
from scene.dataset_readers import load_see3d_cameras
import torch
import numpy as np
from argparse import ArgumentParser
from scene.dataset_readers import load_cameras
from utils.general_utils import safe_state
from guidance.cam_utils import project_points_to_image
import trimesh
from PIL import Image
import json
import time

class PseudoParams:
    def __init__(self, source_path):
        self.sh_degree = 3
        self.source_path = source_path
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.render_items = ['RGB', 'Alpha', 'Normal', 'Depth', 'Edge', 'Curvature']

def get_visible_mask_for_input_views(input_viewpoints, refine_depth_list, pnts, depth_threshold=0.1):

    visible_mask = torch.zeros((pnts.shape[0],), dtype=torch.bool, device='cuda')
    for view_id in range(len(input_viewpoints)):
        input_viewpoint = input_viewpoints[view_id]
        refine_depth_i = refine_depth_list[view_id]
        pnts_i_depth, pnts_i_2d, in_image = project_points_to_image(input_viewpoint, pnts)

        H, W = refine_depth_i.shape
        valid_points_2d = pnts_i_2d[in_image]
        u = torch.clamp(valid_points_2d[:, 0].long(), 0, W-1)
        v = torch.clamp(valid_points_2d[:, 1].long(), 0, H-1)
        valid_points_depth = pnts_i_depth[in_image]
        depth_at_pixels = refine_depth_i[v, u]

        # Check if depth difference is within threshold
        depth_diff = torch.abs(valid_points_depth - depth_at_pixels)
        relative_diff = depth_diff / (valid_points_depth + 1e-6)
        depth_valid = relative_diff < depth_threshold
        depth_valid = depth_valid & (valid_points_depth > 0)            # avoid negative depth

        valid_indices = torch.nonzero(in_image).squeeze(-1)[depth_valid]
        visible_mask[valid_indices] = True

    return visible_mask

def get_all_global_3Dpnts(source_path, plane_root_path, see3d_root_path, output_path, top_k=10):

    safe_state(False)
    
    # Load cameras
    train_viewpoints, _ = load_cameras(PseudoParams(source_path))
    input_view_num = len(train_viewpoints)
    
    camera_path = os.path.join(see3d_root_path, 'see3d_cameras.npz')
    if os.path.exists(camera_path):
        inpaint_root_dir = os.path.join(see3d_root_path, 'inpainted_images')
        print(f'NOTE: Loading See3D cameras from {camera_path}')
        see3d_gs_cameras_list, _ = load_see3d_cameras(camera_path, inpaint_root_dir)
        see3d_view_num = len(see3d_gs_cameras_list)
        train_viewpoints = train_viewpoints + see3d_gs_cameras_list

    # Load plane masks
    plane_masks = []
    for i in range(len(train_viewpoints)):
        plane_mask_path = os.path.join(plane_root_path, f"plane_mask_frame{i:06d}.npy")
        plane_mask = np.load(plane_mask_path)
        plane_masks.append(plane_mask)

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
    plane_point_counts = {}
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

        plane_point_counts[k] = temp_global_3Dplane_points.shape[0]

    sorted_planes = sorted(plane_point_counts.items(), key=lambda x: x[1], reverse=True)
    top_k_planes = sorted_planes[:top_k]

    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
        for plane_id, plane_points_num in top_k_planes:
            plane_points_path = os.path.join(output_path, f"global_3Dplane_points_{plane_id:06d}.ply")
            trimesh.PointCloud(global_3Dplane_points[plane_id]).export(plane_points_path)
            print(f"Saved {plane_points_path}")

    plane_all_points_dict = {}
    for plane_id, plane_points_num in top_k_planes:
        plane_all_points_dict[plane_id] = global_3Dplane_points[plane_id]

    return plane_all_points_dict


def get_none_vis_global_3Dpnts(source_path, plane_root_path, see3d_root_path, output_path, top_k=10, depth_threshold=0.1, return_all_vis_pnts=False):

    safe_state(False)
    
    # Load cameras
    train_viewpoints, _ = load_cameras(PseudoParams(source_path))
    input_view_num = len(train_viewpoints)
    
    camera_path = os.path.join(see3d_root_path, 'see3d_cameras.npz')
    if os.path.exists(camera_path):
        inpaint_root_dir = os.path.join(see3d_root_path, 'inpainted_images')
        print(f'NOTE: Loading See3D cameras from {camera_path}')
        see3d_gs_cameras_list, _ = load_see3d_cameras(camera_path, inpaint_root_dir)
        see3d_view_num = len(see3d_gs_cameras_list)
        train_viewpoints = train_viewpoints + see3d_gs_cameras_list

    # Load plane masks
    plane_masks = []
    for i in range(len(train_viewpoints)):
        plane_mask_path = os.path.join(plane_root_path, f"plane_mask_frame{i:06d}.npy")
        plane_mask = np.load(plane_mask_path)
        plane_masks.append(plane_mask)

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
    plane_point_counts = {}
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

        plane_point_counts[k] = temp_global_3Dplane_points.shape[0]

    sorted_planes = sorted(plane_point_counts.items(), key=lambda x: x[1], reverse=True)
    top_k_planes = sorted_planes[:top_k]

    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
        for plane_id, plane_points_num in top_k_planes:
            plane_points_path = os.path.join(output_path, f"global_3Dplane_points_{plane_id:06d}.ply")
            trimesh.PointCloud(global_3Dplane_points[plane_id]).export(plane_points_path)
            print(f"Saved {plane_points_path}")

    # get none vis in input views points for top k planes
    none_vis_plane_points_dict = {}
    for plane_id, plane_points_num in top_k_planes:
        plane_points = global_3Dplane_points[plane_id]
        plane_points = torch.from_numpy(plane_points).cuda().float()
        plane_vis_mask = get_visible_mask_for_input_views(train_viewpoints[:input_view_num], refine_depth_list[:input_view_num], plane_points, depth_threshold)

        none_vis_plane_points = plane_points[~plane_vis_mask]
        if none_vis_plane_points.shape[0] == 0:
            print(f"Plane {plane_id} points all vis in input views")
            continue
        print(f"Plane {plane_id} has {none_vis_plane_points.shape[0]} none vis points")

        none_vis_plane_points_dict[plane_id] = none_vis_plane_points.cpu().numpy()

        if output_path is not None:
            # save none vis plane points as ply
            none_vis_plane_points_path = os.path.join(output_path, f"none_vis_global_3Dplane_points_{plane_id:06d}.ply")
            trimesh.PointCloud(none_vis_plane_points.cpu().numpy()).export(none_vis_plane_points_path)
            print(f"Saved {none_vis_plane_points_path}")

    if return_all_vis_pnts:
        plane_all_points_dict = {}
        for plane_id, plane_points_num in top_k_planes:
            plane_all_points_dict[plane_id] = global_3Dplane_points[plane_id]

        return none_vis_plane_points_dict, plane_all_points_dict

    return none_vis_plane_points_dict



if __name__ == "__main__":
    # Set up command line arguments
    parser = ArgumentParser(description="Fast 3D Instance Segmentation for Mesh")
    parser.add_argument("--source_path", type=str, required=True, help="Path to source data directory")
    parser.add_argument("--plane_root_path", type=str, required=True, help="Path to plane data directory")
    parser.add_argument("--see3d_root_path", type=str, default=None, help="Path to See3D data")
    parser.add_argument("--output_path", type=str, default=None, help="Output path for colored mesh")
    args = parser.parse_args()

    t1 = time.time()
    get_none_vis_global_3Dpnts(args.source_path, args.plane_root_path, args.see3d_root_path, args.output_path)

    t2 = time.time()
    print(f"Time taken: {t2 - t1} seconds")




