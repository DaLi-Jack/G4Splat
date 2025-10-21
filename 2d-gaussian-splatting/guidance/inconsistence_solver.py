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


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--plane_root_path", type=str, required=True)
    parser.add_argument("--see3d_root_path", type=str, default=None)
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

    # load see3d cameras
    camera_path = os.path.join(see3d_root_path, 'see3d_cameras.npz')
    inpaint_root_dir = os.path.join(see3d_root_path, 'inpainted_images')
    print(f'NOTE: Using training views from camera_path {camera_path}')
    # Load See3D cameras
    see3d_viewpoints, _ = load_see3d_cameras(camera_path, inpaint_root_dir)
    train_viewpoints = input_viewpoints + see3d_viewpoints

    input_view_num = len(input_viewpoints)
    see3d_view_num = len(see3d_viewpoints)

    # load refine points
    print(f'********** load refine points **********')
    pnts_list = []
    file_list = os.listdir(plane_root_path)
    refine_points_file_name = [file for file in file_list if 'refine_points_frame' in file]
    refine_points_file_name.sort()
    for refine_points_file_name_i in refine_points_file_name:
        pnts_path = os.path.join(plane_root_path, refine_points_file_name_i)                # last refine points
        pnts_i = trimesh.load(pnts_path).vertices
        pnts_i = torch.tensor(pnts_i, dtype=torch.float32).cuda()
        pnts_list.append(pnts_i)

    assert len(pnts_list) == len(train_viewpoints)          # check if the number of refine points is the same as the number of training viewpoints
    pnts = torch.cat(pnts_list, dim=0)

    # # save refine points
    # refine_points_path = os.path.join(plane_root_path, 'refine_points.ply')
    # trimesh.PointCloud(pnts.cpu().numpy()).export(refine_points_path)
    # print(f'********** save refine points **********')

    # load refine depth
    print(f'********** load refine depth **********')
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

    assert len(refine_depth_list) == len(train_viewpoints)          # check if the number of refine depth is the same as the number of training viewpoints

    # project pnts to image and get rgb for each pnts
    print(f'********** project pnts to image and get rgb for each pnts **********')

    num_points = pnts.shape[0]
    num_views = len(train_viewpoints)
    
    # Pre-allocate tensors to store projection information for all views
    # point_visibility: (num_points, num_views) - whether each point is visible in each view
    point_visibility = torch.zeros((num_points, num_views), dtype=torch.bool, device='cuda')
    
    # point_coords: (num_points, num_views, 2) - pixel coordinates of each point in each view
    point_coords = torch.zeros((num_points, num_views, 2), dtype=torch.long, device='cuda')
    
    # Collect RGB images from all views
    rgb_images = []

    for idx in range(num_views):
        viewpoint = train_viewpoints[idx]
        refine_depth_i = refine_depth_list[idx]
        pnts_i_depth, pnts_i_2d, in_image = project_points_to_image(viewpoint, pnts)

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

        # # save project pnts
        # pnts_i = pnts[valid_indices]
        # save_pnt_path = os.path.join(plane_root_path, f'refine_points_proj_{idx}.ply')
        # trimesh.PointCloud(pnts_i.cpu().numpy()).export(save_pnt_path)
        # print(f'********** save {idx} projected points **********')

        rgb_image = viewpoint.original_image
        rgb_image = rgb_image.permute(1, 2, 0)  # (3, H, W) -> (H, W, 3)
        rgb_images.append(rgb_image)

        point_visibility[valid_indices, idx] = True
        valid_u = u[depth_valid]
        valid_v = v[depth_valid]
        point_coords[valid_indices, idx, 0] = valid_u
        point_coords[valid_indices, idx, 1] = valid_v

    # get seen points idx in input views
    seen_in_input = torch.any(point_visibility[:, :input_view_num], dim=1)  # (num_points,)
    none_seen_in_input = ~seen_in_input  # (num_points,)
    
    # 2. Assign colors to none_seen_in_input points (randomly select from visible views)
    num_none_seen = torch.sum(none_seen_in_input).item()
    point_colors = torch.zeros((num_points, 3), dtype=torch.uint8, device='cuda')

    # Get all none_seen_in_input point indices
    none_seen_indices = torch.nonzero(none_seen_in_input).squeeze(-1)
    
    # Get visibility matrix for these points: (num_none_seen, num_views)
    point_vis_matrix = point_visibility[none_seen_indices]  # (num_none_seen, num_views)
    
    # Find the first visible view for each point
    # argmax will return the index of the first True value (since True > False)
    selected_view_indices = torch.argmax(point_vis_matrix.float(), dim=1)
    
    # Get coordinates for all selected views (batch operation)
    selected_coords = point_coords[none_seen_indices, selected_view_indices]  # (num_none_seen, 2)
    selected_u = selected_coords[:, 0]
    selected_v = selected_coords[:, 1]
    
    # Batch extract colors using advanced indexing
    batch_colors = torch.stack([
        rgb_images[view_idx][v, u] 
        for view_idx, u, v in zip(selected_view_indices, selected_u, selected_v)
    ])  # (num_none_seen, 3)
    
    batch_colors_uint8 = (batch_colors * 255).to(torch.uint8)
    
    # Assign colors to all points at once
    point_colors[none_seen_indices] = batch_colors_uint8
    
    print(f'Assigned colors to {num_none_seen} points not seen in input views')

    # 3. Generate confident maps for all views in parallel
    confident_maps = {}
    
    for view_idx in range(num_views):
        H, W = refine_depth_list[view_idx].shape
        
        if view_idx < input_view_num:
            # Input view confident map is all 1
            confident_maps[view_idx] = torch.ones((H, W), dtype=torch.uint8, device='cuda')
            print(f'Input view {view_idx} confident map set to all 1')
        else:
            # See3D view needs to calculate confident map based on color matching algorithm
            confident_map = torch.ones((H, W), dtype=torch.uint8, device='cuda')
            
            # Get points visible in current view
            current_view_visible = point_visibility[:, view_idx]  # (num_points,)
            
            if torch.any(current_view_visible):
                # Find indices of points visible in current view
                visible_point_indices = torch.nonzero(current_view_visible).squeeze(-1)
                
                # Batch get coordinates of these points
                visible_coords = point_coords[visible_point_indices, view_idx]  # (num_visible, 2)
                u_coords = visible_coords[:, 0]
                v_coords = visible_coords[:, 1]
                
                # Check if these points have been seen in input views
                visible_seen_in_input = seen_in_input[visible_point_indices]
                
                # For points seen in input views, set confident to 0
                input_seen_mask = visible_seen_in_input
                confident_map[v_coords[input_seen_mask], u_coords[input_seen_mask]] = 0
                
                # For points not seen in input views, perform color matching
                not_in_input_mask = ~visible_seen_in_input
                not_in_input_indices = visible_point_indices[not_in_input_mask]
                
                if len(not_in_input_indices) > 0:
                    # Get assigned colors for these points
                    assigned_colors = point_colors[not_in_input_indices]  # (num_not_in_input, 3)
                    
                    # Set assigned colors to original color
                    rgb_images[view_idx][v_coords[not_in_input_mask], u_coords[not_in_input_mask]] = assigned_colors / 255.0
            
            confident_maps[view_idx] = confident_map

    # save rgb images
    old_inpaint_root_dir = os.path.join(see3d_root_path, 'inpainted_images_ori')
    if os.path.exists(old_inpaint_root_dir):
        os.system(f'rm -rf {old_inpaint_root_dir}')
    os.rename(inpaint_root_dir, old_inpaint_root_dir)
    os.makedirs(inpaint_root_dir, exist_ok=True)
    for view_idx in range(num_views):
        if view_idx < input_view_num:
            continue

        save_view_idx = view_idx - input_view_num               # inpaint images use see3d images index
        rgb_path = os.path.join(inpaint_root_dir, f'predict_warp_frame{save_view_idx:06d}.png')
        rgb_map = rgb_images[view_idx].cpu().numpy() * 255
        rgb_map = Image.fromarray(rgb_map.astype(np.uint8))
        rgb_map.save(rgb_path)
        print(f'Saved assigned rgb image for view {view_idx} to {rgb_path}')

    # Save confident maps
    print(f'********** save confident maps **********')
    for view_idx, confident_map in confident_maps.items():
        # Save as image
        confident_map_np = confident_map.cpu().numpy()
        confident_map_vis = (confident_map_np * 255).astype(np.uint8)
        confident_map_img_path = os.path.join(plane_root_path, f'confident_map_frame{view_idx:06d}.png')
        Image.fromarray(confident_map_vis).save(confident_map_img_path)

        rgb_path = os.path.join(plane_root_path, f'rgb_frame{view_idx:06d}.png')
        rgb_image = Image.open(rgb_path)
        rgb_image = np.array(rgb_image)
        rgb_image = rgb_image * confident_map_np[:,:,None]
        rgb_image = Image.fromarray(rgb_image.astype(np.uint8))
        rgb_image.save(os.path.join(plane_root_path, f'confident_masked_frame{view_idx:06d}.png'))
        
        print(f'Saved confident map for view {view_idx} to {confident_map_img_path}')

    t2 = time.time()

    print(f'Efficient parallel confident map generation completed! Time cost: {t2 - t1:.2f}s')


