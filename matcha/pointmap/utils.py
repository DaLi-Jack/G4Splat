import os
import numpy as np
import torch
from matcha.dm_scene.cameras import CamerasWrapper, load_gs_cameras
from matcha.dm_utils.dataset_readers import read_points3D_binary, read_extrinsics_binary


def load_colmap_sfm_data(
    colmap_source_path:str, 
    device:torch.device,
):
    # Load SfM points    
    pt_ids, xyzs, rgbs, errors, all_track_elems = read_points3D_binary(os.path.join(colmap_source_path, 'sparse', '0', 'points3D.bin'))
    sfm_xyz = torch.tensor(xyzs, device=device, dtype=torch.float32)
    sfm_col = torch.tensor(rgbs / 255., device=device, dtype=torch.float32)

    # Get the mapping from point ids to point indices
    pt_ids = torch.tensor(pt_ids.astype(np.int64), device=device)
    pts_id_to_idx = - torch.ones(pt_ids.max().int().item() + 1, device=device, dtype=torch.int64)  # point ids to point indices
    pts_id_to_idx[pt_ids] = torch.arange(len(pt_ids), device=device)

    # Get, for each image name, the indices of the points that are visible in that image
    image_sfm_points = {}
    parsed_images = read_extrinsics_binary(os.path.join(colmap_source_path, 'sparse', '0', 'images.bin'))
    for id, image in parsed_images.items():
        image_points_ids = torch.tensor(image.point3D_ids)
        image_points_ids = image_points_ids[image_points_ids > -1].unique()
        image_sfm_points[image.name.split('.')[0]] = pts_id_to_idx[image_points_ids]
        
    colmap_sfm_data = {
        'sfm_xyz': sfm_xyz,
        'sfm_col': sfm_col,
        'image_sfm_points': image_sfm_points,
    }
    
    return colmap_sfm_data


def load_colmap_scene(
    source_path,
    load_gt_images=True,
    max_img_size=1920,
    white_background=False,
    eval_split=True,
    eval_split_interval=8,
    device='cuda',
):
    # Load cameras
    _cam_list = load_gs_cameras(
        source_path=source_path,
        load_gt_images=load_gt_images,
        max_img_size=max_img_size,
        white_background=white_background,
    )

    if eval_split:
        cam_list = []
        test_cam_list = []
        for i, cam in enumerate(_cam_list):
            if i % eval_split_interval == 0:
                test_cam_list.append(cam)
            else:
                cam_list.append(cam)
        test_cameras = CamerasWrapper(test_cam_list)

    else:
        cam_list = _cam_list
        test_cam_list = None
        test_cameras = None

    training_cameras = CamerasWrapper(cam_list)
    
    # Load SfM data
    if True:
        colmap_sfm_data = load_colmap_sfm_data(source_path, device=device)
        sfm_xyz = colmap_sfm_data['sfm_xyz']
        sfm_col = colmap_sfm_data['sfm_col']
        image_sfm_points = colmap_sfm_data['image_sfm_points']
    else:
        ply_path = os.path.join(source_path, "sparse/0/points3D.ply")
        pcd = fetchPly(ply_path)
        sfm_xyz = torch.tensor(pcd.points, device=device).float().cuda()
        sfm_col = torch.tensor(pcd.colors, device=device).float().cuda()
        image_sfm_points = None
    
    scene_package = {
        'training_cameras': training_cameras,
        'test_cameras': test_cameras,
        'sfm_xyz': sfm_xyz,
        'sfm_col': sfm_col,
        'image_sfm_points': image_sfm_points,
    }
    
    return scene_package