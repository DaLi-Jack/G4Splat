#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
sys.path.append(os.getcwd())
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import cv2
import torch

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):

    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):

    if eval:
        try:
            cameras_extrinsic_file = os.path.join(path, "all-sparse/0", "images.bin")
            cameras_intrinsic_file = os.path.join(path, "all-sparse/0", "cameras.bin")
            cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        except:
            cameras_extrinsic_file = os.path.join(path, "all-sparse/0", "images.txt")
            cameras_intrinsic_file = os.path.join(path, "all-sparse/0", "cameras.txt")
            cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    else:
        dense_sparse_point3d_file = os.path.join(path, "dense-view-sparse/0/points3D.ply")
        if os.path.exists(dense_sparse_point3d_file):
            print(f'[INFO]: Load cameras from dense-view-sparse for dense view training')
            try:
                cameras_extrinsic_file = os.path.join(path, "dense-view-sparse/0", "images.bin")
                cameras_intrinsic_file = os.path.join(path, "dense-view-sparse/0", "cameras.bin")
                cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
                cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
            except:
                cameras_extrinsic_file = os.path.join(path, "dense-view-sparse/0", "images.txt")
                cameras_intrinsic_file = os.path.join(path, "dense-view-sparse/0", "cameras.txt")
                cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
                cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

        else:
            try:
                cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
                cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
                cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
                cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
            except:
                cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
                cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
                cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
                cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images

    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    # if eval:
    #     train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
    #     test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    # else:
    train_cam_infos = cam_infos
    test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}


# NOTE: load gaussian cameras
def fill_config_args(args):
    args.sh_degree = 3
    args.images = 'images'
    args.resolution = -1
    args.white_background = False
    args.data_device = "cuda"
    args.eval = False
    args.render_items=['RGB', 'Alpha', 'Normal', 'Depth', 'Edge', 'Curvature']

    return args

from utils.camera_utils import cameraList_from_camInfos
def load_cameras(args, resolution_scales=[1.0], scale=1.0):

    args = fill_config_args(args)

    if os.path.exists(os.path.join(args.source_path, "sparse")):
        scene_info = readColmapSceneInfo(args.source_path, args.images, args.eval)
    elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
        print("Found transforms_train.json file, assuming Blender data set!")
        scene_info = readNerfSyntheticInfo(args.source_path, args.white_background, args.eval)
    else:
        assert False, "Could not recognize scene type!"

    train_cameras = {}
    test_cameras = {}

    for resolution_scale in resolution_scales:
        print("Loading Training Cameras")
        train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
        print("Loading Test Cameras")
        test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

    return train_cameras[scale], test_cameras[scale]

from matcha.dm_scene.cameras import GSCamera
def load_see3d_cameras(camera_path, inpainted_image_root_path):
    '''
    Load see3d inpaint view cameras
    '''

    print(f'NOTE: Load See3D inpaint view cameras from camera_path {camera_path}')

    temp_image_name = os.listdir(inpainted_image_root_path)[0]
    postfix = temp_image_name.split('.')[-1]

    see3d_cameras = np.load(camera_path)
    see3d_viewpoints = []
    train_views = see3d_cameras['train_views']
    n_views = see3d_cameras['n_views']
    for i in range(n_views):
        R = see3d_cameras[f'R_{i:06d}']
        T = see3d_cameras[f'T_{i:06d}']
        FoVx = see3d_cameras[f'FoVx_{i:06d}']
        FoVy = see3d_cameras[f'FoVy_{i:06d}']
        image_width = int(see3d_cameras[f'image_width_{i:06d}'])
        image_height = int(see3d_cameras[f'image_height_{i:06d}'])
        image_name = f'predict_warp_frame{i:06d}'

        inpainted_image_path = os.path.join(inpainted_image_root_path, image_name + f'.{postfix}')
        inpainted_image = cv2.imread(inpainted_image_path)
        inpainted_image = cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB) / 255.0
        inpainted_image = torch.from_numpy(inpainted_image).float().to("cuda").permute(2, 0, 1)

        see3d_viewpoints.append(GSCamera(
            colmap_id=i+train_views,                     # avoid colmap id conflict with input viewpoint cameras
            R=R,
            T=T,
            FoVx=FoVx,
            FoVy=FoVy,
            image=inpainted_image,
            image_width=image_width,
            image_height=image_height,
            gt_alpha_mask=None,
            image_name=image_name,
            uid=None,
            data_device='cuda',
        ))

    return see3d_viewpoints, see3d_cameras
