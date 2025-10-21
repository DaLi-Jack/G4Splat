import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import gradio
import numpy as np
import functools
import trimesh
import copy
from scipy.spatial.transform import Rotation
import tempfile
import shutil
import torch

from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess
from mast3r.image_pairs import make_pairs
from mast3r.retrieval.processor import Retriever

import mast3r.utils.path_to_dust3r  # noqa
from mast3r.model import AsymmetricMASt3R
from mast3r.utils.misc import hash_md5

from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.demo import get_args_parser as dust3r_get_args_parser

import cv2
from matcha.dm_utils.dataset_readers import read_intrinsics_binary, read_extrinsics_binary, qvec2rotmat, read_intrinsics_text, read_extrinsics_text
# from colmap.read_write_model import read_cameras_binary, read_images_binary, read_points3D_binary
from colmap.read_write_model import Camera, Image, Point3D,  write_cameras_binary, write_images_binary, write_points3D_binary, write_points3D_text, rotmat2qvec
from colmap.read_write_model import write_cameras_text, write_images_text, write_points3D_text
import open3d as o3d

import argparse
import json
import time


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description='Script to run MASt3R-SfM on a set of images.')
    
    # Input/Output
    parser.add_argument('-s', '--scene_path', type=str, 
                        help='Path to the scene data to use. Can be a directory containing images, or a colmap output directory.')
    parser.add_argument('-o', '--output_dir', type=str, default=None, 
                        help='Path to the output directory.')
    
    # Mast3r parameters
    parser.add_argument('--weights_path', type=str, 
                        default='./checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth', 
                        help='Path to the MASt3R model weights.')
    parser.add_argument('--retrieval_model', type=str, 
                        default='./checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth', 
                        help='Path to the retrieval model weights.')
    
    # Optimization parameters
    parser.add_argument('--min_conf_thr', type=float, default=0., help='Minimum confidence threshold.')
    parser.add_argument('--matching_conf_thr', type=float, default=0., help='Matching confidence threshold.')
    parser.add_argument('--n_coarse_iterations', type=int, default=1000, help='Number of coarse iterations.')
    parser.add_argument('--n_refinement_iterations', type=int, default=1000, help='Number of refinement iterations.')
    parser.add_argument('--TSDF_thresh', type=float, default=0., help='TSDF threshold.')
    parser.add_argument('--fix_focal', action='store_true', help='Fix camera focal length.')
    parser.add_argument('--fix_principal_point', action='store_true', help='Fix camera principal point.')
    parser.add_argument('--fix_rotation', action='store_true', help='Fix camera rotation.')
    parser.add_argument('--fix_translation', action='store_true', help='Fix camera translation.')
    
    # Data parameters
    parser.add_argument('--n_images', type=int, default=10, help='Number of images to use for optimization, sampled with constant spacing.')
    parser.add_argument('--use_all_images', action='store_true', help='Use all images for optimization.')
    parser.add_argument('--image_idx', type=int, nargs='*', default=None, 
        help='View indices to use for optimization (zero-based indexing). If provided, this will override the --n_images and --use_all_images arguments.')
    parser.add_argument('--randomize_images', action='store_true', help='Shuffle training images before sampling with constant spacing.')
    parser.add_argument('--image_size', type=int, default=512, help='Size of input images.')
    parser.add_argument('--max_window_size', type=int, default=20, help='Maximum window size.')
    parser.add_argument('--max_refid', type=int, default=10, help='Maximum reference image id.')
    parser.add_argument('--use_calibrated_poses', action='store_true', help='Use calibrated camera paramters in COLMAP format.')
    
    # Misc
    parser.add_argument('--save_glb', action='store_true', help='Save the optimized scene as a .glb file.')
    parser.add_argument('--gpu', type=int, default=0, help='Index of GPU device to use.')

    # Post processing    
    parser.add_argument('--output_conf_thr', type=float, default=.1, help='Confidence threshold for COLMAP format outputs.')
    parser.add_argument('--align_camera_locations', action='store_true', help='Align camera locations.')

    args = parser.parse_args()
    debug_verbose = False
    def print_debug(*args, **kwargs):
        if debug_verbose:
            print(*args, **kwargs)
    
    # If the user provides a directory containing images, use them.
    # If not, we assume this is a COLMAP dataset, 
    # and images should be located in the scene_path/images subdirectory.
    scene_path = args.scene_path
    use_calibrated_poses = args.use_calibrated_poses
    dir_content = os.listdir(scene_path)
    use_subdir = True
    n_all_images = 0
    for content in dir_content:
        if (
            content.endswith('.jpg') or content.endswith('.png') or content.endswith('.jpeg')
            or content.endswith('.JPG') or content.endswith('.PNG') or content.endswith('.JPEG')
        ):
            use_subdir = False
            n_all_images += 1
    if not use_subdir:
        print(f"{n_all_images} images found in {scene_path}.")
    else:
        dir_content = os.listdir(os.path.join(scene_path, "images"))
        for content in dir_content:
            if (
                content.endswith('.jpg') or content.endswith('.png') or content.endswith('.jpeg')
                or content.endswith('.JPG') or content.endswith('.PNG') or content.endswith('.JPEG')
            ):
                n_all_images += 1
        print(f"{n_all_images} images found in {scene_path}/images/.")
        if use_calibrated_poses:
            src_scale_mats = None
            if os.path.exists(f'{scene_path}/sparse/0'):
                try:
                    src_camera_data = read_intrinsics_binary(f'{scene_path}/sparse/0/cameras.bin')
                    src_image_data = read_extrinsics_binary(f'{scene_path}/sparse/0/images.bin')
                except:
                    src_camera_data = read_intrinsics_text(f'{scene_path}/sparse/0/cameras.txt')
                    src_image_data = read_extrinsics_text(f'{scene_path}/sparse/0/images.txt')

                src_intrinsics = {}
                src_extrinsics = {}
                for k in src_image_data:
                    img_name = src_image_data[k].name

                    cam = src_camera_data[src_image_data[k].camera_id]
                    if cam.model == 'PINHOLE':
                        fx,fy,cx,cy = cam.params
                    else:
                        raise NotImplementedError('Only PINHOLE model is supported for now.')
                    src_intrinsics[img_name] = np.array([
                        [fx, 0., cx],
                        [0., fy, cy],
                        [0., 0., 1.]
                    ])

                    T = np.eye(4)
                    T[:3,:3] = qvec2rotmat(src_image_data[k].qvec)
                    T[:3,3] = src_image_data[k].tvec
                    src_extrinsics[img_name] = T
            elif os.path.exists(f'{scene_path}/cameras.npz'):
                camera_dict = np.load(f'{scene_path}/cameras.npz')
                scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_all_images)]
                world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_all_images)]
                src_intrinsics = {}
                src_extrinsics = {}
                src_scale_mats = {}
                for k, img_name in enumerate(np.sort(os.listdir(f'{scene_path}/images'))):
                    scale_mat, world_mat = scale_mats[k], world_mats[k]
                    P = world_mat @ scale_mat
                    P = P[:3, :4]

                    K,R,t = cv2.decomposeProjectionMatrix(P)[:3]
                    K = K/K[2,2]
                    cam2w = np.eye(4, dtype=np.float32)
                    cam2w[:3, :3] = R.transpose()
                    cam2w[:3,3] = (t[:3] / t[3])[:,0]

                    src_intrinsics[img_name] = K
                    src_extrinsics[img_name] = np.linalg.inv(cam2w)
                    src_scale_mats[img_name] = scale_mat

            elif os.path.exists(f'{scene_path}/traj.txt'):
                print("Loading calibrated poses from Replica traj.txt")
                src_intrinsics = {}
                src_extrinsics = {}
                temp_img_path = os.path.join(scene_path, 'images', '000000_rgb.png')
                temp_img = cv2.imread(temp_img_path)
                height, width = temp_img.shape[:2]
                print(f"Height: {height}, Width: {width}")
                with open(f'{scene_path}/traj.txt', 'r') as f:
                    lines = f.readlines()
                for view_id in range(len(lines)):
                    img_name = f'{view_id:06d}_rgb.png'

                    # get camera to world matrix
                    line = lines[view_id].split()
                    c2w = np.array(list(map(float, line))).reshape(4, 4)
                    src_extrinsics[img_name] = np.linalg.inv(c2w)

                    # get intrinsics
                    focal_length_x = 600.0
                    focal_length_y = 600.0
                    src_intrinsics[img_name] = np.array([
                        [focal_length_x, 0., width/2],
                        [0., focal_length_y, height/2],
                        [0., 0., 1.]
                    ])

            elif os.path.exists(f'{scene_path}/transforms.json'):
                print("Loading calibrated poses from Blender / Mip-NeRF 360 transforms.json")
                src_intrinsics = {}
                src_extrinsics = {}
                with open(f'{scene_path}/transforms.json', 'r') as f:
                    data = json.load(f)
                frames = data['frames']
                for frame in frames:
                    img_name = frame['file_path'].split('/')[-1]

                    # NeRF 'transform_matrix' is a camera-to-world transform
                    c2w = np.array(frame["transform_matrix"])
                    # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
                    c2w[:3, 1:3] *= -1

                    src_extrinsics[img_name] = np.linalg.inv(c2w)
                    
                    # get intrinsics
                    cx = data['cx']
                    cy = data['cy']
                    fl_x = data['fl_x']
                    fl_y = data['fl_y']

                    src_intrinsics[img_name] = np.array([
                        [fl_x, 0., cx],
                        [0., fl_y, cy],
                        [0., 0., 1.]
                    ])
                    
            else:
                raise FileNotFoundError(f'Calibration data ({scene_path}/sparse/0/) not found.')

    image_dir = os.path.join(scene_path, "images") if use_subdir else scene_path
    
    # Parameters
    use_all_images = args.use_all_images
    n_images = n_all_images if use_all_images else args.n_images
    if args.image_idx is not None:
        use_all_images = False
        n_images = len(args.image_idx)
    image_size = args.image_size
    scenegraph_type = 'retrieval'
    min_conf_thr = args.min_conf_thr
    matching_conf_thr = args.matching_conf_thr
    
    winsize = min(args.max_window_size, n_images)  # 0 < winsize <= n_images
    refid = min(n_images - 1, args.max_refid)  # 1 <= refid <= n_images - 1
    
    # shared_intrinsics = True if not use_calibrated_poses else False
    shared_intrinsics = True
    cam_size = 0.2
    mask_sky = False
    clean_depth = True

    fix_focal = args.fix_focal
    fix_pp = args.fix_principal_point
    fix_rotation = args.fix_rotation
    fix_translation = args.fix_translation

    lr1 = 0.07
    niter1 = args.n_coarse_iterations
    lr2 = 0.01
    niter2 = args.n_refinement_iterations  # TODO: Try 500
    optim_level = 'refine+depth'

    TSDF_thresh = args.TSDF_thresh

    output_conf_thr = args.output_conf_thr

    current_scene_state = None
    gradio_delete_cache = False  # TODO: Check if this is correct
    
    if args.output_dir is None:
        output_name = f'mast3r_allimages' if use_all_images else f'mast3r_{n_images}images'
        output_dir = os.path.join(scene_path, output_name)
        temp_exp_name = scene_path.split('/')[-2]
        time_str = time.strftime("%Y-%m-%d-%H-%M-%S")
        exp_name = f'{temp_exp_name}_{time_str}'
    else:
        output_dir = args.output_dir
        exp_name = output_dir.split('/')[-2]
    os.makedirs(output_dir, exist_ok=True)
    _outdir = f'./tmp/{exp_name}'
    
    print_debug("========== PARAMETERS ==========")
    print_debug(f"Scene path: {scene_path}")
    print_debug(f"Image directory: {image_dir}")
    print_debug(f"Output directory: {output_dir}")
    print_debug(f"Number of images: {n_images}")
    print_debug(f"Use all images: {use_all_images}")
    print_debug(f"Image size: {image_size}")
    print_debug(f"Scene graph type: {scenegraph_type}")
    print_debug(f"Minimum confidence threshold: {min_conf_thr}")
    print_debug(f"Matching confidence threshold: {matching_conf_thr}")
    print_debug(f"Window size: {winsize}")
    print_debug(f"Reference image id: {refid}")
    print_debug(f"Shared intrinsics: {shared_intrinsics}")
    print_debug(f"Camera size: {cam_size}")
    print_debug(f"Mask sky: {mask_sky}")
    print_debug(f"Clean depth: {clean_depth}")
    print_debug(f"Learning rate 1: {lr1}")
    print_debug(f"Number of coarse iterations: {niter1}")
    print_debug(f"Learning rate 2: {lr2}")
    print_debug(f"Number of refinement iterations: {niter2}")
    print_debug(f"Optimization level: {optim_level}")
    print_debug(f"TSDF threshold: {TSDF_thresh}")
    print_debug(f"Output confidence threshold: {output_conf_thr}")
    print_debug("================================\n")
    
    # TODO
    print_debug("\n[WARNING] This script needs to be updated to return data as float32.")
    
    # Helper functions
    class SparseGAState:
        def __init__(self, sparse_ga, should_delete=False, cache_dir=None, outfile_name=None):
            self.sparse_ga = sparse_ga
            self.cache_dir = cache_dir
            self.outfile_name = outfile_name
            self.should_delete = should_delete

        def __del__(self):
            if not self.should_delete:
                return
            if self.cache_dir is not None and os.path.isdir(self.cache_dir):
                shutil.rmtree(self.cache_dir)
            self.cache_dir = None
            if self.outfile_name is not None and os.path.isfile(self.outfile_name):
                os.remove(self.outfile_name)
            self.outfile_name = None

            
    def _convert_scene_output_to_glb(outfile, imgs, pts3d, mask, focals, cams2world, cam_size=0.05,
                                    cam_color=None, as_pointcloud=False,
                                    transparent_cams=False, silent=False):
        assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
        pts3d = to_numpy(pts3d)
        imgs = to_numpy(imgs)
        focals = to_numpy(focals)
        cams2world = to_numpy(cams2world)

        scene = trimesh.Scene()

        # full pointcloud
        if as_pointcloud:
            pts = np.concatenate([p[m.ravel()] for p, m in zip(pts3d, mask)]).reshape(-1, 3)
            col = np.concatenate([p[m] for p, m in zip(imgs, mask)]).reshape(-1, 3)
            valid_msk = np.isfinite(pts.sum(axis=1))
            pct = trimesh.PointCloud(pts[valid_msk], colors=col[valid_msk])
            scene.add_geometry(pct)
        else:
            meshes = []
            for i in range(len(imgs)):
                pts3d_i = pts3d[i].reshape(imgs[i].shape)
                msk_i = mask[i] & np.isfinite(pts3d_i.sum(axis=-1))
                meshes.append(pts3d_to_trimesh(imgs[i], pts3d_i, msk_i))
            mesh = trimesh.Trimesh(**cat_meshes(meshes))
            scene.add_geometry(mesh)

        # add each camera
        for i, pose_c2w in enumerate(cams2world):
            if isinstance(cam_color, list):
                camera_edge_color = cam_color[i]
            else:
                camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
            add_scene_cam(scene, pose_c2w, camera_edge_color,
                        None if transparent_cams else imgs[i], focals[i],
                        imsize=imgs[i].shape[1::-1], screen_width=cam_size)

        rot = np.eye(4)
        rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
        scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
        if not silent:
            print('(exporting 3D scene to', outfile, ')')
        scene.export(file_obj=outfile)
        return outfile

    # ========== Start MASt3R-SfM ==========

    # Set device
    torch.cuda.set_device(args.gpu)
    device = torch.device(torch.cuda.current_device())

    # Load MASt3R model
    print('\nLoading MASt3R model...')
    model = AsymmetricMASt3R.from_pretrained(args.weights_path).to(device)
    chkpt_tag = hash_md5(args.weights_path)
    print('MASt3R model loaded.')
    
    # Load images
    print('\nLoading images...')
    all_image_names = np.sort(os.listdir(image_dir))
    if args.image_idx is not None:
        image_names = [all_image_names[i] for i in args.image_idx]
    elif use_all_images:
        image_names = all_image_names.tolist()
    else:
        # image_names = all_image_names[::len(all_image_names) // (n_images - 1)].tolist()  # Better
        image_names = [all_image_names[i * (len(all_image_names) // (n_images - 1))] for i in range(n_images)]
        # image_names = all_image_names[::math.ceil(len(all_image_names) / n_images)].tolist()  # Better

    filelist = [os.path.join(image_dir, image_name) for image_name in image_names]
    silent = True
    imgs = load_images(filelist, size=image_size, verbose=not silent)
    print(f'Loaded {len(imgs)} images.')
    print(f'Images have shape: {imgs[0]["img"].shape}')
    
    if not use_calibrated_poses:
        os.makedirs(f'{output_dir}/images', exist_ok=True)
        for idx_img, img_path in enumerate(filelist):
            img_fname = img_path.split('/')[-1]
            src_img = cv2.imread(img_path, 1)

            # ==========ADDED: Start by Cropping the original image==========
            print_debug(f"\nSource image has shape: {src_img.shape}")
            if True:
                # Get size of MASt3R images
                _height_resized, _width_resized = imgs[idx_img]['img'].size()[-2:]
                
                # Get original image size
                _height_original, _width_original = src_img.shape[:2]                
                
                # Get MASt3R image proportions
                _width_crop = int(round(max(_height_original, _width_original) / image_size * _width_resized))
                _height_crop = int(round(max(_height_original, _width_original) / image_size * _height_resized))
                
                print_debug(f"Width crop: {_width_crop}")
                print_debug(f"Height crop: {_height_crop}")
                
                # Update image and intrinsics
                print_debug(f"Source image shape (before cropping): {src_img.shape}")
                
                src_img = src_img[_height_original//2 - _height_crop//2:_height_original//2 + _height_crop//2, 
                                   _width_original//2 - _width_crop//2:_width_original//2 + _width_crop//2]
                print_debug(f"Source image shape (after cropping): {src_img.shape}")
                cv2.imwrite(f'{output_dir}/images/{img_fname}', src_img)
                filelist[idx_img] = f'{output_dir}/images/{img_fname}'
                
        imgs = load_images(filelist, size=image_size, verbose=not silent)
        print(f'Images have shape: {imgs[0]["img"].shape}')
    
    if use_calibrated_poses:
        print("[INFO] Using calibrated poses.")
        intrinsics = np.stack([src_intrinsics[img_name] for img_name in image_names], axis=0)
        extrinsics = np.stack([src_extrinsics[img_name] for img_name in image_names], axis=0)
        calibrated_img_sizes = []

        print_debug('original intrinsics')
        print_debug(intrinsics)
        print_debug('original extrinsics')
        print_debug(extrinsics)

        # copy instance masks and instance id json files
        src_instance_json_path = os.path.join(scene_path, 'instance_id.json')
        if os.path.exists(src_instance_json_path):
            print("Exist instance annotation file, copy it to the output directory.")
            dst_instance_json_path = os.path.join(output_dir, 'instance_id.json')
            shutil.copy(src_instance_json_path, dst_instance_json_path)

            src_instance_mask_path = os.path.join(scene_path, 'instance_masks')
            dst_instance_mask_path = os.path.join(output_dir, 'instance_masks')
            shutil.copytree(src_instance_mask_path, dst_instance_mask_path)
        else:
            print("No instance annotation file found, skip copying.")

        # Modify images and intrinsics so that the principal points become the center of the images, before feeding them to MASt3R
        src_img_path = os.path.join(scene_path, 'images')
        dst_img_path = os.path.join(output_dir, 'images')
        if not os.path.exists(dst_img_path):
            shutil.copytree(src_img_path, dst_img_path)
        # os.makedirs(f'{output_dir}/images', exist_ok=True)
        for idx_img, img_path in enumerate(filelist):
            img_fname = img_path.split('/')[-1]
            src_img = cv2.imread(img_path, 1)
            src_K = intrinsics[idx_img]
            
            # ==========ADDED==========
            src_height, src_width = src_img.shape[:2]
            src_dist = np.array([0., 0., 0., 0.]) # no distortion
            src_pp = src_K[:2,2]

            # Change the principal point so that it becomes the center of the image.
            tar_pp = np.array([
                int(min(src_pp[0], src_width - src_pp[0])), 
                int(min(src_pp[1], src_height - src_pp[1])), 
            ])
            
            print_debug(f"Tar pp: {tar_pp}")
            centered_height = 2 * tar_pp[1]
            centered_width = 2 * tar_pp[0]
            print_debug(f"Height centered: {centered_height}")
            print_debug(f"Width centered: {centered_width}")
            height_resized = imgs[idx_img]['img'].shape[-2]
            width_resized = imgs[idx_img]['img'].shape[-1]
            print_debug(f"Height resized: {height_resized}")
            print_debug(f"Width resized: {width_resized}")
            width_crop = int(round(max(centered_height, centered_width) / image_size * width_resized))
            height_crop = int(round(max(centered_height, centered_width) / image_size * height_resized))
            tar_pp = np.array([width_crop//2, height_crop//2])
            print_debug(f"New tar pp: {tar_pp}")
            # ==========END==========
            
            # Change the focals so that fx and fy become identical after resizing and cropping in load_images().
            w_, h_ = tar_pp * 2
            s_w = int(round(w_*image_size/max(w_, h_))) / w_
            s_h = int(round(h_*image_size/max(w_, h_))) / h_
            tar_focal = 0.5 * (src_K[0,0] + src_K[1,1])
            tar_focal_x = tar_focal * np.sqrt(s_w * s_h) / s_w
            tar_focal_y = tar_focal * np.sqrt(s_w * s_h) / s_h
            tar_K = src_K.copy()
            tar_K[:2,2] = tar_pp
            tar_K[0,0] = tar_focal_x
            tar_K[1,1] = tar_focal_y
            map1, map2 = cv2.initUndistortRectifyMap(
                src_K, src_dist, None, 
                tar_K, 2 * tar_pp,
                cv2.CV_32FC2
            )
            new_img = cv2.remap(
                src_img, 
                map1, map2, 
                interpolation=cv2.INTER_LANCZOS4, # cv2.INTER_LINEAR
                borderMode=cv2.BORDER_REPLICATE, # cv2.BORDER_CONSTANT
            )
            calibrated_img_sizes.append((new_img.shape[0], new_img.shape[1]))
            
            # ==========ADDED: Start by Cropping the original image==========
            print_debug(f"\nSource image has shape: {src_img.shape}")
            print_debug(f"Calibrated image has shape: {new_img.shape}")
            
            # ==========ADDED========== 
            
            print_debug(f"Source image has shape: {src_img.shape}")
            print_debug(f"Final image has shape: {new_img.shape}")
            # cv2.imwrite(f'{output_dir}/images/{img_fname}', new_img)

            intrinsics[idx_img] = tar_K
            filelist[idx_img] = f'{output_dir}/images/{img_fname}'
        imgs = load_images(filelist, size=image_size, verbose=not silent)
        print_debug(f'Images have shape: {imgs[0]["img"].shape}')

        print_debug('intrinsics after cropping')
        print_debug(intrinsics)

        # Modify intrinsics according to MASt3R resizing
        for idx_img, img in enumerate(imgs):
            # Get size of MASt3R images
            height_resized, width_resized = img['img'].size()[-2:]
            
            # Load original rectified images to get the original size
            new_img = cv2.imread(filelist[idx_img], 1)
            height_original, width_original = new_img.shape[:2]
            # height_original, width_original = calibrated_img_sizes[idx_img]

            # Consider cropping inside load_images()!!!
            width_resized_simple = int(round(width_original*image_size/max(width_original, height_original)))
            height_resized_simple = int(round(height_original*image_size/max(width_original, height_original)))
            ofs_u = width_resized_simple//2 - width_resized//2
            ofs_v = height_resized_simple//2 - height_resized//2

            # Scaling focals and principal points
            intrinsics[idx_img][0,:] *= (width_resized_simple / width_original)
            intrinsics[idx_img][1,:] *= (height_resized_simple / height_original)
            # Adjusting principal points. Centered cropping, so the principal point is still centered
            intrinsics[idx_img][0,2] -= ofs_u  
            intrinsics[idx_img][1,2] -= ofs_v
            
            # ==========ADDED: Save Cropped Image==========
            if True and False:
                from torchvision.transforms.functional import center_crop
                # Get MASt3R image proportions
                width_crop = int(round(max(height_original, width_original) / image_size * width_resized))
                height_crop = int(round(max(height_original, width_original) / image_size * height_resized))
                
                print_debug(f"Width crop: {width_crop}")
                print_debug(f"Height crop: {height_crop}")
                if True:
                    cropped_img = center_crop(
                        torch.tensor(new_img, device=device).permute(2, 0, 1),
                        output_size=(height_crop, width_crop)
                    ).permute(1, 2, 0).cpu().numpy()
                else:
                    src_img = src_img[_height_original//2 - _height_crop//2:_height_original//2 + _height_crop//2, 
                                   _width_original//2 - _width_crop//2:_width_original//2 + _width_crop//2]
                cv2.imwrite(filelist[idx_img], cropped_img)
                print_debug(f"Image saved to {filelist[idx_img]}")
                
                # Update image and intrinsics
                print_debug(f"Cropped image shape: {cropped_img.shape}")       
                
                new_resized_shape = load_images([filelist[idx_img]], size=image_size, verbose=False)[0]['img'][0].shape
                print_debug(f"New resized shape: {new_resized_shape}")
                if new_resized_shape[-2] != height_resized or new_resized_shape[-1] != width_resized:
                    raise ValueError(f"Cropped image shape {new_resized_shape} does not match MASt3R resized shape {(height_resized, width_resized)}")
            # ==========END==========

        imgs = load_images(filelist, size=image_size, verbose=not silent)
        print_debug(f'Images have shape: {imgs[0]["img"].shape}')
        
        print_debug('intrinsics after resizing')
        print_debug(intrinsics)

    # Load retrieval model
    scene_graph_params = [scenegraph_type]
    if scenegraph_type in ["swin", "logwin"]:
        scene_graph_params.append(str(winsize))
    elif scenegraph_type == "oneref":
        scene_graph_params.append(str(refid))
    elif scenegraph_type == "retrieval":
        scene_graph_params.append(str(winsize))  # Na
        scene_graph_params.append(str(refid))  # k
    scene_graph = '-'.join(scene_graph_params)

    sim_matrix = None
    if 'retrieval' in scenegraph_type:
        assert args.retrieval_model is not None
        retriever = Retriever(args.retrieval_model, backbone=model, device=device)
        with torch.no_grad():
            sim_matrix = retriever(filelist)
        # Cleanup
        del retriever
        torch.cuda.empty_cache()
        
    # Build scene graph
    print('\nStart building scene graph...')
    t0 = time.time()
    
    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True, sim_mat=sim_matrix)
    if optim_level == 'coarse':
        niter2 = 0
    # Sparse GA (forward mast3r -> matching -> 3D optim -> 2D refinement -> triangulation)
    if current_scene_state is not None and \
        not current_scene_state.should_delete and \
            current_scene_state.cache_dir is not None:
        cache_dir = current_scene_state.cache_dir
    elif gradio_delete_cache:
        cache_dir = tempfile.mkdtemp(suffix='_cache', dir=_outdir)
    else:
        cache_dir = os.path.join(_outdir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    print('Done building scene graph.')
    
    # Run sparse global alignment
    print('Start global alignment...')
    kw = {
        'init': {},
        'opt_pp': True if not fix_pp else False,
        'opt_focal': True if not fix_focal else False,
        'opt_quat': True if not fix_rotation else False,
        'opt_tran': True, # if not fix_translation else False, # For now, this does not work well.
        'opt_size': True, # if not fix_translation else False, # Instead, we do post alignment.
    }
    if use_calibrated_poses:
        for idx_img, img_path in enumerate(filelist):
            kw['init'][img_path] = {
                'intrinsics': torch.from_numpy(intrinsics[idx_img].astype(np.float32)).to(device),
                'cam2w': torch.inverse(torch.from_numpy(extrinsics[idx_img].astype(np.float32)).to(device)),
            }

    scene = sparse_global_alignment(filelist, pairs, cache_dir,
                                    model, lr1=lr1, niter1=niter1, lr2=lr2, niter2=niter2, device=device,
                                    opt_depth='depth' in optim_level, shared_intrinsics=shared_intrinsics,
                                    matching_conf_thr=matching_conf_thr, **kw)  # TODO: Check that removing kw is fine
    if current_scene_state is not None and \
        not current_scene_state.should_delete and \
            current_scene_state.outfile_name is not None:
        outfile_name = current_scene_state.outfile_name
    else:
        outfile_name = tempfile.mktemp(suffix='_scene.glb', dir=_outdir)
    scene_state = SparseGAState(scene, gradio_delete_cache, cache_dir, outfile_name)
    print('Done global alignment.')
    print(f'Optimization complete in {time.time() - t0:.2f}s.')
    
    if scene_state is None:
        print('No scene state')
    outfile = scene_state.outfile_name
    if outfile is None:
        print('No outfile')

    # Get optimized values from scene
    print('Saving optimized scene...')
    scene = scene_state.sparse_ga
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    pps = scene.get_principal_points().cpu()
    cams2world = scene.get_im_poses().cpu()

    # 3D pointcloud from depthmap, poses and intrinsics
    if TSDF_thresh > 0:
        print('TSDF post-processing...')
        tsdf = TSDFPostProcess(scene, TSDF_thresh=TSDF_thresh)
        pts3d, _, confs = to_numpy(tsdf.get_dense_pts3d(clean_depth=clean_depth))
    else:
        pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=clean_depth))

    # Align the recovered camera locations with calibrated ones
    if use_calibrated_poses and args.align_camera_locations:
        print('[INFO] Aligning camera locations...')
        calib_cam_locations = []
        est_cam_locations = []
        for idx_view, img_name in enumerate(image_names):
            calib_cam_locations.append(np.linalg.inv(src_extrinsics[img_name])[:3,3])
            est_cam_locations.append(cams2world[idx_view][:3,3].cpu().numpy())
        calib_cam_locations = np.stack(calib_cam_locations, axis=0)
        est_cam_locations = np.stack(est_cam_locations, axis=0)

        if True:#fix_rotation:
            # Modify only scale and translation
            x = est_cam_locations - np.mean(est_cam_locations, axis=0, keepdims=True)
            y = calib_cam_locations - np.mean(calib_cam_locations, axis=0, keepdims=True)
            # minimize ||y - scale * x||^2 w.r.t scale
            # scale = \sum{xy} / \sum{x^2}
            scale_est2calib = np.sum(x * y) / np.sum(x**2)
            ofs_est2calib = np.mean(calib_cam_locations, axis=0) - scale_est2calib * np.mean(est_cam_locations, axis=0)

            mean_residual = np.mean(np.linalg.norm(scale_est2calib * est_cam_locations + ofs_est2calib - calib_cam_locations, axis=-1))
            print_debug('Scale and translation offsets:', scale_est2calib, ofs_est2calib)
            print_debug('Mean residual of camera locations:', mean_residual)

            # Modify camera poses and 3D points
            cams2world[:,:3,3] = scale_est2calib * cams2world[:,:3,3] + torch.from_numpy(ofs_est2calib).to(cams2world.device)
            for i_ in range(len(pts3d)):
                pts3d[i_] = scale_est2calib * pts3d[i_] + ofs_est2calib

        else:
            # Modify rotation, scale, and translaion
            raise NotImplementedError()

        if fix_rotation and fix_translation:
            cams2world = torch.from_numpy(np.linalg.inv(extrinsics))
        elif fix_translation:
            cams2world[:,:3,3] = torch.from_numpy(np.linalg.inv(extrinsics))[:,:3,3]
        elif fix_rotation:
            cams2world[:,:3,:3] = torch.from_numpy(np.linalg.inv(extrinsics))[:,:3,:3]

        print_debug('output extrinsics:')
        print_debug(torch.inverse(cams2world).numpy())

        # Unnormalize world scale and translaion for data with cameras.npz
        if src_scale_mats is not None:
            for i_ in range(len(pts3d)):
                S = src_scale_mats[image_names[i_]]
                cam_location = cams2world[i_:i_+1,:3,3].cpu().numpy()
                new_cam_location = (S[:3,:3] @ cam_location.T + S[:3,3:4]).T
                cams2world[i_:i_+1,:3,3] = torch.from_numpy(new_cam_location).to(cams2world.device)
                pts3d[i_] = (S[:3,:3] @ pts3d[i_].T + S[:3,3:4]).T
    
    # Save cameras    
    output_cameras = {
        'filepaths': [img['instance'] for img in imgs],
        'focals': focals.numpy().tolist(),
        'cams2world': cams2world.numpy().tolist(),
    }
    with open(os.path.join(output_dir, 'cameras.json'), 'w') as f:
        json.dump(output_cameras, f)
    

    cameras_colmap = {}
    images_colmap = {}
    points3d_colmap = {}
    points3d_id_ofs = 1
    all_xyzs = []
    all_rgbs = []
    for idx_img in range(len(imgs)):
        camera_id = idx_img + 1

        # intrinsics
        height_original, width_original = cv2.imread(imgs[idx_img]['instance'],0).shape[:2]
        width_resized = int(imgs[idx_img]['img'].size(-1))
        height_resized = int(imgs[idx_img]['img'].size(-2))

        # Consider cropping inside load_images()!!!
        width_resized_simple = int(round(width_original*image_size/max(width_original, height_original)))
        height_resized_simple = int(round(height_original*image_size/max(width_original, height_original)))
        ofs_u = width_resized_simple//2 - width_resized//2
        ofs_v = height_resized_simple//2 - height_resized//2

        scale_w = width_original / width_resized_simple
        scale_h = height_original / height_resized_simple
        cameras_colmap[camera_id] = Camera(
            int(camera_id), 
            'PINHOLE', 
            width=width_original, 
            height=height_original, 
            params=np.array([
                focals[idx_img].item() * scale_w,
                focals[idx_img].item() * scale_h,
                (pps[idx_img][0].item() + ofs_u) * scale_w,
                (pps[idx_img][1].item() + ofs_v) * scale_h,
            ])
        )

        # extrinsics
        w2c = np.linalg.inv(cams2world[idx_img].numpy())

        # points
        pixels = np.mgrid[:width_resized, :height_resized].T.reshape(-1, 2)
        pixels_original = pixels * np.array([scale_w, scale_h])
        point3d_ids = []
        xys = []
        img_original = cv2.imread(imgs[idx_img]['instance'],1)[...,::-1]
        # TODO: avoid slow for-loop
        for idx_pt2d, (xyz, pt2d) in enumerate(zip(pts3d[idx_img], pixels_original)):
            pixel_conf = confs[idx_img][pixels[idx_pt2d][1], pixels[idx_pt2d][0]]
            if pixel_conf < output_conf_thr: # ignore points with low confidence
                continue

            point3d_id = points3d_id_ofs + idx_pt2d
            xys.append(pt2d)
            point3d_ids.append(point3d_id)

            rgb = img_original[int(np.clip(pt2d[1],0,height_original)), int(np.clip(pt2d[0],0,width_original))]
            error = np.array(2., dtype=np.float64) # TODO: use actual reprojection error?

            image_ids = np.array([camera_id,], dtype=np.int64)
            point2D_idxs=np.array([idx_pt2d,], dtype=np.int64)

            points3d_colmap[point3d_id] = Point3D(
                id=point3d_id,
                xyz=xyz.astype(np.float64),
                rgb=rgb.astype(np.int64),
                error=error,
                image_ids=image_ids,
                point2D_idxs=point2D_idxs,
            )
            all_xyzs.append(xyz.astype(np.float64))
            all_rgbs.append(rgb.astype(np.float64) / 255.)
        points3d_id_ofs += len(pts3d[idx_img])

        images_colmap[camera_id] = Image(
            camera_id, # id
            rotmat2qvec(w2c[:3,:3]).astype(np.float64), # qvec
            w2c[:3,3].astype(np.float64), # tvec
            camera_id, # camera_model_id
            imgs[idx_img]['instance'].split('/')[-1], # name
            np.array(xys, dtype=np.float64), # xys
            np.array(point3d_ids, dtype=np.int64) # point3D_ids
        )

    os.makedirs(f'{output_dir}/sparse/0', exist_ok=True)
    write_cameras_binary(cameras_colmap, f'{output_dir}/sparse/0/cameras.bin')
    write_images_binary(images_colmap, f'{output_dir}/sparse/0/images.bin')
    write_points3D_binary(points3d_colmap, f'{output_dir}/sparse/0/points3D.bin')
    if True:
        write_cameras_text(cameras_colmap, f'{output_dir}/sparse/0/cameras.txt')
        write_images_text(images_colmap, f'{output_dir}/sparse/0/images.txt')
        write_points3D_text(points3d_colmap, f'{output_dir}/sparse/0/points3D.txt')

    if use_calibrated_poses:
        # save eval cameras
        save_all_cameras_path = f'{output_dir}/all-sparse/0'
        os.makedirs(save_all_cameras_path, exist_ok=True)
        train_image_names_set = set(image_names)

        all_cameras_colmap = {}
        all_images_colmap = {}

        for idx_img, img_name in enumerate(all_image_names):
            camera_id = idx_img + 1
            img_path = os.path.join(image_dir, img_name)

            img_original = cv2.imread(img_path, 0)
            height_original, width_original = img_original.shape[:2]

            K = src_intrinsics[img_name]
            w2c = src_extrinsics[img_name]

            # camera intrinsics
            all_cameras_colmap[camera_id] = Camera(
                int(camera_id), 
                'PINHOLE', 
                width=width_original, 
                height=height_original, 
                params=np.array([
                    K[0, 0],  # fx
                    K[1, 1],  # fy
                    K[0, 2],  # cx
                    K[1, 2],  # cy
                ])
            )

            # camera extrinsics
            all_images_colmap[camera_id] = Image(
                camera_id,  # id
                rotmat2qvec(w2c[:3, :3]).astype(np.float64),  # qvec
                w2c[:3, 3].astype(np.float64),  # tvec
                camera_id,  # camera_model_id
                img_name,  # name
                np.array([], dtype=np.float64),  # xys
                np.array([], dtype=np.int64)  # point3D_ids
            )

        write_cameras_binary(all_cameras_colmap, f'{save_all_cameras_path}/cameras.bin')
        write_images_binary(all_images_colmap, f'{save_all_cameras_path}/images.bin')

        write_cameras_text(all_cameras_colmap, f'{save_all_cameras_path}/cameras.txt')
        write_images_text(all_images_colmap, f'{save_all_cameras_path}/images.txt')

    dense_view_json_path = os.path.join(scene_path, 'dense_view.json')
    if os.path.exists(dense_view_json_path):
        print(f'[INFO]: Load dense view json in {dense_view_json_path}')

        with open(dense_view_json_path, 'r') as f:
            dense_view_json = json.load(f)
        dense_view_idx_list = dense_view_json['train']
        print(f"Use {len(dense_view_idx_list)} views from {dense_view_json_path} as dense view")

        save_dense_cameras_path = f'{output_dir}/dense-view-sparse/0'
        os.makedirs(save_dense_cameras_path, exist_ok=True)

        dense_cameras_colmap = {}
        dense_images_colmap = {}
        dense_image_names = all_image_names[dense_view_idx_list]

        for idx_img, img_name in enumerate(dense_image_names):
            camera_id = idx_img + 1
            img_path = os.path.join(image_dir, img_name)

            img_original = cv2.imread(img_path, 0)
            height_original, width_original = img_original.shape[:2]

            K = src_intrinsics[img_name]
            w2c = src_extrinsics[img_name]

            dense_cameras_colmap[camera_id] = Camera(
                int(camera_id), 
                'PINHOLE', 
                width=width_original, 
                height=height_original, 
                params=np.array([
                    K[0, 0],  # fx
                    K[1, 1],  # fy
                    K[0, 2],  # cx
                    K[1, 2],  # cy
                ])
            )

            dense_images_colmap[camera_id] = Image(
                camera_id,  # id
                rotmat2qvec(w2c[:3, :3]).astype(np.float64),  # qvec
                w2c[:3, 3].astype(np.float64),  # tvec
                camera_id,  # camera_model_id
                img_name,  # name
                np.array([], dtype=np.float64),  # xys
                np.array([], dtype=np.int64)  # point3D_ids
            )

        write_cameras_binary(dense_cameras_colmap, f'{save_dense_cameras_path}/cameras.bin')
        write_images_binary(dense_images_colmap, f'{save_dense_cameras_path}/images.bin')

        write_cameras_text(dense_cameras_colmap, f'{save_dense_cameras_path}/cameras.txt')
        write_images_text(dense_images_colmap, f'{save_dense_cameras_path}/images.txt')

    # save 
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.stack(all_xyzs, axis=0))
    pcd.colors = o3d.utility.Vector3dVector(np.stack(all_rgbs, axis=0))
    o3d.io.write_point_cloud(f"{output_dir}/points.ply", pcd)

    # Save pointmaps
    pointmaps_dir = os.path.join(output_dir, 'pointmaps')
    os.makedirs(pointmaps_dir, exist_ok=True)
    for i_image in range(len(imgs)):
        img_path = imgs[i_image]['instance']
        img_name = os.path.basename(img_path).split('.')[0]
        output_pointmap = {
            'rgb': None if use_all_images else rgbimg[i_image].tolist(),
            'points': pts3d[i_image].tolist(),
            'confs': confs[i_image].tolist(),
        }
        with open(os.path.join(pointmaps_dir, f'{img_name}.json'), 'w') as f:
            json.dump(output_pointmap, f)
        
    if args.save_glb:
        as_pointcloud = True
        transparent_cams = False
        msk = to_numpy([c > min_conf_thr for c in confs])
        _convert_scene_output_to_glb(
            # outfile, 
            os.path.join(output_dir, 'scene.glb'),
            rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
            transparent_cams=transparent_cams, cam_size=cam_size, silent=silent
        )
    print('Scene saved.')
    
    # Remove temporary files
    print('\nCleaning up temporary files...')
    shutil.rmtree(_outdir)
    print('Temporary files cleaned up.')
    
    print('\nInitial SfM complete.')
