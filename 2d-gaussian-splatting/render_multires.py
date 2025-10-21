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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from scene import Scene
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh
from utils.render_utils import generate_path, create_videos

import open3d as o3d

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_mesh", action="store_true")
    parser.add_argument("--render_all_img", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_path", action="store_true")
    parser.add_argument("--voxel_size", default=-1.0, type=float, help='Mesh: voxel size for TSDF')
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')
    parser.add_argument("--num_cluster", default=50, type=int, help='Mesh: number of connected clusters to export')
    parser.add_argument("--unbounded", action="store_true", help='Mesh: using unbounded mode for meshing')
    parser.add_argument("--mesh_res", default=1024, type=int, help='Mesh: resolution for unbounded mesh extraction')
    parser.add_argument("--multires_factors", default=[2,8,16], nargs='+', type=int, help='Mesh: multiresolution factors')
    parser.add_argument("--output_dir", type=str, default=None, help='Path to save the output mesh.')
    parser.add_argument("--use_default_output_dir", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)


    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    if args.render_all_img:
        dataset.eval = True
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    if args.use_default_output_dir:
        train_dir = os.path.join(args.model_path, 'train', "ours_{}".format(scene.loaded_iter))
    else:
        train_dir = args.output_dir
    test_dir = os.path.join(args.model_path, 'test', "ours_{}".format(scene.loaded_iter))
    gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)    
    
    if not args.skip_train:
        print("export training images ...")
        os.makedirs(train_dir, exist_ok=True)
        gaussExtractor.reconstruction(scene.getTrainCameras())
        gaussExtractor.export_image(train_dir)
        
    
    if (not args.skip_test) and (len(scene.getTestCameras()) > 0):
        print("export rendered testing images ...")
        os.makedirs(test_dir, exist_ok=True)
        gaussExtractor.reconstruction(scene.getTestCameras())
        gaussExtractor.export_image(test_dir)
    
    
    if args.render_path:
        print("render videos ...")
        traj_dir = os.path.join(args.model_path, 'traj', "ours_{}".format(scene.loaded_iter))
        os.makedirs(traj_dir, exist_ok=True)
        n_fames = 240
        cam_traj = generate_path(scene.getTrainCameras(), n_frames=n_fames)
        gaussExtractor.reconstruction(cam_traj)
        gaussExtractor.export_image(traj_dir)
        create_videos(base_dir=traj_dir,
                    input_dir=traj_dir, 
                    out_name='render_traj', 
                    num_frames=n_fames)

    if not args.skip_mesh:
        print("export mesh ...")
        os.makedirs(train_dir, exist_ok=True)
        # set the active_sh to 0 to export only diffuse texture
        gaussExtractor.gaussians.active_sh_degree = 0
        gaussExtractor.reconstruction(scene.getTrainCameras())
        
        
        multires_factors = args.multires_factors
        depth_truncs = []
        meshes = []
        scene_cameras = scene.getTrainCameras()
        from matcha.dm_scene.cameras import CamerasWrapper, GSCamera
        gs_cameras = []
        for scene_camera in scene_cameras:
            gs_cameras.append(GSCamera(
                colmap_id=scene_camera.colmap_id,
                R=scene_camera.R,
                T=scene_camera.T,
                FoVx=scene_camera.FoVx,
                FoVy=scene_camera.FoVy,
                image=scene_camera.original_image,
                gt_alpha_mask=scene_camera.gt_alpha_mask,
                image_name=scene_camera.image_name,
                uid=scene_camera.uid,
                data_device=scene_camera.data_device,
                image_height=scene_camera.image_height,
                image_width=scene_camera.image_width,
            ))
        cameras_wrapper = CamerasWrapper(gs_cameras)
        
        # extract the mesh and save
        for factor in multires_factors:
            print(f'\nExtracting mesh with factor {factor}...')
            depth_trunc = (gaussExtractor.radius * factor)
            voxel_size = depth_trunc / args.mesh_res
            sdf_trunc = 5.0 * voxel_size
            mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
            meshes.append(mesh)
            depth_truncs.append(depth_trunc)
            print(f'Mesh extracted with depth truncation {depth_trunc} and voxel size {voxel_size}.')
        
        p3d_meshes = []
        device = 'cuda'
        import numpy as np
        from matcha.dm_scene.meshes import Meshes, TexturesVertex, render_mesh_with_pytorch3d, remove_faces_from_single_mesh, join_meshes_as_scene
        print("\n===Merging multi-resolution meshes===")
        previous_pix_to_face = None
        current_pix_to_face = None
        for i_mesh, (depth_trunc, mesh) in enumerate(zip(depth_truncs, meshes)):
            print(f"Processing mesh with depth truncation {depth_trunc}...")
            
            verts = torch.from_numpy(np.asarray(mesh.vertices)).float().to(device)
            faces = torch.from_numpy(np.asarray(mesh.triangles)).long().to(device)
            vert_colors = torch.from_numpy(np.asarray(mesh.vertex_colors)).float().to(device)
            
            p3d_mesh = Meshes(
                verts=[verts], 
                faces=[faces],
                textures=TexturesVertex([vert_colors]),
            )
            empty_mesh = False
            
            # Identify which faces from lower resolutions are necessary to keep
            necessary_faces = torch.zeros(faces.shape[0], dtype=torch.bool, device=device)
            
            # Removing faces in the field of view of the cameras, but with depth below the truncation threshold
            if i_mesh > 0:
                # Check which vertices are in the field of view...
                projections = cameras_wrapper.project_points(verts.view(1, -1, 3))  # (n_cameras, n_verts, 2)
                height, width = cameras_wrapper.gs_cameras[0].image_height, cameras_wrapper.gs_cameras[0].image_width
                factors = torch.tensor([[[-width / min(height, width), -height / min(height, width)]]], device=projections.device)  # (1, 1, 2)
                projections = projections / factors  # (n_cameras, n_verts, 2)
                visible_mask = (projections[..., 0] > -1.0) & (projections[..., 0] < 1.0) & (projections[..., 1] > -1.0) & (projections[..., 1] < 1.0)  # (n_cameras, n_verts)
                
                # ... and which are close to the camera
                depths = cameras_wrapper.transform_points_world_to_view(verts.view(1, -1, 3))[..., 2]  # (n_cameras, n_verts)
                close_verts = (depths < depth_truncs[i_mesh - 1])
                
                non_valid_verts = (visible_mask & close_verts).any(dim=0)  # (n_verts)
                non_valid_faces = non_valid_verts[faces].all(dim=-1)  # Should it be any or all? It depends on what we want to remove.

                # Remove the faces corresponding faces
                try:
                    p3d_mesh = remove_faces_from_single_mesh(p3d_mesh, faces_to_keep_mask=(~non_valid_faces) | necessary_faces)
                except:
                    print(f"Error removing faces for mesh {i_mesh}. Empty mesh?")
                    empty_mesh = True
                    
            if not empty_mesh:
                p3d_meshes.append(p3d_mesh)

        from pytorch3d.structures import join_meshes_as_scene
        full_mesh = join_meshes_as_scene(p3d_meshes)
        verts = full_mesh.verts_packed()
        faces = full_mesh.faces_packed()
        vert_colors = full_mesh.textures.verts_features_packed()
        # Creates an open3d mesh from the pytorch3d mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts.cpu().numpy())
        mesh.triangles = o3d.utility.Vector3iVector(faces.cpu().numpy())
        mesh.vertex_colors = o3d.utility.Vector3dVector(vert_colors.cpu().numpy())
        
        name = 'multires_tsdf.ply'
        o3d.io.write_triangle_mesh(os.path.join(train_dir, name), mesh)
        print("mesh saved at {}".format(os.path.join(train_dir, name)))
        # post-process the mesh and save, saving the largest N clusters
        mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
        o3d.io.write_triangle_mesh(os.path.join(train_dir, name.replace('.ply', '_post.ply')), mesh_post)
        print("mesh post processed saved at {}".format(os.path.join(train_dir, name.replace('.ply', '_post.ply'))))