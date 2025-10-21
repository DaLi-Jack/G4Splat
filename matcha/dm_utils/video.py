import json
import torch
import numpy as np
from matcha.dm_scene.cameras import CamerasWrapper, focal2fov, fov2focal, GSCamera
import open3d as o3d
from pytorch3d.transforms import (
    quaternion_to_matrix, 
    quaternion_invert,
    quaternion_multiply,
    Transform3d
)

from matcha.dm_utils.rendering import focal2fov, fov2focal, getWorld2View2, getProjectionMatrix


def load_blender_package(package_path, device):
    # Load package
    package = json.load(open(package_path))
    # Convert lists into tensors
    for key, object in package.items():
        if type(object) is dict:
            for sub_key, sub_object in object.items():
                if type(sub_object) is list:
                    object[sub_key] = torch.tensor(sub_object)
        elif type(object) is list:
            for element in object:
                if element:
                    for sub_key, sub_object in element.items():
                        if type(sub_object) is list:
                            element[sub_key] = torch.tensor(sub_object)
                            
    # Process bones
    bone_to_vertices = []
    bone_to_vertex_weights = []
    for i_mesh, mesh_dict in enumerate(package['bones']):
        if mesh_dict:
            vertex_dict = mesh_dict['vertex']
            armature_dict = mesh_dict['armature']
            
            # Per vertex info
            vertex_dict['matrix_world'] = torch.Tensor(vertex_dict['matrix_world']).to(device)
            vertex_dict['tpose_points'] = torch.Tensor(vertex_dict['tpose_points']).to(device)
            # vertex_dict['groups'] = np.array(vertex_dict['groups'])
            # vertex_dict['weights'] = torch.tensor(vertex_dict['weights']).to(device)
            
            # Per bone info
            armature_dict['matrix_world'] = torch.Tensor(armature_dict['matrix_world']).to(device)
            for key, val in armature_dict['rest_bones'].items():
                armature_dict['rest_bones'][key] = torch.Tensor(val).to(device)
            for key, val in armature_dict['pose_bones'].items():
                armature_dict['pose_bones'][key] = torch.Tensor(val).to(device)
                
            # Build mapping from bone name to corresponding vertices
            vertex_groups_idx = {}
            vertex_groups_weights = {}
            
            # > For each bone of the current armature, we initialize an empty list
            for bone_name in armature_dict['rest_bones']:
                vertex_groups_idx[bone_name] = []
                vertex_groups_weights[bone_name] = []
                
            # > For each vertex, we add the vertex index to the corresponding bone lists
            for i in range(len(vertex_dict['groups'])):
                # groups_in_which_vertex_appears = vertex_dict['groups'][i]
                # weights_of_the_vertex_in_those_groups = vertex_dict['weights'][i]
                groups_in_which_vertex_appears = []
                weights_of_the_vertex_in_those_groups = []

                # We start by filtering out the groups that are not part of the current armature.
                # This is necessary for accurately normalizing the weights.
                for j_group, group in enumerate(vertex_dict['groups'][i]):
                    if group in vertex_groups_idx:
                        groups_in_which_vertex_appears.append(group)
                        weights_of_the_vertex_in_those_groups.append(vertex_dict['weights'][i][j_group])
                
                # We normalize the weights
                normalize_weights = True
                if normalize_weights:
                    sum_of_weights = np.sum(weights_of_the_vertex_in_those_groups)
                    weights_of_the_vertex_in_those_groups = [w / sum_of_weights for w in weights_of_the_vertex_in_those_groups]
                
                # We add the vertex index and the associated weight to the corresponding bone lists
                for j_group, group in enumerate(groups_in_which_vertex_appears):
                    # For safety, we check that the group belongs to the current armature, used for rendering.
                    # Indeed, for editing purposes, one might want to use multiple armatures in the Blender scene, 
                    # but only one (as expected) for the final rendering.
                    if group in vertex_groups_idx:
                        vertex_groups_idx[group].append(i)
                        vertex_groups_weights[group].append(weights_of_the_vertex_in_those_groups[j_group])

            # > Convert the lists to tensors
            for bone_name in vertex_groups_idx:
                if len(vertex_groups_idx[bone_name]) > 0:
                    vertex_groups_idx[bone_name] = torch.tensor(vertex_groups_idx[bone_name], dtype=torch.long, device=device)
                    vertex_groups_weights[bone_name] = torch.tensor(vertex_groups_weights[bone_name], device=device)

            bone_to_vertices.append(vertex_groups_idx)
            bone_to_vertex_weights.append(vertex_groups_weights)
        else:
            bone_to_vertices.append(None)
            bone_to_vertex_weights.append(None)
    package['bone_to_vertices'] = bone_to_vertices
    package['bone_to_vertex_weights'] = bone_to_vertex_weights
    
    return package


def load_cameras_from_blender_package(package, device):
    matrix_world = package['camera']['matrix_world'].to(device)
    angle = package['camera']['angle']
    znear = package['camera']['clip_start']
    zfar = package['camera']['clip_end']
    
    if not 'image_height' in package['camera']:
        print('[WARNING] Image size not found in the package. Using default value 1920 x 1080.')
        height, width = 1080, 1920
    else:
        height, width = package['camera']['image_height'], package['camera']['image_width']

    gs_cameras = []
    for i_cam in range(len(angle)):
        c2w = matrix_world[i_cam]
        c2w[:3, 1:3] *= -1  # Blender to COLMAP convention
        w2c = c2w.inverse()
        R, T = w2c[:3, :3].transpose(-1, -2), w2c[:3, 3]  # R is stored transposed due to 'glm' in CUDA code
        
        fov = angle[i_cam].item()
        
        if width > height:
            fov_x = fov
            fov_y = focal2fov(fov2focal(fov_x, width), height)
        else:
            fov_y = fov
            fov_x = focal2fov(fov2focal(fov_y, height), width)
        
        gs_camera = GSCamera(
            colmap_id=str(i_cam), 
            R=R.cpu().numpy(), 
            T=T.cpu().numpy(), 
            FoVx=fov_x, 
            FoVy=fov_y,
            image=None, 
            gt_alpha_mask=None,
            image_name=f"frame_{i_cam}", 
            uid=i_cam,
            data_device=device,
            image_height=height,
            image_width=width,
        )
        gs_cameras.append(gs_camera)
    
    return CamerasWrapper(gs_cameras)
