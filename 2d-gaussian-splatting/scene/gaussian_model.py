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

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.use_mip_filter = False
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    def freeze_params(self):
        """
        Freeze all parameters of the GaussianModel to prevent gradient updates
        
        This is useful for hierarchical model training or transfer learning scenarios,
        such as when you want to add new points based on an existing model without 
        changing the original points.
        
        The freezing operation affects the following parameters:
        - Positions (_xyz)
        - Features (_features_dc, _features_rest)
        - Scaling (_scaling)
        - Rotation (_rotation)
        - Opacity (_opacity)
        
        Note: This operation modifies the original parameters to no longer require gradients
        """
        # Record the original parameter count
        original_point_count = self._xyz.shape[0]
        print(f"Freezing {original_point_count} points in GaussianModel")
        
        # Freeze each parameter
        self._xyz.requires_grad_(False)
        self._features_dc.requires_grad_(False)
        self._features_rest.requires_grad_(False)
        self._scaling.requires_grad_(False)
        self._rotation.requires_grad_(False)
        self._opacity.requires_grad_(False)
        
        # If optimizer is already set, update the parameters in the optimizer
        if self.optimizer is not None:
            # Replace parameters with frozen versions
            updated_param_groups = []
            for group in self.optimizer.param_groups:
                if group["name"] == "xyz":
                    group["params"] = [self._xyz]
                elif group["name"] == "f_dc":
                    group["params"] = [self._features_dc]
                elif group["name"] == "f_rest":
                    group["params"] = [self._features_rest]
                elif group["name"] == "opacity":
                    group["params"] = [self._opacity]
                elif group["name"] == "scaling":
                    group["params"] = [self._scaling]
                elif group["name"] == "rotation":
                    group["params"] = [self._rotation]
                
                # Set learning rate to 0 to ensure no updates
                group["lr"] = 0.0
                updated_param_groups.append(group)
            
            # Recreate the optimizer with updated parameter groups while retaining state
            self.optimizer = torch.optim.Adam(
                updated_param_groups,
                lr=0.0,
                eps=1e-15
            )

    @property
    def get_scaling(self):
        scales = self.scaling_activation(self._scaling)
        if self.use_mip_filter:
            scales = torch.square(scales) + torch.square(self.mip_filter)
            scales = torch.sqrt(scales)
        return scales #.clamp(max=1)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        opacity = self.opacity_activation(self._opacity) 
        if self.use_mip_filter:
            scales = self.scaling_activation(self._scaling)
            
            scales_square = torch.square(scales)
            det1 = scales_square.prod(dim=1)
            
            scales_after_square = scales_square + torch.square(self.mip_filter) 
            det2 = scales_after_square.prod(dim=1) 
            coef = torch.sqrt(det1 / det2)
            opacity = opacity * coef[..., None]
        return opacity
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
    def create_from_parameters(self, _means, _scales, _quaternions, _colors, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = _means
        fused_color = RGB2SH(_colors)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        scales = torch.log(_scales)
        rots = _quaternions

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        if self.use_mip_filter:
            l.append('mip_filter')
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        
        if self.use_mip_filter:
            mip_filter = self.mip_filter.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        if self.use_mip_filter:
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, mip_filter), axis=1)
        else:
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        
    @torch.no_grad()
    def get_tetra_points(
        self, 
        downsample_ratio : float = None, 
        gaussian_flatness : float = 1e-3, 
        return_idx : bool = False,
        points_idx : torch.Tensor = None,
    ):
        import trimesh
        M = trimesh.creation.box()
        M.vertices *= 2
        
        rots = build_rotation(self._rotation)
        scales_3d = torch.nn.functional.pad(
            self.get_scaling, 
            (0, 1), 
            mode="constant", 
            value=gaussian_flatness,
        )
        print(f"[INFO] Padding 2D scaling with {gaussian_flatness} for tetra points: {scales_3d[0]}")
        
        if (downsample_ratio is None) and (points_idx is None):
            xyz = self.get_xyz
            scale = scales_3d * 3. # TODO test
            # filter points with small opacity for bicycle scene
            # opacity = self.get_opacity_with_3D_filter
            # mask = (opacity > 0.1).squeeze(-1)
            # xyz = xyz[mask]
            # scale = scale[mask]
            # rots = rots[mask]
        else:
            if points_idx is None:
                print(f"[INFO] Downsampling tetra points by {downsample_ratio}.")
                xyz_idx = torch.randperm(self.get_xyz.shape[0])[:int(self.get_xyz.shape[0] * downsample_ratio)]
                xyz = self.get_xyz[xyz_idx]
                scale = scales_3d[xyz_idx] * 3. / (downsample_ratio ** (1/3))
                rots = rots[xyz_idx]
                print(f"[INFO] Number of tetra points after downsampling: {xyz.shape[0]}.")
            else:
                downsample_ratio = len(points_idx) / len(self.get_xyz)
                xyz_idx = points_idx
                xyz = self.get_xyz[xyz_idx]
                scale = scales_3d[xyz_idx] * 3. / (downsample_ratio ** (1/3))
                rots = rots[xyz_idx]
                print(f"[INFO] Number of tetra points after downsampling: {xyz.shape[0]}.")
                
        vertices = M.vertices.T    
        vertices = torch.from_numpy(vertices).float().cuda().unsqueeze(0).repeat(xyz.shape[0], 1, 1)
        # scale vertices first
        vertices = vertices * scale.unsqueeze(-1)
        vertices = torch.bmm(rots, vertices).squeeze(-1) + xyz.unsqueeze(-1)
        vertices = vertices.permute(0, 2, 1).reshape(-1, 3).contiguous()
        # concat center points
        vertices = torch.cat([vertices, xyz], dim=0)
        
        # scale is not a good solution but use it for now
        scale = scale.max(dim=-1, keepdim=True)[0]
        scale_corner = scale.repeat(1, 8).reshape(-1, 1)
        vertices_scale = torch.cat([scale_corner, scale], dim=0)
        if return_idx:
            if downsample_ratio is None:
                print("[WARNING] return_idx might not be needed when downsample_ratio is None")
                xyz_idx = torch.arange(self.get_xyz.shape[0])
            return vertices, vertices_scale, xyz_idx
        else:
            return vertices, vertices_scale
    
    def set_mip_filter(self, use_mip_filter: bool):
        self.use_mip_filter = use_mip_filter
    
    @torch.no_grad()
    def compute_mip_filter(self, cameras, znear=0.2, filter_variance=0.2):
        # Set the flag to use the mip filter
        if not self.use_mip_filter:
            print("[WARNING] Computing mip filter but mip filter is currently disabled.")
        
        #TODO consider focal length and image width
        xyz = self.get_xyz
        distance = torch.ones((xyz.shape[0]), device=xyz.device) * 100000.0
        valid_points = torch.zeros((xyz.shape[0]), device=xyz.device, dtype=torch.bool)
        
        # We should use the focal length of the highest resolution camera
        focal_length = 0.
        for camera in cameras:

            # transform points to camera space
            R = torch.tensor(camera.R, device=xyz.device, dtype=torch.float32)
            T = torch.tensor(camera.T, device=xyz.device, dtype=torch.float32)
            # R is stored transposed due to 'glm' in CUDA code so we don't neet transopse here
            xyz_cam = xyz @ R + T[None, :]
            xyz_to_cam = torch.norm(xyz_cam, dim=1)
            
            # project to screen space
            valid_depth = xyz_cam[:, 2] > znear
            
            
            x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
            z = torch.clamp(z, min=0.001)
            
            x = x / z * camera.focal_x + camera.image_width / 2.0
            y = y / z * camera.focal_y + camera.image_height / 2.0
            
            # use similar tangent space filtering as in the paper
            in_screen = torch.logical_and(torch.logical_and(x >= -0.15 * camera.image_width, x <= camera.image_width * 1.15), torch.logical_and(y >= -0.15 * camera.image_height, y <= 1.15 * camera.image_height))
            
        
            valid = torch.logical_and(valid_depth, in_screen)
            
            # distance[valid] = torch.min(distance[valid], xyz_to_cam[valid])
            distance[valid] = torch.min(distance[valid], z[valid])
            valid_points = torch.logical_or(valid_points, valid)
            if focal_length < camera.focal_x:
                focal_length = camera.focal_x
        
        distance[~valid_points] = distance[valid_points].max()
        
        mip_filter = distance / focal_length * (filter_variance ** 0.5)
        self.mip_filter = mip_filter[..., None]

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        
        if "mip_filter" in [p.name for p in plydata.elements[0].properties]:
            mip_filter = np.asarray(plydata.elements[0]["mip_filter"])[..., np.newaxis]
            use_mip_filter = True
            self.set_mip_filter(use_mip_filter)
            print("[INFO] Loading mip filter from ply file.")
        else:
            print("[INFO] No mip filter found in ply file.")
            use_mip_filter = False

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        if use_mip_filter:
            self.mip_filter = torch.tensor(mip_filter, dtype=torch.float, device="cuda")

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        use_mip_filter = self.use_mip_filter
        if use_mip_filter:
            self.set_mip_filter(False)
            
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()
        if use_mip_filter:
            self.set_mip_filter(True)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def gs_scale_loss(self, max_scale_thresh=0.05):
        max_scale = self.get_scaling.max(dim=1).values
        excess = torch.clamp(max_scale - max_scale_thresh, min=0.0)
        loss = torch.sum(excess ** 2)
        return loss

def combine_gslist(gslist):
    """
    Combine a list of GaussianModel objects into a single GaussianModel object.
    
    Args:
        gslist: List of GaussianModel objects to combine
        
    Returns:
        A new GaussianModel instance containing all parameters from the input models
    """
    # Initialize a new GaussianModel object with the same SH degree as the first model in the list
    combined_model = GaussianModel(gslist[0].max_sh_degree)
    
    # Prepare lists to hold parameters from all models
    xyz_list = []
    features_dc_list = []
    features_rest_list = []
    opacity_list = []
    scaling_list = []
    rotation_list = []
    mip_filter_list = []
    # Collect parameters from each model
    for model in gslist:
        xyz_list.append(model.get_xyz.detach())
        features_dc_list.append(model._features_dc.detach())
        features_rest_list.append(model._features_rest.detach())
        opacity_list.append(model._opacity.detach())
        scaling_list.append(model._scaling.detach())
        rotation_list.append(model._rotation.detach())

        if hasattr(model, "mip_filter"):
            mip_filter_list.append(model.mip_filter.detach())
    
    # Concatenate all parameters
    combined_model._xyz = nn.Parameter(torch.cat(xyz_list, dim=0))
    combined_model._features_dc = nn.Parameter(torch.cat(features_dc_list, dim=0))
    combined_model._features_rest = nn.Parameter(torch.cat(features_rest_list, dim=0))
    combined_model._opacity = nn.Parameter(torch.cat(opacity_list, dim=0))
    combined_model._scaling = nn.Parameter(torch.cat(scaling_list, dim=0))
    combined_model._rotation = nn.Parameter(torch.cat(rotation_list, dim=0))

    if len(mip_filter_list) > 0:
        combined_model.set_mip_filter(True)
        combined_model.mip_filter = nn.Parameter(torch.cat(mip_filter_list, dim=0))
    assert combined_model.mip_filter.shape[0] == combined_model._xyz.shape[0]
    
    # Also initialize/combine other necessary properties
    combined_model.active_sh_degree = gslist[0].active_sh_degree
    
    # Initialize max_radii2D with the right size
    n_points = combined_model._xyz.shape[0]
    combined_model.max_radii2D = torch.zeros(n_points, device=combined_model._xyz.device)
    
    # Initialize other buffers with the appropriate sizes
    combined_model.xyz_gradient_accum = torch.zeros((n_points, 1), device=combined_model._xyz.device)
    combined_model.denom = torch.zeros((n_points, 1), device=combined_model._xyz.device)

    # NOTE: hard code pseudo optimizer
    l = [
        {'params': [combined_model._xyz], 'lr': 0.001, "name": "xyz"},
        {'params': [combined_model._features_dc], 'lr': 0.001, "name": "f_dc"},
        {'params': [combined_model._features_rest], 'lr': 0.001 / 20.0, "name": "f_rest"},
        {'params': [combined_model._opacity], 'lr': 0.001, "name": "opacity"},
        {'params': [combined_model._scaling], 'lr': 0.001, "name": "scaling"},
        {'params': [combined_model._rotation], 'lr': 0.001, "name": "rotation"}
    ]
    combined_model.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
    
    # Copy spatial_lr_scale from the first model
    combined_model.spatial_lr_scale = gslist[0].spatial_lr_scale
    
    # Set percent_dense to the same as the first model
    combined_model.percent_dense = gslist[0].percent_dense
    
    # Print information about the combined model
    print(f"Combined {len(gslist)} models with a total of {n_points} points")
    for i, model in enumerate(gslist):
        print(f"  Model {i}: {model.get_xyz.shape[0]} points")
    
    return combined_model

def combine_gslist_simple(gslist):
    """
    Combine a list of GaussianModel objects into a single GaussianModel object.
    
    Args:
        gslist: List of GaussianModel objects to combine
        
    Returns:
        A new GaussianModel instance containing all parameters from the input models
    """
    # Initialize a new GaussianModel object with the same SH degree as the first model in the list
    combined_model = GaussianModel(gslist[0].max_sh_degree)
    
    # Prepare lists to hold parameters from all models
    xyz_list = []
    features_dc_list = []
    features_rest_list = []
    opacity_list = []
    scaling_list = []
    rotation_list = []

    # Collect parameters from each model
    for model in gslist:
        xyz_list.append(model.get_xyz.detach())
        features_dc_list.append(model._features_dc.detach())
        features_rest_list.append(model._features_rest.detach())
        opacity_list.append(model._opacity.detach())
        scaling_list.append(model._scaling.detach())
        rotation_list.append(model._rotation.detach())
    
    # Concatenate all parameters
    combined_model._xyz = nn.Parameter(torch.cat(xyz_list, dim=0))
    combined_model._features_dc = nn.Parameter(torch.cat(features_dc_list, dim=0))
    combined_model._features_rest = nn.Parameter(torch.cat(features_rest_list, dim=0))
    combined_model._opacity = nn.Parameter(torch.cat(opacity_list, dim=0))
    combined_model._scaling = nn.Parameter(torch.cat(scaling_list, dim=0))
    combined_model._rotation = nn.Parameter(torch.cat(rotation_list, dim=0))

    combined_model.max_radii2D = torch.zeros(combined_model._xyz.shape[0], device=combined_model._xyz.device)
    
    # Count of old Gaussians (for setting learning rates)
    gaussians_num = len(gslist)
    total_count = combined_model._xyz.shape[0]
    
    print(f'{gaussians_num} Gaussians are combined into one model, with {total_count} points')

    return combined_model

def get_obj_gaussian_by_mask(gaussian, obj_gs_mask):
    """
    Extract object-specific Gaussians based on a binary mask.
    
    Args:
        gaussian: Source GaussianModel containing all Gaussians
        obj_gs_mask: Binary mask indicating which Gaussians belong to the object
        
    Returns:
        A new GaussianModel instance containing only the Gaussians of the object
    """
    # Create a new GaussianModel with the same SH degree
    obj_gaussian = GaussianModel(gaussian.max_sh_degree)
    
    # Extract parameters for the selected Gaussians
    obj_gaussian._xyz = torch.nn.Parameter(gaussian._xyz[obj_gs_mask].clone().detach())
    obj_gaussian._features_dc = torch.nn.Parameter(gaussian._features_dc[obj_gs_mask].clone().detach())
    obj_gaussian._features_rest = torch.nn.Parameter(gaussian._features_rest[obj_gs_mask].clone().detach())
    obj_gaussian._scaling = torch.nn.Parameter(gaussian._scaling[obj_gs_mask].clone().detach())
    obj_gaussian._rotation = torch.nn.Parameter(gaussian._rotation[obj_gs_mask].clone().detach())
    obj_gaussian._opacity = torch.nn.Parameter(gaussian._opacity[obj_gs_mask].clone().detach())
    
    # Copy other necessary properties
    obj_gaussian.active_sh_degree = gaussian.active_sh_degree
    obj_gaussian.max_sh_degree = gaussian.max_sh_degree
    obj_gaussian.spatial_lr_scale = gaussian.spatial_lr_scale
    
    # Handle MIP filtering if used
    if hasattr(gaussian, 'use_mip_filter') and gaussian.use_mip_filter:
        obj_gaussian.set_mip_filter(True)
        if hasattr(gaussian, 'mip_filter'):
            obj_gaussian.mip_filter = gaussian.mip_filter[obj_gs_mask].clone().detach()
    
    # Initialize max_radii2D with the right size
    obj_gaussian.max_radii2D = torch.zeros(
        obj_gaussian._xyz.shape[0], device=obj_gaussian._xyz.device
    )
    
    # Print statistics
    print(f"Extracted {obj_gaussian._xyz.shape[0]} Gaussians for the object "
          f"(out of {gaussian._xyz.shape[0]} total Gaussians)")
    
    return obj_gaussian

def get_gaussian_normal(rotation, scaling, scale_modifier=1.0):

    q = torch.nn.functional.normalize(rotation, dim=-1)
    scales_3d = torch.cat([scaling * scale_modifier, torch.ones_like(scaling[:, :1])], dim=-1)
    L = build_scaling_rotation(scales_3d, q)  # (N, 3, 3), L = R * S
    normal = L[:, :, 2]

    return normal
