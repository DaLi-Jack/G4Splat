from typing import List
import torch
from matcha.dm_scene.cameras import GSCamera


def transform_points_world_to_view(
    points:torch.Tensor,
    gs_cameras:List[GSCamera],
    use_p3d_convention:bool=False,
):
    """_summary_

    Args:
        points (torch.Tensor): Should have shape (n_cameras, N, 3).
        gs_cameras (List[GSCamera]): List of GSCameras. Should contain n_cameras elements.
        use_p3d_convention (bool, optional): Defaults to False.
        
    Returns:
        torch.Tensor: Has shape (n_cameras, N, 3).
    """
    world_view_transforms = torch.stack([gs_camera.world_view_transform for gs_camera in gs_cameras], dim=0)  # (n_cameras, 4, 4)
    
    points_h = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)  # (n_cameras, N, 4)
    view_points = (points_h @ world_view_transforms)[..., :3]  # (n_cameras, N, 3)
    if use_p3d_convention:
        factors = torch.tensor([[[-1, -1, 1]]], device=points.device)  # (1, 1, 3)
        view_points = factors * view_points  # (n_cameras, N, 3)
    return view_points


def transform_points_to_pixel_space(
        points:torch.Tensor,
        gs_cameras:List[GSCamera],
        points_are_already_in_view_space:bool=False,
        use_p3d_convention:bool=False,
        znear:float=1e-6,
        keep_float:bool=False,
    ):
        """_summary_

        Args:
            points (torch.Tensor): Should have shape (n_cameras, N, 3).
            gs_cameras (List[GSCamera]): List of GSCameras. Should contain n_cameras elements.
            points_are_already_in_view_space (bool, optional): Defaults to False.
            use_p3d_convention (bool, optional): Defaults to False.
            znear (float, optional): Defaults to 1e-6.

        Returns:
            torch.Tensor: Has shape (n_cameras, N, 2).
        """
        if points_are_already_in_view_space:
            full_proj_transforms = torch.stack([gs_camera.projection_matrix for gs_camera in gs_cameras])  # (n_depth, 4, 4)
            if use_p3d_convention:
                points = torch.tensor([[[-1, -1, 1]]], device=points.device) * points
        else:
            full_proj_transforms = torch.stack([gs_camera.full_proj_transform for gs_camera in gs_cameras])  # (n_cameras, 4, 4)
        
        points_h = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)  # (n_cameras, N, 4)
        proj_points = points_h @ full_proj_transforms  # (n_cameras, N, 4)
        proj_points = proj_points[..., :2] / proj_points[..., 3:4].clamp_min(znear)  # (n_cameras, N, 2)

        height, width = gs_cameras[0].image_height, gs_cameras[0].image_width
        # Should I do [width-1, height-1] instead?
        image_size = torch.tensor([[width, height]], device=points.device)
        proj_points = (1. + proj_points) * image_size / 2

        if keep_float:
            return proj_points        
        else:
            return torch.round(proj_points).long()
        
        
def get_interpolated_value_from_pixel_coordinates(
    value_img:torch.Tensor,
    pix_coords:torch.Tensor,
    interpolation_mode:str='bilinear',
    padding_mode:str='border',
    align_corners:bool=True,
):
    """
    Get a value from a pixel coordinate, by interpolating the value_img.

    Args:
        value_img (torch.Tensor): Has shape (H, W, C).
        pix_coords (torch.Tensor): Has shape (N, 2).
        interpolation_mode (str, optional): Defaults to 'bilinear'.
        padding_mode (str, optional): Defaults to 'border'.
        align_corners (bool, optional): Defaults to True.
        
    Returns:
        torch.Tensor: Has shape (N, C).
    """
    height, width = value_img.shape[:2]
    n_points = pix_coords.shape[0]
    
    # Scale and shift pixel coordinates to the range [-1, 1]
    factors = 0.5 * torch.tensor([[width-1, height-1]], dtype=torch.float32).to(pix_coords.device)  # (1, 2)
    scaled_pix_coords = pix_coords / factors - 1.  # (N, 2)
    scaled_pix_coords = scaled_pix_coords.view(1, -1, 1, 2)  # (1, N, 1, 2)

    # Interpolate the value
    interpolated_value = torch.nn.functional.grid_sample(
        input=value_img.permute(2, 0, 1)[None],  # (1, C, H, W)
        grid=scaled_pix_coords,  # (1, N, 1, 2)
        mode=interpolation_mode,
        padding_mode=padding_mode,  # 'reflection', 'zeros'
        align_corners=align_corners,
    )  # (1, C, N, 1)
    
    # Reshape to (N, C)
    interpolated_value = interpolated_value.reshape(-1, n_points).permute(1, 0)
    return interpolated_value
    

class AdaptiveTSDF:
    def __init__(
        self,
        points:torch.Tensor,
        trunc_margin:float,
        znear:float=1e-6,
        zfar:float=1e6,
        use_binary_opacity:bool=False,
    ):
        """
        A class for computing a TSDF field from a set of points and a collection of posed depth maps.
        
        Args:
            points (torch.Tensor): Points at which to compute the TSDF. Has shape (N, 3).
            trunc_margin (float): Truncation margin for the TSDF.
            znear (float): Near clipping plane.
            zfar (float): Far clipping plane.
            use_binary_opacity (bool): Whether to use a binary opacity field or a TSDF field.
                Please note that the TSDF field can be approximated into a binary opacity field 
                by using a TSDF with softmax weighting and high temperature.
        """
        
        assert trunc_margin >= 0, "Truncation margin must be positive"
        assert points.shape[1] == 3, "Points must have shape (N, 3)"
        assert znear > 0, "znear must be positive"
        assert zfar > znear, "zfar must be greater than znear"
        
        self._n_points = points.shape[0]
        self._points = points
        self._trunc_margin = trunc_margin
        self._znear = znear
        self._zfar = zfar

        # Initialize the field values
        if use_binary_opacity:
            self._tsdf = torch.ones(self._n_points, 1, device=points.device)
        else:
            self._tsdf = - torch.ones(self._n_points, 1, device=points.device)
        self._weights = torch.zeros(self._n_points, 1, device=points.device)
        self._colors = torch.zeros(self._n_points, 3, device=points.device)
        
        self._use_binary_opacity = use_binary_opacity
        
    @property
    def device(self):
        return self._points.device
    
    def integrate(
        self, 
        img:torch.Tensor, 
        depth:torch.Tensor,
        camera:GSCamera, 
        obs_weight=1.0,
        interpolate_depth:bool=True,
        interpolation_mode:str='bilinear',
        weight_interpolation_by_depth_gradient:bool=False,
        depth_gradient_threshold:float=1.0,
        normals:torch.Tensor=None,
        filter_with_depth_gradient:bool=False,
        depth_gradient_threshold_for_filtering:float=1.0,
        filter_with_normal_consistency:bool=False,
        reference_normals:torch.Tensor=None,
        normal_consistency_threshold:float=1.0,
        unbias_depth_using_normals:bool=False,
        weight_by_softmax:bool=False,
        softmax_temperature:float=1.0,
        weight_by_normal_consistency:bool=False,
    ):
        """
        Integrate a new observation into the TSDF.
        
        Args:
            img (torch.Tensor): Image. Has shape (H, W, 3) or (3, H, W).
            depth (torch.Tensor): Depth. Has shape (H, W), (H, W, 1) or (1, H, W).
            camera (GSCamera): Camera.
            obs_weight (float): Weight for the observation.
            interpolate_depth (bool): Whether to interpolate the depth.
            interpolation_mode (str): Interpolation mode.
            weight_interpolation_by_depth_gradient (bool): Whether to weight the interpolation by the depth gradient.
            depth_gradient_threshold (float): Threshold for the depth gradient.
            normals (torch.Tensor): Normal maps. Has shape (H, W, 3) or (3, H, W). If provided, will be used for unbiasing the TSDF values.
            filter_with_depth_gradient (bool): Whether to filter out points with high depth gradient.
            depth_gradient_threshold_for_filtering (float): Threshold for the depth gradient for filtering.
            filter_with_normal_consistency (bool): Whether to filter out points with low normal consistency.
            normal_consistency_threshold (float): Threshold for the normal consistency.
        """
        
        # Reshape image and depth to (H, W, 3) and (H, W) respectively
        if img.shape[0] == 3:
            img = img.permute(1, 2, 0)
        depth = depth.squeeze()
        H, W = img.shape[:2]
        if (normals is not None) and (normals.shape[0] == 3):
            normals = normals.permute(1, 2, 0)
        if (reference_normals is not None) and (reference_normals.shape[0] == 3):
            reference_normals = reference_normals.permute(1, 2, 0)
            
        # Compute depth gradient if needed
        if (interpolate_depth and weight_interpolation_by_depth_gradient) or filter_with_depth_gradient:
            padded_depth = torch.nn.functional.pad(
                input=depth.squeeze()[None], pad=(1, 1, 1, 1), mode='replicate',
            ).squeeze()  # (H+2, W+2)
            depth_x_gradient = padded_depth[2:, 1:-1] - padded_depth[:-2, 1:-1]
            depth_y_gradient = padded_depth[1:-1, 2:] - padded_depth[1:-1, :-2]
            depth_gradient = torch.sqrt(depth_x_gradient**2 + depth_y_gradient**2)  # (H, W)
        
        # Transform points to view space
        view_points = transform_points_world_to_view(
            self._points.view(1, self._n_points, 3),
            gs_cameras=[camera],
        )[0]  # (N, 3)
        
        # Project points to pixel space
        pix_pts = transform_points_to_pixel_space(
            view_points.view(1, self._n_points, 3),
            gs_cameras=[camera],
            points_are_already_in_view_space=True,
            keep_float=interpolate_depth,
        )[0]  # (N, 2)
        pix_x, pix_y, pix_z = pix_pts[..., 0], pix_pts[..., 1], view_points[..., 2]
        if interpolate_depth:
            int_pix_pts = pix_pts.round().long()  # (N, 2)
            int_pix_x, int_pix_y = int_pix_pts[:, 0], int_pix_pts[:, 1]
        else:
            int_pix_x, int_pix_y = pix_x, pix_y
        
        # Remove points outside view frustum and outside depth range
        valid_mask = (
            (pix_x >= 0) & (pix_x <= W-1) 
            & (pix_y >= 0) & (pix_y <= H-1) 
            & (pix_z > self._znear) & (pix_z < self._zfar)
        )  # (N,)
        
        # Filter points with depth gradient
        if filter_with_depth_gradient:
            _int_pix_pts = int_pix_pts[valid_mask]
            gradient_depth_at_pix = depth_gradient[_int_pix_pts[:, 1], _int_pix_pts[:, 0]]  # (N_valid,)
            valid_mask[valid_mask.clone()] = gradient_depth_at_pix < depth_gradient_threshold_for_filtering
            
        # Filter points with normal consistency
        if filter_with_normal_consistency and (reference_normals is not None):
            normal_consistency = (reference_normals * normals).sum(dim=-1)  # (H, W)
            normal_consistency_at_valid_pix = normal_consistency[int_pix_y[valid_mask], int_pix_x[valid_mask]]  # (N_valid,)
            valid_mask[valid_mask.clone()] = normal_consistency_at_valid_pix > normal_consistency_threshold
            
        # Get depth values at pixel locations
        depth_at_pix = torch.zeros(len(valid_mask), device=self.device)
        if interpolate_depth:
            depth_at_pix[valid_mask] = get_interpolated_value_from_pixel_coordinates(
                value_img=depth[..., None],
                pix_coords=pix_pts[valid_mask],
                interpolation_mode=interpolation_mode,
                padding_mode='border',
                align_corners=True,
            )[..., 0]
            if weight_interpolation_by_depth_gradient:
                # Get depth and depth gradients at valid pixel locations
                _int_pix_pts = int_pix_pts[valid_mask]  # (N_valid, 2)
                gradient_depth_at_pix = depth_gradient[_int_pix_pts[:, 1], _int_pix_pts[:, 0]]  # (N_valid,)
                nearest_depth_at_pix = depth[_int_pix_pts[:, 1], _int_pix_pts[:, 0]]  # (N_valid,)
                
                # Fix depth values at pixel locations with high depth gradients
                fixed_depth_at_pix = depth_at_pix[valid_mask]  # (N_valid,)
                fixed_depth_at_pix[gradient_depth_at_pix > depth_gradient_threshold] = nearest_depth_at_pix[gradient_depth_at_pix > depth_gradient_threshold]
                depth_at_pix[valid_mask] = fixed_depth_at_pix
        else:
            depth_at_pix[valid_mask] = depth[pix_y[valid_mask], pix_x[valid_mask]]
        
        # Compute distance
        depth_diff = depth_at_pix - pix_z
        valid_mask = valid_mask & (depth_at_pix > 0.) & (depth_diff >= -self._trunc_margin)

        if (normals is not None) and unbias_depth_using_normals:
            # Unbias the TSDF values using the normals (experimental, tends to inflate the surface)
            # TODO: Try unbiasing after updating the valid mask
            camera_center = camera.world_view_transform.inverse()[3:4, :3]
            pt_rays_at_valid_pix = torch.nn.functional.normalize(view_points[valid_mask] - camera_center, dim=-1)  # (N_valid, 3)
            normals_at_valid_pix = normals[int_pix_y[valid_mask], int_pix_x[valid_mask]]  # (N_valid, 3)
            normals_proj = torch.sum(pt_rays_at_valid_pix * normals_at_valid_pix, dim=-1).abs()  # (N_valid,)
            depth_diff[valid_mask] = depth_diff[valid_mask] * normals_proj  # (N_valid,)
                    
        dist = (depth_diff[valid_mask] / self._trunc_margin).clamp_max(1.).unsqueeze(-1)
        # dist = (depth_diff[valid_mask] / self._trunc_margin).clamp(min=-1., max=1.).unsqueeze(-1)
        
        # Compute observation weight
        _obs_weight = obs_weight
        if weight_by_softmax:
            _obs_weight = _obs_weight * torch.exp(softmax_temperature * dist)  # (N_valid, 1)
        if weight_by_normal_consistency:
            normal_consistency = (reference_normals * normals).sum(dim=-1)  # (H, W)
            normal_consistency_at_valid_pix = normal_consistency[int_pix_y[valid_mask], int_pix_x[valid_mask]].unsqueeze(-1)  # (N_valid, 1)
            _obs_weight = _obs_weight * normal_consistency_at_valid_pix.abs()  # (N_valid, 1)
            
        # Update TSDF
        if self._use_binary_opacity:
            opacity = (dist < 0.).float()
            self._tsdf[valid_mask] = torch.min(self._tsdf[valid_mask], opacity)
            
            old_weights = self._weights[valid_mask]
            new_weights = old_weights + _obs_weight
            self._weights[valid_mask] = new_weights
        else:
            old_weights = self._weights[valid_mask]
            old_tsdf = self._tsdf[valid_mask]
            
            new_weights = old_weights + _obs_weight
            new_tsdf = (old_tsdf * old_weights + dist * _obs_weight) / new_weights
            
            self._weights[valid_mask] = new_weights
            self._tsdf[valid_mask] = new_tsdf
        
        # Integrate color
        if interpolate_depth:
            img_at_valid_pix = get_interpolated_value_from_pixel_coordinates(
                value_img=img,
                pix_coords=pix_pts[valid_mask],
                interpolation_mode=interpolation_mode,
                padding_mode='border',
                align_corners=True,
            )
        else:
            img_at_valid_pix = img[pix_y[valid_mask], pix_x[valid_mask]]
        old_colors = self._colors[valid_mask]
        new_colors = (old_colors * old_weights + img_at_valid_pix * _obs_weight) / new_weights
        self._colors[valid_mask] = new_colors.clamp(min=0., max=1.)

    def return_field_values(self):
        output_pkg = {
            "tsdf": 0.5 - self._tsdf if self._use_binary_opacity else self._tsdf,
            "colors": self._colors,
        }
        return output_pkg
