import random
import math
from typing import Union
import numpy as np
import torch
from torch.nn.functional import normalize as torch_normalize
from pytorch3d.structures import Meshes
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix

try:  # Vanilla 3DGS or Gaussian Surfel rasterization (3DGS with normal and depth rendering)
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
except ImportError:
    print("[INFO] Vanilla 3D Gaussian rasterization or Gaussian Surfel rasterization not available.")

try:  # Gaustudio rasterization (3DGS with depth and median depth rendering)
    from gaustudio_diff_gaussian_rasterization import GaussianRasterizationSettings as GaustudioGaussianRasterizationSettings
    from gaustudio_diff_gaussian_rasterization import GaussianRasterizer as GaustudioGaussianRasterizer
except ImportError:
    print("[INFO] Gaustudio rasterization not available.")

try:  # 2DGS rasterization (2D gaussian rasterization with depth, normal, and distortion rendering)
    from diff_surfel_rasterization import GaussianRasterizationSettings as SurfelGaussianRasterizationSettings
    from diff_surfel_rasterization import GaussianRasterizer as SurfelGaussianRasterizer
except ImportError:
    print("[INFO] 2DGS rasterization not available.")


def get_regular_triangle_bary_coords(n:int, device='cpu'):
    """Returns regular barycentric coordinates in a triangle.
    The barycentric coordinates correspond to a regular set of N points,
    where N=n*(n+1)/2 is the n-th triangular number.

    Args:
        n (int): Level of the triangular number.
        device (str, optional): Defaults to 'cpu'.

    Returns:
        torch.Tensor: Has shape (N, 3) where N=n*(n+1)/2.
    """
    l = torch.arange(n, device=device).view(-1, 1, 1).repeat(1, n, 1) + 1
    k = torch.arange(n, device=device).view(1, -1, 1).repeat(n, 1, 1) + 1
    
    gamma = 1 - (l + 1) / (n + 2)
    alpha = k / (l + 1) * (1 - gamma)
    beta = (l - k + 1) / (l + 1) * (1 - gamma)
    mask = k[..., 0] <= l[..., 0]
    
    return torch.cat([alpha[mask], beta[mask], gamma[mask]], dim=-1)


def get_gaussian_surfel_parameters_from_mesh(
    barycentric_coords:Union[torch.Tensor, int], 
    mesh:Meshes=None, 
    verts:torch.Tensor=None, faces:torch.Tensor=None, verts_features:torch.Tensor=None,
    normalized_scales:Union[float, torch.Tensor]=None,
    use_biggest_axis_as_first_axis_for_gram_schmidt:bool=True,
    normal_scale:float=1e-10,
    get_colors_from_mesh:bool=False,
    get_opacity_from_mesh:bool=False,
    ):
    """_summary_

    Args:
        barycentric_coords (Union[torch.Tensor, int]): Has shape (n_faces, n_gaussians_per_face, 3) or (n_gaussians_per_face, 3).
            If an int is provided, the regular barycentric coordinates of a triangle with the corresponding triangular number will be used.
        mesh (Meshes, optional): Defaults to None.
        verts (torch.Tensor, optional): Has shape (n_verts, 3). Defaults to None.
        faces (torch.Tensor, optional): Has shape (n_faces, 3). Defaults to None.
        normalized_scales (Union[float, torch.Tensor], optional): Has shape (n_faces, n_gaussians_per_face, 2). Can be a float. Defaults to None.
        use_biggest_axis_as_first_axis_for_gram_schmidt (bool, optional): Defaults to True.
        normal_scale (float, optional): Defaults to 1e-10.
        get_colors_from_mesh (bool, optional): Defaults to False.
        get_opacity_from_mesh (bool, optional): Defaults to False.

    Raises:
        ValueError: Either mesh or (verts, faces) should be provided.
        
    Returns:
        tuple: (means, scales, quaternion).
            means (torch.Tensor): Has shape (n_faces * n_gaussians_per_face, 3).
            scales (torch.Tensor): Has shape (n_faces * n_gaussians_per_face, 3).
            quaternions (torch.Tensor): Has shape (n_faces * n_gaussians_per_face, 4).
    """
    
    mesh_is_provided = mesh is not None
    verts_and_faces_are_provided = (verts is not None) and (faces is not None)  # and (verts_features is not None)
    if not mesh_is_provided and not verts_and_faces_are_provided:
        raise ValueError("Either mesh or (verts, faces) should be provided.")
    if verts is None:
        verts = mesh.verts_packed()
    if faces is None:
        faces = mesh.faces_packed()
    if verts_features is None and mesh_is_provided:
        verts_features = mesh.textures.verts_features_packed()
    if type(normalized_scales) == torch.Tensor:
        norm_scales = normalized_scales[..., None]
    else:
        norm_scales = normalized_scales
    device = verts.device
        
    # If barycentric_coords is an int, then the regular triangle barycentric coordinates are used as default
    if isinstance(barycentric_coords, int):
        barycentric_coords = get_regular_triangle_bary_coords(barycentric_coords, device=device)
    
    # Get the means of the Gaussians using the barycentric coordinates and the vertices of the mesh
    n_gaussians_per_triangle = barycentric_coords.shape[-2]
    if barycentric_coords.dim() == 2:
        bary_coords = barycentric_coords[None]  # (1, n_gaussians_per_triangle, 3). Same barycentric coordinates will be used for all faces
    else:
        bary_coords = barycentric_coords  # (n_faces, n_gaussians_per_triangle, 3)
    faces_verts = verts[faces]  # (n_faces, 3, d) where d = 3
    faces_points = (bary_coords[:, :, :, None] * faces_verts[:, None]).sum(dim=-2)  # (n_faces, n_points, d)

    # To compute the rotations of the Gaussians, we first compute the barycentric coordinates of two orthonormal axes in a regular triangle.
    axis_bary_shifts = torch.tensor([[
        [-np.sqrt(2)/2, np.sqrt(2)/2, 0.],
        [-1/np.sqrt(6), -1/np.sqrt(6), 2/np.sqrt(6)],
    ]], device=device, dtype=torch.float32)  # Shape (1, 2, 3)

    # Then, we transform the axes to match the triangles of the mesh, using the mesh vertices and barycentric coordinates.
    transformed_axis = (axis_bary_shifts[..., None] * faces_verts[:, None]).sum(dim=-2)  # (n_faces, 2, d)

    # Now, we orthogonalize the two transformed axes with Gram-Schmidt to get valid axes for a 2D Gaussian.
    # ortho_axis = transformed_axis.clone()  # (n_faces, 2, d)
    # We propose to use the biggest resulting axis (corresponding to the main direction of the deformation) as the first, main axis for the Gram-Schmidt process.
    if use_biggest_axis_as_first_axis_for_gram_schmidt:
        sorted_ortho_idx = torch.argsort(transformed_axis.norm(dim=-1, keepdim=True), dim=1, descending=True).repeat(1, 1, transformed_axis.shape[-1])
        if False:
            ortho_axis = transformed_axis.clone()  # (n_faces, 2, d)
            ortho_axis = ortho_axis.gather(dim=1, index=sorted_ortho_idx)
            ortho_axis[:, 1] = ortho_axis[:, 1] - (ortho_axis[:, 1] * ortho_axis[:, 0]).sum(dim=-1, keepdim=True) * ortho_axis[:, 0] / (ortho_axis[:, 0] ** 2).sum(dim=-1, keepdim=True)
            ortho_axis = ortho_axis.gather(dim=1, index=sorted_ortho_idx)
        else:
            ortho_axis = transformed_axis.gather(dim=1, index=sorted_ortho_idx).contiguous()
            second_ortho_axis = ortho_axis[:, 1] - (ortho_axis[:, 1] * ortho_axis[:, 0]).sum(dim=-1, keepdim=True) * ortho_axis[:, 0] / (ortho_axis[:, 0] ** 2).sum(dim=-1, keepdim=True)
            ortho_axis = torch.cat(
                [ortho_axis[:, 0:1], second_ortho_axis[:, None]], 
                dim=1
            ).gather(dim=1, index=sorted_ortho_idx)
        # TODO: I heard that torch.gather is pretty slow. Should I remove it?
    else:
        ortho_axis[:, 1] = ortho_axis[:, 1] - (ortho_axis[:, 1] * ortho_axis[:, 0]).sum(dim=-1, keepdim=True) * ortho_axis[:, 0] / (ortho_axis[:, 0] ** 2).sum(dim=-1, keepdim=True)
    
    # Means of the Gaussian Surfels
    means = faces_points.reshape(-1, 3)  # (n_faces * n_points, 3)
    
    # We normalize the orthogonal axes and compute the normal to get the 3D rotations of all Gaussians
    orthonormal_axis = torch_normalize(ortho_axis, dim=-1)
    normals = torch.cross(orthonormal_axis[:, 0], orthonormal_axis[:, 1], dim=-1)  # (n_faces, d)
    orthonormal_axis = orthonormal_axis[:, None, :, :].repeat(1, n_gaussians_per_triangle, 1, 1)  # (n_faces, n_points, 2, d)
    normals = normals[:, None, :].repeat(1, n_gaussians_per_triangle, 1)  # (n_faces, n_points, d)
    rotations = torch.cat([orthonormal_axis, normals[:, :, None]], dim=-2).transpose(-1, -2)  # (n_faces, n_points, 3, d)
    quaternions = matrix_to_quaternion(rotations.reshape(-1, 3, 3))  # (n_faces * n_points, 4)
    
    # We multiply the norm of the transformed orthogonal axes by the provided canonical scales to get the final scales of the Gaussian Surfels.
    # A very small value is used for the scale of the normal axis.
    scales = (
        ortho_axis[:, None, :, :].repeat(1, n_gaussians_per_triangle, 1, 1) # (n_faces, n_points, 2, d)
        * norm_scales  # float or (n_faces, n_gaussians_per_triangle, 2, 1)
    ).norm(dim=-1).reshape(-1, 2)  # (n_faces * n_points, 2)
    scales = torch.nn.functional.pad(scales, pad=(0, 1), value=normal_scale)  # (n_faces * n_points, 3)
    
    package = {
        'means': means,  # (n_faces * n_points, 3)
        'scales': scales,  # (n_faces * n_points, 3)
        'quaternions': quaternions,  # (n_faces * n_points, 4)
    }
    
    if get_colors_from_mesh or get_opacity_from_mesh:
        feature_size = min(verts_features.shape[-1], 4)
        features = (
            verts_features[:, :4][faces][:, None]  # (n_faces, 1, 3, 4)
            * bary_coords[:, :, :, None]  # (n_faces, n_points, 3, 1)
        ).sum(dim=-2).reshape(-1, feature_size)  # (n_faces * n_points, 3)

        if get_colors_from_mesh:
            package['colors'] = features[..., :3]

        if get_opacity_from_mesh:
            package['opacities'] = features[..., 3:]
    
    return package


# TODO: Normals should be flipped to face the camera
def render_gaussian_surfels(
    means, scales, quaternions, opacities, sh_coordinates_dc, sh_coordinates_rest,
    nerf_cameras, camera_idx,
    bg_color=[0., 0., 0.],
    sh_degree=None,
    point_colors=None,
    return_whole_package=True,
    rasterizer_type='2dgs',
    use_integration=False,  # New parameter to choose between render/integrate for GOF
    integration_points=None,
    ):
    
    # Settings
    n_gaussians = len(means)
    gs_camera = nerf_cameras.gs_cameras[camera_idx]
    tanfovx = math.tan(gs_camera.FoVx * 0.5)
    tanfovy = math.tan(gs_camera.FoVy * 0.5)
    image_height, image_width = nerf_cameras.height[camera_idx].item(), nerf_cameras.width[camera_idx].item()
    device = means.device
    if sh_coordinates_rest is None:
        sh_degree = 0
    else:
        if sh_degree is None:
            sh_degree = int(np.sqrt(sh_coordinates_rest.shape[-2] + 1) - 1)

    # Build rasterizer
    if (rasterizer_type == 'gaustudio'):
        raster_settings = GaustudioGaussianRasterizationSettings(
            image_height=int(image_height),
            image_width=int(image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=torch.tensor(bg_color).to(device),
            scale_modifier=1.,
            viewmatrix=gs_camera.world_view_transform,
            projmatrix=gs_camera.full_proj_transform,
            sh_degree=sh_degree,
            campos=gs_camera.camera_center,
            prefiltered=False,
            debug=False
        )
        rasterizer = GaustudioGaussianRasterizer(raster_settings=raster_settings)
    elif rasterizer_type == 'surfel':
        h_size, w_size = float('inf'), float('inf')
        h_size, w_size = min(h_size, image_height), min(w_size, image_width)
        h0, w0 = random.randint(0, image_height - h_size), random.randint(0, image_width - w_size)
        h1, w1 = h0 + h_size, w0 + w_size
        patch_bbox = torch.tensor([h0, w0, h1, w1]).to(torch.float32).to(device)
        _raster_config = torch.tensor([1., 1., 1., 0.], dtype=torch.float32, device=device)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(image_height),
            image_width=int(image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=torch.tensor(bg_color).to(device),
            scale_modifier=1.,
            viewmatrix=gs_camera.world_view_transform,
            projmatrix=gs_camera.full_proj_transform,
            patch_bbox=patch_bbox,  # TODO: To check
            prcppoint=gs_camera.prcppoint,  # TODO: Implement this
            sh_degree=sh_degree,
            campos=gs_camera.camera_center,
            prefiltered=False,
            debug=False,
            config=_raster_config,
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    elif rasterizer_type == '2dgs':
        raster_settings = SurfelGaussianRasterizationSettings(
            image_height=int(image_height),
            image_width=int(image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=torch.tensor(bg_color).to(device),
            scale_modifier=1.,
            viewmatrix=gs_camera.world_view_transform,
            projmatrix=gs_camera.full_proj_transform,
            sh_degree=sh_degree,
            campos=gs_camera.camera_center,
            prefiltered=False,
            debug=False,
        )
        rasterizer = SurfelGaussianRasterizer(raster_settings=raster_settings)
    elif rasterizer_type == 'gof':
        raster_settings = GaussianRasterizationSettings(
            image_height=int(image_height),
            image_width=int(image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            kernel_size=0.,
            subpixel_offset=torch.zeros((int(image_height), int(image_width), 2), dtype=torch.float32, device=device),
            bg=torch.tensor(bg_color).to(device),
            scale_modifier=1.,
            viewmatrix=gs_camera.world_view_transform,
            projmatrix=gs_camera.full_proj_transform,
            sh_degree=sh_degree,
            campos=gs_camera.camera_center,
            prefiltered=False,
            debug=False
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    else: 
        raise ValueError("Invalid rasterizer type.")

    if point_colors is None:
        shs = torch.cat([sh_coordinates_dc, sh_coordinates_rest], dim=-2)
        splat_colors = None
    else:
        splat_colors = point_colors
        shs = None
    screenspace_points = torch.zeros(n_gaussians, 3, dtype=means.dtype, device=device)
    means2D = screenspace_points
    
    # Render images
    if (rasterizer_type == 'gaustudio'):
        rendered_image, radii, rendered_depth, rendered_median_depth, rendered_opac = rasterizer(
            means3D=means,
            means2D=means2D,
            shs=shs,
            colors_precomp=splat_colors,
            opacities=opacities,
            scales=scales,
            rotations=quaternions,
            cov3D_precomp=None)
        rendered_image = rendered_image.permute(1, 2, 0)
        rendered_depth = rendered_depth[0]
        rendered_median_depth = rendered_median_depth[0]
        rendered_normal = None
        render_distortion = None
        
    elif rasterizer_type == 'surfel':
        rendered_image, rendered_normal, rendered_depth, rendered_opac, radii = rasterizer(
            means3D=means,
            means2D=means2D,
            shs=shs,
            colors_precomp=splat_colors,
            opacities=opacities,
            scales=scales,
            rotations=quaternions,
            cov3D_precomp=None,
            )
        rendered_image = rendered_image.permute(1, 2, 0)
        rendered_depth = rendered_depth[0]
        rendered_median_depth = None
        rendered_normal = torch_normalize(rendered_normal.permute(1, 2, 0), dim=-1)
        render_distortion = None
    
    elif rasterizer_type == '2dgs':
        rendered_image, radii, allmap = rasterizer(
            means3D=means,
            means2D=means2D,
            shs=shs,
            colors_precomp=splat_colors,
            opacities=opacities,
            scales=scales[..., :2],
            rotations=quaternions,
            cov3D_precomp=None,
        )
        # additional regularizations
        rendered_opac = allmap[1:2]
        
        rendered_normal = allmap[2:5]
        if False:  # Transforming normals to world space        
            rendered_normal = (rendered_normal.permute(1, 2, 0) @ (gs_camera.world_view_transform[:3,:3].T))
        else:  # Not transforming normals to world space
            rendered_normal = rendered_normal.permute(1, 2, 0)
        
        rendered_median_depth = allmap[5:6]
        rendered_median_depth = torch.nan_to_num(rendered_median_depth, 0, 0)

        rendered_depth = allmap[0:1]
        rendered_depth = (rendered_depth / rendered_opac)
        rendered_depth = torch.nan_to_num(rendered_depth, 0, 0)
        
        render_distortion = allmap[6:7]
        
        rendered_image = rendered_image.permute(1, 2, 0)
        rendered_depth = rendered_depth[0]
        rendered_median_depth = rendered_median_depth[0]
        
        if False:
            print(f"Rendered opac:", rendered_opac.shape, rendered_opac.min(), rendered_opac.max())
            print(f"Rendered depth:", rendered_depth.shape, rendered_depth.min(), rendered_depth.max())
            print(f"Rendered median depth:", rendered_median_depth.shape, rendered_median_depth.min(), rendered_median_depth.max())
            print(f"Rendered normal:", rendered_normal.shape, rendered_normal.min(), rendered_normal.max())
            print(f"Rendered distortion:", render_distortion.shape, render_distortion.min(), render_distortion.max())
            
    elif rasterizer_type == 'gof':
        if use_integration:
            if integration_points is None:
                raise ValueError("integration_points must be provided for GOF integration")
            allmap, alpha_integrated, color_integrated, radii = rasterizer.integrate(
                points3D=integration_points,
                means3D=means,
                means2D=means2D,
                shs=shs,
                colors_precomp=splat_colors,
                opacities=opacities,
                scales=scales,
                rotations=quaternions,
                cov3D_precomp=None,
            )
        else:
            # Standard rendering for GOF
            allmap, radii = rasterizer(
                means3D=means,
                means2D=means2D,
                shs=shs,
                colors_precomp=splat_colors,
                opacities=opacities,
                scales=scales,
                rotations=quaternions,
                cov3D_precomp=None,
                view2gaussian_precomp=None,
            )
        rendered_image = allmap[0:3, :, :].permute(1, 2, 0)  # Shape (H, W, 3)
        rendered_normal = allmap[3:6, :, :].permute(1, 2, 0)  # Shape (H, W, 3)
        rendered_depth = allmap[6, :, :]  # Shape (H, W)
        rendered_opac = allmap[7:8, :, :]  # Shape (1, H, W)
        render_distortion = allmap[8:9, :, :]  # Shape (1, H, W)
        rendered_median_depth = rendered_depth.clone()
        
        # Normals are normalized for GOF!
        rendered_normal = torch_normalize(rendered_normal, p=2, dim=-1)
 
    render_pkg = {
        "rgb": rendered_image,  # Shape (H, W, 3)
        "radii": radii,  # Shape (n_viewspace_points, )
        "viewspace_points": screenspace_points,  # Shape (n_viewspace_points, 3)
        "depth": rendered_depth,  # Shape (H, W)
        "median_depth": rendered_median_depth,  # Shape (H, W)
        "normal": rendered_normal,  # Shape (H, W, 3)
        "opacity": rendered_opac,  # Shape (1, H, W)
        "distortion": render_distortion,  # Shape (1, H, W)
        "color_integrated": color_integrated if rasterizer_type == 'gof' and use_integration else None,
        "alpha_integrated": alpha_integrated if rasterizer_type == 'gof' and use_integration else None,
    }

    if return_whole_package:
        return render_pkg
    else:
        return render_pkg["render"]