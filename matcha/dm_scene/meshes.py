import os
import torch
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.renderer import TexturesVertex, TexturesUV
from matcha.dm_scene.cameras import CamerasWrapper
from pytorch3d.renderer import (
    RasterizationSettings, 
    MeshRasterizer,  
    AmbientLights,
    SoftPhongShader,
)
from pytorch3d.renderer.blending import BlendParams
from pytorch3d.renderer.mesh.shader import ShaderBase


def remove_faces_from_single_mesh(
    mesh:Meshes, 
    faces_idx_to_keep:torch.Tensor=None,
    faces_to_keep_mask:torch.Tensor=None,
    ):
    """Returns a new Meshes object with only the faces specified by faces_idx_to_keep.
    If faces_to_keep_mask is provided instead, it is used to filter out the faces to remove.

    Args:
        mesh (Meshes): _description_
        faces_idx_to_keep (torch.Tensor): Has shape (n_faces_to_keep, ).
        faces_to_keep_mask (torch.Tensor): Has shape (n_faces, ).

    Returns:
        Meshes: _description_
    """
    assert len(mesh) == 1, "This function only works with single mesh objects."
    if (faces_to_keep_mask is None) and (faces_idx_to_keep is None):
        raise ValueError("Either faces_idx_to_keep or faces_to_keep_mask must be provided.")
    
    if faces_idx_to_keep is None:
        faces_idx_to_keep = torch.arange(0, faces_to_keep_mask.shape[0], device=faces_to_keep_mask.device)[faces_to_keep_mask]
    return mesh.submeshes([[faces_idx_to_keep]])


def remove_verts_from_single_mesh(
    mesh:Meshes, 
    verts_idx_to_keep:torch.Tensor=None,
    verts_to_keep_mask:torch.Tensor=None,
    ):
    """Returns a new Meshes object with only the vertices specified by verts_idx_to_keep.
    If verts_to_keep_mask is provided instead, it is used to filter out the vertices to remove.

    Args:
        mesh (Meshes): _description_
        verts_idx_to_keep (torch.Tensor): Has shape (n_verts_to_keep, ).
        verts_to_keep_mask (torch.Tensor): Has shape (n_verts, ).

    Returns:
        Meshes: _description_
    """
    assert len(mesh) == 1, "This function only works with single mesh objects."
    if (verts_idx_to_keep is None) and (verts_to_keep_mask is None):
        raise ValueError("Either verts_idx_to_keep or verts_to_keep_mask must be provided.")
    
    if verts_to_keep_mask is None:
        verts_to_keep_mask = torch.zeros(mesh.verts_packed().shape[0], device=verts_idx_to_keep.device, dtype=torch.bool)
        verts_to_keep_mask[verts_idx_to_keep] = True
    faces = mesh.faces_packed()
    faces_mask = verts_to_keep_mask[faces].any(dim=-1)
    return remove_faces_from_single_mesh(mesh, faces_to_keep_mask=faces_mask)


def get_vertices_with_n_neighbors(
    n_neighbors:int,
    mesh:Meshes=None,
    edges:torch.Tensor=None,
    ):
    """Returns the indices of the vertices that have n_neighbors neighbors, and the indices of their neighbors.

    Args:
        n_neighbors (int): _description_
        mesh (Meshes, optional): _description_. Defaults to None.
        edges (torch.Tensor, optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if mesh is None and edges is None:
        raise ValueError("Either mesh or edges must be provided.")
    if edges is None:
        edges = mesh.edges_packed()
    
    # Get oriented edges
    oriented_edges = torch.cat([edges, edges.flip(1)], dim=0)
    
    # Sort edges based on the first vertex of each edge
    first_vertex_of_edge = oriented_edges.index_select(dim=1, index=torch.tensor([0], dtype=torch.long, device=edges.device))[..., 0]
    sorted_idx = first_vertex_of_edge.argsort(dim=0)
    sorted_first_vertex_of_edge = first_vertex_of_edge[sorted_idx]
    sorted_oriented_edges = oriented_edges[sorted_idx]

    # Get vertices with n neighbors
    _, inverse, counts = sorted_first_vertex_of_edge.unique(dim=0, return_counts=True, return_inverse=True)
    mask = (counts == n_neighbors)[inverse]
    selected_edges = sorted_oriented_edges[mask].view(-1, n_neighbors, 2)
    verts_that_have_neighbors = selected_edges[..., 0, 0]
    neighbors = selected_edges[..., :, 1]
    
    # print(selected_edges[..., :, 0].float().std(dim=-1).max().item() == 0.)
    return verts_that_have_neighbors, neighbors


def get_faces_with_n_neighbors(
    n_neighbors,
    mesh=None,
    faces=None,
):
    """Returns the indices of the faces that have n_neighbors neighbors, and the indices of their neighbors.
    Each face should have no more than 3 neighbors.

    Args:
        n_neighbors (_type_): _description_
        mesh (_type_, optional): _description_. Defaults to None.
        faces (_type_, optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if mesh is None and faces is None:
        raise ValueError("Either mesh or faces must be provided.")
    if faces is None:
        faces = mesh.faces_packed()
        
    # Build connections between faces (there is a connection if two faces share an edge)
    sorted_faces = faces.sort(dim=-1)[0]
    edge_0 = sorted_faces[..., :2]
    edge_1 = sorted_faces[..., 1:]
    edge_2 = sorted_faces[..., ::2]
    
    face_idx = torch.arange(len(faces), device=faces.device)
    
    all_edges = torch.cat([edge_0, edge_1, edge_2], dim=0)
    all_face_idx = torch.cat([face_idx.clone(), face_idx.clone(), face_idx.clone()], dim=0)
    
    _, inverse, counts = all_edges.unique(dim=0, return_counts=True, return_inverse=True)
    mask = (counts == 2)[inverse]
    selected_edges = all_edges[mask]
    selected_face_idx = all_face_idx[mask]
    # Sort along each dim successively, so that all duo of faces are sorted together
    tmp = selected_edges[..., 1].argsort(dim=0)
    selected_edges = selected_edges[tmp]
    selected_face_idx = selected_face_idx[tmp]
    tmp = selected_edges[..., 0].argsort(dim=0)
    selected_edges = selected_edges[tmp]
    selected_face_idx = selected_face_idx[tmp]
    
    # Apply the same argsort to the indices, and reshape the indices as (-1, 2) to get connexions
    face_connexions = selected_face_idx.reshape(-1, 2)
    
    # See faces as verts and connections as edges, then apply verts method
    faces_that_have_neighbors, neighbors = get_vertices_with_n_neighbors(
        n_neighbors,
        mesh=None,
        edges=face_connexions,
    )
    
    return faces_that_have_neighbors, neighbors


def get_manifold_meshes_from_pointmaps(
    points3d:torch.Tensor, 
    imgs:torch.Tensor, 
    masks=None,
    return_single_mesh_object=False,
    return_manifold_idx=False,
    return_verts_neighbors=False,
    return_faces_neighbors=False,
    device=None,
    ):
    """Creates a list of Meshes objects from a list of pointmaps and images.
    Masks can be provided to filter out points in the pointmaps. 

    Args:
        points3d (:torch.Tensor): Has shape (n_images, height, width, 3)
        imgs (:torch.Tensor): Has shape (n_images, height, width, 3)
        masks (:torch.Tensor): Has shape (n_images, height, width)
        return_single_mesh_object (bool, optional): If True, the meshes are joined into a single Meshes object. Defaults to False.
        device (torch.device, optional): Device for the output Meshes object. Defaults to None. 
            If using many images, we recommend passing points3d and other tensors on cpu and providing the GPU device in this variable.
            Vertices will be filtered with masks and progressively moved to the device to avoid OOM issues.

    Returns:
        Meshes: _description_
    """
    # TODO: Replace vertex colors with a UVTexture + an Image of a given input resolution.
    
    if device is None:
        device = points3d[0].device
    
    n_points_per_col = points3d[0].shape[0]
    n_points_per_row = points3d[0].shape[1]

    verts_idx = torch.arange(n_points_per_row * n_points_per_col)[..., None].to(device)
    verts_idx = verts_idx.reshape(n_points_per_col, n_points_per_row)[:-1, :-1].reshape(-1, 1)

    faces_1 = torch.cat([
        verts_idx,
        verts_idx + n_points_per_row,
        verts_idx + 1,
    ], dim=-1)
    faces_2 = torch.cat([
        verts_idx + n_points_per_row + 1,
        verts_idx + 1,
        verts_idx + n_points_per_row,
    ], dim=-1)

    faces = torch.cat([faces_1, faces_2], dim=0)

    manifolds = []
    manifold_idx = torch.zeros(0, device=device, dtype=torch.int64)
    for i_ptmap in range(len(points3d)):
        vert_features = torch.nn.functional.pad(
            imgs[i_ptmap].view(1, -1, 3).clamp(0, 1).to(device), pad=(0, 1), value=1.,
        )
        manifold = Meshes(
            verts=[points3d[i_ptmap].view(-1, 3)], 
            faces=[faces],
            textures=TexturesVertex(verts_features=vert_features)
        ).to(device)
        if (masks is not None) and masks.any().item():
            manifold = remove_verts_from_single_mesh(manifold, verts_to_keep_mask=masks[i_ptmap].to(device).view(-1))
        manifolds.append(manifold)
        manifold_idx = torch.cat([
            manifold_idx, 
            torch.full(size=(manifold.verts_packed().shape[0],), fill_value=i_ptmap, device=device, dtype=torch.int64)
        ])
    if return_single_mesh_object:
        manifolds = join_meshes_as_scene(manifolds)
        
    if return_manifold_idx:
        return manifolds, manifold_idx
    return manifolds


def render_mesh_with_pytorch3d(
    mesh:Meshes, nerf_cameras:CamerasWrapper, camera_idx:int=0,
    rasterizer:MeshRasterizer=None, shader:ShaderBase=None,
    faces_per_pixel:int=1, max_faces_per_bin:int=50_000,
    background_color=(0., 0., 0.),
    use_gaussian_surfels_convention=True,
    flip_normals_toward_camera=True,
    ):
    """Renders a mesh using PyTorch3D's rasterizer and shader.
    Quite slow, should just be used for testing purposes.
    
    Args:
        mesh (Meshes): _description_
        nerf_cameras (CamerasWrapper): _description_
        camera_idx (int, optional): _description_. Defaults to 0.
        rasterizer (MeshRasterizer, optional): A custom mesh rasterizer can be provided. Defaults to None.
        shader (ShaderBase, optional): A custom shader can be provided. Defaults to None.
        faces_per_pixel (int, optional): _description_. Defaults to 1.
        max_faces_per_bin (int, optional): _description_. Defaults to 50_000.
        background_color (tuple, optional): _description_. Defaults to (0., 0., 0.).

    Returns:
        dict: Dictionary containing the rendered RGB, Depth and Normal images.
        RGB has shape (height, width, 3).
        Depth has shape (height, width). 
        Normal has shape (height, width, 3).
    """
    
    height, width = nerf_cameras.height[camera_idx].item(), nerf_cameras.width[camera_idx].item()
    p3d_camera = nerf_cameras.p3d_cameras[camera_idx]
    device = nerf_cameras.device

    if rasterizer is None:
        rasterizer = MeshRasterizer(
            cameras=p3d_camera, 
            raster_settings=RasterizationSettings(
                image_size=(height, width),
                blur_radius=0.0, 
                faces_per_pixel=faces_per_pixel,
                # max_faces_per_bin=max_faces_per_bin
            ),
        )
    if shader is None:
        shader = SoftPhongShader(
            device=device, 
            cameras=p3d_camera,
            lights=AmbientLights(device=device),
            blend_params=BlendParams(background_color=background_color),
        )

    # Rasterize mesh
    _mesh = mesh.clone()
    if isinstance(mesh.textures, TexturesVertex):
        _mesh.textures = TexturesVertex(verts_features=[_mesh.textures.verts_features_packed()[..., :3]])
    elif isinstance(mesh.textures, TexturesUV):
        pass
    else:
        raise ValueError("Textures type not supported.")
    fragments = rasterizer(_mesh, cameras=p3d_camera)

    # Compute RGB image
    rgb_img = shader(fragments, _mesh)[0, ..., :3]

    # Compute Depth
    depth_img = fragments.zbuf[0, ..., 0].clamp_min(0.)

    # Compute Normal
    mesh_normals = _mesh.faces_normals_list()[0].clone()
    
    # New
    if True:
        if flip_normals_toward_camera:
            faces_to_camera = nerf_cameras.p3d_cameras.get_camera_center()[camera_idx:camera_idx+1] - mesh.verts_packed()[mesh.faces_packed()].mean(dim=1)
            mesh_normals = torch.sign((faces_to_camera * mesh_normals).sum(dim=-1, keepdim=True)) * mesh_normals
        mesh_normals = p3d_camera.get_world_to_view_transform().transform_normals(mesh_normals)
    
    # Old
    else:
        mesh_normals = p3d_camera.get_world_to_view_transform().transform_normals(mesh_normals)
        if flip_normals_toward_camera:
            # In PyTorch3D, the camera looks in the positive z direction.
            # We want the normals to be oriented towards the camera, i.e. to have a negative z component.
            mesh_normals = - mesh_normals[..., 2:3].sign() * mesh_normals
    
    if use_gaussian_surfels_convention:
        mesh_normals[..., :2] = - mesh_normals[..., :2]  # Converts to COLMAP convention? This line is here to match Gaussian Surfel's normal images.
    pix_to_face = fragments.pix_to_face[0, ..., 0]
    bary_coords = fragments.bary_coords[0, ..., 0, :]
    
    normal_img = torch.zeros(depth_img.shape[0], depth_img.shape[1], 3, dtype=torch.float32, device=device)
    normal_img[pix_to_face != -1] = mesh_normals[pix_to_face][pix_to_face != -1]
    
    return {
        'rgb': rgb_img,
        'depth': depth_img,
        'normal': normal_img,
        'pix_to_face': pix_to_face,
        'bary_coords': bary_coords,
    }


def render_mesh_with_nvdiffrast(mesh, nerf_cameras):
    # TODO
    raise NotImplementedError("NVDiffRast rasterizer is not implemented yet.")


def save_mesh_with_vertex_colors_as_ply(
    save_path:str, 
    mesh:Meshes,
    make_dir_if_needed:bool=False,
):
    import open3d as o3d
    
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.verts_packed().cpu().numpy())
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces_packed().cpu().numpy())
    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(mesh.textures.verts_features_packed()[..., :3].cpu().numpy())
    
    if make_dir_if_needed:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    o3d.io.write_triangle_mesh(
        save_path, 
        o3d_mesh, 
        write_triangle_uvs=True, 
        write_vertex_colors=True, 
        write_vertex_normals=True
    )
