import numpy as np
import torch
from matcha.dm_scene.meshes import Meshes
import open3d as o3d
from pytorch3d.ops import knn_points


def project_mesh_on_points(
    pts:torch.Tensor,
    mesh,  # Open3D mesh
):
    verts = torch.tensor(np.asarray(mesh.vertices), device=pts.device, dtype=torch.float32)
    faces = torch.tensor(np.asarray(mesh.triangles), device=pts.device, dtype=torch.int64)
    colors = torch.tensor(np.asarray(mesh.vertex_colors), device=pts.device, dtype=torch.float32)
    
    proj_knn_idx = knn_points(
        verts[None], 
        pts[None], 
        K=1,
    ).idx[0][..., 0]
    
    new_mesh_verts = pts[proj_knn_idx]
    new_mesh_faces = faces
    new_mesh_colors = colors
    
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(new_mesh_verts.cpu().numpy())
    o3d_mesh.triangles = o3d.utility.Vector3iVector(new_mesh_faces.cpu().numpy())
    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(new_mesh_colors.cpu().numpy())
    o3d_mesh.compute_vertex_normals()
    
    print("Cleaning projected mesh...")
    o3d_mesh.remove_duplicated_vertices()
    o3d_mesh.remove_degenerate_triangles()
    o3d_mesh.remove_duplicated_triangles()
    o3d_mesh.remove_non_manifold_edges()
    print("Projection done.")
    
    return o3d_mesh


def project_points_on_mesh(
    mesh:Meshes, 
    pts:torch.Tensor,
    n_closest_triangles:int=6,
):
    face_centers = mesh.verts_packed()[mesh.faces_packed()].mean(dim=1)
    face_normals = mesh.faces_normals_packed()
    
    # We compute the K nearest faces to each point
    knn_idx = knn_points(
        pts[None],
        face_centers[None],
        K=n_closest_triangles,
    ).idx[0]  # Shape (N, K)
    closest_centers = face_centers[knn_idx]  # Shape (N, K, 3)
    closest_normals = face_normals[knn_idx]  # Shape (N, K, 3)
    closest_faces_verts = mesh.verts_packed()[mesh.faces_packed()[knn_idx]]  # Shape (N, K, 3, 3)
    
    # We project the points on the planes defined by the closest faces
    new_pts = pts[:, None] - ((pts[:, None] - closest_centers) * closest_normals).sum(dim=-1, keepdim=True) * closest_normals  # Shape (N, K, 3)
    print("New pts:", new_pts[:3])
    
    # We now identify the faces for which the projected points are inside the triangles...
    hyperplane_0_normals = torch.cross(closest_normals, closest_faces_verts[:, :, 2] - closest_faces_verts[:, :, 1])  # Shape (N, K, 3)
    hyperplane_1_normals = torch.cross(closest_normals, closest_faces_verts[:, :, 0] - closest_faces_verts[:, :, 2])  # Shape (N, K, 3)
    hyperplane_2_normals = torch.cross(closest_normals, closest_faces_verts[:, :, 1] - closest_faces_verts[:, :, 0])  # Shape (N, K, 3)
    
    inside_space_0 = (
        (hyperplane_0_normals * (new_pts - closest_faces_verts[:, :, 1])).sum(dim=-1).sign() 
        == (hyperplane_0_normals * (closest_faces_verts[:, :, 0] - closest_faces_verts[:, :, 1])).sum(dim=-1).sign()
    )  # Shape (N, K)
    inside_space_1 = (
        (hyperplane_1_normals * (new_pts - closest_faces_verts[:, :, 2])).sum(dim=-1).sign() 
        == (hyperplane_1_normals * (closest_faces_verts[:, :, 1] - closest_faces_verts[:, :, 2])).sum(dim=-1).sign()
    )  # Shape (N, K)
    inside_space_2 = (
        (hyperplane_2_normals * (new_pts - closest_faces_verts[:, :, 0])).sum(dim=-1).sign() 
        == (hyperplane_2_normals * (closest_faces_verts[:, :, 2] - closest_faces_verts[:, :, 0])).sum(dim=-1).sign()
    )  # Shape (N, K)
    inside_triangle = inside_space_0 & inside_space_1 & inside_space_2  # Shape (N, K)
    
    # ...And we keep the closest one.
    distances = (new_pts - pts[:, None]).norm(dim=-1)  # Shape (N, K)
    _, sorted_idx = distances.sort(dim=-1)  # Shape (N, K), Shape (N, K)
    sorted_pts = torch.gather(input=new_pts, dim=1, index=sorted_idx.unsqueeze(-1).expand(-1, -1, 3))  # Shape (N, K, 3)
    sorted_inside_triangle = torch.gather(input=inside_triangle, dim=1, index=sorted_idx)  # Shape (N, K)
    closest_pts_in_triangle_idx = torch.argmax(sorted_inside_triangle.float(), dim=-1)  # Shape (N,)
    closest_pts_in_triangle = sorted_pts[torch.arange(sorted_pts.shape[0]), closest_pts_in_triangle_idx]  # Shape (N, 3)
    
    # If there is no face for which the projected point is inside the triangle, 
    # it means that the point is "between" the faces, and we should project it on the closest vertex (not the closest face!)
    no_triangle = ~(sorted_inside_triangle.any(dim=-1))  # Shape (N,)
    points_with_no_triangle = pts[no_triangle]
    closest_vert_idx = knn_points(
        points_with_no_triangle[None],
        mesh.verts_packed()[None],
        K=1,
    ).idx[0][..., 0]  # Shape (N,)
    closest_pts_in_triangle[no_triangle] = mesh.verts_packed()[closest_vert_idx]
    
    return closest_pts_in_triangle