import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), '2d-gaussian-splatting'))
from scene.dataset_readers import load_see3d_cameras
import torch
import numpy as np
from argparse import ArgumentParser
from arguments import ModelParams
from scene.dataset_readers import load_cameras
from utils.general_utils import safe_state
from guidance.cam_utils import project_points_to_image
import trimesh
from PIL import Image
import json
from matcha.dm_scene.cameras import GSCamera
import cv2
from scipy.stats import mode
from collections import Counter
from tqdm import tqdm
import time

def generate_distinct_colors(n_colors):
    """
    Generate n_colors distinct colors
    """
    colors = []
    for i in range(n_colors):
        # Use HSV color space to generate evenly distributed colors
        hue = i / n_colors
        saturation = 0.8
        value = 0.9
        
        # Convert to RGB
        h = int(hue * 360)
        s = int(saturation * 255)
        v = int(value * 255)
        
        # Use OpenCV for conversion
        rgb = cv2.cvtColor(np.array([[[h, s, v]]]).astype(np.uint8), cv2.COLOR_HSV2RGB)
        colors.append(rgb[0, 0].tolist())
    
    return colors

def get_global_plane_ids_for_view(camera, points, plane_mask, global_3Dplane_ID_dict, cur_view_id):
    """
    Get global plane IDs for all points in a single view
    
    Args:
        camera: GSCamera object
        points: torch.Tensor, [N, 3], all points
        plane_mask: torch.Tensor, [H, W], plane mask for this view
        global_3Dplane_ID_dict: dict, global 3D plane ID mapping
        cur_view_id: int, current view ID
    
    Returns:
        global_plane_ids: torch.Tensor, [N], global plane IDs for all points (-1 for invalid)
    """
    # Project all points to this view
    points_depth, points_2d, in_image = project_points_to_image(camera, points)
    in_image = in_image & (points_depth > 0)
    
    # Initialize result with default value -1
    global_plane_ids = torch.full((points.shape[0],), -1, dtype=torch.int, device=points.device)

    # Only process points within the image
    valid_indices = torch.nonzero(in_image).squeeze(-1)
    if len(valid_indices) == 0:
        return global_plane_ids
    
    valid_points_2d = points_2d[valid_indices]
    
    # Convert to integer pixel coordinates
    pixel_coords = torch.round(valid_points_2d).int()
    pixel_coords[:, 0] = torch.clamp(pixel_coords[:, 0], 0, plane_mask.shape[1] - 1)
    pixel_coords[:, 1] = torch.clamp(pixel_coords[:, 1], 0, plane_mask.shape[0] - 1)
    
    # Get corresponding plane mask values
    local_plane_ids = plane_mask[pixel_coords[:, 1], pixel_coords[:, 0]]
    
    # Map local plane ID to global plane ID
    for global_plane_id, local_plane_list in global_3Dplane_ID_dict.items():
        # Find local plane IDs belonging to this global plane
        for view_id, local_plane_id_in_dict in local_plane_list:
            if view_id != cur_view_id:
                continue

            # Here we need to match based on actual view_id, temporarily assume all views correspond
            mask = (local_plane_ids == local_plane_id_in_dict)
            global_plane_ids[valid_indices[mask]] = global_plane_id
    
    return global_plane_ids

def get_majority_global_plane_ids(global_plane_matrix):
    """
    For each point, compute the mode across views (excluding -1)
    
    Args:
        global_plane_matrix: torch.Tensor, [N, V], global plane IDs for each point and view
    
    Returns:
        final_global_plane_ids: torch.Tensor, [N], final global plane IDs
    """
    N, V = global_plane_matrix.shape
    final_global_plane_ids = torch.full((N,), -1, dtype=torch.int, device=global_plane_matrix.device)
    
    # Compute mode for each point
    for i in tqdm(range(N)):
        # Get global plane IDs for this point across all views (excluding -1)
        valid_plane_ids = global_plane_matrix[i]
        valid_plane_ids = valid_plane_ids[valid_plane_ids != -1]
        
        if len(valid_plane_ids) > 0:
            # Calculate mode
            counts = Counter(valid_plane_ids.cpu().numpy())
            most_common_plane_id = max(counts.items(), key=lambda x: x[1])[0]
            final_global_plane_ids[i] = most_common_plane_id
    
    return final_global_plane_ids

def get_majority_global_plane_ids_ultra_fast(global_plane_matrix):
    """
    Ultra-fast vectorized version using advanced PyTorch operations
    
    Args:
        global_plane_matrix: torch.Tensor, [N, V], global plane IDs for each point and view
    
    Returns:
        final_global_plane_ids: torch.Tensor, [N], final global plane IDs
    """
    N, V = global_plane_matrix.shape
    device = global_plane_matrix.device
    
    # Create mask for valid plane IDs (not -1)
    valid_mask = global_plane_matrix != -1
    
    # Initialize result tensor
    final_global_plane_ids = torch.full((N,), -1, dtype=torch.int, device=device)
    
    # Find points that have at least one valid plane ID
    has_valid_planes = valid_mask.sum(dim=1) > 0
    
    if not has_valid_planes.any():
        return final_global_plane_ids
    
    # Only process points with valid planes
    valid_points_mask = has_valid_planes
    valid_matrix = global_plane_matrix[valid_points_mask]
    valid_mask_subset = valid_mask[valid_points_mask]
    
    # Get the range of plane IDs
    if valid_mask.any():
        min_plane_id = global_plane_matrix[valid_mask].min().item()
        max_plane_id = global_plane_matrix[valid_mask].max().item()
        
        # Shift plane IDs to start from 0 for bincount
        shifted_matrix = valid_matrix - min_plane_id
        
        # Create a flattened view for vectorized processing
        n_valid_points = valid_points_mask.sum().item()
        
        # Process each valid point
        for i in tqdm(range(n_valid_points)):
            valid_planes = shifted_matrix[i][valid_mask_subset[i]]
            if len(valid_planes) > 0:
                # Use bincount to count occurrences
                counts = torch.bincount(valid_planes, minlength=max_plane_id - min_plane_id + 1)
                # Find the most frequent plane ID and shift back
                mode_plane_id = torch.argmax(counts).item() + min_plane_id
                
                # Find the original index in the full matrix
                original_idx = torch.nonzero(valid_points_mask)[i].item()
                final_global_plane_ids[original_idx] = mode_plane_id
    
    return final_global_plane_ids


def get_majority_global_plane_ids_scipy(global_plane_matrix):
    """
    Ultra-fast vectorized version using advanced PyTorch operations
    
    Args:
        global_plane_matrix: torch.Tensor, [N, V], global plane IDs for each point and view
    
    Returns:
        final_global_plane_ids: torch.Tensor, [N], final global plane IDs
    """

    device = global_plane_matrix.device
    global_plane_matrix = global_plane_matrix.cpu().numpy()

    mask_value = global_plane_matrix.max() + 1

    # Check if each row has valid values (not -1)
    has_valid = np.any(global_plane_matrix != -1, axis=1)

    # Replace -1 with mask_value (invalid value, to avoid becoming the mode)
    temp_map = np.where(global_plane_matrix == -1, mask_value, global_plane_matrix)

    # Calculate the mode
    modes, _ = mode(temp_map, axis=1, keepdims=False)

    # Replace the substituted values back
    modes[modes == mask_value] = -1

    # For cases where there are valid values but mode is -1, fallback to the minimum non-(-1) value
    fallback_mask = (modes == -1) & has_valid
    if np.any(fallback_mask):
        # Extract minimum non-(-1) value from global_plane_matrix (row by row)
        valid_min = np.min(np.where(global_plane_matrix[fallback_mask] == -1, mask_value, global_plane_matrix[fallback_mask]), axis=1)
        modes[fallback_mask] = valid_min

    final_global_plane_ids = torch.tensor(modes, dtype=torch.int, device=device)

    return final_global_plane_ids

def colorize_mesh_by_global_planes_fast(vertices, cameras, plane_masks, global_3Dplane_ID_dict):
    """
    Fast color assignment for mesh vertices based on global planes
    
    Args:
        vertices: torch.Tensor, [N, 3], mesh vertices
        cameras: list of GSCamera objects
        plane_masks: list of torch.Tensor, plane masks for each view
        global_3Dplane_ID_dict: dict, global 3D plane ID mapping
    
    Returns:
        vertex_colors: torch.Tensor, [N, 3], vertex colors
        vertex_plane_ids: torch.Tensor, [N], assigned plane IDs
    """
    num_vertices = vertices.shape[0]
    num_views = len(cameras)
    num_global_planes = len(global_3Dplane_ID_dict)
    
    print(f"Starting batch processing for {num_vertices} vertices, {num_views} views...")
    
    # Initialize [N, V] matrix with default value -1
    global_plane_matrix = torch.full((num_vertices, num_views), -1, dtype=torch.int, device=vertices.device)
    
    # Process each view in batch
    for view_id in range(num_views):
        print(f"Processing view {view_id + 1}/{num_views}")
        
        camera = cameras[view_id]
        plane_mask = plane_masks[view_id]
        
        # Get global plane IDs for all points in this view
        view_global_plane_ids = get_global_plane_ids_for_view(
            camera, vertices, plane_mask, global_3Dplane_ID_dict, view_id
        )
        
        # Fill into matrix
        global_plane_matrix[:, view_id] = view_global_plane_ids
    
    # Compute final global plane ID for each point (mode)
    print("Computing final global plane ID for each point...")
    # final_global_plane_ids = get_majority_global_plane_ids(global_plane_matrix)           # precise but slow
    final_global_plane_ids = get_majority_global_plane_ids_scipy(global_plane_matrix)       # fast but a bit not precise
    print("Computing final global plane ID for each point done!")
    
    # Generate colors
    colors = generate_distinct_colors(num_global_planes + 1)  # +1 for unassigned vertices
    
    # Assign colors to vertices
    vertex_colors = torch.zeros(num_vertices, 3, dtype=torch.uint8, device=vertices.device)
    
    for i in range(num_vertices):
        global_plane_id = final_global_plane_ids[i]
        if global_plane_id >= 0:
            vertex_colors[i] = torch.tensor(colors[global_plane_id], dtype=torch.uint8, device=vertices.device)
        else:
            # Use black for unassigned vertices
            vertex_colors[i] = torch.tensor([0, 0, 0], dtype=torch.uint8, device=vertices.device)
    
    return vertex_colors, final_global_plane_ids

def save_colored_mesh(mesh, vertex_colors, output_path):
    """
    Save colored mesh
    
    Args:
        mesh: trimesh.Trimesh, input mesh
        vertex_colors: torch.Tensor, [N, 3], vertex colors
        output_path: str, output file path
    """
    # Create new mesh object
    colored_mesh = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        vertex_colors=vertex_colors.cpu().numpy()
    )
    
    # Save as PLY file
    colored_mesh.export(output_path)
    print(f"Colored mesh saved to: {output_path}")



if __name__ == "__main__":
    # Set up command line arguments
    parser = ArgumentParser(description="Fast 3D Instance Segmentation for Mesh")
    parser.add_argument("--mesh_path", type=str, required=True, help="Path to input mesh file")
    parser.add_argument("--plane_root_path", type=str, required=True, help="Path to plane data directory")
    parser.add_argument("--see3d_root_path", type=str, default=None, help="Path to See3D data")
    parser.add_argument("--output_path", type=str, default=None, help="Output path for colored mesh")
    model = ModelParams(parser, sentinel=True)
    args = parser.parse_args()

    t1 = time.time()
    
    print('NOTE: Using training views for fast 3D instance segmentation')
    safe_state(False)
    
    # Load cameras
    train_viewpoints, _ = load_cameras(model.extract(args))
    
    if args.see3d_root_path is not None:
        camera_path = os.path.join(args.see3d_root_path, 'see3d_cameras.npz')
        inpaint_root_dir = os.path.join(args.see3d_root_path, 'inpainted_images')
        print(f'NOTE: Loading See3D cameras from {camera_path}')
        see3d_gs_cameras_list, _ = load_see3d_cameras(camera_path, inpaint_root_dir)
        train_viewpoints = train_viewpoints + see3d_gs_cameras_list
    
    # Load plane masks
    plane_root_path = args.plane_root_path
    plane_masks = []
    for i in range(len(train_viewpoints)):
        plane_mask_path = os.path.join(plane_root_path, f"plane_mask_frame{i:06d}.npy")
        plane_mask = np.load(plane_mask_path)
        plane_mask = torch.tensor(plane_mask, dtype=torch.int).cuda()
        plane_masks.append(plane_mask)
    
    # Load global 3D plane dict
    global_3Dplane_dict_path = os.path.join(plane_root_path, 'global_3Dplane_ID_dict.json')
    if not os.path.exists(global_3Dplane_dict_path):
        raise FileNotFoundError(f"Global 3D plane dict file does not exist: {global_3Dplane_dict_path}")
    
    with open(global_3Dplane_dict_path, 'r') as f:
        global_3Dplane_ID_dict = json.load(f)
    
    # Convert keys to int
    global_3Dplane_ID_dict = {int(k): v for k, v in global_3Dplane_ID_dict.items()}
    
    print(f"Loaded {len(global_3Dplane_ID_dict)} global 3D planes")
    
    # Load mesh
    mesh = trimesh.load(args.mesh_path)
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32).cuda()
    print(f"Loaded mesh with {len(vertices)} vertices")
    
    # Fast color assignment for vertices
    print("Starting fast batch processing...")
    vertex_colors, vertex_plane_ids = colorize_mesh_by_global_planes_fast(
        vertices, train_viewpoints, plane_masks, global_3Dplane_ID_dict
    )
    
    # Statistics
    assigned_vertices = (vertex_plane_ids >= 0).sum().item()
    total_vertices = len(vertices)
    print(f"Vertex assignment statistics:")
    print(f"  Total vertices: {total_vertices}")
    print(f"  Assigned vertices: {assigned_vertices}")
    print(f"  Assignment rate: {assigned_vertices/total_vertices*100:.2f}%")
    
    # Statistics for each global plane
    for global_plane_id in sorted(global_3Dplane_ID_dict.keys()):
        count = (vertex_plane_ids == global_plane_id).sum().item()
        print(f"  Global Plane {global_plane_id}: {count} vertices")
    
    save_colored_mesh(mesh, vertex_colors, args.output_path)
    
    # Save vertex plane ID mapping
    plane_id_map_path = args.output_path.replace('.ply', '_plane_ids.npy')
    np.save(plane_id_map_path, vertex_plane_ids.cpu().numpy())
    print(f"Vertex plane ID mapping saved to: {plane_id_map_path}")

    t2 = time.time()
    
    print(f"Fast 3D Instance Segmentation completed! Time cost: {t2 - t1:.2f}s")

