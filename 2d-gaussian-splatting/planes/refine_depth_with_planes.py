import torch
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), '2d-gaussian-splatting'))
from scene.dataset_readers import load_see3d_cameras
from scene.dataset_readers import load_cameras
from utils.general_utils import safe_state
from utils.render_utils import save_img_f32, save_img_u8
from argparse import ArgumentParser
from arguments import ModelParams
from matcha.dm_scene.charts import depths_to_points_parallel
from matcha.dm_scene.cameras import GSCamera
from matcha.pointmap.depthanythingv2 import depth_linear_align
import trimesh
import numpy as np
import cv2
import pickle
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt
import json
from sklearn.base import BaseEstimator, RegressorMixin


class GeneralPlaneRegressor(BaseEstimator, RegressorMixin):
    """
    General plane regressor using ax + by + cz + d = 0 representation
    """
    def __init__(self, alpha=1.0, prior_normal=None):
        """
        Args:
            alpha: Regularization strength
            prior_normal: Prior normal direction [n1, n2, n3], should be normalized
        """
        self.alpha = alpha
        self.prior_normal = prior_normal
        self.coef_ = None  # [a, b, c, d] coefficients
        
    def fit(self, X, y=None):
        """
        Fit general plane equation ax + by + cz + d = 0
        
        Args:
            X: Points array of shape [N, 3]
            y: Not used (for sklearn compatibility)
        """
        points = X  # Shape [N, 3]
        N = points.shape[0]
        
        if self.prior_normal is not None:
            # Use regularized fitting with normal constraint
            self.coef_ = self._fit_with_normal_constraint(points)
        else:
            # Use SVD-based fitting
            self.coef_ = self._fit_with_svd(points)
            
        return self
    
    def _fit_with_svd(self, points):
        """
        Fit plane using SVD (Principal Component Analysis approach)
        """
        # Center the points
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        
        # Compute SVD
        U, S, Vt = np.linalg.svd(centered_points, full_matrices=False)
        
        # The normal vector is the last column of V (row of Vt)
        normal = Vt[-1, :]  # Shape [3]
        
        # Compute d: ax + by + cz + d = 0, so d = -(a*cx + b*cy + c*cz)
        d = -np.dot(normal, centroid)
        
        return np.array([normal[0], normal[1], normal[2], d])
    
    def _fit_with_normal_constraint(self, points):
        """
        Fit plane with normal direction constraint using optimization
        """

        points = points.astype(np.float64)
        if self.prior_normal is not None:
            self.prior_normal = np.array(self.prior_normal, dtype=np.float64)

        def objective(params):
            a, b, c, d = params
            
            # Normalize the normal vector
            normal_norm = np.sqrt(a*a + b*b + c*c)
            if normal_norm < 1e-8:
                return 1e10
            
            a_norm, b_norm, c_norm = a/normal_norm, b/normal_norm, c/normal_norm
            d_norm = d/normal_norm
            
            # Compute point-to-plane distances
            distances = np.abs(points[:, 0] * a_norm + 
                             points[:, 1] * b_norm + 
                             points[:, 2] * c_norm + d_norm)
            
            # Main loss: mean squared distance
            mse_loss = np.mean(distances ** 2)
            
            # Regularization: constrain normal directions to be collinear
            if self.prior_normal is not None:
                current_normal = np.array([a_norm, b_norm, c_norm])
                dot_product = np.dot(self.prior_normal, current_normal)
                # Regularization: minimize (1 - |dot_product|)^2
                regularization = self.alpha * (1.0 - np.abs(dot_product)) ** 2
            else:
                regularization = 0.0
            
            return mse_loss + regularization
        
        # Initial guess using SVD
        initial_coef = self._fit_with_svd(points)
        
        # If prior normal is available, adjust initial guess
        if self.prior_normal is not None:
            # Use prior normal as initial normal, compute d from centroid
            centroid = np.mean(points, axis=0)
            initial_coef[:3] = self.prior_normal
            initial_coef[3] = -np.dot(self.prior_normal, centroid)
        
        # Optimization with normalization constraint
        def constraint_fun(params):
            a, b, c, d = params
            return a*a + b*b + c*c - 1.0  # ||normal|| = 1
        
        from scipy.optimize import minimize
        constraint = {'type': 'eq', 'fun': constraint_fun}
        
        result = minimize(objective, initial_coef, method='SLSQP', constraints=constraint)
        
        if result.success:
            return result.x.astype(np.float64)
        else:
            # Fallback to SVD result
            return initial_coef
    
    def predict(self, X):
        """
        Compute signed distances to the plane (for compatibility with RANSAC)
        """
        if self.coef_ is None:
            raise ValueError("Model not fitted yet")
        
        a, b, c, d = self.coef_
        # Normalize coefficients
        normal_norm = np.sqrt(a*a + b*b + c*c)
        if normal_norm < 1e-8:
            return np.zeros(X.shape[0])
        
        a_norm, b_norm, c_norm, d_norm = a/normal_norm, b/normal_norm, c/normal_norm, d/normal_norm
        
        # Return signed distances
        return X[:, 0] * a_norm + X[:, 1] * b_norm + X[:, 2] * c_norm + d_norm
    
    def get_plane_params(self):
        """
        Get normalized plane parameters
        """
        if self.coef_ is None:
            raise ValueError("Model not fitted yet")
        
        a, b, c, d = self.coef_
        normal_norm = np.sqrt(a*a + b*b + c*c)
        if normal_norm < 1e-8:
            return np.array([0, 0, 1]), 0
        
        normal = np.array([a, b, c]) / normal_norm
        if abs(c) > 1e-5:
            z_offset = -1.0 * d / c
            center = np.array([0, 0, z_offset])
        else:
            if abs(b) > 1e-5:
                y_offset = -1.0 * d / b
                center = np.array([0, y_offset, 0])
            else:
                x_offset = -1.0 * d / a
                center = np.array([x_offset, 0, 0])

        return normal, center


def save_tensor_as_pcd(pcd, path, pcd_colors=None):

    if isinstance(pcd, torch.Tensor):
        pcd = pcd.detach().cpu().numpy()
    pcd = trimesh.PointCloud(pcd)
    if pcd_colors is not None:
        if isinstance(pcd_colors, torch.Tensor):
            pcd_colors = pcd_colors.detach().cpu().numpy()
        pcd.colors = pcd_colors
    pcd.export(path)

def normals_cluster_1d(valid_normals_1d, n_init_clusters=8, n_clusters=6, min_size_ratio=0.004):
    """
    Cluster 1D normal vectors and return 1D cluster masks
    
    Args:
        valid_normals_1d: Normal vectors from valid region (N, 3)
        n_init_clusters: Initial number of clusters for KMeans
        n_clusters: Number of clusters to keep after filtering
        min_size_ratio: Minimum size ratio for valid clusters
    
    Returns:
        cluster_masks: List of 1D cluster masks (each is boolean array of length N)
        cluster_centers: Cluster centers (normal directions)
    """
    min_cluster_size = valid_normals_1d.shape[0] * min_size_ratio

    # KMeans clustering on 1D normals
    kmeans = KMeans(n_clusters=n_init_clusters, random_state=0, n_init=1).fit(valid_normals_1d)
    pred_1d = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    # Select top clusters by size
    count_values = np.bincount(pred_1d)
    topk = np.argpartition(count_values, -n_clusters)[-n_clusters:]
    sorted_topk_idx = np.argsort(count_values[topk])
    sorted_topk = topk[sorted_topk_idx][::-1]
    
    cluster_masks = []
    cluster_centers = []
    
    for cluster_id in sorted_topk:
        # Create 1D mask for this cluster
        cluster_mask_1d = (pred_1d == cluster_id)
        
        # Filter by minimum size
        if cluster_mask_1d.sum() < min_cluster_size:
            continue
        
        cluster_masks.append(cluster_mask_1d)
        
        # Normalize cluster center
        center_norm = centers[cluster_id] / np.linalg.norm(centers[cluster_id])
        cluster_centers.append(center_norm)
    
    return cluster_masks, np.array(cluster_centers)

def compute_plane_aligned_depth(plane_normal, plane_center, camera, img_shape):
    """
    Compute depth map by intersecting camera rays with 3D plane
    
    Args:
        plane_normal: Plane normal vector (3,) in world coordinates
        plane_center: Point on the plane (3,) in world coordinates  
        camera: Camera object with intrinsics and extrinsics
        img_shape: Image shape (H, W)
    
    Returns:
        aligned_depth: Depth map aligned to the plane (H, W)
    """
    H, W = img_shape
    device = plane_normal.device
    
    # Get camera parameters
    fx = camera.focal_x
    fy = camera.focal_y
    K = torch.tensor([[fx, 0, W/2], [0, fy, H/2], [0, 0, 1]], device=device).float()  # Intrinsic matrix (3, 3)
    c2w = camera.world_view_transform.inverse().T  # Camera to world transform (4, 4)
    
    # Camera center in world coordinates
    camera_center = c2w[:3, 3]  # (3,)
    
    # Create pixel coordinates
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    pixels = torch.stack([x, y, torch.ones_like(x)], dim=-1).float()  # (H, W, 3)
    
    # Convert to normalized camera coordinates
    K_inv = torch.inverse(K)
    ray_dirs_cam = torch.matmul(pixels, K_inv.T)  # (H, W, 3)
    
    # Transform ray directions to world coordinates
    ray_dirs_world = torch.matmul(ray_dirs_cam, c2w[:3, :3].T)  # (H, W, 3)
    ray_dirs_world = torch.nn.functional.normalize(ray_dirs_world, dim=-1)  # Normalize
    
    # Compute ray-plane intersection for each pixel
    # Plane equation: n·(P - P0) = 0
    # Ray equation: P = O + t*D
    # Solve: n·(O + t*D - P0) = 0 => t = n·(P0 - O) / (n·D)
    
    plane_normal = plane_normal.view(1, 1, 3)  # (1, 1, 3)
    plane_center = plane_center.view(1, 1, 3)  # (1, 1, 3)
    camera_center = camera_center.view(1, 1, 3)  # (1, 1, 3)
    
    # Compute dot products
    n_dot_d = torch.sum(plane_normal * ray_dirs_world, dim=-1)  # (H, W)
    n_dot_pc_minus_o = torch.sum(plane_normal * (plane_center - camera_center), dim=-1)  # (H, W)
    
    # Avoid division by zero (rays parallel to plane)
    eps = 1e-8
    n_dot_d = torch.clamp(torch.abs(n_dot_d), min=eps) * torch.sign(n_dot_d)
    
    # Compute intersection parameter t
    t = n_dot_pc_minus_o / n_dot_d  # (H, W)
    
    # Compute intersection points
    intersection_points = camera_center + t.unsqueeze(-1) * ray_dirs_world  # (H, W, 3)
    
    # Convert intersection points from world coordinates to camera coordinates
    # Transform to homogeneous coordinates
    intersection_points_homo = torch.cat([intersection_points, torch.ones_like(intersection_points[..., :1])], dim=-1)  # (H, W, 4)
    
    # World to camera transform (inverse of camera to world)
    w2c = torch.inverse(c2w)  # (4, 4)
    
    # Transform intersection points to camera coordinates
    intersection_points_cam = torch.matmul(intersection_points_homo, w2c.T)  # (H, W, 4)
    
    # Extract z-depth (distance along camera z-axis)
    aligned_depth = intersection_points_cam[..., 2]  # (H, W) - z coordinate in camera space
    
    # Handle invalid intersections (behind camera or negative depth)
    valid_mask = (t > 0) & (aligned_depth > 0)
    aligned_depth[~valid_mask] = 0

    valid_intersection_points = intersection_points[valid_mask]
    
    return aligned_depth, valid_intersection_points

def create_overlay_visualization(rgb_image, mask_obj, transparency=0.6, color=[0, 0, 255]):
    """
    Create overlay visualization of mask on RGB image
    
    Args:
        rgb_image: RGB image (H, W, 3)
        mask_obj: Binary mask (H, W)
        transparency: Mask transparency
        color: Specified color, default is blue
    
    Returns:
        blended: Overlaid image
    """
    # Ensure correct input format
    if rgb_image.dtype != np.uint8:
        rgb_image = (rgb_image * 255).astype(np.uint8)
    
    # Create colored mask
    colored_mask = np.zeros_like(rgb_image)
    colored_mask[mask_obj] = color
    
    # Blend images
    blended = rgb_image.copy()
    mask_indices = mask_obj
    
    for i in range(3):
        blended[:, :, i] = np.where(
            mask_indices,
            rgb_image[:, :, i] * (1 - transparency) + colored_mask[:, :, i] * transparency,
            rgb_image[:, :, i]
        )
    
    return blended.astype(np.uint8)

def vis_3Dplane(plane_normal, plane_center, grid_size=1.0, num_points_per_side=21, save_path=None):
    """
    Visualize 3D plane
    
    Args:
        plane_normal: Plane normal vector (3,) in world coordinates
        plane_center: Center point on the plane (3,) in world coordinates
        grid_size: Size of the grid in meters (e.g., 1.0 for 1m x 1m)
        num_points_per_side: Number of points per side of the grid
        save_path: Path to save the plane points
    
    Returns:
        plane_points: Points on the plane (N, 3)
    """
    # Convert to numpy for easier computation
    if isinstance(plane_normal, torch.Tensor):
        plane_normal = plane_normal.cpu().numpy()
    if isinstance(plane_center, torch.Tensor):
        plane_center = plane_center.cpu().numpy()
    
    # Find two orthogonal vectors in the plane
    # Choose an arbitrary vector not parallel to the normal
    if abs(plane_normal[0]) < 0.9:
        arbitrary_vec = np.array([1.0, 0.0, 0.0])
    else:
        arbitrary_vec = np.array([0.0, 1.0, 0.0])
    
    # First tangent vector (cross product)
    tangent1 = np.cross(plane_normal, arbitrary_vec)
    tangent1 = tangent1 / np.linalg.norm(tangent1)
    
    # Second tangent vector (cross product of normal and first tangent)
    tangent2 = np.cross(plane_normal, tangent1)
    tangent2 = tangent2 / np.linalg.norm(tangent2)
    
    # Generate grid coordinates
    half_size = grid_size / 2.0
    coords = np.linspace(-half_size, half_size, num_points_per_side)
    u_coords, v_coords = np.meshgrid(coords, coords)
    
    # Flatten the grid
    u_flat = u_coords.flatten()
    v_flat = v_coords.flatten()
    
    # Generate 3D points on the plane
    plane_points = []
    for u, v in zip(u_flat, v_flat):
        point = plane_center + u * tangent1 + v * tangent2
        plane_points.append(point)

    plane_points_tensor = torch.tensor(plane_points).float()
    if save_path is not None:
        save_tensor_as_pcd(plane_points_tensor, save_path)
        print(f"Saved plane points to {save_path}")

    return plane_points_tensor

def fit_plane_ransac(pnts, threshold=0.01, min_samples=3, max_trials=1000, 
                    alpha=1.0, prior_normal=None):
    """
    Fit a plane to 3D points using RANSAC with general plane equation ax + by + cz + d = 0
    
    Args:
        pnts: numpy array of shape [N, 3], representing N 3D points
        threshold: Distance threshold to determine inliers
        min_samples: Minimum number of data points to fit the model
        max_trials: Maximum number of iterations for RANSAC
        use_normal_constraint: Whether to use normal direction constraints
        alpha: Regularization strength (only used when use_normal_constraint=True)
        prior_normal: Prior normal direction [n1, n2, n3] (only used when use_normal_constraint=True)
    
    Returns:
        plane_normal: Normal vector of the plane [a, b, c] (normalized)
        d: Offset in the plane equation ax + by + cz + d = 0 (normalized)
        inlier_mask: Boolean mask of inliers, shape [N,]
    """

    if prior_normal is not None:
        use_normal_constraint = True
    else:
        use_normal_constraint = False
    
    # Normalize prior_normal if provided
    if prior_normal is not None:
        prior_normal = np.array(prior_normal)
        prior_normal = prior_normal / np.linalg.norm(prior_normal)
    
    # Create general plane regressor
    if use_normal_constraint:
        base_regressor = GeneralPlaneRegressor(alpha=alpha, prior_normal=prior_normal)
    else:
        base_regressor = GeneralPlaneRegressor()
    
    # Use RANSAC for robust fitting
    ransac = RANSACRegressor(
        estimator=base_regressor,
        residual_threshold=threshold,
        min_samples=min_samples,
        max_trials=max_trials,
        random_state=42
    )
    
    # Fit using all 3D coordinates (no y needed for general plane fitting)
    ransac.fit(pnts, np.zeros(pnts.shape[0]))  # Dummy y for sklearn compatibility
    
    # Get plane parameters
    plane_normal, plane_center = ransac.estimator_.get_plane_params()
    
    return plane_normal, plane_center, ransac.inlier_mask_


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    parser.add_argument("--plane_root_path", required=True, type=str)
    parser.add_argument("--see3d_root_path", type=str, default=None)
    args = parser.parse_args()

    print('NOTE: Using training views from data_path')
    # Initialize system state (RNG)
    safe_state(False)

    train_viewpoints, _ = load_cameras(model.extract(args))
    input_view_num = len(train_viewpoints)

    if args.see3d_root_path is not None:
        camera_path = os.path.join(args.see3d_root_path, 'see3d_cameras.npz')
        inpaint_root_dir = os.path.join(args.see3d_root_path, 'inpainted_images')
        print(f'NOTE: Using training views from camera_path {camera_path}')
        # Load See3D cameras
        see3d_gs_cameras_list, _ = load_see3d_cameras(camera_path, inpaint_root_dir)

        train_viewpoints = train_viewpoints + see3d_gs_cameras_list

    data_path = args.plane_root_path

    # load data
    view_points_list = []
    rend_normal_list = []
    plane_mask_list = []
    conf_map_list = []
    depth_list = []
    mono_normal_world_list = []

    for i in range(len(train_viewpoints)):
        depth_path = os.path.join(data_path, f"depth_frame{i:06d}.tiff")
        depth = np.array(Image.open(depth_path))
        depth = torch.from_numpy(depth).to('cuda')

        # get pnts
        view_points = depths_to_points_parallel(depth, [train_viewpoints[i]])       # [1, H*W, 3]
        view_points = view_points.squeeze(0)

        # get rend normal in world coordinate
        rend_normal_path = os.path.join(data_path, f"depth_normal_world_frame{i:06d}.npy")
        rend_normal = np.load(rend_normal_path)
        rend_normal = torch.from_numpy(rend_normal).to('cuda')

        # get mono normal in world coordinate
        mono_normal_world_path = os.path.join(data_path, f"mono_normal_world_frame{i:06d}.npy")
        mono_normal_world = np.load(mono_normal_world_path)
        mono_normal_world = torch.from_numpy(mono_normal_world).to('cuda')

        # load plane mask
        plane_mask_path = os.path.join(data_path, f"plane_mask_frame{i:06d}.npy")
        plane_mask = np.load(plane_mask_path)

        conf_path = os.path.join(data_path, f"visibility_frame{i:06d}.npy")
        conf_map = np.load(conf_path)
        conf_map = (conf_map > 0.5)

        view_points_list.append(view_points)
        rend_normal_list.append(rend_normal)
        plane_mask_list.append(plane_mask)
        conf_map_list.append(conf_map)
        depth_list.append(depth)
        mono_normal_world_list.append(mono_normal_world)

    # load global 3Dplane ID dict
    global_3Dplane_ID_dict_path = os.path.join(data_path, 'global_3Dplane_ID_dict.json')
    with open(global_3Dplane_ID_dict_path, 'r') as f:
        temp_global_3Dplane_ID_dict = json.load(f)
    global_3Dplane_ID_dict = {int(k): v for k, v in temp_global_3Dplane_ID_dict.items()}

    # align global 3D plane
    for key, value in global_3Dplane_ID_dict.items():
        global_3Dplane_ID = int(key)
        local_3Dplane_idx = value                               # [[view_id, plane_id], ...]

        global_3Dplane_pnts = []                                # save all points for this global 3D plane
        global_3Dplane_mono_normal_world = []
        for view_id, plane_id in local_3Dplane_idx:
            view_points = view_points_list[view_id]
            rend_normal = rend_normal_list[view_id]
            plane_mask = plane_mask_list[view_id]
            conf_map = conf_map_list[view_id]

            # get valid points for alignment
            mask = (plane_mask == plane_id).astype(np.float32)
            mask = (mask > 0.5)

            valid_mask = mask & conf_map
            valid_mask = torch.from_numpy(valid_mask).to('cuda')
            valid_points = view_points[valid_mask.reshape(-1)]
            if valid_points.shape[0] < 20:                          # too few points, skip
                continue

            # get max valid plane regions from rend normal
            valid_rend_normal = (rend_normal[valid_mask]).reshape(-1, 3)
            cluster_masks, cluster_centers = normals_cluster_1d(valid_rend_normal.cpu().numpy())
            max_plane_mask = cluster_masks[0]           # already sorted by area
            max_plane_points = valid_points[max_plane_mask]
            global_3Dplane_pnts.append(max_plane_points)

            # # get mono normal max cluster center in world coordinate
            # mono_normal_world = mono_normal_world_list[view_id]
            # # valid_mono_normal_world = (mono_normal_world[valid_mask]).reshape(-1, 3)
            # valid_mono_normal_world = (mono_normal_world[mask]).reshape(-1, 3)
            # _, cluster_centers_mono = normals_cluster_1d(valid_mono_normal_world.cpu().numpy())
            # global_3Dplane_mono_normal_world.append(cluster_centers_mono[0])

            # use rend normal max cluster center in world coordinate
            global_3Dplane_mono_normal_world.append(cluster_centers[0])

        if len(global_3Dplane_pnts) == 0:
            continue

        # get prior normal
        global_3Dplane_prior_normal = np.stack(global_3Dplane_mono_normal_world, axis=0)
        global_3Dplane_prior_normal = global_3Dplane_prior_normal.mean(axis=0)
        global_3Dplane_prior_normal = global_3Dplane_prior_normal / np.linalg.norm(global_3Dplane_prior_normal)

        # fit global 3D plane use all points
        global_3Dplane_pnts = torch.cat(global_3Dplane_pnts, dim=0)
        global_3Dplane_normal, global_3Dplane_center, _ = fit_plane_ransac(global_3Dplane_pnts.cpu().numpy(), prior_normal=global_3Dplane_prior_normal)
        global_3Dplane_normal = torch.from_numpy(global_3Dplane_normal).to('cuda').float()
        global_3Dplane_center = torch.from_numpy(global_3Dplane_center).to('cuda').float()

        # # vis global 3D plane
        # vis_3Dplane(global_3Dplane_normal, global_3Dplane_center, grid_size=3.0, save_path=os.path.join(data_path, f"global_3Dplane_{global_3Dplane_ID}.ply"))
        # print(f'global 3Dplane {global_3Dplane_ID} saved')

        # get plane aligned depth for each local 3D plane
        for view_id, plane_id in local_3Dplane_idx:

            plane_mask = plane_mask_list[view_id]
            conf_map = conf_map_list[view_id]
            mask = (plane_mask == plane_id).astype(np.float32)
            mask = (mask > 0.5)

            aligned_depth, valid_intersection_points = compute_plane_aligned_depth(global_3Dplane_normal, global_3Dplane_center, train_viewpoints[view_id], depth_list[view_id].shape)
            
            # replace depth use aligned depth in mask region
            mask = torch.from_numpy(mask).to('cuda')
            depth_list[view_id][mask] = aligned_depth[mask]

    # refine non-plane region for see3d views
    for view_id in range(len(train_viewpoints)):

        if view_id < input_view_num:
            continue

        plane_refined_depth = depth_list[view_id]
        plane_region_mask = (plane_mask_list[view_id] > 0.5)
        plane_region_mask = torch.from_numpy(plane_region_mask).to('cuda')
        mono_depth_path = os.path.join(data_path, f"mono_depth_frame{view_id:06d}.tiff")
        mono_depth = np.array(Image.open(mono_depth_path))
        mono_depth = torch.from_numpy(mono_depth).to('cuda').float()
        mono_disp = 1. / mono_depth

        # linear align mono_depth to real scale according to plane_refined_depth in plane region
        conf_map = conf_map_list[view_id]
        conf_map = torch.from_numpy(conf_map).to('cuda')
        mono_aligned_depth = depth_linear_align(disp=mono_disp, render_depth=plane_refined_depth, visible_mask=conf_map)
        non_plane_region_mask = ~plane_region_mask
        non_conf_region_mask = ~conf_map
        replace_region_mask = non_plane_region_mask & non_conf_region_mask
        depth_list[view_id][replace_region_mask] = mono_aligned_depth[replace_region_mask]


    # save refine depth
    for view_id in range(len(train_viewpoints)):
        save_img_f32(depth_list[view_id].cpu().numpy(), os.path.join(data_path, f"refine_depth_frame{view_id:06d}.tiff"))
        print(f'refine depth saved to {os.path.join(data_path, f"refine_depth_frame{view_id:06d}.tiff")}')
    
        # save refine points
        view_points = depths_to_points_parallel(depth_list[view_id], [train_viewpoints[view_id]])
        view_points = view_points.squeeze(0)
        save_tensor_as_pcd(view_points, os.path.join(data_path, f"refine_points_frame{view_id:06d}.ply"))
        print(f'refine points saved to {os.path.join(data_path, f"refine_points_frame{view_id:06d}.ply")}')
    
    print('done')


