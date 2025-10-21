import math
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d
import pytransform3d.visualizer as pv

import torch
import random

from utils.graphics_utils import getProjectionMatrix
from matcha.dm_scene.charts import project_points, transform_points_world_to_view

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def to_tensor_safe(data, dtype=torch.float32, device='cuda'):
    """Safely convert data to tensor, handling both numpy arrays and tensors"""
    if isinstance(data, torch.Tensor):
        return data.to(dtype=dtype, device=device)
    else:
        return torch.from_numpy(data).to(dtype=dtype, device=device)

def vis_camera_pose(poses, mesh_path=None):
    fig = pv.figure()
    
    # NOTE: Hard code camera intrinsic matrix
    fovy_deg = 60
    fovy = np.deg2rad(fovy_deg)
    fovx = fovy
    w, h = 512, 512
    K = np.zeros((3, 3))
    K[0, 0] = fov2focal(fovx, w)
    K[1, 1] = fov2focal(fovy, h)
    K[0, 2] = w // 2
    K[1, 2] = h // 2
    K[2, 2] = 1
    sensor_size = (float(w), float(h))

    if mesh_path is not None:
        fig.plot_mesh(mesh_path)
    for pose in poses:
        fig.plot_transform(A2B=pose, s=0.1)
        fig.plot_camera(M=K, cam2world=pose, virtual_image_distance=0.1, sensor_size=sensor_size)
    fig.show()

def dot(x, y):
    if isinstance(x, np.ndarray):
        return np.sum(x * y, -1, keepdims=True)
    else:
        return torch.sum(x * y, -1, keepdim=True)


def length(x, eps=1e-20):
    if isinstance(x, np.ndarray):
        return np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), eps))
    else:
        return torch.sqrt(torch.clamp(dot(x, x), min=eps))


def safe_normalize(x, eps=1e-20):
    return x / length(x, eps)


def look_at(campos, target):
    # campos: [N, 3], camera/eye position
    # target: [N, 3], object to look at
    # return: [N, 3, 3], rotation matrix

    forward_vector = safe_normalize(target - campos)
    up_vector = np.array([0, 0, 1], dtype=np.float32)
    right_vector = safe_normalize(np.cross(forward_vector, up_vector))
    up_vector = safe_normalize(np.cross(right_vector, forward_vector))

    R = np.stack([right_vector, up_vector, forward_vector], axis=1)
    return R


def interpolate_camera_path(poses, num_views, add_random_trans=False):
    positions = poses[:, :3, 3]  # shape: (N, 3)

    rotations = Rotation.from_matrix(poses[:, :3, :3])
    quats = rotations.as_quat()  # shape: (N, 4)

    t = np.linspace(0, 1, len(poses))
    t_new = np.linspace(0, 1, num_views)

    if len(poses) < 3:
        kind = "linear"
    elif len(poses) < 4:
        kind = "quadratic"
    else:
        kind = "cubic"
    pos_interpolator = interp1d(t, positions, axis=0, kind=kind)
    new_positions = pos_interpolator(t_new)

    key_rots = Rotation.from_quat(quats)
    slerp = Slerp(t, key_rots)
    new_rots = slerp(t_new)

    new_poses = np.zeros((num_views, 4, 4))
    new_poses[:, :3, :3] = new_rots.as_matrix()
    new_poses[:, :3, 3] = new_positions
    
    if add_random_trans:
        trans = np.random.uniform(0, 0.5)
        back_dir = new_poses[:, :3, 2]
        new_poses[:, :3, 3] = new_poses[:, :3, 3] + back_dir * trans
    
    new_poses[:, 3, 3] = 1.0

    return new_poses

def generate_random_perturbed_camera_poses(
    gs_cameras,
    visibility_grid,
    n_poses=10,
    position_std=0.05,
    rotation_std=0.03,
    width=512, height=512, fovy_deg=60, fovx_deg=None
):
    """
    Generate N camera poses with small perturbations around a given camera pose (NumPy version).
    
    Args:
        gs_cameras: gs camera views
        visibility_grid: visibility grid
        n_poses: Number of perturbed poses to generate
        position_std: Standard deviation for position perturbation (in same units as camera)
        rotation_std: Standard deviation for rotation perturbation (in radians)
        
    Returns:
        perturbed_poses: List of N perturbed camera poses
    """
    # Initialize lists to store perturbed poses
    perturbed_poses = []
    device = visibility_grid.device

    fovy = np.deg2rad(fovy_deg)
    fovx = fovy

    for gs_camera in gs_cameras:

        # Extract rotation and translation from original pose
        R = gs_camera.R             # c2w R
        temp_T = gs_camera.T        # w2c T
        T = -np.matmul(R, temp_T)   # c2w T
        
        # Convert rotation matrix to scipy rotation object
        rot = Rotation.from_matrix(R)
        
        for i in range(n_poses):
            # 1. Perturb camera position
            # Add Gaussian noise to position
            found = False
            max_try_times = 10
            while not found:
                pos_noise = np.random.normal(0, position_std, size=3)
                perturbed_T = T + pos_noise
                valid_mask = visibility_grid.check_valid_camera_center(torch.from_numpy(perturbed_T).unsqueeze(0).to(device))
                if valid_mask.sum() > 0:
                    found = True
                else:
                    max_try_times -= 1
                    if max_try_times <= 0:
                        break
            
            # 2. Perturb camera rotation
            # Create small random rotation using axis-angle representation
            random_axis = np.random.normal(0, 1, size=3)
            random_axis = random_axis / np.linalg.norm(random_axis)  # normalize to unit vector
            random_angle = np.random.normal(0, rotation_std)
            
            # Create small rotation
            small_rot = Rotation.from_rotvec(random_axis * random_angle)
            
            # Apply small rotation to original rotation
            perturbed_rot = small_rot * rot
            
            # Get rotation matrix
            perturbed_R = perturbed_rot.as_matrix()
            
            # Create perturbed camera pose
            perturbed_pose = np.eye(4)
            perturbed_pose[:3, :3] = perturbed_R
            perturbed_pose[:3, 3] = perturbed_T
            perturbed_poses.append(perturbed_pose.astype(np.float32))

    # generate camera
    cur_cams = []
    for idx in range(len(perturbed_poses)):
        cur_cam = MiniCam(perturbed_poses[idx], width, height, fovy=fovy, fovx=fovx)
        cur_cams.append(cur_cam)
    
    return perturbed_poses, cur_cams

def generate_perturbed_camera_poses(
    gs_camera,
    horizontal_angles=[-20, -10, 10, 20],         # Horizontal angles list (degrees)
    vertical_angles=[-20, -10, 10, 20],           # Vertical angles list (degrees)
    random_translation=True,                    # Whether to perturb translation
    width=512, height=512, fovy_deg=60
):
    """
    Generate a grid of camera poses with multiple angle variations in horizontal and vertical directions
    
    Args:
        gs_camera: Original camera
        horizontal_angles: List of horizontal angles (degrees), negative values for left, positive for right
        vertical_angles: List of vertical angles (degrees), negative values for up, positive for down
        random_translation: Whether to perturb translation
    
    Returns:
        perturbed_poses: List of all generated camera poses
        cur_cams: List of all generated camera objects
    """
    # Extract rotation and translation from original camera
    R = gs_camera.R             # c2w R
    temp_T = gs_camera.T        # w2c T
    T = -np.matmul(R, temp_T)   # c2w T

    # get fovy and fovx
    fovy = np.deg2rad(fovy_deg)
    fovx = fovy

    # Get the three axis directions of the camera coordinate system
    x_axis = R[:, 0]  # Camera's right direction
    y_axis = R[:, 1]  # Camera's up direction
    z_axis = R[:, 2]  # Camera's forward direction (actual direction is -z)
    
    # Initialize result lists
    perturbed_poses = []
    
    # Generate camera poses for each combination of horizontal and vertical angles
    for h_angle in horizontal_angles:
        for v_angle in vertical_angles:
            # Convert angles to radians
            h_rad = np.radians(h_angle + np.random.uniform(-1.5, 1.5))
            v_rad = np.radians(v_angle + np.random.uniform(-1.5, 1.5))
            
            # Create horizontal rotation (around y-axis)
            h_rotation = Rotation.from_rotvec(y_axis / np.linalg.norm(y_axis) * h_rad)
            
            # Create vertical rotation (around x-axis)
            v_rotation = Rotation.from_rotvec(x_axis / np.linalg.norm(x_axis) * v_rad)
            
            # Apply rotation to original camera rotation
            rot = Rotation.from_matrix(R)
            # First horizontal rotation, then vertical rotation
            perturbed_rot = v_rotation * h_rotation * rot
            perturbed_R = perturbed_rot.as_matrix()
            
            # Calculate translation
            perturbed_T = T.copy()
            
            if random_translation:
                # Add Gaussian noise to position
                pos_noise = np.random.normal(0, 0.1, size=3)
                
                # Apply translation
                perturbed_T = perturbed_T + pos_noise
            
            # Create camera pose matrix after rotation
            perturbed_pose = np.eye(4)
            perturbed_pose[:3, :3] = perturbed_R
            perturbed_pose[:3, 3] = perturbed_T
            perturbed_poses.append(perturbed_pose.astype(np.float32))
    
    # Generate camera objects
    cur_cams = []
    for idx in range(len(perturbed_poses)):
        cur_cam = MiniCam(perturbed_poses[idx], width, height, fovy, fovx)
        cur_cams.append(cur_cam)
    
    return perturbed_poses, cur_cams


def focus_point_fn(poses: np.ndarray) -> np.ndarray:
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt


def generate_ellipse_path(
    poses: np.ndarray,
    n_frames: int = 120,
    const_speed: bool = False,
    z_variation: float = 0.0,
    z_phase: float = 0.0,
    scale: float = 1.0,
) -> np.ndarray:
    """Generate an elliptical render path based on the given poses."""
    # Calculate the focal point for the path (cameras point toward this).
    center = focus_point_fn(poses)
    # Path height sits at z=0 (in middle of zero-mean capture pattern).
    offset = np.array([center[0], center[1], 0])
    # Calculate scaling for ellipse axes based on input camera positions.
    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)
    # Use ellipse that is symmetric about the focal point in xy.
    low = (-sc + offset) * 1.2
    high = (sc + offset) * 1.2
    # Optional height variation need not be symmetric
    # z_low = scale_z*np.percentile((poses[:, :3, 3]), 10, axis=0)
    # z_high = scale_z*np.percentile((poses[:, :3, 3]), 90, axis=0)
    z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
    z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)

    def get_positions(theta):
        # Interpolate between bounds with trig functions to get ellipse in x-y.
        # Optionally also interpolate in z to change camera height along path.
        return np.stack(
            [
                low[0] + (high - low)[0] * (np.cos(theta) * 0.5 + 0.5),
                low[1] + (high - low)[1] * (np.sin(theta) * 0.5 + 0.5),
                z_variation
                * (
                    z_low[2]
                    + (z_high - z_low)[2]
                    * (np.cos(theta + 2 * np.pi * z_phase) * 0.5 + 0.5)
                ),
            ],
            -1,
        )
    
    def viewmatrix(lookdir: np.ndarray, up: np.ndarray, position: np.ndarray) -> np.ndarray:
        """Construct lookat view matrix in COLMAP convention.

        Args:
            lookdir: Looking direction (will be aligned with +Z axis)
            up: Up direction (will be aligned close to +Y axis)
            position: Camera position
        Returns:
            4x4 view matrix where:
            - Z axis is the looking direction (forward)
            - Y axis is up
            - X axis is right
        """
        vec2 = safe_normalize(-lookdir)
        vec1 = safe_normalize(up)
        vec0 = safe_normalize(np.cross(vec1, vec2))
        vec1 = safe_normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, position], axis=1)
        return m

    theta = np.linspace(0, 2.0 * np.pi, n_frames + 1, endpoint=True)
    positions = get_positions(theta)

    # if const_speed:
    #     # Resample theta angles so that the velocity is closer to constant.
    #     lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
    #     theta = stepfun.sample(None, theta, np.log(lengths), n_frames + 1)
    #     positions = get_positions(theta)

    # Throw away duplicated last position.
    positions = positions[:-1]

    # Set path's up vector to axis closest to average of input pose up vectors.
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])
    # center = center + np.array([0, 0, -0.5])
    new_poses = np.stack([viewmatrix(p - center, up, p) for p in positions])
    angle_radians = np.radians(2 * scale)
    sign = np.random.choice([-1, 1])
    rotation_matrix = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle_radians), -np.sin(angle_radians)],
            [0, np.sin(angle_radians), np.cos(angle_radians)],
        ]
    )

    for i in range(new_poses.shape[0]):
        new_poses[i, :, :3] = np.matmul(
            new_poses[i, :, :3], rotation_matrix
        )  # np.dot(, )

    return new_poses

def generate_control_ellipse_path(
    center: np.ndarray,             # [x_center, y_center, z_center]
    low: np.ndarray,                # [x_low, y_low, z_low]
    high: np.ndarray,               # [x_high, y_high, z_high]
    rotation_angle: float = 0.0,    # in degree, rotation angle of the ellipse along z axis
    n_frames: int = 120,
    const_speed: bool = False,
    z_variation: float = 0.0,
    z_phase: float = 0.0,
    scale: float = 1.0,
) -> np.ndarray:
    """Generate an elliptical render path based on the given poses."""

    def get_positions(theta):
        # Interpolate between bounds with trig functions to get ellipse in x-y
        # Optionally also interpolate in z to change camera height along path
        positions = np.stack(
            [
                low[0] + (high[0] - low[0]) * (np.cos(theta) * 0.5 + 0.5),
                low[1] + (high[1] - low[1]) * (np.sin(theta) * 0.5 + 0.5),
                z_variation * (low[2] + (high[2] - low[2]) * (np.cos(theta + 2 * np.pi * z_phase) * 0.5 + 0.5)),
            ],
            -1,
        )
        
        if rotation_angle != 0:
            angle_rad = np.deg2rad(rotation_angle)
            rot_matrix = np.array([
                [np.cos(angle_rad), -np.sin(angle_rad), 0],
                [np.sin(angle_rad), np.cos(angle_rad), 0],
                [0, 0, 1]
            ])
            positions = np.dot(positions - center, rot_matrix.T) + center
            
        return positions

    def viewmatrix(lookdir: np.ndarray, up: np.ndarray, position: np.ndarray) -> np.ndarray:
        """Construct lookat view matrix."""
        vec2 = safe_normalize(-lookdir)
        vec1 = safe_normalize(up)
        vec0 = safe_normalize(np.cross(vec1, vec2))
        vec1 = safe_normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, position], axis=1)
        return m

    theta = np.linspace(0, 2.0 * np.pi, n_frames + 1, endpoint=True)
    positions = get_positions(theta)

    # if const_speed:
    #     # Resample theta angles so that the velocity is closer to constant.
    #     lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
    #     theta = stepfun.sample(None, theta, np.log(lengths), n_frames + 1)
    #     positions = get_positions(theta)

    positions = positions[:-1]

    # NOTE: hard code up vector
    up = np.array([0, 0, 1])
    new_poses = np.stack([viewmatrix(p - center, up, p) for p in positions])
    angle_radians = np.radians(2 * scale)
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(angle_radians), -np.sin(angle_radians)],
        [0, np.sin(angle_radians), np.cos(angle_radians)],
    ])

    for i in range(new_poses.shape[0]):
        new_poses[i, :, :3] = np.matmul(new_poses[i, :, :3], rotation_matrix)
    
    return new_poses

def generate_see3d_camera(input_c2ws, interpolate_num=10, camera_type='ellipse', ellipse_num=50, scale=5, width=512, height=512, fovy_deg=60, fovx_deg=None):
    """Generate a camera path for See3D dataset."""

    # get fovy and fovx
    fovy = np.deg2rad(fovy_deg)
    fovx = fovy if fovx_deg is None else np.deg2rad(fovx_deg)
    
    # generate novel c2w matrix
    z_variation = 1.5 - 0.20 * scale
    z_phase = np.random.random()
    poses = np.stack(input_c2ws, axis=0)

    if interpolate_num > 1:
        interpolate_poses = interpolate_camera_path(poses, interpolate_num)
    else:
        interpolate_poses = poses
    
    if camera_type == 'only_interpolate':
        random_poses = interpolate_poses
    elif camera_type == 'control_ellipse':
        random_poses = generate_control_ellipse_path(
            center=poses[:, :3, 3].mean(0),
            low=poses[:, :3, 3].min(0),
            high=poses[:, :3, 3].max(0),
        )

        homogeneous_row = np.zeros((len(random_poses), 1, 4))
        homogeneous_row[:, 0, 3] = 1
        random_poses = np.concatenate([random_poses, homogeneous_row], axis=1)          # c2w matrix

    elif camera_type == 'ellipse':
        random_poses = generate_ellipse_path(
            interpolate_poses[:, :3],
            ellipse_num,
            z_variation=z_variation,
            z_phase=z_phase,
            scale=scale,
        )

        homogeneous_row = np.zeros((len(random_poses), 1, 4))
        homogeneous_row[:, 0, 3] = 1
        random_poses = np.concatenate([random_poses, homogeneous_row], axis=1)          # c2w matrix

    else:
        raise ValueError(f'Invalid camera type: {camera_type}')

    # generate camera
    cur_cams = []
    for idx in range(len(random_poses)):
        c2w = random_poses[idx].astype(np.float32)
        cur_cam = MiniCam(c2w, width, height, fovy=fovy, fovx=fovx)
        cur_cams.append(cur_cam)

    return random_poses, cur_cams

def generate_interpolated_camera_poses(train_cams, visibility_grid, interpolate_num=10, width=512, height=512, fovy_deg=60, fovx_deg=None, device='cuda'):
    """Generate interpolated camera poses."""
    # get fovy and fovx
    fovy = np.deg2rad(fovy_deg)
    fovx = fovy if fovx_deg is None else np.deg2rad(fovx_deg)

    # get train c2w matrix
    train_w2cs = [train_cam.world_view_transform.transpose(0, 1) for train_cam in train_cams]
    train_w2cs = [train_w2c.cpu().numpy() for train_w2c in train_w2cs]
    train_c2ws = [np.linalg.inv(w2c) for w2c in train_w2cs]
    train_c2ws = np.stack(train_c2ws, axis=0)

    # get interpolated c2w matrix
    interpolated_c2ws = interpolate_camera_path(train_c2ws, interpolate_num)

    # check if interpolated camera center is valid
    interpolated_cam_centers = interpolated_c2ws[:, :3, 3]
    interpolated_cam_centers = torch.tensor(interpolated_cam_centers, dtype=torch.float32, device=device)
    valid_mask = visibility_grid.check_valid_camera_center(interpolated_cam_centers)
    valid_mask = valid_mask.cpu().numpy()
    interpolated_c2ws = interpolated_c2ws[valid_mask]

    # generate camera
    cur_cams = []
    for idx in range(len(interpolated_c2ws)):
        c2w = interpolated_c2ws[idx].astype(np.float32)
        cur_cam = MiniCam(c2w, width, height, fovy=fovy, fovx=fovx)
        cur_cams.append(cur_cam)
    
    return interpolated_c2ws, cur_cams

def generate_see3d_camera_by_lookat(train_cams, visibility_grid, train_depths, train_view_points, traj_center=None, n_frames=60, width=512, height=512, fovy_deg=60, fovx_deg=None):

    def viewmatrix(lookdir: np.ndarray, up: np.ndarray, position: np.ndarray) -> np.ndarray:
        """Construct lookat view matrix."""
        vec2 = safe_normalize(-lookdir)
        vec1 = safe_normalize(up)
        vec0 = safe_normalize(np.cross(vec1, vec2))
        vec1 = safe_normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, position], axis=1)
        return m

    # get fovy and fovx
    fovy = np.deg2rad(fovy_deg)
    fovx = fovy if fovx_deg is None else np.deg2rad(fovx_deg)

    device = train_depths.device

    train_cam_centers = torch.stack([cam.camera_center for cam in train_cams], dim=0)
    x_range = (train_cam_centers[:, 0].max() - train_cam_centers[:, 0].min()) / 2.0
    y_range = (train_cam_centers[:, 1].max() - train_cam_centers[:, 1].min()) / 2.0
    z_range = (train_cam_centers[:, 2].max() - train_cam_centers[:, 2].min()) / 2.0

    # get traj center
    if traj_center is None:
        traj_center = torch.mean(train_cam_centers, dim=0)

    # NOTE: hard code for range scale
    x_range_scale = [0.4, 0.9]
    y_range_scale = [0.4, 0.9]
    z_range_scale = [0.1, 0.3]

    # generate novel camera center
    theta = torch.linspace(0, 2.0 * torch.pi, n_frames + 1, device=device)
    novel_cam_centers = torch.stack([
        (torch.rand(n_frames+1, device=device) * (x_range_scale[1] - x_range_scale[0]) + x_range_scale[0]) * x_range * torch.cos(theta) + traj_center[0],
        (torch.rand(n_frames+1, device=device) * (y_range_scale[1] - y_range_scale[0]) + y_range_scale[0]) * y_range * torch.sin(theta) + traj_center[1],
        (torch.rand(n_frames+1, device=device) * (z_range_scale[1] - z_range_scale[0]) + z_range_scale[0]) * z_range * torch.cos(theta) + traj_center[2],
    ], dim=-1)
    novel_cam_centers = novel_cam_centers[:-1]          # Throw away duplicated last position.

    # check valid novel camera center
    novel_cam_centers = torch.tensor(novel_cam_centers, dtype=torch.float32, device=device)

    if visibility_grid is not None:
        valid_mask = visibility_grid.check_valid_camera_center(novel_cam_centers)
    else:
        valid_mask = torch.ones_like(novel_cam_centers[:, 0], dtype=torch.bool)

    novel_cam_centers = novel_cam_centers[valid_mask]

    # get lookat points
    lookat_points = get_novel_cams_lookat_points(train_cam_centers, train_view_points, novel_cam_centers)

    novel_cam_centers = novel_cam_centers.cpu().numpy()
    lookat_points = lookat_points.cpu().numpy()
    
    # NOTE: hard code up vector for colmap coords
    up = np.array([0, 0, -1])
    new_poses = np.stack([viewmatrix(p - lookat, up, p) for p, lookat in zip(novel_cam_centers, lookat_points)])

    homogeneous_row = np.zeros((len(new_poses), 1, 4))
    homogeneous_row[:, 0, 3] = 1
    new_poses = np.concatenate([new_poses, homogeneous_row], axis=1)

    # generate camera
    cur_cams = []
    for idx in range(len(new_poses)):
        c2w = new_poses[idx].astype(np.float32)
        cur_cam = MiniCam(c2w, width, height, fovy=fovy, fovx=fovx)
        cur_cams.append(cur_cam)

    return new_poses, cur_cams

def generate_see3d_camera_by_view_angle(train_cams, visibility_grid, traj_center=None, n_frames=60, width=512, height=512, fovy_deg=60, fovx_deg=None):
    """
    Generate see3d camera by view angle.
    """
    # get fovy and fovx
    fovy = np.deg2rad(fovy_deg)
    fovx = fovy if fovx_deg is None else np.deg2rad(fovx_deg)

    train_cam_centers = torch.stack([cam.camera_center for cam in train_cams], dim=0)
    x_range = (train_cam_centers[:, 0].max() - train_cam_centers[:, 0].min()) / 2.0
    y_range = (train_cam_centers[:, 1].max() - train_cam_centers[:, 1].min()) / 2.0
    z_range = (train_cam_centers[:, 2].max() - train_cam_centers[:, 2].min()) / 2.0

    # get traj center
    if traj_center is None:
        traj_center = torch.mean(train_cam_centers, dim=0)

    device = traj_center.device

    # NOTE: hard code for range scale
    x_range_scale = [0.4, 0.9]
    y_range_scale = [0.4, 0.9]
    z_range_scale = [0.1, 0.3]

    # generate novel camera center
    theta = torch.linspace(0, 2.0 * torch.pi, n_frames + 1, device=device)
    novel_cam_centers = torch.stack([
        (torch.rand(n_frames+1, device=device) * (x_range_scale[1] - x_range_scale[0]) + x_range_scale[0]) * x_range * torch.cos(theta) + traj_center[0],
        (torch.rand(n_frames+1, device=device) * (y_range_scale[1] - y_range_scale[0]) + y_range_scale[0]) * y_range * torch.sin(theta) + traj_center[1],
        (torch.rand(n_frames+1, device=device) * (z_range_scale[1] - z_range_scale[0]) + z_range_scale[0]) * z_range * torch.cos(theta) + traj_center[2],
    ], dim=-1)
    novel_cam_centers = novel_cam_centers[:-1]          # Throw away duplicated last position.

    # check valid novel camera center
    novel_cam_centers = torch.tensor(novel_cam_centers, dtype=torch.float32, device=device)

    if visibility_grid is not None:
        valid_mask = visibility_grid.check_valid_camera_center(novel_cam_centers)
    else:
        valid_mask = torch.ones_like(novel_cam_centers[:, 0], dtype=torch.bool)

    novel_cam_centers = novel_cam_centers[valid_mask]

    vec = traj_center - novel_cam_centers  # [N, 3]
    norm = torch.norm(vec, dim=-1, keepdim=True)
    vec_norm = vec / (norm + 1e-8)
    azimuths = torch.atan2(vec_norm[:, 1], vec_norm[:, 0])  # [N]
    elevations = torch.asin(vec_norm[:, 2])                 # [N]

    delta_azimuth_degs = torch.rand(novel_cam_centers.shape[0], device=device) * 20 - 10  # [-10, 10]
    delta_azimuths = torch.deg2rad(delta_azimuth_degs)
    azimuths_perturbeds = azimuths + delta_azimuths

    delta_elevation_degs = torch.rand(novel_cam_centers.shape[0], device=device) * 55 - 55  # [-55, 0]
    delta_elevations = torch.deg2rad(delta_elevation_degs)
    elevations_perturbeds = elevations + delta_elevations

    new_poses = []
    cur_cams = []
    assert width == height
    render_resolution = width
    for idx in range(len(novel_cam_centers)):
        
        cam_center = novel_cam_centers[idx]
        azimuth_perturbed = torch.rad2deg(azimuths_perturbeds[idx])
        elevation_perturbed = torch.rad2deg(elevations_perturbeds[idx])

        pose, cam = get_pose_and_cam(elevation_perturbed.cpu().numpy(), azimuth_perturbed.cpu().numpy(), fovx, fovy, cam_center.cpu().numpy(), render_resolution=render_resolution)
        new_poses.append(pose)
        cur_cams.append(cam)

    return new_poses, cur_cams

def generate_see3d_camera_by_lookat_none_vis_plane(train_cams, visibility_grid, none_vis_plane_points_dict, traj_center=None, width=512, height=512, fovy_deg=60, fovx_deg=None):
    """
    Generate see3d camera by lookat none vis plane points.
    """

    def viewmatrix(lookdir: np.ndarray, up: np.ndarray, position: np.ndarray) -> np.ndarray:
        """Construct lookat view matrix."""
        vec2 = safe_normalize(-lookdir)
        vec1 = safe_normalize(up)
        vec0 = safe_normalize(np.cross(vec1, vec2))
        vec1 = safe_normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, position], axis=1)
        return m

    # get fovy and fovx
    fovy = np.deg2rad(fovy_deg)
    fovx = fovy if fovx_deg is None else np.deg2rad(fovx_deg)

    train_cam_centers = torch.stack([cam.camera_center for cam in train_cams], dim=0)
    device = train_cam_centers.device
    traj_center = torch.mean(train_cam_centers, dim=0)

    # check traj_center is valid
    traj_center_valid_mask = visibility_grid.check_valid_camera_center(traj_center)
    if traj_center_valid_mask:
        cam_center = traj_center
        print(f"traj_center is valid, use it as cam_center")
    else:
        x_min, x_max = train_cam_centers[:, 0].min(), train_cam_centers[:, 0].max()
        y_min, y_max = train_cam_centers[:, 1].min(), train_cam_centers[:, 1].max()
        z_min, z_max = train_cam_centers[:, 2].min(), train_cam_centers[:, 2].max()
        
        voxel_res = 8
        x_coords = torch.linspace(x_min, x_max, voxel_res, device=device)
        y_coords = torch.linspace(y_min, y_max, voxel_res, device=device)
        z_coords = torch.linspace(z_min, z_max, voxel_res, device=device) 
        X, Y, Z = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        voxel_grid_points = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)  # [voxel_res^3, 3]
        
        valid_mask = visibility_grid.check_valid_camera_center(voxel_grid_points)
        valid_voxel_points = voxel_grid_points[valid_mask]
        distances = torch.norm(valid_voxel_points - traj_center.unsqueeze(0), dim=1)
        closest_idx = torch.argmin(distances)
        cam_center = valid_voxel_points[closest_idx]
        print(f"traj_center is not valid, use the closest voxel point as cam_center")

    cam_center = cam_center.cpu().numpy()
    lookat_points = []
    for plane_id, none_vis_plane_points in none_vis_plane_points_dict.items():
        # random choose one lookat point
        lookat_point = none_vis_plane_points[np.random.randint(0, none_vis_plane_points.shape[0])]
        lookat_points.append(lookat_point)

    # NOTE: hard code up vector for colmap coords
    up = np.array([0, 0, -1])
    new_poses = np.stack([viewmatrix(cam_center - lookat, up, cam_center) for lookat in lookat_points])

    homogeneous_row = np.zeros((len(new_poses), 1, 4))
    homogeneous_row[:, 0, 3] = 1
    new_poses = np.concatenate([new_poses, homogeneous_row], axis=1)

    # generate camera
    cur_cams = []
    for idx in range(len(new_poses)):
        c2w = new_poses[idx].astype(np.float32)
        cur_cam = MiniCam(c2w, width, height, fovy=fovy, fovx=fovx)
        cur_cams.append(cur_cam)

    return new_poses, cur_cams

def generate_see3d_camera_by_lookat_all_plane(train_cams, visibility_grid, plane_all_points_dict, traj_center=None, width=512, height=512, fovy_deg=60, fovx_deg=None):
    """
    Generate see3d camera by lookat all plane points.
    """

    def viewmatrix(lookdir: np.ndarray, up: np.ndarray, position: np.ndarray) -> np.ndarray:
        """Construct lookat view matrix."""
        vec2 = safe_normalize(-lookdir)
        vec1 = safe_normalize(up)
        vec0 = safe_normalize(np.cross(vec1, vec2))
        vec1 = safe_normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, position], axis=1)
        return m
    
    def get_point_to_plane_distance(points, plane_normal, plane_point):
        """
        Calculate distance from points to plane
        """
        # Calculate distance from points to plane: |(p - p0) · n|
        # where p is the point, p0 is a point on the plane, n is the plane normal vector
        point_to_plane = points - plane_point
        distances = np.abs(np.sum(point_to_plane * plane_normal, axis=1))
        return distances
    
    def find_optimal_camera_position(plane_points, plane_normal, visible_points, lookat_point):
        """Find optimal camera position that can see all plane points."""
        # Calculate plane bounding box
        min_coords = np.min(plane_points, axis=0)
        max_coords = np.max(plane_points, axis=0)
        plane_size = np.max(max_coords - min_coords)
        
        # Calculate camera distance to plane (based on plane size and FOV)
        distance_factor = plane_size / (2 * np.tan(fovx / 2))
        optimal_distance = distance_factor * 1.5  # Add some margin
        
        # Move optimal_distance along normal direction from lookat_point
        camera_direction = -plane_normal  # Camera looks at plane, so direction is opposite to normal

        # Find optimal camera position in visible_points
        # Calculate direction from each visible_point to lookat_point
        directions_to_lookat = visible_points - lookat_point
        directions_to_lookat = directions_to_lookat / (np.linalg.norm(directions_to_lookat, axis=1, keepdims=True) + 1e-6)
        
        # Calculate similarity between each direction and ideal camera direction (dot product)
        ideal_direction = camera_direction / np.linalg.norm(camera_direction)
        similarities = np.abs(np.dot(directions_to_lookat, ideal_direction))            # plane normal direction may be negative
        similarities_thresh = np.max(similarities) * 0.95
        
        # Select points with direction closest to ideal direction and appropriate distance
        # Distance should be within reasonable range (not too close or too far)
        distances_to_lookat = np.linalg.norm(visible_points - lookat_point, axis=1)
        distance_scores = np.exp(-np.abs(distances_to_lookat - optimal_distance) / optimal_distance)
        
        # Combined score: direction similarity + distance appropriateness
        combined_scores = similarities + distance_scores

        # Select best position from filtered points
        high_similarity_mask = similarities > similarities_thresh
        high_similarity_indices = np.nonzero(high_similarity_mask)[0]
        high_similarity_scores = combined_scores[high_similarity_mask]
        best_local_idx = np.argmax(high_similarity_scores)
        best_idx = high_similarity_indices[best_local_idx]
        
        return visible_points[best_idx]

    # get fovy and fovx
    fovy = np.deg2rad(fovy_deg)
    fovx = fovy if fovx_deg is None else np.deg2rad(fovx_deg)

    train_cam_centers = torch.stack([cam.camera_center for cam in train_cams], dim=0)
    x_range = (train_cam_centers[:, 0].max() - train_cam_centers[:, 0].min()) / 2.0
    y_range = (train_cam_centers[:, 1].max() - train_cam_centers[:, 1].min()) / 2.0
    z_range = (train_cam_centers[:, 2].max() - train_cam_centers[:, 2].min()) / 2.0

    # get traj center
    if traj_center is None:
        traj_center = torch.mean(train_cam_centers, dim=0).cpu().numpy()

    all_visible_pnts = visibility_grid.get_all_visible_pnts()
    all_visible_pnts = all_visible_pnts.detach().cpu().numpy()

    x_dis = abs(all_visible_pnts[:, 0] - traj_center[0])
    y_dis = abs(all_visible_pnts[:, 1] - traj_center[1])
    z_dis = abs(all_visible_pnts[:, 2] - traj_center[2])

    x_dis_valid_mask = x_dis < x_range.cpu().numpy()
    y_dis_valid_mask = y_dis < y_range.cpu().numpy()
    z_dis_valid_mask = z_dis < z_range.cpu().numpy()

    valid_novel_cam_centers = all_visible_pnts[x_dis_valid_mask & y_dis_valid_mask & z_dis_valid_mask]

    lookat_points = []
    novel_cam_centers = []
    for plane_id, plane_all_points in plane_all_points_dict.items():

        # get plane normal
        indices = np.random.choice(len(plane_all_points), 2, replace=False)
        sample_points = plane_all_points[indices]
        dir1 = sample_points[1] - sample_points[0]
        while True:
            # sample another point from plane points
            new_sample_point = plane_all_points[np.random.randint(0, len(plane_all_points))]
            dir2 = new_sample_point - sample_points[0]
            cos_angle = np.dot(dir1, dir2) / (np.linalg.norm(dir1) * np.linalg.norm(dir2) + 1e-6)
            if abs(cos_angle) < 0.95:        # avoid parallel
                break
        plane_normal = np.cross(dir1, dir2)
        plane_normal = plane_normal / np.linalg.norm(plane_normal)

        # 2. Select lookat point (plane center)
        lookat_point = np.mean(plane_all_points, axis=0)

        ref_distance = get_point_to_plane_distance(traj_center[None, :], plane_normal, lookat_point)
        all_cam_dis = get_point_to_plane_distance(valid_novel_cam_centers, plane_normal, lookat_point)
        dis_valid_mask = all_cam_dis < 1.2 * ref_distance[0]
        plane_valid_cam_centers = valid_novel_cam_centers[dis_valid_mask]
        
        # 3. Select camera center (choose optimal position from visible points)
        camera_center = find_optimal_camera_position(
            plane_all_points, plane_normal, plane_valid_cam_centers, lookat_point
        )

        lookat_points.append(lookat_point)
        novel_cam_centers.append(camera_center)

    # NOTE: hard code up vector for colmap coords
    up = np.array([0, 0, -1])
    new_poses = np.stack([viewmatrix(p - lookat, up, p) for p, lookat in zip(novel_cam_centers, lookat_points)])

    homogeneous_row = np.zeros((len(new_poses), 1, 4))
    homogeneous_row[:, 0, 3] = 1
    new_poses = np.concatenate([new_poses, homogeneous_row], axis=1)

    # generate camera
    cur_cams = []
    for idx in range(len(new_poses)):
        c2w = new_poses[idx].astype(np.float32)
        cur_cam = MiniCam(c2w, width, height, fovy=fovy, fovx=fovx)
        cur_cams.append(cur_cam)

    return new_poses, cur_cams

def select_need_inpaint_views(novel_cams, gs_none_visible_rate, gaussians, select_num=10, none_visible_rate_low_bound=0.05, none_visible_rate_high_bound=0.5, covisible_rate_high_bound=0.8):
    """
    Select views that need inpainting
    
    Args:
        novel_cams: list of GSCamera objects
        gs_none_visible_rate: list of float, none visible rate of each view
        gaussians: GaussianModel object
        select_num: int, number of views to select
        none_visible_rate_low_bound: float, lower bound of none visible rate
        none_visible_rate_high_bound: float, upper bound of none visible rate
        covisible_rate_high_bound: float, upper bound of co-visibility rate
        
    Returns:
        selected_view_ids: list of int, ids of selected views
    """

    # Create pairs of (view_id, none_visible_rate)
    view_rates = [(i, rate) for i, rate in enumerate(gs_none_visible_rate)]
    
    # Step 1: shuffle the view_rates
    random.shuffle(view_rates)
    
    # Step 2: Filter views within desired none_visible_rate range
    filtered_views = [(i, rate) for i, rate in view_rates 
                     if none_visible_rate_low_bound <= rate <= none_visible_rate_high_bound]
    
    # Step 3: Select views with low co-visibility
    selected_view_ids = []
    
    # If we have filtered views, select the first one
    if filtered_views:
        first_view_id = filtered_views[0][0]
        selected_view_ids.append(first_view_id)
    
    # Try to select remaining views from filtered views
    for view_id, _ in filtered_views:
        # Skip if this view is already selected
        if view_id in selected_view_ids:
            continue
        
        # Check co-visibility with all previously selected views
        is_covisible = False
        for selected_id in selected_view_ids:
            covisible_ratio = covisibility_check_by_gs(
                novel_cams[selected_id], novel_cams[view_id], gaussians
            )
            
            if covisible_ratio > covisible_rate_high_bound:
                is_covisible = True
                break
        
        # If this view has low co-visibility with all selected views, add it
        if not is_covisible:
            selected_view_ids.append(view_id)
            
        # Stop if we have enough views
        if len(selected_view_ids) >= select_num:
            break

    # Step 4: If we still don't have enough views, relax the constraints
    if len(selected_view_ids) < select_num:
        print(f"Only found {len(selected_view_ids)} views with optimal none_visible_rate. Relaxing constraints...")
        
        # First try views with none_visible_rate < lower bound
        low_rate_views = [(i, rate) for i, rate in view_rates 
                         if rate < none_visible_rate_low_bound and i not in selected_view_ids]
        
        for view_id, _ in low_rate_views:
            # Check co-visibility with all previously selected views
            is_covisible = False
            for selected_id in selected_view_ids:
                covisible_ratio = covisibility_check_by_gs(
                    novel_cams[selected_id], novel_cams[view_id], gaussians
                )
                
                if covisible_ratio > covisible_rate_high_bound:
                    is_covisible = True
                    break
            
            # If this view has low co-visibility with all selected views, add it
            if not is_covisible:
                selected_view_ids.append(view_id)
                
            # Stop if we have enough views
            if len(selected_view_ids) >= select_num:
                break

    # Step 5: If we still don't have enough views, just add any remaining views regardless of co-visibility
    if len(selected_view_ids) < select_num:
        print(f"Only found {len(selected_view_ids)} views with optimal none_visible_rate. Adding remaining views...")
        remaining_views = [i for i in range(len(novel_cams)) if i not in selected_view_ids and gs_none_visible_rate[i] <= none_visible_rate_high_bound]
        random.shuffle(remaining_views)
        selected_view_ids.extend(remaining_views[:select_num - len(selected_view_ids)])
    
    # print(f"Selected {len(selected_view_ids)} views for inpainting")
    return selected_view_ids

def generate_see3d_camera_by_lookat_object_centric(train_cams, visibility_grid, traj_center=None, n_frames=60, width=512, height=512, fovy_deg=60, fovx_deg=None):
    """
    Select views that need inpainting
    For object-centric scenes, e.g. Mip-NeRF 360, CO3D
    
    Args:
        train_cams: list of GSCamera objects
        visibility_grid: VisibilityGrid object
        traj_center: torch.Tensor, [3], center of the object
        n_frames: int, number of views to generate
        width: int, width of the image
        height: int, height of the image
        fovy_deg: float, field of view in degrees
        fovx_deg: float, field of view in degrees
        
    Returns:
        new_poses: torch.Tensor, [n_frames, 4, 4], new poses
        cur_cams: list of GSCamera objects, new cameras
    """

    def viewmatrix(lookdir: np.ndarray, up: np.ndarray, position: np.ndarray) -> np.ndarray:
        """Construct lookat view matrix."""
        vec2 = safe_normalize(-lookdir)
        vec1 = safe_normalize(up)
        vec0 = safe_normalize(np.cross(vec1, vec2))
        vec1 = safe_normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, position], axis=1)
        return m

    # get fovy and fovx
    fovy = np.deg2rad(fovy_deg)
    fovx = fovy if fovx_deg is None else np.deg2rad(fovx_deg)

    train_cam_centers = torch.stack([cam.camera_center for cam in train_cams], dim=0)
    x_range = (train_cam_centers[:, 0].max() - train_cam_centers[:, 0].min()) / 2.0
    y_range = (train_cam_centers[:, 1].max() - train_cam_centers[:, 1].min()) / 2.0
    z_range = (train_cam_centers[:, 2].max() - train_cam_centers[:, 2].min()) / 2.0

    device = train_cam_centers.device

    # get traj center
    if traj_center is None:
        traj_center = torch.mean(train_cam_centers, dim=0)
    
    # NOTE: hard code for range scale
    x_range_scale = [0.9, 1.1]
    y_range_scale = [0.9, 1.1]
    z_range_scale = [0.9, 1.1]

    # generate novel camera center
    theta = torch.linspace(0, 2.0 * torch.pi, n_frames + 1, device=device)
    novel_cam_centers = torch.stack([
        (torch.rand(n_frames+1, device=device) * (x_range_scale[1] - x_range_scale[0]) + x_range_scale[0]) * x_range * torch.cos(theta) + traj_center[0],
        (torch.rand(n_frames+1, device=device) * (y_range_scale[1] - y_range_scale[0]) + y_range_scale[0]) * y_range * torch.sin(theta) + traj_center[1],
        (torch.rand(n_frames+1, device=device) * (z_range_scale[1] - z_range_scale[0]) + z_range_scale[0]) * z_range * torch.cos(theta) + traj_center[2],
    ], dim=-1)
    novel_cam_centers = novel_cam_centers[:-1]          # Throw away duplicated last position.

    # NOTE: hard code to make the camera look at ground
    origin_train_cam_centers = train_cam_centers.clone()
    max_z = train_cam_centers[:, 2].max()
    novel_cam_centers[:, 2] = max_z
    
    if visibility_grid is not None:
        # check valid camera center
        valid_mask = visibility_grid.check_valid_camera_center(novel_cam_centers)
        if valid_mask.sum() == 0:
            print("No valid camera centers found. Using original camera centers.")
            novel_cam_centers = origin_train_cam_centers
            valid_mask = visibility_grid.check_valid_camera_center(novel_cam_centers)

            if valid_mask.sum() == 0:
                print("No valid camera centers found. Skip this stage.")
                return [], []
    else:
        valid_mask = torch.ones_like(novel_cam_centers[:, 0], dtype=torch.bool)

    novel_cam_centers = novel_cam_centers[valid_mask]

    # NOTE: hard code lookat points as traj_center
    lookat_points = torch.zeros_like(novel_cam_centers) + traj_center

    # NOTE: hard code to make the camera look at ground
    min_z = train_cam_centers[:, 2].min()
    lookat_points[:, 2] = min_z

    novel_cam_centers = novel_cam_centers.cpu().numpy()
    lookat_points = lookat_points.cpu().numpy()
    
    # NOTE: hard code up vector for colmap coords
    up = np.array([0, 0, -1])
    new_poses = np.stack([viewmatrix(p - lookat, up, p) for p, lookat in zip(novel_cam_centers, lookat_points)])

    homogeneous_row = np.zeros((len(new_poses), 1, 4))
    homogeneous_row[:, 0, 3] = 1
    new_poses = np.concatenate([new_poses, homogeneous_row], axis=1)

    # generate camera
    cur_cams = []
    for idx in range(len(new_poses)):
        c2w = new_poses[idx].astype(np.float32)
        cur_cam = MiniCam(c2w, width, height, fovy=fovy, fovx=fovx)
        cur_cams.append(cur_cam)

    return new_poses, cur_cams

# add random sample cameras in not covered area
def generate_random_sample_cameras(selected_cams, visibility_grid, train_cams, max_side_res=5, min_side_res=3, width=512, height=512, fovy_deg=60, fovx_deg=None):
    """
    Generate random sample cameras in not covered area.

    Args:
        selected_cams: List of selected camera objects
        visibility_grid: VisibilityGrid object
        train_cams: List of train view points
        max_side_res: Maximum grid resolution for the longer side
        min_side_res: Minimum grid resolution for the shorter side
        width, height: Image size for new cameras
        fovy_deg, fovx_deg: Field of view for new cameras

    Returns:
        new_poses: torch.Tensor, [n_frames, 4, 4], new poses
        cur_cams: list of GSCamera objects, new cameras
    """

    def viewmatrix(lookdir: np.ndarray, up: np.ndarray, position: np.ndarray) -> np.ndarray:
        """Construct lookat view matrix."""
        vec2 = safe_normalize(-lookdir)
        vec1 = safe_normalize(up)
        vec0 = safe_normalize(np.cross(vec1, vec2))
        vec1 = safe_normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, position], axis=1)
        return m

    # get fovy and fovx
    fovy = np.deg2rad(fovy_deg)
    fovx = fovy if fovx_deg is None else np.deg2rad(fovx_deg)

    x_min, y_min, _, x_max, y_max, _ = visibility_grid.get_visible_boundary()

    train_cam_centers = torch.stack([cam.camera_center for cam in train_cams], dim=0)
    device = train_cam_centers.device
    z_min, z_max = train_cam_centers[:, 2].min(), train_cam_centers[:, 2].max()

    x_min, y_min, z_min = x_min.item(), y_min.item(), z_min.item()
    x_max, y_max, z_max = x_max.item(), y_max.item(), z_max.item()

    traj_center = torch.tensor([(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2], device=device)

    x_range = x_max - x_min
    y_range = y_max - y_min

    # Determine grid resolution for x and y based on their range
    if x_range > y_range:
        x_res, y_res = max_side_res, min_side_res
    else:
        x_res, y_res = min_side_res, max_side_res
    z_res = 1  # Only divide in x and y, not z

    # Get selected camera centers
    selected_cam_centers = torch.stack([cam.camera_center for cam in selected_cams], dim=0)
    selected_cam_centers = selected_cam_centers.to(device)

    # Compute grid edges
    x_edges = torch.linspace(x_min, x_max, x_res + 1)
    y_edges = torch.linspace(y_min, y_max, y_res + 1)
    z_edges = torch.linspace(z_min, z_max, z_res + 1)

    new_cam_centers = []

    # Iterate over all grid cells
    for xi in range(x_res):
        for yi in range(y_res):
            for zi in range(z_res):
                # Compute the bounds of the current cell
                cell_min = torch.tensor([x_edges[xi], y_edges[yi], z_edges[zi]])
                cell_max = torch.tensor([x_edges[xi+1], y_edges[yi+1], z_edges[zi+1]])

                # Check if any selected camera center falls into this cell
                in_cell_mask = (
                    (selected_cam_centers[:, 0] >= cell_min[0]) & (selected_cam_centers[:, 0] < cell_max[0]) &
                    (selected_cam_centers[:, 1] >= cell_min[1]) & (selected_cam_centers[:, 1] < cell_max[1]) &
                    (selected_cam_centers[:, 2] >= cell_min[2]) & (selected_cam_centers[:, 2] < cell_max[2])
                )
                if in_cell_mask.any():
                    continue  # This cell is already covered

                ###### Find visible camera center in the cell
                # Collect all candidate centers in priority order
                candidate_centers = []
                
                # 1. Cell center (highest priority)
                cell_center = (cell_min + cell_max) / 2
                candidate_centers.append(cell_center)
                
                # 2. 8 subdivision centers (2x2x2)
                i, j, k = torch.meshgrid(torch.arange(2), torch.arange(2), torch.arange(2), indexing='ij')
                ijk_coords = torch.stack([i.flatten(), j.flatten(), k.flatten()], dim=1).float()  # [8, 3]
                sub_min_8 = cell_min + ijk_coords * (cell_max - cell_min) / 2
                sub_max_8 = cell_min + (ijk_coords + 1) * (cell_max - cell_min) / 2
                sub_centers_8 = (sub_min_8 + sub_max_8) / 2
                candidate_centers.extend(sub_centers_8)
                
                # 3. 27 subdivision centers (3x3x3)
                i, j, k = torch.meshgrid(torch.arange(3), torch.arange(3), torch.arange(3), indexing='ij')
                ijk_coords = torch.stack([i.flatten(), j.flatten(), k.flatten()], dim=1).float()  # [27, 3]
                sub_min_27 = cell_min + ijk_coords * (cell_max - cell_min) / 3
                sub_max_27 = cell_min + (ijk_coords + 1) * (cell_max - cell_min) / 3
                sub_centers_27 = (sub_min_27 + sub_max_27) / 2
                candidate_centers.extend(sub_centers_27)
                
                # Convert to tensor and check all candidates at once
                candidate_centers = torch.stack(candidate_centers).to(device)
                visibility_mask = visibility_grid.check_valid_camera_center(candidate_centers)

                # NOTE: check candidate centers not too close with traj_center
                dist = torch.norm(candidate_centers - traj_center, dim=-1)
                valid_mask = dist > 0.01
                visibility_mask = visibility_mask & valid_mask
                
                # Find the first visible center (highest priority)
                visible_indices = torch.nonzero(visibility_mask).squeeze(-1)
                if len(visible_indices) > 0:
                    first_visible_idx = visible_indices[0]
                    new_cam_centers.append(candidate_centers[first_visible_idx])

    if len(new_cam_centers) == 0:
        print("No new visible camera centers found in uncovered regions.")
        return None, None

    novel_cam_centers = torch.stack(new_cam_centers, dim=0)
    lookat_points = torch.zeros_like(novel_cam_centers) + traj_center

    novel_cam_centers = novel_cam_centers.cpu().numpy()
    lookat_points = lookat_points.cpu().numpy()

    # NOTE: hard code up vector for colmap coords
    up = np.array([0, 0, -1])
    new_poses = np.stack([viewmatrix(p - lookat, up, p) for p, lookat in zip(novel_cam_centers, lookat_points)])

    homogeneous_row = np.zeros((len(new_poses), 1, 4))
    homogeneous_row[:, 0, 3] = 1
    new_poses = np.concatenate([new_poses, homogeneous_row], axis=1)

    # generate camera
    cur_cams = []
    for idx in range(len(new_poses)):
        c2w = new_poses[idx].astype(np.float32)
        cur_cam = MiniCam(c2w, width, height, fovy=fovy, fovx=fovx)
        cur_cams.append(cur_cam)

    return new_poses, cur_cams

def generate_look_around_camera_poses(train_cams, visibility_grid, azimuth_bin=10, elevation_bin=5, width=512, height=512, fovy_deg=60, fovx_deg=None):
    """
    Generate look around camera poses.
    """

    # get fovy and fovx
    fovy = np.deg2rad(fovy_deg)
    fovx = fovy if fovx_deg is None else np.deg2rad(fovx_deg)

    train_cam_centers = torch.stack([cam.camera_center for cam in train_cams], dim=0)
    device = train_cam_centers.device
    traj_center = torch.mean(train_cam_centers, dim=0)

    # check traj_center is valid
    traj_center_valid_mask = visibility_grid.check_valid_camera_center(traj_center)
    if traj_center_valid_mask:
        cam_center = traj_center
        print(f"traj_center is valid, use it as cam_center")
    else:
        x_min, x_max = train_cam_centers[:, 0].min(), train_cam_centers[:, 0].max()
        y_min, y_max = train_cam_centers[:, 1].min(), train_cam_centers[:, 1].max()
        z_min, z_max = train_cam_centers[:, 2].min(), train_cam_centers[:, 2].max()
        
        voxel_res = 8
        x_coords = torch.linspace(x_min, x_max, voxel_res, device=device)
        y_coords = torch.linspace(y_min, y_max, voxel_res, device=device)
        z_coords = torch.linspace(z_min, z_max, voxel_res, device=device) 
        X, Y, Z = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        voxel_grid_points = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)  # [voxel_res^3, 3]
        
        valid_mask = visibility_grid.check_valid_camera_center(voxel_grid_points)
        valid_voxel_points = voxel_grid_points[valid_mask]
        distances = torch.norm(valid_voxel_points - traj_center.unsqueeze(0), dim=1)
        closest_idx = torch.argmin(distances)
        cam_center = valid_voxel_points[closest_idx]
        print(f"traj_center is not valid, use the closest voxel point as cam_center")

    # generate look around camera poses
    azimuth_list = np.linspace(-180, 180, azimuth_bin)
    elevation_list = np.linspace(-30, 5, elevation_bin)

    new_poses = []
    cur_cams = []
    assert width == height
    render_resolution = width
    for azimuth in azimuth_list:
        for elevation in elevation_list:
            pose, cam = get_pose_and_cam(elevation, azimuth, fovx=fovx, fovy=fovy, cam_center=cam_center.cpu().numpy(), render_resolution=render_resolution)
            new_poses.append(pose)
            cur_cams.append(cam)
    return new_poses, cur_cams

def get_pose_and_cam(elevation_deg, azimuth_deg, fovx, fovy, cam_center, radius=1, render_resolution=512):

    def viewmatrix(lookdir: np.ndarray, up: np.ndarray, position: np.ndarray) -> np.ndarray:
        """Construct lookat view matrix."""
        vec2 = safe_normalize(-lookdir)
        vec1 = safe_normalize(up)
        vec0 = safe_normalize(np.cross(vec1, vec2))
        vec1 = safe_normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, position], axis=1)
        return m

    elevation = np.deg2rad(elevation_deg)
    azimuth = np.deg2rad(azimuth_deg)

    x = radius * np.cos(elevation) * np.cos(azimuth)
    y = radius * np.cos(elevation) * np.sin(azimuth)
    z = radius * np.sin(elevation)
    lookat_point = np.array([x, y, z]) + cam_center

    # NOTE: hard code up vector for colmap coords
    up = np.array([0, 0, -1])
    pose_raw = viewmatrix(cam_center - lookat_point, up, cam_center)
    pose = np.eye(4).astype(np.float32)
    pose[:3, :] = pose_raw[:3, :]

    cam = MiniCam(pose, render_resolution, render_resolution, fovy=fovy, fovx=fovx)
    return pose, cam

def covisibility_check_by_gs(camera1, camera2, gaussians):
    """
    Determine co-visibility by checking the number of Gaussian points visible from both cameras
    
    Args:
        camera1, camera2: Two GSCamera objects
        gaussians: GaussianModel object
        
    Returns:
        bool: True if the ratio of shared visible points exceeds the threshold
        float: The maximum co-visibility ratio
    """
    # Get Gaussian point visibility from camera1
    visible_points1_mask = get_visible_points_mask(camera1, gaussians.get_xyz)
    
    # Get Gaussian point visibility from camera2
    visible_points2_mask = get_visible_points_mask(camera2, gaussians.get_xyz)
    
    # Calculate the number of points visible from both cameras
    common_visible = torch.logical_and(visible_points1_mask, visible_points2_mask).sum().item()
    
    # Calculate visibility ratios
    ratio1 = common_visible / visible_points1_mask.sum().item() if visible_points1_mask.sum().item() > 0 else 0
    ratio2 = common_visible / visible_points2_mask.sum().item() if visible_points2_mask.sum().item() > 0 else 0
    
    # Take the larger ratio, if it exceeds the threshold, consider the cameras to have co-visibility
    max_ratio = max(ratio1, ratio2)
    return max_ratio

def get_covisible_points(camera1, camera2, points):
    """
    Get points that are visible from both cameras
    
    Args:
        camera1, camera2: Two GSCamera objects
        points: torch.Tensor, [N, 3], points in world coordinate
        
    Returns:
        visible_points: torch.Tensor, [N], boolean mask indicating which points are visible from both cameras
    """

    visible_points1_mask = get_visible_points_mask(camera1, points)
    visible_points2_mask = get_visible_points_mask(camera2, points)

    visible_points_mask = torch.logical_and(visible_points1_mask, visible_points2_mask)
    visible_points = points[visible_points_mask]

    return visible_points

def project_points_to_image(camera, points):
    """
    Project points to image plane
    
    Args:
        camera: GSCamera object where camera.R is c2w rotation and camera.T is w2c translation
        points: torch.Tensor, [N, 3], points in world coordinate
        
    Returns:
        points_depth: torch.Tensor, [N], depth of points in camera coordinate
        points_2d: torch.Tensor, [N, 2], 2D coordinates of points in image plane
        in_image: torch.Tensor, [N], boolean mask indicating which points are within the field of view
    """
    # Get camera parameters (note the special convention)
    R_c2w = camera.R  # This is already camera-to-world rotation
    T_w2c = camera.T  # This is world-to-camera translation
    
    # Calculate world-to-camera rotation (transpose of camera-to-world rotation)
    R_w2c = R_c2w.T

    # Convert to tensors
    T_w2c = to_tensor_safe(T_w2c)
    R_w2c = to_tensor_safe(R_w2c)
    
    # Get camera frustum parameters
    image_height, image_width = camera.image_height, camera.image_width
    fx = image_width / (2 * np.tan(camera.FoVx / 2))
    fy = image_height / (2 * np.tan(camera.FoVy / 2))
    
    # Transform points to camera coordinate system
    # First apply world-to-camera rotation
    points_cam = torch.matmul(R_w2c, points.T).T
    # Then apply world-to-camera translation
    points_cam = points_cam + T_w2c
    
    # Calculate depth values (z-coordinate in camera space)
    points_depth = points_cam[:, 2]

    # Project to image plane
    points_2d = points_cam[:, :2] / points_cam[:, 2:3]
    points_2d[:, 0] = points_2d[:, 0] * fx + image_width / 2
    points_2d[:, 1] = points_2d[:, 1] * fy + image_height / 2

    # Check if points are within the field of view
    in_image = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < camera.image_width) & \
               (points_2d[:, 1] >= 0) & (points_2d[:, 1] < camera.image_height)

    return points_depth, points_2d, in_image

def get_pixel_to_points_tensor(camera, points, use_depth_threshold=True, depth_thresh = 0.01):
    """
    Get pixel-to-points mapping using tensor operations (more efficient version)
    
    Args:
        camera: GSCamera object
        points: torch.Tensor, [N, 3], points in world coordinate
        use_depth_threshold: bool, whether to use depth threshold to filter out points
        
    Returns:
        pixel_map: torch.Tensor, [H, W, max_points_per_pixel], point IDs stored at each pixel position
        valid_mask: torch.Tensor, [H, W, max_points_per_pixel], mask indicating valid point IDs
    """
    # Use existing function to get projection results
    points_depth, points_2d, in_image = project_points_to_image(camera, points)
    in_image = in_image & (points_depth > 0)
    
    H, W = camera.image_height, camera.image_width
    device = points.device
    
    valid_indices = torch.nonzero(in_image).squeeze(-1)
    valid_points_2d = points_2d[valid_indices]
    
    # Convert to integer pixel coordinates
    pixel_coords = torch.round(valid_points_2d).int()
    pixel_coords[:, 0] = torch.clamp(pixel_coords[:, 0], 0, W - 1)
    pixel_coords[:, 1] = torch.clamp(pixel_coords[:, 1], 0, H - 1)
    
    # Calculate flattened pixel indices
    pixel_indices = pixel_coords[:, 1] * W + pixel_coords[:, 0]
    
    # Sort by pixel indices to group points by pixel
    sorted_pixel_indices, sort_order = torch.sort(pixel_indices)
    sorted_valid_indices = valid_indices[sort_order]
    
    # Find unique pixel indices and their counts
    unique_pixels, pixel_counts = torch.unique_consecutive(sorted_pixel_indices, return_counts=True)
    max_points_per_pixel = min(pixel_counts.max().item(), 500)
    
    # Create output tensors
    pixel_map = torch.zeros((H, W, max_points_per_pixel), dtype=torch.int, device=device)
    
    # Calculate within-pixel indices for each point using cumsum
    # Create a mask for the start of each new pixel group
    is_new_pixel = torch.ones_like(sorted_pixel_indices, dtype=torch.bool)
    is_new_pixel[1:] = sorted_pixel_indices[1:] != sorted_pixel_indices[:-1]
    
    group_ids = torch.cumsum(is_new_pixel.int(), dim=0) - 1
    start_indices = torch.nonzero(is_new_pixel).squeeze(-1)
    group_start_positions = start_indices[group_ids]
    within_pixel_indices = torch.arange(len(sorted_pixel_indices), device=device) - group_start_positions

    # Convert flat pixel indices back to 2D coordinates
    pixel_v = sorted_pixel_indices // W
    pixel_u = sorted_pixel_indices % W
    
    # Filter out points that exceed max_points_per_pixel
    valid_fill_mask = within_pixel_indices < max_points_per_pixel
    if valid_fill_mask.any():
        final_v = pixel_v[valid_fill_mask]
        final_u = pixel_u[valid_fill_mask]
        final_within_indices = within_pixel_indices[valid_fill_mask]
        final_point_indices = sorted_valid_indices[valid_fill_mask]
        
        # Fill pixel_map and valid_mask in parallel
        pixel_map[final_v, final_u, final_within_indices] = final_point_indices.int()

    if use_depth_threshold:
        temp_points_depth = points_depth.clone()
        safe_max_points_depth = (points_depth[valid_indices].max())             # avoid nan in points_depth
        temp_points_depth[0] = safe_max_points_depth + 100.0                     # set the first point depth to a large value, as default point id is 0

        pixel_map_flatten = pixel_map.reshape(-1)
        pixel_pnts_depth_flatten = temp_points_depth[pixel_map_flatten]
        pixel_pnts_depth_map = pixel_pnts_depth_flatten.reshape(H, W, max_points_per_pixel)
        min_depth_map = pixel_pnts_depth_map.min(dim=-1).values                                 # [H, W]
        min_depth_map[min_depth_map > safe_max_points_depth] = safe_max_points_depth                # avoid only 0 point idx

        high_depth_map = min_depth_map + depth_thresh                                           # [H, W]
        # copy high_depth_map to [H, W, max_points_per_pixel]
        high_depth_map = high_depth_map.unsqueeze(-1).repeat(1, 1, max_points_per_pixel)

        # filter out points that are behind the high_depth_map
        valid_mask = pixel_pnts_depth_map < high_depth_map
        valid_pnts_sum = valid_mask.sum(dim=-1)
        max_valid_pnts_sum = valid_pnts_sum.max()
        refine_pixel_map = torch.zeros(H, W, max_valid_pnts_sum, dtype=torch.int, device=device)

        valid_coords = torch.nonzero(valid_mask)  # [num_valid_entries, 3] (h, w, point_idx)
        if len(valid_coords) > 0:
            h_coords = valid_coords[:, 0]
            w_coords = valid_coords[:, 1]
            point_slot_coords = valid_coords[:, 2]
            valid_point_indices = pixel_map[h_coords, w_coords, point_slot_coords]

            # get pos in refine_pixel_map
            pixel_indices = h_coords * W + w_coords
            sorted_indices = torch.argsort(pixel_indices)

            sorted_pixel_indices = pixel_indices[sorted_indices]
            sorted_h_coords = h_coords[sorted_indices]
            sorted_w_coords = w_coords[sorted_indices]
            sorted_point_indices = valid_point_indices[sorted_indices]

            is_new_pixel = torch.ones_like(sorted_pixel_indices, dtype=torch.bool)
            is_new_pixel[1:] = sorted_pixel_indices[1:] != sorted_pixel_indices[:-1]
            
            group_ids = torch.cumsum(is_new_pixel.int(), dim=0) - 1
            start_indices = torch.nonzero(is_new_pixel).squeeze(-1)
            group_start_positions = start_indices[group_ids]
            within_pixel_offsets = torch.arange(len(sorted_pixel_indices), device=device) - group_start_positions

            refine_pixel_map[sorted_h_coords, sorted_w_coords, within_pixel_offsets] = sorted_point_indices

            return refine_pixel_map

    return pixel_map

def get_visible_points_mask(camera, points):
    """
    Get points visible from a camera viewpoint
    
    Args:
        camera: GSCamera object where camera.R is c2w rotation and camera.T is w2c translation
        points: torch.Tensor, [N, 3], points in world coordinate
        
    Returns:
        torch.Tensor: Boolean mask indicating which points are visible
    """

    points_depth, _, in_image = project_points_to_image(camera, points)
    
    # Filter out points behind the camera
    front_mask = points_depth > 0
    
    # Visibility mask
    visible_mask = front_mask & in_image
    
    return visible_mask

def check_valid_camera_center_by_depth(train_cams, train_depths, novel_cam_centers):
    """
    Check if the novel camera center is visible from any of the training cameras
    
    Args:
        train_cams: List of GSCamera objects representing training cameras
        train_depths: List of torch.Tensor, [M, H, W], depths of points in camera coordinate
        novel_cam_centers: torch.Tensor, [N, 3], centers of novel cameras
    
    Returns:
        valid_mask: torch.Tensor, [N], boolean mask indicating which novel camera centers are visible
    """
    # Initialize valid mask (all False)
    valid_mask = torch.zeros(novel_cam_centers.shape[0], dtype=torch.bool, device=novel_cam_centers.device)

    for idx, train_cam in enumerate(train_cams):
        train_depth = train_depths[idx]  # [H, W]
        if isinstance(train_depth, np.ndarray):
            train_depth = torch.from_numpy(train_depth).cuda()
        
        # Project novel camera centers to this training camera's image plane
        points_depth, points_2d, in_image = project_points_to_image(train_cam, novel_cam_centers)
        
        # Get height and width of the training depth map
        H, W = train_depth.shape
        
        # Only process points that are within the image
        if not torch.any(in_image):
            continue
        
        # Get coordinates for points that are in the image
        valid_points_2d = points_2d[in_image]
        
        # Convert to integer coordinates and clamp to image boundaries
        u = torch.clamp(valid_points_2d[:, 0].long(), 0, W-1)
        v = torch.clamp(valid_points_2d[:, 1].long(), 0, H-1)
        
        # Get the depths of these valid points
        valid_points_depth = points_depth[in_image]
        
        # Get corresponding depths from the training depth map
        depth_at_pixels = train_depth[v, u]
        
        # Create masks for visibility conditions
        visible_points_mask = (valid_points_depth < depth_at_pixels) & (valid_points_depth > 0)
        
        # Map back to original indices and update valid_mask
        visible_indices = torch.nonzero(in_image).squeeze(-1)[visible_points_mask]
        valid_mask[visible_indices] = True
    
    return valid_mask

def build_visibility_masks(cameras, depths, points, mast3r_matching=None, depth_threshold=0.1, least_num_views=1, return_origin_masks=False):
    """
    Build visibility masks for each viewpoint camera
    
    Args:
        cameras: List of GSCamera objects representing cameras
        depths: List of torch.Tensor, [M, H, W], depths of points in camera coordinate
        points: torch.Tensor, [M, N, 3], points in world coordinate
        mast3r_matching: dict, {namei_namej: {i: [u1, v1], j: [u2, v2], ...}}, matching points between images
        depth_threshold: float, depth threshold for visibility check
        least_num_views: int, least number of views for visibility check
        return_origin_masks: bool, whether to return the original times visibility masks

    Returns:
        visibility_masks: List of torch.Tensor, [M, H, W], boolean mask indicating which viewpoint camera centers are visible
    """

    view_num = len(cameras)
    visibility_masks = []
    visibility_times_masks = []
    for i in range(view_num):
        name_i = cameras[i].image_name
        src_depth, src_pnts = depths[i], points[i]
        _, H, W = src_depth.shape
        src_visibility_mask = torch.zeros(H, W, device=src_depth.device)

        for j in range(view_num):
            if i == j:
                continue
            name_j = cameras[j].image_name
            tgt_cam, tgt_depth = cameras[j], depths[j].squeeze(0)

            # Project source points to target camera view
            tgt_pnts_depth, tgt_pnts_2d, tgt_pnts_in_image = project_points_to_image(tgt_cam, src_pnts)
            if not torch.any(tgt_pnts_in_image):
                continue

            # Create temporary visibility mask for this target view
            temp_visibility_mask = torch.zeros_like(src_visibility_mask)
            
            # Get valid points that are inside the target image
            valid_indices = torch.nonzero(tgt_pnts_in_image).squeeze(-1)
            valid_points_2d = tgt_pnts_2d[valid_indices]
            valid_points_depth = tgt_pnts_depth[valid_indices]
            
            # Convert to integer coordinates and clamp to image boundaries
            u = torch.clamp(valid_points_2d[:, 0].long(), 0, tgt_depth.shape[1]-1)
            v = torch.clamp(valid_points_2d[:, 1].long(), 0, tgt_depth.shape[0]-1)
            
            # Get corresponding depths from the target depth map
            depth_at_pixels = tgt_depth[v, u]
            
            # Check if depth difference is within threshold
            depth_diff = torch.abs(valid_points_depth - depth_at_pixels)
            relative_diff = depth_diff / (valid_points_depth + 1e-6)
            depth_valid = relative_diff < depth_threshold
            depth_valid = depth_valid & (valid_points_depth > 0)            # avoid negative depth
            
            # Map back to source coordinates (from flattened to 2D)
            src_indices = valid_indices[depth_valid]
            src_v = src_indices // W
            src_u = src_indices % W

            if mast3r_matching is not None:

                pair_name = f"{name_i}_{name_j}" if i<j else f"{name_j}_{name_i}"
                if pair_name in mast3r_matching:
                    if i < j:
                        kpts_i = mast3r_matching[pair_name]['kpts1']
                        kpts_j = mast3r_matching[pair_name]['kpts2']
                        assert name_i == mast3r_matching[pair_name]['img1']
                    else:
                        kpts_i = mast3r_matching[pair_name]['kpts2']
                        kpts_j = mast3r_matching[pair_name]['kpts1']
                        assert name_i == mast3r_matching[pair_name]['img2']

                    src_mast3r_u = [x[0] for x in kpts_i]
                    src_mast3r_v = [x[1] for x in kpts_i]
                    temp_visibility_mask[src_mast3r_v, src_mast3r_u] = 1.0              # use mast3r matching to update visibility mask

            else:
                # Update temporary visibility mask
                temp_visibility_mask[src_v, src_u] = 1.0
            
            # Add to source visibility mask
            src_visibility_mask += temp_visibility_mask
        
        # Check if the number of views is greater than or equal to least_num_views
        visibility_times_masks.append(src_visibility_mask.float().unsqueeze(0))
        src_visibility_mask = (src_visibility_mask >= least_num_views).float().unsqueeze(0)     # [1, H, W]
        visibility_masks.append(src_visibility_mask)

    if return_origin_masks:
        return visibility_times_masks
    else:
        return visibility_masks
    
def build_visibility_masks_2(tgt_cameras, tgt_depths, src_points, depth_threshold=0.1, least_num_views=1, return_origin_masks=False):
    """
    Build visibility masks for each viewpoint camera
    
    Args:
        tgt_cameras: List of GSCamera objects representing target cameras
        tgt_depths: List of torch.Tensor, [M, H, W], depths of points in camera coordinate
        src_points: torch.Tensor, [K, N, 3], points in world coordinate
        depth_threshold: float, depth threshold for visibility check
        least_num_views: int, least number of views for visibility check
        return_origin_masks: bool, whether to return the original times visibility masks

    Returns:
        visibility_masks: List of torch.Tensor, [M, H, W], boolean mask indicating which viewpoint camera centers are visible
    """

    src_view_num = src_points.shape[0]
    tgt_view_num = len(tgt_cameras)

    visibility_masks = []
    visibility_times_masks = []
    for i in range(tgt_view_num):
        tgt_cam, tgt_depth = tgt_cameras[i], tgt_depths[i].squeeze(0)
        H, W = tgt_depth.shape
        tgt_visibility_mask = torch.zeros(H, W, device=tgt_depth.device)
        
        for j in range(src_view_num):
            src_pnts = src_points[j]
            src_pnts_depth, src_pnts_2d, src_pnts_in_image = project_points_to_image(tgt_cam, src_pnts)
            if not torch.any(src_pnts_in_image):
                continue
            
            # Create temporary visibility mask for this target view
            temp_visibility_mask = torch.zeros_like(tgt_visibility_mask)

            # Get valid points that are inside the target image
            valid_indices = torch.nonzero(src_pnts_in_image).squeeze(-1)
            valid_points_2d = src_pnts_2d[valid_indices]
            valid_points_depth = src_pnts_depth[valid_indices]
            
            # Convert to integer coordinates and clamp to image boundaries
            u = torch.clamp(valid_points_2d[:, 0].long(), 0, W-1)
            v = torch.clamp(valid_points_2d[:, 1].long(), 0, H-1)
            
            # Get corresponding depths from the target depth map
            depth_at_pixels = tgt_depth[v, u]
            
            # Check if depth difference is within threshold
            depth_diff = torch.abs(valid_points_depth - depth_at_pixels)
            relative_diff = depth_diff / (valid_points_depth + 1e-6)
            depth_valid = relative_diff < depth_threshold
            depth_valid = depth_valid & (valid_points_depth > 0)  # Avoid negative depth
            
            # Update temporary visibility mask for valid depth points
            if torch.any(depth_valid):
                valid_u = u[depth_valid]
                valid_v = v[depth_valid]
                temp_visibility_mask[valid_v, valid_u] = 1.0
            
            # Add to target visibility mask
            tgt_visibility_mask += temp_visibility_mask
        
        # Store the visibility times mask (before thresholding)
        visibility_times_masks.append(tgt_visibility_mask.float().unsqueeze(0))
        
        # Apply threshold to create final visibility mask
        tgt_visibility_mask = (tgt_visibility_mask >= least_num_views).float().unsqueeze(0)  # [1, H, W]
        visibility_masks.append(tgt_visibility_mask)

    if return_origin_masks:
        return visibility_times_masks
    else:
        return visibility_masks

def farthest_point_sample(points, num_samples):
    """
    FPS sampling of points
    """
    num_points = points.shape[0]
    # If we have fewer points than requested samples, return all points
    if num_points <= num_samples:
        return points
        
    # Initialize with the first point
    selected_indices = torch.zeros(num_samples, dtype=torch.int, device=points.device)
    # Distances to the selected points
    distances = torch.ones(num_points, device=points.device) * 1e10
    
    # Randomly select the first point
    selected_indices[0] = torch.randint(0, num_points, (1,), device=points.device)
    
    # Iteratively select the farthest point
    for i in range(1, num_samples):
        # Last selected point
        last_idx = selected_indices[i-1]
        # Calculate distances to the last selected point
        dist = torch.sum((points - points[last_idx].unsqueeze(0)) ** 2, dim=1)
        # Update distances (minimum distance to any selected point)
        distances = torch.min(distances, dist)
        # Select the farthest point
        selected_indices[i] = torch.argmax(distances)
        
    # Return the sampled points
    return points[selected_indices]

def get_novel_cams_lookat_points(train_cam_centers, gs_train_view_points, novel_cam_centers, fps_num=10):
    """
    Get the lookat points of valid novel cameras
    
    Args:
        train_cam_centers: torch.Tensor, [M, 3], centers of training cameras
        gs_train_view_points: torch.Tensor, [M, N, 3], points of training cameras
        novel_cam_centers: torch.Tensor, [K, 3], centers of valid novel cameras
        fps_num: int, number of points to sample from training view points
    
    Returns:
        lookat_points: torch.Tensor, [K, 3], lookat points of valid novel cameras
    """

    device = novel_cam_centers.device

    # Calculate distances between novel camera centers and training camera centers
    distances = torch.cdist(novel_cam_centers, train_cam_centers)       # [K, M]

    # Find the closest training camera indices for each novel camera
    closest_train_indices = torch.argmin(distances, dim=1)

    # only sample once for each training camera
    train_fps_points = []
    for i in range(train_cam_centers.shape[0]):
        # Get points from the training camera
        points = gs_train_view_points[i]
        # Sample points from the training camera
        sampled_points = farthest_point_sample(points, fps_num)
        train_fps_points.append(sampled_points)

    train_fps_points = torch.stack(train_fps_points, dim=0)           # [M, fps_num, 3]

    # random choose one point from train_fps_points for each novel camera
    random_ids = torch.randint(0, fps_num, (novel_cam_centers.shape[0],), device=device)
    lookat_points = train_fps_points[closest_train_indices, random_ids]

    return lookat_points

class MiniCam:
    def __init__(self, c2w, width, height, fovy, fovx, znear=0.01, zfar=100):
        # c2w (pose) should be in NeRF convention.

        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar

        self.R = c2w[:3, :3]            # gs use c2w R
        w2c = np.linalg.inv(c2w)
        self.T = w2c[:3, 3]             # gs use w2c T

        self.world_view_transform = torch.tensor(w2c).transpose(0, 1).cuda()
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = torch.tensor(c2w[:3, 3]).cuda()                     # TODO: need check whether this is correct, camera center used to compute the SH

