import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
from matcha.dm_scene.cameras import GSCamera
from guidance.cam_utils import check_valid_camera_center_by_depth
import trimesh
from matcha.dm_scene.charts import depths_to_sample_points_parallel

class VisibilityGrid:
    """
    A 3D visibility grid that tracks which regions of space are visible from input views.
    
    The grid divides a 3D bounding box into voxels and marks each voxel as:
    - 1: visible (grid center is visible from at least one input camera)
    - 0: invisible (grid center is not visible from any input camera)
    """
    
    def __init__(
        self, 
        bbox_min: torch.Tensor,
        bbox_max: torch.Tensor, 
        resolution: int,
        input_cameras: List[GSCamera],
        input_depths: List[torch.Tensor],
        device: str = "cuda"
    ):
        """
        Initialize the visibility grid.
        
        Args:
            bbox_min: Minimum corner of 3D bounding box, shape (3,)
            bbox_max: Maximum corner of 3D bounding box, shape (3,)
            resolution: Grid resolution
            input_cameras: List of input view cameras (GSCamera objects)
            input_depths: List of depth maps corresponding to input cameras
            device: Device to run computations on
        """
        self.device = device
        self.bbox_min = bbox_min.to(device)
        self.bbox_max = bbox_max.to(device)
        self.resolution = resolution
        self.input_cameras = input_cameras
        self.input_depths = input_depths
        
        # Calculate grid properties
        self.grid_size = (self.bbox_max - self.bbox_min) / torch.tensor(resolution, device=device)
        self.min_grid_size = self.grid_size.min().item()
        
        # Initialize visibility grid, default to invisible (0)
        self.visibility_grid = torch.zeros((resolution, resolution, resolution), dtype=torch.float32, device=device)
        
        # Build the grid
        self._build_grid()
    
    def _build_grid(self):
        """Build the initial visibility grid based on input camera visibility."""
        print(f"Building visibility grid with resolution {self.resolution}")
        
        # Generate all grid center points
        nx, ny, nz = self.resolution, self.resolution, self.resolution
        
        # Use meshgrid to generate all grid indices efficiently
        x_indices = torch.arange(nx, device=self.device)
        y_indices = torch.arange(ny, device=self.device)
        z_indices = torch.arange(nz, device=self.device)
        
        # Create meshgrid for all grid positions
        X, Y, Z = torch.meshgrid(x_indices, y_indices, z_indices, indexing='ij')
        
        # Convert indices to world coordinates (grid centers)
        grid_centers = torch.stack([
            self.bbox_min[0] + (X + 0.5) * self.grid_size[0],
            self.bbox_min[1] + (Y + 0.5) * self.grid_size[1], 
            self.bbox_min[2] + (Z + 0.5) * self.grid_size[2]
        ], dim=-1)  # Shape: (nx, ny, nz, 3)
        
        # Flatten for batch processing
        grid_centers_flat = grid_centers.reshape(-1, 3)  # Shape: (nx*ny*nz, 3)
        
        print(f"Checking visibility for {grid_centers_flat.shape[0]} grid points")
        
        # Check visibility using the provided function
        valid_mask = check_valid_camera_center_by_depth(
            self.input_cameras, 
            self.input_depths, 
            grid_centers_flat
        )
        
        # Reshape back to grid shape and set visibility values
        valid_mask_grid = valid_mask.reshape(self.resolution, self.resolution, self.resolution)
        
        # Set visibility: 1 for visible, 0 for invisible
        self.visibility_grid[valid_mask_grid] = 1.0
        self.visibility_grid[~valid_mask_grid] = 0.0
        
        visible_count = valid_mask.sum().item()
        total_count = grid_centers_flat.shape[0]
        print(f"Grid initialized: {visible_count}/{total_count} ({100*visible_count/total_count:.1f}%) voxels are visible")
    
    def _world_to_grid_indices(self, points: torch.Tensor) -> torch.Tensor:
        """
        Convert world coordinates to grid indices.
        
        Args:
            points: World coordinates, shape (..., 3)
            
        Returns:
            Grid indices, shape (..., 3), values in [0, resolution-1]
        """
        # Normalize to [0, 1] within bbox
        normalized = (points - self.bbox_min) / (self.bbox_max - self.bbox_min)
        
        # Convert to grid indices
        indices = normalized * torch.tensor(self.resolution, device=self.device)
        
        # Clamp to valid range
        indices = torch.clamp(indices, 0, torch.tensor(self.resolution, device=self.device) - 1)
        
        return indices.int()
    
    def _sample_visibility_at_points(self, points: torch.Tensor, max_batch_point_num: int = 100000) -> torch.Tensor:
        """
        Sample visibility values at given 3D points using nearest neighbor interpolation.
        
        Args:
            points: World coordinates, shape (..., 3)
            max_batch_point_num: Maximum number of points to process in a single batch
            
        Returns:
            Visibility values, shape (...,)
        """
        original_shape = points.shape[:-1]
        points_flat = points.reshape(-1, 3)
        
        # If number of points is less than batch size, process directly
        if points_flat.shape[0] <= max_batch_point_num:
            # Convert to grid indices
            grid_indices = self._world_to_grid_indices(points_flat)  # Shape: (N, 3)
            
            # Sample from visibility grid
            visibility_values = self.visibility_grid[
                grid_indices[:, 0], 
                grid_indices[:, 1], 
                grid_indices[:, 2]
            ]
        else:
            # Process in batches
            visibility_values_list = []
            num_points = points_flat.shape[0]
            num_batches = (num_points + max_batch_point_num - 1) // max_batch_point_num
            
            for i in range(num_batches):
                start_idx = i * max_batch_point_num
                end_idx = min((i + 1) * max_batch_point_num, num_points)
                
                # Get current batch points
                batch_points = points_flat[start_idx:end_idx]
                
                # Convert to grid indices
                grid_indices = self._world_to_grid_indices(batch_points)  # Shape: (batch_size, 3)
                
                # Sample from visibility grid
                batch_visibility_values = self.visibility_grid[
                    grid_indices[:, 0], 
                    grid_indices[:, 1], 
                    grid_indices[:, 2]
                ]
                
                visibility_values_list.append(batch_visibility_values)
            
            # Concatenate results from all batches
            visibility_values = torch.cat(visibility_values_list, dim=0)
        
        return visibility_values.reshape(original_shape)

    def check_valid_camera_center(self, cam_centers: torch.Tensor) -> torch.Tensor:
        """
        Check camera center is visible or not.

        Args:
            cam_centers: World coordinates, shape (N, 3)
            
        Returns:
            valid_mask: 1 -> visible, 0 -> invisible, shape (N,)
        """
        valid_mask = self._sample_visibility_at_points(cam_centers)
        valid_mask = valid_mask > 0.5

        return valid_mask

    def render_visibility_map(
        self, 
        novel_cameras: List[GSCamera], 
        novel_depths: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Render visibility maps for novel cameras.
        
        Args:
            novel_cameras: List of novel view cameras
            novel_depths: List of depth maps for novel cameras
            
        Returns:
            List of visibility maps, each with shape (H, W). 1 for visible, 0 for occluded
        """
        visibility_maps = []
        
        for cam_idx, (camera, depth_map) in enumerate(zip(novel_cameras, novel_depths)):
            print(f"Rendering visibility map for camera {cam_idx+1}/{len(novel_cameras)}")
            
            # Ensure depth map is on correct device
            if isinstance(depth_map, np.ndarray):
                depth_map = torch.from_numpy(depth_map).to(self.device)
            else:
                depth_map = depth_map.to(self.device)

            H, W = depth_map.shape

            invalid_depth_mask = depth_map <= 1e-6
            depth_map[invalid_depth_mask] = 1e-3  # NOTE: set invalid depth to 1e-3, avoid error when sampling

            # Get points in depth map
            max_samples = int(depth_map.max().item() / self.min_grid_size) + 1
            sample_points = depths_to_sample_points_parallel(depth_map.unsqueeze(0), max_samples, [camera])  # (1, H * W, max_samples, 3)
            sample_points = sample_points[:, :, :-10, :]  # NOTE: delete last 10 points, avoid being too close to surface boundary
            max_samples = sample_points.shape[2]
            sample_points_flat = sample_points.reshape(-1, 3)

            # Sample visibility values
            visibility_values_flat = self._sample_visibility_at_points(sample_points_flat)
            visibility_values = visibility_values_flat.reshape(H, W, max_samples)

            # Check if any point along each ray has visibility < 0.5 (i.e., invisible)
            occlusion_map = (visibility_values < 0.5).any(dim=-1).float()

            # Handle invalid depths, set them as occluded (1)
            occlusion_map[invalid_depth_mask] = 1.0
            visibility_map = 1 - occlusion_map
            
            visibility_maps.append(visibility_map)
        
        return visibility_maps
    
    def get_visible_boundary(self):
        """
        Get visible boundary of the grid.
        """
        # Get grid dimensions
        nx, ny, nz = self.resolution, self.resolution, self.resolution
        
        # Find visible voxels (value > 0.5)
        visible_mask = self.visibility_grid > 0.5
        
        if not visible_mask.any():
            print("No invisible voxels found.")
            return
        
        # Generate grid center coordinates for all voxels
        x_indices = torch.arange(nx, device=self.device)
        y_indices = torch.arange(ny, device=self.device)
        z_indices = torch.arange(nz, device=self.device)
        
        # Create meshgrid for all grid positions
        X, Y, Z = torch.meshgrid(x_indices, y_indices, z_indices, indexing='ij')
        
        # Convert indices to world coordinates (grid centers)
        grid_centers = torch.stack([
            self.bbox_min[0] + (X + 0.5) * self.grid_size[0],
            self.bbox_min[1] + (Y + 0.5) * self.grid_size[1], 
            self.bbox_min[2] + (Z + 0.5) * self.grid_size[2]
        ], dim=-1)  # Shape: (nx, ny, nz, 3)

        visible_points = grid_centers[visible_mask]
        x_min, y_min, z_min = visible_points[:, 0].min(), visible_points[:, 1].min(), visible_points[:, 2].min()
        x_max, y_max, z_max = visible_points[:, 0].max(), visible_points[:, 1].max(), visible_points[:, 2].max()

        return x_min, y_min, z_min, x_max, y_max, z_max
    
    def get_all_visible_pnts(self):
        """
        Get all visible points of the grid.
        """
        # Get grid dimensions
        nx, ny, nz = self.resolution, self.resolution, self.resolution
        
        # Find visible voxels (value > 0.5)
        visible_mask = self.visibility_grid > 0.5
        
        if not visible_mask.any():
            print("No invisible voxels found.")
            return
        
        # Generate grid center coordinates for all voxels
        x_indices = torch.arange(nx, device=self.device)
        y_indices = torch.arange(ny, device=self.device)
        z_indices = torch.arange(nz, device=self.device)
        
        # Create meshgrid for all grid positions
        X, Y, Z = torch.meshgrid(x_indices, y_indices, z_indices, indexing='ij')
        
        # Convert indices to world coordinates (grid centers)
        grid_centers = torch.stack([
            self.bbox_min[0] + (X + 0.5) * self.grid_size[0],
            self.bbox_min[1] + (Y + 0.5) * self.grid_size[1], 
            self.bbox_min[2] + (Z + 0.5) * self.grid_size[2]
        ], dim=-1)  # Shape: (nx, ny, nz, 3)

        visible_points = grid_centers[visible_mask]

        return visible_points

    def vis_invisible_pnts(self, save_path: str):
        """
        Visualize invisible grid centers as points and save to file.
        
        Args:
            save_path: Path to save the point cloud file (should end with .ply)
        """
        import os
        
        # Get grid dimensions
        nx, ny, nz = self.resolution, self.resolution, self.resolution
        
        # Find invisible voxels (value < 0.5)
        invisible_mask = self.visibility_grid < 0.5
        
        if not invisible_mask.any():
            print("No invisible voxels found.")
            return
        
        # Generate grid center coordinates for all voxels
        x_indices = torch.arange(nx, device=self.device)
        y_indices = torch.arange(ny, device=self.device)
        z_indices = torch.arange(nz, device=self.device)
        
        # Create meshgrid for all grid positions
        X, Y, Z = torch.meshgrid(x_indices, y_indices, z_indices, indexing='ij')
        
        # Convert indices to world coordinates (grid centers)
        grid_centers = torch.stack([
            self.bbox_min[0] + (X + 0.5) * self.grid_size[0],
            self.bbox_min[1] + (Y + 0.5) * self.grid_size[1], 
            self.bbox_min[2] + (Z + 0.5) * self.grid_size[2]
        ], dim=-1)  # Shape: (nx, ny, nz, 3)
        
        # Extract invisible points
        invisible_points = grid_centers[invisible_mask]  # Shape: (N, 3)
        
        # Convert to numpy for saving
        points_np = invisible_points.detach().cpu().numpy()
        pm = trimesh.PointCloud(points_np)
        pm.export(save_path)
        print(f"Invisible points saved to {save_path}")
