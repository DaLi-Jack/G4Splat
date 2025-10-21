from typing import Union
import numpy as np
import torch
from torch import nn


def _preprocess_data(data:Union[np.ndarray, torch.Tensor]):
    if isinstance(data, list):
        if isinstance(data[0], torch.Tensor):
            new_data = torch.stack(data)
        else:
            new_data = np.array(data)
    else:
        new_data = data
    return new_data


def _process_data(data:Union[np.ndarray, torch.Tensor], convert_to_tensor, device='cpu'):    
    if convert_to_tensor and not isinstance(data, torch.Tensor):
        new_data = torch.tensor(data, device=device)
    elif isinstance(data, torch.Tensor):
        new_data = data.to(device)
    else:
        new_data = data
    return new_data


class PointMap(nn.Module):
    def __init__(
        self, 
        img_paths:list, 
        images:Union[list, np.ndarray, torch.Tensor],
        original_images:Union[list, np.ndarray, torch.Tensor],
        focals:Union[list, np.ndarray, torch.Tensor],
        poses:Union[list, np.ndarray, torch.Tensor],
        points3d:Union[list, np.ndarray, torch.Tensor],
        confidence:Union[list, np.ndarray, torch.Tensor],  # Optional
        masks:Union[list, np.ndarray, torch.Tensor],
        device='cpu',
    ):
        """The class is structured such that elements can stay on CPU and be moved to GPU when needed.
        Indeed, the scene could contain too many images and pointmaps to fit in GPU memory.
        However, if needed, the user can move the entire scene to GPU by providing tensors 
        or by calling the method move_everything_to_device.

        Args:
            img_paths (list): Has length n_images.
            images (Union[list, np.ndarray, torch.Tensor]): Has shape (n_images, height, width, 3).
            original_images (Union[list, np.ndarray, torch.Tensor]): Has shape (n_images, original_height, original_width, 3).
            focals (Union[list, np.ndarray, torch.Tensor]): Has shape (n_images, 1) or (n_images, 2).
            poses (Union[list, np.ndarray, torch.Tensor]): Has shape (n_images, 4, 4).
            points3d (Union[list, np.ndarray, torch.Tensor]): Has shape (n_images, height, width, 3).
            confidence (Union[list, np.ndarray, torch.Tensor]): Has shape (n_images, height, width).
            masks (Union[list, np.ndarray, torch.Tensor]): Has shape (n_images, height, width).
            device (str, optional): _description_. Defaults to 'cpu'.
        """
        super(PointMap, self).__init__()
        
        # TODO: Change _poses and _focals for camera objects with intrinsics. In a more general scenario, we also want to have principal points, etc.
        self._img_paths = _preprocess_data(img_paths)
        self._images = _preprocess_data(images)
        self._original_images = _preprocess_data(original_images)
        self._focals = _preprocess_data(focals)
        self._poses = _preprocess_data(poses)
        self._points3d = _preprocess_data(points3d)
        self._confidence = _preprocess_data(confidence)
        self._masks = _preprocess_data(masks)
        self.device = torch.device(device) if isinstance(device, str) else device
        self._convert_to_tensors = True
    
    def switch_conversion_to_tensors(self, activate:bool):
        self._convert_to_tensors = activate
        
    def move_everything_to_device(self, device):
        self.device = torch.device(device) if isinstance(device, str) else device
        self._images = _process_data(self._images, self._convert_to_tensors, self.device)
        self._original_images = _process_data(self._original_images, self._convert_to_tensors, self.device)
        self._focals = _process_data(self._focals, self._convert_to_tensors, self.device)
        self._poses = _process_data(self._poses, self._convert_to_tensors, self.device)
        self._points3d = _process_data(self._points3d, self._convert_to_tensors, self.device)
        self._confidence = _process_data(self._confidence, self._convert_to_tensors, self.device)
        self._masks = _process_data(self._masks, self._convert_to_tensors, self.device)
    
    @property
    def img_paths(self):
        return self._img_paths
    
    @property
    def images(self):
        return _process_data(self._images, self._convert_to_tensors, self.device)
    
    @property
    def original_images(self):
        return _process_data(self._original_images, self._convert_to_tensors, self.device)
    
    @property
    def focals(self):
        return _process_data(self._focals, self._convert_to_tensors, self.device)
    
    @property
    def poses(self):
        return _process_data(self._poses, self._convert_to_tensors, self.device)
    
    @property
    def points3d(self):
        return _process_data(self._points3d, self._convert_to_tensors, self.device)
    
    @property
    def confidence(self):
        return _process_data(self._confidence, self._convert_to_tensors, self.device)
    
    @property
    def masks(self):
        return _process_data(self._masks, self._convert_to_tensors, self.device)

    def upsample(self, scale_factor:int):
        # TODO: Implement a function to upsample the images, original_images, and points3d to higher resolution.
        # If camera objects are used, the intrinsics should be updated accordingly.
        raise NotImplementedError

    def forward(self, camera_idx):
        if isinstance(camera_idx, int):
            return {
                'img_path': self.img_paths[camera_idx],
                'image': self.images[camera_idx],
                'original_image': self.original_images[camera_idx],
                'focal': self.focals[camera_idx],
                'pose': self.poses[camera_idx],
                'points3d': self.points3d[camera_idx],
                'confidence': self.confidence[camera_idx],
                'masks': self.masks[camera_idx],
            }
        elif isinstance(camera_idx, list):
            return {
                'img_path': [self.img_paths[i] for i in camera_idx],
                'image': [self.images[i] for i in camera_idx],
                'original_image': [self.original_images[i] for i in camera_idx],
                'focal': [self.focals[i] for i in camera_idx],
                'pose': [self.poses[i] for i in camera_idx],
                'points3d': [self.points3d[i] for i in camera_idx],
                'confidence': [self.confidence[i] for i in camera_idx],
                'masks': [self.masks[i] for i in camera_idx],
            }
