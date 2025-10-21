import os
import sys
sys.path.append(os.getcwd())

from dataclasses import dataclass, field
from typing import Type, Union
from pathlib import Path
from easydict import EasyDict as edict

import gc
import torch
import numpy as np
import cv2
from sklearn.cluster import KMeans
import argparse
from PIL import Image

from mask_generator import setup_sam, infer_masks
from tools import to_world_space, remove_small_isolated_areas, merge_normal_clusters
sys.path.append(os.path.join(os.getcwd(), '2d-gaussian-splatting'))
from utils.general_utils import seed_everything

def normals_cluster(normals: np.ndarray, img_shape: tuple, n_init_clusters: int = 8, n_clusters: int = 6, min_size_ratio: float = 0.004):
    """
    Cluster the surface normals.
    
    Args:
        normals: Normal vectors array (H*W, 3) or (H, W, 3)
        img_shape: Image shape (H, W)
        n_init_clusters: Initial number of clusters for KMeans
        n_clusters: Number of clusters to keep after filtering
        min_size_ratio: Minimum size ratio for valid clusters
    
    Returns:
        normal_masks: List of 2D cluster masks
    """
    # Ensure normals are in correct shape (N, 3)
    if len(normals.shape) == 3:
        normals_flat = normals.reshape(-1, 3)
    else:
        normals_flat = normals
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=n_init_clusters, random_state=0, n_init=1).fit(normals_flat)
    pred = kmeans.labels_
    centers = kmeans.cluster_centers_

    # Select the first `n_clusters` clusters c1, c2, ..., where Size(c1) > Size(c2) > ... (sorted by size)
    count_values = np.bincount(pred)
    topk = np.argpartition(count_values, -n_clusters)[-n_clusters:]
    sorted_topk_idx = np.argsort(count_values[topk])
    sorted_topk = topk[sorted_topk_idx][::-1]

    pred, sorted_topk, num_clusters = merge_normal_clusters(pred, sorted_topk, centers)

    min_plane_size = img_shape[0] * img_shape[1] * min_size_ratio
    
    count_valid_cluster = 0
    normal_masks = []
    for i in range(num_clusters):
        mask = (pred == sorted_topk[count_valid_cluster])
        mask_clean = remove_small_isolated_areas((mask > 0).reshape(*img_shape) * 255, min_size=min_plane_size).reshape(-1)
        mask[mask_clean == 0] = 0

        num_labels, labels = cv2.connectedComponents((mask * 255).reshape(img_shape).astype(np.uint8))
        for label in range(1, num_labels):
            normal_masks.append(labels == label)
        count_valid_cluster += 1
    return normal_masks

@dataclass
class PlaneExcavatorConfig():

    min_size_ratio: float = 0.004 # 0.4%
    """The minimum size of a desired plane segment, as a ratio of the total number of pixels in the image."""
    n_init_normal_clusters: int = 8
    """The number of clusters to form as well as the number of centroids to generate when use KMeans to cluster the surface normals."""
    n_normal_clusters: int = 6
    """The number of normal clusters to keep after KMeans, i.e., we only keep the first `num_max_clusters` clusters c1, c2, ..., where Size(c1) > Size(c2) > ... (sorted by size).
    """
    num_sam_prompts: int = 256
    """The number of SAM prompts to use for inference."""

class PlaneExcavator:
    def __init__(self, config: PlaneExcavatorConfig, device, img_height: int, img_width: int, use_normal_estimator: bool = False, normal_model_type: str = 'stablenormal'):
        self.n_init_normal_clusters = config.n_init_normal_clusters
        self.n_normal_clusters = config.n_normal_clusters
        self.num_sam_prompts = config.num_sam_prompts
        self.img_shape = (img_height, img_width)  # Currently only support images of the same size
        self.min_plane_size = self.img_shape[0] * self.img_shape[1] * config.min_size_ratio
        self.device = device

        self.sam_model = setup_sam(device=self.device)

        self.use_normal_estimator = use_normal_estimator
        if self.use_normal_estimator:
            if normal_model_type == 'stablenormal':
                self.normal_estimator = torch.hub.load("Stable-X/StableNormal", "StableNormal", trust_repo=True)
            # elif normal_model_type == 'snu':
            #     model_configs = edict(
            #         {
            #             "architecture": "BN",
            #             "pretrained": self.config.pretrain_dataset,
            #             "sampling_ratio": 0.4,
            #             "importance_ratio": 0.7,
            #             "input_height": self.kwargs.get("img_height", 480),
            #             "input_width": self.kwargs.get("img_width", 640),
            #         }
            #     )
            #     self.model = snu.models.snu_NNET(model_configs).to(self.device)
            else:
                raise ValueError(f'normal_model_type {normal_model_type} is not supported')

    def _normals_cluster(self, normals: np.ndarray):
        """
        Cluster the surface normals.
        """
        kmeans = KMeans(n_clusters=self.n_init_normal_clusters, random_state=0, n_init=1).fit(normals.reshape(-1, 3))
        pred = kmeans.labels_
        centers = kmeans.cluster_centers_

        # Select the first `num_max_clusters` clusters c1, c2, ..., where Size(c1) > Size(c2) > ... (sorted by size)
        count_values = np.bincount(pred)
        topk = np.argpartition(count_values,-self.n_normal_clusters)[-self.n_normal_clusters:]
        sorted_topk_idx = np.argsort(count_values[topk])
        sorted_topk = topk[sorted_topk_idx][::-1]

        pred, sorted_topk, num_clusters = merge_normal_clusters(pred, sorted_topk, centers)

        count_valid_cluster = 0
        normal_masks = []
        for i in range(num_clusters):
            mask = (pred==sorted_topk[count_valid_cluster])
            mask_clean = remove_small_isolated_areas((mask>0).reshape(*self.img_shape)*255, min_size=self.min_plane_size).reshape(-1)
            mask[mask_clean==0] = 0

            num_labels, labels = cv2.connectedComponents((mask*255).reshape(self.img_shape).astype(np.uint8))
            for label in range(1, num_labels):
                normal_masks.append(labels == label)
            count_valid_cluster += 1
        return normal_masks

    def __call__(self, img: np.ndarray, normals: np.ndarray = None, vis: bool = False):
        """
        img: np.ndarray, shape: (H, W, 3)
            The input image, in the form of a numpy array.
        normals: np.ndarray, shape: (H, W, 3)
            The surface normals, in the form of a numpy array.
        """

        assert normals is not None or self.use_normal_estimator, "normals must be provided if use_normal_estimator is False"

        if normals is None:
            normals = self.normal_estimator(Image.fromarray(img))
            normals = np.array(normals)         # 0-255
            normals = normals / 255.0            # 0-1
            normals = (0.5 - normals) * 2       # -1-1

        normal_clusters = self._normals_cluster(normals)

        # Generate masks
        normalized_prompts = torch.rand(self.num_sam_prompts, 2, device=self.device) * 2 - 1
        sam_outputs = infer_masks(self.sam_model, img, keypoints=normalized_prompts, device=self.device, num_pts_active=0)['masks']
        masks = sam_outputs['masks'].cpu().numpy()
        masks = sorted(masks, key=lambda x: np.sum(x))

        seg_mask = np.zeros(self.img_shape, dtype=np.uint8)  # 0 indicates background (non-plane region)
        count = 0
        for mask in masks:
            for normal_mask in normal_clusters:
                intersect = mask & normal_mask
                size = np.sum(intersect)
                if size < self.min_plane_size:
                    continue
                count += 1
                seg_mask[intersect] = count

        new_seg_mask =np.zeros_like(seg_mask)
        masks_avg_normals = []
        masks_areas = []
        plane_instances = []

        new_count = 0
        for i in range(np.min([100, count])):
            mask = (seg_mask == i + 1)
            area = mask.sum()
            if area < self.min_plane_size:
                continue
            new_count += 1
            new_seg_mask[mask] = new_count
            masks_areas.append(area)
            plane_instances.append(mask)

            avg_normal = np.mean(normals[mask], axis=0)
            avg_normal /= np.sqrt((avg_normal ** 2).sum())  # normalize
            masks_avg_normals.append(avg_normal)

        masks_avg_normals = np.stack(masks_avg_normals)
        masks_areas = np.array(masks_areas)

        outputs = edict(
            {
                "seg_mask": new_seg_mask,
                "normal": masks_avg_normals,
                "areas": masks_areas,
            }
        )

        if vis:
            img_batch = {
                "image": img,
            }
            pred_norm_rgb = ((normals + 1) * 0.5) * 255
            pred_norm_rgb = np.clip(pred_norm_rgb, a_min=0, a_max=255)
            pred_norm_rgb = pred_norm_rgb.astype(np.uint8)
            img_batch["pred_norm"] = pred_norm_rgb

            from disp import overlay_masks
            img_batch['sam_masks'] = overlay_masks(img, sam_outputs['masks'].cpu().numpy())
            img_batch["normal_mask"] = overlay_masks(img, normal_clusters)
            img_batch["plane_mask"] = overlay_masks(img, plane_instances)

            outputs["vis"] = img_batch

        return outputs
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--plane_root_path", type=str, required=True)
    parser.add_argument("--use_normal_estimator", action='store_true')
    args = parser.parse_args()

    seed_everything()

    data_path = args.plane_root_path
    file_list = os.listdir(data_path)
    rgb_list = [file for file in file_list if file.endswith('.png') and 'rgb_frame' in file]
    rgb_list.sort()

    temp_rgb = Image.open(os.path.join(data_path, rgb_list[0]))
    img_height, img_width = temp_rgb.height, temp_rgb.width
    print(f'img_height: {img_height}, img_width: {img_width}')

    normal_list = [file for file in file_list if file.endswith('.npy') and 'mono_normal_frame' in file]      # normal from depth-anything-v2 (MAtCha use this as normal prior)
    normal_list.sort()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = PlaneExcavatorConfig(min_size_ratio=0.01)
    plane_excavator = PlaneExcavator(config, device, img_height, img_width, use_normal_estimator=args.use_normal_estimator)

    if args.use_normal_estimator:
        print('NOTE: use normal estimator for plane extraction')

    print('********** start plane extraction **********')

    for idx, (rgb_file, normal_file) in enumerate(zip(rgb_list, normal_list)):
        rgb_path = os.path.join(data_path, rgb_file)
        normal_path = os.path.join(data_path, normal_file)
        rgb = Image.open(rgb_path)
        rgb = np.array(rgb)
        normal = np.load(normal_path)

        print(f'********** processing frame {idx:06d} **********')

        with torch.no_grad():
            if args.use_normal_estimator:
                output = plane_excavator(rgb, normals=None, vis=True)
            else:
                output = plane_excavator(rgb, normals=normal, vis=True)

        # save plane mask results
        plane_mask = output['seg_mask']
        npy_save_path = os.path.join(data_path, f'plane_mask_frame{idx:06d}.npy')
        np.save(npy_save_path, plane_mask)
        print(f'save to {npy_save_path}')

        # for visualization
        # normal_map = output['vis']['pred_norm']
        # save_path = os.path.join(data_path, f'plane_normal_map_frame{idx:06d}.png')
        # normal_rgb = Image.fromarray(normal_map)
        # normal_rgb.save(save_path)
        # print(f'save to {save_path}')

        vis_map = output['vis']['plane_mask']
        save_path = os.path.join(data_path, f'plane_vis_frame{idx:06d}.png')
        vis_img = Image.fromarray(vis_map)
        vis_img.save(save_path)
        print(f'save to {save_path}')

        # vis_sam_map = output['vis']['sam_masks']
        # save_path = os.path.join(data_path, f'plane_sam_vis_frame{idx:06d}.png')
        # vis_sam_img = Image.fromarray(vis_sam_map)
        # vis_sam_img.save(save_path)
        # print(f'save to {save_path}')

        # vis_normal_mask_map = output['vis']['normal_mask']
        # save_path = os.path.join(data_path, f'plane_normal_mask_vis_frame{idx:06d}.png')
        # vis_normal_mask_img = Image.fromarray(vis_normal_mask_map)
        # vis_normal_mask_img.save(save_path)
        # print(f'save to {save_path}')

        # del output, plane_mask, vis_map, vis_sam_map, vis_normal_mask_map
        del output, plane_mask, vis_map
        torch.cuda.empty_cache()
        gc.collect()
