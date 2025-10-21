import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import yaml

from matcha.pointmap.depthanythingv2 import get_pointmap_from_mast3r_scene_with_depthanything
from matcha.dm_scene.cameras import CamerasWrapper, rescale_cameras, create_gs_cameras_from_pointmap
from matcha.dm_trainers.charts_alignment import align_charts_in_parallel

from rich.console import Console


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Scene arguments
    parser.add_argument('-s', '--source_path', type=str, required=True)
    parser.add_argument('-m', '--mast3r_scene', type=str, required=True)
    parser.add_argument('-o', '--output_path', type=str, default=None)
    parser.add_argument('--depth_model', type=str, default="depthanythingv2")
    parser.add_argument('--white_background', type=bool, default=False)
    
    # DepthAnything arguments
    parser.add_argument('--depthanythingv2_checkpoint_dir', type=str, default='./Depth-Anything-V2/checkpoints/')
    parser.add_argument('--depthanything_encoder', type=str, default='vitl')
    
    # Deprecated arguments (should not be used)
    parser.add_argument('--image_indices', type=str, default=None)
    parser.add_argument('--n_charts', type=int, default=None)
    
    # Config
    parser.add_argument('-c', '--config', type=str, default='default')
    
    args = parser.parse_args()
    
    # Set console
    CONSOLE = Console(width=120)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set output path
    if args.output_path is None:
        args.output_path = args.mast3r_scene
    else:
        os.makedirs(args.output_path, exist_ok=True)
    CONSOLE.print(f"[INFO] Aligned charts will be saved to: {args.output_path}")
    
    # Load config
    config_path = os.path.join('configs/charts_alignment', args.config + '.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    pm_config = config['pointmap']
    scene_config = config['scene']
    align_config = config['alignment']
    masking_config = config['masking']
    
    # Reprojection loss
    if align_config['use_reprojection_loss']:
        raise NotImplementedError("Reprojection loss is not implemented yet.")
    
    # Build pointmap from MASt3R-SfM data
    scene_pm, sfm_data, mast3r_pm = get_pointmap_from_mast3r_scene_with_depthanything(
        scene_source_path=args.source_path,
        n_images_in_pointmap=args.n_charts,
        image_indices=args.image_indices,
        white_background=args.white_background,
        # MASt3R
        mast3r_scene_source_path=args.mast3r_scene,
        # DepthAnything
        depthanything_checkpoint_dir=args.depthanythingv2_checkpoint_dir,
        depthanything_encoder=args.depthanything_encoder,
        # Misc
        device=device,
        return_sfm_data=True,
        return_mast3r_pointmap=True,
        **pm_config,
    )
    
    # Compute rescaling factor
    _cam_list = create_gs_cameras_from_pointmap(
        scene_pm,
        image_resolution=1,
        load_gt_images=True, 
        max_img_size=pm_config['max_img_size'], 
        use_original_image_size=True,
        average_focal_distances=False,
        verbose=False,
    )
    _pointmap_cameras = CamerasWrapper(_cam_list, no_p3d_cameras=False)
    _scale_factor = scene_config['target_scale'] / _pointmap_cameras.get_spatial_extent()
    
    # Rescale cameras
    _pointmap_cameras = rescale_cameras(_pointmap_cameras, _scale_factor)
    
    # Rescale and prepare reference data based on SFM method
    reference_data = torch.cat([
        _pointmap_cameras.p3d_cameras[i_chart].get_world_to_view_transform().transform_points(
            _scale_factor * sfm_data['sfm_xyz'][sfm_data['image_sfm_points'][_pointmap_cameras.gs_cameras[i_chart].image_name.split('.')[0]]]
        )[..., 2].view(scene_pm.points3d[i_chart][..., 0].shape)[None]
        for i_chart in range(len(_pointmap_cameras))
    ], dim=0)
    if masking_config['use_masks_for_alignment']:
        mast3r_masks = mast3r_pm.confidence > masking_config['sfm_mask_threshold']
        CONSOLE.print(f"[INFO] {mast3r_masks.sum()} points in masks.")
    else:
        mast3r_masks = None
        CONSOLE.print("[INFO] All MASt3R-SfM points will be used for charts alignment.")
    
    # Align the charts
    output = align_charts_in_parallel(
        # Scene
        scene_pm,
        # Data parameters
        reference_data,
        masks=mast3r_masks,
        rendering_size=pm_config['max_img_size'],
        target_scale=scene_config['target_scale'],
        verbose=True,
        return_training_losses=True,
        reprojection_matches_file=None,
        save_charts_data=True,
        charts_data_path=args.output_path,
        **align_config,
    )

    if align_config['use_learnable_confidence']:
        output_verts, output_depths, output_confs, training_losses = output
        output_confs = output_confs - 1.
    else:
        output_verts, output_depths, training_losses = output

    CONSOLE.print("\nInitialization complete!")
    CONSOLE.print("Output vertices shape:", output_verts.shape)
    CONSOLE.print("Output depths shape:", output_depths.shape)
    if align_config['use_learnable_confidence']:
        CONSOLE.print("Output confidence shape:", output_confs.shape)
