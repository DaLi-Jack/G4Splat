import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml

from rich.console import Console

def run_command_safe(command):
    print(f"Running command: {command}")
    exit_code = os.system(command)
    if exit_code != 0:
        print("Command failed!")
        sys.exit(1)
    else:
        print("Command succeeded!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Scene arguments
    parser.add_argument('-s', '--mast3r_scene', type=str, required=True, help='Path to the MASt3R-SfM scene.')
    parser.add_argument('-m', '--model_path', type=str, required=True, help='Path to the 2D Gaussian Splatting model.')
    parser.add_argument('-o', '--output_path', type=str, default=None, help='Path to save the output mesh.')
    
    # Dense supervision (Optional)
    parser.add_argument('--dense_data_path', type=str, default=None, help='Path to the dense supervision data.')
    
    # Interpolate views
    parser.add_argument('--interpolate_views', action='store_true', default=False, help='Interpolate views for mesh extraction.')
    
    # Config
    parser.add_argument('-c', '--config', type=str, default='default', help='Config for adaptive tetrahedralization.')
    parser.add_argument('--downsample_ratio', type=float, default=0.5, help='Downsample ratio for tetrahedralization.')
    
    args = parser.parse_args()
    
    # Set console
    CONSOLE = Console(width=120)
    
    # Set output path
    if args.output_path is None:
        args.output_path = os.path.join(args.model_path, 'tetra_meshes')
    os.makedirs(args.output_path, exist_ok=True)
    CONSOLE.print(f"[INFO] Tetrahedralized mesh will be saved to: {args.output_path}")
    
    # Load config
    config_path = os.path.join('configs/adaptive_tetrahedralization', args.config + '.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Dense supervision (Optional)
    if args.dense_data_path is not None:
        dense_arg = " ".join([
            "--dense_data_path", args.dense_data_path,
        ])
    else:
        dense_arg = ""
        
    # Define command
    iteration_to_load_for_tetra = max(
        [int(fname.split("_")[-1]) for fname in os.listdir(os.path.join(args.model_path, "point_cloud"))]
    )
    tetra_command = " ".join([
        # "python", "2d-gaussian-splatting/extract_mesh.py",
        "python", "2d-gaussian-splatting/extract_mesh_adaptive_tsdf.py",
        "--source_path", args.mast3r_scene,
        "--model_path", args.model_path,
        "--iteration", str(iteration_to_load_for_tetra),
        "--downsample_ratio", str(args.downsample_ratio),
        "--gaussian_flatness", str(config['gaussian_flatness']),
        "--depth_ratio", str(config['depth_ratio']),
        "--filter_mesh" if config['filter_mesh'] else "",
        "--texture_mesh",
        "--output_dir", args.output_path,
        "--interpolate_cameras" if args.interpolate_views else "",
        dense_arg,
        ###
        "--interpolate_depth" if config['interpolate_depth'] else "",
        "--interpolation_mode", config['interpolation_mode'],
        "--weight_interpolation_by_depth_gradient" if config['weight_interpolation_by_depth_gradient'] else "",
        "--truncation_margin", str(config['truncation_margin']),
        #
        "--use_dilated_depth" if config['use_dilated_depth'] else "",
        "--use_sdf_tolerance" if config['use_sdf_tolerance'] else "",
        "--use_unbiased_tsdf" if config['use_unbiased_tsdf'] else "",
        "--use_binary_opacity" if config['use_binary_opacity'] else "",
        #
        "--filter_with_depth_gradient" if config['filter_with_depth_gradient'] else "",
        "--filter_with_normal_consistency" if config['filter_with_normal_consistency'] else "",
        #
        "--weight_by_softmax" if config['weight_by_softmax'] else "",
        "--weight_by_normal_consistency" if config['weight_by_normal_consistency'] else "",
        "--softmax_temperature", str(config['softmax_temperature']),
        #
        "--n_neighbors_to_interpolate", str(config['n_neighbors_to_interpolate']),
        "--n_interpolated_cameras_for_each_neighbor", str(config['n_interpolated_cameras_for_each_neighbor']),
    ])
    
    # Run command
    CONSOLE.print(f"[INFO] Running command:\n{tetra_command}")
    run_command_safe(tetra_command)