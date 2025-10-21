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
    
    # Config
    parser.add_argument('-c', '--config', type=str, default='default', help='Config for multi-resolution TSDF fusion.')
    
    args = parser.parse_args()
    
    # Set console
    CONSOLE = Console(width=120)
    
    # Set output path
    if args.output_path is None:
        args.output_path = os.path.join(args.model_path, 'tsdf_meshes')
    os.makedirs(args.output_path, exist_ok=True)
    CONSOLE.print(f"[INFO] TSDF mesh will be saved to: {args.output_path}")
    
    # Load config
    config_path = os.path.join('configs/multiresolution_tsdf', args.config + '.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Define command
    tsdf_command = " ".join([
        "python", "2d-gaussian-splatting/render_multires.py",
        "--source_path", args.mast3r_scene,
        "--model_path", args.model_path,
        "--output_dir", args.output_path,
        "--depth_ratio", str(config['depth_ratio']),
        "--num_cluster", str(config['num_cluster']),
        "--mesh_res", str(config['mesh_res']),
        "--multires_factors", *[str(factor) for factor in config['multires_factors']],
        "--skip_train",
        "--skip_test",
    ])
    
    # Run command
    CONSOLE.print(f"[INFO] Running command:\n{tsdf_command}")
    run_command_safe(tsdf_command)