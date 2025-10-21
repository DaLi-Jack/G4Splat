import os
import sys
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

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
    
    args = parser.parse_args()
    
    # Define command
    render_command = " ".join([
        "python", "2d-gaussian-splatting/render_multires.py",
        "--source_path", args.mast3r_scene,
        "--model_path", args.model_path,
        "--skip_test",
        "--skip_mesh",
        "--render_all_img",
    ])
    
    # Run command
    print(render_command)
    run_command_safe(render_command)