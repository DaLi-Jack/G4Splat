import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import subprocess

import argparse
import yaml

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
    
    parser.add_argument('-s', '--source_path', type=str, 
        help='Path to the source data to use. Can be a directory containing images, or a colmap output directory.')
    parser.add_argument('-o', '--output_path', type=str, default=None, 
        help='Path to the output directory.')
    
    # Data parameters
    # parser.add_argument('--use_all_images', action='store_true', help='Use all images for optimization.')
    parser.add_argument('--n_images', type=int, default=None, 
        help='Number of images to use for optimization, sampled with constant spacing. If not provided, all images will be used.')
    parser.add_argument('--image_idx', type=int, nargs='*', default=None, 
        help='View indices to use for optimization (zero-based indexing). If provided, this will override the --n_images.')
    parser.add_argument('--randomize_images', action='store_true', 
        help='Shuffle training images before sampling with constant spacing. If image_idx is provided, this will be ignored.')
    
    # Config
    parser.add_argument('-c', '--config', type=str, default='unposed',
        help='name of the config to use. Should be either "unposed", "posed" or a custom config.')
    
    # Environment
    # parser.add_argument('--env', type=str, default='matcha',
    #     help='name of the environment to use.')

    # Set image arguments
    args = parser.parse_args()
    if args.n_images is None: 
        if args.image_idx is None:
            args.use_all_images = True
            args.randomize_images = False
            args.n_images = -1  # Just a placeholder
            print(f"[INFO] Using all images for optimization.")
        else:
            args.use_all_images = False
            args.randomize_images = False
            args.n_images = len(args.image_idx)
            print(f"[INFO] Using {args.n_images} images for optimization.")
    else:
        if args.image_idx is not None:
            raise ValueError("Cannot provide both --n_images and --image_idx arguments.")
        else:
            args.use_all_images = False
            print(f"[INFO] Using {args.n_images} images for optimization.")
            
    # Set output path
    if args.output_path is None:
        if args.source_path.endswith(os.sep):
            output_dir_name = args.source_path.split(os.sep)[-2]
        else:
            output_dir_name = args.source_path.split(os.sep)[-1]
        args.output_path = os.path.join('output', output_dir_name)
        args.output_path = os.path.join(args.output_path, 'mast3r_sfm')
    os.makedirs(args.output_path, exist_ok=True)
    print(f"[INFO] Scene will be saved to: {args.output_path}")
            
    # Load config
    config_path = os.path.join('configs/mast3r', args.config + '.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Define command
    tmp_idx = args.image_idx if args.image_idx is not None else []
    command = " ".join([
        # "conda",  "run", "-n", args.env, 
        "python", "mast3r/run_mast3r.py",
        "--scene_path", args.source_path,
        "--output_dir", args.output_path,
        "--weights_path", config['weights_path'],
        "--retrieval_model", config['retrieval_model'],
        "--min_conf_thr", str(config['min_conf_thr']),
        "--matching_conf_thr", str(config['matching_conf_thr']),
        "--n_coarse_iterations", str(config['n_coarse_iterations']),
        "--n_refinement_iterations", str(config['n_refinement_iterations']),
        "--TSDF_thresh", str(config['TSDF_thresh']),
        "--fix_focal" if config['fix_focal'] else "",
        "--fix_principal_point" if config['fix_principal_point'] else "",
        "--fix_rotation" if config['fix_rotation'] else "",
        "--fix_translation" if config['fix_translation'] else "",
        "--n_images", str(args.n_images),
        "--use_all_images" if args.use_all_images else "",
        "--image_idx" if args.image_idx is not None else "", 
        *[str(i) for i in tmp_idx],
        "--randomize_images" if args.randomize_images else "",
        "--image_size", str(config['image_size']),
        "--max_window_size", str(config['max_window_size']),
        "--max_refid", str(config['max_refid']),
        "--use_calibrated_poses" if config['use_calibrated_poses'] else "",
        "--save_glb" if config['save_glb'] else "",
        "--output_conf_thr", str(config['output_conf_thr']),
        "--align_camera_locations" if config['align_camera_locations'] else "",
    ])
    
    # Run command
    print(f"[INFO] Running command:\n", command)
    run_command_safe(command)
    
    # subprocess.run([
    #     "conda",  "run", "-n", args.env, "python", "mast3r/run_mast3r.py",
    #     "--scene_path", args.source_path,
    #     "--output_dir", args.output_path,
    #     "--weights_path", config['weights_path'],
    #     "--retrieval_model", config['retrieval_model'],
    #     "--min_conf_thr", str(config['min_conf_thr']),
    #     "--matching_conf_thr", str(config['matching_conf_thr']),
    #     "--n_coarse_iterations", str(config['n_coarse_iterations']),
    #     "--n_refinement_iterations", str(config['n_refinement_iterations']),
    #     "--TSDF_thresh", str(config['TSDF_thresh']),
    #     "--fix_focal" if config['fix_focal'] else "",
    #     "--fix_principal_point" if config['fix_principal_point'] else "",
    #     "--fix_rotation" if config['fix_rotation'] else "",
    #     "--fix_translation" if config['fix_translation'] else "",
    #     "--n_images", str(args.n_images),
    #     "--use_all_images" if args.use_all_images else "",
    #     "--image_idx" if args.image_idx is not None else "", 
    #     *[str(i) for i in tmp_idx],
    #     "--randomize_images" if args.randomize_images else "",
    #     "--image_size", str(config['image_size']),
    #     "--max_window_size", str(config['max_window_size']),
    #     "--max_refid", str(config['max_refid']),
    #     "--use_calibrated_poses" if config['use_calibrated_poses'] else "",
    #     "--save_glb" if config['save_glb'] else "",
    #     "--output_conf_thr", str(config['output_conf_thr']),
    #     "--align_camera_locations" if config['align_camera_locations'] else "",
    # ])
