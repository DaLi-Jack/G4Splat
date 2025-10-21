import os
import sys
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
    parser.add_argument('--source_path', type=str, required=True)
    parser.add_argument('--plane_root_path', type=str, required=True)
    parser.add_argument("--pnts_path", type=str, required=True)
    parser.add_argument("--see3d_root_path", type=str, default=None)
    parser.add_argument("--vis_plane_path", type=str, default=None)
    parser.add_argument("--anchor_view_id_json_path", type=str, default=None)
    args = parser.parse_args()

    # get global 3D plane
    if args.see3d_root_path is not None:
        if args.vis_plane_path is not None:
            command = f"python 2d-gaussian-splatting/planes/merge_global_3Dplane.py --source_path {args.source_path} --pnts_path {args.pnts_path} --plane_root_path {args.plane_root_path} --see3d_root_path {args.see3d_root_path} --vis_plane_path {args.vis_plane_path}"
        else:
            command = f"python 2d-gaussian-splatting/planes/merge_global_3Dplane.py --source_path {args.source_path} --pnts_path {args.pnts_path} --plane_root_path {args.plane_root_path} --see3d_root_path {args.see3d_root_path}"
    else:
        if args.vis_plane_path is not None:
            command = f"python 2d-gaussian-splatting/planes/merge_global_3Dplane.py --source_path {args.source_path} --pnts_path {args.pnts_path} --plane_root_path {args.plane_root_path} --vis_plane_path {args.vis_plane_path}"
        else:
            command = f"python 2d-gaussian-splatting/planes/merge_global_3Dplane.py --source_path {args.source_path} --pnts_path {args.pnts_path} --plane_root_path {args.plane_root_path}"
    run_command_safe(command)

    # refine depth with planes
    if args.see3d_root_path is not None:
        command = f"python 2d-gaussian-splatting/planes/refine_depth_with_planes.py --source_path {args.source_path} --plane_root_path {args.plane_root_path} --see3d_root_path {args.see3d_root_path}"
    else:
        command = f"python 2d-gaussian-splatting/planes/refine_depth_with_planes.py --source_path {args.source_path} --plane_root_path {args.plane_root_path}"
    run_command_safe(command)

    # get confident map
    if args.see3d_root_path is None:
        command = f"python 2d-gaussian-splatting/guidance/inconsistence_solver.py --source_path {args.source_path} --plane_root_path {args.plane_root_path}"
    else:
        if args.anchor_view_id_json_path is None:
            command = f"python 2d-gaussian-splatting/guidance/inconsistence_solver.py --source_path {args.source_path} --plane_root_path {args.plane_root_path} --see3d_root_path {args.see3d_root_path}"
        else:
            command = f"python 2d-gaussian-splatting/guidance/plane_inconsistency_solver.py --source_path {args.source_path} --plane_root_path {args.plane_root_path} --see3d_root_path {args.see3d_root_path} --anchor_view_id_json_path {args.anchor_view_id_json_path}"
    run_command_safe(command)

    print('Plane refine depth done!')

