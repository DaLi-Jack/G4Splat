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
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--plane_root_dir", type=str, required=True)
    parser.add_argument("--iteration", required=True, type=str)
    parser.add_argument("--see3d_stage", required=True, type=int)
    parser.add_argument("--select_inpaint_num", required=True, type=str)
    args = parser.parse_args()

    # 1. render novel views
    command = f"python 2d-gaussian-splatting/render_novel_views.py --source_path {args.source_path} --model_path {args.model_path} --iteration {args.iteration} --see3d_stage {args.see3d_stage} --select_inpaint_num {args.select_inpaint_num}"
    run_command_safe(command)

    ref_image_path = os.path.join(args.source_path, 'see3d_render', 'ref-views')
    warp_image_path = os.path.join(args.source_path, 'see3d_render', f'stage{args.see3d_stage}', 'select-gs')
    output_root_dir = os.path.join(args.source_path, 'see3d_render', f'stage{args.see3d_stage}', 'select-gs-inpainted')

    # 2. inpaint rgb
    command = f"python 2d-gaussian-splatting/guidance/see3d_util.py --ref_imgs_dir {ref_image_path} --warp_root_dir {warp_image_path} --output_root_dir {output_root_dir}"
    run_command_safe(command)

    # 3. generate depth and normal
    command = f"python 2d-gaussian-splatting/guidance/see3d_dn_util.py --source_path {args.source_path} --see3d_stage {args.see3d_stage}"
    run_command_safe(command)

    # 4. generate 2D planes
    cur_plane_root_dir = os.path.join(args.source_path, 'see3d_render', f'stage{args.see3d_stage}', 'select-gs-planes')
    command = f'python 2d-gaussian-splatting/planes/plane_excavator.py --plane_root_path {cur_plane_root_dir}'
    run_command_safe(command)

    # 5. merge results
    anchor_view_id_json_path = os.path.join(args.source_path, 'see3d_render', f'stage{args.see3d_stage}', 'anchor_view_id.json')
    command = f"python 2d-gaussian-splatting/guidance/merge_util.py --source_path {args.source_path} --see3d_stage {args.see3d_stage} --plane_root_dir {args.plane_root_dir} --anchor_view_id_json_path {anchor_view_id_json_path} --none_replace"
    run_command_safe(command)

    print(f'See3D stage {args.see3d_stage} inpaint done!')
