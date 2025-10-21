import os
import numpy as np
from PIL import Image
from argparse import ArgumentParser


def replace_inpaint_results(warp_root_dir, inpaint_root_dir, save_root_dir):
    os.makedirs(save_root_dir, exist_ok=True)
    inpaint_img_list = os.listdir(inpaint_root_dir)
    inpaint_img_list = [img for img in inpaint_img_list if '.png' in img]
    img_num = len(inpaint_img_list)
    for idx in range(img_num):
        gs_render_img_path = os.path.join(warp_root_dir, f'warp_frame{idx:06d}.png')
        mask_img_path = os.path.join(warp_root_dir, f'mask_frame{idx:06d}.png')
        inpaint_img_path = os.path.join(inpaint_root_dir, f'predict_warp_frame{idx:06d}.png')

        gs_render_img = Image.open(gs_render_img_path)
        mask_img = Image.open(mask_img_path)
        inpaint_img = Image.open(inpaint_img_path)

        mask_map = np.array(mask_img) / 255
        gs_render_img = np.array(gs_render_img)
        inpaint_img = np.array(inpaint_img)

        save_img = inpaint_img.copy()
        save_img[mask_map == 1] = gs_render_img[mask_map == 1]          # NOTE: visible part is gs_render_img, invisible part is inpaint_img
        save_img = Image.fromarray(save_img)
        save_img.save(os.path.join(save_root_dir, f'predict_replaced_warp_frame{idx:06d}.png'))

        print(f'Inpaint {idx} replace done!')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--warp_root_dir', type=str)
    parser.add_argument('--inpaint_root_dir', type=str)
    parser.add_argument('--save_root_dir', type=str)
    args = parser.parse_args()

    replace_inpaint_results(args.warp_root_dir, args.inpaint_root_dir, args.save_root_dir)
    print('Replace inpaint results done!')

