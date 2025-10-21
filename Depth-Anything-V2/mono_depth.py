import cv2
import torch
import numpy as np

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from depth_anything_v2.dpt import DepthAnythingV2

import matplotlib.pyplot as plt

from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--input_path", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    model = model.to(DEVICE).eval()

    input_path = args.input_path
    output_path = args.output_path

    img_list = os.listdir(input_path)
    img_list = sorted(img_list)
    img_list = [os.path.join(input_path, file) for file in img_list]

    for idx, img_path in enumerate(img_list):
        raw_img = cv2.imread(img_path)
        depth = model.infer_image(raw_img) # HxW raw depth map in numpy

        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-5) + 1e-6
        depth = 1. - depth_normalized                  # convert to depth
        save_name = f'mono_depth_frame{idx:06d}.png'
        save_path = os.path.join(output_path, save_name)

        # save mono depth .npy
        npy_save_depth_path = os.path.join(output_path, save_name.replace('.png', '.npy'))
        np.save(npy_save_depth_path, depth)

        plt.imsave(save_path, depth, cmap='viridis')

    print(f'Mono depth generated!')
