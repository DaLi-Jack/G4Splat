import os
import torch
import numpy as np
from PIL import Image

# Create predictor instance
predictor = torch.hub.load("Stable-X/StableNormal", "StableNormal", trust_repo=True)

root_path = '/home/nijunfeng/mycode/project/gs-recon/priorgs/output/mipnerf360-6-views-ori'

scene_list = ['bicycle', 'bonsai', 'counter', 'flowers', 'garden', 'kitchen', 'room', 'stump', 'treehill']
for scene_name in scene_list:
    scene_path = os.path.join(root_path, f'{scene_name}-t1-scratch/free_gaussians/test_see3d_render/vis-train-views')
    file_list = os.listdir(scene_path)
    img_list = [os.path.join(scene_path, file) for file in file_list if 'rgb_frame' in file]
    for img_path in img_list:

        input_image = Image.open(img_path)
        # Apply the model to the image
        normal_image = predictor(input_image)

        normal_npy = np.array(normal_image)         # 0-255
        normal_npy = normal_npy / 255.0             # 0-1
        normal_npy = 1 - normal_npy                 # flip the normal map, consistent with omnidata

        # Save or display the result
        save_name = os.path.basename(img_path).replace('rgb_frame', 'mono_normal_frame')
        save_path = os.path.join(scene_path, save_name)
        convert_normal_image = Image.fromarray((normal_npy * 255).astype(np.uint8))
        convert_normal_image.save(save_path)

    print(f'{scene_name} done!')
