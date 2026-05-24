import os
import json
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--config_view_num', type=int, default=20)
args = parser.parse_args()

chart_view_num = args.config_view_num
scan_path = 'data/denseview/scan1'

split_save_path = os.path.join(scan_path, f'split-{chart_view_num}views.json')
scan_image_path = os.path.join(scan_path, 'images')

all_view_num = len(os.listdir(scan_image_path))
all_view_list = list(range(0, all_view_num))

if all_view_num < chart_view_num:
    split_dict = {
        "train": all_view_list,
        "test": []
    }

else:
    # train list: uniform sampling from all views
    train_view_list = np.linspace(0, all_view_num - 1, chart_view_num).astype(int).tolist()
    test_view_list = list(set(all_view_list) - set(train_view_list))
    print(f'train view num: {len(train_view_list)}, test view num: {len(test_view_list)}')

    split_dict = {
        "train": train_view_list,
        "test": test_view_list
    }

with open(split_save_path, 'w') as f:
    json.dump(split_dict, f)

print(f'Generated split saved to {split_save_path}')
