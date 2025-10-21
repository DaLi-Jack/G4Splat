# modify from https://github.com/graphdeco-inria/gaussian-splatting/blob/main/metrics.py
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), '2d-gaussian-splatting'))

from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from tqdm import tqdm
from utils.image_utils import psnr


def eval_images(source_path, pred_img_root_path, test_views_list):

    renders = []
    gts = []

    img_data_path = os.path.join(source_path, 'images')
    img_data_list = os.listdir(img_data_path)
    img_data_list.sort()
    for view_id in test_views_list:
        pred_img_path = os.path.join(pred_img_root_path, f'{view_id:05d}.png')
        # gt_img_path = os.path.join(source_path, 'images', f'{view_id:06d}_rgb.png')
        gt_img_path = os.path.join(img_data_path, img_data_list[view_id])
        render = Image.open(pred_img_path)
        gt = Image.open(gt_img_path)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())

    ssims = []
    psnrs = []
    lpipss = []

    for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
        ssims.append(ssim(renders[idx], gts[idx]))
        psnrs.append(psnr(renders[idx], gts[idx]))
        lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

    return ssims, psnrs, lpipss

def eval_images_temp(source_path, pred_img_root_path, test_views_list, only_psnr=False):

    renders = []
    gts = []

    img_data_path = source_path
    img_data_list = os.listdir(img_data_path)
    img_data_list.sort()

    test_img_list = os.listdir(pred_img_root_path)
    test_img_list.sort()
    for view_id in test_views_list:
        pred_img_path = os.path.join(pred_img_root_path, test_img_list[view_id])
        # gt_img_path = os.path.join(source_path, 'images', f'{view_id:06d}_rgb.png')
        gt_img_path = os.path.join(img_data_path, img_data_list[view_id])
        render = Image.open(pred_img_path)
        gt = Image.open(gt_img_path)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())

    ssims = []
    psnrs = []
    lpipss = []

    for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
        psnrs.append(psnr(renders[idx], gts[idx]))
        if not only_psnr:
            ssims.append(ssim(renders[idx], gts[idx]))
            lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

    return ssims, psnrs, lpipss
