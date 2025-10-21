import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from glob import glob
from transformers import CLIPTextModel, CLIPTokenizer

from See3D_modules.mv_diffusion import mvdream_diffusion_model
from See3D_modules.mv_diffusion_SR import mvdream_diffusion_model as mvdream_diffusion_model_SR
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import gc

import time

class See3D(nn.Module):
    def __init__(
        self,
        device,
        base_model_path='./checkpoint/MVD_weights/',
        model_type='sparse',                                # single or sparse
        use_SR=False,
        seed=12345,
    ):
        super().__init__()

        self.device = device
        mv_unet_path = os.path.join(base_model_path, f"unet/{model_type}/ema-checkpoint")

        tokenizer = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")
        self.rgb_model = mvdream_diffusion_model(base_model_path, mv_unet_path, tokenizer, seed=seed)
        if use_SR:
            self.rgb_model_SR = mvdream_diffusion_model_SR(base_model_path, mv_unet_path, tokenizer, seed=seed)

    def PIL2tensor(self, height, width, num_frames, masks, warps, logicalNot=False):
        channels = 3
        pixel_values = torch.empty((num_frames, channels, height, width))
        condition_pixel_values = torch.empty((num_frames, channels, height, width))
        masks_pixel_values = torch.ones((num_frames, 1, height, width))
        
        # input_ids
        prompt = ''

        for i, img in enumerate(masks):
            img = masks[i]
            img = img.convert('L') # make sure channel 1
            img_resized = img.resize((width, height)) # hard code here
            img_tensor = torch.from_numpy(np.array(img_resized)).float()

            # Normalize the image by scaling pixel values to [0, 1]
            img_normalized = img_tensor / 255
            mask_condition = (img_normalized > 0.9).float()
            
            masks_pixel_values[i] = mask_condition
        
        for i, img in enumerate(warps):
            # Resize the image and convert it to a tensor
            img_resized = img.resize((width, height)) # hard code here
            img_tensor = torch.from_numpy(np.array(img_resized)).float()

            # Normalize the image by scaling pixel values to [-1, 1]
            img_normalized = img_tensor / 127.5 - 1

            img_normalized = img_normalized.permute(2, 0, 1)  # For RGB images

            if(logicalNot):
                img_normalized = torch.logical_not(masks_pixel_values[i])*(-1) + masks_pixel_values[i]*img_normalized
            condition_pixel_values[i] = img_normalized
            
        return [prompt], {
                'conditioning_pixel_values': condition_pixel_values, # [-1,1]
                'masks': masks_pixel_values# [0,1]
                }
        
    def get_image_files(self, folder_path):
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.webp', '*.JPG', '*.JPEG', '*.PNG', '*.GIF', '*.BMP', '*.TIFF', '*.WEBP']
        
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob(os.path.join(folder_path, ext)))

        image_names = [os.path.basename(file) for file in image_files]
        
        return image_names
    
    def load_ref_images(self, folder_path, height_mvd, width_mvd):
        temp_image_names = self.get_image_files(folder_path)
        temp_img_path = os.path.join(folder_path, temp_image_names[0])
        temp_img = Image.open(temp_img_path)
        width_ref, height_ref = temp_img.size

        ref_images = []
        ref_image_names = []
        if height_ref != height_mvd or width_ref != width_mvd:
            # split into two images
            if height_ref > width_ref:
                # resize width_ref to width_mvd
                height_tgt = int(height_ref * width_mvd / width_ref)
                for image_name in temp_image_names:
                    img = Image.open(os.path.join(folder_path, image_name))
                    img = img.resize((width_mvd, height_tgt))

                    # split into two images
                    img_top = img.crop((0, 0, width_mvd, height_mvd))
                    img_bottom = img.crop((0, height_tgt-height_mvd, width_mvd, height_tgt))

                    img_name_top = image_name.split('.')[0] + '_top.png'
                    img_name_bottom = image_name.split('.')[0] + '_bottom.png'

                    ref_images.append(img_top)
                    ref_images.append(img_bottom)
                    ref_image_names.append(img_name_top)
                    ref_image_names.append(img_name_bottom)
            elif width_ref > height_ref:
                # resize height_ref to height_mvd
                width_tgt = int(width_ref * height_mvd / height_ref)
                for image_name in temp_image_names:
                    img = Image.open(os.path.join(folder_path, image_name))
                    img = img.resize((width_tgt, height_mvd))

                    # split into two images
                    img_left = img.crop((0, 0, width_mvd, height_mvd))
                    img_right = img.crop((width_tgt-width_mvd, 0, width_tgt, height_mvd))

                    img_name_left = image_name.split('.')[0] + '_left.png'
                    img_name_right = image_name.split('.')[0] + '_right.png'

                    ref_images.append(img_left)
                    ref_images.append(img_right)
                    ref_image_names.append(img_name_left)
                    ref_image_names.append(img_name_right)
            else:
                # resize both height_ref and width_ref to height_mvd and width_mvd
                for image_name in temp_image_names:
                    img = Image.open(os.path.join(folder_path, image_name))
                    img = img.resize((width_mvd, height_mvd))

                    ref_images.append(img)
                    ref_image_names.append(image_name)

        return ref_images, ref_image_names

    def inpainting(self, source_imgs_dir, warp_root_dir, output_root_dir, super_resolution=False):

        os.makedirs(output_root_dir, exist_ok=True)

        height_mvd = 512
        width_mvd = 512
        masks_infer = []
        warps_infer = []
        input_names = []

        # load source images
        gt_num_b = 0
        mask2 = np.ones((height_mvd, width_mvd), dtype=np.float32)

        # image_names_ref = self.get_image_files(source_imgs_dir)
        # fimage = Image.open(os.path.join(source_imgs_dir, image_names_ref[0]))
        # (width, height)= fimage.size

        ref_images, ref_image_names = self.load_ref_images(source_imgs_dir, height_mvd, width_mvd)

        for imn, ref_img in zip(ref_image_names, ref_images):
            masks_infer.append(Image.fromarray(np.repeat(np.expand_dims(np.round(mask2*255.).astype(np.uint8),axis=2),3,axis=2)).resize((width_mvd, height_mvd)))
            warps_infer.append(ref_img)
            input_names.append(imn)
            gt_num_b = gt_num_b + 1

        # load warp images and masks
        image_files = glob(os.path.join(warp_root_dir, "warp_*"))
        image_names = [os.path.basename(image) for image in image_files]
        image_names.sort()

        fimage = Image.open(os.path.join(warp_root_dir, image_names[0]))
        (width, height)= fimage.size

        for ins in image_names:
            warps_infer.append(Image.open(os.path.join(warp_root_dir, ins))) 
            masks_infer.append(Image.open(os.path.join(warp_root_dir, ins.replace('warp','mask'))))
            input_names.append(ins)
        print('all inpainting sequence length:', len(warps_infer))

        # inpainting
        images_predict = []
        images_mask_p = []
        images_predict_names = []

        grounp_size = len(masks_infer)
        for i in range(0, len(masks_infer[gt_num_b:]), grounp_size):
            if(len(images_predict)!=0):
                masks_infer_batch = masks_infer[:gt_num_b] + [masks_infer_batch[-1]] + masks_infer[(gt_num_b+i):(i+gt_num_b+grounp_size)]
                warp_infer_batch = warps_infer[:gt_num_b] + [images_predict[-1]] + warps_infer[(gt_num_b+i):(i+gt_num_b+grounp_size)]
                input_names_batch = input_names[:gt_num_b] + [input_names_batch[len(masks_infer_batch)//2]] + [input_names_batch[-1]] + input_names[(gt_num_b+i):(i+gt_num_b+grounp_size)]
            else:
                masks_infer_batch = masks_infer[:gt_num_b] + masks_infer[(gt_num_b+i):(i+gt_num_b+grounp_size)]
                warp_infer_batch = warps_infer[:gt_num_b] + warps_infer[(gt_num_b+i):(i+gt_num_b+grounp_size)]
                input_names_batch = input_names[:gt_num_b] + input_names[(gt_num_b+i):(i+gt_num_b+grounp_size)]

            prompt, batch = self.PIL2tensor(height_mvd,width_mvd,len(masks_infer_batch),masks_infer_batch,warp_infer_batch,logicalNot=False)
            if(len(images_predict)!=0):
                images_predict_batch = self.rgb_model.inference_next_frame(prompt,batch,len(masks_infer_batch),height_mvd,width_mvd,gt_num_frames=gt_num_b,output_type='pil')
                for jj in range(gt_num_b+1,len(images_predict_batch)):
                    images_predict.append(images_predict_batch[jj])
                    images_mask_p.append(batch['masks'][0][jj][0].cpu().numpy())
                    images_predict_names.append(input_names_batch[jj])
            else:
                images_predict_batch = self.rgb_model.inference_next_frame(prompt,batch,len(masks_infer_batch),height_mvd,width_mvd,gt_num_frames=gt_num_b,output_type='pil')
                for jj in range(gt_num_b,len(images_predict_batch)):
                    images_predict.append(images_predict_batch[jj])
                    images_mask_p.append(batch['masks'][0][jj][0].cpu().numpy())
                    images_predict_names.append(input_names_batch[jj])
                
        for jj in range(len(images_predict)):
            images_predict[jj].resize((width, height)).save(os.path.join(output_root_dir,"predict_{}".format(images_predict_names[jj])))

        print(f'end inpainting, result saved in {output_root_dir}')

        if super_resolution:
            print('start SR inpainting')
            del self.rgb_model
            gc.collect()
            torch.cuda.empty_cache()

            masks_infer_SR = []
            warps_infer_SR = []
            mask2 = np.ones((height_mvd*2,width_mvd*2), dtype=np.float32)

            ref_images, ref_image_names = self.load_ref_images(source_imgs_dir, height_mvd, width_mvd)

            for imn, ref_img in zip(ref_image_names, ref_images):
                masks_infer_SR.append(Image.fromarray(np.repeat(np.expand_dims(np.round(mask2*255.).astype(np.uint8),axis=2),3,axis=2)).resize((width_mvd, height_mvd)))
                warps_infer_SR.append(ref_img)

            for i in range(len(images_predict)):
                masks_infer_SR.append(masks_infer[i])
                warps_infer_SR.append(images_predict[i])

            images_predict = []
            images_predict_names = []
            # grounp_size = min((len(masks_infer_SR) + 5)//2,50)
            grounp_size = (len(masks_infer_SR) + 3) // 2
            # grounp_size = (len(masks_infer_SR) + 3)
            print('grounp_size:',grounp_size)
            for i in range(0, len(masks_infer_SR[gt_num_b:]), grounp_size):
                if(len(images_predict)!=0):
                    masks_infer_batch = masks_infer_SR[:gt_num_b] + [masks_infer_batch[len(masks_infer_batch)//2]] + [masks_infer_batch[-1]] + masks_infer_SR[(gt_num_b+i):(i+gt_num_b+grounp_size)]
                    warp_infer_batch = warps_infer_SR[:gt_num_b] + [images_predict[len(images_predict)//2]] + [images_predict[-1]] + warps_infer_SR[(gt_num_b+i):(i+gt_num_b+grounp_size)]
                    input_names_batch = input_names[:gt_num_b] + [input_names_batch[len(masks_infer_batch)//2]] + [input_names_batch[-1]] + input_names[(gt_num_b+i):(i+gt_num_b+grounp_size)]
                else:
                    masks_infer_batch = masks_infer_SR[:gt_num_b] + masks_infer_SR[(gt_num_b+i):(i+gt_num_b+grounp_size)]
                    warp_infer_batch = warps_infer_SR[:gt_num_b] + warps_infer_SR[(gt_num_b+i):(i+gt_num_b+grounp_size)]
                    input_names_batch = input_names[:gt_num_b] + input_names[(gt_num_b+i):(i+gt_num_b+grounp_size)]

                
                prompt, batch = self.PIL2tensor(height_mvd*2,width_mvd*2,len(masks_infer_batch),masks_infer_batch,warp_infer_batch)
                if(len(images_predict)!=0):
                    images_predict_batch = self.rgb_model_SR.inference_next_frame(prompt,batch,len(masks_infer_batch),height_mvd*2,width_mvd*2,gt_num_frames=gt_num_b,output_type='pil')
                    for jj in range(gt_num_b+2,len(images_predict_batch)):
                        images_predict.append(images_predict_batch[jj])
                        images_predict_names.append(input_names_batch[jj])
                else:
                    images_predict_batch = self.rgb_model_SR.inference_next_frame(prompt,batch,len(masks_infer_batch),height_mvd*2,width_mvd*2,gt_num_frames=gt_num_b,output_type='pil')
                    for jj in range(gt_num_b,len(images_predict_batch)):
                        images_predict.append(images_predict_batch[jj])
                        images_predict_names.append(input_names_batch[jj])
                gc.collect()
                torch.cuda.empty_cache()


            for jj in range(len(images_predict)):
                images_predict[jj].resize((width_mvd*2, height_mvd*2)).save(os.path.join(output_root_dir,"SR_predict_{}".format(images_predict_names[jj])))

            print(f'end SR inpainting, result saved in {output_root_dir}')
        
        
if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--ref_imgs_dir', type=str)
    parser.add_argument('--warp_root_dir', type=str)
    parser.add_argument('--output_root_dir', type=str)
    parser.add_argument('--use_SR', action='store_true', help='Use super resolution for inpainting')
    args = parser.parse_args()

    source_imgs_dir = args.ref_imgs_dir
    warp_root_dir = args.warp_root_dir
    output_root_dir = args.output_root_dir

    t1 = time.time()

    see3d = See3D(device='cuda', use_SR=args.use_SR)
    see3d.inpainting(source_imgs_dir=source_imgs_dir, warp_root_dir=warp_root_dir, output_root_dir=output_root_dir, super_resolution=args.use_SR)

    # save cat img
    cat_save_root_path = os.path.join(os.path.dirname(output_root_dir), 'cat_img')
    os.makedirs(cat_save_root_path, exist_ok=True)
    inpaint_img_list = os.listdir(output_root_dir)
    inpaint_img_list = [img for img in inpaint_img_list if '.png' in img]
    img_num = len(inpaint_img_list)
    none_visible_rate_list = []
    for idx in range(img_num):
        gs_render_img_path = os.path.join(warp_root_dir, f'warp_frame{idx:06d}.png')
        mask_img_path = os.path.join(warp_root_dir, f'mask_frame{idx:06d}.png')
        inpaint_img_path = os.path.join(output_root_dir, f'predict_warp_frame{idx:06d}.png')

        mask_img = Image.open(mask_img_path)
        mask_img = np.array(mask_img) / 255
        total_pixels = mask_img.shape[0] * mask_img.shape[1]
        mask_pixels = np.sum(mask_img)
        none_visible_rate = 1 - mask_pixels / total_pixels
        none_visible_rate_list.append(none_visible_rate)

        gs_render_img = Image.open(gs_render_img_path)
        inpaint_img = Image.open(inpaint_img_path)

        padding = 10
        cat_img = Image.new('RGB', (gs_render_img.width + inpaint_img.width + padding, gs_render_img.height))
        cat_img.paste(gs_render_img, (0, 0))
        cat_img.paste(inpaint_img, (gs_render_img.width + padding, 0))

        cat_img.save(os.path.join(cat_save_root_path, f'{idx:06d}-{none_visible_rate:.2f}.png'))

    plt.figure(figsize=(10, 6))
    plt.plot(none_visible_rate_list, label='None Visible Rate')
    plt.xlabel('Frame Index')
    plt.ylabel('None Visible Rate')
    plt.title('None Visible Rate of GS Render and PCD Render')
    plt.legend()
    plt.savefig(os.path.join(cat_save_root_path, 'none_visible_rate.png'))
    plt.close()

    print(f'cat img saved in {cat_save_root_path}')

    t2 = time.time()
    print(f'Time cost: {t2 - t1:.2f}s')
    
