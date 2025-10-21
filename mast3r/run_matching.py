import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np
import json
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from PIL.ImageOps import exif_transpose
import cv2
import random
# Import necessary MASt3R modules
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from mast3r.retrieval.processor import Retriever

import mast3r.utils.path_to_dust3r
# from dust3r.utils.image import load_images
from dust3r.inference import inference

import torchvision.transforms as tvf
ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = Image.LANCZOS
    elif S <= long_edge_size:
        interp = Image.BICUBIC
    new_size = tuple(int(round(x*long_edge_size/S)) for x in img.size)
    return img.resize(new_size, interp)


def load_images(folder_or_list, size, square_ok=False, verbose=True):
    """ open and convert all images in a list or folder to proper input format for DUSt3R
    """
    if isinstance(folder_or_list, str):
        if verbose:
            print(f'>> Loading images from {folder_or_list}')
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))

    elif isinstance(folder_or_list, list):
        if verbose:
            print(f'>> Loading a list of {len(folder_or_list)} images')
        root, folder_content = '', folder_or_list

    else:
        raise ValueError(f'bad {folder_or_list=} ({type(folder_or_list)})')

    supported_images_extensions = ['.jpg', '.jpeg', '.png']
    supported_images_extensions = tuple(supported_images_extensions)

    imgs = []
    for path in folder_content:
        if not path.lower().endswith(supported_images_extensions):
            continue
        img = exif_transpose(Image.open(os.path.join(root, path))).convert('RGB')
        W1, H1 = img.size
        if size == 224:
            # resize short side to 224 (then crop)
            img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
        else:
            # resize long side to 512
            img = _resize_pil_image(img, size)
        W, H = img.size
        cx, cy = W//2, H//2
        if size == 224:
            half = min(cx, cy)
            img = img.crop((cx-half, cy-half, cx+half, cy+half))
        else:
            halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
            if not (square_ok) and W == H:
                halfh = 3*halfw/4
            img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))

        W2, H2 = img.size
        if verbose:
            print(f' - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}')
        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
            [img.size[::-1]]), idx=len(imgs), instance=os.path.basename(path)))

    assert imgs, 'no images foud at '+root
    if verbose:
        print(f' (Found {len(imgs)} images)')
    return imgs



def check_image_sizes(image_paths):
    """
    Check if all images have the required size 512x512.
    
    Args:
        image_paths (list): List of image paths
    
    Returns:
        bool: True if all images have the required size, False otherwise
    """
    for path in image_paths:
        img = cv2.imread(path)
        if img.shape[0] != 512 or img.shape[1] != 512:
            print(f"Error: Image {path} has size {img.shape[0]}x{img.shape[1]}, expected 512x512")
            return False
    return True


def split_images(image_paths, output_dir):
    """
    Split each 512x512 image into two 384x512 images (top and bottom).
    
    Args:
        image_paths (list): List of image paths
        output_dir (str): Directory to save split images
    
    Returns:
        dict: Dictionary mapping original image paths to split image paths
        list: List of all split image paths
    """
    split_dir = os.path.join(output_dir, 'split_images')
    os.makedirs(split_dir, exist_ok=True)
    
    split_mapping = {}
    all_split_paths = []
    
    for path in tqdm(image_paths, desc="Splitting images"):
        img = cv2.imread(path)
        
        # Extract filename without extension
        base_name = os.path.basename(path)
        name_without_ext = os.path.splitext(base_name)[0]
        
        # Split into top (0-384) and bottom (128-512)
        top_img = img[:384, :]
        bottom_img = img[128:, :]
        
        # Save split images
        top_path = os.path.join(split_dir, f"{name_without_ext}_top.png")
        bottom_path = os.path.join(split_dir, f"{name_without_ext}_bottom.png")
        
        cv2.imwrite(top_path, top_img)
        cv2.imwrite(bottom_path, bottom_img)
        
        # Store mapping
        split_mapping[path] = {
            'top': top_path,
            'bottom': bottom_path,
            'name': base_name
        }
        
        all_split_paths.extend([top_path, bottom_path])
    
    return split_mapping, all_split_paths


def get_pixel_pairs_from_novel_views(
    novel_views_dir,
    weights_path,
    output_dir,
    image_size=512,
    matching_conf_thr=0.0,
    device=None,
    visualize=False,
    n_viz_pairs=5,
    n_viz_matches=20
):
    """
    Extract pixel correspondences from see3d novel views using the MASt3R model.
    Handles 512x512 images by splitting them into 384x512 images for processing.
    
    Args:
        novel_views_dir (str): Directory containing see3d inpainted novel views
        weights_path (str): Path to MASt3R model weights
        output_dir (str): Directory to save the pixel pairs
        image_size (int): Size to resize images for processing
        matching_conf_thr (float): Confidence threshold for matching
        device (torch.device): Device to run computation on (default: CUDA if available)
        visualize (bool): Whether to visualize the matches
        n_viz_pairs (int): Number of image pairs to visualize
        n_viz_matches (int): Number of matches to visualize per pair
    
    Returns:
        dict: Dictionary containing pixel correspondences between images
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all image files in the directory
    image_paths = sorted([
        os.path.join(novel_views_dir, f) for f in os.listdir(novel_views_dir)
        if f.endswith(('.png', '.jpg', '.jpeg'))
    ])
    print(f"Found {len(image_paths)} images in {novel_views_dir}")
    
    # Check if all images have the required size
    if not check_image_sizes(image_paths):
        raise ValueError("All input images must have size 512x512")
    
    # Split images into top and bottom parts
    split_mapping, all_split_paths = split_images(image_paths, output_dir)
    
    # Load MASt3R model
    print('Loading MASt3R model...')
    model = AsymmetricMASt3R.from_pretrained(weights_path).to(device)
    print('MASt3R model loaded.')
    
    # Load split images
    print('Loading split images...')
    split_imgs = load_images(all_split_paths, size=image_size, verbose=True)
    print(f'Loaded {len(split_imgs)} split images.')
    
    # Create pairs between original images (will be filled with processed matches)
    original_correspondences = {}
    for i in range(len(image_paths)):
        for j in range(i+1, len(image_paths)):
            img1_name = os.path.basename(image_paths[i])
            img2_name = os.path.basename(image_paths[j])
            pair_key = f"{img1_name}_{img2_name}"
            original_correspondences[pair_key] = {
                'img1': img1_name,
                'img2': img2_name,
                'kpts1': [],
                'kpts2': []
            }
    
    # Create all possible pairs of split images that come from different original images
    split_pairs = []
    split_pair_mapping = {}  # Maps split pair index to original pair
    
    for i, img1 in enumerate(split_imgs):
        img1_path = img1['instance']
        for j, img2 in enumerate(split_imgs):
            img2_path = img2['instance']
            
            # Find original images these splits come from
            orig_img1 = None
            orig_img2 = None
            img1_is_top = None
            img2_is_top = None

            orig_img1 = img1_path.replace('_top.png', '.png').replace('_bottom.png', '.png')
            orig_img2 = img2_path.replace('_top.png', '.png').replace('_bottom.png', '.png')
            img1_is_top = img1_path.endswith('_top.png')
            img2_is_top = img2_path.endswith('_top.png')
            
            # Only process pairs from different original images
            if orig_img1 and orig_img2 and orig_img1 != orig_img2:
                # Add to processing list if we haven't seen this pair of split images yet
                if j > i:
                    split_pairs.append((i, j))
                    
                    # Save mapping to original images
                    # orig_img1_name = os.path.basename(orig_img1)
                    # orig_img2_name = os.path.basename(orig_img2)
                    orig_img1_name = orig_img1
                    orig_img2_name = orig_img2
                    orig_pair_key = f"{orig_img1_name}_{orig_img2_name}"
                    if orig_img1_name > orig_img2_name:
                        orig_pair_key = f"{orig_img2_name}_{orig_img1_name}"
                        # Swap variables to maintain consistency
                        orig_img1, orig_img2 = orig_img2, orig_img1
                        img1_is_top, img2_is_top = img2_is_top, img1_is_top
                    
                    split_pair_mapping[len(split_pairs)-1] = {
                        'orig_pair_key': orig_pair_key,
                        'img1_is_top': img1_is_top,
                        'img2_is_top': img2_is_top
                    }
    
    print(f"Created {len(split_pairs)} pairs of split images to process")
    
    # Process each pair of split images    
    for idx, (idx1, idx2) in enumerate(tqdm(split_pairs, desc="Processing image pairs")):
        # Get pair data using MASt3R inference
        output = inference([(split_imgs[idx1], split_imgs[idx2])], model, device, batch_size=1, verbose=False)
        
        # Extract descriptors
        view1, pred1 = output['view1'], output['pred1']
        view2, pred2 = output['view2'], output['pred2']
        desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()
        
        # Find 2D-2D matches between the two images
        matches_im0, matches_im1 = fast_reciprocal_NNs(
            desc1, desc2, 
            subsample_or_initxy1=8,
            device=device, 
            dist='dot', 
            block_size=2**13
        )
        
        # Ignore small border around the edge
        H0, W0 = view1['true_shape'][0]
        valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
            matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)
        
        H1, W1 = view2['true_shape'][0]
        valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
            matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)
        
        valid_matches = valid_matches_im0 & valid_matches_im1
        matches_im0_np, matches_im1_np = matches_im0[valid_matches], matches_im1[valid_matches]
        
        # Get mapping information for this pair
        mapping_info = split_pair_mapping[idx]
        orig_pair_key = mapping_info['orig_pair_key']
        img1_is_top = mapping_info['img1_is_top']
        img2_is_top = mapping_info['img2_is_top']
        
        # Transform coordinates back to original image space
        if len(matches_im0) > 0:

            # Apply offsets for bottom images (y coordinate + 128)
            if not img1_is_top:  # if img1 is from the bottom part
                matches_im0_np[:, 1] += 128  # Offset: 512 - 384 = 128
            
            if not img2_is_top:  # if img2 is from the bottom part
                matches_im1_np[:, 1] += 128  # Offset: 512 - 384 = 128
            
            # Filter out matches in the overlap region if they're from the bottom image
            valid_indices = np.ones(len(matches_im0_np), dtype=bool)
            
            if not img1_is_top:
                # For bottom images, exclude matches in the overlap region (128-384 in y)
                overlap_mask1 = matches_im0_np[:, 1] < 384
                valid_indices = valid_indices & ~overlap_mask1
            
            if not img2_is_top:
                # For bottom images, exclude matches in the overlap region (128-384 in y)
                overlap_mask2 = matches_im1_np[:, 1] < 384
                valid_indices = valid_indices & ~overlap_mask2
            
            # Only keep valid matches (outside of the overlap)
            matches_im0_np = matches_im0_np[valid_indices]
            matches_im1_np = matches_im1_np[valid_indices]
            
            # Add to the original correspondences if we have any valid matches left
            if len(matches_im0_np) > 0:
                original_correspondences[orig_pair_key]['kpts1'].extend(matches_im0_np.tolist())
                original_correspondences[orig_pair_key]['kpts2'].extend(matches_im1_np.tolist())
    
    # Remove pairs with no matches
    original_correspondences = {k: v for k, v in original_correspondences.items() if len(v['kpts1']) > 0}
    
    # Save all correspondences
    output_file = os.path.join(output_dir, 'pixel_correspondences.json')
    with open(output_file, 'w') as f:
        json.dump(original_correspondences, f)
    
    print(f"Saved pixel correspondences to {output_file}")
    print(f"Found {len(original_correspondences)} image pairs with valid correspondences")
    
    # Visualize matches if requested
    if visualize:
        vis_pair = []
        corres_keys = list(original_correspondences.keys())
        random.shuffle(corres_keys)

        for idx in range(len(corres_keys)):
            orig_pair_key = corres_keys[idx]
            img1_name = original_correspondences[orig_pair_key]['img1']
            img2_name = original_correspondences[orig_pair_key]['img2']
            ori_img1_path = os.path.join(novel_views_dir, img1_name)
            ori_img2_path = os.path.join(novel_views_dir, img2_name)

            ori_img1 = exif_transpose(Image.open(ori_img1_path)).convert('RGB')
            ori_img2 = exif_transpose(Image.open(ori_img2_path)).convert('RGB')
            ori_img1 = ImgNorm(ori_img1)[None]
            ori_img2 = ImgNorm(ori_img2)[None]

            vis_pair.append({
                'view1': ori_img1,
                'view2': ori_img2,
                'matches_im0': np.array(original_correspondences[orig_pair_key]['kpts1']),
                'matches_im1': np.array(original_correspondences[orig_pair_key]['kpts2']),
                'img1_name': img1_name,
                'img2_name': img2_name,
            })

        visualize_matches(vis_pair, n_viz_matches, output_dir)
    
    return original_correspondences

def visualize_matches(vis_pairs, n_viz_matches=20, output_dir=None):
    """
    Visualize matches between image pairs.
    
    Args:
        vis_pairs (list): List of dicts containing match data
        n_viz_matches (int): Number of matches to visualize per pair
        output_dir (str): Directory to save visualizations
    """
    image_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
    image_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
    
    for i, pair_data in enumerate(vis_pairs):
        view1 = pair_data['view1']
        view2 = pair_data['view2']
        matches_im0 = pair_data['matches_im0']
        matches_im1 = pair_data['matches_im1']
        img1_name = pair_data['img1_name']
        img2_name = pair_data['img2_name']
        
        # Sample matches for visualization
        num_matches = matches_im0.shape[0]
        match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz_matches)).astype(int)
        viz_matches_im0 = matches_im0[match_idx_to_viz]
        viz_matches_im1 = matches_im1[match_idx_to_viz]
        
        # Prepare images for visualization
        viz_imgs = []
        for view in [view1, view2]:
            rgb_tensor = view * image_std + image_mean
            viz_imgs.append(rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())
        
        # Create combined image
        H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
        img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        img = np.concatenate((img0, img1), axis=1)
        
        # Plot matches
        plt.figure(figsize=(15, 10))
        plt.imshow(img)
        plt.title(f"Matches between {img1_name} and {img2_name}")
        cmap = plt.get_cmap('jet')
        for j in range(n_viz_matches):
            x0, y0 = viz_matches_im0[j]
            x1, y1 = viz_matches_im1[j]
            plt.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(j / (n_viz_matches - 1)), linewidth=1.5, markersize=8)
        
        # Save visualization
        if output_dir:
            viz_dir = os.path.join(output_dir, 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            plt.savefig(os.path.join(viz_dir, f"matches_{img1_name}_{img2_name}.png"), dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def get_matching_points_from_pixel_pairs(
    pixel_correspondences, 
    points3d_data, 
    output_dir=None
):
    """
    Get 3D point correspondences from pixel correspondences.
    
    Args:
        pixel_correspondences (dict): Pixel correspondences between images
        points3d_data (dict): 3D points data for each image
        output_dir (str): Directory to save the 3D point correspondences
    
    Returns:
        dict: Dictionary containing 3D point correspondences between images
    """
    point_matches = {}
    
    for pair_key, pair_data in tqdm(pixel_correspondences.items(), desc="Processing 3D correspondences"):
        img1_name = pair_data['img1']
        img2_name = pair_data['img2']
        kpts1 = np.array(pair_data['kpts1'])
        kpts2 = np.array(pair_data['kpts2'])
        
        # Get 3D points data for each image
        points1 = points3d_data.get(img1_name, None)
        points2 = points3d_data.get(img2_name, None)
        
        if points1 is None or points2 is None:
            continue
        
        # Extract 3D coordinates for each matching pixel
        point_pairs = []
        for (x1, y1), (x2, y2) in zip(kpts1, kpts2):
            # Convert to integers for indexing
            x1, y1 = int(round(x1)), int(round(y1))
            x2, y2 = int(round(x2)), int(round(y2))
            
            # Check if coordinates are within bounds
            if (0 <= x1 < points1.shape[1] and 0 <= y1 < points1.shape[0] and
                0 <= x2 < points2.shape[1] and 0 <= y2 < points2.shape[0]):
                
                # Get 3D points
                p1 = points1[y1, x1]
                p2 = points2[y2, x2]
                
                # Check if points are valid (not NaN or infinite)
                if (np.isfinite(p1).all() and np.isfinite(p2).all()):
                    point_pairs.append({
                        'point1': p1.tolist(),
                        'point2': p2.tolist(),
                        'pixel1': [float(x1), float(y1)],
                        'pixel2': [float(x2), float(y2)]
                    })
        
        if point_pairs:
            point_matches[pair_key] = {
                'img1': img1_name,
                'img2': img2_name,
                'point_pairs': point_pairs
            }
    
    # Save point correspondences if requested
    if output_dir and point_matches:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, '3d_point_correspondences.json')
        with open(output_file, 'w') as f:
            json.dump(point_matches, f)
        print(f"Saved 3D point correspondences to {output_file}")
    
    return point_matches


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract pixel correspondences from see3d novel views using MASt3R')
    
    # Input/Output
    parser.add_argument('-i', '--novel_views_dir', type=str, required=True,
                        help='Directory containing see3d inpainted novel views')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help='Directory to save the pixel pairs')
    
    # MASt3R model
    parser.add_argument('--weights_path', type=str, 
                        default='./mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth',
                        help='Path to the MASt3R model weights')
    
    # Parameters
    parser.add_argument('--image_size', type=int, default=512,
                        help='Size to resize images for processing')
    parser.add_argument('--matching_conf_thr', type=float, default=0.0,
                        help='Confidence threshold for matching')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID to use')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize matches between image pairs')
    parser.add_argument('--n_viz_pairs', type=int, default=5,
                        help='Number of image pairs to visualize')
    parser.add_argument('--n_viz_matches', type=int, default=20,
                        help='Number of matches to visualize per pair')
    
    args = parser.parse_args()
    
    # Set CUDA device
    torch.cuda.set_device(args.gpu)
    device = torch.device(torch.cuda.current_device())
    
    # Run the main function
    correspondences = get_pixel_pairs_from_novel_views(
        novel_views_dir=args.novel_views_dir,
        weights_path=args.weights_path,
        output_dir=args.output_dir,
        image_size=args.image_size,
        matching_conf_thr=args.matching_conf_thr,
        device=device,
        visualize=args.visualize,
        n_viz_pairs=args.n_viz_pairs,
        n_viz_matches=args.n_viz_matches
    )