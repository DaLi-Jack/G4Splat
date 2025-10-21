import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import json
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from PIL.ImageOps import exif_transpose
import cv2
import torchvision.transforms as tvf

# Import necessary MASt3R modules - assuming these are available in your environment
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from dust3r.inference import inference

# Normalize image (same as in the provided code)
ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class ResizingMatcher:
    """
    A class that handles image resizing, MASt3R matching, and coordinate rescaling.
    """
    def __init__(self, weights_path, device=None):
        """
        Initialize the matching system with MASt3R model.
        
        Args:
            weights_path (str): Path to MASt3R model weights
            device (torch.device): Device to run computation on
        """
        # Set device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        print(f"Using device: {device}")
        
        # Load MASt3R model
        print('Loading MASt3R model...')
        self.model = AsymmetricMASt3R.from_pretrained(weights_path).to(device)
        print('MASt3R model loaded.')
        
        # Target size for MASt3R
        self.target_height = 384
        self.target_width = 512

    def resize_image(self, image_path):
        """
        Resize an image to 384x512 for MASt3R, tracking the scaling factors.
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            dict: Dictionary with resized image, original dimensions, and scaling factors
        """
        # Load image
        original_img = exif_transpose(Image.open(image_path)).convert('RGB')
        original_width, original_height = original_img.size
        
        # Resize to target size
        resized_img = original_img.resize((self.target_width, self.target_height), Image.BICUBIC)
        
        # Calculate scaling factors
        width_scale = original_width / self.target_width
        height_scale = original_height / self.target_height
        
        # Normalize for model input
        norm_img = ImgNorm(resized_img)[None]
        
        return {
            'img': norm_img,
            'true_shape': np.int32([resized_img.size[::-1]]),  # [H, W] format
            'idx': 0,
            'instance': os.path.basename(image_path),
            'original_size': (original_width, original_height),
            'scale_factors': (width_scale, height_scale)
        }

    def match_images(self, image_paths, output_dir, matching_conf_thr=0.0, visualize=False, n_viz_matches=20):
        """
        Match features between images, resizing as needed and rescaling results.
        
        Args:
            image_paths (list): List of paths to images
            output_dir (str): Directory to save results
            matching_conf_thr (float): Confidence threshold for matching
            visualize (bool): Whether to visualize matches
            n_viz_matches (int): Number of matches to visualize
            
        Returns:
            dict: Dictionary with pixel correspondences in original image coordinates
        """
        # Create output dictionary
        correspondences = {}
        
        # Process each image pair
        for i in range(len(image_paths)):
            for j in range(i+1, len(image_paths)):
                img1_path = image_paths[i]
                img2_path = image_paths[j]
                
                img1_name = os.path.basename(img1_path).split('.')[0]
                img2_name = os.path.basename(img2_path).split('.')[0]
                pair_key = f"{img1_name}_{img2_name}"
                
                print(f"Processing pair: {pair_key}")
                
                # Resize images for MASt3R
                img1_data = self.resize_image(img1_path)
                img2_data = self.resize_image(img2_path)
                
                # Perform inference
                output = inference([(img1_data, img2_data)], self.model, self.device, batch_size=1, verbose=False)
                
                # Extract descriptors
                view1, pred1 = output['view1'], output['pred1']
                view2, pred2 = output['view2'], output['pred2']
                desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()
                
                # Find 2D-2D matches between the two images
                matches_im0, matches_im1 = fast_reciprocal_NNs(
                    desc1, desc2, 
                    subsample_or_initxy1=8,
                    device=self.device, 
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
                
                # Scale matches back to original dimensions
                width_scale1, height_scale1 = img1_data['scale_factors']
                width_scale2, height_scale2 = img2_data['scale_factors']
                
                # Apply scaling to coordinates
                if len(matches_im0_np) > 0:
                    # Scale x and y coordinates separately
                    orig_matches_im0 = matches_im0_np.copy()
                    orig_matches_im0[:, 0] = (width_scale1 * matches_im0_np[:, 0]).astype(np.int64)  # x coordinates
                    orig_matches_im0[:, 1] = (height_scale1 * matches_im0_np[:, 1]).astype(np.int64)  # y coordinates
                    
                    orig_matches_im1 = matches_im1_np.copy()
                    orig_matches_im1[:, 0] = (width_scale2 * matches_im1_np[:, 0]).astype(np.int64)  # x coordinates
                    orig_matches_im1[:, 1] = (height_scale2 * matches_im1_np[:, 1]).astype(np.int64)  # y coordinates
                    
                    # Store the correspondences
                    correspondences[pair_key] = {
                        'img1': img1_name,
                        'img2': img2_name,
                        'kpts1': orig_matches_im0.tolist(),
                        'kpts2': orig_matches_im1.tolist(),
                        'original_size1': img1_data['original_size'],
                        'original_size2': img2_data['original_size']
                    }
                    
                    print(f"Found {len(orig_matches_im0)} matches for {pair_key}")
                else:
                    print(f"No matches found for {pair_key}")
                
                # Visualize if requested
                if visualize and len(matches_im0_np) > 0:
                    self.visualize_original_matches(
                        img1_path, img2_path,
                        orig_matches_im0, orig_matches_im1,
                        n_viz_matches,
                        output_dir
                    )
        
        return correspondences
    
    def visualize_original_matches(self, img1_path, img2_path, matches_im0, matches_im1, n_viz_matches=20, output_dir=None):
        """
        Visualize matches between original images.
        
        Args:
            img1_path (str): Path to first image
            img2_path (str): Path to second image
            matches_im0 (np.ndarray): Array of coordinates in first image
            matches_im1 (np.ndarray): Array of coordinates in second image
            n_viz_matches (int): Number of matches to visualize
        """
        # Load original images
        img1 = np.array(Image.open(img1_path).convert('RGB'))
        img2 = np.array(Image.open(img2_path).convert('RGB'))

        img1_name = os.path.basename(img1_path).split('.')[0]
        img2_name = os.path.basename(img2_path).split('.')[0]
        
        # Sample matches for visualization
        num_matches = matches_im0.shape[0]
        if num_matches > n_viz_matches:
            match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz_matches)).astype(int)
            viz_matches_im0 = matches_im0[match_idx_to_viz]
            viz_matches_im1 = matches_im1[match_idx_to_viz]
        else:
            viz_matches_im0 = matches_im0
            viz_matches_im1 = matches_im1
        
        # Create combined image
        H0, W0, _ = img1.shape
        H1, W1, _ = img2.shape
        H_max = max(H0, H1)
        
        img0 = np.pad(img1, ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        img1 = np.pad(img2, ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        img = np.concatenate((img0, img1), axis=1)
        
        # Plot matches
        plt.figure(figsize=(15, 10))
        plt.imshow(img)
        plt.title(f"Matches between original images")
        cmap = plt.get_cmap('jet')
        for j in range(len(viz_matches_im0)):
            x0, y0 = viz_matches_im0[j]
            x1, y1 = viz_matches_im1[j]
            plt.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(j / (len(viz_matches_im0) - 1 or 1)), 
                     linewidth=1.5, markersize=8)
        
        # Save visualization
        if output_dir:
            viz_dir = os.path.join(output_dir, 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            plt.savefig(os.path.join(viz_dir, f"matches_{img1_name}_{img2_name}.png"), dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def match_and_rescale_images(
    image_paths,
    weights_path,
    output_dir=None,
    matching_conf_thr=0.0,
    device=None,
    visualize=False,
    n_viz_matches=20
):
    """
    Main function to match images and rescale coordinates.
    
    Args:
        image_paths (list): List of paths to images
        weights_path (str): Path to MASt3R model weights
        output_dir (str): Directory to save results
        matching_conf_thr (float): Confidence threshold for matching
        device (torch.device): Device to run computation on
        visualize (bool): Whether to visualize matches
        n_viz_matches (int): Number of matches to visualize
        
    Returns:
        dict: Dictionary with pixel correspondences in original image coordinates
    """
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the matcher
    matcher = ResizingMatcher(weights_path, device)
    
    # Match images and rescale
    correspondences = matcher.match_images(
        image_paths,
        output_dir,
        matching_conf_thr=matching_conf_thr,
        visualize=visualize,
        n_viz_matches=n_viz_matches
    )
    
    # Save correspondences
    if output_dir and correspondences:
        output_file = os.path.join(output_dir, 'pixel_correspondences.json')
        with open(output_file, 'w') as f:
            json.dump(correspondences, f)
        print(f"Saved pixel correspondences to {output_file}")
    
    return correspondences

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Match features between images with resizing')
    parser.add_argument('-i', '--image_dir', type=str, required=True,
                        help='Directory containing input images')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help='Directory to save the results')
    parser.add_argument('--weights_path', type=str, 
                        default='./mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth',
                        help='Path to the MASt3R model weights')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID to use')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize matches between image pairs')
    
    args = parser.parse_args()
    
    # Get image paths
    image_paths = sorted([
        os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.tif'))
    ])
    
    # Set CUDA device
    torch.cuda.set_device(args.gpu)
    device = torch.device(torch.cuda.current_device())
    
    # Run the matching function
    correspondences = match_and_rescale_images(
        image_paths=image_paths,
        weights_path=args.weights_path,
        output_dir=args.output_dir,
        device=device,
        visualize=args.visualize
    )