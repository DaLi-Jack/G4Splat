import argparse
from pathlib import Path
import itertools
import torch
from tqdm import tqdm
import numpy as np

from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from dust3r.utils.image import load_images
from dust3r.inference import inference


def load_matches_as_tensors(npz_path):
    """
    Load matches from a .npz file and convert them to tensors.
    
    Args:
        npz_path: Path to the .npz file containing matches
        
    Returns:
        match_to_img: Tensor of shape (n_match, 2) containing image indices for each match
        match_to_pix: Tensor of shape (n_match, 2, 2) containing UV coordinates for each match
    """
    
    # Example usage:
    # match_to_img, match_to_pix = load_matches_as_tensors('matches.npz')
    # print(f"Number of matches: {len(match_to_img)}")
    # print(f"match_to_img shape: {match_to_img.shape}")  # (n_match, 2)
    # print(f"match_to_pix shape: {match_to_pix.shape}")  # (n_match, 2, 2)
    
    # Load the matches
    data = np.load(npz_path, allow_pickle=True)
    
    # Create a mapping from image names to indices
    all_images = sorted(list(set(
        name for pair in data.keys() 
        for name in pair.split('___')
    )))
    image_to_idx = {name: idx for idx, name in enumerate(all_images)}
    idx_to_image = {idx: name for idx, name in enumerate(all_images)}
    
    # Initialize lists to store matches
    all_img_indices = []
    all_pixels = []

    # Process each pair
    for pair_key, matches in data.items():
        img1_name, img2_name = pair_key.split('___')
        img1_idx = image_to_idx[img1_name]
        img2_idx = image_to_idx[img2_name]
        
        # Get matches for this pair
        matches_im0 = matches.item()['matches_im0']
        matches_im1 = matches.item()['matches_im1']
        n_matches = len(matches_im0)
        
        # Add image indices for each match
        pair_indices = np.array([[img1_idx, img2_idx]] * n_matches)
        all_img_indices.append(pair_indices)
        
        # Stack pixel coordinates
        pair_pixels = np.stack([matches_im0, matches_im1], axis=1)
        all_pixels.append(pair_pixels)

    # Concatenate all matches
    match_to_img = torch.from_numpy(np.concatenate(all_img_indices, axis=0))
    match_to_pix = torch.from_numpy(np.concatenate(all_pixels, axis=0))
    
    return match_to_img, match_to_pix, idx_to_image


def process_image_pairs(directory: str, checkpoint_path: str, device: str = 'cuda'):
    """
    Process all image pairs in a directory using MASt3R.
    
    Args:
        directory: Path to directory containing images
        checkpoint_path: Path to MASt3R checkpoint
        device: Device to run inference on
    """
    # Load model
    model = AsymmetricMASt3R.from_pretrained(checkpoint_path).to(device)
    
    # Get all image paths
    image_dir = Path(directory)
    image_files = sorted([f for f in image_dir.glob('*') 
                         if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    
    # Process all possible pairs
    results = {}
    for img1_path, img2_path in tqdm(list(itertools.combinations(image_files, 2))):
        # Load image pair
        images = load_images([str(img1_path), str(img2_path)], size=512, verbose=False)
        
        # Run inference
        output = inference([tuple(images)], model, device, batch_size=1, verbose=False)
        
        # Extract predictions
        view1, pred1 = output['view1'], output['pred1']
        view2, pred2 = output['view2'], output['pred2']
        desc1 = pred1['desc'].squeeze(0).detach()
        desc2 = pred2['desc'].squeeze(0).detach()
        
        # Find matches
        matches_im0, matches_im1 = fast_reciprocal_NNs(
            desc1, desc2, 
            subsample_or_initxy1=8,
            device=device, 
            dist='dot', 
            block_size=2**13
        )
        
        # Filter border matches
        H0, W0 = view1['true_shape'][0]
        H1, W1 = view2['true_shape'][0]
        
        valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
            matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)
        valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
            matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)
        
        valid_matches = valid_matches_im0 & valid_matches_im1
        matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]
        
        # Store results
        pair_key = f"{img1_path.name}___{img2_path.name}"
        results[pair_key] = {
            'matches_im0': matches_im0,
            'matches_im1': matches_im1,
        }
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Process image pairs using MASt3R')
    parser.add_argument('--directory', type=str, help='Directory containing images')
    parser.add_argument('--checkpoint', type=str, 
                      default='./checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth',
                      help='Path to MASt3R checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run inference on')
    parser.add_argument('--output', type=str, default='matches.npz', 
                      help='Output file to save matches')
    
    args = parser.parse_args()
    
    # Process all pairs
    results = process_image_pairs(args.directory, args.checkpoint, args.device)
    
    # Save results
    np.savez(args.output, **results)
    print(f"Results saved to {args.output}")

if __name__ == '__main__':
    main() 