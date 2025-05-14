import argparse
import os
from pathlib import Path
import math # Ensure math is imported

import numpy as np
import torch
import torch.nn.functional as F # Ensure F is imported
from PIL import Image
from sklearn.cluster import KMeans
from torchvision import transforms as pth_transforms # For preprocessing if needed
from tqdm import tqdm

# --- IMPORTANT: Adjust this import based on your project structure ---
# Assuming your DinoV2Backbone class is in src/backbones.py
# And your script is in scripts/, so you need to adjust sys.path or use relative imports if it's a package
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Add project root to path
from src.dinov2_backbone import DinoV2Backbone # Now you can import it

import pysaliency # For loading stimuli lists

# --- Helper Functions (No DINOv2 specific helpers needed here if using your class) ---

def reconstruct_mask_from_patches(patch_labels, h_patches, w_patches, target_size_hw):
    """
    Reconstructs a 2D segmentation mask from patch labels.

    Args:
        patch_labels (np.ndarray): 1D array of labels for each patch, shape (N_patches,).
        h_patches (int): Number of patches in height.
        w_patches (int): Number of patches in width.
        target_size_hw (tuple): Target (height, width) for the output mask.

    Returns:
        mask_pil (PIL.Image): The reconstructed segmentation mask as a PIL Image (mode 'L').
    """
    mask_2d = patch_labels.reshape(h_patches, w_patches)
    mask_pil = Image.fromarray(mask_2d.astype(np.uint8), mode='L')
    mask_pil_resized = mask_pil.resize((target_size_hw[1], target_size_hw[0]), Image.NEAREST)
    return mask_pil_resized

# --- Main Script Logic ---
def main():
    parser = argparse.ArgumentParser(description="Generate segmentation masks using your DinoV2Backbone and K-Means.")
    parser.add_argument('--stimuli_location', type=str, required=True,
                        help="Path to the dataset (e.g., SALICON or MIT1003 base directory for pysaliency).")
    parser.add_argument('--dataset_name', type=str, required=True, choices=['SALICON_train', 'SALICON_val', 'MIT1003_twosize'],
                        help="Name of the dataset to process (used for loading via pysaliency).")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory to save the generated masks.")
    parser.add_argument('--dinov2_model_name', type=str, default='dinov2_vitl14', # Match names in your backbone
                        help="DINOv2 model name used by DinoV2Backbone (e.g., dinov2_vitb14, dinov2_vitl14).")
    parser.add_argument('--dinov2_patch_size', type=int, default=14,
                        help="Patch size of the DINOv2 model (must match DinoV2Backbone).")
    parser.add_argument('--dinov2_feature_layers_indices', type=int, nargs='+', default=[-1],
                        help="List of layer indices for DinoV2Backbone (e.g., -1 for last, or -3 -2 -1). For K-Means, usually one layer's features are best.")
    parser.add_argument('--num_segments', type=int, default=16,
                        help="Number of segments (K) for K-Means clustering.")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Device to use (cuda or cpu).")
    parser.add_argument('--file_format', type=str, default='png', choices=['png', 'npy'],
                        help="Format to save the masks (png or npy).")
    parser.add_argument('--image_batch_size', type=int, default=1, # Process one image at a time for simplicity
                        help="Batch size for passing images through DINOv2 (usually 1 for this script).")


    args = parser.parse_args()

    print(f"Using device: {args.device}")
    device = torch.device(args.device)

    # 1. Load DinoV2Backbone
    print(f"Loading DinoV2Backbone with model: {args.dinov2_model_name}, layers: {args.dinov2_feature_layers_indices}...")
    try:
        # Instantiate your DinoV2Backbone. It will load from torch.hub.
        # Ensure it's frozen and in eval mode by default as per your class.
        dino_feature_extractor = DinoV2Backbone(
            layers=args.dinov2_feature_layers_indices,
            model_name=args.dinov2_model_name,
            patch_size=args.dinov2_patch_size,
            freeze=True # Explicitly set, though your class defaults to True
        ).to(device)
        dino_feature_extractor.eval() # Ensure it's in eval mode
    except Exception as e:
        print(f"Error loading DinoV2Backbone: {e}")
        # This might indicate torch.hub cannot download the model (network, firewall, or model name issue)
        return
    print("DinoV2Backbone loaded.")

    # 2. Load Stimuli List (same as before)
    print(f"Loading stimuli for {args.dataset_name} from {args.stimuli_location}...")
    try:
        if args.dataset_name == 'SALICON_train':
            stimuli, _ = pysaliency.get_SALICON_train(location=args.stimuli_location)
        elif args.dataset_name == 'SALICON_val':
            stimuli, _ = pysaliency.get_SALICON_val(location=args.stimuli_location)
        elif args.dataset_name == 'MIT1003_twosize':
            mit_converted_stimuli_path = Path(args.stimuli_location) / "MIT1003_twosize" / "stimuli.pkl"
            if not mit_converted_stimuli_path.exists():
                 print(f"MIT1003_twosize stimuli.pkl not found at {mit_converted_stimuli_path}")
                 return
            with open(mit_converted_stimuli_path, "rb") as f:
                 stimuli = torch.load(f) # Or pickle.load(f)
            if not isinstance(stimuli, pysaliency.FileStimuli):
                print("Error: Loaded MIT1003_twosize object is not a pysaliency.FileStimuli.")
                return
        else:
            print(f"Unknown dataset_name: {args.dataset_name}")
            return
    except Exception as e:
        print(f"Error loading stimuli: {e}")
        return
    print(f"Loaded {len(stimuli)} stimuli.")


    # 3. Prepare Output Directory (same as before)
    mask_output_path = Path(args.output_dir)
    mask_output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving masks to: {mask_output_path}")

    # Image preprocessing (minimal, as DinoV2Backbone handles padding and float conversion)
    # We just need to ensure images are RGB and in the B,C,H,W format torch expects.
    # DinoV2Backbone expects float input (0-1) or uint8.
    # Let's load as PIL and convert to tensor.
    # The torchvision transform here is mainly for ToTensor and potential normalization if
    # DinoV2Backbone didn't handle the uint8 -> float/255.0 part.
    # Your DinoV2Backbone's forward handles x.float().div_(255.0) if x.dtype == torch.uint8.

    # 4. Process Each Image
    for i in tqdm(range(len(stimuli)), desc="Generating Masks"):
        stimulus_filename_abs = stimuli.filenames[i]
        try:
            stimulus_pil = Image.open(stimulus_filename_abs)
            if stimulus_pil.mode != 'RGB':
                stimulus_pil = stimulus_pil.convert('RGB')
            original_size_wh = stimulus_pil.size # (width, height)

            # Convert PIL image to tensor (B, C, H, W)
            # Using a simple ToTensor which scales to [0,1] if PIL Image
            img_tensor = pth_transforms.ToTensor()(stimulus_pil).unsqueeze(0).to(device) # (1, C, H, W)

        except Exception as img_load_e:
            print(f"Error loading or preprocessing image {stimulus_filename_abs}: {img_load_e}")
            continue # Skip this image

        base_fname = Path(stimulus_filename_abs).stem
        if args.file_format == 'png':
            output_mask_file = mask_output_path / f"{base_fname}.png"
        else:
            output_mask_file = mask_output_path / f"{base_fname}.npy"

        if output_mask_file.exists():
            continue

        try:
            # A. Extract DINOv2 features using your DinoV2Backbone
            with torch.no_grad():
                # DinoV2Backbone returns a list of feature maps
                # For K-Means, we typically use features from one specific layer output,
                # or an average/concatenation if args.dinov2_feature_layers_indices has multiple.
                # Let's assume for K-Means we use the first one specified in the list.
                list_of_feature_maps = dino_feature_extractor(img_tensor) # List[(B, C, Hp, Wp)]
                
                if not list_of_feature_maps:
                    print(f"Warning: DinoV2Backbone returned no features for {base_fname}. Skipping.")
                    continue

                # Select the feature map (e.g., from the last specified layer)
                # If multiple layers are given to DinoV2Backbone, list_of_feature_maps will have multiple tensors.
                # For K-Means, usually one set of features is best.
                # We'll take the first one from the list returned by your backbone for simplicity.
                # This corresponds to the first layer index in args.dinov2_feature_layers_indices
                feature_map_tensor = list_of_feature_maps[0] # (B, C_layer, H_patch, W_patch)
                
            B, C_layer, h_patches, w_patches = feature_map_tensor.shape
            
            # Reshape for K-Means: (N_patches, C_layer)
            # N_patches = B * h_patches * w_patches
            # We process one image at a time, so B=1
            patch_features_for_kmeans = feature_map_tensor.permute(0, 2, 3, 1).reshape(-1, C_layer) # (N_patches, C_layer)
            patch_features_np = patch_features_for_kmeans.cpu().numpy()

            # B. K-Means Clustering
            if patch_features_np.shape[0] < args.num_segments:
                print(f"Warning: Number of patches ({patch_features_np.shape[0]}) is less than num_segments ({args.num_segments}) for {base_fname}. Creating dummy mask.")
                dummy_mask_arr = np.zeros((original_size_wh[1], original_size_wh[0]), dtype=np.uint8)
                if args.file_format == 'png': Image.fromarray(dummy_mask_arr, mode='L').save(output_mask_file)
                else: np.save(output_mask_file, dummy_mask_arr)
                continue

            kmeans = KMeans(n_clusters=args.num_segments, random_state=0, n_init='auto')
            patch_labels = kmeans.fit_predict(patch_features_np) # (N_patches,)

            # C. Reconstruct and Resize Mask
            mask_pil_resized = reconstruct_mask_from_patches(
                patch_labels, h_patches, w_patches,
                target_size_hw=(original_size_wh[1], original_size_wh[0]) # (height, width)
            )

            # D. Save Mask
            if args.file_format == 'png':
                mask_pil_resized.save(output_mask_file)
            else:
                np.save(output_mask_file, np.array(mask_pil_resized))

        except Exception as e:
            print(f"Error processing stimulus {stimulus_filename_abs} with DinoV2Backbone: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
            # Save dummy mask on error to avoid reprocessing
            dummy_mask_arr = np.zeros((original_size_wh[1], original_size_wh[0]), dtype=np.uint8)
            if args.file_format == 'png': Image.fromarray(dummy_mask_arr, mode='L').save(output_mask_file)
            else: np.save(output_mask_file, dummy_mask_arr)

    print("Mask generation complete.")

if __name__ == '__main__':
    main()