# generate_masks_Dinov2.py
import argparse
import os
from pathlib import Path
import math
import yaml # For YAML config

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import KMeans
from torchvision import transforms as pth_transforms
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dinov2_backbone import DinoV2Backbone

import pysaliency

# --- Helper Function from your script ---
def reconstruct_mask_from_patches(patch_labels, h_patches, w_patches, target_size_hw):
    mask_2d = patch_labels.reshape(h_patches, w_patches)
    mask_pil = Image.fromarray(mask_2d.astype(np.uint8), mode='L')
    mask_pil_resized = mask_pil.resize((target_size_hw[1], target_size_hw[0]), Image.NEAREST)
    return mask_pil_resized

# --- Helper Functions for Building Mask Banks (from previous discussion) ---
def build_fixed_size_mask_bank(
    stimuli_filenames: list[str],
    mask_source_dir: Path,
    output_bank_file: Path,
    mask_format: str = "png",
    mask_dtype: np.dtype = np.uint8
):
    if not stimuli_filenames:
        print("Error: No stimuli filenames provided for fixed-size bank.")
        return False
    if output_bank_file.exists():
        print(f"Fixed-size output file {output_bank_file} already exists. Skipping bank creation.")
        return True

    print(f"Building FIXED-SIZE mask bank for {len(stimuli_filenames)} stimuli...")
    first_stim_basename = Path(stimuli_filenames[0]).stem
    sample_mask_path = mask_source_dir / f"{first_stim_basename}.{mask_format}"
    if not sample_mask_path.exists():
        print(f"Error: Sample mask not found at {sample_mask_path} for fixed-size bank.")
        return False
    try:
        if mask_format == "png": sample_array = np.array(Image.open(sample_mask_path).convert('L'))
        elif mask_format == "npy": sample_array = np.load(sample_mask_path)
        else: raise ValueError(f"Unsupported mask format: {mask_format}")
    except Exception as e:
        print(f"Error loading sample mask {sample_mask_path}: {e}"); return False

    if sample_array.ndim != 2: print(f"Error: Expected 2D masks, sample has shape {sample_array.shape}"); return False
    H, W = sample_array.shape; N = len(stimuli_filenames)
    print(f"Fixed-size bank: N={N}, H={H}, W={W}, Dtype: {mask_dtype}")
    output_bank_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        mmap_array = np.memmap(output_bank_file, dtype=mask_dtype, mode='w+', shape=(N, H, W))
    except Exception as e: print(f"Error creating memmap file {output_bank_file}: {e}"); return False

    errors = 0
    for idx, stim_path_str in enumerate(tqdm(stimuli_filenames, desc="Building fixed-size bank")):
        mask_basename = Path(stim_path_str).stem
        current_mask_path = mask_source_dir / f"{mask_basename}.{mask_format}"
        try:
            if not current_mask_path.exists(): raise FileNotFoundError(f"Mask not found: {current_mask_path}")
            if mask_format == "png": mask_array = np.array(Image.open(current_mask_path).convert('L'))
            elif mask_format == "npy": mask_array = np.load(current_mask_path)
            if mask_array.shape != (H, W):
                print(f"\nError: Mask {current_mask_path} shape {mask_array.shape} != expected ({H},{W}). Zeroing.")
                mmap_array[idx] = np.zeros((H,W), dtype=mask_dtype); errors+=1; continue
            mmap_array[idx] = mask_array.astype(mask_dtype)
        except Exception as e:
            print(f"\nError processing {current_mask_path} for fixed bank: {e}. Zeroing.");
            mmap_array[idx] = np.zeros((H,W), dtype=mask_dtype); errors+=1
    mmap_array.flush(); del mmap_array
    if errors > 0: print(f"Warning: {errors} errors during fixed-size bank creation.")
    print(f"✅ Wrote {N-errors} masks to {output_bank_file} ({output_bank_file.stat().st_size / 1e6:.1f} MB)")
    return True

def build_variable_size_mask_bank(
    stimuli_filenames: list[str],
    mask_source_dir: Path,
    output_payload_file: Path,
    output_header_file: Path,
    mask_format: str = "png",
    mask_dtype: np.dtype = np.uint8
):
    if not stimuli_filenames: print("No stimuli filenames for variable-size bank."); return False
    if output_payload_file.exists() and output_header_file.exists():
        print(f"Variable-size output files {output_payload_file}, {output_header_file} exist. Skipping.")
        return True

    print(f"Building VARIABLE-SIZE mask bank for {len(stimuli_filenames)} stimuli...")
    num_masks = len(stimuli_filenames)
    header_data = np.zeros((num_masks, 3), dtype=np.int64) # [offset, H, W]
    current_offset = 0
    output_payload_file.parent.mkdir(parents=True, exist_ok=True)
    output_header_file.parent.mkdir(parents=True, exist_ok=True)
    errors = 0

    with open(output_payload_file, 'wb') as payload_f:
        for idx, stim_path_str in enumerate(tqdm(stimuli_filenames, desc="Building variable bank")):
            mask_basename = Path(stim_path_str).stem
            current_mask_path = mask_source_dir / f"{mask_basename}.{mask_format}"
            try:
                if not current_mask_path.exists(): raise FileNotFoundError(f"Mask not found: {current_mask_path}")
                if mask_format == "png": mask_array = np.array(Image.open(current_mask_path).convert('L'), dtype=mask_dtype)
                elif mask_format == "npy": mask_array = np.load(current_mask_path).astype(mask_dtype)
                else: raise ValueError(f"Unsupported format {mask_format}")

                H, W = mask_array.shape
                mask_bytes = mask_array.tobytes()
                header_data[idx] = [current_offset, H, W]
                payload_f.write(mask_bytes)
                current_offset += len(mask_bytes)
            except Exception as e:
                print(f"\nError processing {current_mask_path} for variable bank: {e}. Placeholder.");
                header_data[idx] = [current_offset, 0, 0]; errors+=1 # Still need to advance offset if planning to pack tightly
    np.save(output_header_file, header_data)
    if errors > 0: print(f"Warning: {errors} errors during variable-size bank creation.")
    print(f"✅ Wrote variable mask bank: {output_payload_file} ({output_payload_file.stat().st_size/1e6:.1f} MB), {output_header_file}")
    return True

# --- Main Script Logic ---
def run_mask_generation(config):
    print(f"Using device: {config['device']}")
    device = torch.device(config['device'])

    # 1. Load DinoV2Backbone
    print(f"Loading DinoV2Backbone: {config['dinov2_model_name']}, layers: {config['dinov2_feature_layers_indices']}...")
    try:
        dino_feature_extractor = DinoV2Backbone(
            layers=config['dinov2_feature_layers_indices'],
            model_name=config['dinov2_model_name'],
            patch_size=config['dinov2_patch_size'],
            freeze=True
        ).to(device)
        dino_feature_extractor.eval()
    except Exception as e:
        print(f"Error loading DinoV2Backbone: {e}"); return
    print("DinoV2Backbone loaded.")

    # 2. Load Stimuli List
    print(f"Loading stimuli for {config['dataset_name']} from {config['stimuli_location']}...")
    stimuli_paths_for_bank = [] # Store absolute paths for bank building
    try:
        # Assuming pysaliency.get_... functions return FileStimuli which have a .filenames attribute
        if config['dataset_name'] == 'SALICON_train':
            stimuli_obj, _ = pysaliency.get_SALICON_train(location=config['stimuli_location'])
        elif config['dataset_name'] == 'SALICON_val':
            stimuli_obj, _ = pysaliency.get_SALICON_val(location=config['stimuli_location'])
        elif config['dataset_name'] == 'MIT1003_twosize': # Path to your converted MIT1003 .pkl
            mit_converted_stimuli_path = Path(config['stimuli_location']) / "stimuli.pkl" # Assuming stimuli_location is the dir of the pkl
            if not mit_converted_stimuli_path.exists():
                 print(f"MIT1003_twosize stimuli.pkl not found at {mit_converted_stimuli_path}"); return
            with open(mit_converted_stimuli_path, "rb") as f:
                 stimuli_obj = pysaliency.load_stimuli(f) # pysaliency has its own load/save
        else:
            print(f"Unknown dataset_name: {config['dataset_name']}"); return
        
        stimuli_paths_for_bank = stimuli_obj.filenames # List of absolute or resolvable stimulus image paths
        num_stimuli = len(stimuli_paths_for_bank)
    except Exception as e:
        print(f"Error loading stimuli: {e}"); return
    print(f"Loaded {num_stimuli} stimuli.")

    # 3. Prepare Output Directory for individual masks
    mask_output_path = Path(config['output_dir'])
    mask_output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving individual masks to: {mask_output_path}")

    # 4. Process Each Image and Generate Individual Masks
    generated_stimuli_filenames = [] # Keep track of successfully processed stimuli for bank building

    for i in tqdm(range(num_stimuli), desc="Generating Individual Masks"):
        stimulus_filename_abs = stimuli_paths_for_bank[i]
        try:
            stimulus_pil = Image.open(stimulus_filename_abs)
            if stimulus_pil.mode != 'RGB': stimulus_pil = stimulus_pil.convert('RGB')
            original_size_wh = stimulus_pil.size
            img_tensor = pth_transforms.ToTensor()(stimulus_pil).unsqueeze(0).to(device)
        except Exception as img_load_e:
            print(f"Error loading image {stimulus_filename_abs}: {img_load_e}"); continue

        base_fname = Path(stimulus_filename_abs).stem
        if config['file_format'] == 'png':
            output_mask_file = mask_output_path / f"{base_fname}.png"
        else:
            output_mask_file = mask_output_path / f"{base_fname}.npy"

        if output_mask_file.exists() and not config.get('overwrite_individual_masks', False): # Add overwrite flag
            generated_stimuli_filenames.append(stimulus_filename_abs) # Assume existing is fine
            continue

        try:
            with torch.no_grad():
                list_of_feature_maps = dino_feature_extractor(img_tensor)
                if not list_of_feature_maps:
                    print(f"Warning: No DINO features for {base_fname}. Skipping."); continue
                feature_map_tensor = list_of_feature_maps[0] # Use the first (or only) specified layer

            B, C_layer, h_patches, w_patches = feature_map_tensor.shape
            patch_features_for_kmeans = feature_map_tensor.permute(0, 2, 3, 1).reshape(-1, C_layer)
            patch_features_np = patch_features_for_kmeans.cpu().numpy()

            if patch_features_np.shape[0] < config['num_segments']:
                print(f"Warning: Patches ({patch_features_np.shape[0]}) < num_segments ({config['num_segments']}) for {base_fname}. Dummy mask.")
                dummy_mask_arr = np.zeros((original_size_wh[1], original_size_wh[0]), dtype=np.uint8)
                if config['file_format'] == 'png': Image.fromarray(dummy_mask_arr, mode='L').save(output_mask_file)
                else: np.save(output_mask_file, dummy_mask_arr)
                generated_stimuli_filenames.append(stimulus_filename_abs) # Still add for bank completeness
                continue

            kmeans = KMeans(n_clusters=config['num_segments'], random_state=0, n_init='auto')
            patch_labels = kmeans.fit_predict(patch_features_np)
            mask_pil_resized = reconstruct_mask_from_patches(
                patch_labels, h_patches, w_patches,
                target_size_hw=(original_size_wh[1], original_size_wh[0])
            )

            if config['file_format'] == 'png': mask_pil_resized.save(output_mask_file)
            else: np.save(output_mask_file, np.array(mask_pil_resized))
            generated_stimuli_filenames.append(stimulus_filename_abs)

        except Exception as e:
            print(f"Error processing {stimulus_filename_abs}: {e}")
            # import traceback; traceback.print_exc() # Uncomment for detailed debug

    print("Individual mask generation complete.")

    # 5. Build Mask Banks if requested
    if generated_stimuli_filenames: # Only build if some masks were generated/found
        mask_dtype_for_bank = np.dtype(config.get('mask_bank_dtype', 'uint8')) # Default to uint8 for bank

        if config.get('build_fixed_size_bank_output_file'):
            fixed_bank_path = Path(config['build_fixed_size_bank_output_file'])
            print(f"\nAttempting to build fixed-size mask bank at: {fixed_bank_path}")
            build_fixed_size_mask_bank(
                stimuli_filenames=generated_stimuli_filenames,
                mask_source_dir=mask_output_path, # Source from where individual masks were just saved
                output_bank_file=fixed_bank_path,
                mask_format=config['file_format'], # Format of the *individual* masks
                mask_dtype=mask_dtype_for_bank
            )

        if config.get('build_variable_size_bank_payload_file') and config.get('build_variable_size_bank_header_file'):
            var_payload_path = Path(config['build_variable_size_bank_payload_file'])
            var_header_path = Path(config['build_variable_size_bank_header_file'])
            print(f"\nAttempting to build variable-size mask bank: Payload at {var_payload_path}, Header at {var_header_path}")
            build_variable_size_mask_bank(
                stimuli_filenames=generated_stimuli_filenames,
                mask_source_dir=mask_output_path,
                output_payload_file=var_payload_path,
                output_header_file=var_header_path,
                mask_format=config['file_format'],
                mask_dtype=mask_dtype_for_bank
            )
    else:
        print("No stimuli were processed successfully, skipping mask bank creation.")

    print("Script finished.")

# ──────────────────────────────────────────────────────────────────────────────
#  Main entry -- CLI + YAML handling (CLI overrides YAML)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    # ------------------------------------------------------------------
    # 1. Tiny “pre-parser” – we only want to grab --config_file here
    #    (Doing this avoids “conflicting option string” errors later.)
    # ------------------------------------------------------------------
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument(
        '--config_file', type=str, default=None,
        help='YAML configuration file.  Values inside become defaults; '
             'explicit CLI flags still win.'
    )
    _cfg, _remaining_cli = _pre.parse_known_args()   # _remaining_cli keeps *all*
                                                    # other CLI arguments

    # ------------------------------------------------------------------
    # 2. Full parser (inherits the --config_file flag from the pre-parser)
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        parents=[_pre],
        description=(
            "Generate segmentation masks using DinoV2 + K-Means and optionally "
            "package them into a mem-mapped mask bank."
        )
    )

    # ------------------------------------------------------------------
    # 2a.  FULL SET OF ARGUMENTS  (same as your script)                  |
    # ------------------------------------------------------------------
    # Individual-mask generation
    parser.add_argument('--stimuli_location', type=str,
                        help="Path to dataset root (e.g. SALICON base dir).")
    parser.add_argument('--dataset_name', type=str,
                        choices=['SALICON_train', 'SALICON_val',
                                 'MIT1003_twosize'],
                        help="Which dataset split to process.")
    parser.add_argument('--output_dir', type=str,
                        help="Directory that receives the individual mask files.")

    parser.add_argument('--dinov2_model_name', type=str,
                        default='dinov2_vitl14')
    parser.add_argument('--dinov2_patch_size', type=int, default=14)
    parser.add_argument('--dinov2_feature_layers_indices', type=int, nargs='+',
                        default=[-1],
                        help="Indices of DinoV2 transformer blocks to take "
                             "features from.")
    parser.add_argument('--num_segments', type=int, default=16,
                        help="K in K-Means.")
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--file_format', type=str, default='png',
                        choices=['png', 'npy'],
                        help="On-disk format for individual masks.")
    parser.add_argument('--image_batch_size', type=int, default=1)
    parser.add_argument('--overwrite_individual_masks', action='store_true',
                        help="Re-generate existing individual mask files.")

    # Mask-bank building
    parser.add_argument('--mask_bank_dtype', type=str, default='uint8',
                        choices=['uint8', 'uint16'],
                        help="dtype for mem-mapped mask bank.")
    parser.add_argument('--build_fixed_size_bank_output_file', type=str,
                        help="If set, create a fixed-size bank (.npy).")
    parser.add_argument('--build_variable_size_bank_payload_file', type=str,
                        help="If set, create variable-size bank payload (.bin).")
    parser.add_argument('--build_variable_size_bank_header_file', type=str,
                        help="Header (.npy) for variable-size bank.")

    # ------------------------------------------------------------------
    # 3.  Read YAML and make its entries the *defaults*
    # ------------------------------------------------------------------
    if _cfg.config_file:
        try:
            with open(_cfg.config_file, 'r') as f:
                yaml_cfg = yaml.safe_load(f) or {}
            print(f"📖  Loaded configuration from: {_cfg.config_file}")
            # Everything from YAML becomes the new default
            parser.set_defaults(**yaml_cfg)
        except Exception as e:
            print(f"⚠️  Could not read YAML file '{_cfg.config_file}': {e}")
            # we silently fall back to built-in defaults

    # ------------------------------------------------------------------
    # 4.  Final parse – CLI flags now beat YAML/defaults
    # ------------------------------------------------------------------
    args = parser.parse_args(_remaining_cli)
    cfg = vars(args)          # convert Namespace → dict for convenience

    # ------------------------------------------------------------------
    # 5.  Minimal sanity-checking
    # ------------------------------------------------------------------
    _required = ('stimuli_location', 'dataset_name', 'output_dir')
    missing = [k for k in _required if not cfg.get(k)]
    if missing:
        parser.error(f"Missing required argument(s): {', '.join(missing)}")

    # (Optional) pretty-print the resolved configuration on first run
    print("──────────── Effective configuration ────────────")
    for k, v in sorted(cfg.items()):
        print(f"{k:40s}: {v}")
    print("──────────────────────────────────────────────────")

    # ------------------------------------------------------------------
    # 6.  Execute main routine
    # ------------------------------------------------------------------
    run_mask_generation(cfg)
