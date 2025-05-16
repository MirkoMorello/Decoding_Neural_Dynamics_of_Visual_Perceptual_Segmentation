# scripts/generate_masks_SAM.py

import argparse
import os
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F # For interpolate if needed elsewhere, not directly here
from PIL import Image
from tqdm import tqdm
import yaml
import logging
import time # For simple timing

# SAM specific imports
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from torch.multiprocessing import spawn, set_start_method

# For AMP (Automatic Mixed Precision)
from torch import amp

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pysaliency # For loading stimuli lists

# Configure basic logging for the main process and workers
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(name)s (%(processName)s): %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__) # Main logger for the script

# --- Helper Functions for Building Mask Banks (Keep your existing ones) ---
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

# --- Postprocessing Function (Keep your existing one for now) ---
def postprocess_sam_masks_to_k_labels(
    sam_masks_data: list,
    image_shape_hw: tuple,
    num_target_labels: int = 64,
    background_label: int = 0,
    min_area_ratio: float = 0.001,
    iou_threshold_for_nms: float = 0.7
    ) -> np.ndarray:
    # ... (Your existing postprocess_sam_masks_to_k_labels implementation) ...
    # ... (No changes here for now, focus on parallelizing SAM inference first) ...
    if not sam_masks_data:
        return np.full(image_shape_hw, background_label, dtype=np.uint8)

    height, width = image_shape_hw
    total_pixels = height * width
    min_pixel_area = int(total_pixels * min_area_ratio)

    valid_masks = []
    for m_data in sam_masks_data:
        area = m_data['area']
        if area < min_pixel_area:
            continue
        valid_masks.append({
            'segmentation': m_data['segmentation'], 'area': area,
            'score': m_data.get('predicted_iou', m_data.get('stability_score', 0.0)),
            'bbox': m_data['bbox']
        })
    valid_masks.sort(key=lambda x: x['score'], reverse=True)

    selected_sam_masks_info = []
    for current_mask_info in valid_masks:
        current_binary_mask = current_mask_info['segmentation']
        is_redundant = False
        if iou_threshold_for_nms < 1.0:
            for selected_info in selected_sam_masks_info:
                selected_binary_mask = selected_info['segmentation']
                intersection = np.logical_and(current_binary_mask, selected_binary_mask).sum()
                if intersection == 0: continue
                union = np.logical_or(current_binary_mask, selected_binary_mask).sum()
                if union == 0: continue
                iou = intersection / union
                if iou > iou_threshold_for_nms:
                    is_redundant = True; break
        if not is_redundant:
            selected_sam_masks_info.append(current_mask_info)

    num_fg_labels_to_assign = min(len(selected_sam_masks_info), num_target_labels - (1 if background_label is not None and background_label==0 else 0) )
    
    final_label_map = np.full(image_shape_hw, background_label, dtype=np.uint8)
    current_label_id = background_label + 1 # Start foreground labels from 1 if background is 0

    for i in range(num_fg_labels_to_assign):
        mask_info = selected_sam_masks_info[i]
        binary_mask = mask_info['segmentation']
        final_label_map[binary_mask] = current_label_id
        current_label_id += 1
        if current_label_id >= 256: # Max for uint8
            logger.warning("Exceeded 255 labels for uint8 mask during postprocessing. Some segments will be merged or overwritten.")
            break
    return final_label_map

# --- Worker function for multi-GPU processing ---
def run_sam_mask_generation_worker(rank: int, world_size: int, config: dict):
    """
    Worker function for SAM mask generation on a specific GPU/shard.
    """
    # Configure device for this worker
    if config['device'] == 'cuda' and torch.cuda.is_available() and world_size > 0:
        device_id = rank # For multi-GPU, rank is the GPU index
        torch.cuda.set_device(device_id)
        device_str = f"cuda:{device_id}"
    else: # Fallback to CPU or if world_size is 1 (single process)
        device_str = "cpu"
        if config['device'] == 'cuda' and not torch.cuda.is_available():
            logger.warning(f"[Worker {rank}] CUDA specified but not available. Using CPU.")
        elif config['device'] == 'cuda' and world_size == 0: # Should not happen with spawn
             logger.warning(f"[Worker {rank}] CUDA specified but world_size is 0. Using CPU.")


    device = torch.device(device_str)
    logger.info(f"[Worker {rank}/{world_size}] Initialized on device: {device_str}")

    # 1. Load SAM Model (each worker loads its own instance)
    logger.info(f"[Worker {rank}] Loading SAM model: {config['sam_model_type']} from {config['sam_checkpoint_path']}...")
    try:
        sam_model = sam_model_registry[config['sam_model_type']](checkpoint=config['sam_checkpoint_path'])
        sam_model.to(device=device)
        # FP16 is handled by torch.amp.autocast during generate call, no sam_model.half() needed here
        
        # Optionally compile the model (PyTorch 2.0+)
        if config.get('compile_sam', False) and hasattr(torch, 'compile'):
            logger.info(f"[Worker {rank}] Compiling SAM model with torch.compile()...")
            try:
                sam_model = torch.compile(sam_model, mode=config.get('compile_mode', 'reduce-overhead'))
            except Exception as e_compile:
                logger.warning(f"[Worker {rank}] torch.compile(sam) failed: {e_compile}. Using uncompiled model.")

        mask_generator = SamAutomaticMaskGenerator(
            model=sam_model,
            points_per_side=config.get('sam_points_per_side', 32),
            pred_iou_thresh=config.get('sam_pred_iou_thresh', 0.88),
            stability_score_thresh=config.get('sam_stability_score_thresh', 0.95),
            crop_n_layers=config.get('sam_crop_n_layers', 0),
            crop_n_points_downscale_factor=config.get('sam_crop_n_points_downscale_factor', 1),
            min_mask_region_area=config.get('sam_min_mask_region_area', 0),
        )
    except Exception as e:
        logger.error(f"[Worker {rank}] Critical error loading SAM model or creating generator: {e}", exc_info=True)
        return # Worker cannot proceed
    logger.info(f"[Worker {rank}] SAM model and AutomaticMaskGenerator ready.")

    # 2. Load Full Stimuli List (each worker gets it, then shards)
    # This could be optimized by having rank 0 load and broadcast, but direct load is simpler for now.
    stimuli_paths_full_list = []
    try:
        if config['dataset_name'] == 'SALICON_train':
            stimuli_obj, _ = pysaliency.get_SALICON_train(location=config['stimuli_location'])
        elif config['dataset_name'] == 'SALICON_val':
            stimuli_obj, _ = pysaliency.get_SALICON_val(location=config['stimuli_location'])
        elif config['dataset_name'] == 'MIT1003_twosize': # Assumes this is a path to a dir containing stimuli.pkl
            mit_base_path = Path(config['stimuli_location'])
            if not mit_base_path.is_dir(): # If stimuli_location points directly to pkl or its parent
                 mit_base_path = mit_base_path.parent
            mit_converted_stimuli_path = mit_base_path / "MIT1003_twosize" / "stimuli.pkl"
            if not mit_converted_stimuli_path.exists():
                 logger.error(f"[Worker {rank}] MIT1003_twosize stimuli.pkl not found at {mit_converted_stimuli_path}"); return
            with open(mit_converted_stimuli_path, "rb") as f:
                 stimuli_obj = pysaliency.load_stimuli(f)
        else:
            logger.error(f"[Worker {rank}] Unknown dataset_name: {config['dataset_name']}"); return
        stimuli_paths_full_list = stimuli_obj.filenames
    except Exception as e:
        logger.error(f"[Worker {rank}] Error loading stimuli list: {e}", exc_info=True); return
    
    num_total_stimuli = len(stimuli_paths_full_list)
    if rank == 0: logger.info(f"Total stimuli to process across all workers: {num_total_stimuli}")

    # --- Shard the dataset for this worker ---
    if world_size > 1:
        stimuli_shard = stimuli_paths_full_list[rank::world_size]
    else: # Single process run
        stimuli_shard = stimuli_paths_full_list
    
    num_stimuli_this_worker = len(stimuli_shard)
    if num_stimuli_this_worker == 0:
        logger.info(f"[Worker {rank}] No stimuli assigned to this shard. Exiting worker peacefully.")
        return
    logger.info(f"[Worker {rank}] Assigned {num_stimuli_this_worker} stimuli for processing.")

    # 3. Prepare Output Directory (master/rank 0 creates, all assume it exists)
    mask_output_path = Path(config['output_dir'])
    if rank == 0:
        mask_output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Individual SAM-derived integer label masks will be saved to: {mask_output_path}")
    # Barrier to ensure directory is created before non-master workers proceed (if filesystem is shared and slow to sync)
    # For local multi-GPU, this might not be strictly necessary but is safer.
    # This requires initializing a temporary process group if not already in DDP context.
    # Simpler: assume filesystem ops are fast enough locally or handle FileNotFoundError.

    # 4. Process Shard of Images
    # Determine if tqdm progress bar should be active for this worker
    # Only rank 0 shows the main progress bar in a multi-worker setup to avoid clutter.
    # Other workers can have their own (silent or positioned) bars if needed for debugging.
    disable_tqdm_worker = (world_size > 1 and rank != 0)

    for i in tqdm(range(num_stimuli_this_worker), 
                  desc=f"Worker {rank} SAM Masks", 
                  position=rank, # For distinct progress bar lines if terminal supports
                  disable=disable_tqdm_worker):
        stimulus_filename_abs = stimuli_shard[i]
        img_load_start_time = time.time()
        try:
            image_bgr = cv2.imread(stimulus_filename_abs)
            if image_bgr is None:
                logger.warning(f"[Worker {rank}] cv2.imread failed for {stimulus_filename_abs}. Skipping.")
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_shape_hw = image_rgb.shape[:2]
        except Exception as img_load_e:
            logger.error(f"[Worker {rank}] Error loading image {stimulus_filename_abs}: {img_load_e}", exc_info=True); continue
        
        img_load_duration = time.time() - img_load_start_time

        base_fname = Path(stimulus_filename_abs).stem
        output_mask_file = mask_output_path / f"{base_fname}.{config['file_format']}"

        if output_mask_file.exists() and not config.get('overwrite_individual_masks', False):
            continue

        sam_gen_start_time = time.time()
        try:
            with torch.inference_mode():
                with amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(config.get('use_fp16_sam', False) and device.type=='cuda')):
                    sam_raw_masks = mask_generator.generate(image_rgb)
            sam_gen_duration = time.time() - sam_gen_start_time

            postproc_start_time = time.time()
            integer_label_mask_np = postprocess_sam_masks_to_k_labels(
                sam_raw_masks, image_shape_hw=image_shape_hw,
                num_target_labels=config.get('num_target_segments_from_sam', 64),
                background_label=config.get('sam_background_label', 0),
                min_area_ratio=config.get('sam_postprocess_min_area_ratio', 0.0005),
                iou_threshold_for_nms=config.get('sam_postprocess_iou_thresh_nms', 0.8)
            )
            postproc_duration = time.time() - postproc_start_time

            if integer_label_mask_np is None: # Should be handled by postprocess returning default
                logger.warning(f"[Worker {rank}] Postprocessing returned None for {base_fname}, saving dummy.")
                integer_label_mask_np = np.zeros(image_shape_hw, dtype=np.uint8)

            save_start_time = time.time()
            if config['file_format'] == 'png':
                Image.fromarray(integer_label_mask_np.astype(np.uint8), mode='L').save(output_mask_file)
            else: # npy
                np.save(output_mask_file, integer_label_mask_np.astype(np.uint8))
            save_duration = time.time() - save_start_time
            
            if i % 50 == 0 and rank == 0 : # Log timing occasionally for rank 0
                 logger.debug(f"[Worker {rank}] Timings for {base_fname}: ImgLoad={img_load_duration:.3f}s, SAMGen={sam_gen_duration:.3f}s, PostProc={postproc_duration:.3f}s, Save={save_duration:.3f}s")

        except Exception as e:
            logger.error(f"[Worker {rank}] Error processing {stimulus_filename_abs} with SAM: {e}", exc_info=True)

    logger.info(f"[Worker {rank}] Finished processing its shard of {num_stimuli_this_worker} images.")


if __name__ == '__main__':
    # Set start method for multiprocessing if using spawn, 'fork' can be problematic with CUDA
    # Do this once at the beginning of the main execution block.
    try:
        set_start_method('spawn', force=True) # 'spawn' is generally safer with CUDA
        logger.info("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        logger.warning("Could not set multiprocessing start method to 'spawn' (might be already set or unsupported).")


    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument('--config_file', type=str, default=None, help="Path to YAML config file.")
    _cfg_ns, _remaining_cli = _pre.parse_known_args()

    parser = argparse.ArgumentParser(parents=[_pre], description="Generate segmentation masks using SAM with multi-GPU support.")
    
    # --- Arguments (Copied from your script, with additions) ---
    # Stimuli and Output
    parser.add_argument('--stimuli_location', type=str, help="Path to dataset root (e.g., contains SALICON_train folder or MIT1003_twosize/stimuli.pkl).")
    parser.add_argument('--dataset_name', type=str, choices=['SALICON_train', 'SALICON_val', 'MIT1003_twosize'])
    parser.add_argument('--output_dir', type=str, help="Directory for individual SAM-derived integer masks.")
    parser.add_argument('--file_format', type=str, default='png', choices=['png', 'npy'])
    parser.add_argument('--overwrite_individual_masks', action='store_true')

    # SAM Model Configuration
    parser.add_argument('--sam_model_type', type=str, default='vit_b', choices=['vit_h', 'vit_l', 'vit_b'], help="SAM model architecture.")
    parser.add_argument('--sam_checkpoint_path', type=str, help="Path to SAM model checkpoint .pth file (REQUIRED if not in default cache path).")
    parser.add_argument('--device', type=str, default='cuda', help="Primary device ('cuda' or 'cpu'). For multi-GPU, 'cuda' implies using all available via spawn.")
    parser.add_argument('--use_fp16_sam', action='store_true', help="Enable FP16 inference for SAM using torch.amp.autocast.")
    parser.add_argument('--compile_sam', action='store_true', help="Enable torch.compile() for SAM model (PyTorch 2.0+).")
    parser.add_argument('--compile_mode', type=str, default='reduce-overhead', help="Mode for torch.compile().")


    # SAM AutomaticMaskGenerator Parameters
    parser.add_argument('--sam_points_per_side', type=int, default=32)
    parser.add_argument('--sam_pred_iou_thresh', type=float, default=0.88)
    parser.add_argument('--sam_stability_score_thresh', type=float, default=0.95)
    parser.add_argument('--sam_crop_n_layers', type=int, default=0, help="0 processes whole image; >0 uses multi-crop.")
    parser.add_argument('--sam_crop_n_points_downscale_factor', type=int, default=1)
    parser.add_argument('--sam_min_mask_region_area', type=int, default=0, help="SAM's internal min mask area filter (pixels). Default 0 in SAM, but often set higher.")

    # Postprocessing SAM Masks to K Labels
    parser.add_argument('--num_target_segments_from_sam', type=int, default=64,
                        help="Target K for the final integer mask (e.g., 64 segments).")
    parser.add_argument('--sam_background_label', type=int, default=0)
    parser.add_argument('--sam_postprocess_min_area_ratio', type=float, default=0.0005)
    parser.add_argument('--sam_postprocess_iou_thresh_nms', type=float, default=0.7) # NMS Threshold

    # Multi-GPU / Process Control
    parser.add_argument('--max_gpus', type=int, default=None, help="Maximum number of GPUs to use if device is 'cuda'. Uses all available by default.")

    # Mask Bank Building (Optional)
    parser.add_argument('--mask_bank_dtype', type=str, default='uint8', choices=['uint8', 'uint16'])
    parser.add_argument('--build_fixed_size_bank_output_file', type=str, default=None)
    parser.add_argument('--build_variable_size_bank_payload_file', type=str, default=None)
    parser.add_argument('--build_variable_size_bank_header_file', type=str, default=None)
    
    # --- YAML Config Loading & Final Parse ---
    cfg_from_yaml = {}
    if _cfg_ns.config_file:
        try:
            with open(_cfg_ns.config_file, 'r') as f: cfg_from_yaml = yaml.safe_load(f) or {}
            logger.info(f"📖 Loaded configuration from: {_cfg_ns.config_file}")
            parser.set_defaults(**cfg_from_yaml) # YAML values become defaults
        except Exception as e:
            logger.warning(f"⚠️ Could not read/parse YAML file '{_cfg_ns.config_file}': {e}. Using CLI/default args.")
    
    args = parser.parse_args(_remaining_cli) # CLI args override YAML/defaults
    final_config = vars(args)

    # --- Sanity Checks for Required Args ---
    _required_args = ('stimuli_location', 'dataset_name', 'output_dir') # sam_checkpoint_path handled by SAM itself if not found
    if not final_config.get('sam_checkpoint_path'):
         logger.warning("sam_checkpoint_path not specified. SAM will try to download/use cached if model type is standard.")
         # SAM might download if checkpoint path is a model name like "sam_vit_b", but explicit path is safer.

    missing_args = [k_req for k_req in _required_args if not final_config.get(k_req)]
    if missing_args:
        parser.error(f"Missing one or more required arguments: {', '.join(missing_args)}")

    logger.info("──────────── Effective SAM Mask Generation Configuration ────────────")
    for k_cfg, v_cfg in sorted(final_config.items()): logger.info(f"{k_cfg:40s}: {v_cfg}")
    logger.info("───────────────────────────────────────────────────────────────────────")

    # --- Determine World Size for Spawning ---
    world_size_spawn = 1 # Default to single process
    if final_config['device'] == 'cuda' and torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        if available_gpus == 0:
            logger.warning("CUDA specified, but no GPUs detected by PyTorch. Falling back to CPU.")
            final_config['device'] = 'cpu' # Override config
        else:
            gpus_to_use_spawn = available_gpus
            if final_config.get('max_gpus') is not None and final_config['max_gpus'] > 0:
                gpus_to_use_spawn = min(available_gpus, final_config['max_gpus'])
            world_size_spawn = gpus_to_use_spawn
    
    if world_size_spawn <= 0 : world_size_spawn = 1 # Ensure at least 1 process

    # --- Execute ---
    if world_size_spawn > 1 and final_config['device'] == 'cuda':
        logger.info(f"Spawning {world_size_spawn} worker processes for SAM mask generation on GPUs...")
        spawn(run_sam_mask_generation_worker,
              args=(world_size_spawn, final_config),
              nprocs=world_size_spawn,
              join=True)
    else: # Single process (CPU or 1 GPU)
        logger.info("Running SAM mask generation in a single process...")
        run_sam_mask_generation_worker(rank=0, world_size=1, config=final_config)

    logger.info("\nAll SAM mask generation workers finished (if spawned).")

    # --- Bank Building (runs in main process after workers complete) ---
    # Ensure paths in config are absolute or resolvable from CWD if bank building is enabled.
    # This part needs careful path management if workers wrote to a shared output_dir.
    # The current script assumes all workers write to the same config['output_dir'].
    if final_config.get('build_fixed_size_bank_output_file') or \
       (final_config.get('build_variable_size_bank_payload_file') and final_config.get('build_variable_size_bank_header_file')):
        
        logger.info("\nAttempting to build mask banks from generated individual masks...")
        # Reload full stimuli list for banking to ensure all are covered
        try:
            if final_config['dataset_name'] == 'SALICON_train':
                stimuli_obj_bank, _ = pysaliency.get_SALICON_train(location=final_config['stimuli_location'])
            elif final_config['dataset_name'] == 'SALICON_val':
                stimuli_obj_bank, _ = pysaliency.get_SALICON_val(location=final_config['stimuli_location'])
            elif final_config['dataset_name'] == 'MIT1003_twosize':
                mit_base_path_bank = Path(final_config['stimuli_location'])
                if not mit_base_path_bank.is_dir(): mit_base_path_bank = mit_base_path_bank.parent
                mit_pkl_bank = mit_base_path_bank / "MIT1003_twosize" / "stimuli.pkl"
                with open(mit_pkl_bank, "rb") as f_bank: stimuli_obj_bank = pysaliency.load_stimuli(f_bank)
            else: raise ValueError(f"Unknown dataset for banking: {final_config['dataset_name']}")
            all_stimuli_for_bank = stimuli_obj_bank.filenames
        except Exception as e_bank_stim:
            logger.error(f"Error loading stimuli list for bank building: {e_bank_stim}. Skipping bank creation.", exc_info=True)
            all_stimuli_for_bank = []

        if all_stimuli_for_bank:
            mask_dtype_bank = np.dtype(final_config.get('mask_bank_dtype', 'uint8'))
            # Source directory for masks is the same output_dir where workers saved them
            mask_source_dir_bank = Path(final_config['output_dir'])

            if fixed_bank_path_str := final_config.get('build_fixed_size_bank_output_file'):
                build_fixed_size_mask_bank(
                    stimuli_filenames=all_stimuli_for_bank, mask_source_dir=mask_source_dir_bank,
                    output_bank_file=Path(fixed_bank_path_str), mask_format=final_config['file_format'],
                    mask_dtype=mask_dtype_bank)
            
            if (var_payload_str := final_config.get('build_variable_size_bank_payload_file')) and \
               (var_header_str := final_config.get('build_variable_size_bank_header_file')):
                build_variable_size_mask_bank(
                    stimuli_filenames=all_stimuli_for_bank, mask_source_dir=mask_source_dir_bank,
                    output_payload_file=Path(var_payload_str), output_header_file=Path(var_header_str),
                    mask_format=final_config['file_format'], mask_dtype=mask_dtype_bank)
    else:
        logger.info("Mask bank building not configured or no stimuli available for banking.")

    logger.info("Script generate_masks_SAM.py finished.")