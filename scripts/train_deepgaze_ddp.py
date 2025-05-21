#!/usr/bin/env python
"""
Multi-GPU-ready training script for the original DeepGaze III with DenseNet-201.
Based on the DINOv2 and DeepGaze SPADE training script structures.
"""
import os
import sys

# Add project root to sys.path for local module imports
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import yaml
import argparse
import logging
from pathlib import Path
from collections import OrderedDict
import pickle # For baseline LL caching

import torch
import torch.nn as nn
import torch.nn.functional as F # Not directly used here, but good for general PyTorch scripts
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# Pysaliency and dataset related
import pysaliency
import pysaliency.external_datasets.mit 
from PIL import Image
Image.MAX_IMAGE_PIXELS = None 
from pysaliency.baseline_utils import BaselineModel, CrossvalidatedBaselineModel
import cloudpickle as cpickle # For robust serialization
from tqdm import tqdm
from boltons.fileutils import atomic_save # For safe saving

# --- Local Project Imports ---
try:
    # Data handling classes
    from src.data import (
        ImageDataset, FixationDataset, FixationMaskTransform, ImageDatasetSampler,
        convert_stimuli, convert_fixation_trains # For MIT data preprocessing
    )
    # Original DeepGaze modules and layers
    from src.modules import DeepGazeIII, FeatureExtractor
    from src.layers import (
        Bias, LayerNorm, LayerNormMultiInput,
        Conv2dMultiInput, FlexibleScanpathHistoryEncoding
    )
    # DenseNet backbone
    from DeepGaze.deepgaze_pytorch.features.densenet import RGBDenseNet201

    # Metrics and Training loop
    from src.metrics import log_likelihood, nss, auc # auc might be auc_cpu_fn in other scripts
    from src.training import (_train)

except ImportError as e:
    actual_error_message = str(e)
    print(f"PYTHON IMPORT ERROR: {actual_error_message}")
    print(f"Current sys.path: {sys.path}")
    print("Ensure 'src' is in sys.path, contains __init__.py, and all required .py files with correct internal imports.")
    print("Ensure 'DeepGaze' (deepgaze_pytorch submodule/library) is in PYTHONPATH or accessible.")
    sys.exit(1)

# --- Logging Setup (Configured properly in main() after DDP init) ---
_logger = logging.getLogger("train_deepgaze_original_ddp")

# --- Distributed Utils (Copied from DINOv2/SPADE script) ---
def init_distributed() -> tuple[torch.device, int, int, bool, bool]:
    """Initialise torch.distributed (NCCL) if environment variables are present."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_distributed = world_size > 1

    if is_distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device(f"cuda:{local_rank}")
        is_master = rank == 0
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_master = True
    return device, rank, world_size, is_master, is_distributed

def cleanup_distributed():
    """Clean up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()

# ============================================================================
# == Original DeepGaze III Component Builders (from Notebook Reference) ==
# ============================================================================

def build_saliency_network(input_channels):
    return nn.Sequential(OrderedDict([
        ('layernorm0', LayerNorm(input_channels)),
        ('conv0', nn.Conv2d(input_channels, 8, (1, 1), bias=False)),
        ('bias0', Bias(8)),
        ('softplus0', nn.Softplus()),

        ('layernorm1', LayerNorm(8)),
        ('conv1', nn.Conv2d(8, 16, (1, 1), bias=False)),
        ('bias1', Bias(16)),
        ('softplus1', nn.Softplus()),

        ('layernorm2', LayerNorm(16)),
        ('conv2', nn.Conv2d(16, 1, (1, 1), bias=False)),
        ('bias2', Bias(1)),
        ('softplus2', nn.Softplus()), # Ensures non-negative before finalizer
    ]))


def build_scanpath_network():
    # Based on notebook: 4 history fixations, 3 channels per fixation (xy, t/duration-related)
    # Output 128 channels, then reduced to 16.
    return nn.Sequential(OrderedDict([
        ('encoding0', FlexibleScanpathHistoryEncoding(in_fixations=4, channels_per_fixation=3, out_channels=128, kernel_size=(1, 1), bias=True)),
        ('softplus0', nn.Softplus()),

        ('layernorm1', LayerNorm(128)),
        ('conv1', nn.Conv2d(128, 16, (1, 1), bias=False)),
        ('bias1', Bias(16)),
        ('softplus1', nn.Softplus()),
    ]))


def build_fixation_selection_network(scanpath_features=16):
    """ Builds the network combining saliency and scanpath features for fixation selection. """
    saliency_channels = 1 # Output of saliency network's core path before combination
    in_channels_list = [saliency_channels, scanpath_features if scanpath_features > 0 else 0]

    return nn.Sequential(OrderedDict([
        ('layernorm0', LayerNormMultiInput(in_channels_list)), # Now gets [1, 0] or [1, 16]
        ('conv0', Conv2dMultiInput(in_channels_list, 128, (1, 1), bias=False)), # Now gets [1, 0] or [1, 16]
        ('bias0', Bias(128)),
        ('softplus0', nn.Softplus()),
        ('layernorm1', LayerNorm(128)),
        ('conv1', nn.Conv2d(128, 16, (1, 1), bias=False)),
        ('bias1', Bias(16)),
        ('softplus1', nn.Softplus()),
        ('conv2', nn.Conv2d(16, 1, (1, 1), bias=False)), # Final output layer
    ]))

# ============================================================================
# == Data Preparation Helpers (Adapted from DINOgaze script for DDP) ========
# ============================================================================

def prepare_spatial_dataset_ddp(stimuli, fixations, centerbias_model, batch_size, num_workers,
                                is_distributed, is_master, device_for_ops, # device_for_ops not used here but kept for signature
                                lmdb_path_str, logger_instance, average='image'):
    if lmdb_path_str and is_master: # Create LMDB cache dir only on master
        Path(lmdb_path_str).mkdir(parents=True, exist_ok=True)
    if is_distributed:
        dist.barrier() # Ensure master creates dir before others might try to access

    dataset = ImageDataset(
        stimuli=stimuli,
        fixations=fixations,
        centerbias_model=centerbias_model,
        transform=FixationMaskTransform(sparse=False), # For spatial density maps
        average=average,
        lmdb_path=str(lmdb_path_str) if lmdb_path_str else None,
        # Additional args like cache_size could be added if ImageDataset supports them
    )
    
    if is_distributed:
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
    else:
        # For single GPU, ImageDatasetSampler is good if available and compatible,
        # otherwise standard shuffle in DataLoader.
        # The notebook used ImageDatasetSampler. Let's assume it's fine for non-DDP.
        # If ImageDatasetSampler is not DDP-aware, use regular shuffle.
        sampler = ImageDatasetSampler(dataset, batch_size=batch_size) if not is_distributed else None

    if sampler and hasattr(sampler, 'set_epoch') and is_distributed: # For DDP sampler
         logger_instance.debug("DDP sampler detected, set_epoch will be handled by _train loop.")
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size if not is_distributed else batch_size, # sampler handles batching in notebook version
        sampler=sampler if is_distributed else None, # DDP sampler
        batch_sampler=sampler if not is_distributed else None, # ImageDatasetSampler if not DDP
        shuffle=False if sampler else True, # Shuffle if no sampler (e.g. non-DDP without ImageDatasetSampler)
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_distributed, # Drop last for DDP to ensure even batch sizes
        persistent_workers=num_workers > 0
    )
    if is_master:
        logger_instance.info(f"Spatial Dataloader: {len(loader)} batches. LMDB: '{lmdb_path_str}'. Distributed: {is_distributed}. Sampler: {type(sampler).__name__ if sampler else 'Default/Shuffle'}")
    return loader


def prepare_scanpath_dataset_ddp(stimuli, fixations, centerbias_model, batch_size, num_workers,
                                 is_distributed, is_master, device_for_ops, # device_for_ops not used here
                                 lmdb_path_str, logger_instance, included_fixations, average='image'):
    if lmdb_path_str and is_master:
        Path(lmdb_path_str).mkdir(parents=True, exist_ok=True)
    if is_distributed:
        dist.barrier()

    dataset = FixationDataset(
        stimuli=stimuli,
        fixations=fixations, # These are FixationTrains for FixationDataset
        centerbias_model=centerbias_model,
        included_fixations=included_fixations, # e.g. [-1, -2, -3, -4]
        allow_missing_fixations=True,
        transform=FixationMaskTransform(sparse=False), # Target is still a density map
        average=average,
        lmdb_path=str(lmdb_path_str) if lmdb_path_str else None,
    )

    if is_distributed:
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
    else:
        sampler = ImageDatasetSampler(dataset, batch_size=batch_size) if not is_distributed else None

    if sampler and hasattr(sampler, 'set_epoch') and is_distributed:
        logger_instance.debug("DDP sampler detected, set_epoch will be handled by _train loop.")

    loader = DataLoader(
        dataset,
        batch_size=batch_size if not is_distributed else batch_size,
        sampler=sampler if is_distributed else None,
        batch_sampler=sampler if not is_distributed else None,
        shuffle=False if sampler else True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_distributed,
        persistent_workers=num_workers > 0
    )
    if is_master:
        logger_instance.info(f"Scanpath Dataloader: {len(loader)} batches. LMDB: '{lmdb_path_str}'. Distributed: {is_distributed}. Sampler: {type(sampler).__name__ if sampler else 'Default/Shuffle'}")
    return loader

# ============================================================================
# == MAIN FUNCTION ==
# ============================================================================
def main(args: argparse.Namespace):
    device, rank, world_size, is_master, is_distributed = init_distributed()

    # --- Logging Setup ---
    log_level_str = args.log_level.upper() if args.log_level else "INFO"
    log_level = getattr(logging, log_level_str, logging.INFO)
    if not is_master and log_level < logging.WARNING: # Non-masters more quiet
        log_level = logging.WARNING
        
    logging.basicConfig(level=log_level, format=f"%(asctime)s Rank{rank} %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", force=True)
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING) 
    _logger.setLevel(log_level)

    if is_master:
        _logger.info("================== Effective Configuration ==================")
        for arg_name, arg_value in sorted(vars(args).items()):
            _logger.info(f"  {arg_name}: {arg_value}")
        _logger.info(f"  DDP Info: Rank {rank}/{world_size}, Master: {is_master}, Distributed: {is_distributed}, Device: {device}")
        _logger.info(f"  Torch: {torch.__version__}, CUDA: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
        _logger.info("===========================================================")

    # --- Path Setup ---
    dataset_directory = Path(args.dataset_dir).resolve()
    train_output_base_dir = Path(args.train_dir).resolve()
    lmdb_image_cache_dir = Path(args.lmdb_dir).resolve()

    if is_master:
        for p in [dataset_directory, train_output_base_dir, lmdb_image_cache_dir]:
            p.mkdir(parents=True, exist_ok=True)
    if is_distributed: dist.barrier()

    # --- Backbone Setup: DenseNet-201 ---
    if is_master: _logger.info("Initializing DenseNet-201 backbone...")
    densenet_base_model = RGBDenseNet201() # Already pretrained on ImageNet by default
    
    # Hooks from the original DeepGaze III notebook for DenseNet-201
    densenet_feature_nodes = [
        '1.features.denseblock4.denselayer32.norm1',
        '1.features.denseblock4.denselayer32.conv1',
        '1.features.denseblock4.denselayer31.conv2',
    ]
    features_module = FeatureExtractor(densenet_base_model, densenet_feature_nodes)
    
    # Freeze the backbone
    for param in features_module.parameters(): param.requires_grad = False
    features_module.eval().to(device) # Set to eval mode and move to device
    
    # Input channels to saliency network from these hooks is 2048 for DenseNet-201
    # This was confirmed in thought process. C_in = 1888+128+32 = 2048
    saliency_network_input_channels = 2048
    if is_master: 
        _logger.info(f"DenseNet-201 with FeatureExtractor (hooks: {densenet_feature_nodes}) initialized and frozen.")
        _logger.info(f"Input channels to saliency network: {saliency_network_input_channels}")

    # Constants from notebook
    READOUT_FACTOR = 4
    SALIENCY_MAP_FACTOR = 4 # For Finalizer

    # ======================= STAGE DISPATCH ============================
    if args.stage == 'salicon_pretrain':
        current_lr = args.lr_salicon
        current_downsample = args.downsample_salicon
        current_milestones = args.lr_milestones_salicon
        output_dir_stage = train_output_base_dir / 'salicon_pretraining_original_densenet'
        
        if is_master: _logger.info(f"--- Preparing SALICON Pretraining (Original DeepGaze III) ---")

        if is_master: _logger.info(f"Loading SALICON data from {dataset_directory}...")
        # Ensure SALICON is downloaded if not present
        if is_master:
            try:
                if not (dataset_directory / 'SALICON' / 'stimuli' / 'train').exists():
                    pysaliency.get_SALICON_train(location=str(dataset_directory))
                if not (dataset_directory / 'SALICON' / 'stimuli' / 'val').exists():
                    pysaliency.get_SALICON_val(location=str(dataset_directory))
            except Exception as e:
                _logger.critical(f"Failed to download/access SALICON data: {e}")
                if is_distributed: dist.barrier()
                sys.exit(1)
        if is_distributed: dist.barrier()
        
        try:
            SALICON_train_stimuli, SALICON_train_fixations = pysaliency.get_SALICON_train(location=str(dataset_directory))
            SALICON_val_stimuli, SALICON_val_fixations = pysaliency.get_SALICON_val(location=str(dataset_directory))
        except Exception as e:
            _logger.critical(f"Failed to load SALICON metadata: {e}")
            if is_distributed: dist.barrier()
            sys.exit(1)
        if is_master: _logger.info("SALICON data loaded.")

        if is_master: _logger.info("Initializing SALICON BaselineModel for centerbias...")
        SALICON_centerbias = BaselineModel(stimuli=SALICON_train_stimuli, fixations=SALICON_train_fixations, bandwidth=0.0217, eps=2e-13, caching=False)
        
        train_ll_cache_file = dataset_directory / 'salicon_baseline_train_ll_dg_original.pkl'
        val_ll_cache_file = dataset_directory / 'salicon_baseline_val_ll_dg_original.pkl'
        train_baseline_log_likelihood, val_baseline_log_likelihood = None, None

        if is_master:
            try:
                with open(train_ll_cache_file, 'rb') as f: train_baseline_log_likelihood = cpickle.load(f)
                _logger.info(f"Loaded cached train baseline LL from: {train_ll_cache_file}")
            except Exception:
                _logger.warning(f"Train LL cache miss ({train_ll_cache_file}). Computing...");
                train_baseline_log_likelihood = SALICON_centerbias.information_gain(SALICON_train_stimuli, SALICON_train_fixations, verbose=True, average='image')
                with open(train_ll_cache_file, 'wb') as f: cpickle.dump(train_baseline_log_likelihood, f)
            try:
                with open(val_ll_cache_file, 'rb') as f: val_baseline_log_likelihood = cpickle.load(f)
                _logger.info(f"Loaded cached val baseline LL from: {val_ll_cache_file}")
            except Exception:
                _logger.warning(f"Val LL cache miss ({val_ll_cache_file}). Computing...");
                val_baseline_log_likelihood = SALICON_centerbias.information_gain(SALICON_val_stimuli, SALICON_val_fixations, verbose=True, average='image')
                with open(val_ll_cache_file, 'wb') as f: cpickle.dump(val_baseline_log_likelihood, f)
            _logger.info(f"Master Baseline LLs - Train: {train_baseline_log_likelihood or float('nan'):.5f}, Val: {val_baseline_log_likelihood or float('nan'):.5f}")

        ll_bcast = [train_baseline_log_likelihood, val_baseline_log_likelihood]
        if is_distributed:
            if is_master and (train_baseline_log_likelihood is None or val_baseline_log_likelihood is None): ll_bcast = [float('nan'), float('nan')] # Should not happen with try/except
            dist.broadcast_object_list(ll_bcast, src=0)
        train_baseline_log_likelihood, val_baseline_log_likelihood = ll_bcast
        if train_baseline_log_likelihood is None or val_baseline_log_likelihood is None or \
           (isinstance(train_baseline_log_likelihood, float) and torch.isnan(torch.tensor(train_baseline_log_likelihood))): # Check for NaN
            _logger.critical(f"Baseline LLs invalid on rank {rank}. Exiting."); sys.exit(1)
        _logger.info(f"Rank {rank} Baseline LLs - Train: {train_baseline_log_likelihood:.5f}, Val: {val_baseline_log_likelihood:.5f}")
        
        model = DeepGazeIII(
            features=features_module, # Already on device, frozen
            saliency_network=build_saliency_network(saliency_network_input_channels),
            scanpath_network=None, # No scanpath for spatial pretraining
            fixation_selection_network=build_fixation_selection_network(scanpath_features=0),
            downsample=current_downsample,
            readout_factor=READOUT_FACTOR,
            saliency_map_factor=SALIENCY_MAP_FACTOR,
            included_fixations=[] # No scanpath history
        ).to(device)

        if is_distributed:
            # find_unused_parameters=False because scanpath_network is None, so no params there.
            # If parts of the model might not be used in forward, set to True.
            # For this spatial pretraining, all used params should get grads.
            model = DDP(model, device_ids=[device.index], find_unused_parameters=False) 
            if is_master: _logger.info("Wrapped model with DDP.")

        # Optimizer for head parameters (backbone is frozen)
        head_params = [p for p in model.parameters() if p.requires_grad]
        if not head_params: _logger.critical("No trainable parameters found!"); sys.exit(1)
        optimizer = optim.Adam(head_params, lr=current_lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=current_milestones)

        train_loader = prepare_spatial_dataset_ddp(
            SALICON_train_stimuli, SALICON_train_fixations, SALICON_centerbias, 
            args.batch_size, args.num_workers, is_distributed, is_master, device,
            lmdb_image_cache_dir / 'SALICON_train_dg_original' if args.use_lmdb_images else None, _logger
        )
        validation_loader = prepare_spatial_dataset_ddp(
            SALICON_val_stimuli, SALICON_val_fixations, SALICON_centerbias,
            args.batch_size, args.num_workers, is_distributed, is_master, device,
            lmdb_image_cache_dir / 'SALICON_val_dg_original' if args.use_lmdb_images else None, _logger
        )

        if is_master: _logger.info(f"Starting SALICON pretraining. Output to: {output_dir_stage}")
        _train(
            this_directory=str(output_dir_stage), model=model,
            train_loader=train_loader, train_baseline_log_likelihood=train_baseline_log_likelihood,
            val_loader=validation_loader, val_baseline_log_likelihood=val_baseline_log_likelihood,
            optimizer=optimizer, lr_scheduler=lr_scheduler,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            minimum_learning_rate=args.min_lr,
            validation_metrics=['LL', 'IG', 'NSS', 'AUC'], # Assuming AUC is desired, maps to auc_cpu_fn
            startwith=args.resume_checkpoint, # General resume, _train handles specific step
            device=device,
            is_distributed=is_distributed, is_master=is_master, logger=_logger,
            validation_epochs=args.validation_epochs,
        )
        if is_master: _logger.info("--- SALICON Pretraining Finished ---")

    elif args.stage in ['mit_spatial', 'mit_scanpath_frozen', 'mit_scanpath_full']:
        fold = args.fold
        if fold is None or not (0 <= fold < 10):
            _logger.critical("--fold required for MIT stages and must be 0-9."); sys.exit(1)
        
        # --- Stage Specific Configs (MIT) ---
        if args.stage == 'mit_spatial':
            stage_lr = args.lr_mit_spatial
            stage_milestones = args.lr_milestones_mit_spatial
            stage_downsample = args.downsample_mit
            scanpath_active = False
            prev_stage_checkpoint_dir = train_output_base_dir / 'salicon_pretraining_original_densenet'
            output_dir_stage = train_output_base_dir / 'mit_spatial_original_densenet' / f'crossval-10-{fold}'
            apply_specific_freezing = False
        elif args.stage == 'mit_scanpath_frozen':
            stage_lr = args.lr_mit_scanpath_frozen
            stage_milestones = args.lr_milestones_mit_scanpath_frozen
            stage_downsample = args.downsample_mit
            scanpath_active = True
            prev_stage_checkpoint_dir = train_output_base_dir / 'mit_spatial_original_densenet' / f'crossval-10-{fold}'
            output_dir_stage = train_output_base_dir / 'mit_scanpath_frozen_original_densenet' / f'crossval-10-{fold}'
            apply_specific_freezing = True
        elif args.stage == 'mit_scanpath_full': # mit_scanpath_full
            stage_lr = args.lr_mit_scanpath_full
            stage_milestones = args.lr_milestones_mit_scanpath_full
            stage_downsample = args.downsample_mit
            scanpath_active = True
            prev_stage_checkpoint_dir = train_output_base_dir / 'mit_scanpath_frozen_original_densenet' / f'crossval-10-{fold}'
            output_dir_stage = train_output_base_dir / 'mit_scanpath_full_original_densenet' / f'crossval-10-{fold}'
            apply_specific_freezing = False
        else: # Should not happen due to choices in argparser
            _logger.critical(f"Unknown MIT stage: {args.stage}"); sys.exit(1)

        if is_master: _logger.info(f"--- Preparing MIT1003 Stage: {args.stage} (Fold {fold}) ---")
        
        # --- MIT Data Handling (Conversion, Splits, Baseline) ---
        mit_converted_stimuli_path = train_output_base_dir / 'MIT1003_converted_dg_original' # Store converted data here
        mit_stimuli_file_cache = mit_converted_stimuli_path / "stimuli.pkl" # pysaliency FileStimuli are pickled
        mit_scanpaths_file_cache = mit_converted_stimuli_path / "scanpaths.pkl"

        needs_conversion = True # Assume conversion is needed
        if mit_stimuli_file_cache.exists() and mit_scanpaths_file_cache.exists():
            if is_master: _logger.info("Found cached converted MIT1003 data.")
            needs_conversion = False
        
        if is_distributed: # Sync conversion decision
            needs_conversion_tensor = torch.tensor(int(needs_conversion), device=device, dtype=torch.int)
            dist.broadcast(needs_conversion_tensor, src=0)
            needs_conversion = bool(needs_conversion_tensor.item())

        mit_stimuli_twosize, mit_scanpaths_twosize = None, None
        if needs_conversion:
            if is_master:
                _logger.info(f"Converting MIT1003 data. Original from: {dataset_directory}, Processed to: {mit_converted_stimuli_path}")
                mit_converted_stimuli_path.mkdir(parents=True, exist_ok=True)
                try: # Ensure original MIT data is present
                    pysaliency.external_datasets.mit.get_mit1003_with_initial_fixation(location=str(dataset_directory), replace_initial_invalid_fixations=True)
                except ImportError: # Handle if pysaliency.external_datasets.mit is not found
                     _logger.error("pysaliency.external_datasets.mit module not found. Please ensure pysaliency is correctly installed and includes this module.")
                     if is_distributed: dist.barrier(); sys.exit(1)
                except Exception as e:
                    _logger.critical(f"Failed to get original MIT1003 data: {e}")
                    if is_distributed: dist.barrier(); sys.exit(1)
            if is_distributed: dist.barrier()

            # Load original MIT data (all ranks do this after master ensures it's available)
            try:
                mit_stimuli_orig, mit_scanpaths_orig = pysaliency.external_datasets.mit.get_mit1003_with_initial_fixation(location=str(dataset_directory), replace_initial_invalid_fixations=True)
            except ImportError:
                 _logger.error("pysaliency.external_datasets.mit module not found on non-master rank. Ensure pysaliency installation consistency.")
                 if is_distributed: dist.barrier(); sys.exit(1)
            except Exception as e:
                 _logger.critical(f"Failed to load original MIT1003 metadata on rank {rank}: {e}")
                 if is_distributed: dist.barrier(); sys.exit(1)

            # Use src.data conversion functions
            mit_stimuli_twosize = convert_stimuli(mit_stimuli_orig, mit_converted_stimuli_path, is_master, is_distributed, device, _logger)
            mit_scanpaths_twosize = convert_fixation_trains(mit_stimuli_orig, mit_scanpaths_orig, is_master, _logger) # This is FixationTrains
            
            if mit_stimuli_twosize is None or mit_scanpaths_twosize is None:
                _logger.critical("MIT1003 data conversion failed."); sys.exit(1)

            if is_master:
                try:
                    with atomic_save(str(mit_stimuli_file_cache), text_mode=False, overwrite_part=True) as f:
                        pickle.dump(mit_stimuli_twosize, f)
                    with atomic_save(str(mit_scanpaths_file_cache), text_mode=False, overwrite_part=True) as f:
                        cpickle.dump(mit_scanpaths_twosize, f) # Use cpickle for FixationTrains
                    _logger.info(f"Saved converted MIT1003 data to {mit_converted_stimuli_path}")
                except Exception as e: _logger.error(f"Failed to save converted MIT data: {e}")
            if is_distributed: dist.barrier()
        else: # Load from cache
            if is_master: _logger.info(f"Loading pre-converted MIT1003 data from {mit_converted_stimuli_path}")
            try:
                with open(mit_stimuli_file_cache, "rb") as f: mit_stimuli_twosize = pickle.load(f)
                with open(mit_scanpaths_file_cache, "rb") as f: mit_scanpaths_twosize = cpickle.load(f)
            except Exception as e:
                _logger.critical(f"Failed to load cached converted MIT data: {e}"); sys.exit(1)
        
        # Get fixations from scanpaths (FixationDataset needs FixationTrains, ImageDataset needs Fixations)
        # The notebook uses mit_fixations_twosize = mit_scanpaths_twosize[mit_scanpaths_twosize.scanpath_history_length > 0]
        # This converts FixationTrains to Fixations.
        mit_fixations_for_spatial = mit_scanpaths_twosize.to_fixations() # Convert to flat Fixations for ImageDataset if needed for centerbias
        
        if is_master: _logger.info("Initializing MIT1003 CrossvalidatedBaselineModel for centerbias...")
        MIT1003_centerbias = CrossvalidatedBaselineModel(
            mit_stimuli_twosize,
            mit_fixations_for_spatial[mit_fixations_for_spatial.scanpath_history_length > 0] if hasattr(mit_fixations_for_spatial, 'scanpath_history_length') else mit_fixations_for_spatial, # Use fixations after first if available
            bandwidth=10**-1.6667673342543432, eps=10**-14.884189168516073, caching=False
        )

        # Split data for the current fold
        # For spatial model (ImageDataset), we need Fixations.
        # For scanpath model (FixationDataset), we need FixationTrains.
        MIT1003_stimuli_train, mit_fixations_train_flat = pysaliency.dataset_config.train_split(mit_stimuli_twosize, mit_fixations_for_spatial, crossval_folds=10, fold_no=fold)
        MIT1003_stimuli_val, mit_fixations_val_flat = pysaliency.dataset_config.validation_split(mit_stimuli_twosize, mit_fixations_for_spatial, crossval_folds=10, fold_no=fold)
        
        _, mit_scanpaths_train = pysaliency.dataset_config.train_split(mit_stimuli_twosize, mit_scanpaths_twosize, crossval_folds=10, fold_no=fold)
        _, mit_scanpaths_val = pysaliency.dataset_config.validation_split(mit_stimuli_twosize, mit_scanpaths_twosize, crossval_folds=10, fold_no=fold)


        train_baseline_log_likelihood, val_baseline_log_likelihood = None, None
        if is_master:
            _logger.info(f"Computing baseline LLs for MIT1003 Fold {fold}...")
            try:
                train_baseline_log_likelihood = MIT1003_centerbias.information_gain(MIT1003_stimuli_train, mit_fixations_train_flat, verbose=False, average='image')
                val_baseline_log_likelihood = MIT1003_centerbias.information_gain(MIT1003_stimuli_val, mit_fixations_val_flat, verbose=False, average='image')
                _logger.info(f"Fold {fold} Master Baseline LLs - Train: {train_baseline_log_likelihood:.5f}, Val: {val_baseline_log_likelihood:.5f}")
            except Exception as e:
                _logger.critical(f"Failed to compute MIT baseline LLs: {e}")
                train_baseline_log_likelihood, val_baseline_log_likelihood = float('nan'), float('nan')
        
        ll_bcast_mit = [train_baseline_log_likelihood, val_baseline_log_likelihood]
        if is_distributed:
            if is_master and (train_baseline_log_likelihood is None or val_baseline_log_likelihood is None): ll_bcast_mit = [float('nan'), float('nan')]
            dist.broadcast_object_list(ll_bcast_mit, src=0)
        train_baseline_log_likelihood, val_baseline_log_likelihood = ll_bcast_mit
        if train_baseline_log_likelihood is None or val_baseline_log_likelihood is None or \
           (isinstance(train_baseline_log_likelihood, float) and torch.isnan(torch.tensor(train_baseline_log_likelihood))):
            _logger.critical(f"MIT Baseline LLs invalid on rank {rank}. Exiting."); sys.exit(1)
        _logger.info(f"Rank {rank} MIT Fold {fold} Baseline LLs - Train: {train_baseline_log_likelihood:.5f}, Val: {val_baseline_log_likelihood:.5f}")

        # --- Model Build & Load Checkpoint for MIT stages ---
        current_scanpath_network = build_scanpath_network() if scanpath_active else None
        current_fixation_selection_network = build_fixation_selection_network(scanpath_features=16 if scanpath_active else 0)
        included_fixations_list = [-1, -2, -3, -4] if scanpath_active else []

        # Build model on CPU first for checkpoint loading
        model_cpu = DeepGazeIII(
            features=features_module.cpu(), # Ensure backbone is on CPU for loading
            saliency_network=build_saliency_network(saliency_network_input_channels),
            scanpath_network=current_scanpath_network,
            fixation_selection_network=current_fixation_selection_network,
            downsample=stage_downsample,
            readout_factor=READOUT_FACTOR,
            saliency_map_factor=SALIENCY_MAP_FACTOR,
            included_fixations=included_fixations_list
        )
        # Move backbone back to device after model_cpu uses it for structure
        features_module.to(device)


        # Load checkpoint from previous stage
        start_checkpoint_path = None
        if args.resume_checkpoint: # Explicit resume takes precedence
             start_checkpoint_path = Path(args.resume_checkpoint)
        elif prev_stage_checkpoint_dir:
            # Try to find best validation checkpoint, then final general checkpoint
            chkpt_options = [prev_stage_checkpoint_dir / 'final_best_val.pth', prev_stage_checkpoint_dir / 'final.pth']
            for p_opt in chkpt_options:
                if p_opt.exists():
                    start_checkpoint_path = p_opt
                    break
        
        if start_checkpoint_path and start_checkpoint_path.exists():
            if is_master: _logger.info(f"Loading checkpoint for model_cpu from: {start_checkpoint_path}")
            # Load on CPU, ensure weights_only=False if it's a full checkpoint from _train
            state_dict = torch.load(start_checkpoint_path, map_location='cpu', weights_only=False) 
            # Handle cases where state_dict might be nested (e.g. 'model_state_dict')
            if 'model_state_dict' in state_dict: state_dict = state_dict['model_state_dict']
            elif 'state_dict' in state_dict: state_dict = state_dict['state_dict']

            # If loading from a DDP model, keys might have 'module.' prefix
            # Create a new state_dict without 'module.' prefix
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            
            # For scanpath stages, the previous model might not have scanpath_network weights.
            # strict=False allows loading even if some keys are missing (new layers) or unexpected.
            missing_keys, unexpected_keys = model_cpu.load_state_dict(new_state_dict, strict=False)
            if is_master:
                if missing_keys: _logger.warning(f"Missing keys when loading checkpoint: {missing_keys}")
                if unexpected_keys: _logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")
        elif args.stage != 'salicon_pretrain': # Don't warn if it's the first stage and no resume specified
            _logger.warning(f"No checkpoint found at {prev_stage_checkpoint_dir} or specified by --resume_checkpoint. Starting with fresh head weights for stage {args.stage}.")
        
        model = model_cpu.to(device) # Move the potentially populated model to the target device

        # Apply specific freezing for mit_scanpath_frozen
        if apply_specific_freezing:
            if is_master: _logger.info(f"Freezing parts of saliency network for stage '{args.stage}'.")
            frozen_scopes = [
                "saliency_network.layernorm0", "saliency_network.conv0", "saliency_network.bias0",
                "saliency_network.layernorm1", "saliency_network.conv1", "saliency_network.bias1",
            ]
            for name, param in model.named_parameters():
                if any(name.startswith(scope) for scope in frozen_scopes):
                    param.requires_grad = False
                    if is_master: _logger.debug(f"Froze parameter: {name}")
        
        # Ensure all head parameters are trainable for mit_scanpath_full, overriding previous freezes
        if args.stage == 'mit_scanpath_full':
            if is_master: _logger.info(f"Ensuring all head parameters (non-backbone) are unfrozen for '{args.stage}'.")
            for name, param in model.named_parameters():
                if not name.startswith('features.'): # Backbone is features_module
                    if hasattr(model.features, name): # Should not happen if features_module is separate
                        continue
                    param.requires_grad = True


        if is_distributed:
            # find_unused_parameters=True can be safer if some parts are conditionally used or frozen.
            # For scanpath_frozen, some saliency_network parts are frozen, and scanpath_network is new.
            # For scanpath_full, all head parts are trained.
            find_unused = scanpath_active or apply_specific_freezing 
            model = DDP(model, device_ids=[device.index], find_unused_parameters=find_unused)
            if is_master: _logger.info(f"Wrapped model with DDP (find_unused_parameters={find_unused}).")

        head_params_mit = [p for p in model.parameters() if p.requires_grad]
        if not head_params_mit: _logger.critical("No trainable parameters found for MIT stage!"); sys.exit(1)
        optimizer = optim.Adam(head_params_mit, lr=stage_lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=stage_milestones)

        if scanpath_active:
            train_loader = prepare_scanpath_dataset_ddp(
                MIT1003_stimuli_train, mit_scanpaths_train, MIT1003_centerbias,
                args.batch_size, args.num_workers, is_distributed, is_master, device,
                lmdb_image_cache_dir / f'MIT1003_train_scanpath_dg_original_{fold}' if args.use_lmdb_images else None, 
                _logger, included_fixations_list
            )
            validation_loader = prepare_scanpath_dataset_ddp(
                MIT1003_stimuli_val, mit_scanpaths_val, MIT1003_centerbias,
                args.batch_size, args.num_workers, is_distributed, is_master, device,
                lmdb_image_cache_dir / f'MIT1003_val_scanpath_dg_original_{fold}' if args.use_lmdb_images else None,
                _logger, included_fixations_list
            )
        else: # mit_spatial
            train_loader = prepare_spatial_dataset_ddp(
                MIT1003_stimuli_train, mit_fixations_train_flat, MIT1003_centerbias,
                args.batch_size, args.num_workers, is_distributed, is_master, device,
                lmdb_image_cache_dir / f'MIT1003_train_spatial_dg_original_{fold}' if args.use_lmdb_images else None, _logger
            )
            validation_loader = prepare_spatial_dataset_ddp(
                MIT1003_stimuli_val, mit_fixations_val_flat, MIT1003_centerbias,
                args.batch_size, args.num_workers, is_distributed, is_master, device,
                lmdb_image_cache_dir / f'MIT1003_val_spatial_dg_original_{fold}' if args.use_lmdb_images else None, _logger
            )
        
        # For MIT stages, startwith should be None as we handled checkpoint loading.
        # _train will then look for checkpoints *within* output_dir_stage for resumption of *this specific stage*.
        if is_master: _logger.info(f"Starting MIT stage '{args.stage}' (Fold {fold}). Output to: {output_dir_stage}")
        _train(
            this_directory=str(output_dir_stage), model=model,
            train_loader=train_loader, train_baseline_log_likelihood=train_baseline_log_likelihood,
            val_loader=validation_loader, val_baseline_log_likelihood=val_baseline_log_likelihood,
            optimizer=optimizer, lr_scheduler=lr_scheduler,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            minimum_learning_rate=args.min_lr,
            validation_metrics=['LL', 'IG', 'NSS', 'AUC'],
            startwith=None, # _train handles resumption from output_dir_stage
            device=device,
            is_distributed=is_distributed, is_master=is_master, logger=_logger,
            validation_epochs=args.validation_epochs,
        )
        if is_master: _logger.info(f"--- MIT Stage {args.stage} (Fold {fold}) Finished ---")

    else:
        _logger.critical(f"Unknown or unsupported stage: {args.stage}"); sys.exit(1)

    cleanup_distributed()
    if is_master: _logger.info("Training script finished successfully.")


# ============================================================================
# == ARGUMENT PARSING AND MAIN ENTRY POINT ==
# ============================================================================
if __name__ == "__main__":
    _pre_parser = argparse.ArgumentParser(add_help=False)
    _pre_parser.add_argument('--config_file', type=str, default=None, help="Path to YAML configuration file.")
    _cfg_namespace, _remaining_cli_args = _pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        parents=[_pre_parser],
        description="Train Original DeepGaze III (DenseNet-201) with DDP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Core arguments ---
    parser.add_argument('--stage',
        choices=['salicon_pretrain', 'mit_spatial', 'mit_scanpath_frozen', 'mit_scanpath_full'],
        help='Training stage to execute.')
    parser.add_argument('--log_level', type=str, default="INFO", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    
    # --- Training hyper-parameters ---
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size per GPU.') # Notebook uses 32, adjust based on memory
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--min_lr', type=float, default=1e-7, help='Lower LR bound for schedulers.')
    parser.add_argument('--fold', type=int, help='Cross-validation fold for MIT stages (0-9).')
    parser.add_argument('--validation_epochs', type=int, default=1, help='Run validation every N epochs.')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='Explicit path to a checkpoint to resume from (overrides stage-to-stage flow).')

    # --- Stage-specific LRs and Milestones ---
    parser.add_argument('--lr_salicon', type=float, default=0.001)
    parser.add_argument('--lr_milestones_salicon', type=int, nargs='+', default=[15, 30, 45, 60, 75, 90, 105, 120])
    parser.add_argument('--lr_mit_spatial', type=float, default=0.001)
    parser.add_argument('--lr_milestones_mit_spatial', type=int, nargs='+', default=[3, 6, 9, 12, 15, 18, 21, 24])
    parser.add_argument('--lr_mit_scanpath_frozen', type=float, default=0.001)
    parser.add_argument('--lr_milestones_mit_scanpath_frozen', type=int, nargs='+', default=[10, 20, 30, 31, 32, 33, 34, 35]) # From notebook
    parser.add_argument('--lr_mit_scanpath_full', type=float, default=1e-5)
    parser.add_argument('--lr_milestones_mit_scanpath_full', type=int, nargs='+', default=[3, 6, 9, 12, 15, 18, 21, 24])

    # --- Model specific ---
    parser.add_argument('--downsample_salicon', type=float, default=1.5, help="Initial image downsampling for SALICON.")
    parser.add_argument('--downsample_mit', type=float, default=2.0, help="Initial image downsampling for MIT stages.")

    # --- Dataloading & system ---
    parser.add_argument('--num_workers', type=str, default='auto', help="'auto' or integer for dataloader workers per rank.")
    parser.add_argument('--train_dir', type=str, default='./experiments_deepgaze_original', help="Base output directory for experiments.")
    parser.add_argument('--dataset_dir', type=str, default='./data/pysaliency_datasets', help="Pysaliency datasets directory.")
    parser.add_argument('--lmdb_dir', type=str, default='./data/lmdb_caches_deepgaze_original', help="LMDB image cache directory.")
    parser.add_argument('--use_lmdb_images', action=argparse.BooleanOptionalAction, default=True, help='Use LMDB for stimuli images.')

    # Load YAML and set defaults
    if _cfg_namespace.config_file:
        try:
            with open(_cfg_namespace.config_file, 'r') as f: yaml_cfg = yaml.safe_load(f) or {}
            logging.info(f"Loaded YAML config from: {_cfg_namespace.config_file}")
            parser.set_defaults(**yaml_cfg)
        except FileNotFoundError: logging.warning(f"YAML config file not found: {_cfg_namespace.config_file}")
        except Exception as e: logging.warning(f"Could not read/parse YAML '{_cfg_namespace.config_file}': {e}")

    final_args = parser.parse_args(_remaining_cli_args)

    # Post-process num_workers
    world_size_env = int(os.environ.get("WORLD_SIZE", 1)) # Get world size for num_workers calc
    if isinstance(final_args.num_workers, str) and final_args.num_workers.lower() == 'auto':
        final_args.num_workers = None # Signal for auto-calculation
    if final_args.num_workers is None:
        try: cpu_count = len(os.sched_getaffinity(0))
        except AttributeError: cpu_count = os.cpu_count() or 1
        final_args.num_workers = max(0, cpu_count // world_size_env if world_size_env > 0 else cpu_count)
        if final_args.num_workers == 0 and world_size_env > 1 and cpu_count > world_size_env : final_args.num_workers = 1 # Min 1 worker if cpus allow
    else:
        try: final_args.num_workers = int(final_args.num_workers)
        except ValueError: logging.warning(f"Invalid num_workers='{final_args.num_workers}', using 0."); final_args.num_workers = 0
        if final_args.num_workers < 0: logging.warning(f"Negative num_workers='{final_args.num_workers}', using 0."); final_args.num_workers = 0
    
    if final_args.stage is None: # Should be caught by 'required=True'
        parser.error("--stage is required.")

    try:
        main(final_args)
    except KeyboardInterrupt:
        _logger.warning("Training interrupted by user (Ctrl+C).")
        cleanup_distributed()
        sys.exit(130)
    except Exception:
        _logger.critical("Unhandled exception during main execution:", exc_info=True)
        cleanup_distributed()
        sys.exit(1)