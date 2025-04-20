#!/usr/bin/env python
"""
Multi‑GPU‑ready training script for DeepGaze III with DINOv2 backbone.
Launch with torchrun, e.g.:
    torchrun --standalone --nproc_per_node=4 train_dinogaze_ddp.py \
        --stage salicon_pretrain --batch_size 4 --num_workers 8 --model_name dinov2_vitl14 --lr 0.0005

The script falls back to single‑GPU/CPU when no distributed environment
variables are present, so you can still run it exactly like the original.
"""

import argparse
import logging
import os
import pickle
import shutil
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import pysaliency
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from imageio.v3 import imread, imwrite
from PIL import Image
from pysaliency.baseline_utils import (BaselineModel,
                                       CrossvalidatedBaselineModel)
from tqdm import tqdm

# -----------------------------------------------------------------------------
# UTILS – DISTRIBUTED SETUP ----------------------------------------------------
# -----------------------------------------------------------------------------

def init_distributed() -> tuple[torch.device, int, int, bool]:
    """Initialise torch.distributed (NCCL) if environment variables are present.

    Returns
    -------
    device : torch.device     CUDA device for this rank (or cpu)
    rank   : int              Global rank (0 for single‑GPU case)
    world  : int              World size (1 for single‑GPU case)
    is_master : bool          True on rank 0 – use for logging / checkpointing
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # ---- torchrun sets these automatically --------------------------------
        rank = int(os.environ["RANK"], 10)
        world_size = int(os.environ["WORLD_SIZE"], 10)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", local_rank)
        is_master = rank == 0
        print(f"DDP: Rank {rank}/{world_size} on device cuda:{local_rank}") # Add print for confirmation
    else:
        # -------- single‑GPU / CPU fallback ------------------------------------
        rank = 0
        world_size = 1
        is_master = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DDP: Not detected. Running on device: {device}") # Add print for confirmation
    return device, rank, world_size, is_master


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

# -----------------------------------------------------------------------------
# PATH MANGLING so we can `import deepgaze_pytorch.*` no matter where script is.
# -----------------------------------------------------------------------------
script_dir = Path(__file__).parent.resolve()
sys.path.append(str(script_dir / "../"))  # deepgaze_pytorch is one level up

# -----------------------------------------------------------------------------
# --- DeepGaze III Imports -----------------------------------------------------
# -----------------------------------------------------------------------------
try:
    from deepgaze_pytorch.data import (
        FixationDataset,
        FixationMaskTransform,
        ImageDataset,
        ImageDatasetSampler,
    )
    from deepgaze_pytorch.dinov2_backbone import DinoV2Backbone
    from deepgaze_pytorch.layers import (
        Bias,
        Conv2dMultiInput,
        FlexibleScanpathHistoryEncoding,
        LayerNorm,
        LayerNormMultiInput,
    )
    from deepgaze_pytorch.modules import DeepGazeIII, FeatureExtractor
    from deepgaze_pytorch.training import _train
except ImportError as e:
    print(f"Error importing DeepGaze modules: {e}")
    print("Please ensure 'deepgaze_pytorch' directory is accessible.")
    sys.exit(1)

# -----------------------------------------------------------------------------
# LOGGING – silence non‑master ranks ------------------------------------------
# -----------------------------------------------------------------------------
# Basic config for master, silent for others initially
log_level = logging.INFO if os.environ.get("RANK", "0") == "0" else logging.CRITICAL
logging.basicConfig(
    level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
)
_logger = logging.getLogger("train_dinogaze_ddp")


# -----------------------------------------------------------------------------
# MODEL‑BUILDING HELPERS (Copied from original) -----------------------------
# -----------------------------------------------------------------------------

def build_saliency_network(input_channels):
    # Using _logger.info here is fine, it will only log on master
    _logger.info(f"Building saliency network with {input_channels} input channels.")
    return nn.Sequential(OrderedDict([
        ('layernorm0', LayerNorm(input_channels)),
        ('conv0', nn.Conv2d(input_channels, 64, (1, 1), bias=False)), # Increased capacity slightly
        ('bias0', Bias(64)),
        ('softplus0', nn.Softplus()),

        ('layernorm1', LayerNorm(64)),
        ('conv1', nn.Conv2d(64, 32, (1, 1), bias=False)), # Increased capacity slightly
        ('bias1', Bias(32)),
        ('softplus1', nn.Softplus()),

        ('layernorm2', LayerNorm(32)),
        ('conv2', nn.Conv2d(32, 1, (1, 1), bias=False)),
        ('bias2', Bias(1)),
        ('softplus2', nn.Softplus()),
    ]))

def build_scanpath_network():
    return nn.Sequential(OrderedDict([
        ('encoding0', FlexibleScanpathHistoryEncoding(in_fixations=4, channels_per_fixation=3, out_channels=128, kernel_size=[1, 1], bias=True)),
        ('softplus0', nn.Softplus()),
        ('layernorm1', LayerNorm(128)),
        ('conv1', nn.Conv2d(128, 16, (1, 1), bias=False)),
        ('bias1', Bias(16)),
        ('softplus1', nn.Softplus()),
    ]))

def build_fixation_selection_network(scanpath_features=16):
    _logger.info(f"Building fixation selection network with scanpath features={scanpath_features}")
    saliency_channels = 1 # Output of saliency network's core path before combination
    in_channels_list = [saliency_channels, scanpath_features if scanpath_features > 0 else 0]
    return nn.Sequential(OrderedDict([
        ('layernorm0', LayerNormMultiInput(in_channels_list)),
        ('conv0', Conv2dMultiInput(in_channels_list, 128, (1, 1), bias=False)),
        ('bias0', Bias(128)),
        ('softplus0', nn.Softplus()),
        ('layernorm1', LayerNorm(128)),
        ('conv1', nn.Conv2d(128, 16, (1, 1), bias=False)),
        ('bias1', Bias(16)),
        ('softplus1', nn.Softplus()),
        ('conv2', nn.Conv2d(16, 1, (1, 1), bias=False)),
    ]))

# -----------------------------------------------------------------------------
# DATASET HELPERS – add DistributedSampler support ----------------------------
# -----------------------------------------------------------------------------

def prepare_spatial_dataset(
    stimuli,
    fixations,
    centerbias,
    batch_size,
    num_workers,
    is_distributed: bool,
    is_master: bool,
    path: Path | None = None,
):
    lmdb_path = str(path) if path else None
    if lmdb_path:
        # Only master creates dir, others wait
        if is_master:
            path.mkdir(parents=True, exist_ok=True)
            _logger.info(f"Using LMDB cache for spatial dataset at: {lmdb_path}")
        if is_distributed:
            dist.barrier() # Ensure dir exists before other ranks proceed

    dataset = ImageDataset(
        stimuli=stimuli,
        fixations=fixations,
        centerbias_model=centerbias,
        transform=FixationMaskTransform(sparse=False),
        average="image",
        lmdb_path=lmdb_path,
    )

    if is_distributed:
        # Use DistributedSampler for DDP
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True, drop_last=True)
        # Ensure batch_size is per GPU
        loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size, # This is now batch_size per GPU
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
            drop_last=True # Important for DDP
        )
        # Need to set epoch for sampler shuffling
        loader.sampler.set_epoch(0) # Initialize epoch, _train loop should update this
    else:
        # Original sampler for single device
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=ImageDatasetSampler(dataset, batch_size=batch_size),
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )
    return loader


def prepare_scanpath_dataset(
    stimuli,
    fixations,
    centerbias,
    batch_size,
    num_workers,
    is_distributed: bool,
    is_master: bool,
    path: Path | None = None,
):
    lmdb_path = str(path) if path else None
    if lmdb_path:
        if is_master:
            path.mkdir(parents=True, exist_ok=True)
            _logger.info(f"Using LMDB cache for scanpath dataset at: {lmdb_path}")
        if is_distributed:
            dist.barrier()

    dataset = FixationDataset(
        stimuli=stimuli,
        fixations=fixations,
        centerbias_model=centerbias,
        included_fixations=[-1, -2, -3, -4],
        allow_missing_fixations=True,
        transform=FixationMaskTransform(sparse=False),
        average="image",
        lmdb_path=lmdb_path,
    )

    if is_distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True, drop_last=True)
        loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size, # Per GPU
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
            drop_last=True # Important for DDP
        )
        loader.sampler.set_epoch(0) # Initialize epoch
    else:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=ImageDatasetSampler(dataset, batch_size=batch_size),
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )
    return loader

# -----------------------------------------------------------------------------
# MIT Data Conversion Functions (Copied from original) ------------------------
# -----------------------------------------------------------------------------
def convert_stimulus(input_image):
    size = input_image.shape[:2]
    new_size = (768, 1024) if size[0] < size[1] else (1024, 768)
    new_size_pil = tuple(list(new_size)[::-1]) # pillow uses width, height
    return np.array(Image.fromarray(input_image).resize(new_size_pil, Image.BILINEAR))

def convert_stimuli(stimuli, new_location: Path, is_master: bool):
    assert isinstance(stimuli, pysaliency.FileStimuli)
    new_stimuli_location = new_location / 'stimuli'
    if is_master:
        new_stimuli_location.mkdir(parents=True, exist_ok=True)
        _logger.info(f"Converting stimuli resolution and saving to {new_stimuli_location}...")

    new_filenames = []
    # Use tqdm only on master rank
    filenames_iterable = tqdm(stimuli.filenames, desc="Converting Stimuli") if is_master else stimuli.filenames

    for filename in filenames_iterable:
        stimulus = imread(filename)
        # Ensure 3 channels if grayscale
        if stimulus.ndim == 2:
            stimulus = np.stack([stimulus]*3, axis=-1)
        elif stimulus.shape[2] == 1:
             stimulus = np.concatenate([stimulus]*3, axis=-1)
        elif stimulus.shape[2] == 4: # Handle RGBA
             stimulus = stimulus[:,:,:3]

        if stimulus.shape[2] != 3:
             if is_master: _logger.warning(f"Skipping stimulus {filename} with unexpected shape {stimulus.shape}")
             continue # Or handle differently

        new_stimulus = convert_stimulus(stimulus)
        basename = os.path.basename(filename)
        new_filename = new_stimuli_location / basename

        # Only master writes files
        if is_master:
            if new_stimulus.shape != stimulus.shape: # Only write if changed
                try:
                    imwrite(new_filename, new_stimulus)
                except Exception as e:
                     _logger.error(f"Failed to write {new_filename}: {e}")
                     continue # Skip this image if writing fails
            else:
                shutil.copy(filename, new_filename)

        # All ranks need the list of filenames for the FileStimuli object
        new_filenames.append(new_filename)

    # Ensure all ranks wait for master to finish writing
    if dist.is_initialized():
        dist.barrier()

    # All ranks create the FileStimuli object (needs identical filenames)
    # The master rank will also save the stimuli.json metadata
    return pysaliency.FileStimuli(new_filenames, store_json=is_master)


def convert_fixation_trains(stimuli, fixations, is_master: bool):
    if is_master: _logger.info("Converting fixation coordinates...")
    train_xs = fixations.train_xs.copy()
    train_ys = fixations.train_ys.copy()
    shapes_cache = stimuli.shapes # Cache shapes for efficiency

    # Use tqdm only on master rank
    range_iterable = tqdm(range(len(train_xs)), desc="Converting Fixations") if is_master else range(len(train_xs))

    for i in range_iterable:
        n = fixations.train_ns[i]
        size = shapes_cache[n][:2] # Use cached shape
        new_size = (768, 1024) if size[0] < size[1] else (1024, 768)
        x_factor = new_size[1] / size[1]
        y_factor = new_size[0] / size[0]
        train_xs[i] *= x_factor
        train_ys[i] *= y_factor

    return pysaliency.FixationTrains(
        train_xs=train_xs, train_ys=train_ys, train_ts=fixations.train_ts.copy(),
        train_ns=fixations.train_ns.copy(), train_subjects=fixations.train_subjects.copy(),
        attributes={k: getattr(fixations, k).copy() for k in fixations.__attributes__ if k not in ['subjects', 'scanpath_index']}
    )

# -----------------------------------------------------------------------------
# MAIN ------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def main(args):
    device, rank, world, is_master = init_distributed()

    # Setup logging level again based on rank AFTER init_distributed
    log_level = logging.INFO if is_master else logging.CRITICAL
    _logger.setLevel(log_level)
    # Ensure handlers also respect the level
    for handler in logging.root.handlers:
        handler.setLevel(log_level)

    if is_master:
        _logger.info(f"DDP Initialised: Rank {rank}/{world} | Master: {is_master} | Device: {device}")
        _logger.info(f"Running stage: {args.stage}")
        _logger.info(f"Using DINOv2 model: {args.model_name}")
        _logger.info(f"Extracting from layers: {args.layers}")
        _logger.info(f"Global Batch Size: {args.batch_size * world} ({args.batch_size} per GPU)")
        _logger.info(f"Initial LR: {args.lr}")
        _logger.info(f"Num Workers: {args.num_workers}")
        _logger.info(f"Train Dir: {args.train_dir}")
        _logger.info(f"Dataset Dir: {args.dataset_dir}")
        _logger.info(f"LMDB Dir: {args.lmdb_dir}")

    # ------------------------------------------------------------------
    # Directory setup - Master creates, others wait
    # ------------------------------------------------------------------
    dataset_directory = Path(args.dataset_dir).resolve()
    train_directory = Path(args.train_dir).resolve()
    lmdb_directory = Path(args.lmdb_dir).resolve()

    if is_master:
        dataset_directory.mkdir(parents=True, exist_ok=True)
        train_directory.mkdir(parents=True, exist_ok=True)
        lmdb_directory.mkdir(parents=True, exist_ok=True)

    # Barrier to ensure directories exist before any rank tries to access/write
    if dist.is_initialized():
        dist.barrier()

    # ------------------------------------------------------------------
    # Feature extractor (Identical for all ranks)
    # ------------------------------------------------------------------
    features = DinoV2Backbone(
        layers=args.layers,
        model_name=args.model_name,
        freeze=True # Keep frozen for all stages initially
    )
    C_in = len(features.layers) * features.num_channels
    if is_master:
        _logger.info(f"Feature extractor initialized. Input channels to saliency network: {C_in}")

    # ===================================================================
    # ======================= STAGE DISPATCH ============================
    # ===================================================================

    if args.stage == 'salicon_pretrain':
        if is_master: _logger.info("--- Starting SALICON Pretraining ---")

        # --- Load Data (All ranks load metadata, master might download) ---
        if is_master: _logger.info(f"Loading SALICON data from {dataset_directory}...")
        # pysaliency handles download potentially, let master do it first.
        if is_master:
            pysaliency.get_SALICON_train(location=dataset_directory)
            pysaliency.get_SALICON_val(location=dataset_directory)
        if dist.is_initialized(): dist.barrier() # Ensure data is downloaded/present

        SALICON_train_stimuli, SALICON_train_fixations = pysaliency.get_SALICON_train(location=dataset_directory)
        SALICON_val_stimuli, SALICON_val_fixations = pysaliency.get_SALICON_val(location=dataset_directory)
        if is_master: _logger.info("SALICON data loaded.")

        # --- Baseline Model & LLs (Master computes/caches, all load) ---
        if is_master: _logger.info("Initializing SALICON BaselineModel...")
        # Centerbias model instantiation is cheap, ok for all ranks
        SALICON_centerbias = BaselineModel(stimuli=SALICON_train_stimuli, fixations=SALICON_train_fixations, bandwidth=0.0217, eps=2e-13, caching=False)

        train_ll_cache_file = dataset_directory / 'salicon_baseline_train_ll.pkl'
        val_ll_cache_file = dataset_directory / 'salicon_baseline_val_ll.pkl'
        train_baseline_log_likelihood = None
        val_baseline_log_likelihood = None

        if is_master:
            try:
                with open(train_ll_cache_file, 'rb') as f: train_baseline_log_likelihood = pickle.load(f)
                _logger.info(f"Loaded cached train baseline LL from: {train_ll_cache_file}")
            except Exception as e:
                _logger.warning(f"Cache not found or invalid ({e}). Computing train baseline LL...")
                train_baseline_log_likelihood = SALICON_centerbias.information_gain(SALICON_train_stimuli, SALICON_train_fixations, verbose=True, average='image')
                try:
                    with open(train_ll_cache_file, 'wb') as f: pickle.dump(train_baseline_log_likelihood, f)
                    _logger.info(f"Saved train baseline LL to: {train_ll_cache_file}")
                except Exception as save_e: _logger.error(f"Error saving cache file {train_ll_cache_file}: {save_e}")

            try:
                with open(val_ll_cache_file, 'rb') as f: val_baseline_log_likelihood = pickle.load(f)
                _logger.info(f"Loaded cached validation baseline LL from: {val_ll_cache_file}")
            except Exception as e:
                _logger.warning(f"Cache not found or invalid ({e}). Computing validation baseline LL...")
                val_baseline_log_likelihood = SALICON_centerbias.information_gain(SALICON_val_stimuli, SALICON_val_fixations, verbose=True, average='image')
                try:
                    with open(val_ll_cache_file, 'wb') as f: pickle.dump(val_baseline_log_likelihood, f)
                    _logger.info(f"Saved validation baseline LL to: {val_ll_cache_file}")
                except Exception as save_e: _logger.error(f"Error saving cache file {val_ll_cache_file}: {save_e}")

            _logger.info(f"Master Train Baseline Log Likelihood: {train_baseline_log_likelihood}")
            _logger.info(f"Master Validation Baseline Log Likelihood: {val_baseline_log_likelihood}")

        # Ensure master finished computation/saving, then broadcast/load results
        if dist.is_initialized():
            dist.barrier()
            # Broadcast the computed LLs from master (rank 0) to all other ranks
            ll_list = [train_baseline_log_likelihood, val_baseline_log_likelihood] if is_master else [None, None]
            dist.broadcast_object_list(ll_list, src=0)
            if not is_master:
                train_baseline_log_likelihood, val_baseline_log_likelihood = ll_list

        if train_baseline_log_likelihood is None or val_baseline_log_likelihood is None:
             _logger.error("Baseline log likelihoods were not computed or broadcast correctly.")
             sys.exit(1)
        # Now all ranks have the baseline LL values


        # --- Model Definition (Spatial Only) ---
        model = DeepGazeIII(
            features=features,
            saliency_network=build_saliency_network(C_in),
            scanpath_network=None,
            fixation_selection_network=build_fixation_selection_network(scanpath_features=0),
            downsample=1, readout_factor=14, saliency_map_factor=4, included_fixations=[]
        ).to(device) # Move to device before wrapping

        # --- Wrap model with DDP if using distributed training ---
        if dist.is_initialized():
            model = DDP(
                model,
                device_ids=[device.index], # device.index is the local rank cuda id
                output_device=device.index,
                broadcast_buffers=False,
                # Consider find_unused_parameters=True if complex freezing/subgraphs later
            )
            if is_master: _logger.info("Wrapped model with DDP.")

        # --- Optimizer & Scheduler ---
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        # Adjusted milestones for potentially longer pretraining
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90, 120, 150, 180])

        # --- DataLoaders (Use DDP-aware helper) ---
        train_loader = prepare_spatial_dataset(
            SALICON_train_stimuli, SALICON_train_fixations, SALICON_centerbias,
            args.batch_size, args.num_workers,
            is_distributed=dist.is_initialized(), is_master=is_master,
            path=lmdb_directory / 'SALICON_train'
        )
        validation_loader = prepare_spatial_dataset(
            SALICON_val_stimuli, SALICON_val_fixations, SALICON_centerbias,
            args.batch_size, args.num_workers,
            is_distributed=dist.is_initialized(), is_master=is_master,
            path=lmdb_directory / 'SALICON_val'
        )

        # --- Training ---
        output_dir = train_directory / 'salicon_pretraining'
        if is_master:
            _logger.info(f"Starting training, outputting to: {output_dir}")
        # _train should handle DDP internally (e.g., checkpoint saving only on master)
        _train(
            output_dir, model,
            train_loader, train_baseline_log_likelihood,
            validation_loader, val_baseline_log_likelihood,
            optimizer, lr_scheduler,
            minimum_learning_rate=args.min_lr, device=device, # Pass the rank-specific device
            validation_metrics=['LL', 'IG', 'NSS', 'AUC_CPU'], # Use CPU AUC for stability
            is_distributed=dist.is_initialized(), # Inform _train about DDP status
            is_master=is_master # Inform _train if it's the master rank
        )
        if is_master: _logger.info("--- SALICON Pretraining Finished ---")

    elif args.stage in ['mit_spatial', 'mit_scanpath_frozen', 'mit_scanpath_full']:
        fold = args.fold
        if fold is None or not (0 <= fold < 10):
            # Error should be raised on all ranks to stop execution
            _logger.error("A valid --fold (0-9) is required for MIT1003 stages.")
            sys.exit(1)

        if is_master: _logger.info(f"--- Starting MIT1003 Stage: {args.stage} (Fold {fold}) ---")

        # --- Load/Convert MIT Data (Master converts, all load) ---
        mit_converted_stimuli_path = train_directory / 'MIT1003_twosize'
        mit_converted_stimuli_file = mit_converted_stimuli_path / 'stimuli.json' # pysaliency saves metadata here
        scanpath_cache_file = mit_converted_stimuli_path / 'scanpaths_twosize.pkl'

        # Check if conversion is needed (only master needs to check file existence for decision)
        needs_conversion = False
        if is_master:
            if not mit_converted_stimuli_file.exists() or not scanpath_cache_file.exists():
                needs_conversion = True
                _logger.warning(f"Converted MIT1003 data not found at {mit_converted_stimuli_path}. Will convert now...")

        # Broadcast the decision from master
        if dist.is_initialized():
            needs_conversion_tensor = torch.tensor(int(needs_conversion), device=device)
            dist.broadcast(needs_conversion_tensor, src=0)
            needs_conversion = bool(needs_conversion_tensor.item())

        if needs_conversion:
            # Load original data (potentially downloaded by master first)
            if is_master:
                _logger.info(f"Loading original MIT1003 from {dataset_directory} for conversion...")
                # Let master download/extract if needed
                pysaliency.external_datasets.mit.get_mit1003_with_initial_fixation(
                     location=dataset_directory, replace_initial_invalid_fixations=True
                 )
            if dist.is_initialized(): dist.barrier() # Ensure data is available

            # All ranks load original metadata
            mit_stimuli_orig, mit_scanpaths_orig = pysaliency.external_datasets.mit.get_mit1003_with_initial_fixation(
                 location=dataset_directory, replace_initial_invalid_fixations=True
            )

            # Conversion (master does file I/O, all get results)
            mit_stimuli_twosize = convert_stimuli(mit_stimuli_orig, mit_converted_stimuli_path, is_master)
            mit_scanpaths_twosize = convert_fixation_trains(mit_stimuli_twosize, mit_scanpaths_orig, is_master)

            # Master saves the converted scanpaths cache
            if is_master:
                 with open(scanpath_cache_file, 'wb') as f:
                     pickle.dump(mit_scanpaths_twosize, f)
                 _logger.info(f"Saved converted scanpaths to {scanpath_cache_file}")

            # Barrier to ensure master finished saving
            if dist.is_initialized(): dist.barrier()

        else:
            # Load pre-converted data
            if is_master: _logger.info(f"Loading pre-converted MIT1003 data from {mit_converted_stimuli_path}")
            # All ranks load metadata and cache
            mit_stimuli_twosize = pysaliency.read_json(mit_converted_stimuli_file)
            try:
                 with open(scanpath_cache_file, 'rb') as f:
                     mit_scanpaths_twosize = pickle.load(f)
                 if is_master: _logger.info(f"Loaded converted scanpaths from cache.")
            except FileNotFoundError:
                 _logger.error(f"Scanpath cache file {scanpath_cache_file} not found even after checking. Corrupted state?")
                 sys.exit(1)

        # --- Common MIT Setup (All Ranks) ---
        mit_fixations_twosize = mit_scanpaths_twosize[mit_scanpaths_twosize.lengths > 0]

        # Crossvalidated Baseline Model (Instantiate on all ranks, computation only on master if needed)
        if is_master: _logger.info("Initializing MIT1003 Crossvalidated BaselineModel...")
        MIT1003_centerbias = CrossvalidatedBaselineModel(
            mit_stimuli_twosize, mit_fixations_twosize,
            bandwidth=10**-1.6667673342543432, eps=10**-14.884189168516073, caching=False
        )

        # Get train/val splits for the current fold (identical for all ranks)
        MIT1003_stimuli_train, MIT1003_fixations_train = pysaliency.dataset_config.train_split(mit_stimuli_twosize, mit_fixations_twosize, crossval_folds=10, fold_no=fold)
        MIT1003_stimuli_val, MIT1003_fixations_val = pysaliency.dataset_config.validation_split(mit_stimuli_twosize, mit_fixations_twosize, crossval_folds=10, fold_no=fold)

        # Baseline LLs for the fold (Compute on master, broadcast)
        train_baseline_log_likelihood = None
        val_baseline_log_likelihood = None
        if is_master:
            _logger.info(f"Computing baseline log likelihoods for Fold {fold}...")
            # Compute LLs on the fly for the specific fold
            train_baseline_log_likelihood = MIT1003_centerbias.information_gain(MIT1003_stimuli_train, MIT1003_fixations_train, verbose=False, average='image')
            val_baseline_log_likelihood = MIT1003_centerbias.information_gain(MIT1003_stimuli_val, MIT1003_fixations_val, verbose=False, average='image')
            _logger.info(f"Fold {fold} Master Train Baseline LL: {train_baseline_log_likelihood}")
            _logger.info(f"Fold {fold} Master Validation Baseline LL: {val_baseline_log_likelihood}")

        # Broadcast LLs
        if dist.is_initialized():
            dist.barrier()
            ll_list = [train_baseline_log_likelihood, val_baseline_log_likelihood] if is_master else [None, None]
            dist.broadcast_object_list(ll_list, src=0)
            if not is_master:
                train_baseline_log_likelihood, val_baseline_log_likelihood = ll_list

        if train_baseline_log_likelihood is None or val_baseline_log_likelihood is None:
             _logger.error(f"Fold {fold} baseline LLs were not computed or broadcast correctly.")
             sys.exit(1)


        # --- Stage-Specific Model/Training Setup ---

        if args.stage == 'mit_spatial':
            # --- Model Definition (Spatial Only) ---
            model = DeepGazeIII(
                features=features, saliency_network=build_saliency_network(C_in), scanpath_network=None,
                fixation_selection_network=build_fixation_selection_network(scanpath_features=0),
                downsample=1, readout_factor=14, saliency_map_factor=4, included_fixations=[]
            ).to(device) # Move to device before DDP

            # --- Wrap with DDP ---
            if dist.is_initialized():
                model = DDP(model, device_ids=[device.index], output_device=device.index, broadcast_buffers=False)
                if is_master: _logger.info("Wrapped model with DDP for MIT Spatial.")

            optimizer = optim.Adam(model.parameters(), lr=args.lr) # Use provided LR
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15, 20]) # Shorter schedule

            # --- DataLoaders ---
            train_loader = prepare_spatial_dataset(
                MIT1003_stimuli_train, MIT1003_fixations_train, MIT1003_centerbias,
                args.batch_size, args.num_workers,
                is_distributed=dist.is_initialized(), is_master=is_master,
                path=lmdb_directory / f'MIT1003_train_spatial_{fold}'
            )
            validation_loader = prepare_spatial_dataset(
                MIT1003_stimuli_val, MIT1003_fixations_val, MIT1003_centerbias,
                args.batch_size, args.num_workers,
                is_distributed=dist.is_initialized(), is_master=is_master,
                path=lmdb_directory / f'MIT1003_val_spatial_{fold}'
            )

            # --- Checkpoint & Training ---
            start_checkpoint = train_directory / 'salicon_pretraining' / 'final.pth'
            if is_master and not start_checkpoint.exists():
                _logger.error(f"SALICON pretraining checkpoint not found at {start_checkpoint}. Run pretraining first.")
                # No easy way to signal exit to other ranks here besides them also erroring in _train or hanging
                # Best practice: Checkpoint existence should be verified before starting multi-GPU job.
                # For simplicity, we'll let _train handle the error if the file is missing.
                # sys.exit(1) # Avoid exiting only master

            output_dir = train_directory / 'mit_spatial' / f'crossval-10-{fold}'
            if is_master: _logger.info(f"Starting spatial fine-tuning, outputting to: {output_dir}")
            _train(
                output_dir, model, train_loader, train_baseline_log_likelihood,
                validation_loader, val_baseline_log_likelihood, optimizer, lr_scheduler,
                minimum_learning_rate=args.min_lr, device=device, startwith=start_checkpoint,
                validation_metrics=['LL', 'IG', 'NSS', 'AUC_CPU'],
                is_distributed=dist.is_initialized(), is_master=is_master
            )
            if is_master: _logger.info(f"--- MIT Spatial Fine-tuning Finished (Fold {fold}) ---")

        elif args.stage == 'mit_scanpath_frozen':
            # --- Model Definition (Full, with Scanpath) ---
            model = DeepGazeIII(
                features=features, saliency_network=build_saliency_network(C_in),
                scanpath_network=build_scanpath_network(),
                fixation_selection_network=build_fixation_selection_network(scanpath_features=16), # Default 16
                downsample=1, readout_factor=14, saliency_map_factor=4,
                included_fixations=[-1, -2, -3, -4] # Include history
            ).to(device) # Move to device first

            # --- Freeze Layers (Before DDP) ---
            # Note: Freezing needs to happen identically on all ranks
            frozen_scopes = [
                "saliency_network.layernorm0", "saliency_network.conv0", "saliency_network.bias0",
                "saliency_network.layernorm1", "saliency_network.conv1", "saliency_network.bias1",
            ]
            if is_master: _logger.info("Freezing parameters in scopes: {}".format(', '.join(frozen_scopes)))
            for scope in frozen_scopes:
                for name, param in model.named_parameters():
                    if name.startswith(scope):
                        param.requires_grad = False

            # --- Wrap with DDP (Consider find_unused_parameters=True due to freezing) ---
            if dist.is_initialized():
                model = DDP(
                    model,
                    device_ids=[device.index], output_device=device.index,
                    broadcast_buffers=False,
                    find_unused_parameters=True # Potentially needed because parts are frozen
                )
                if is_master: _logger.info("Wrapped model with DDP for MIT Frozen Scanpath (find_unused_parameters=True).")

            # Optimizer only for trainable parameters
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30]) # Adjust schedule

            # --- DataLoaders ---
            train_loader = prepare_scanpath_dataset(
                 MIT1003_stimuli_train, MIT1003_fixations_train, MIT1003_centerbias,
                 args.batch_size, args.num_workers,
                 is_distributed=dist.is_initialized(), is_master=is_master,
                 path=lmdb_directory / f'MIT1003_train_scanpath_{fold}'
            )
            validation_loader = prepare_scanpath_dataset(
                 MIT1003_stimuli_val, MIT1003_fixations_val, MIT1003_centerbias,
                 args.batch_size, args.num_workers,
                 is_distributed=dist.is_initialized(), is_master=is_master,
                 path=lmdb_directory / f'MIT1003_val_scanpath_{fold}'
            )

            # --- Checkpoint & Training ---
            start_checkpoint = train_directory / 'mit_spatial' / f'crossval-10-{fold}' / 'final.pth'
            # Check existence on master before starting
            if is_master and not start_checkpoint.exists():
                 _logger.error(f"MIT spatial checkpoint not found at {start_checkpoint}. Run spatial tuning first.")
                 # sys.exit(1) # Avoid exiting only master

            output_dir = train_directory / 'mit_scanpath_frozen' / f'crossval-10-{fold}'
            if is_master: _logger.info(f"Starting frozen scanpath training, outputting to: {output_dir}")
            _train(
                output_dir, model, train_loader, train_baseline_log_likelihood,
                validation_loader, val_baseline_log_likelihood, optimizer, lr_scheduler,
                minimum_learning_rate=args.min_lr, device=device, startwith=start_checkpoint,
                validation_metrics=['LL', 'IG', 'NSS', 'AUC_CPU'],
                is_distributed=dist.is_initialized(), is_master=is_master
            )
            if is_master: _logger.info(f"--- MIT Frozen Scanpath Training Finished (Fold {fold}) ---")

        elif args.stage == 'mit_scanpath_full':
            # --- Model Definition (Full, all trainable) ---
            model = DeepGazeIII(
                features=features, saliency_network=build_saliency_network(C_in),
                scanpath_network=build_scanpath_network(),
                fixation_selection_network=build_fixation_selection_network(scanpath_features=16),
                downsample=1, readout_factor=14, saliency_map_factor=4,
                included_fixations=[-1, -2, -3, -4]
            ).to(device) # Move to device first

            # --- Wrap with DDP ---
            if dist.is_initialized():
                model = DDP(model, device_ids=[device.index], output_device=device.index, broadcast_buffers=False)
                if is_master: _logger.info("Wrapped model with DDP for MIT Full Scanpath.")

            # Very low LR for final fine-tuning (already adjusted in argparse logic)
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15]) # Short schedule

            # --- DataLoaders (Can reuse LMDB from frozen stage) ---
            train_loader = prepare_scanpath_dataset(
                 MIT1003_stimuli_train, MIT1003_fixations_train, MIT1003_centerbias,
                 args.batch_size, args.num_workers,
                 is_distributed=dist.is_initialized(), is_master=is_master,
                 path=lmdb_directory / f'MIT1003_train_scanpath_{fold}' # Reuse LMDB
            )
            validation_loader = prepare_scanpath_dataset(
                 MIT1003_stimuli_val, MIT1003_fixations_val, MIT1003_centerbias,
                 args.batch_size, args.num_workers,
                 is_distributed=dist.is_initialized(), is_master=is_master,
                 path=lmdb_directory / f'MIT1003_val_scanpath_{fold}' # Reuse LMDB
            )

            # --- Checkpoint & Training ---
            start_checkpoint = train_directory / 'mit_scanpath_frozen' / f'crossval-10-{fold}' / 'final.pth'
            if is_master and not start_checkpoint.exists():
                 _logger.error(f"MIT frozen scanpath checkpoint not found at {start_checkpoint}. Run frozen scanpath training first.")
                 # sys.exit(1) # Avoid exiting only master

            output_dir = train_directory / 'mit_scanpath_full' / f'crossval-10-{fold}'
            if is_master: _logger.info(f"Starting full scanpath fine-tuning, outputting to: {output_dir}")
            _train(
                output_dir, model, train_loader, train_baseline_log_likelihood,
                validation_loader, val_baseline_log_likelihood, optimizer, lr_scheduler,
                minimum_learning_rate=args.min_lr, device=device, startwith=start_checkpoint,
                validation_metrics=['LL', 'IG', 'NSS', 'AUC_CPU'],
                is_distributed=dist.is_initialized(), is_master=is_master
            )
            if is_master: _logger.info(f"--- MIT Full Scanpath Fine-tuning Finished (Fold {fold}) ---")

    else:
        # Should not happen due to argparse choices, but good practice
        _logger.error(f"Unknown stage: {args.stage}")
        sys.exit(1)

    cleanup_distributed()


# -----------------------------------------------------------------------------
# CLI -------------------------------------------------------------------------
# (Using the more complete parser from the original script)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeepGazeIII with DINOv2 Backbone (Multi-GPU enabled)")
    parser.add_argument('--stage', required=True, choices=['salicon_pretrain', 'mit_spatial', 'mit_scanpath_frozen', 'mit_scanpath_full'], help='Training stage to execute.')
    parser.add_argument('--model_name', default='dinov2_vitg14', choices=['dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'], help='DINOv2 model variant.')
    parser.add_argument('--layers', type=int, nargs='+', default=[-3, -2, -1], help='Indices of transformer layers to extract features from.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size *per GPU*. Reduce for larger models!')
    parser.add_argument('--lr', type=float, default=0.0005, help='Initial learning rate.')
    parser.add_argument('--min_lr', type=float, default=1e-7, help='Minimum learning rate for scheduler.')
    parser.add_argument('--fold', type=int, help='Cross-validation fold for MIT1003 stages (0-9).')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of dataloader workers per GPU. Defaults to cpu_count()/world_size. Set 0 to disable.')
    parser.add_argument('--train_dir', default='./train_dinogaze_vitg', help='Base directory for training outputs (checkpoints, logs).')
    parser.add_argument('--dataset_dir', default='./pysaliency_datasets', help='Directory to store/cache datasets.')
    parser.add_argument('--lmdb_dir', default='./lmdb_cache_dinogaze_vitg', help='Directory for LMDB data caches.')

    args = parser.parse_args()

    # Determine world size early for num_workers calculation
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if args.num_workers is None:
        try:
            cpu_count = len(os.sched_getaffinity(0)) # More reliable than os.cpu_count() on Linux
        except AttributeError:
            cpu_count = os.cpu_count()
        args.num_workers = max(1, cpu_count // world_size) # Distribute workers among GPUs
        # Log only on master after logging is fully set up in main()
        # print(f"Auto-setting num_workers per GPU to: {args.num_workers}")
    elif args.num_workers < 0:
         # print("Warning: num_workers cannot be negative. Setting to 0.") # Log later
         args.num_workers = 0

    # Adjust default LR for final scanpath stage if needed (log later in main)
    if args.stage == 'mit_scanpath_full' and args.lr > 1e-5:
         # print(f"Warning: Setting LR for full scanpath stage to 1e-5 (was {args.lr})") # Log later
         args.lr = 1e-5 # Override default if too high for this stage

    main(args)