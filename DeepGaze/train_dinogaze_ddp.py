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
import glob            # Needed for checkpoint management in copied _train
import tempfile        # Note: Large import, consider lazy import if startup time critical
import warnings        # Import warnings module
from collections import OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path
from collections.abc import Sequence

import contextlib # Needed for gradient accumulation

import numpy as np
import pandas as pd    # Note: Large import
import pysaliency
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torchmetrics    # For GPU AUC etc.
from torchmetrics.metric import Metric
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter # Needed for copied _train
from imageio.v3 import imread, imwrite
from PIL import Image
Image.MAX_IMAGE_PIXELS = None # Disable PIL decompression bomb check for large images
from pysaliency.baseline_utils import (BaselineModel,
                                       CrossvalidatedBaselineModel)
#import pysaliency.external_datasets.mit as mit_module
# Monkey-patch out the broken Octave step
#mit_module.get_mit1003_with_initial_fixation = mit_module.get_mit1003

from tqdm import tqdm
from boltons.fileutils import atomic_save, mkdir_p # Needed for copied _train

# Needed for AMP
from torch import amp

# ── MONKEY-PATCH torchmetrics.Metric.reset to not call .clear() on plain Tensors ──

_original_reset = Metric.reset
def _safe_reset(self):
    for attr in self._defaults:                         # each registered state name
        state = getattr(self, attr)
        if hasattr(state, "clear"):                     # list-like states
            state.clear()
        elif isinstance(state, torch.Tensor):            # tensor states
            state.zero_()
        # otherwise leave it alone
# Override globally before you instantiate/use any metrics
Metric.reset = _safe_reset

# -----------------------------------------------------------------------------
# UTILS – DISTRIBUTED SETUP ----------------------------------------------------
# -----------------------------------------------------------------------------

def init_distributed() -> tuple[torch.device, int, int, bool, bool]:
    """Initialise torch.distributed (NCCL) if environment variables are present.

    Returns
    -------
    device : torch.device     CUDA device for this rank (or cpu)
    rank   : int              Global rank (0 for single‑GPU case)
    world  : int              World size (1 for single‑GPU case)
    is_master : bool          True on rank 0 – use for logging / checkpointing
    is_distributed : bool     True if world_size > 1
    """
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_distributed = world_size > 1

    if is_distributed:
        # ---- torchrun sets these automatically --------------------------------
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        # Correct device after setting cuda device
        device = torch.device(f"cuda:{local_rank}")
        is_master = rank == 0
        # Logging configured later after rank is known
    else:
        # -------- single‑GPU / CPU fallback ------------------------------------
        rank = 0
        world_size = 1
        is_master = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return device, rank, world_size, is_master, is_distributed


def cleanup_distributed():
    """ Clean up the distributed environment. """
    if dist.is_initialized():
        dist.destroy_process_group()

# -----------------------------------------------------------------------------
# PATH MANGLING so we can import deepgaze_pytorch.* no matter where script is.
# -----------------------------------------------------------------------------
script_dir = Path(__file__).parent.resolve()
sys.path.append(str(script_dir / "../"))  # deepgaze_pytorch is one level up

# -----------------------------------------------------------------------------
# --- DeepGaze III Imports -----------------------------------------------------
# -----------------------------------------------------------------------------
try:
    # Data handling classes
    from deepgaze_pytorch.data import (
        FixationDataset, FixationMaskTransform,
        ImageDataset, ImageDatasetSampler)
    # Backbone feature extractor
    from deepgaze_pytorch.dinov2_backbone import DinoV2Backbone
    # Custom layers used in DeepGaze models
    from deepgaze_pytorch.layers import (
        Bias, Conv2dMultiInput, FlexibleScanpathHistoryEncoding,
        LayerNorm, LayerNormMultiInput, SelfAttention) # Added SelfAttention
    # Core model modules
    from deepgaze_pytorch.modules import DeepGazeIII, FeatureExtractor, DeepGazeII # DeepGazeII needed for copied _train
    # Original metric functions (CPU AUC specifically)
    from deepgaze_pytorch.metrics import log_likelihood, nss, auc as auc_cpu_fn # Keep original CPU AUC separate
    # NOTE: The original _train function is NO LONGER imported.
    # It's copied and modified within this script below.
except ImportError as e:
    # Use logger if available, otherwise print
    try:
        _logger.critical(f"Error importing DeepGaze modules: {e}")
        _logger.critical("Please ensure 'deepgaze_pytorch' directory is accessible.")
    except NameError:
        print(f"Error importing DeepGaze modules: {e}")
        print("Please ensure 'deepgaze_pytorch' directory is accessible.")
    sys.exit(1)

# -----------------------------------------------------------------------------
# LOGGING SETUP – Configured properly in main() after DDP init
# -----------------------------------------------------------------------------
_logger = logging.getLogger("train_dinogaze_ddp") # Get logger instance

# -----------------------------------------------------------------------------
# MODEL‑BUILDING HELPERS (Copied from original script with comments)
# -----------------------------------------------------------------------------

def build_saliency_network(input_channels, add_sa_head=False):
    """ Builds the saliency prediction head network. """
    # Using _logger.info here is fine, it will only log on master
    _logger.info(f"Building saliency network with {input_channels} input channels. Add SA Head: {add_sa_head}")
    layers = OrderedDict()

    if add_sa_head:
        # Add SelfAttention as the first layer
        # NOTE: SelfAttention outputs the same number of channels as input by default.
        layers['sa_head'] = SelfAttention(input_channels, key_channels=input_channels // 8, return_attention=False)
        layers['layernorm_sa_out'] = LayerNorm(input_channels) # Normalize SA output
        layers['softplus_sa_out'] = nn.Softplus() # Add non-linearity after SA

    # Reduced complexity slightly given potentially richer ViT features
    layers['layernorm0'] = LayerNorm(input_channels)
    layers['conv0'] = nn.Conv2d(input_channels, 64, (1, 1), bias=False) # Increased capacity slightly
    layers['bias0'] = Bias(64)
    layers['softplus0'] = nn.Softplus()

    layers['layernorm1'] = LayerNorm(64)
    layers['conv1'] = nn.Conv2d(64, 32, (1, 1), bias=False) # Increased capacity slightly
    layers['bias1'] = Bias(32)
    layers['softplus1'] = nn.Softplus()

    layers['layernorm2'] = LayerNorm(32)
    layers['conv2'] = nn.Conv2d(32, 1, (1, 1), bias=False)
    layers['bias2'] = Bias(1)
    layers['softplus2'] = nn.Softplus() # Ensures non-negative output before final log-likelihood

    return nn.Sequential(layers)


def build_scanpath_network():
    """ Builds the network processing scanpath history. """
    return nn.Sequential(OrderedDict([
        ('encoding0', FlexibleScanpathHistoryEncoding(in_fixations=4, channels_per_fixation=3, out_channels=128, kernel_size=[1, 1], bias=True)),
        ('softplus0', nn.Softplus()),
        ('layernorm1', LayerNorm(128)),
        ('conv1', nn.Conv2d(128, 16, (1, 1), bias=False)), # Output 16 channels
        ('bias1', Bias(16)),
        ('softplus1', nn.Softplus()),
    ]))

def build_fixation_selection_network(scanpath_features=16):
    """ Builds the network combining saliency and scanpath features for fixation selection. """
    _logger.info(f"Building fixation selection network with scanpath features={scanpath_features}")
    saliency_channels = 1 # Output of saliency network's core path before combination

    # <<< FIX: DO NOT filter the channel list >>>
    # Always provide two channel counts, even if one is 0.
    # The multi-input layers should handle the 0-channel case.
    in_channels_list = [saliency_channels, scanpath_features if scanpath_features > 0 else 0]
    # in_channels_list = [ch for ch in in_channels_list if ch > 0] # <<< REMOVED THIS LINE >>>
    _logger.info(f"  -> Configured multi-input layers for channel counts: {in_channels_list}")
    # <<< END FIX >>>

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

# -----------------------------------------------------------------------------
# DATASET HELPERS – Modified for DistributedSampler support
# -----------------------------------------------------------------------------

def prepare_spatial_dataset(
    stimuli,
    fixations,
    centerbias,
    batch_size,
    num_workers,
    is_distributed: bool,
    is_master: bool,
    device: torch.device, # Added device for barrier call
    path: Path | None = None,
):
    """ Prepares the DataLoader for spatial (image-based) saliency prediction. """
    lmdb_path = str(path) if path else None
    if lmdb_path:
        # Only master creates dir, others wait via barrier
        if is_master:
            try:
                path.mkdir(parents=True, exist_ok=True)
                _logger.info(f"Using LMDB cache for spatial dataset at: {lmdb_path}")
            except OSError as e:
                 _logger.error(f"Failed to create LMDB directory {path}: {e}")
                 lmdb_path = None # Fallback
        if is_distributed:
            # dist.barrier(device_ids=[device.index] if device.type == 'cuda' else None)
            dist.barrier() # Removed device_ids

    dataset = ImageDataset(
        stimuli=stimuli,
        fixations=fixations,
        centerbias_model=centerbias,
        transform=FixationMaskTransform(sparse=False), # Use dense masks
        average="image", # Average fixations per image
        lmdb_path=lmdb_path,
    )

    if is_distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True, drop_last=True)
        loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
            drop_last=True
        )
    else:
        # Set seed for reproducibility if not distributed
        # torch.manual_seed(42) # Example seed setting
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=ImageDatasetSampler(dataset, batch_size=batch_size), # This sampler shuffles each epoch
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
    device: torch.device, # Added device for barrier call
    path: Path | None = None,
):
    """ Prepares the DataLoader for scanpath prediction (fixation-based). """
    lmdb_path = str(path) if path else None
    if lmdb_path:
        if is_master:
            try:
                path.mkdir(parents=True, exist_ok=True)
                _logger.info(f"Using LMDB cache for scanpath dataset at: {lmdb_path}")
            except OSError as e:
                 _logger.error(f"Failed to create LMDB directory {path}: {e}")
                 lmdb_path = None # Fallback
        if is_distributed:
            # dist.barrier(device_ids=[device.index] if device.type == 'cuda' else None)
            dist.barrier() # Removed device_ids

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
            batch_size=batch_size,
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
            drop_last=True
        )
    else:
        # Set seed for reproducibility if not distributed
        # torch.manual_seed(42) # Example seed setting
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=ImageDatasetSampler(dataset, batch_size=batch_size), # This sampler shuffles each epoch
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )
    return loader

# -----------------------------------------------------------------------------
# MIT Data Conversion Functions (Copied from original with DDP modifications)
# -----------------------------------------------------------------------------
def convert_stimulus(input_image):
    h, w = input_image.shape[:2]
    # Pillow wants (width, height)
    if h < w:         # landscape
        new_size = (1024, 768)
    else:             # portrait
        new_size = (768, 1024)
    return np.array(
        Image.fromarray(input_image)
             .resize(new_size, Image.BILINEAR)
    )


def convert_stimuli(stimuli, new_location: Path, is_master: bool, is_distributed: bool, device: torch.device):
    """ Converts all stimuli in a FileStimuli object to standard sizes and saves them. """
    assert isinstance(stimuli, pysaliency.FileStimuli)
    new_stimuli_location = new_location / 'stimuli'
    if is_master:
        try:
            new_stimuli_location.mkdir(parents=True, exist_ok=True)
            _logger.info(f"Converting stimuli resolution and saving to {new_stimuli_location}...")
        except OSError as e:
            _logger.critical(f"Failed to create stimuli conversion directory {new_stimuli_location}: {e}")
            return None

    new_filenames = []
    filenames_iterable = tqdm(stimuli.filenames, desc="Converting Stimuli", disable=not is_master)
    conversion_errors = 0

    for filename in filenames_iterable:
        try:
            stimulus = imread(filename)
            if stimulus.ndim == 2:
                stimulus = np.stack([stimulus]*3, axis=-1)
            elif stimulus.shape[2] == 1:
                 stimulus = np.concatenate([stimulus]*3, axis=-1)
            elif stimulus.shape[2] == 4:
                 stimulus = stimulus[:,:,:3]
            if stimulus.shape[2] != 3:
                 if is_master: _logger.warning(f"Skipping stimulus {filename} with unexpected shape {stimulus.shape}")
                 conversion_errors += 1
                 continue
            new_stimulus = convert_stimulus(stimulus)
            basename = os.path.basename(filename)
            new_filename = new_stimuli_location / basename
            if is_master:
                # Store original shape in the JSON metadata if using FileStimuli's save method
                # Or, handle metadata storage separately if needed.
                if new_stimulus.shape != stimulus.shape or not new_filename.exists():
                    try: imwrite(new_filename, new_stimulus)
                    except Exception as e: _logger.error(f"Failed to write {new_filename}: {e}"); conversion_errors +=1; continue
                elif not new_filename.exists():
                    try: shutil.copy(filename, new_filename)
                    except Exception as e: _logger.error(f"Failed to copy {filename} to {new_filename}: {e}"); conversion_errors += 1; continue
            new_filenames.append(new_filename)
        except Exception as read_err:
             if is_master: _logger.exception(f"Failed to read or process stimulus {filename}")
             conversion_errors += 1
             continue

    if is_master and conversion_errors > 0:
        _logger.warning(f"Encountered {conversion_errors} errors during stimuli conversion.")

    if is_distributed:
        # dist.barrier(device_ids=[device.index] if device.type == 'cuda' else None)
        dist.barrier() # Removed device_ids

    return pysaliency.FileStimuli(new_filenames)



def convert_fixation_trains(stimuli, fixations, is_master: bool):
    """
    Converts fixation coordinates and filters attributes for FixationTrains,
    incorporating vectorized scaling and stricter attribute checks.
    """
    # --- Initial Checks ---
    if stimuli is None or not hasattr(stimuli, 'shapes'):
        _logger.error("Stimuli object is None or missing 'shapes'. Cannot convert fixations.")
        return None
    core_attrs = ['train_xs', 'train_ys', 'train_ts', 'train_ns', 'train_subjects']
    if not all(hasattr(fixations, attr) for attr in core_attrs):
        _logger.error("Input fixations object is missing one or more core train_* attributes.")
        return None

    if is_master: _logger.info("Converting fixation coordinates...")

    try:
        # --- Pull out the original 2D scanpath matrices ---
        train_xs       = np.asarray(fixations.train_xs,      dtype=float)   # shape (T, L)
        train_ys       = np.asarray(fixations.train_ys,      dtype=float)   # shape (T, L)
        train_ts       = np.asarray(fixations.train_ts,      dtype=float)   # shape (T, L)
        train_ns       = np.asarray(fixations.train_ns,      dtype=int)     # shape (T, L)
        train_subjects = np.asarray(fixations.train_subjects,dtype=int)     # shape (T, L)

        original_fixation_count = len(train_xs)
        if original_fixation_count == 0:
            _logger.warning("Input fixations has zero fixations after potential collapsing.")
            return pysaliency.FixationTrains( # Return empty but valid object
                train_xs=np.array([]), train_ys=np.array([]), train_ts=np.array([]),
                train_ns=np.array([], dtype=int), train_subjects=np.array([]), attributes={}
            )
    except Exception as e:
         _logger.exception(f"Error during initial coordinate processing/collapsing: {e}")
         return None

    
    
    try:
        # --- Compute Scaling Factors (once per stimulus) ---
        shapes_cache = stimuli.shapes
        scale_factors_map = {}
        for n, shape in enumerate(shapes_cache):
            size = shape[:2] # Original (height, width)
            if len(size) != 2 or size[0] <= 0 or size[1] <= 0:
                scale_factors_map[n] = (np.nan, np.nan) # Mark as invalid
                if is_master: _logger.warning(f"Invalid original size {size} for stimulus index {n}.")
            else:
                new_size = (768, 1024) if size[0] < size[1] else (1024, 768) # (h, w)
                scale_factors_map[n] = (new_size[1] / size[1], new_size[0] / size[0]) # (x_factor, y_factor)
    except Exception as e:
        _logger.exception(f"Failed to compute scaling factors from stimuli shapes: {e}")
        return None

    # --- Vectorized Coordinate Scaling ---
    try:
        N_stim = len(shapes_cache)
        lut_x  = np.full(N_stim, np.nan, dtype=float)
        lut_y  = np.full(N_stim, np.nan, dtype=float)
        for n, (xf, yf) in scale_factors_map.items():
            lut_x[n] = xf
            lut_y[n] = yf

        # look‑ups are 1‑D (N,) here
        x_factors_1d = lut_x[train_ns]          # (N,)
        y_factors_1d = lut_y[train_ns]

        # match the second dimension of the coordinate matrices
        L = train_xs.shape[1]                   # 16
        x_factors = np.repeat(x_factors_1d[:, None], L, axis=1)   # (N,16)
        y_factors = np.repeat(y_factors_1d[:, None], L, axis=1)

        # apply element‑wise
        train_xs = train_xs.astype(float) * x_factors
        train_ys = train_ys.astype(float) * y_factors
        
                # --- clip to image bounds (avoid "positions out of bound") -------------
        # build LUTs with the *max legal index* for each stimulus
        width_new  = np.array([nf[0]            # w
                               for nf in ( (1024,768) if s[0] < s[1] else (768,1024)
                                           for s in shapes_cache )], dtype=float) - 1.0
        height_new = np.array([nf[1]            # h
                               for nf in ( (1024,768) if s[0] < s[1] else (768,1024)
                                           for s in shapes_cache )], dtype=float) - 1.0

        max_x = width_new[train_ns][:, None]    # (N,16)
        max_y = height_new[train_ns][:, None]

        # clip; keep a small epsilon below the edge
        train_xs = np.clip(train_xs, 0.0, max_x)
        train_ys = np.clip(train_ys, 0.0, max_y)


        # --- Identify Valid Fixations (Post‐Scaling) ---
        valid_mask = ~np.isnan(x_factors) & ~np.isnan(y_factors) \
                     & ~np.isnan(train_xs) & ~np.isnan(train_ys)
        final_valid_indices = np.where(valid_mask)[0]

        num_filtered = original_fixation_count - len(final_valid_indices)
        if is_master and num_filtered > 0:
            _logger.warning(f"Filtered out {num_filtered} fixations due to invalid coordinates or scaling issues.")

    except Exception as e:
        _logger.exception(f"Error during vectorized coordinate scaling or validation: {e}")
        return None


    # --- Safe Attribute Transfer ("Bullet-Proof Fix" Refined) ---
    # Only explicitly skip the core attributes passed as positional args
    skip_keys = set(core_attrs)
    # Add others sometimes present/problematic, though filtering handles most cases
    skip_keys.update(['subjects', 'scanpath_index', 'lengths', 'average_duration'])

    attributes_dict = {}
    if hasattr(fixations, '__attributes__'):
        attributes_iterable = tqdm(fixations.__attributes__, desc="Filtering Attributes", disable=not is_master) if is_master else fixations.__attributes__
        for k in attributes_iterable:
            if k in skip_keys:
                continue # Skip core/explicitly excluded

            try:
                attr_val = getattr(fixations, k)

                # Keep scalars/0D arrays
                is_scalar_like = not isinstance(attr_val, (Sequence, np.ndarray)) or \
                                 (isinstance(attr_val, np.ndarray) and attr_val.ndim == 0)
                is_string_like = isinstance(attr_val, (str, bytes))

                if is_scalar_like and not is_string_like: # Keep non-sequence, non-string data
                    attributes_dict[k] = attr_val
                    if is_master: _logger.debug(f"Kept scalar/0D attribute '{k}'")
                    continue

                # Process valid sequences (excluding strings/bytes)
                if isinstance(attr_val, Sequence) and not is_string_like:
                    try:
                        arr = np.asarray(attr_val) # Convert to numpy array
                    except Exception as np_err:
                        if is_master: _logger.warning(f"Could not convert sequence attribute '{k}' to NumPy array: {np_err}. Skipping.")
                        continue

                    # Check: Must be exactly 1D AND match original fixation count
                    if arr.ndim == 1 and arr.shape[0] == original_fixation_count:
                        attributes_dict[k] = arr.copy()[final_valid_indices] # Filter valid ones
                        if is_master: _logger.debug(f"Kept and filtered 1D attribute '{k}' (Shape: {arr.shape})")
                    else:
                         if is_master: # Log skip reason
                            shape_info = arr.shape if hasattr(arr, 'shape') else 'N/A'
                            reason = f"ndim={arr.ndim}!=1" if arr.ndim != 1 else f"shape[0]={arr.shape[0]}!={original_fixation_count}"
                            _logger.debug(f"Skipped attribute '{k}' – Shape {shape_info}, Reason: {reason}")
                # else: implicitly skip non-sequence, non-scalar, or string types

            except AttributeError:
                 if is_master: _logger.warning(f"Attribute '{k}' in __attributes__ but not found. Skipping.")
            except Exception as attr_err:
                if is_master: _logger.exception(f"Unexpected error processing attribute '{k}'. Skipping.")

    else:
         if is_master: _logger.warning("fixations object lacks __attributes__. Cannot transfer extra attributes.")

    # --- Construct Final Object ---
    try:
        new_fixations = pysaliency.FixationTrains(
            train_xs=train_xs[final_valid_indices],
            train_ys=train_ys[final_valid_indices],
            train_ts=train_ts[final_valid_indices],
            train_ns=train_ns[final_valid_indices],
            train_subjects=train_subjects[final_valid_indices],
            attributes=attributes_dict
        )
        if is_master: _logger.info(f"Created new FixationTrains with {len(final_valid_indices)} valid fixations.")
        return new_fixations
    except Exception as e:
         _logger.exception(f"Error creating final FixationTrains object: {e}")
         return None


# -----------------------------------------------------------------------------
# TRAINING FUNCTIONS (Copied and Modified for DDP)
# -----------------------------------------------------------------------------

def eval_epoch(model, dataset, baseline_information_gain, device, metrics=None, is_distributed=False, is_master=True):
    """ Evaluates the model for one epoch on the validation set. Handles DDP aggregation. """
    model.eval()
    default_metrics = ['LL', 'IG', 'NSS', 'AUC_CPU']
    if metrics is None: metrics = default_metrics
    if 'IG' in metrics and 'LL' not in metrics: metrics.append('LL')
    if is_master: _logger.debug(f"Evaluating metrics: {metrics}")

    total_metric_sums = {
        name: torch.tensor(0.0, device=device, dtype=torch.float64)
        for name in metrics if name in ['LL', 'NSS', 'AUC_CPU']
    }
    total_weight = torch.tensor(0.0, device=device, dtype=torch.float64)

    auroc_metric_gpu = None
    if 'AUC_GPU' in metrics:
        if is_master: _logger.debug("Initializing GPU AUROC (max_fpr=1.0)")
        auroc_metric_gpu = torchmetrics.AUROC(task="binary", max_fpr=1.0).to(device)

    metric_functions_avg = {}
    if 'LL' in metrics: metric_functions_avg['LL'] = log_likelihood
    if 'NSS' in metrics: metric_functions_avg['NSS'] = nss
    if 'AUC_CPU' in metrics:
        if is_master: _logger.debug("Will calculate CPU AUC using original function.")
        metric_functions_avg['AUC_CPU'] = auc_cpu_fn

    oom_error_gpu_auc = False
    pbar_desc = "Validating" + (f" (Rank {dist.get_rank()})" if is_distributed else "")
    pbar = tqdm(dataset, desc=pbar_desc, disable=not is_master, leave=False)

    with torch.no_grad():
        batch_process_count = 0
        for batch in pbar:
            try:
                image = batch.pop('image').to(device)
                centerbias = batch.pop('centerbias').to(device)
                fixation_mask = batch.pop('fixation_mask').to(device)
                x_hist = batch.pop('x_hist', torch.tensor([])).to(device)
                y_hist = batch.pop('y_hist', torch.tensor([])).to(device)
                weights = batch.pop('weight').to(device)
                durations = batch.pop('durations', torch.tensor([])).to(device)
                kwargs = {k: v.to(device) for k, v in batch.items()}
            except Exception as e:
                _logger.error(f"Error moving batch data to device {device}: {e}")
                continue

            log_density = None
            try:
                underlying_model = model.module if is_distributed else model

                # No AMP during validation as inference might be sensitive
                if isinstance(underlying_model, DeepGazeII):
                    _logger.debug("Using DeepGazeII forward.")
                    log_density = model(image, centerbias, **kwargs)

                elif getattr(underlying_model, 'scanpath_network', None) is None:
                    # Spatial-only: Call model's forward with only image and centerbias
                    _logger.debug("Using spatial-only via model.forward(image, centerbias)")
                    log_density = model(image, centerbias)
                else:
                    # Full scanpath model: Call model's forward with all relevant arguments
                    _logger.debug("Using full model forward with scanpath.")
                    log_density = model(
                        image, centerbias, x_hist=x_hist, y_hist=y_hist, durations=durations, **kwargs
                    )

            except torch.cuda.OutOfMemoryError as e:
                current_rank = dist.get_rank() if is_distributed else 0
                _logger.error(
                    f"\n\n!!! OOM ERROR during Validation Forward Pass (Rank {current_rank}) !!!\n"
                    f"Image shape: {image.shape if 'image' in locals() else 'N/A'}, "
                    f"Centerbias shape: {centerbias.shape if 'centerbias' in locals() else 'N/A'}\n"
                    f"Error: {e}\nTry reducing validation batch size or model size."
                )
                if 'image' in locals(): del image
                if 'centerbias' in locals(): del centerbias
                if 'fixation_mask' in locals(): del fixation_mask
                if 'x_hist' in locals(): del x_hist
                if 'y_hist' in locals(): del y_hist
                if 'weights' in locals(): del weights
                if 'durations' in locals(): del durations
                if 'kwargs' in locals(): del kwargs
                if 'log_density' in locals(): del log_density
                torch.cuda.empty_cache()
                raise e
            except Exception as forward_e:
                 _logger.exception("Error during validation forward pass")
                 continue

            if log_density is None:
                _logger.error("log_density is None after forward pass, skipping batch.")
                continue

            try:
                if isinstance(fixation_mask, torch.sparse.Tensor):
                    target_mask_dense = fixation_mask.to_dense()
                else:
                    target_mask_dense = fixation_mask
                target_mask_int = target_mask_dense.long()
                target_mask_binary = (target_mask_int > 0).long() # Binary mask used for most metrics
            except Exception as e:
                 _logger.error(f"Error preparing target masks: {e}")
                 continue

            batch_weight_sum = weights.sum()

            # --- Accumulate Batch-Averaged Metrics (LL, NSS, AUC_CPU) ---
            if batch_weight_sum > 0:
                current_batch_weight = batch_weight_sum
                total_weight += current_batch_weight
                batch_process_count += 1
                for metric_name, metric_fn in metric_functions_avg.items():
                    if metric_name in total_metric_sums:
                        try:
                            # <<< FIX: Pass the ORIGINAL fixation_mask >>>
                            # LL, NSS, and the original AUC function expect the mask
                            # potentially containing fixation counts, not just binary values.
                            mask_for_metric = fixation_mask
                            value = metric_fn(log_density, mask_for_metric, weights=weights)
                            # <<< END FIX >>>

                            # Accumulate results (rest of the logic is the same)
                            if isinstance(value, torch.Tensor) and value.ndim == 0:
                                total_metric_sums[metric_name] += value * current_batch_weight
                            elif isinstance(value, (int, float)):
                                total_metric_sums[metric_name] += torch.tensor(value, device=device, dtype=torch.float64) * current_batch_weight
                            else:
                                if is_master: warnings.warn(f"Metric {metric_name} returned non-scalar {type(value)}. Skipping.", RuntimeWarning)
                        except Exception as e:
                            if is_master: warnings.warn(f"Error calculating {metric_name} for batch: {e}. Skipping.", RuntimeWarning)

            if auroc_metric_gpu is not None and not oom_error_gpu_auc:
                try:
                    preds_flat = log_density.flatten()
                    targets_flat = target_mask_binary.flatten() # Use binary target here too
                    if torch.unique(targets_flat).numel() > 1:
                        auroc_metric_gpu.update(preds_flat, targets_flat)
                except torch.cuda.OutOfMemoryError as e:
                    if is_master: warnings.warn("\n!!! OOM ERROR during GPU AUROC update !!!\nGPU AUC will be NaN. Reduce validation batch size.", RuntimeWarning)
                    oom_error_gpu_auc = True
                    del auroc_metric_gpu; auroc_metric_gpu = None; torch.cuda.empty_cache()
                except Exception as e:
                    if is_master: warnings.warn(f"Error updating GPU AUROC: {e}. GPU AUC may be incorrect.", RuntimeWarning)

            if is_master and total_weight > 0 and batch_process_count > 0:
                 desc = "Validating"
                 if 'LL' in total_metric_sums:
                     local_avg_ll = (total_metric_sums["LL"] / total_weight).item()
                     desc += f' LL_local {local_avg_ll:.5f}'
                 if 'AUC_CPU' in total_metric_sums:
                     local_avg_auc_cpu = (total_metric_sums["AUC_CPU"] / total_weight).item()
                     desc += f' AUC_CPU_local {local_avg_auc_cpu:.5f}'
                 pbar.set_description(desc)

    if is_distributed:
        dist.all_reduce(total_weight, op=dist.ReduceOp.SUM)
        for metric_name in total_metric_sums:
            dist.all_reduce(total_metric_sums[metric_name], op=dist.ReduceOp.SUM)

    final_metrics = {}
    total_weight_cpu = total_weight.item()

    if total_weight_cpu > 0:
        for metric_name, metric_sum in total_metric_sums.items():
            if metric_name in metrics:
                final_metrics[metric_name] = (metric_sum / total_weight).item()

        if 'AUC_GPU' in metrics:
            if auroc_metric_gpu is not None and not oom_error_gpu_auc:
                try:
                    metric_has_state = False
                    if hasattr(auroc_metric_gpu, 'update_count') and auroc_metric_gpu.update_count > 0: metric_has_state = True
                    elif hasattr(auroc_metric_gpu, 'preds') and hasattr(auroc_metric_gpu, 'target') and len(auroc_metric_gpu.preds) > 0: metric_has_state = True

                    if metric_has_state:
                        final_metrics['AUC_GPU'] = auroc_metric_gpu.compute().item()
                    else:
                         if is_master: warnings.warn("GPU AUROC metric has no state/updates. Setting AUC_GPU to NaN.", RuntimeWarning)
                         final_metrics['AUC_GPU'] = float('nan')
                except Exception as e:
                    if is_master: warnings.warn(f"Failed to compute final GPU AUC: {e}. Setting AUC_GPU to NaN.", RuntimeWarning)
                    final_metrics['AUC_GPU'] = float('nan')
            else:
                final_metrics['AUC_GPU'] = float('nan')
    else:
        for metric_name in metrics:
            if metric_name != 'IG': final_metrics[metric_name] = float('nan')

    for metric_name in metrics:
        if metric_name != 'IG' and metric_name not in final_metrics:
             final_metrics[metric_name] = float('nan')

    if 'IG' in metrics:
        ll_value = final_metrics.get('LL', float('nan'))
        if not np.isnan(ll_value) and baseline_information_gain is not None:
            final_metrics['IG'] = ll_value - baseline_information_gain
        else:
            final_metrics['IG'] = float('nan')

    if auroc_metric_gpu is not None:
        auroc_metric_gpu.reset()

    return final_metrics


def train_epoch(model, dataset, optimizer, device, scaler, gradient_accumulation_steps=1, is_distributed=False, is_master=True):
    """ Trains the model for one epoch. Handles DDP gradient sync, AMP, and Gradient Accumulation. """
    model.train()
    local_losses = []
    local_batch_weights = []
    total_batches = len(dataset)
    # Estimate global batch size for logging (handle non-DDP case)
    sampler = getattr(dataset, 'sampler', None) # Check if sampler exists
    batch_size_attr = getattr(sampler, 'batch_size', getattr(dataset, 'batch_size', None)) # Try sampler first, then dataset
    if batch_size_attr is None and hasattr(dataset, 'batch_sampler') and dataset.batch_sampler is not None: # Check batch_sampler
         batch_size_attr = getattr(dataset.batch_sampler, 'batch_size', None)

    if batch_size_attr is not None:
         global_batch_size_est = batch_size_attr * (dist.get_world_size() if is_distributed else 1) * gradient_accumulation_steps
    else:
         global_batch_size_est = "Unknown" # Fallback if size cannot be determined
    if is_master: _logger.debug(f"Estimated global batch size for logging: {global_batch_size_est}")

    # Reset optimizer gradients only at the beginning of the accumulation cycle
    optimizer.zero_grad()

    pbar_desc = "Training" + (f" (Rank {dist.get_rank()})" if is_distributed else "")
    pbar = tqdm(dataset, desc=pbar_desc, disable=not is_master, leave=False)

    for batch_idx, batch in enumerate(pbar):

        # Determine if this is the last micro-batch for gradient accumulation
        is_accumulation_step = (batch_idx + 1) % gradient_accumulation_steps == 0
        is_last_batch_in_epoch = (batch_idx + 1) == total_batches

        # Context manager for DDP gradient synchronization
        sync_context = model.no_sync() if (is_distributed and not is_accumulation_step and not is_last_batch_in_epoch) else contextlib.nullcontext()

        with sync_context:
            try:
                image = batch.pop('image').to(device)
                centerbias = batch.pop('centerbias').to(device)
                fixation_mask = batch.pop('fixation_mask').to(device)
                x_hist = batch.pop('x_hist', torch.tensor([])).to(device)
                y_hist = batch.pop('y_hist', torch.tensor([])).to(device)
                weights = batch.pop('weight').to(device)
                durations = batch.pop('durations', torch.tensor([])).to(device)
                kwargs = {k: v.to(device) for k, v in batch.items()}
            except Exception as e:
                _logger.error(f"Error moving batch data to device {device}: {e}")
                continue

            log_density = None
            try:
                underlying_model = model.module if is_distributed else model

                # Use Automatic Mixed Precision (AMP)
                with amp.autocast('cuda', dtype=torch.float16): # Use float16 for speed/memory
                    if isinstance(underlying_model, DeepGazeII):
                        _logger.debug("Using DeepGazeII forward.")
                        log_density = model(image, centerbias, **kwargs)
                    elif getattr(underlying_model, 'scanpath_network', None) is None:
                        _logger.debug("Using spatial-only via model.forward(image, centerbias)")
                        log_density = model(image, centerbias)
                    else:
                        _logger.debug("Using full model forward with scanpath.")
                        log_density = model(
                            image, centerbias, x_hist=x_hist, y_hist=y_hist, durations=durations, **kwargs
                        )

                    if log_density is None:
                        _logger.error("log_density is None after forward pass, skipping batch.")
                        continue

                    loss = -log_likelihood(log_density, fixation_mask, weights=weights)
                    # Scale loss for gradient accumulation
                    loss = loss / gradient_accumulation_steps

                if torch.isnan(loss) or torch.isinf(loss):
                    _logger.error(f"NaN or Inf loss detected! Skipping backward/step.")
                    # If NaN/Inf, reset gradients accumulated so far for this step
                    if not is_accumulation_step and not is_last_batch_in_epoch:
                         optimizer.zero_grad() # Reset gradients for the current accumulation cycle
                    continue

            except Exception as forward_loss_e:
                _logger.exception("Error during training forward pass or loss calculation")
                if not is_accumulation_step and not is_last_batch_in_epoch:
                    optimizer.zero_grad() # Reset potentially corrupted gradients
                continue

            # Backward pass with GradScaler
            try:
                # scaler.scale(loss).backward() will sync grads if sync_context is nullcontext
                scaler.scale(loss).backward()
            except Exception as backward_e:
                _logger.exception("Error during backward pass")
                if not is_accumulation_step and not is_last_batch_in_epoch:
                     optimizer.zero_grad() # Reset potentially corrupted gradients
                continue

        # Optimizer step only after accumulating gradients or at the very end
        if is_accumulation_step or is_last_batch_in_epoch:
            try:
                # Unscales gradients and calls optimizer.step()
                scaler.step(optimizer)
                # Updates the scale for next iteration
                scaler.update()
                # Zero the gradients *after* the step for the next accumulation cycle
                optimizer.zero_grad()
            except Exception as opt_step_e:
                _logger.exception("Error during optimizer step or scaler update")
                optimizer.zero_grad() # Ensure grads are zeroed if step failed
                continue

        # Log loss (optional: log per micro-batch or only after accumulation step)
        detached_loss = loss.detach() * gradient_accumulation_steps # Unscale for logging
        local_losses.append(detached_loss.cpu().numpy())
        local_batch_weights.append(weights.detach().cpu().numpy().sum())
        if is_master and len(local_losses) > 0:
            current_avg_loss = np.nanmean(local_losses) if np.any(np.isnan(local_batch_weights)) else np.average(local_losses, weights=local_batch_weights)
            pbar_desc_str = f'Training Loss (Rank 0 Avg): {current_avg_loss:.5f}'
            if gradient_accumulation_steps > 1:
                 pbar_desc_str += f' (Acc {batch_idx % gradient_accumulation_steps + 1}/{gradient_accumulation_steps})'
            pbar.set_description(pbar_desc_str)


    if len(local_losses) > 0:
        final_avg_loss = np.nanmean(local_losses) if np.any(np.isnan(local_batch_weights)) else np.average(local_losses, weights=local_batch_weights)
        return final_avg_loss
    else:
        return np.nan


def restore_from_checkpoint(model, optimizer, scheduler, scaler, path, device, is_distributed=False):
    """ Restores training state from a checkpoint file, handling DDP 'module.' prefix and GradScaler. """
    if not os.path.exists(path):
        _logger.error(f"Checkpoint path not found: {path}. Cannot restore.")
        return 0, np.nan, False  # No checkpoint, start from scratch

    _logger.info(f"Restoring checkpoint from: {path} (Distributed: {is_distributed})")
    try:
        # Load either a full checkpoint dict or a bare state_dict
        data = torch.load(path, map_location=device)
        if not isinstance(data, dict) or 'model' not in data:
            state_dict = data
        else:
            state_dict = data['model']
    except Exception as e:
        _logger.exception(f"Failed to load checkpoint file {path}")
        return 0, np.nan, False

    # --- Restore Model State ---
    model_state_dict = state_dict
    adjusted_state_dict = OrderedDict()
    is_ddp_checkpoint = any(k.startswith('module.') for k in model_state_dict.keys())
    current_model_is_ddp = isinstance(model, DDP)

    if current_model_is_ddp:
        # Model is wrapped in DDP, ensure keys start with 'module.'
        if not is_ddp_checkpoint:
            _logger.warning("Current model is DDP, but checkpoint is not. Adding 'module.' prefix.")
            for k, v in model_state_dict.items():
                adjusted_state_dict[f'module.{k}'] = v
        else:
            _logger.debug("Current model and checkpoint are both DDP. Checking prefixes.")
            for k, v in model_state_dict.items():
                if not k.startswith('module.'):
                    _logger.warning(f"Adding missing 'module.' prefix to key '{k}'.")
                    adjusted_state_dict[f'module.{k}'] = v
                else:
                    adjusted_state_dict[k] = v
    else:
        # Model is not DDP, strip 'module.' if present
        if is_ddp_checkpoint:
            _logger.warning("Current model is not DDP, but checkpoint is. Removing 'module.' prefix.")
            for k, v in model_state_dict.items():
                if k.startswith('module.'):
                    adjusted_state_dict[k[len('module.'):]] = v
                else:
                    adjusted_state_dict[k] = v
        else:
            adjusted_state_dict = model_state_dict

    try:
        missing_keys, unexpected_keys = model.load_state_dict(adjusted_state_dict, strict=False)
        if missing_keys:
            _logger.warning(f"Missing keys when loading model state: {missing_keys}")
        if unexpected_keys:
            _logger.warning(f"Unexpected keys when loading model state: {unexpected_keys}")
    except Exception as load_err:
        _logger.exception("Error loading model state_dict")
        return 0, np.nan, False

    # --- Restore Optimizer State ---
    if isinstance(data, dict) and 'optimizer' in data and optimizer is not None:
        try:
            optimizer.load_state_dict(data['optimizer'])
            _logger.info("Optimizer state restored.")
            # Move optimizer tensors to correct device
            for state in optimizer.state.values():
                for key, val in state.items():
                    if isinstance(val, torch.Tensor):
                        state[key] = val.to(device)
        except Exception as e:
            _logger.warning(f"Could not restore optimizer state: {e}. Starting with fresh optimizer.")
    elif optimizer is not None:
        _logger.warning("Optimizer state not found in checkpoint. Starting with fresh optimizer.")

    # --- Restore Scheduler State ---
    scheduler_restored = False
    if isinstance(data, dict) and 'scheduler' in data and scheduler is not None:
        try:
            scheduler.load_state_dict(data['scheduler'])
            _logger.info("Scheduler state restored.")
            scheduler_restored = True
        except Exception as e:
            _logger.warning(f"Could not restore scheduler state: {e}. Starting with fresh scheduler.")
    elif scheduler is not None:
        _logger.warning("Scheduler state not found in checkpoint. Starting with fresh scheduler.")

    # --- Restore GradScaler State (for AMP) ---
    if isinstance(data, dict) and 'grad_scaler' in data and scaler is not None:
        try:
            scaler.load_state_dict(data['grad_scaler'])
            _logger.info("GradScaler state restored.")
        except Exception as e:
            _logger.warning(f"Could not restore GradScaler state: {e}. Starting with fresh GradScaler.")
    elif scaler is not None:
        _logger.warning("GradScaler state not found in checkpoint. Starting with fresh GradScaler.")

    # --- Warn about RNG state if present ---
    if isinstance(data, dict) and ('rng_state' in data or 'cuda_rng_state' in data):
        _logger.warning("Checkpoint contains RNG state, but it is not being restored by default.")

    # --- Final bookkeeping ---
    step = data.get('step', 0) if isinstance(data, dict) else 0
    loss = data.get('loss', np.nan) if isinstance(data, dict) else np.nan
    if step > 0:
        _logger.info(f"Restored to step {step} with loss {loss:.5f}")
    else:
        _logger.info("No previous step/loss found. Starting from step 0.")

    return step, loss, scheduler_restored



def save_training_state(model, optimizer, scheduler, scaler, step, loss, path, is_distributed=False, is_master=True):
    """ Saves the training state to a checkpoint file. Only master rank writes. """
    if not is_master:
        return

    try:
        model_to_save = model.module if is_distributed else model
        data = {
            'model': model_to_save.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'grad_scaler': scaler.state_dict(), # Save GradScaler state for AMP
            # --- RNG State Saving (Optional - see note in restore_from_checkpoint) ---
            # 'rng_state': torch.get_rng_state(),
            # 'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None, # Saves list for all GPUs
            # --- End RNG State ---
            'step': step,
            'loss': loss,
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with atomic_save(path, text_mode=False, overwrite_part=True) as f:
            torch.save(data, f)
        _logger.debug(f"Saved checkpoint to {path} at step {step}")
    except Exception as e:
        _logger.exception(f"Failed to save checkpoint {path} at step {step}")


def _train(this_directory,
          model,
          train_loader, train_baseline_log_likelihood,
          val_loader, val_baseline_log_likelihood,
          optimizer, lr_scheduler,
          gradient_accumulation_steps, # Added for train_epoch call
          minimum_learning_rate,
          validation_metric='IG',
          validation_metrics=['IG', 'LL', 'AUC_CPU', 'NSS'],
          validation_epochs=1,
          startwith=None,
          device=None,
          is_distributed=False,
          is_master=True):
    """ Main training loop function, adapted for DDP, AMP, Grad Accum. """

    output_dir_path = Path(this_directory)
    final_checkpoint_path = output_dir_path / 'final.pth'
    finished_training = False

    if is_master:
        if final_checkpoint_path.exists():
            _logger.info(f"Final checkpoint {final_checkpoint_path} already exists. Training previously finished.")
            finished_training = True

    finished_flag = torch.tensor([int(finished_training)], device=device, dtype=torch.int)
    if is_distributed:
        dist.broadcast(finished_flag, src=0)
    if finished_flag.item() == 1:
        current_rank = dist.get_rank() if is_distributed else 0
        _logger.info(f"Rank {current_rank} exiting: Training already finished.")
        return

    if is_master:
        try:
            output_dir_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
             _logger.critical(f"Failed to create output directory {output_dir_path}: {e}")
             if is_distributed: dist.barrier()
             sys.exit(1)
    if device is None:
        raise ValueError("Device must be specified for _train function.")
    current_rank = dist.get_rank() if is_distributed else 0
    _logger.info(f"Rank {current_rank} using device: {device}")

    # Initialize GradScaler for AMP
    scaler  = amp.GradScaler('cuda')

    val_metrics_history = defaultdict(list)
    columns = ['epoch', 'timestamp', 'learning_rate', 'loss'] + [f'validation_{m}' for m in validation_metrics]
    progress_df = pd.DataFrame(columns=columns)
    step = 0
    last_train_loss = np.nan
    scheduler_restored = False

    writer = None
    if is_master:
        try:
            log_dir = output_dir_path / 'log'
            log_dir.mkdir(parents=True, exist_ok=True)
            _logger.info(f"TensorBoard logs will be written to: {log_dir}")
            writer = SummaryWriter(log_dir, flush_secs=30)
        except Exception as e:
             _logger.error(f"Failed to create TensorBoard writer: {e}. TensorBoard logging disabled.")

    checkpoint_to_load = None
    if startwith and os.path.exists(startwith):
         checkpoint_to_load = startwith
         if is_master: _logger.info(f"Attempting to restore from specified checkpoint: {startwith}")
    else:
         step_files = sorted(output_dir_path.glob('step-*.pth'))
         if step_files:
             checkpoint_to_load = step_files[-1]
             if is_master: _logger.info(f"No startwith specified, found latest checkpoint in output dir: {checkpoint_to_load}")

    if checkpoint_to_load:
        # <<< CHANGE: Pass scaler to restore_from_checkpoint >>>
        step, last_train_loss, scheduler_restored = restore_from_checkpoint(
            model, optimizer, lr_scheduler, scaler, checkpoint_to_load, device, is_distributed
        )
        # <<< END CHANGE >>>
        if is_master: _logger.info(f"Restored training state from {checkpoint_to_load} to step {step}.")
    elif startwith:
         _logger.warning(f"Start checkpoint specified ({startwith}) but not found. Starting from scratch.")
    else:
        if is_master: _logger.info("No checkpoint found or specified. Starting training from scratch.")

    log_csv_path = output_dir_path / 'log.csv'
    if is_master and os.path.exists(log_csv_path):
        try:
            loaded_df = pd.read_csv(log_csv_path)
            if 'epoch' not in loaded_df.columns and loaded_df.index.name == 'epoch':
                 loaded_df.reset_index(inplace=True)
            loaded_df['epoch'] = pd.to_numeric(loaded_df['epoch'], errors='coerce')
            loaded_df.dropna(subset=['epoch'], inplace=True)
            loaded_df['epoch'] = loaded_df['epoch'].astype(int)

            progress_df = loaded_df[loaded_df['epoch'] <= step].copy()
            for metric_name in validation_metrics:
                col = f'validation_{metric_name}'
                if col in progress_df.columns:
                     numeric_col = pd.to_numeric(progress_df[col], errors='coerce')
                     val_metrics_history[metric_name] = numeric_col.dropna().tolist()

            _logger.info(f"Loaded previous progress from {log_csv_path} up to step {step}")
        except Exception as e:
            _logger.warning(f"Could not load or parse progress from {log_csv_path}: {e}. Starting log fresh.")
            progress_df = pd.DataFrame(columns=columns)
            val_metrics_history = defaultdict(list)

    def save_and_log_step(current_step, current_loss):
        nonlocal progress_df, val_metrics_history

        _val_metrics_epoch = {}
        run_validation_this_step = (current_step % validation_epochs == 0)

        if run_validation_this_step:
            if is_master: _logger.info(f"Running validation for step {current_step}...")
            _val_metrics_epoch = eval_epoch(model, val_loader, val_baseline_log_likelihood, device, metrics=validation_metrics, is_distributed=is_distributed, is_master=is_master)
            if is_master: _logger.info(f"Validation results step {current_step}: {_val_metrics_epoch}")
        else:
            if is_master: _logger.info(f"Skipping validation for step {current_step}.")
            for m in validation_metrics: _val_metrics_epoch[m] = np.nan

        if is_master:
            if run_validation_this_step:
                for key, value in _val_metrics_epoch.items():
                    if not np.isnan(value):
                         val_metrics_history[key].append(value)

            if writer:
                try:
                    if not np.isnan(current_loss): writer.add_scalar('Loss/train', current_loss, current_step)
                    lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('Meta/learning_rate', lr, current_step)
                    m = model.module if is_distributed else model
                    # Access finalizer parameters safely
                    finalizer = getattr(m, 'finalizer', None)
                    if finalizer:
                        gauss = getattr(finalizer, 'gauss', None)
                        if gauss and hasattr(gauss, 'sigma'):
                            try:
                                writer.add_scalar('Params/sigma', gauss.sigma.item(), current_step)
                            except Exception: pass # Ignore if sigma is not a tensor or item fails
                        cb_weight = getattr(finalizer, 'center_bias_weight', None)
                        if cb_weight is not None:
                            try:
                                writer.add_scalar('Params/center_bias_weight', cb_weight.item(), current_step)
                            except Exception: pass # Ignore if not tensor or item fails

                    for key, value in _val_metrics_epoch.items():
                        if not np.isnan(value): writer.add_scalar(f'Val/{key}', value, current_step)
                except Exception as tb_err:
                    _logger.error(f"Error writing to TensorBoard: {tb_err}")

            new_row_data = {
                'epoch': current_step,
                'timestamp': datetime.utcnow().isoformat(),
                'learning_rate': optimizer.param_groups[0]['lr'],
                'loss': current_loss
            }
            for key, value in _val_metrics_epoch.items():
                new_row_data[f'validation_{key}'] = value

            if current_step in progress_df['epoch'].values:
                 progress_df = progress_df[progress_df['epoch'] != current_step]

            try:
                new_row_df = pd.DataFrame([new_row_data])
                progress_df = pd.concat([progress_df, new_row_df], ignore_index=True)
                progress_df.reset_index(drop=True, inplace=True)
            except Exception as df_err:
                _logger.error(f"Error updating progress DataFrame: {df_err}")

            _logger.info(f"Step {current_step} Summary:\n{progress_df.iloc[-1:].to_string()}")

            best_val_step = -1
            best_val_score = np.nan
            validation_metric_col = f'validation_{validation_metric}'
            if validation_metric_col in progress_df.columns and not progress_df[validation_metric_col].isnull().all():
                 progress_df[validation_metric_col] = pd.to_numeric(progress_df[validation_metric_col], errors='coerce')
                 # Determine if higher is better based on common metrics
                 higher_is_better = validation_metric in ['IG', 'AUC_CPU', 'AUC_GPU', 'NSS']
                 if higher_is_better:
                     best_val_score = progress_df[validation_metric_col].max(skipna=True)
                 else: # Assume lower is better (e.g., LL)
                      best_val_score = progress_df[validation_metric_col].min(skipna=True)

                 if not np.isnan(best_val_score):
                     # Find the first epoch that achieved the best score
                     matching_epochs = progress_df.loc[progress_df[validation_metric_col] == best_val_score, 'epoch']
                     if not matching_epochs.empty:
                         best_val_step = int(matching_epochs.iloc[0])
                         _logger.info(f"Best validation ({validation_metric}) score so far {best_val_score:.5f} occurred at step {best_val_step}")

            chkpt_path = output_dir_path / f'step-{current_step:04d}.pth'
            # Pass scaler to save its state
            save_training_state(model, optimizer, lr_scheduler, scaler, current_step, current_loss, str(chkpt_path), is_distributed, is_master)


            try:
                 with atomic_save(str(log_csv_path), text_mode=True, overwrite_part=True) as f:
                     progress_df.to_csv(f, index=False)
            except Exception as e:
                 _logger.exception(f"Failed to save progress log {log_csv_path}")

            # Clean up old checkpoints, keeping current and best validation
            all_checkpoints = sorted(output_dir_path.glob('step-*.pth'))
            for cp_path in all_checkpoints:
                try:
                    cp_step = int(cp_path.stem.split('-')[1])
                    # Keep current step and best validation step (if valid)
                    if cp_step != current_step and (best_val_step == -1 or cp_step != best_val_step):
                        cp_path.unlink()
                        _logger.debug(f"Removed old checkpoint: {cp_path}")
                except (ValueError, IndexError, OSError) as e:
                     _logger.warning(f"Could not parse step or remove checkpoint {cp_path}: {e}")

    needs_initial_eval = False
    if is_master:
        if step == 0 or (step > 0 and step not in progress_df['epoch'].values):
             _logger.info(f"Step {step} needs initial evaluation.")
             needs_initial_eval = True

    if is_distributed:
        needs_initial_eval_tensor = torch.tensor(int(needs_initial_eval), device=device, dtype=torch.int)
        dist.broadcast(needs_initial_eval_tensor, src=0)
        needs_initial_eval = bool(needs_initial_eval_tensor.item())

    if needs_initial_eval:
        save_and_log_step(step, last_train_loss)

    if step > 0 and scheduler_restored:
        try:
             # lr_scheduler.step() # Step was already called at the end of the last completed epoch
             # We only need to ensure the scheduler's internal state (_step_count) is correct.
             # Most PyTorch schedulers handle this correctly when state_dict is loaded.
             # If using a custom scheduler, might need manual adjustment here.
             if is_master: _logger.info(f"LR scheduler state restored for step {step}. Will step at end of next epoch.")
        except Exception as e:
             _logger.error(f"Error checking scheduler after restore: {e}")
    elif step > 0 and not scheduler_restored:
        if is_master: _logger.warning(f"Restored to step {step}, but scheduler state was not restored. Scheduler starts fresh.")


    if is_master: _logger.info("Starting training loop...")
    while True:
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < minimum_learning_rate:
            if is_master: _logger.info(f"Learning rate ({current_lr:.2e}) reached minimum ({minimum_learning_rate:.2e}). Stopping training.")
            break

        step += 1
        epoch_start_time = datetime.now()

        if is_distributed and hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(step)
            if hasattr(val_loader, 'sampler') and hasattr(val_loader.sampler, 'set_epoch'):
                 val_loader.sampler.set_epoch(step) # Sync validation sampler too

        if is_master: _logger.info(f"--- Starting Epoch {step} (LR: {current_lr:.2e}) ---")
        # Pass scaler and gradient_accumulation_steps to train_epoch
        last_train_loss = train_epoch(
            model, train_loader, optimizer, device, scaler,
            gradient_accumulation_steps, is_distributed, is_master
        )
        if is_master:
             epoch_duration = datetime.now() - epoch_start_time
             if np.isnan(last_train_loss):
                  _logger.warning(f"Epoch {step} finished with NaN loss. Duration: {epoch_duration}")
             else:
                  _logger.info(f"Epoch {step} finished. Rank 0 Avg Train Loss: {last_train_loss:.5f}. Duration: {epoch_duration}")

        save_and_log_step(step, last_train_loss)

        try:
            # Step scheduler AFTER saving checkpoint for the current epoch
            lr_scheduler.step()
        except Exception as e:
             _logger.error(f"Error stepping scheduler at end of step {step}: {e}")


    if is_master:
        _logger.info("Training loop finished.")
        try:
            final_path = output_dir_path / 'final.pth'
            model_to_save = model.module if is_distributed else model
            # Save only the model state_dict for the final model
            torch.save(model_to_save.state_dict(), str(final_path))
            _logger.info(f"Final model state dict saved to {final_path}")
        except Exception as e:
             _logger.exception("Failed to save final model state dict")

        # --- Find and copy best validation checkpoint ---
        best_val_step = -1
        best_val_score = np.nan
        validation_metric_col = f'validation_{validation_metric}'
        if validation_metric_col in progress_df.columns and not progress_df[validation_metric_col].isnull().all():
             progress_df[validation_metric_col] = pd.to_numeric(progress_df[validation_metric_col], errors='coerce')
             higher_is_better = validation_metric in ['IG', 'AUC_CPU', 'AUC_GPU', 'NSS']
             if higher_is_better:
                 best_val_score = progress_df[validation_metric_col].max(skipna=True)
             else:
                  best_val_score = progress_df[validation_metric_col].min(skipna=True)
             if not np.isnan(best_val_score):
                 matching_epochs = progress_df.loc[progress_df[validation_metric_col] == best_val_score, 'epoch']
                 if not matching_epochs.empty:
                     best_val_step = int(matching_epochs.iloc[0]) # Take the first epoch that hit best score

        if best_val_step > 0:
            best_chkpt_path = output_dir_path / f'step-{best_val_step:04d}.pth'
            if best_chkpt_path.exists():
                final_best_path = output_dir_path / 'final_best_val.pth'
                try:
                    # Load the full checkpoint data
                    best_data = torch.load(best_chkpt_path, map_location='cpu')
                    # Save only the model state_dict from the best checkpoint
                    if 'model' in best_data:
                         torch.save(best_data['model'], str(final_best_path))
                         _logger.info(f"Saved best validation model state dict (Step {best_val_step}, Score: {best_val_score:.5f}) to {final_best_path}")
                    else:
                         _logger.error(f"Best checkpoint {best_chkpt_path} does not contain 'model' key.")

                    # Optional: Copy full best checkpoint as well
                    # shutil.copyfile(str(best_chkpt_path), str(output_dir_path / 'full_best_val.pth'))
                    # _logger.info(f"Copied full best validation checkpoint (Step {best_val_step}) to {output_dir_path / 'full_best_val.pth'}")

                except Exception as e:
                    _logger.exception(f"Failed to save/copy best checkpoint model state {best_chkpt_path}")
            else:
                 _logger.warning(f"Best validation checkpoint file {best_chkpt_path} not found, cannot save best model separately.")

        # Optional cleanup of remaining step checkpoints
        # for step_file in output_dir_path.glob('step-*.pth'):
        #      try: step_file.unlink()
        #      except OSError as e: _logger.error(f"Failed to remove final intermediate checkpoint {step_file}: {e}")

        if writer:
            writer.close()

    if is_distributed:
        # dist.barrier(device_ids=[device.index] if device.type == 'cuda' else None)
        dist.barrier() # Removed device_ids


# -----------------------------------------------------------------------------
# MAIN EXECUTION BLOCK
# -----------------------------------------------------------------------------

def main(args):
    """ Main function to parse arguments, set up DDP, and run training stages. """
    device, rank, world, is_master, is_distributed = init_distributed()

    log_level = logging.INFO if is_master else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format=f"%(asctime)s Rank{rank} %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True # Override any root logger setup
    )
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING) # Quieten PIL logs
    _logger.setLevel(log_level) # Ensure our logger uses the correct level

    if is_master:
        _logger.info("==================================================")
        _logger.info(f"Starting Training Run: Stage '{args.stage}'")
        _logger.info("==================================================")
        _logger.info(f"Torch Version: {torch.__version__}")
        _logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available(): _logger.info(f"CUDA Version: {torch.version.cuda}")
        _logger.info(f"DDP Initialised: Rank {rank}/{world} | Master: {is_master} | Distributed: {is_distributed} | Device: {device}")
        _logger.info(f"Model: {args.model_name} | Layers: {args.layers}")
        _logger.info(f"Batch Size Per GPU: {args.batch_size}")
        _logger.info(f"Gradient Accumulation Steps: {args.gradient_accumulation_steps}")
        effective_batch_size = args.batch_size * world * args.gradient_accumulation_steps
        _logger.info(f"Effective Global Batch Size: {effective_batch_size}")
        _logger.info(f"LR: {args.lr} | Min LR: {args.min_lr}")
        _logger.info(f"Workers Per Rank: {args.num_workers}")
        _logger.info(f"Train Dir: {args.train_dir}")
        _logger.info(f"Dataset Dir: {args.dataset_dir}")
        _logger.info(f"LMDB Dir: {args.lmdb_dir}")
        _logger.info(f"Add SA Head: {args.add_sa_head}")
        _logger.info(f"Unfreeze ViT Layers: {args.unfreeze_vit_layers if args.unfreeze_vit_layers else 'None (All Frozen)'}")
        if args.unfreeze_vit_layers:
            _logger.info(f"Backbone LR (if unfrozen): {args.backbone_lr}")
        if args.stage in ['mit_spatial', 'mit_scanpath_frozen', 'mit_scanpath_full']:
            if args.fold is None:
                 _logger.critical("--fold parameter is required for MIT stages.")
                 if is_distributed: dist.barrier()
                 sys.exit(1)
            _logger.info(f"MIT Fold: {args.fold}")
        if args.num_workers == 0 and is_master:
             _logger.warning("Number of dataloader workers is 0. Data loading might be a bottleneck.")
        if args.stage == 'mit_scanpath_full' and args.lr != 1e-5:
             _logger.warning(f"Using LR={args.lr} for 'mit_scanpath_full'. Recommended LR is 1e-5.")
        if args.gradient_accumulation_steps > 1 and is_master:
             _logger.info("Gradient accumulation enabled. Optimizer step occurs every "
                          f"{args.gradient_accumulation_steps} micro-batches.")


    dataset_directory = Path(args.dataset_dir).resolve()
    train_directory = Path(args.train_dir).resolve()
    lmdb_directory = Path(args.lmdb_dir).resolve()
    if is_master:
        for p in (dataset_directory, train_directory, lmdb_directory):
             try: p.mkdir(parents=True, exist_ok=True)
             except OSError as e:
                 _logger.critical(f"Failed to create directory {p}: {e}. Check permissions.")
                 if is_distributed: dist.barrier()
                 sys.exit(1)
    if is_distributed:
        # dist.barrier(device_ids=[device.index] if device.type == 'cuda' else None)
        dist.barrier() # Removed device_ids

    try:
        # Always initialize frozen, unfreeze later if needed
        features = DinoV2Backbone(
            layers=args.layers,
            model_name=args.model_name,
            freeze=True # Start frozen
        )
    except Exception as e:
        _logger.critical(f"Failed to initialize DinoV2Backbone: {e}")
        if is_distributed: dist.barrier()
        sys.exit(1)

    # Determine readout factor based on model size
    if args.model_name in ['dinov2_vitl14', 'dinov2_vitg14']:
         readout_factor = 14 # features.patch_size # Should be 14 for L/G models
         if is_master: _logger.info(f"Using readout_factor={readout_factor} for {args.model_name}")
    else: # Assume vitb14
         readout_factor = 14 # Default from original notebook/script
         if is_master: _logger.info(f"Using default readout_factor={readout_factor} for {args.model_name}")


    # Selective Unfreezing (before DDP wrap)
    unfrozen_params_exist = False
    if args.unfreeze_vit_layers:
        if is_master: _logger.info(f"Attempting to unfreeze DINOv2 layers: {args.unfreeze_vit_layers}")
        unfrozen_count = 0
        total_backbone_params = sum(p.numel() for p in features.backbone.parameters())
        unfrozen_backbone_params_count = 0

        for name, param in features.backbone.named_parameters():
            # Layer indices are typically named like 'blocks.11...'
            # Check if the parameter's name corresponds to any of the layers to unfreeze
            should_unfreeze = False
            if name.startswith('blocks.'):
                try:
                    block_index = int(name.split('.')[1])
                    if block_index in args.unfreeze_vit_layers:
                        should_unfreeze = True
                except (IndexError, ValueError):
                    pass # Not a block parameter or unexpected format

            # Also consider unfreezing related components like norm layers or projection layers
            # Example: Unfreeze final norm layer if last block is unfrozen
            if name == 'norm' and features.backbone.norm is not None: # Handle models without final norm
                 if args.unfreeze_vit_layers and max(args.unfreeze_vit_layers) == (len(features.backbone.blocks) - 1):
                     should_unfreeze = True
                     if is_master: _logger.info(f"  - Also unfreezing final norm layer.")

            # Add other specific layers if needed (e.g., 'cls_token', 'pos_embed') - less common

            if should_unfreeze:
                 param.requires_grad = True
                 unfrozen_count += 1
                 unfrozen_backbone_params_count += param.numel()
                 if is_master: _logger.debug(f"  - Unfroze parameter: {name} (Size: {param.numel()})")
            else:
                 param.requires_grad = False # Ensure others stay frozen

        if is_master:
             _logger.info(f"Successfully unfroze {unfrozen_count} parameter tensors in DINOv2 backbone.")
             _logger.info(f"Total backbone params: {total_backbone_params:,} | Unfrozen backbone params: {unfrozen_backbone_params_count:,}")
             if unfrozen_count > 0: unfrozen_params_exist = True
             elif args.unfreeze_vit_layers: _logger.warning("Requested unfreezing, but no matching parameters found.")
    else:
         if is_master: _logger.info("Keeping DINOv2 backbone fully frozen.")


    C_in = len(features.layers) * features.num_channels
    if is_master: _logger.info(f"Feature extractor initialized. Input channels to saliency network: {C_in}")

    # ======================= STAGE DISPATCH ============================
    if args.stage == 'salicon_pretrain':
        if is_master: _logger.info("--- Preparing SALICON Pretraining Stage ---")
        if is_master: _logger.info(f"Loading SALICON data from {dataset_directory}...")
        salicon_train_loc = dataset_directory
        salicon_val_loc = dataset_directory
        if is_master:
            try:
                pysaliency.get_SALICON_train(location=salicon_train_loc)
                pysaliency.get_SALICON_val(location=salicon_val_loc)
            except Exception as e:
                _logger.critical(f"Failed to download/access SALICON data: {e}")
                if is_distributed: dist.barrier(); sys.exit(1)
        if is_distributed:
            # dist.barrier(device_ids=[device.index] if device.type == 'cuda' else None)
            dist.barrier() # Removed device_ids
        try:
            SALICON_train_stimuli, SALICON_train_fixations = pysaliency.get_SALICON_train(location=salicon_train_loc)
            SALICON_val_stimuli, SALICON_val_fixations = pysaliency.get_SALICON_val(location=salicon_val_loc)
        except Exception as e:
            _logger.critical(f"Failed to load SALICON metadata: {e}")
            if is_distributed: dist.barrier(); sys.exit(1)
        if is_master: _logger.info("SALICON data loaded.")

        if is_master: _logger.info("Initializing SALICON BaselineModel...")
        SALICON_centerbias = BaselineModel(stimuli=SALICON_train_stimuli, fixations=SALICON_train_fixations, bandwidth=0.0217, eps=2e-13, caching=False)
        train_ll_cache_file = dataset_directory / 'salicon_baseline_train_ll.pkl'
        val_ll_cache_file = dataset_directory / 'salicon_baseline_val_ll.pkl'
        train_baseline_log_likelihood, val_baseline_log_likelihood = None, None
        if is_master:
            # Load or compute train LL
            try:
                with open(train_ll_cache_file, 'rb') as f: train_baseline_log_likelihood = pickle.load(f)
                _logger.info(f"Loaded cached train baseline LL from: {train_ll_cache_file}")
            except Exception as e:
                _logger.warning(f"Train LL cache fail ({e}). Computing...");
                try:
                    train_baseline_log_likelihood = SALICON_centerbias.information_gain(SALICON_train_stimuli, SALICON_train_fixations, verbose=False, average='image')
                    with open(train_ll_cache_file, 'wb') as f: pickle.dump(train_baseline_log_likelihood, f)
                    _logger.info(f"Saved computed train baseline LL to: {train_ll_cache_file}")
                except Exception as compute_e:
                     _logger.error(f"Error computing/saving train LL cache: {compute_e}")
                     train_baseline_log_likelihood = -999.9
            # Load or compute val LL
            try:
                with open(val_ll_cache_file, 'rb') as f: val_baseline_log_likelihood = pickle.load(f)
                _logger.info(f"Loaded cached validation baseline LL from: {val_ll_cache_file}")
            except Exception as e:
                _logger.warning(f"Val LL cache fail ({e}). Computing...");
                try:
                    val_baseline_log_likelihood = SALICON_centerbias.information_gain(SALICON_val_stimuli, SALICON_val_fixations, verbose=False, average='image')
                    with open(val_ll_cache_file, 'wb') as f: pickle.dump(val_baseline_log_likelihood, f)
                    _logger.info(f"Saved computed validation baseline LL to: {val_ll_cache_file}")
                except Exception as compute_e:
                     _logger.error(f"Error computing/saving val LL cache: {compute_e}")
                     val_baseline_log_likelihood = -999.9

            if train_baseline_log_likelihood == -999.9 or val_baseline_log_likelihood == -999.9:
                _logger.critical("Failed to obtain baseline LLs.")
                if train_baseline_log_likelihood is None: train_baseline_log_likelihood = -999.9
                if val_baseline_log_likelihood is None: val_baseline_log_likelihood = -999.9
            else:
                _logger.info(f"Master Baseline LLs - Train: {train_baseline_log_likelihood:.5f}, Val: {val_baseline_log_likelihood:.5f}")

        ll_list_to_broadcast = [train_baseline_log_likelihood, val_baseline_log_likelihood]
        if is_distributed: dist.broadcast_object_list(ll_list_to_broadcast, src=0)
        train_baseline_log_likelihood, val_baseline_log_likelihood = ll_list_to_broadcast
        if train_baseline_log_likelihood == -999.9 or val_baseline_log_likelihood == -999.9:
             _logger.critical(f"Baseline LLs invalid on rank {rank}. Exiting.")
             if is_distributed: dist.barrier(); sys.exit(1)

        model = DeepGazeIII(
            features=features,
            saliency_network=build_saliency_network(C_in, add_sa_head=args.add_sa_head), # Pass SA flag
            scanpath_network=None,
            fixation_selection_network=build_fixation_selection_network(scanpath_features=0),
            downsample=1, # Keep original resolution for SALICON pretrain
            readout_factor=readout_factor, # Use determined factor
            saliency_map_factor=4,
            included_fixations=[]
        ).to(device)

        # Optimizer setup with potential parameter groups for backbone
        head_params = [p for n, p in model.named_parameters() if not n.startswith('features.backbone') and p.requires_grad]
        param_groups = [{'params': head_params, 'lr': args.lr}]
        if unfrozen_params_exist:
            backbone_params = [p for n, p in model.named_parameters() if n.startswith('features.backbone') and p.requires_grad]
            if backbone_params: # Only add group if there are actually unfrozen params
                param_groups.append({'params': backbone_params, 'lr': args.backbone_lr})
                if is_master: _logger.info(f"Created separate parameter group for unfrozen backbone layers with LR={args.backbone_lr}")
            elif is_master:
                 _logger.warning("Backbone unfreezing requested, but no unfrozen backbone parameters found for optimizer group.")
        optimizer = optim.Adam(param_groups, lr=args.lr) # Default LR applies to head params

        if is_distributed:
            # find_unused_parameters might be needed if SA head or parts of backbone are conditionally used/frozen
            model = DDP(model, device_ids=[device.index], output_device=device.index, broadcast_buffers=False, find_unused_parameters=unfrozen_params_exist or args.add_sa_head)
            if is_master: _logger.info(f"Wrapped model with DDP (find_unused_parameters={unfrozen_params_exist or args.add_sa_head}).")


        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 45, 60, 75, 90, 105, 120])
        train_loader = prepare_spatial_dataset(SALICON_train_stimuli, SALICON_train_fixations, SALICON_centerbias, args.batch_size, args.num_workers, is_distributed, is_master, device, lmdb_directory / 'SALICON_train')
        validation_loader = prepare_spatial_dataset(SALICON_val_stimuli, SALICON_val_fixations, SALICON_centerbias, args.batch_size, args.num_workers, is_distributed, is_master, device, lmdb_directory / 'SALICON_val')
        output_dir = train_directory / 'salicon_pretraining'
        _train(
            this_directory=str(output_dir), model=model,
            train_loader=train_loader, train_baseline_log_likelihood=train_baseline_log_likelihood,
            val_loader=validation_loader, val_baseline_log_likelihood=val_baseline_log_likelihood,
            optimizer=optimizer, lr_scheduler=lr_scheduler,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            minimum_learning_rate=args.min_lr,
            validation_metrics=['LL', 'IG', 'NSS', 'AUC_CPU'],
            startwith=None, device=device,
            is_distributed=is_distributed, is_master=is_master
        )
        if is_master: _logger.info("--- SALICON Pretraining Finished ---")

    elif args.stage in ['mit_spatial', 'mit_scanpath_frozen', 'mit_scanpath_full']:
        fold = args.fold
        if fold is None or not (0 <= fold < 10):
            _logger.critical("--fold required for MIT stages. Exiting.")
            if is_distributed: dist.barrier(); sys.exit(1)
        if is_master: _logger.info(f"--- Preparing MIT1003 Stage: {args.stage} (Fold {fold}) ---")

        mit_converted_stimuli_path = train_directory / 'MIT1003_twosize'
        mit_converted_stimuli_file = mit_converted_stimuli_path / 'stimuli.json'
        scanpath_cache_file = mit_converted_stimuli_path / 'scanpaths_twosize.pkl'
        needs_conversion = False
        if is_master:
            if not mit_converted_stimuli_file.exists() or not scanpath_cache_file.exists():
                needs_conversion = True
                _logger.warning(f"Converted MIT1003 data incomplete. Will convert...")
        if is_distributed:
            needs_conversion_tensor = torch.tensor(int(needs_conversion), device=device, dtype=torch.int)
            dist.broadcast(needs_conversion_tensor, src=0)
            needs_conversion = bool(needs_conversion_tensor.item())

        mit_stimuli_twosize, mit_scanpaths_twosize = None, None
        if needs_conversion:
            if is_master:
                _logger.info(f"Loading original MIT1003 from {dataset_directory}...")
                try: pysaliency.external_datasets.mit.get_mit1003_with_initial_fixation(location=dataset_directory, replace_initial_invalid_fixations=True)
                except Exception as e: _logger.critical(f"Failed to get original MIT1003: {e}"); dist.barrier(); sys.exit(1)
            if is_distributed:
                # dist.barrier(device_ids=[device.index] if device.type == 'cuda' else None)
                dist.barrier() # Removed device_ids
            try: mit_stimuli_orig, mit_scanpaths_orig = pysaliency.external_datasets.mit.get_mit1003_with_initial_fixation(location=dataset_directory, replace_initial_invalid_fixations=True)
            except Exception as e: _logger.critical(f"Failed to load original MIT1003 meta: {e}"); dist.barrier(); sys.exit(1)

            mit_stimuli_twosize = convert_stimuli(mit_stimuli_orig, mit_converted_stimuli_path, is_master, is_distributed, device)
            mit_scanpaths_twosize = convert_fixation_trains(mit_stimuli_twosize, mit_scanpaths_orig, is_master)
            if mit_stimuli_twosize is None or mit_scanpaths_twosize is None: _logger.critical("MIT1003 conversion failed."); dist.barrier(); sys.exit(1)
            if is_master:
                 try:
                     with atomic_save(str(scanpath_cache_file), text_mode=False, overwrite_part=True) as f: pickle.dump(mit_scanpaths_twosize, f)
                     _logger.info(f"Saved converted scanpaths to {scanpath_cache_file}")
                 except Exception as e: _logger.error(f"Failed to save scanpath cache: {e}")
            if is_distributed:
                # dist.barrier(device_ids=[device.index] if device.type == 'cuda' else None)
                dist.barrier() # Removed device_ids
        else:
            if is_master: _logger.info(f"Loading pre-converted MIT1003 from {mit_converted_stimuli_path}...")
            try: mit_stimuli_twosize = pysaliency.read_json(mit_converted_stimuli_file)
            except Exception as e: _logger.critical(f"Failed load stimuli JSON: {e}"); dist.barrier(); sys.exit(1)
            try:
                 with open(scanpath_cache_file, 'rb') as f: mit_scanpaths_twosize = pickle.load(f)
                 if is_master: _logger.info(f"Loaded converted scanpaths.")
            except Exception as e: _logger.critical(f"Error loading scanpath cache: {e}"); dist.barrier(); sys.exit(1)

        mit_fixations_twosize = mit_scanpaths_twosize[mit_scanpaths_twosize.lengths > 0]
        if is_master: _logger.info("Initializing MIT1003 Baseline...")
        MIT1003_centerbias = CrossvalidatedBaselineModel(mit_stimuli_twosize, mit_fixations_twosize, bandwidth=10**-1.6667673342543432, eps=10**-14.884189168516073, caching=False)
        MIT1003_stimuli_train, MIT1003_fixations_train = pysaliency.dataset_config.train_split(mit_stimuli_twosize, mit_fixations_twosize, crossval_folds=10, fold_no=fold)
        MIT1003_stimuli_val, MIT1003_fixations_val = pysaliency.dataset_config.validation_split(mit_stimuli_twosize, mit_fixations_twosize, crossval_folds=10, fold_no=fold)

        train_baseline_log_likelihood, val_baseline_log_likelihood = None, None
        if is_master:
            _logger.info(f"Computing baseline LLs for Fold {fold}...")
            try:
                train_baseline_log_likelihood = MIT1003_centerbias.information_gain(MIT1003_stimuli_train, MIT1003_fixations_train, verbose=False, average='image')
                val_baseline_log_likelihood = MIT1003_centerbias.information_gain(MIT1003_stimuli_val, MIT1003_fixations_val, verbose=False, average='image')
                _logger.info(f"Fold {fold} Master Baseline LLs - Train: {train_baseline_log_likelihood:.5f}, Val: {val_baseline_log_likelihood:.5f}")
            except Exception as e: _logger.critical(f"Failed compute baseline LLs: {e}"); train_baseline_log_likelihood, val_baseline_log_likelihood = -999.9, -999.9
        ll_list_to_broadcast = [train_baseline_log_likelihood, val_baseline_log_likelihood]
        if is_distributed: dist.broadcast_object_list(ll_list_to_broadcast, src=0)
        train_baseline_log_likelihood, val_baseline_log_likelihood = ll_list_to_broadcast
        if train_baseline_log_likelihood == -999.9 or val_baseline_log_likelihood == -999.9: _logger.critical(f"Baseline LLs invalid on rank {rank}."); dist.barrier(); sys.exit(1)

        output_dir_base = train_directory / args.stage / f'crossval-10-{fold}'
        start_checkpoint_path = None
        model = None # Define model inside stage block
        start_state_dict_path = None


        if args.stage == 'mit_spatial':
            model = DeepGazeIII(
                features=features,
                saliency_network=build_saliency_network(C_in, add_sa_head=args.add_sa_head),
                scanpath_network=None,
                fixation_selection_network=build_fixation_selection_network(scanpath_features=0),
                downsample=1, # Downsample by 2 for MIT
                readout_factor= readout_factor, #readout_factor, # Use determined factor
                saliency_map_factor=4,
                included_fixations=[]
            ).to(device)
            

            head_params = [p for n, p in model.named_parameters() if not n.startswith('features.backbone') and p.requires_grad]
            param_groups = [{'params': head_params, 'lr': args.lr}]
            if unfrozen_params_exist:
                backbone_params = [p for n, p in model.named_parameters() if n.startswith('features.backbone') and p.requires_grad]
                if backbone_params: param_groups.append({'params': backbone_params, 'lr': args.backbone_lr})
            optimizer = optim.Adam(param_groups, lr=args.lr)

            if is_distributed:
                 model = DDP(model, device_ids=[device.index], output_device=device.index, broadcast_buffers=False, find_unused_parameters=unfrozen_params_exist or args.add_sa_head)
                 _logger.info(f"Wrapped DDP MIT Spatial (find_unused={unfrozen_params_exist or args.add_sa_head}).")

            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15, 20])
            train_loader = prepare_spatial_dataset(MIT1003_stimuli_train, MIT1003_fixations_train, MIT1003_centerbias, args.batch_size, args.num_workers, is_distributed, is_master, device, lmdb_directory / f'MIT1003_train_spatial_{fold}')
            validation_loader = prepare_spatial_dataset(MIT1003_stimuli_val, MIT1003_fixations_val, MIT1003_centerbias, args.batch_size, args.num_workers, is_distributed, is_master, device, lmdb_directory / f'MIT1003_val_spatial_{fold}')
            start_checkpoint_path = train_directory / 'salicon_pretraining' / 'final.pth' # Use final state dict from pretraining
            output_dir = output_dir_base

        elif args.stage == 'mit_scanpath_frozen':
            model = DeepGazeIII(
                features=features,
                saliency_network=build_saliency_network(C_in, add_sa_head=args.add_sa_head),
                scanpath_network=build_scanpath_network(),
                fixation_selection_network=build_fixation_selection_network(scanpath_features=16),
                downsample=1, # Downsample by 2 for MIT
                readout_factor=readout_factor, # Use determined factor
                saliency_map_factor=4,
                included_fixations=[-1, -2, -3, -4]
            ).to(device)

            # Freeze early saliency layers + backbone (unless parts were unfrozen earlier)
            frozen_scopes = [
                "saliency_network.layernorm0", "saliency_network.conv0", "saliency_network.bias0",
                "saliency_network.layernorm1", "saliency_network.conv1", "saliency_network.bias1",
                # "features.backbone" # Freeze the whole backbone for this stage - Let user control backbone freezing via args
            ]
            if is_master: _logger.info(f"Freezing parameters explicitly for mit_scanpath_frozen: {frozen_scopes}")
            # Freeze specified scopes, keep backbone freezing as per args.unfreeze_vit_layers
            for name, param in model.named_parameters():
                if any(name.startswith(scope) for scope in frozen_scopes):
                     param.requires_grad = False
                elif name.startswith("features.backbone"):
                     # Check if this specific backbone param should be trainable based on user args
                     should_be_trainable = False
                     if unfrozen_params_exist:
                         if name.startswith('blocks.'):
                              try:
                                  block_index = int(name.split('.')[1])
                                  if block_index in args.unfreeze_vit_layers:
                                      should_be_trainable = True
                              except (IndexError, ValueError): pass
                         elif name == 'norm' and features.backbone.norm is not None:
                              if args.unfreeze_vit_layers and max(args.unfreeze_vit_layers) == (len(features.backbone.blocks) - 1):
                                  should_be_trainable = True
                     param.requires_grad = should_be_trainable
                # else: Keep other head params trainable by default

            # Setup optimizer with potentially separate group for backbone
            head_params = [p for n, p in model.named_parameters() if not n.startswith('features.backbone') and p.requires_grad]
            param_groups = [{'params': head_params, 'lr': args.lr}]
            if unfrozen_params_exist:
                 backbone_params = [p for n, p in model.named_parameters() if n.startswith('features.backbone') and p.requires_grad]
                 if backbone_params: param_groups.append({'params': backbone_params, 'lr': args.backbone_lr})

            optimizer = optim.Adam(param_groups, lr=args.lr)

            if is_distributed:
                 # Must use find_unused_parameters=True because parts are frozen
                 model = DDP(model, device_ids=[device.index], output_device=device.index, broadcast_buffers=False, find_unused_parameters=True)
                 _logger.info("Wrapped DDP MIT Frozen Scanpath (find_unused=True).")

            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30])
            train_loader = prepare_scanpath_dataset(MIT1003_stimuli_train, MIT1003_fixations_train, MIT1003_centerbias, args.batch_size, args.num_workers, is_distributed, is_master, device, lmdb_directory / f'MIT1003_train_scanpath_{fold}')
            validation_loader = prepare_scanpath_dataset(MIT1003_stimuli_val, MIT1003_fixations_val, MIT1003_centerbias, args.batch_size, args.num_workers, is_distributed, is_master, device, lmdb_directory / f'MIT1003_val_scanpath_{fold}')
            # Use the best validation model state dict from the previous stage
            start_checkpoint_path = train_directory / 'mit_spatial' / f'crossval-10-{fold}' / 'final_best_val.pth'
            if not start_checkpoint_path.exists():
                # Fallback to final.pth if best_val doesn't exist
                 start_checkpoint_path = train_directory / 'mit_spatial' / f'crossval-10-{fold}' / 'final.pth'
                 if is_master: _logger.warning(f"final_best_val.pth not found, falling back to {start_checkpoint_path}")

            output_dir = output_dir_base

        elif args.stage == 'mit_scanpath_full':
            model = DeepGazeIII(
                features=features,
                saliency_network=build_saliency_network(C_in, add_sa_head=args.add_sa_head),
                scanpath_network=build_scanpath_network(),
                fixation_selection_network=build_fixation_selection_network(scanpath_features=16),
                downsample=1, # Downsample by 2 for MIT
                readout_factor=readout_factor, # Use determined factor
                saliency_map_factor=4,
                included_fixations=[-1, -2, -3, -4]
            ).to(device)

            # Unfreeze everything *except* the backbone parts that were meant to stay frozen based on user args
            if is_master: _logger.info("Unfreezing all head parameters for mit_scanpath_full.")
            head_params = []
            backbone_params = []
            for name, param in model.named_parameters():
                if name.startswith('features.backbone'):
                    # Only keep backbone params trainable if explicitly unfrozen by user arg
                    should_be_trainable = False
                    if unfrozen_params_exist:
                         if name.startswith('blocks.'):
                              try:
                                  block_index = int(name.split('.')[1])
                                  if block_index in args.unfreeze_vit_layers:
                                      should_be_trainable = True
                              except (IndexError, ValueError): pass
                         elif name == 'norm' and features.backbone.norm is not None:
                              if args.unfreeze_vit_layers and max(args.unfreeze_vit_layers) == (len(features.backbone.blocks) - 1):
                                  should_be_trainable = True
                    param.requires_grad = should_be_trainable
                    if should_be_trainable:
                        backbone_params.append(param)

                else:
                    # Unfreeze all other (head) parameters
                    param.requires_grad = True
                    head_params.append(param)

            param_groups = [{'params': head_params, 'lr': args.lr}]
            if backbone_params: # Add backbone group only if non-empty
                 param_groups.append({'params': backbone_params, 'lr': args.backbone_lr})
                 if is_master: _logger.info(f"Using separate param group for backbone (LR={args.backbone_lr})")

            optimizer = optim.Adam(param_groups, lr=args.lr) # Default LR for head

            if is_distributed:
                 model = DDP(model, device_ids=[device.index], output_device=device.index, broadcast_buffers=False, find_unused_parameters=unfrozen_params_exist or args.add_sa_head)
                 _logger.info(f"Wrapped DDP MIT Full Scanpath (find_unused={unfrozen_params_exist or args.add_sa_head}).")

            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15])
            train_loader = prepare_scanpath_dataset(MIT1003_stimuli_train, MIT1003_fixations_train, MIT1003_centerbias, args.batch_size, args.num_workers, is_distributed, is_master, device, lmdb_directory / f'MIT1003_train_scanpath_{fold}')
            validation_loader = prepare_scanpath_dataset(MIT1003_stimuli_val, MIT1003_fixations_val, MIT1003_centerbias, args.batch_size, args.num_workers, is_distributed, is_master, device, lmdb_directory / f'MIT1003_val_scanpath_{fold}')
            # Use the best validation model state dict from the previous stage
            start_checkpoint_path = train_directory / 'mit_scanpath_frozen' / f'crossval-10-{fold}' / 'final_best_val.pth'
            if not start_checkpoint_path.exists():
                 start_checkpoint_path = train_directory / 'mit_scanpath_frozen' / f'crossval-10-{fold}' / 'final.pth'
                 if is_master: _logger.warning(f"final_best_val.pth not found, falling back to {start_checkpoint_path}")
            output_dir = output_dir_base

        # --- Start Training for MIT stage ---
        if not model:
             _logger.critical(f"Model was not initialized for stage {args.stage}")
             dist.barrier(); sys.exit(1)

        if start_checkpoint_path and not start_checkpoint_path.exists():
             _logger.critical(f"Required start checkpoint not found: {start_checkpoint_path}")
             dist.barrier(); sys.exit(1)
        elif start_checkpoint_path and is_master:
            # Make sure we load the state_dict, not the full checkpoint for stages > pretrain
            if args.stage != 'salicon_pretrain':
                 _logger.info(f"Loading model state_dict from: {start_checkpoint_path}")
                 start_state_dict_path = start_checkpoint_path
                 start_checkpoint_path = None # Don't pass full checkpoint path to _train
            else:
                 _logger.info(f"Starting from full checkpoint: {start_checkpoint_path}")
                 start_state_dict_path = None
        elif not start_checkpoint_path and args.stage != 'salicon_pretrain':
             _logger.warning(f"No start checkpoint specified or found for stage {args.stage}, but it usually depends on a previous stage.")
             start_state_dict_path = None

        # Load state dict if necessary (before DDP wrap if possible, but done here for simplicity)
        if start_state_dict_path:
            _logger.info(f"Loading state_dict from {start_state_dict_path} (via CPU)")
            try:
                # 1) Load entire checkpoint into CPU RAM
                cpu_state = torch.load(start_state_dict_path, map_location="cpu")

                # 2) Fix key prefixes for DDP vs non-DDP
                adjusted_state_dict = OrderedDict()
                is_ddp_state_dict = any(k.startswith("module.") for k in cpu_state.keys())
                current_model_is_ddp = isinstance(model, DDP)

                if current_model_is_ddp:
                    if not is_ddp_state_dict:
                        # add 'module.' prefix to every key
                        for k, v in cpu_state.items():
                            adjusted_state_dict[f"module.{k}"] = v
                    else:
                        adjusted_state_dict = cpu_state
                else:
                    if is_ddp_state_dict:
                        # strip off 'module.' prefix
                        for k, v in cpu_state.items():
                            if k.startswith("module."):
                                adjusted_state_dict[k[len("module."):]] = v
                            else:
                                adjusted_state_dict[k] = v
                    else:
                        adjusted_state_dict = cpu_state

                # 3) Load into model (this will copy each tensor into the correct GPU)
                missing, unexpected = model.load_state_dict(adjusted_state_dict, strict=False)
                if missing:
                    _logger.warning(f"Missing keys when loading state_dict: {missing}")
                if unexpected:
                    _logger.warning(f"Unexpected keys when loading state_dict: {unexpected}")
                _logger.info("Successfully loaded model state_dict.")
            except Exception as e:
                _logger.error(f"Failed to load state_dict from {start_state_dict_path}: {e}. Starting with initial weights.")

        if is_master: _logger.info(f"Starting training {args.stage} (Fold {fold}), output: {output_dir}")
        _train(
            this_directory=str(output_dir), model=model,
            train_loader=train_loader, train_baseline_log_likelihood=train_baseline_log_likelihood,
            val_loader=validation_loader, val_baseline_log_likelihood=val_baseline_log_likelihood,
            optimizer=optimizer, lr_scheduler=lr_scheduler,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            minimum_learning_rate=args.min_lr,
            device=device,
            startwith=str(start_checkpoint_path) if start_checkpoint_path else None, # Pass full checkpoint only if needed (pretrain)
            validation_metrics=['LL', 'IG', 'NSS', 'AUC_CPU'],
            is_distributed=is_distributed, is_master=is_master
        )
        if is_master: _logger.info(f"--- Stage {args.stage} Finished (Fold {fold}) ---")

    else:
        _logger.critical(f"Unknown stage: {args.stage}"); dist.barrier(); sys.exit(1)

    cleanup_distributed()
    if is_master:
        _logger.info("==================================================")
        _logger.info("Training script finished successfully.")
        _logger.info("==================================================")


# -----------------------------------------------------------------------------
# CLI ARGUMENT PARSER and Script Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeepGazeIII with DINOv2 Backbone (Multi-GPU via torchrun)")
    # --- Core Arguments ---
    parser.add_argument('--stage', required=True, choices=['salicon_pretrain', 'mit_spatial', 'mit_scanpath_frozen', 'mit_scanpath_full'], help='Training stage to execute.')
    parser.add_argument('--model_name', default='dinov2_vitg14', choices=['dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'], help='DINOv2 model variant.')
    parser.add_argument('--layers', type=int, nargs='+', default=[-3, -2, -1], help='Indices of transformer blocks to extract features from.')
    # --- Training Hyperparameters ---
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size *per GPU*. Effective batch size is batch_size * world_size * gradient_accumulation_steps.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of forward passes to accumulate gradients before optimizer step.')
    parser.add_argument('--lr', type=float, default=0.0005, help='Initial learning rate for the model head.')
    parser.add_argument('--backbone_lr', type=float, default=1e-5, help='Learning rate for unfrozen backbone layers (if any).')
    parser.add_argument('--min_lr', type=float, default=1e-7, help='Minimum learning rate threshold for scheduler.')
    parser.add_argument('--fold', type=int, help='Cross-validation fold for MIT1003 stages (0-9). Required for MIT stages.')
    # --- Dataloading & System ---
    parser.add_argument('--num_workers', type=int, default=None, help='Dataloader workers per rank. Default: auto (cores // world_size). Set 0 for main process loading.')
    parser.add_argument('--train_dir', default='./train_dinogaze_vitg', help='Base directory for training outputs (checkpoints, logs).')
    parser.add_argument('--dataset_dir', default='./pysaliency_datasets', help='Directory for datasets (download/cache location).')
    parser.add_argument('--lmdb_dir', default='./lmdb_cache_dinogaze_vitg', help='Directory for LMDB data caches.')
    # --- Model Modifications (Experimental) ---
    parser.add_argument('--add_sa_head', action='store_true', help='Add a SelfAttention layer before the main saliency network.')
    parser.add_argument('--unfreeze_vit_layers', type=int, nargs='+', default=[], help='Indices (e.g., 10 11 for ViT-B/14) of DINOv2 blocks to unfreeze and fine-tune.')


    args = parser.parse_args()

    # --- Automatic Worker Count ---
    world_size = int(os.environ.get("WORLD_SIZE", 1)) # Get world size early for worker calc
    if args.num_workers is None:
        try:
            # Use sched_getaffinity for more accurate core count on Linux
            cpu_count = len(os.sched_getaffinity(0))
        except AttributeError:
            cpu_count = os.cpu_count() or 1 # Fallback for non-Linux/Windows
        # Allocate workers evenly, ensure at least 0
        args.num_workers = max(0, cpu_count // world_size)
    elif args.num_workers < 0:
         args.num_workers = 0 # Treat negative as 0


    # --- LR Adjustment Warning ---
    if args.stage == 'mit_scanpath_full' and args.lr > 1e-5:
         # Note: We only warn here, the user might intentionally use a different LR.
         # The actual LR is set in the main() function logic based on stage.
         # Consider setting it directly here if the override should always happen:
         # args.lr = 1e-5
         # print(f"INFO: Overriding head LR for 'mit_scanpath_full' stage to 1e-5 (was {args.lr})")
         print(f"WARNING: Recommended head LR for 'mit_scanpath_full' is 1e-5, but got {args.lr}. Using provided value.")


    try:
        main(args)
    except KeyboardInterrupt:
        # Use logger if initialized, otherwise print
        try:
            _logger.warning("Training interrupted by user (KeyboardInterrupt). Cleaning up...")
        except NameError:
            print("Training interrupted by user (KeyboardInterrupt). Cleaning up...")
        cleanup_distributed()
        sys.exit(130) # Standard exit code for Ctrl+C
    except Exception as e:
        # Use logger if initialized, otherwise print and traceback
        try:
            _logger.critical("Unhandled exception during main execution:")
            _logger.exception(e)
        except NameError:
            print("Unhandled exception during main execution:")
            import traceback
            traceback.print_exc()
        cleanup_distributed()
        sys.exit(1) # General error exit code