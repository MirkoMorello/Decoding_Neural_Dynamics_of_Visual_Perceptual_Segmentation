#!/usr/bin/env python
"""
Multi‑GPU‑ready training script for DeepGaze III with DINOv2 backbone.
Launch with torchrun, e.g.:
    torchrun --standalone --nproc_per_node=4 train_dinogaze_ddp.py \
        --stage salicon_pretrain --batch_size 4 --num_workers 8 --model_name dinov2_vitl14 --lr 0.0005

The script falls back to single‑GPU/CPU when no distributed environment
variables are present, so you can still run it exactly like the original.
"""
import os
import sys

current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import logging
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
import json

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
import pysaliency.external_datasets.mit
from imageio.v3 import imread, imwrite
from PIL import Image
Image.MAX_IMAGE_PIXELS = None # Disable PIL decompression bomb check for large images
from pysaliency.baseline_utils import (BaselineModel,
                                    CrossvalidatedBaselineModel)
#from deepgaze_pytorch.data import AspectRatioBatchSampler
import cloudpickle as cpickle
import pysaliency
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
# --- DeepGaze III Imports -----------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# --- DeepGaze III Imports (from your local src directory) --------------------
# -----------------------------------------------------------------------------
try:
    # Data handling classes
    from src.data import (  # CHANGED
        FixationDataset, FixationMaskTransform,
        ImageDataset, ImageDatasetSampler)
    # Backbone feature extractor
    from src.dinov2_backbone import DinoV2Backbone # CHANGED
    from src.dinogaze import (build_saliency_network, build_scanpath_network, build_fixation_selection_network)
    # Custom layers used in DeepGaze models
    from src.layers import (
        Bias, Conv2dMultiInput, FlexibleScanpathHistoryEncoding,
        LayerNorm, LayerNormMultiInput, SelfAttention)
    # Core model modules
    from src.modules import DeepGazeIII, DeepGazeII # CHANGED
    # Original metric functions (CPU AUC specifically)
    from src.metrics import log_likelihood, nss, auc as auc_cpu_fn # CHANGED
    # NOTE: The original _train function is NO LONGER imported.
    # It's copied and modified within this script below.
except ImportError as e:
    actual_error_message = str(e) # Get the real Python error
    logger_instance_name = "_logger"
    if logger_instance_name in locals() and locals()[logger_instance_name] is not None:
        locals()[logger_instance_name].critical(f"PYTHON IMPORT ERROR: {actual_error_message}") # Print the real error
        locals()[logger_instance_name].critical(f"Current sys.path: {sys.path}")
        locals()[logger_instance_name].critical("Ensure 'src' is in sys.path and contains __init__.py and all required .py files with correct internal imports.")
    else:
        print(f"PYTHON IMPORT ERROR: {actual_error_message}") # Print the real error
        print(f"Current sys.path: {sys.path}")
        print("Ensure 'src' is in sys.path and contains __init__.py and all required .py files with correct internal imports.")
    sys.exit(1)

# -----------------------------------------------------------------------------
# LOGGING SETUP – Configured properly in main() after DDP init
# -----------------------------------------------------------------------------
_logger = logging.getLogger("train_dinogaze_ddp") # Get logger instance


# -----------------------------------------------------------------------------
# DATASET HELPERS – Modified for DistributedSampler support
# -----------------------------------------------------------------------------

def prepare_spatial_dataset(
    stimuli,
    fixations,
    centerbias,
    batch_size: int,        # This is the PER-GPU batch size
    num_workers: int,
    is_distributed: bool,
    is_master: bool,
    device: torch.device,
    path: Path | None = None,
    current_epoch: int = 0, # Added: current epoch for DistributedSampler
):
    """
    Prepares the DataLoader for spatial (image-based) saliency prediction,
    handling DDP with shape-aware batching correctly.
    """
    # ----------  LMDB bookkeeping  ----------
    lmdb_path_str = str(path) if path else None
    if lmdb_path_str:
        if is_master:
            try:
                if path: path.mkdir(parents=True, exist_ok=True)
                _logger.info(f"Using LMDB cache for spatial dataset at: {lmdb_path_str}")
            except OSError as e:
                _logger.error(f"Failed to create LMDB directory {path}: {e}")
                lmdb_path_str = None # Fallback
        if is_distributed:
            # dist.barrier(device_ids=[device.index] if device.type == 'cuda' else None)
            dist.barrier() # Simplified barrier call

    # ----------  build Full Dataset ----------
    full_dataset = ImageDataset(
        stimuli=stimuli,
        fixations=fixations,
        centerbias_model=centerbias,
        transform=FixationMaskTransform(sparse=False), # Use dense masks
        average="image", # Average fixations per image
        lmdb_path=lmdb_path_str,
    )

    loader: torch.utils.data.DataLoader # Type hint

    # ----------  choose samplers and build DataLoader ----------
    if is_distributed:
        # DDP Path:
        # 1. Create a DistributedSampler for the full dataset.
        distributed_sampler = torch.utils.data.DistributedSampler(
            full_dataset,
            shuffle=True,
            drop_last=True
        )
        # CRITICAL: Set the epoch for the DistributedSampler
        distributed_sampler.set_epoch(current_epoch)

        # 2. Create a Subset of the full_dataset for the current rank.
        rank_subset_indices = list(iter(distributed_sampler))
        rank_dataset_subset = torch.utils.data.Subset(full_dataset, rank_subset_indices)

        # 3. Use ImageDatasetSampler on this rank-specific subset.
        shape_aware_batch_sampler = ImageDatasetSampler(
            data_source=rank_dataset_subset, # Operates on the subset for this rank
            batch_size=batch_size,           # Per-GPU batch size
            shuffle=True                     # Shuffle items within the subset before batching by shape
        )
        # Optional: if ImageDatasetSampler has its own epoch setting
        # if hasattr(shape_aware_batch_sampler, 'set_epoch'):
        #     shape_aware_batch_sampler.set_epoch(current_epoch)

        # 4. Create the DataLoader for DDP
        loader = torch.utils.data.DataLoader(
            rank_dataset_subset,
            batch_sampler=shape_aware_batch_sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )
    else:
        # Non-Distributed Path (Single GPU or CPU):
        # Use ImageDatasetSampler on the full dataset directly.
        batch_sampler_single_gpu = ImageDatasetSampler(
            full_dataset,
            batch_size=batch_size,
            shuffle=True # Assuming this enables shuffling within ImageDatasetSampler
        )
        # Optional: if ImageDatasetSampler has its own epoch setting
        # if hasattr(batch_sampler_single_gpu, 'set_epoch'):
        #    batch_sampler_single_gpu.set_epoch(current_epoch)

        loader = torch.utils.data.DataLoader(
            full_dataset,
            batch_sampler=batch_sampler_single_gpu,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )

    return loader


def prepare_scanpath_dataset(
    stimuli,
    fixations,
    centerbias,
    batch_size: int,        # This is the PER-GPU batch size
    num_workers: int,
    is_distributed: bool,
    is_master: bool,
    device: torch.device,   # device is kept for consistency, used by barrier
    path: Path | None = None,
    current_epoch: int = 0, # Added: current epoch for DistributedSampler
):
    """
    Prepares the DataLoader for scanpath prediction (fixation-based),
    handling DDP with shape-aware batching correctly.
    """
    lmdb_path_str = str(path) if path else None
    if lmdb_path_str:
        if is_master:
            try:
                if path: path.mkdir(parents=True, exist_ok=True)
                _logger.info(f"Using LMDB cache for scanpath dataset at: {lmdb_path_str}")
            except OSError as e:
                _logger.error(f"Failed to create LMDB directory {path}: {e}")
                lmdb_path_str = None # Fallback
        if is_distributed:
            # The original barrier call didn't use device_ids if device was CPU.
            # dist.barrier(device_ids=[device.index] if device.type == 'cuda' else None)
            dist.barrier() # Simplified barrier call as in your provided code

    # 1. Create the full dataset instance
    full_dataset = FixationDataset(
        stimuli=stimuli,
        fixations=fixations,
        centerbias_model=centerbias,
        included_fixations=[-1, -2, -3, -4],
        allow_missing_fixations=True,
        transform=FixationMaskTransform(sparse=False),
        average="image",
        lmdb_path=lmdb_path_str,
    )

    loader: torch.utils.data.DataLoader # Type hint for clarity

    if is_distributed:
        # DDP Path:
        # 2. Create a DistributedSampler for the full dataset.
        #    drop_last=True is important for consistent batch counts across GPUs when using Subset.
        distributed_sampler = torch.utils.data.DistributedSampler(
            full_dataset,
            shuffle=True,       # Shuffles the global list of indices
            drop_last=True      # Ensures all GPUs get the same number of samples
        )
        # CRITICAL: Set the epoch for the DistributedSampler for proper shuffling each epoch
        distributed_sampler.set_epoch(current_epoch)

        # 3. Create a Subset of the full_dataset for the current rank.
        #    iter(distributed_sampler) yields the indices assigned to the current rank.
        rank_subset_indices = list(iter(distributed_sampler))
        rank_dataset_subset = torch.utils.data.Subset(full_dataset, rank_subset_indices)

        # 4. Use ImageDatasetSampler on this rank-specific subset.
        #    It will perform shape-aware batching ONLY on the data for this GPU.
        #    Set shuffle=True if ImageDatasetSampler should shuffle items within the
        #    rank's subset before forming shape-compatible batches.
        shape_aware_batch_sampler = ImageDatasetSampler(
            data_source=rank_dataset_subset, # Operates on the subset for this rank
            batch_size=batch_size,           # Per-GPU batch size
            shuffle=True                     # Shuffle items within the subset before batching by shape
        )
        # Optional: if ImageDatasetSampler itself has epoch-aware internal shuffling
        # if hasattr(shape_aware_batch_sampler, 'set_epoch'):
        #     shape_aware_batch_sampler.set_epoch(current_epoch)

        # 5. Create the DataLoader for DDP
        loader = torch.utils.data.DataLoader(
            rank_dataset_subset,        # Use the subset for this rank
            batch_sampler=shape_aware_batch_sampler, # Custom batch sampler
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
            # drop_last is effectively handled by ImageDatasetSampler and DistributedSampler(drop_last=True)
        )
    else:
        # Non-Distributed Path (Single GPU or CPU):
        # Use ImageDatasetSampler on the full dataset directly, as in your "before" code.
        # This assumes ImageDatasetSampler's shuffle=True handles epoch shuffling correctly
        # for the non-DDP case, or that you call set_epoch on it if available.
        batch_sampler_single_gpu = ImageDatasetSampler(
            full_dataset,
            batch_size=batch_size,
            shuffle=True  # Assuming this enables shuffling within ImageDatasetSampler
        )
        # Optional: if ImageDatasetSampler itself has epoch-aware internal shuffling
        # if hasattr(batch_sampler_single_gpu, 'set_epoch'):
        #    batch_sampler_single_gpu.set_epoch(current_epoch)

        loader = torch.utils.data.DataLoader(
            full_dataset,
            batch_sampler=batch_sampler_single_gpu,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )

    return loader

# -----------------------------------------------------------------------------
# MIT Data Conversion Functions (Copied from original with DDP modifications)
# -----------------------------------------------------------------------------
# ── shared size helper ─────────────────────────────────────────────
def target_size(h_orig, w_orig):          # returns (width, height)
    if h_orig < w_orig:                   # landscape
        return 1024, 768
    else:                                 # portrait or square
        return 768, 1024



def convert_stimulus(input_image):
    h, w = input_image.shape[:2]
    new_w, new_h = target_size(h, w)      # <- use helper
    return np.array(
        Image.fromarray(input_image).resize((new_w, new_h), Image.BILINEAR)
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
            new_filenames.append(str(new_filename.resolve()))   # <- absolute
        except Exception as read_err:
            if is_master: _logger.exception(f"Failed to read or process stimulus {filename}")
            conversion_errors += 1
            continue

    if is_master and conversion_errors > 0:
        _logger.warning(f"Encountered {conversion_errors} errors during stimuli conversion.")

    # synchronize if distributed
    if is_distributed:
        dist.barrier()

    # persist cache and metadata
    if is_master:
        with open(new_location / "stimuli.pkl", "wb") as f:
            cpickle.dump(pysaliency.FileStimuli(new_filenames), f, protocol=pickle.HIGHEST_PROTOCOL)
        meta = {
            "filenames": new_filenames,
            "shapes":    [list(s) for s in pysaliency.FileStimuli(new_filenames).shapes],
        }
        with open(new_location / "stimuli.json", "w") as f:
            json.dump(meta, f, indent=2)

    # return a FileStimuli made from the *string* paths
    return pysaliency.FileStimuli(new_filenames)




def convert_fixation_trains(stimuli: pysaliency.FileStimuli,
                            fixations: pysaliency.FixationTrains,
                            is_master: bool) -> pysaliency.ScanpathFixations:
    """
    Rescale MIT‑1003 FixationTrains to 1024×768 / 768×1024 and return
    a pysaliency.ScanpathFixations object **with correct history & time‑stamps**.
    """

    # -------------- 1. pre‑compute per‑stimulus scale factors --------------
    orig_h = np.array([h for h, w, _ in stimuli.shapes])
    orig_w = np.array([w for h, w, _ in stimuli.shapes])
    tgt_w  = np.where(orig_h < orig_w, 1024, 768)
    tgt_h  = np.where(orig_h < orig_w,  768,1024)
    sx     = tgt_w / orig_w
    sy     = tgt_h / orig_h

    # -------------- 2. walk every scan‑path -------------------------------
    new_xs, new_ys, new_ts  = [], [], []
    new_x_hist, new_y_hist, new_dur = [], [], []

    skipped = 0
    for xs, ys, ts, ns in zip(fixations.train_xs,
                              fixations.train_ys,
                              fixations.train_ts,
                              fixations.train_ns):

        f_sx, f_sy = sx[ns], sy[ns]                       # scalar
        xs_ = np.clip(xs * f_sx, 0, tgt_w[ns]-1e-6)
        ys_ = np.clip(ys * f_sy, 0, tgt_h[ns]-1e-6)

        new_xs.append(xs_)
        new_ys.append(ys_)

        # ---------- timestamps ----------
        if ts is None or len(ts)==0:
            new_ts.append(np.full_like(xs_, np.nan))
        else:
            new_ts.append(np.asarray(ts, dtype=float))

        # ---------- scan‑path history ----------
        # history arrays are (n_fixations, 4, 2) in MIT1003; scale both coordinates
        if hasattr(fixations, "x_hist"):
            hist_x = np.clip(fixations.x_hist[skipped] * f_sx, 0, tgt_w[ns]-1e-6)
            hist_y = np.clip(fixations.y_hist[skipped] * f_sy, 0, tgt_h[ns]-1e-6)
            new_x_hist.append(hist_x)
            new_y_hist.append(hist_y)

        if hasattr(fixations, "durations"):
            new_dur.append(fixations.durations[skipped].copy())

        skipped += 1

    # -------------- 3. assemble pysaliency.Scanpaths -----------------------
    scanpaths = pysaliency.Scanpaths(xs=new_xs, ys=new_ys, ts=new_ts,
                                     n=fixations.train_ns,
                                     scanpath_attributes={"subject": fixations.train_subjects})

    # add optional arrays if we have them
    if new_x_hist:  scanpaths.x_hist = new_x_hist
    if new_y_hist:  scanpaths.y_hist = new_y_hist
    if new_dur:     scanpaths.durations = new_dur

    # -------------- 4. sanity‑check & return ------------------------------
    xx = np.concatenate(new_xs)
    yy = np.concatenate(new_ys)
    assert (xx >= 0).all() and (yy >= 0).all(), "negative coords after scaling!"
    assert (xx < tgt_w.max()).all() and (yy < tgt_h.max()).all(), "coords out of bounds!"

    if is_master:
        _logger.info(f"✅ converted {len(scanpaths)} scan‑paths "
                     f"({xx.size:,} fixations total)")

    return pysaliency.ScanpathFixations(scanpaths=scanpaths)




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
                    best_data = torch.load(best_chkpt_path, map_location='cpu', weights_only=False)
                    # Save only the model state_dict from the best checkpoint
                    if 'model' in best_data:
                        model_state = best_data['model']
                        torch.save(model_state, str(final_best_path))
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
        dist.barrier() # Wait for master

    try:
        features = DinoV2Backbone(
            layers=args.layers,
            model_name=args.model_name,
            freeze=True # Start frozen
        )
        features = features.to(device)

    except Exception as e:
        _logger.critical(f"Failed to initialize DinoV2Backbone: {e}")
        if is_distributed: dist.barrier()
        sys.exit(1)

    if args.model_name in ['dinov2_vitg14']:
        readout_factor = 14
        if is_master: _logger.info(f"Using readout_factor={readout_factor} for {args.model_name}")
    else:
        readout_factor = 7
        if is_master: _logger.info(f"Using default readout_factor={readout_factor} for {args.model_name}")

    unfrozen_params_exist = False
    if args.unfreeze_vit_layers:
        if is_master: _logger.info(f"Attempting to unfreeze DINOv2 layers: {args.unfreeze_vit_layers}")
        unfrozen_count = 0
        total_backbone_params = sum(p.numel() for p in features.backbone.parameters())
        unfrozen_backbone_params_count = 0

        for name, param in features.backbone.named_parameters():
            should_unfreeze = False
            if name.startswith('blocks.'):
                try:
                    block_index = int(name.split('.')[1])
                    if block_index in args.unfreeze_vit_layers:
                        should_unfreeze = True
                except (IndexError, ValueError):
                    pass

            if name == 'norm' and hasattr(features.backbone, 'norm') and features.backbone.norm is not None:
                if args.unfreeze_vit_layers and max(args.unfreeze_vit_layers) == (len(features.backbone.blocks) - 1):
                    should_unfreeze = True
                    if is_master: _logger.info(f"  - Also unfreezing final norm layer.")

            if should_unfreeze:
                param.requires_grad = True
                unfrozen_count += 1
                unfrozen_backbone_params_count += param.numel()
                if is_master: _logger.debug(f"  - Unfroze parameter: {name} (Size: {param.numel()})")
            else:
                param.requires_grad = False

        if is_master:
            _logger.info(f"Successfully unfroze {unfrozen_count} parameter tensors in DINOv2 backbone.")
            _logger.info(f"Total backbone params: {total_backbone_params:,} | Unfrozen backbone params: {unfrozen_backbone_params_count:,}")
            if unfrozen_count > 0: unfrozen_params_exist = True
            elif args.unfreeze_vit_layers: _logger.warning("Requested unfreezing, but no matching parameters found.")
    else:
        if is_master: _logger.info("Keeping DINOv2 backbone fully frozen.")
    
    _logger.info(f"Effective saliency stride = "
             f"{readout_factor * 1} px")

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
            dist.barrier()
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
            try:
                with open(train_ll_cache_file, 'rb') as f: train_baseline_log_likelihood = cpickle.load(f)
                _logger.info(f"Loaded cached train baseline LL from: {train_ll_cache_file}")
            except Exception as e:
                _logger.warning(f"Train LL cache fail ({e}). Computing...");
                try:
                    train_baseline_log_likelihood = SALICON_centerbias.information_gain(SALICON_train_stimuli, SALICON_train_fixations, verbose=False, average='image')
                    with open(train_ll_cache_file, 'wb') as f: cpikle.dump(train_baseline_log_likelihood, f)
                    _logger.info(f"Saved computed train baseline LL to: {train_ll_cache_file}")
                except Exception as compute_e:
                    _logger.error(f"Error computing/saving train LL cache: {compute_e}")
                    train_baseline_log_likelihood = -999.9
            try:
                with open(val_ll_cache_file, 'rb') as f: val_baseline_log_likelihood = cpickle.load(f)
                _logger.info(f"Loaded cached validation baseline LL from: {val_ll_cache_file}")
            except Exception as e:
                _logger.warning(f"Val LL cache fail ({e}). Computing...");
                try:
                    val_baseline_log_likelihood = SALICON_centerbias.information_gain(SALICON_val_stimuli, SALICON_val_fixations, verbose=False, average='image')
                    with open(val_ll_cache_file, 'wb') as f: cpikle.dump(val_baseline_log_likelihood, f)
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
        
        _logger.info(f"Building saliency network with {args.C_in} input channels. Add SA Head: {args.add_sa_head}")
        _logger.info(f"Building fixation selection network with {args.scanpath_features} scanpath features.")
        
        model = DeepGazeIII(
            features=features, # features already on device
            saliency_network=build_saliency_network(C_in, add_sa_head=args.add_sa_head),
            scanpath_network=None,
            fixation_selection_network=build_fixation_selection_network(scanpath_features=0),
            downsample=1,
            readout_factor=readout_factor,
            saliency_map_factor=4,
            included_fixations=[]
        ).to(device)

        if is_distributed:
            ddp_find_unused = unfrozen_params_exist or args.add_sa_head
            model = DDP(model, device_ids=[device.index], output_device=device.index, broadcast_buffers=False, find_unused_parameters=ddp_find_unused)
            if is_master: _logger.info(f"Wrapped model with DDP (find_unused_parameters={ddp_find_unused}).")

        head_params = [p for n, p in model.named_parameters() if not n.startswith('features.backbone') and p.requires_grad]
        param_groups = [{'params': head_params, 'lr': args.lr}]
        if unfrozen_params_exist:
            backbone_params = [p for n, p in model.named_parameters() if n.startswith('features.backbone') and p.requires_grad]
            if backbone_params:
                param_groups.append({'params': backbone_params, 'lr': args.backbone_lr})
                if is_master: _logger.info(f"Created separate parameter group for unfrozen backbone layers with LR={args.backbone_lr}")
            elif is_master and args.unfreeze_vit_layers:
                _logger.warning("Backbone unfreezing requested, but no unfrozen backbone parameters found for optimizer group.")
        optimizer = optim.Adam(param_groups, lr=args.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 45, 60, 75, 90, 105, 120])

        train_loader = prepare_spatial_dataset(SALICON_train_stimuli, SALICON_train_fixations, SALICON_centerbias, args.batch_size, args.num_workers, is_distributed, is_master, device, lmdb_directory / 'SALICON_train')
        validation_loader = prepare_spatial_dataset(SALICON_val_stimuli, SALICON_val_fixations, SALICON_centerbias, args.batch_size, args.num_workers, is_distributed, is_master, device, lmdb_directory / 'SALICON_val')

        output_dir = train_directory / 'salicon_pretraining'
        if is_master: _logger.info(f"Starting training {args.stage}, output: {output_dir}")
        _train(
            this_directory=str(output_dir), model=model,
            train_loader=train_loader, train_baseline_log_likelihood=train_baseline_log_likelihood,
            val_loader=validation_loader, val_baseline_log_likelihood=val_baseline_log_likelihood,
            optimizer=optimizer, lr_scheduler=lr_scheduler,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            minimum_learning_rate=args.min_lr,
            validation_metrics=['LL', 'IG', 'NSS', 'AUC_CPU'],
            startwith=None,
            device=device,
            is_distributed=is_distributed, is_master=is_master
        )
        if is_master: _logger.info("--- SALICON Pretraining Finished ---")

    elif args.stage in ['mit_spatial', 'mit_scanpath_frozen', 'mit_scanpath_full']:
        fold = args.fold
        if fold is None or not (0 <= fold < 10):
            _logger.critical("--fold required for MIT stages. Exiting.")
            if is_distributed: dist.barrier(); sys.exit(1)
        if is_master: _logger.info(f"--- Preparing MIT1003 Stage: {args.stage} (Fold {fold}) ---")

        # --- Determine Stage Specific Config ---
        output_dir_base = train_directory / args.stage / f'crossval-10-{fold}'

        if args.stage == 'mit_spatial':
            stage_downsample = 1
            stage_scanpath_features = 0
            stage_included_fixations = []
            prev_stage_dir = train_directory / 'salicon_pretraining'
            stage_lr = args.lr
            scheduler_milestones = [5, 10, 15, 20]
            ddp_find_unused = unfrozen_params_exist or args.add_sa_head
            apply_specific_freezing = False
        elif args.stage == 'mit_scanpath_frozen':
            stage_downsample = 1
            stage_scanpath_features = 16
            stage_included_fixations = [-1, -2, -3, -4]
            prev_stage_dir = train_directory / 'mit_spatial' / f'crossval-10-{fold}'
            stage_lr = args.lr
            scheduler_milestones = [10, 20, 30]
            ddp_find_unused = True
            apply_specific_freezing = True
        elif args.stage == 'mit_scanpath_full':
            stage_downsample = 1
            stage_scanpath_features = 16
            stage_included_fixations = [-1, -2, -3, -4]
            prev_stage_dir = train_directory / 'mit_scanpath_frozen' / f'crossval-10-{fold}'
            recommended_lr = 1e-5
            stage_lr = recommended_lr if args.lr != recommended_lr else args.lr
            if stage_lr != args.lr and is_master:
                _logger.warning(f"Overriding LR for 'mit_scanpath_full' to {stage_lr} (was {args.lr})")
            scheduler_milestones = [5, 10, 15]
            ddp_find_unused = unfrozen_params_exist or args.add_sa_head
            apply_specific_freezing = False
        else:
            _logger.critical(f"Invalid MIT stage configuration for {args.stage}")
            if is_distributed: dist.barrier(); sys.exit(1)

        mit_converted_stimuli_path = train_directory / 'MIT1003_twosize'
        mit_converted_stimuli_file = mit_converted_stimuli_path / 'stimuli.json'
        scanpath_cache_file = mit_converted_stimuli_path / 'scanpaths_twosize.pkl'
        needs_conversion = False
        if is_master:
            if not mit_converted_stimuli_file.exists() or not scanpath_cache_file.exists():
                needs_conversion = True
                _logger.warning(f"Converted MIT1003 data incomplete ({mit_converted_stimuli_file}, {scanpath_cache_file}). Will convert...")
        if is_distributed:
            needs_conversion_tensor = torch.tensor(int(needs_conversion), device=device, dtype=torch.int)
            dist.broadcast(needs_conversion_tensor, src=0)
            needs_conversion = bool(needs_conversion_tensor.item())

        mit_stimuli_twosize, mit_scanpaths_twosize = None, None
        if needs_conversion:
            if is_master:
                _logger.info(f"Loading original MIT1003 from {dataset_directory}...")
                try:
                    pysaliency.external_datasets.mit.get_mit1003_with_initial_fixation(location=dataset_directory, replace_initial_invalid_fixations=True)
                except ImportError:
                    _logger.error("pysaliency.external_datasets.mit not found. Cannot get dataset.")
                    if is_distributed: dist.barrier(); sys.exit(1)
                except Exception as e: _logger.critical(f"Failed to get original MIT1003: {e}"); dist.barrier(); sys.exit(1)
            if is_distributed:
                dist.barrier()
            try:
                mit_stimuli_orig, mit_scanpaths_orig = pysaliency.external_datasets.mit.get_mit1003_with_initial_fixation(location=dataset_directory, replace_initial_invalid_fixations=True)
            except ImportError:
                _logger.error("pysaliency.external_datasets.mit not found. Cannot load dataset.")
                if is_distributed: dist.barrier(); sys.exit(1)
            except Exception as e: _logger.critical(f"Failed to load original MIT1003 meta: {e}"); dist.barrier(); sys.exit(1)

            mit_stimuli_twosize = convert_stimuli(mit_stimuli_orig, mit_converted_stimuli_path, is_master, is_distributed, device)
            mit_scanpaths_twosize = convert_fixation_trains(mit_stimuli_orig, mit_scanpaths_orig, is_master)
            if mit_stimuli_twosize is None or mit_scanpaths_twosize is None: _logger.critical("MIT1003 conversion failed."); dist.barrier(); sys.exit(1)
            if is_master:
                try:
                    with atomic_save(str(scanpath_cache_file), text_mode=False, overwrite_part=True) as f: pickle.dump(mit_scanpaths_twosize, f)
                    _logger.info(f"Saved converted scanpaths to {scanpath_cache_file}")
                except Exception as e: _logger.error(f"Failed to save scanpath cache: {e}")
            if is_distributed:
                dist.barrier()
        else:
            if is_master: _logger.info(f"Loading pre-converted MIT1003 from {mit_converted_stimuli_path}...")
            try: 
                stimuli_pkl = mit_converted_stimuli_path / "stimuli.pkl"
                with open(stimuli_pkl, "rb") as f:
                    mit_stimuli_twosize = pickle.load(f)      # ← this is a FileStimuli object
            except Exception as e: _logger.critical(f"Failed load stimuli JSON: {e}"); dist.barrier(); sys.exit(1)
            try:
                with open(scanpath_cache_file, 'rb') as f: mit_scanpaths_twosize = cpickle.load(f)
                if is_master: _logger.info(f"Loaded converted scanpaths from {scanpath_cache_file}.")
            except Exception as e: _logger.critical(f"Error loading scanpath cache: {e}"); dist.barrier(); sys.exit(1)

        mit_fixations_twosize = mit_scanpaths_twosize[mit_scanpaths_twosize.scanpath_history_length > 0]
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

        _logger.info("Building base model on CPU for state dict loading...")
        saliency_net = build_saliency_network(C_in, add_sa_head=args.add_sa_head)
        scanpath_net = build_scanpath_network() if stage_scanpath_features > 0 else None
        fixsel_net = build_fixation_selection_network(scanpath_features=stage_scanpath_features)

        base_model = Dinogaze(
            features=features.cpu(),
            saliency_network=saliency_net,
            scanpath_network=scanpath_net,
            fixation_selection_network=fixsel_net,
            downsample=stage_downsample,
            readout_factor=readout_factor,
            saliency_map_factor=4,
            included_fixations=stage_included_fixations
        )

        start_state_dict_path = prev_stage_dir / 'final_best_val.pth'
        if not start_state_dict_path.exists():
            start_state_dict_path = prev_stage_dir / 'final.pth'
            if is_master: _logger.warning(f"'final_best_val.pth' not found in {prev_stage_dir}, falling back to 'final.pth'")

        if not start_state_dict_path.exists():
            _logger.critical(f"Required start checkpoint not found: {start_state_dict_path}")
            if is_distributed: dist.barrier(); sys.exit(1)
        else:
            _logger.info(f"Loading state_dict from {start_state_dict_path} (via CPU)")
            try:
                previous_stage_state_dict = torch.load(start_state_dict_path, map_location="cpu")
                missing, unexpected = base_model.load_state_dict(previous_stage_state_dict, strict=False)
                if is_master:
                    _logger.warning(f"Loaded previous stage state dict. Missing keys: {missing}")
                    _logger.warning(f"Loaded previous stage state dict. Unexpected keys: {unexpected}")
                    saliency_keys_missing = any(k not in previous_stage_state_dict for k in base_model.saliency_network.state_dict().keys())
                    fixsel_keys_missing = any(k not in previous_stage_state_dict for k in base_model.fixation_selection_network.state_dict().keys())
                    scanpath_keys_missing = (base_model.scanpath_network is not None and
                                            any(k not in previous_stage_state_dict for k in base_model.scanpath_network.state_dict().keys()))
                    _logger.info(f"Saliency Network keys potentially missing from loaded dict: {saliency_keys_missing}")
                    _logger.info(f"Fixation Selection Network keys potentially missing from loaded dict: {fixsel_keys_missing}")
                    if base_model.scanpath_network is not None:
                        _logger.info(f"Scanpath Network keys potentially missing from loaded dict: {scanpath_keys_missing}")
                _logger.info("Successfully loaded previous stage state_dict into base CPU model.")
            except Exception as e:
                _logger.error(f"Failed to load state_dict from {start_state_dict_path}: {e}. Starting with potentially random weights in head.")

        model = base_model.to(device)

        if apply_specific_freezing: # True only for mit_scanpath_frozen
            frozen_scopes = [
                "saliency_network.layernorm0", "saliency_network.conv0", "saliency_network.bias0",
                "saliency_network.layernorm1", "saliency_network.conv1", "saliency_network.bias1",
            ]
            if is_master: _logger.info(f"Freezing parameters explicitly for {args.stage}: {frozen_scopes}")
            for name, param in model.named_parameters():
                if any(name.startswith(scope) for scope in frozen_scopes):
                    param.requires_grad = False

        elif args.stage == 'mit_scanpath_full': # Specific logic only for this stage
            if is_master: _logger.info(f"Ensuring all head parameters are unfrozen for {args.stage}.")
            for name, param in model.named_parameters():
                if not name.startswith('features.backbone'):
                    param.requires_grad = True

        # --- Common training setup for all MIT stages ---
        if is_distributed:
            model = DDP(model, device_ids=[device.index], output_device=device.index, broadcast_buffers=False, find_unused_parameters=ddp_find_unused)
            if is_master: _logger.info(f"Wrapped DDP {args.stage} (find_unused={ddp_find_unused}).")

        head_params = [p for n, p in model.named_parameters() if not n.startswith('features.backbone') and p.requires_grad]
        param_groups = [{'params': head_params, 'lr': stage_lr}]
        if unfrozen_params_exist:
            backbone_params = [p for n, p in model.named_parameters() if n.startswith('features.backbone') and p.requires_grad]
            if backbone_params:
                param_groups.append({'params': backbone_params, 'lr': args.backbone_lr})
                if is_master: _logger.info(f"Using separate param group for backbone (LR={args.backbone_lr})")
        optimizer = optim.Adam(param_groups, lr=stage_lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_milestones)

        if args.stage == 'mit_spatial':
            train_loader = prepare_spatial_dataset(MIT1003_stimuli_train, MIT1003_fixations_train, MIT1003_centerbias, args.batch_size, args.num_workers, is_distributed, is_master, device, lmdb_directory / f'MIT1003_train_spatial_{fold}')
            validation_loader = prepare_spatial_dataset(MIT1003_stimuli_val, MIT1003_fixations_val, MIT1003_centerbias, args.batch_size, args.num_workers, is_distributed, is_master, device, lmdb_directory / f'MIT1003_val_spatial_{fold}')
        else: # mit_scanpath_frozen or mit_scanpath_full
            train_loader = prepare_scanpath_dataset(MIT1003_stimuli_train, MIT1003_fixations_train, MIT1003_centerbias, args.batch_size, args.num_workers, is_distributed, is_master, device, lmdb_directory / f'MIT1003_train_scanpath_{fold}')
            validation_loader = prepare_scanpath_dataset(MIT1003_stimuli_val, MIT1003_fixations_val, MIT1003_centerbias, args.batch_size, args.num_workers, is_distributed, is_master, device, lmdb_directory / f'MIT1003_val_scanpath_{fold}')

        output_dir = output_dir_base
        # Clean intermediate checkpoints only if starting a *new* fine-tuning stage from a *previous* stage's checkpoint
        if is_master and args.stage in ['mit_scanpath_frozen', 'mit_scanpath_full']: # Apply to stages that fine-tune from previous
            intermediate_checkpoints = list(output_dir.glob('step-*.pth'))
            previous_checkpoint_exists = start_state_dict_path and start_state_dict_path.exists()
            # Only clean if we successfully loaded a *previous* stage and found *current* stage intermediates
            if previous_checkpoint_exists and intermediate_checkpoints:
                _logger.warning(f"Starting fine-tuning stage {args.stage} from {start_state_dict_path}, "
                                f"but intermediate checkpoints found in {output_dir}. "
                                f"Removing them to ensure fresh optimizer/scheduler state.")
                for cp in intermediate_checkpoints:
                    try: cp.unlink()
                    except OSError as e: _logger.error(f"Failed to remove {cp}: {e}")
            elif not previous_checkpoint_exists:
                _logger.error(f"Previous stage checkpoint {start_state_dict_path} not found, cannot start fine-tuning!")
                if is_distributed: dist.barrier(); sys.exit(1)

        if is_master: _logger.info(f"Starting training {args.stage} (Fold {fold}), output: {output_dir}")
        _train(
            this_directory=str(output_dir), model=model,
            train_loader=train_loader, train_baseline_log_likelihood=train_baseline_log_likelihood,
            val_loader=validation_loader, val_baseline_log_likelihood=val_baseline_log_likelihood,
            optimizer=optimizer, lr_scheduler=lr_scheduler,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            minimum_learning_rate=args.min_lr,
            device=device,
            startwith=None, # _train handles resumption from within output_dir
            validation_metrics=['LL', 'IG', 'NSS', 'AUC_CPU'],
            is_distributed=is_distributed, is_master=is_master
        )
        if is_master: _logger.info(f"--- Stage {args.stage} Finished (Fold {fold}) ---")

    else:
        _logger.critical(f"Unknown stage: {args.stage}");
        if is_distributed: dist.barrier();
        sys.exit(1)

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
    parser.add_argument('--dataset_dir', default='../data/pysaliency_datasets', help='Directory for datasets (download/cache location).')
    parser.add_argument('--lmdb_dir', default='../data/lmdb_cache_dinogaze_vitg', help='Directory for LMDB data caches.')
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