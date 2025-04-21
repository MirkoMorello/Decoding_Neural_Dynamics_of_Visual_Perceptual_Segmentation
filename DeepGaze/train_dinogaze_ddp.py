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

import numpy as np
import pandas as pd    # Note: Large import
import pysaliency
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torchmetrics    # For GPU AUC etc.
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter # Needed for copied _train
from imageio.v3 import imread, imwrite
from PIL import Image
Image.MAX_IMAGE_PIXELS = None # Disable PIL decompression bomb check for large images
from pysaliency.baseline_utils import (BaselineModel,
                                       CrossvalidatedBaselineModel)
from tqdm import tqdm
from boltons.fileutils import atomic_save, mkdir_p # Needed for copied _train

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
        device = torch.device("cuda", local_rank)
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
# PATH MANGLING so we can `import deepgaze_pytorch.*` no matter where script is.
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
        LayerNorm, LayerNormMultiInput)
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

def build_saliency_network(input_channels):
    """ Builds the saliency prediction head network. """
    # Using _logger.info here is fine, it will only log on master
    _logger.info(f"Building saliency network with {input_channels} input channels.")
    # Reduced complexity slightly given potentially richer ViT features
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
        ('softplus2', nn.Softplus()), # Ensures non-negative output before final log-likelihood
    ]))

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
            dist.barrier(device_ids=[device.index] if device.type == 'cuda' else None)

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
            dist.barrier(device_ids=[device.index] if device.type == 'cuda' else None)

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
# MIT Data Conversion Functions (Copied from original with DDP modifications)
# -----------------------------------------------------------------------------
def convert_stimulus(input_image):
    """ Resizes a single stimulus image to one of two standard sizes. """
    size = input_image.shape[:2] # (height, width)
    new_size = (768, 1024) if size[0] < size[1] else (1024, 768)
    new_size_pil = tuple(list(new_size)[::-1])
    return np.array(Image.fromarray(input_image).resize(new_size_pil, Image.Resampling.BILINEAR))

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
        dist.barrier(device_ids=[device.index] if device.type == 'cuda' else None)

    try:
        return pysaliency.FileStimuli(new_filenames, store_json=is_master)
    except Exception as e:
         _logger.critical(f"Failed to create FileStimuli object after conversion: {e}")
         return None

def convert_fixation_trains(stimuli, fixations, is_master: bool):
    """ Converts fixation coordinates to match resized stimuli. """
    if stimuli is None:
        _logger.error("Stimuli conversion failed previously, cannot convert fixations.")
        return None
    if is_master: _logger.info("Converting fixation coordinates...")
    train_xs = fixations.train_xs.copy()
    train_ys = fixations.train_ys.copy()
    try:
        shapes_cache = stimuli.shapes
    except Exception as e:
        _logger.error(f"Failed to get shapes from converted stimuli: {e}")
        return None

    range_iterable = tqdm(range(len(train_xs)), desc="Converting Fixations", disable=not is_master)
    valid_indices_mask = np.ones(len(train_xs), dtype=bool)
    conversion_errors = 0

    for i in range_iterable:
        n = fixations.train_ns[i]
        try:
            size = shapes_cache[n][:2]
            if size[0] <= 0 or size[1] <= 0: raise ValueError(f"Invalid original image size {size} for stimulus index {n}")
            new_size = (768, 1024) if size[0] < size[1] else (1024, 768)
            x_factor = new_size[1] / size[1]
            y_factor = new_size[0] / size[0]
            train_xs[i] *= x_factor
            train_ys[i] *= y_factor
            if np.isnan(train_xs[i]) or np.isnan(train_ys[i]):
                raise ValueError(f"NaN result after scaling fixation {i} for stimulus index {n}")
        except (IndexError, ValueError, ZeroDivisionError, TypeError) as e:
             if is_master: _logger.warning(f"Error converting fixation {i} for stimulus index {n}: {e}. Marking as invalid.")
             train_xs[i] = np.nan
             train_ys[i] = np.nan
             valid_indices_mask[i] = False
             conversion_errors += 1

    if is_master and conversion_errors > 0:
        _logger.warning(f"Encountered {conversion_errors} errors during fixation conversion.")

    final_valid_indices = np.where(valid_indices_mask & ~np.isnan(train_xs) & ~np.isnan(train_ys))[0]
    num_filtered = len(train_xs) - len(final_valid_indices)
    if is_master and num_filtered > 0:
        _logger.warning(f"Filtered out {num_filtered} fixations due to conversion errors or invalid original data.")

    attributes_dict = {}
    try:
        for k in fixations.__attributes__:
            if k in ['subjects', 'scanpath_index']: continue
            attr_val = getattr(fixations, k)
            if hasattr(attr_val, '__len__') and hasattr(attr_val, 'copy') and len(attr_val) == len(fixations.train_xs):
                try:
                     attributes_dict[k] = attr_val.copy()[final_valid_indices]
                except IndexError:
                     if is_master: _logger.warning(f"Could not index attribute '{k}' during fixation filtering.")
            elif not (hasattr(attr_val, '__len__') and hasattr(attr_val, 'copy')):
                 try:
                     attributes_dict[k] = attr_val.copy() if hasattr(attr_val, 'copy') else attr_val
                 except Exception as copy_e:
                     if is_master: _logger.warning(f"Could not copy attribute '{k}': {copy_e}")

        return pysaliency.FixationTrains(
            train_xs=train_xs[final_valid_indices], train_ys=train_ys[final_valid_indices],
            train_ts=fixations.train_ts.copy()[final_valid_indices],
            train_ns=fixations.train_ns.copy()[final_valid_indices],
            train_subjects=fixations.train_subjects.copy()[final_valid_indices],
            attributes=attributes_dict
        )
    except Exception as e:
         _logger.error(f"Error filtering attributes or creating final FixationTrains object: {e}")
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
                target_mask_binary = (target_mask_int > 0).long()
            except Exception as e:
                 _logger.error(f"Error preparing target masks: {e}")
                 continue

            batch_weight_sum = weights.sum()

            if batch_weight_sum > 0:
                current_batch_weight = batch_weight_sum
                total_weight += current_batch_weight
                batch_process_count += 1
                for metric_name, metric_fn in metric_functions_avg.items():
                    if metric_name in total_metric_sums:
                        try:
                            mask_for_metric = target_mask_binary.float() if metric_name != 'AUC_CPU' else fixation_mask
                            value = metric_fn(log_density, mask_for_metric, weights=weights)
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
                    targets_flat = target_mask_binary.flatten()
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


def train_epoch(model, dataset, optimizer, device, is_distributed=False, is_master=True):
    """ Trains the model for one epoch. Handles DDP gradient sync. """
    model.train()
    local_losses = []
    local_batch_weights = []

    pbar_desc = "Training" + (f" (Rank {dist.get_rank()})" if is_distributed else "")
    pbar = tqdm(dataset, desc=pbar_desc, disable=not is_master, leave=False)

    for batch in pbar:
        optimizer.zero_grad()

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

            if isinstance(underlying_model, DeepGazeII):
                _logger.debug("Using DeepGazeII forward.")
                log_density = model(image, centerbias, **kwargs)

            # <<< START CORRECTED LOGIC >>>
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
            # <<< END CORRECTED LOGIC >>>

        except Exception as forward_e:
             _logger.exception("Error during training forward pass")
             continue

        if log_density is None:
             _logger.error("log_density is None after forward pass, skipping batch.")
             continue

        try:
            loss = -log_likelihood(log_density, fixation_mask, weights=weights)
            if torch.isnan(loss) or torch.isinf(loss):
                 _logger.error(f"NaN or Inf loss detected! Skipping batch.")
                 continue
        except Exception as loss_e:
             _logger.exception("Error during loss calculation")
             continue

        try:
             loss.backward()
             optimizer.step()
        except Exception as backward_e:
            _logger.exception("Error during backward/optimizer step")
            optimizer.zero_grad()
            continue

        detached_loss = loss.detach()
        local_losses.append(detached_loss.cpu().numpy())
        local_batch_weights.append(weights.detach().cpu().numpy().sum())
        if is_master and len(local_losses) > 0:
             current_avg_loss = np.nanmean(local_losses) if np.isnan(local_batch_weights).any() else np.average(local_losses, weights=local_batch_weights)
             pbar.set_description(f'Training Loss (Rank 0 Avg): {current_avg_loss:.5f}')

    if len(local_losses) > 0:
        final_avg_loss = np.nanmean(local_losses) if np.isnan(local_batch_weights).any() else np.average(local_losses, weights=local_batch_weights)
        return final_avg_loss
    else:
        return np.nan


def restore_from_checkpoint(model, optimizer, scheduler, path, device, is_distributed=False):
    """ Restores training state from a checkpoint file, handling DDP 'module.' prefix. """
    if not os.path.exists(path):
        _logger.error(f"Checkpoint path not found: {path}. Cannot restore.")
        return 0, np.nan, False # Added scheduler_restored flag

    _logger.info(f"Restoring checkpoint from: {path} (Distributed: {is_distributed})")
    try:
        data = torch.load(path, map_location=device)
    except Exception as e:
        _logger.exception(f"Failed to load checkpoint file {path}")
        return 0, np.nan, False

    # --- Restore Model State ---
    if 'model' not in data:
        _logger.error(f"Checkpoint {path} does not contain 'model' key.")
        return 0, np.nan, False
    model_state_dict = data['model']
    adjusted_state_dict = OrderedDict()
    is_ddp_checkpoint = any(k.startswith('module.') for k in model_state_dict.keys())
    current_model_is_ddp = isinstance(model, DDP)

    if current_model_is_ddp:
        if not is_ddp_checkpoint:
            _logger.warning("Current model is DDP, but checkpoint is not. Adding 'module.' prefix.")
            for k, v in model_state_dict.items():
                adjusted_state_dict[f'module.{k}'] = v
        else:
            _logger.debug("Current model and checkpoint are DDP. Ensuring 'module.' prefix.")
            for k, v in model_state_dict.items():
                if not k.startswith('module.'):
                    _logger.warning(f"Found key '{k}' without 'module.' prefix in DDP checkpoint. Adding prefix.")
                    adjusted_state_dict[f'module.{k}'] = v
                else:
                    adjusted_state_dict[k] = v
    else: # Current model is not DDP
        if is_ddp_checkpoint:
            _logger.warning("Current model is not DDP, but checkpoint is. Removing 'module.' prefix.")
            for k, v in model_state_dict.items():
                if k.startswith('module.'):
                    adjusted_state_dict[k.removeprefix('module.')] = v
                else:
                     _logger.warning(f"Found key '{k}' without 'module.' prefix in DDP checkpoint when loading to non-DDP model.")
                     adjusted_state_dict[k] = v
        else:
            adjusted_state_dict = model_state_dict

    try:
        missing_keys, unexpected_keys = model.load_state_dict(adjusted_state_dict, strict=False)
        if missing_keys: _logger.warning(f"Missing keys when loading model state: {missing_keys}")
        if unexpected_keys: _logger.warning(f"Unexpected keys when loading model state: {unexpected_keys}")
    except Exception as load_err:
         _logger.exception(f"Error loading model state_dict")
         return 0, np.nan, False

    # --- Restore Optimizer State ---
    if 'optimizer' in data and optimizer is not None:
        try:
            optimizer.load_state_dict(data['optimizer'])
            _logger.info("Optimizer state restored.")
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        except Exception as e:
            _logger.warning(f"Could not restore optimizer state: {e}. Optimizer will start fresh.")
    elif optimizer is not None:
         _logger.warning("Optimizer state not found in checkpoint. Optimizer will start fresh.")

    # --- Restore Scheduler State ---
    scheduler_restored = False
    if 'scheduler' in data and scheduler is not None:
        try:
            scheduler.load_state_dict(data['scheduler'])
            _logger.info("Scheduler state restored.")
            scheduler_restored = True
        except Exception as e:
            _logger.warning(f"Could not restore scheduler state: {e}. Scheduler will start fresh.")
    elif scheduler is not None:
        _logger.warning("Scheduler state not found in checkpoint. Scheduler will start fresh.")

    # --- Restore RNG State ---
    if 'rng_state' in data:
        try:
            torch.set_rng_state(data['rng_state'].cpu())
            if torch.cuda.is_available() and 'cuda_rng_state' in data and data['cuda_rng_state']:
                 cuda_rng_list = data['cuda_rng_state']
                 if isinstance(cuda_rng_list, list) and all(isinstance(s, torch.Tensor) for s in cuda_rng_list):
                     torch.cuda.set_rng_state_all(cuda_rng_list)
                 elif isinstance(cuda_rng_list, torch.Tensor):
                      torch.cuda.set_rng_state(cuda_rng_list, device=device.index)
                 else:
                     _logger.warning(f"CUDA RNG state in checkpoint has unexpected format: {type(cuda_rng_list)}. Skipping restore.")
            _logger.info("Global RNG state restored.")
        except Exception as e:
            _logger.warning(f"Could not restore RNG state: {e}")
    else:
         _logger.warning("RNG state not found in checkpoint.")

    step = data.get('step', 0)
    loss = data.get('loss', np.nan)
    if step > 0:
        _logger.info(f"Restored training state to step {step} with loss {loss:.5f}")
    else:
         _logger.info("Checkpoint loaded, but no previous step/loss found. Starting training from step 0.")

    return step, loss, scheduler_restored


def save_training_state(model, optimizer, scheduler, step, loss, path, is_distributed=False, is_master=True):
    """ Saves the training state to a checkpoint file. Only master rank writes. """
    if not is_master:
        return

    try:
        model_to_save = model.module if is_distributed else model
        data = {
            'model': model_to_save.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
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
          minimum_learning_rate,
          validation_metric='IG',
          validation_metrics=['IG', 'LL', 'AUC_CPU', 'NSS'],
          validation_epochs=1,
          startwith=None,
          device=None,
          is_distributed=False,
          is_master=True):
    """ Main training loop function, adapted for DDP. """

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
        step, last_train_loss, scheduler_restored = restore_from_checkpoint(
            model, optimizer, lr_scheduler, checkpoint_to_load, device, is_distributed
        )
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
                    if hasattr(m, 'finalizer'):
                        if hasattr(m.finalizer, 'gauss') and m.finalizer.gauss is not None and hasattr(m.finalizer.gauss, 'sigma'): writer.add_scalar('Params/sigma', m.finalizer.gauss.sigma.item(), current_step)
                        if hasattr(m.finalizer, 'center_bias_weight') and m.finalizer.center_bias_weight is not None: writer.add_scalar('Params/center_bias_weight', m.finalizer.center_bias_weight.item(), current_step)
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
                 higher_is_better = validation_metric in ['IG', 'AUC_CPU', 'AUC_GPU', 'NSS']
                 if higher_is_better:
                     best_val_score = progress_df[validation_metric_col].max()
                 else:
                      best_val_score = progress_df[validation_metric_col].min()

                 if not np.isnan(best_val_score):
                     matching_epochs = progress_df.loc[progress_df[validation_metric_col] == best_val_score, 'epoch']
                     if not matching_epochs.empty:
                         best_val_step = int(matching_epochs.iloc[0])
                         _logger.info(f"Best validation ({validation_metric}) score so far {best_val_score:.5f} occurred at step {best_val_step}")

            chkpt_path = output_dir_path / f'step-{current_step:04d}.pth'
            save_training_state(model, optimizer, lr_scheduler, current_step, current_loss, str(chkpt_path), is_distributed, is_master)

            try:
                 with atomic_save(str(log_csv_path), text_mode=True, overwrite_part=True) as f:
                     progress_df.to_csv(f, index=False)
            except Exception as e:
                 _logger.exception(f"Failed to save progress log {log_csv_path}")

            all_checkpoints = sorted(output_dir_path.glob('step-*.pth'))
            for cp_path in all_checkpoints:
                try:
                    cp_step = int(cp_path.stem.split('-')[1])
                    if cp_step != current_step and cp_step != best_val_step:
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
             lr_scheduler.step()
             if is_master: _logger.info(f"Stepped LR scheduler once after restoring to step {step}.")
        except Exception as e:
             _logger.error(f"Error stepping scheduler after restore: {e}")
    elif step > 0 and not scheduler_restored:
        if is_master: _logger.warning(f"Restored to step {step}, but scheduler state was not restored. Scheduler starts fresh, not stepping.")

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
                 val_loader.sampler.set_epoch(step)

        if is_master: _logger.info(f"--- Starting Epoch {step} (LR: {current_lr:.2e}) ---")
        last_train_loss = train_epoch(model, train_loader, optimizer, device, is_distributed, is_master)
        if is_master:
             epoch_duration = datetime.now() - epoch_start_time
             if np.isnan(last_train_loss):
                  _logger.warning(f"Epoch {step} finished with NaN loss. Duration: {epoch_duration}")
             else:
                  _logger.info(f"Epoch {step} finished. Rank 0 Avg Train Loss: {last_train_loss:.5f}. Duration: {epoch_duration}")

        save_and_log_step(step, last_train_loss)

        try:
             lr_scheduler.step()
        except Exception as e:
             _logger.error(f"Error stepping scheduler at end of step {step}: {e}")

    if is_master:
        _logger.info("Training loop finished.")
        try:
            final_path = output_dir_path / 'final.pth'
            model_to_save = model.module if is_distributed else model
            torch.save(model_to_save.state_dict(), str(final_path))
            _logger.info(f"Final model state dict saved to {final_path}")
        except Exception as e:
             _logger.exception("Failed to save final model state dict")

        best_val_step = -1
        best_val_score = np.nan
        validation_metric_col = f'validation_{validation_metric}'
        if validation_metric_col in progress_df.columns and not progress_df[validation_metric_col].isnull().all():
             progress_df[validation_metric_col] = pd.to_numeric(progress_df[validation_metric_col], errors='coerce')
             higher_is_better = validation_metric in ['IG', 'AUC_CPU', 'AUC_GPU', 'NSS']
             if higher_is_better:
                 best_val_score = progress_df[validation_metric_col].max()
             else:
                  best_val_score = progress_df[validation_metric_col].min()
             if not np.isnan(best_val_score):
                 matching_epochs = progress_df.loc[progress_df[validation_metric_col] == best_val_score, 'epoch']
                 if not matching_epochs.empty:
                     best_val_step = int(matching_epochs.iloc[0])

        if best_val_step > 0:
            best_chkpt_path = output_dir_path / f'step-{best_val_step:04d}.pth'
            if best_chkpt_path.exists():
                final_best_path = output_dir_path / 'final_best_val.pth'
                try:
                    shutil.copyfile(str(best_chkpt_path), str(final_best_path))
                    _logger.info(f"Copied best validation checkpoint (Step {best_val_step}, Score: {best_val_score:.5f}) to {final_best_path}")
                except Exception as e:
                    _logger.exception(f"Failed to copy best checkpoint {best_chkpt_path}")
            else:
                 _logger.warning(f"Best validation checkpoint file {best_chkpt_path} not found, cannot copy.")

        # Optional cleanup
        # for step_file in output_dir_path.glob('step-*.pth'):
        #      try: step_file.unlink()
        #      except OSError as e: _logger.error(f"Failed to remove final intermediate checkpoint {step_file}: {e}")

        if writer:
            writer.close()

    if is_distributed:
        dist.barrier(device_ids=[device.index] if device.type == 'cuda' else None)


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
        force=True
    )
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)
    _logger.setLevel(log_level)

    if is_master:
        _logger.info("==================================================")
        _logger.info(f"Starting Training Run: Stage '{args.stage}'")
        _logger.info("==================================================")
        _logger.info(f"Torch Version: {torch.__version__}")
        _logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available(): _logger.info(f"CUDA Version: {torch.version.cuda}")
        _logger.info(f"DDP Initialised: Rank {rank}/{world} | Master: {is_master} | Distributed: {is_distributed} | Device: {device}")
        _logger.info(f"Model: {args.model_name} | Layers: {args.layers}")
        _logger.info(f"Batch Size Per GPU: {args.batch_size} | Global Batch Size: {args.batch_size * world}")
        _logger.info(f"LR: {args.lr} | Min LR: {args.min_lr}")
        _logger.info(f"Workers Per Rank: {args.num_workers}")
        _logger.info(f"Train Dir: {args.train_dir}")
        _logger.info(f"Dataset Dir: {args.dataset_dir}")
        _logger.info(f"LMDB Dir: {args.lmdb_dir}")
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
    if is_distributed: dist.barrier(device_ids=[device.index] if device.type == 'cuda' else None)

    try:
        features = DinoV2Backbone(
            layers=args.layers,
            model_name=args.model_name,
            freeze=True
        )
    except Exception as e:
        _logger.critical(f"Failed to initialize DinoV2Backbone: {e}")
        if is_distributed: dist.barrier()
        sys.exit(1)

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
        if is_distributed: dist.barrier(device_ids=[device.index] if device.type == 'cuda' else None)
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

        model = DeepGazeIII(features=features, saliency_network=build_saliency_network(C_in),
                           scanpath_network=None, fixation_selection_network=build_fixation_selection_network(scanpath_features=0),
                           downsample=1, readout_factor=14, saliency_map_factor=4, included_fixations=[]).to(device)
        if is_distributed:
            model = DDP(model, device_ids=[device.index], output_device=device.index, broadcast_buffers=False)
            if is_master: _logger.info("Wrapped model with DDP.")

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90, 120, 150, 180])
        train_loader = prepare_spatial_dataset(SALICON_train_stimuli, SALICON_train_fixations, SALICON_centerbias, args.batch_size, args.num_workers, is_distributed, is_master, device, lmdb_directory / 'SALICON_train')
        validation_loader = prepare_spatial_dataset(SALICON_val_stimuli, SALICON_val_fixations, SALICON_centerbias, args.batch_size, args.num_workers, is_distributed, is_master, device, lmdb_directory / 'SALICON_val')
        output_dir = train_directory / 'salicon_pretraining'
        _train(str(output_dir), model, train_loader, train_baseline_log_likelihood, validation_loader, val_baseline_log_likelihood, optimizer, lr_scheduler, args.min_lr, validation_metrics=['LL', 'IG', 'NSS', 'AUC_CPU', 'AUC_GPU'], startwith=None, device=device, is_distributed=is_distributed, is_master=is_master)
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
            if is_distributed: dist.barrier(device_ids=[device.index] if device.type == 'cuda' else None)
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
            if is_distributed: dist.barrier(device_ids=[device.index] if device.type == 'cuda' else None)
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

        if args.stage == 'mit_spatial':
            model = DeepGazeIII(features=features, saliency_network=build_saliency_network(C_in), scanpath_network=None, fixation_selection_network=build_fixation_selection_network(scanpath_features=0), downsample=1, readout_factor=14, saliency_map_factor=4, included_fixations=[]).to(device)
            if is_distributed: model = DDP(model, device_ids=[device.index], output_device=device.index, broadcast_buffers=False); _logger.info("Wrapped DDP MIT Spatial.")
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15, 20])
            train_loader = prepare_spatial_dataset(MIT1003_stimuli_train, MIT1003_fixations_train, MIT1003_centerbias, args.batch_size, args.num_workers, is_distributed, is_master, device, lmdb_directory / f'MIT1003_train_spatial_{fold}')
            validation_loader = prepare_spatial_dataset(MIT1003_stimuli_val, MIT1003_fixations_val, MIT1003_centerbias, args.batch_size, args.num_workers, is_distributed, is_master, device, lmdb_directory / f'MIT1003_val_spatial_{fold}')
            start_checkpoint_path = train_directory / 'salicon_pretraining' / 'final.pth'
            output_dir = output_dir_base
        elif args.stage == 'mit_scanpath_frozen':
            model = DeepGazeIII(features=features, saliency_network=build_saliency_network(C_in), scanpath_network=build_scanpath_network(), fixation_selection_network=build_fixation_selection_network(scanpath_features=16), downsample=1, readout_factor=14, saliency_map_factor=4, included_fixations=[-1, -2, -3, -4]).to(device)
            frozen_scopes = ["saliency_network.layernorm0", "saliency_network.conv0", "saliency_network.bias0", "saliency_network.layernorm1", "saliency_network.conv1", "saliency_network.bias1"]
            if is_master: _logger.info(f"Freezing parameters: {frozen_scopes}")
            for name, param in model.named_parameters():
                if any(name.startswith(scope) for scope in frozen_scopes): param.requires_grad = False
            if is_distributed: model = DDP(model, device_ids=[device.index], output_device=device.index, broadcast_buffers=False, find_unused_parameters=True); _logger.info("Wrapped DDP MIT Frozen Scanpath (find_unused=True).")
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30])
            train_loader = prepare_scanpath_dataset(MIT1003_stimuli_train, MIT1003_fixations_train, MIT1003_centerbias, args.batch_size, args.num_workers, is_distributed, is_master, device, lmdb_directory / f'MIT1003_train_scanpath_{fold}')
            validation_loader = prepare_scanpath_dataset(MIT1003_stimuli_val, MIT1003_fixations_val, MIT1003_centerbias, args.batch_size, args.num_workers, is_distributed, is_master, device, lmdb_directory / f'MIT1003_val_scanpath_{fold}')
            start_checkpoint_path = train_directory / 'mit_spatial' / f'crossval-10-{fold}' / 'final.pth'
            output_dir = output_dir_base
        elif args.stage == 'mit_scanpath_full':
            model = DeepGazeIII(features=features, saliency_network=build_saliency_network(C_in), scanpath_network=build_scanpath_network(), fixation_selection_network=build_fixation_selection_network(scanpath_features=16), downsample=1, readout_factor=14, saliency_map_factor=4, included_fixations=[-1, -2, -3, -4]).to(device)
            if is_distributed: model = DDP(model, device_ids=[device.index], output_device=device.index, broadcast_buffers=False); _logger.info("Wrapped DDP MIT Full Scanpath.")
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15])
            train_loader = prepare_scanpath_dataset(MIT1003_stimuli_train, MIT1003_fixations_train, MIT1003_centerbias, args.batch_size, args.num_workers, is_distributed, is_master, device, lmdb_directory / f'MIT1003_train_scanpath_{fold}')
            validation_loader = prepare_scanpath_dataset(MIT1003_stimuli_val, MIT1003_fixations_val, MIT1003_centerbias, args.batch_size, args.num_workers, is_distributed, is_master, device, lmdb_directory / f'MIT1003_val_scanpath_{fold}')
            start_checkpoint_path = train_directory / 'mit_scanpath_frozen' / f'crossval-10-{fold}' / 'final.pth'
            output_dir = output_dir_base

        if start_checkpoint_path and not start_checkpoint_path.exists(): _logger.critical(f"Start checkpoint not found: {start_checkpoint_path}"); dist.barrier(); sys.exit(1)
        elif start_checkpoint_path and is_master: _logger.info(f"Starting from checkpoint: {start_checkpoint_path}")

        if is_master: _logger.info(f"Starting training {args.stage} (Fold {fold}), output: {output_dir}")
        _train(str(output_dir), model, train_loader, train_baseline_log_likelihood, validation_loader, val_baseline_log_likelihood, optimizer, lr_scheduler, args.min_lr, device=device, startwith=str(start_checkpoint_path) if start_checkpoint_path else None, validation_metrics=['LL', 'IG', 'NSS', 'AUC_CPU', 'AUC_GPU'], is_distributed=is_distributed, is_master=is_master)
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
    parser.add_argument('--stage', required=True, choices=['salicon_pretrain', 'mit_spatial', 'mit_scanpath_frozen', 'mit_scanpath_full'], help='Training stage to execute.')
    parser.add_argument('--model_name', default='dinov2_vitg14', choices=['dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'], help='DINOv2 model variant.')
    parser.add_argument('--layers', type=int, nargs='+', default=[-3, -2, -1], help='Indices of transformer layers to extract features from.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size *per GPU*.')
    parser.add_argument('--lr', type=float, default=0.0005, help='Initial learning rate.')
    parser.add_argument('--min_lr', type=float, default=1e-7, help='Minimum learning rate threshold.')
    parser.add_argument('--fold', type=int, help='Cross-validation fold for MIT1003 stages (0-9).')
    parser.add_argument('--num_workers', type=int, default=None, help='Dataloader workers per rank. Default: auto. Set 0 for main process loading.')
    parser.add_argument('--train_dir', default='./train_dinogaze_vitg', help='Base directory for training outputs.')
    parser.add_argument('--dataset_dir', default='./pysaliency_datasets', help='Directory for datasets.')
    parser.add_argument('--lmdb_dir', default='./lmdb_cache_dinogaze_vitg', help='Directory for LMDB caches.')

    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if args.num_workers is None:
        try: cpu_count = len(os.sched_getaffinity(0))
        except AttributeError: cpu_count = os.cpu_count() or 1
        args.num_workers = max(0, cpu_count // world_size)
    elif args.num_workers < 0: args.num_workers = 0

    if args.stage == 'mit_scanpath_full' and args.lr > 1e-5:
         args.lr = 1e-5 # Adjustment logged in main()

    try:
        main(args)
    except KeyboardInterrupt:
        try: _logger.warning("Training interrupted by user (KeyboardInterrupt). Cleaning up...")
        except Exception: print("Training interrupted by user (KeyboardInterrupt). Cleaning up...")
        cleanup_distributed()
        sys.exit(130)
    except Exception as e:
        try:
            _logger.critical("Unhandled exception during main execution:")
            _logger.exception(e)
        except Exception:
            print("Unhandled exception during main execution:")
            import traceback
            traceback.print_exc()
        cleanup_distributed()
        sys.exit(1)