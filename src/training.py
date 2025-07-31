# src/training.py
# flake8: noqa E501
# pylint: disable=not-callable, unused-import, import-error, no-name-in-module
# E501: line too long

from collections import defaultdict, OrderedDict # Added OrderedDict
from datetime import datetime
import os
# import glob # Not used in the provided snippet, but might be in your full _train
# import tempfile # Not used in the provided snippet
# from torch.serialization import safe_globals # Not used

from boltons.cacheutils import cached, LRU
from boltons.fileutils import atomic_save #, mkdir_p # mkdir_p not used here but maybe in _train
import numpy as np
import pandas as pd
# import pysaliency # Not directly used in these functions
# from pysaliency.filter_datasets import iterate_crossvalidation # Not used
# from pysaliency.plotting import visualize_distribution # Not used
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml # Not used here

from torch import amp # For Automatic Mixed Precision
import torchmetrics
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP # For type hinting
from pathlib import Path
import sys
import warnings
import contextlib # For model.no_sync()

# Assuming these are correctly imported from your project structure or deepgaze_pytorch
# Adjust paths if these are in deepgaze_pytorch and training.py is in src
try:
    from .metrics import log_likelihood, nss, auc as auc_cpu_fn # Use your metrics
    from src.modules import DeepGazeII # Example, if used
    # If DeepGazeIII is also used for type hinting:
    # from deepgaze_pytorch.modules import DeepGazeIII
except ImportError:
    # Fallback for direct execution or different structure
    from src.metrics import log_likelihood, nss, auc as auc_cpu_fn
    from src.modules import DeepGazeII

import logging

# baseline_performance = cached(LRU(max_size=3))(lambda model, *args, **kwargs: model.information_gain(*args, **kwargs))
# This seems to be for a pysaliency model object, not directly used in train/eval epoch here.

def eval_epoch(model, dataset, baseline_information_gain, device, metrics=None,
               is_distributed=False, is_master=True, logger=None):
    """ Evaluates the model for one epoch on the validation set. Handles DDP aggregation. """
    if logger is None:
        logger = logging.getLogger(__name__) # Basic fallback logger

    model.eval()
    default_metrics = ['LL', 'IG', 'NSS', 'AUC_CPU']
    if metrics is None: metrics = default_metrics
    if 'IG' in metrics and 'LL' not in metrics: metrics.append('LL') # IG needs LL
    if is_master: logger.debug(f"Evaluating with metrics: {metrics}")

    # --- Initialize Accumulators & Metrics ---
    total_metric_sums = {
        name: torch.tensor(0.0, device=device, dtype=torch.float64)
        for name in metrics if name in ['LL', 'NSS', 'AUC_CPU'] # Metrics averaged over batches
    }
    total_weight = torch.tensor(0.0, device=device, dtype=torch.float64)

    auroc_metric_gpu = None
    if 'AUC_GPU' in metrics:
        if is_master: logger.debug("Initializing GPU AUROC (binary task, max_fpr=1.0)")
        auroc_metric_gpu = torchmetrics.AUROC(task="binary", max_fpr=1.0).to(device)

    metric_functions_avg = {}
    if 'LL' in metrics: metric_functions_avg['LL'] = log_likelihood
    if 'NSS' in metrics: metric_functions_avg['NSS'] = nss
    if 'AUC_CPU' in metrics:
        if is_master: logger.debug("Using original CPU AUC function for AUC_CPU.")
        metric_functions_avg['AUC_CPU'] = auc_cpu_fn

    # --- Validation Loop ---
    oom_error_gpu_auc = False
    pbar_desc = "Validating" + (f" (Rank {dist.get_rank()})" if is_distributed and dist.is_initialized() else "")
    pbar = tqdm(dataset, desc=pbar_desc, disable=not is_master, leave=False)
    
    processed_batches_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            try:
                # --- Move batch to GPU & Pop data ---
                image = batch.pop('image').to(device, non_blocking=True)
                centerbias = batch.pop('centerbias').to(device, non_blocking=True)
                fixation_mask = batch.pop('fixation_mask').to(device, non_blocking=True) # Target for loss
                
                # --- NEW: Pop segmentation_mask ---
                segmentation_mask = batch.pop('segmentation_mask', None) # Default to None if not present
                if segmentation_mask is not None:
                    segmentation_mask = segmentation_mask.to(device, non_blocking=True)
                # --- END NEW ---

                x_hist = batch.pop('x_hist', torch.tensor([], device=device)).to(device, non_blocking=True)
                y_hist = batch.pop('y_hist', torch.tensor([], device=device)).to(device, non_blocking=True)
                weights = batch.pop('weight').to(device, non_blocking=True)
                durations = batch.pop('durations', torch.tensor([], device=device)).to(device, non_blocking=True)
                
                # Any remaining items in batch are considered extra kwargs
                # This was in your DINOv2 script, ensure it's intended.
                # If not, handle batch items explicitly.
                remaining_kwargs = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            except Exception as e:
                logger.error(f"Rank {device}: Error moving batch {batch_idx} data to device: {e}", exc_info=True)
                continue # Skip this batch

            # --- Model Forward Pass ---
            log_density = None
            try:
                # DDP models are wrapped in model.module
                underlying_model = model.module if is_distributed else model
                
                # Construct model_args ensuring order for positional args, then kwargs
                # This depends heavily on your model's forward signature
                # Assuming DeepGazeII/III signature: model(image, centerbias, x_hist*, y_hist*, durations*, segmentation_mask*, **kwargs)
                
                model_call_args = [image, centerbias]
                model_call_kwargs = {'segmentation_mask': segmentation_mask} # Pass new mask

                if hasattr(underlying_model, 'scanpath_network') and underlying_model.scanpath_network is not None:
                    # This indicates a DeepGazeIII-like model that might use scanpath history
                    model_call_kwargs['x_hist'] = x_hist
                    model_call_kwargs['y_hist'] = y_hist
                    model_call_kwargs['durations'] = durations
                
                model_call_kwargs.update(remaining_kwargs) # Add any other batch items

                # No AMP autocast during validation
                log_density = model(*model_call_args, **model_call_kwargs)

            except torch.cuda.OutOfMemoryError:
                current_rank_str = f"Rank {dist.get_rank()}" if is_distributed and dist.is_initialized() else "SingleOp"
                logger.error(f"\n!!! OOM ERROR (Validation Forward) on {current_rank_str} !!!\n"
                             f"Image: {image.shape}, Centerbias: {centerbias.shape}\n"
                             "Try reducing validation batch size or model complexity.", exc_info=True)
                # Critical error, re-raise to stop epoch or training
                # Consider how _train handles this. For now, re-raise.
                torch.cuda.empty_cache()
                raise
            except Exception:
                logger.error(f"Rank {device}: Error during validation forward pass for batch {batch_idx}.", exc_info=True)
                continue # Skip this batch

            if log_density is None:
                logger.warning(f"Rank {device}: log_density is None after forward pass for batch {batch_idx}, skipping metrics.")
                continue

            # --- Prep Targets (Dense & Binary) for metrics ---
            try:
                # fixation_mask is the target for LL, NSS, AUC
                # It might be sparse from dataloader, convert to dense for metrics
                if isinstance(fixation_mask, torch.sparse.Tensor):
                    target_mask_for_metrics = fixation_mask.to_dense()
                else:
                    target_mask_for_metrics = fixation_mask
                
                target_mask_for_metrics_int = target_mask_for_metrics.long()
                # Binary mask often used for AUC, sometimes for NSS if definition requires it
                target_mask_binary = (target_mask_for_metrics_int > 0).long()
            except Exception as e:
                logger.error(f"Rank {device}: Error preparing target masks for batch {batch_idx}: {e}", exc_info=True)
                continue

            # --- Calculate Batch Weight ---
            batch_weight_sum = weights.sum() # Sum of weights for all samples in the batch

            # --- Accumulate Batch-Averaged Metrics (LL, NSS, AUC_CPU) ---
            if batch_weight_sum > 0:
                current_batch_total_weight = batch_weight_sum # This is the sum of individual sample weights
                total_weight += current_batch_total_weight # Accumulate total weight across all batches
                processed_batches_count +=1

                for metric_name, metric_fn in metric_functions_avg.items():
                    if metric_name in total_metric_sums: # Check if this metric is requested
                        try:
                            # Use target_mask_for_metrics (dense, potentially counts) for LL, NSS, AUC_CPU
                            metric_value_batch_avg = metric_fn(log_density, target_mask_for_metrics, weights=weights)
                            
                            if not (isinstance(metric_value_batch_avg, torch.Tensor) and metric_value_batch_avg.ndim == 0):
                                if is_master: warnings.warn(
                                    f"Metric function {metric_name} did not return scalar tensor! Got: {metric_value_batch_avg}", RuntimeWarning)
                                if isinstance(metric_value_batch_avg, (int, float)):
                                     metric_value_batch_avg = torch.tensor(metric_value_batch_avg, device=device, dtype=torch.float64)
                                else:
                                     continue # Skip accumulation if conversion fails

                            # Accumulate sum of (metric * weight_sum_for_batch)
                            total_metric_sums[metric_name] += metric_value_batch_avg * current_batch_total_weight
                        except Exception as e:
                            if is_master: warnings.warn(
                                f"Error calculating metric {metric_name} for batch {batch_idx}: {e}. Skipping accumulation.", RuntimeWarning)
            
            # --- Update GPU AUC (if active and no previous OOM) ---
            if auroc_metric_gpu is not None and not oom_error_gpu_auc:
                try:
                    preds_for_auc_flat = log_density.flatten() # Predictions (log-probabilities or logits)
                    targets_for_auc_flat = target_mask_binary.flatten() # Binary targets

                    if torch.unique(targets_for_auc_flat).numel() > 1: # Check for both classes present
                        auroc_metric_gpu.update(preds_for_auc_flat, targets_for_auc_flat)
                except torch.cuda.OutOfMemoryError:
                    if is_master: warnings.warn(
                        "\n!!! OOM ERROR during GPU AUROC update !!!\nGPU AUC will be NaN. Reduce validation batch size.", RuntimeWarning)
                    oom_error_gpu_auc = True
                    del auroc_metric_gpu; auroc_metric_gpu = None # Free memory
                    torch.cuda.empty_cache()
                except Exception as e:
                    if is_master: warnings.warn(f"Error updating GPU AUROC for batch {batch_idx}: {e}. GPU AUC may be incorrect.", RuntimeWarning)

            # --- Update progress bar (master only) ---
            if is_master and total_weight.item() > 0 and processed_batches_count > 0:
                desc_suffix = ""
                if 'LL' in total_metric_sums:
                    current_avg_ll_local = (total_metric_sums['LL'] / total_weight).item()
                    desc_suffix += f' LL_local {current_avg_ll_local:.4f}'
                if 'AUC_CPU' in total_metric_sums:
                    current_avg_auc_cpu_local = (total_metric_sums['AUC_CPU'] / total_weight).item()
                    desc_suffix += f' AUC_CPU_local {current_avg_auc_cpu_local:.3f}'
                pbar.set_description(pbar_desc + desc_suffix)
    # --- End Validation Loop ---

    # --- DDP Synchronization of accumulated sums ---
    if is_distributed:
        dist.all_reduce(total_weight, op=dist.ReduceOp.SUM)
        for metric_name in total_metric_sums: # Iterate over keys present in total_metric_sums
            if metric_name in metrics : # Check if this metric was requested and accumulated
                dist.all_reduce(total_metric_sums[metric_name], op=dist.ReduceOp.SUM)

    # --- Final Metric Calculation ---
    final_metrics = {}
    total_weight_cpu_val = total_weight.item() # Get scalar value for total weight

    if total_weight_cpu_val > 0:
        # Calculate final averages for LL, NSS, AUC_CPU
        for metric_name in metrics: # Iterate over requested metrics
            if metric_name in total_metric_sums:
                final_metrics[metric_name] = (total_metric_sums[metric_name] / total_weight).item()

        # Compute final GPU AUC (if requested and didn't fail)
        if 'AUC_GPU' in metrics:
            if auroc_metric_gpu is not None and not oom_error_gpu_auc:
                try:
                    # Check if metric has accumulated state before computing
                    # This check depends on the torchmetrics version and specific metric state attributes
                    metric_has_state = False
                    if hasattr(auroc_metric_gpu, '_update_count') and auroc_metric_gpu._update_count > 0: # Older torchmetrics
                        metric_has_state = True
                    elif hasattr(auroc_metric_gpu, 'update_count') and auroc_metric_gpu.update_count > 0: # Newer torchmetrics
                        metric_has_state = True
                    elif hasattr(auroc_metric_gpu, 'preds') and hasattr(auroc_metric_gpu, 'target') and len(getattr(auroc_metric_gpu, 'preds',[])) > 0:
                        metric_has_state = True
                        
                    if metric_has_state:
                        final_metrics['AUC_GPU'] = auroc_metric_gpu.compute().item()
                    else:
                        if is_master: warnings.warn("GPU AUROC metric may not have been updated (no state). Setting AUC_GPU to NaN.", RuntimeWarning)
                        final_metrics['AUC_GPU'] = float('nan')
                except Exception as e:
                    if is_master: warnings.warn(f"Failed to compute final GPU AUC: {e}. Setting AUC_GPU to NaN.", RuntimeWarning)
                    final_metrics['AUC_GPU'] = float('nan')
            else: # OOM happened or GPU AUC was not initialized
                final_metrics['AUC_GPU'] = float('nan')
    else: # Handle zero total weight case (e.g., empty validation set)
        if is_master: logger.warning("Total weight for validation is zero. All metrics will be NaN.")
        for metric_name in metrics:
            if metric_name != 'IG': # IG is calculated from LL
                final_metrics[metric_name] = float('nan')

    # Ensure all requested metrics have a value (even if NaN)
    for metric_name in metrics:
        if metric_name != 'IG' and metric_name not in final_metrics:
            final_metrics[metric_name] = float('nan')

    # Calculate IG from LL
    if 'IG' in metrics:
        ll_value = final_metrics.get('LL', float('nan'))
        # baseline_information_gain is a scalar (already IG, so LL - baseline_LL)
        # If it's baseline_LL, then IG = LL_model - baseline_LL
        if not np.isnan(ll_value) and baseline_information_gain is not None and not np.isnan(baseline_information_gain):
            final_metrics['IG'] = ll_value - baseline_information_gain
        else:
            if is_master and (baseline_information_gain is None or np.isnan(baseline_information_gain)):
                 logger.warning("Cannot calculate IG: baseline_information_gain is None or NaN.")
            final_metrics['IG'] = float('nan')
            
    # Reset GPU metric state for next epoch
    if auroc_metric_gpu is not None:
        try:
            auroc_metric_gpu.reset()
        except Exception as e:
            if is_master: logger.warning(f"Error resetting AUROC metric: {e}")

    return final_metrics


def train_epoch(model, dataset, optimizer, device, scaler, gradient_accumulation_steps=1,
                is_distributed=False, is_master=True, logger=None):
    """ Trains the model for one epoch. Handles DDP, AMP, Grad Accum, and segmentation_mask. """
    if logger is None:
        logger = logging.getLogger(__name__) # Basic fallback logger

    model.train() # Set model to training mode
    
    # Accumulators for average loss calculation for this rank
    local_epoch_losses = []
    local_epoch_batch_weights = [] # Sum of sample weights in each batch

    # Determine effective batch size for logging
    # This is a rough estimate as batch_sampler might not always expose batch_size directly
    world_size_for_log = dist.get_world_size() if is_distributed and dist.is_initialized() else 1
    try:
        # Attempt to get batch_size from loader or its sampler
        bs_attr = getattr(dataset, 'batch_size', None)
        if bs_attr is None and hasattr(dataset, 'batch_sampler') and dataset.batch_sampler is not None:
            bs_attr = getattr(dataset.batch_sampler, 'batch_size', None)
        
        per_gpu_batch_size = bs_attr if bs_attr is not None else "Unknown"
        if isinstance(per_gpu_batch_size, int):
            global_batch_size_log_est = per_gpu_batch_size * world_size_for_log * gradient_accumulation_steps
        else:
            global_batch_size_log_est = "Unknown (per-GPU BS not found)"
    except Exception:
        global_batch_size_log_est = "Unknown (error getting BS)"

    if is_master:
        logger.debug(f"Train Epoch: Est. Global Batch Size for logging: {global_batch_size_log_est}")

    # Optimizer zero_grad is handled per accumulation cycle
    optimizer.zero_grad() # Initial zero_grad for the first cycle

    pbar_desc = "Training" + (f" (Rank {dist.get_rank()})" if is_distributed and dist.is_initialized() else "")
    pbar = tqdm(dataset, desc=pbar_desc, disable=not is_master, leave=False)
    total_batches_in_epoch = len(dataset)

    for batch_idx, batch in enumerate(pbar):
        # Determine if DDP gradient sync should happen for this micro-batch
        # Sync if it's an accumulation step OR the very last micro-batch of the epoch
        is_accumulation_boundary = (batch_idx + 1) % gradient_accumulation_steps == 0
        is_last_micro_batch_of_epoch = (batch_idx + 1) == total_batches_in_epoch
        
        # `model.no_sync()` is a context manager to disable DDP gradient sync
        # It should be used if distributed, AND it's NOT an accumulation boundary, AND it's NOT the last batch.
        needs_ddp_sync = is_accumulation_boundary or is_last_micro_batch_of_epoch
        sync_context = contextlib.nullcontext() # Default: sync
        if is_distributed and not needs_ddp_sync:
            sync_context = model.no_sync()

        with sync_context:
            try:
                # --- Move batch to GPU & Pop data ---
                image = batch.pop('image').to(device, non_blocking=True)
                centerbias = batch.pop('centerbias').to(device, non_blocking=True)
                fixation_mask = batch.pop('fixation_mask').to(device, non_blocking=True) # Target for loss
                
                # --- NEW: Pop segmentation_mask ---
                segmentation_mask = batch.pop('segmentation_mask', None)
                if segmentation_mask is not None:
                    segmentation_mask = segmentation_mask.to(device, non_blocking=True)
                # --- END NEW ---

                x_hist = batch.pop('x_hist', torch.tensor([], device=device)).to(device, non_blocking=True)
                y_hist = batch.pop('y_hist', torch.tensor([], device=device)).to(device, non_blocking=True)
                weights = batch.pop('weight').to(device, non_blocking=True)
                durations = batch.pop('durations', torch.tensor([], device=device)).to(device, non_blocking=True)
                
                remaining_kwargs = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            except Exception as e:
                logger.error(f"Rank {device}: Error moving batch {batch_idx} data to device: {e}", exc_info=True)
                # If data loading fails for a batch, critical to reset grads if in accumulation cycle
                if not needs_ddp_sync: optimizer.zero_grad(set_to_none=True)
                continue

            # --- Model Forward and Loss Calculation (with AMP) ---
            log_density = None
            loss = None # Initialize loss
            try:
                underlying_model = model.module if is_distributed else model
                
                # Construct model_args for forward pass
                model_call_args = [image, centerbias]
                model_call_kwargs = {'segmentation_mask': segmentation_mask}

                if hasattr(underlying_model, 'scanpath_network') and underlying_model.scanpath_network is not None:
                    model_call_kwargs['x_hist'] = x_hist
                    model_call_kwargs['y_hist'] = y_hist
                    model_call_kwargs['durations'] = durations
                
                model_call_kwargs.update(remaining_kwargs)

                with amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type=='cuda')):
                    log_density = model(*model_call_args, **model_call_kwargs)
                    
                    if log_density is None: # Should not happen if model forward is correct
                        logger.error(f"Rank {device}: log_density is None after model forward for batch {batch_idx}.")
                        # Skip loss calculation if log_density is None
                        raise ValueError("log_density is None from model.")


                    # Ensure fixation_mask is dense for loss calculation
                    if isinstance(fixation_mask, torch.sparse.Tensor):
                        loss_target_mask = fixation_mask.to_dense()
                    else:
                        loss_target_mask = fixation_mask
                        
                    current_loss_unscaled = -log_likelihood(log_density, loss_target_mask, weights=weights)
                    
                    # Scale loss for gradient accumulation
                    loss = current_loss_unscaled / gradient_accumulation_steps

                if torch.isnan(loss) or torch.isinf(loss):
                    logger.error(f"Rank {device}: NaN or Inf loss detected for batch {batch_idx}! Loss: {loss.item()}. Skipping backward/step.")
                    # If NaN/Inf, reset gradients accumulated so far for this step
                    # This prevents propagation of bad gradients.
                    if not needs_ddp_sync : optimizer.zero_grad(set_to_none=True)
                    loss = None # Mark loss as invalid for backward pass
                    continue # Skip to next micro-batch

            except Exception as e: # Catch errors from forward pass or loss calc
                logger.error(f"Rank {device}: Error during training forward/loss for batch {batch_idx}: {e}", exc_info=True)
                if not needs_ddp_sync : optimizer.zero_grad(set_to_none=True)
                loss = None # Mark loss as invalid
                continue

            # --- Backward pass (scaled loss) ---
            # Only proceed if loss is valid
            if loss is not None:
                try:
                    # scaler.scale(loss).backward() handles DDP sync if sync_context is nullcontext
                    scaler.scale(loss).backward()
                except Exception as e:
                    logger.error(f"Rank {device}: Error during backward pass for batch {batch_idx}: {e}", exc_info=True)
                    if not needs_ddp_sync : optimizer.zero_grad(set_to_none=True) # Reset if backward fails mid-accumulation
                    continue # Skip optimizer step for this cycle
            else: # Loss was NaN/Inf or None from previous error
                if is_master: pbar.set_description(pbar_desc + " Invalid Loss - Skipping optim step")
                # If this was supposed to be an accumulation step, we need to ensure optimizer.step isn't called
                # with potentially corrupted gradients from previous micro-batches of this cycle.
                # The `if is_accumulation_boundary or is_last_micro_batch_of_epoch:` below handles this.
                # However, if an invalid loss occurs, we might want to zero grads for the current cycle.
                if not needs_ddp_sync: optimizer.zero_grad(set_to_none=True)


        # --- Optimizer Step (at accumulation boundary or end of epoch) ---
        if needs_ddp_sync: # This is equivalent to (is_accumulation_boundary or is_last_micro_batch_of_epoch)
            if loss is not None: # Only step if the last micro-batch's loss was valid
                try:
                    # Optional: Gradient clipping (before scaler.step)
                    # scaler.unscale_(optimizer) # Unscale gradients first if clipping
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    scaler.step(optimizer) # Unscales gradients and calls optimizer.step()
                    scaler.update()        # Updates the scale for next iteration
                except Exception as e:
                    logger.error(f"Rank {device}: Error during optimizer step or scaler update for cycle ending at batch {batch_idx}: {e}", exc_info=True)
                finally:
                    # Zero gradients for the *next* accumulation cycle, regardless of step success/failure
                    optimizer.zero_grad(set_to_none=True)
            else: # Loss for the micro-batch that completed the cycle was invalid
                logger.warning(f"Rank {device}: Skipping optimizer step at batch {batch_idx} due to invalid loss in current/last micro-batch of cycle.")
                # Gradients should already be zeroed if loss was invalid, or zero them now for next cycle
                optimizer.zero_grad(set_to_none=True)


        # --- Log Loss and Update Progress Bar ---
        if loss is not None: # Only log if loss was valid for this micro-batch
            # Unscale loss for logging (loss was already divided by grad_accum_steps)
            detached_loss_value = loss.detach().item() * gradient_accumulation_steps
            local_epoch_losses.append(detached_loss_value)
            local_epoch_batch_weights.append(weights.sum().item()) # Sum of weights in this micro-batch

            if is_master and len(local_epoch_losses) > 0:
                # Calculate running average loss for the current epoch on this rank
                current_avg_loss_disp = np.average(local_epoch_losses, weights=local_epoch_batch_weights if np.sum(local_epoch_batch_weights)>0 else None)
                
                pbar_desc_str = f"{pbar_desc} Loss (R0 Avg): {current_avg_loss_disp:.4f}"
                if gradient_accumulation_steps > 1:
                    pbar_desc_str += f" (Acc {(batch_idx % gradient_accumulation_steps) + 1}/{gradient_accumulation_steps})"
                pbar.set_description(pbar_desc_str)
    # --- End Epoch Loop ---

    # Calculate average loss for the epoch for this rank
    if len(local_epoch_losses) > 0 and sum(local_epoch_batch_weights) > 0:
        epoch_avg_loss_local = np.average(local_epoch_losses, weights=local_epoch_batch_weights)
    elif len(local_epoch_losses) > 0: # Unweighted average if all batch weights were zero (unlikely)
        epoch_avg_loss_local = np.mean(local_epoch_losses)
    else: # No valid batches processed
        epoch_avg_loss_local = np.nan
        if is_master: logger.warning("No valid batches processed in train_epoch, returning NaN loss.")

    return epoch_avg_loss_local


def _extract_model_state_dict_from_checkpoint(checkpoint_content, logger):
    """
    Attempts to extract the model's state_dict from various common checkpoint structures.
    Args:
        checkpoint_content: The loaded content from a .pth file.
        logger: Logger instance.
    Returns:
        An OrderedDict if a model state_dict is found, otherwise None.
    """
    if not isinstance(checkpoint_content, dict):
        logger.info("Checkpoint content is not a dictionary. Assuming it IS the model state_dict directly.")
        # Check if it "looks" like a state_dict (all values are tensors or specific structures)
        if checkpoint_content is not None and hasattr(checkpoint_content, 'keys') and \
           all(isinstance(v, torch.Tensor) or (isinstance(v, tuple) and all(isinstance(t, torch.Tensor) for t in v))
               for v in checkpoint_content.values()):
            return checkpoint_content if isinstance(checkpoint_content, OrderedDict) else OrderedDict(checkpoint_content)
        else:
            logger.error("Checkpoint content is not a dict and does not appear to be a valid state_dict.")
            return None

    # Common top-level keys for the model's state_dict
    # Order can matter if multiple could be present, though unlikely for distinct meanings.
    potential_model_keys = ['model', 'state_dict', 'model_state_dict', 'state_dict_model', 'net', 'weights']

    for key in potential_model_keys:
        if key in checkpoint_content:
            candidate = checkpoint_content[key]
            if isinstance(candidate, dict) and \
               all(isinstance(v, torch.Tensor) or (isinstance(v, tuple) and all(isinstance(t, torch.Tensor) for t in v))
                   for v in candidate.values()): # Basic check: are values tensors?
                logger.info(f"Found model state_dict under top-level key: '{key}'")
                return candidate if isinstance(candidate, OrderedDict) else OrderedDict(candidate)
            elif isinstance(candidate, dict):
                # Check for common nested structures, e.g., checkpoint['model']['state_dict']
                logger.debug(f"Key '{key}' is a dict, checking for nested state_dict...")
                nested_potential_keys = ['state_dict', 'model_state_dict']
                for nested_key in nested_potential_keys:
                    if nested_key in candidate and isinstance(candidate[nested_key], dict) and \
                       all(isinstance(v, torch.Tensor) or (isinstance(v, tuple) and all(isinstance(t, torch.Tensor) for t in v))
                           for v in candidate[nested_key].values()):
                        logger.info(f"Found model state_dict under nested key: '{key}.{nested_key}'")
                        return candidate[nested_key] if isinstance(candidate[nested_key], OrderedDict) else OrderedDict(candidate[nested_key])

    # If no specific key worked, but the top-level checkpoint_content itself looks like a state_dict
    if all(isinstance(v, torch.Tensor) or (isinstance(v, tuple) and all(isinstance(t, torch.Tensor) for t in v))
           for v in checkpoint_content.values()):
        logger.info("No common model state_dict key found, but the top-level checkpoint content appears to be a state_dict itself.")
        return checkpoint_content if isinstance(checkpoint_content, OrderedDict) else OrderedDict(checkpoint_content)

    logger.error("Could not automatically extract a valid model state_dict from the checkpoint content.")
    return None

def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Helper to repeatedly unwrap a model from DDP and torch.compile wrappers."""
    while hasattr(model, "module") or hasattr(model, "_orig_mod"):
        if hasattr(model, "module"): # DDP wrapper
            model = model.module
        if hasattr(model, "_orig_mod"): # torch.compile wrapper
            model = model._orig_mod
    return model


def restore_from_checkpoint(model: torch.nn.Module,
                            optimizer: torch.optim.Optimizer = None,
                            scheduler: torch.optim.lr_scheduler._LRScheduler = None, # type: ignore
                            scaler: torch.cuda.amp.GradScaler = None, # type: ignore
                            path: str = None,
                            device: torch.device = None,
                            is_distributed: bool = False, # Kept for consistency, DDP handled via model instance
                            logger: logging.Logger = None,
                            load_weights_only: bool = False):
    """
    Restores training state (model, optimizer, scheduler, scaler, step, loss) from a checkpoint.
    Args:
        model: The PyTorch model instance (can be DDP wrapped).
        optimizer: The optimizer instance.
        scheduler: The LR scheduler instance.
        scaler: The GradScaler instance for AMP.
        path: Path to the checkpoint file.
        device: The device to load onto (e.g., torch.device('cuda:0')).
        is_distributed: (Deprecated here) Flag indicating if DDP is used. DDP status is inferred from `model`.
        logger: Logger instance.
    Returns:
        tuple: (step (int), loss (float), scheduler_restored (bool))
    """
    if logger is None: logger = _logger_restore # Use the module-level logger as fallback
    
    if not path or not os.path.exists(path):
        logger.error(f"Checkpoint path '{path}' not found or not specified. Cannot restore.")
        return 0, np.nan, False

    logger.info(f"Attempting to restore checkpoint from: {path}")
    try:
        # Load the entire checkpoint file first
        checkpoint_content = torch.load(path, map_location=device, weights_only=False)
    except Exception as e:
        logger.error(f"Failed to load checkpoint file '{path}'. Error: {e}", exc_info=True)
        return 0, np.nan, False
        
    # Restore RNG states first, if they exist in the checkpoint.
    # This is crucial for bit-for-bit reproducibility of data shuffling.
    if isinstance(checkpoint_content, dict):
        if 'rng_state' in checkpoint_content:
            try:
                torch.set_rng_state(checkpoint_content['rng_state'].cpu()) # RNG state must be on CPU
                logger.info("CPU RNG state restored.")
            except Exception as e:
                logger.warning(f"Could not restore CPU RNG state. Sampler sequence may differ. Error: {e}")
        
        if 'cuda_rng_state' in checkpoint_content and scaler is not None and device.type == 'cuda':
            try:
                # This should be a list of tensors, one for each CUDA device
                torch.cuda.set_rng_state_all(checkpoint_content['cuda_rng_state'])
                logger.info("CUDA RNG states restored.")
            except Exception as e:
                logger.warning(f"Could not restore CUDA RNG states. Error: {e}")

    model_state_dict = _extract_model_state_dict_from_checkpoint(checkpoint_content, logger)

    if model_state_dict:
        target_model_for_loading = _unwrap_model(model)
        try:
            missing_keys, unexpected_keys = target_model_for_loading.load_state_dict(model_state_dict, strict=False)
            if missing_keys: logger.warning(f"Model Load: Missing parameter keys: {missing_keys}")
            if unexpected_keys: logger.warning(f"Model Load: Unexpected parameter keys: {unexpected_keys}")
            logger.info("Model state successfully loaded into the unwrapped target model.")
        except Exception as e:
            logger.error(f"Error during model.load_state_dict: {e}", exc_info=True)
            return 0, np.nan, False
    else:
        logger.error("No model state_dict could be extracted. Model not restored.")
        return 0, np.nan, False
    
    if load_weights_only:
        logger.info("`load_weights_only` is True. Skipping optimizer, scheduler, and epoch state. Starting fresh.")
        return 0, np.nan, False

    scheduler_restored_flag = False
    if isinstance(checkpoint_content, dict):
        if 'optimizer' in checkpoint_content and optimizer is not None:
            try:
                optimizer.load_state_dict(checkpoint_content['optimizer'])
                logger.info("Optimizer state restored.")
                # Move optimizer state tensors to the correct device
                for state in optimizer.state.values(): 
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor): state[k] = v.to(device)
            except Exception:
                logger.warning("Could not restore optimizer state. Optimizer starts fresh.")
        
        if 'scheduler' in checkpoint_content and scheduler is not None:
            try:
                scheduler.load_state_dict(checkpoint_content['scheduler'])
                logger.info("Scheduler state restored.")
                scheduler_restored_flag = True
            except Exception:
                logger.warning("Could not restore scheduler state. Scheduler starts fresh.")
        
        if 'grad_scaler' in checkpoint_content and scaler is not None and device.type == 'cuda':
            try:
                scaler.load_state_dict(checkpoint_content['grad_scaler'])
                logger.info("GradScaler state restored.")
            except Exception:
                logger.warning("Could not restore GradScaler state. Scaler starts fresh.")

    restored_step = checkpoint_content.get('step', 0) if isinstance(checkpoint_content, dict) else 0
    restored_loss = checkpoint_content.get('loss', np.nan) if isinstance(checkpoint_content, dict) else np.nan
    
    logger.info(f"Restored to step/epoch {restored_step}. Last recorded loss: {restored_loss if not np.isnan(restored_loss) else 'N/A'}.")
    logger.info(f"Checkpoint restoration from '{path}' complete.")
    return restored_step, restored_loss, scheduler_restored_flag


def save_training_state(model, optimizer, scheduler, scaler, step, loss, path,
                        is_distributed=False, is_master=True, logger=None):
    if logger is None: logger = logging.getLogger(__name__)
    if not is_master: return

    try:
        model_to_save = model.module if is_distributed else model
        data = {
            'model': model_to_save.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'grad_scaler': scaler.state_dict(),
            'step': step,
            'loss': loss,
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True) # Ensure dir exists
        with atomic_save(path, text_mode=False, overwrite_part=True) as f:
            torch.save(data, f)
        logger.debug(f"Saved checkpoint to {path} at step {step}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint {path} at step {step}.", exc_info=True)


# --- _train function (Main Loop) ---
def _train(this_directory, model,
           train_loader, train_baseline_log_likelihood,
           val_loader, val_baseline_log_likelihood,
           optimizer, lr_scheduler,
           gradient_accumulation_steps,
           minimum_learning_rate,
           validation_metric='IG',
           validation_metrics=None,
           validation_epochs=1,
           startwith=None,
           device=None,
           is_distributed=False, is_master=True,
           logger=None,
           train_sampler=None):
    """ Main training loop. Now uses logger passed from the main script. """

    if logger is None:
        logger = logging.getLogger("train_loop")
        logger.setLevel(logging.INFO if is_master else logging.WARNING)
        if not logger.hasHandlers():
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(f"%(asctime)s Rank{dist.get_rank() if is_distributed and dist.is_initialized() else 0} %(levelname)s %(name)s: %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

    if validation_metrics is None:
        validation_metrics = ['IG', 'LL', 'AUC_CPU', 'NSS']

    output_dir_path = Path(this_directory)
    if is_master:
        logger.info(f"Output directory for this run: {output_dir_path}")
        output_dir_path.mkdir(parents=True, exist_ok=True)

    final_checkpoint_path = output_dir_path / 'final.pth'
    finished_training_flag_val = 0
    if is_master and final_checkpoint_path.exists():
        logger.info(f"Final checkpoint {final_checkpoint_path} found. Training assumed complete for this directory.")
        finished_training_flag_val = 1
    
    finished_flag_tensor = torch.tensor([finished_training_flag_val], device=device, dtype=torch.int)
    if is_distributed:
        dist.broadcast(finished_flag_tensor, src=0)
    
    if finished_flag_tensor.item() == 1:
        current_rank_str = f"Rank {dist.get_rank()}" if is_distributed and dist.is_initialized() else "SingleOp"
        logger.info(f"{current_rank_str} exiting: Training already marked as finished in {this_directory}.")
        return

    if device is None:
        logger.error("Device not specified to _train function. This is required.")
        raise ValueError("Device must be specified for _train.")
    
    current_rank_str = f"Rank {dist.get_rank()}" if is_distributed and dist.is_initialized() else "SingleOp"
    logger.info(f"{current_rank_str} using device: {device}")

    scaler = amp.GradScaler(enabled=(device.type == 'cuda'))
    if is_master and device.type == 'cuda': logger.info("AMP GradScaler initialized for CUDA.")
    elif is_master: logger.info("AMP GradScaler disabled (not using CUDA or not master).")

    val_metrics_history = defaultdict(list)
    progress_log_columns = ['epoch', 'timestamp', 'learning_rate', 'train_loss'] + \
                           [f'validation_{m}' for m in validation_metrics]
    progress_df = pd.DataFrame(columns=progress_log_columns)
    
    current_epoch_step = 0
    last_avg_train_loss_epoch = np.nan
    scheduler_state_restored = False

    tb_writer = None
    if is_master:
        tb_log_dir = output_dir_path / 'tensorboard_logs'
        try:
            tb_log_dir.mkdir(parents=True, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=str(tb_log_dir), flush_secs=30)
            logger.info(f"TensorBoard logs will be written to: {tb_log_dir}")
        except Exception as e:
            logger.error(f"Failed to create TensorBoard SummaryWriter at {tb_log_dir}: {e}. TB logging disabled.", exc_info=True)

    # Priority 1: Check for a local checkpoint to resume an interrupted run.
    intermediate_checkpoints = sorted(output_dir_path.glob('step-*.pth'))
    if intermediate_checkpoints:
        checkpoint_path_to_attempt_restore = intermediate_checkpoints[-1]
        is_finetuning_from_external_ckpt = False # This is a resumption, not fine-tuning
        if is_master:
            logger.info(f"Found local checkpoint. Resuming interrupted run from: {checkpoint_path_to_attempt_restore}")
    
    # Priority 2: If no local checkpoint, check for an external one from the config for fine-tuning.
    elif startwith and Path(startwith).exists():
        checkpoint_path_to_attempt_restore = startwith
        is_finetuning_from_external_ckpt = True # This is for fine-tuning
        if is_master:
            logger.info(f"No local checkpoint found. Starting fine-tuning from external checkpoint: {startwith}")
    
    # Priority 3: No checkpoints found at all.
    else:
        if is_master:
            logger.info("No local or external checkpoint found. Starting training from scratch.")

    # Now, attempt to restore if a path was determined.
    if checkpoint_path_to_attempt_restore:
        current_epoch_step, last_avg_train_loss_epoch, scheduler_state_restored = restore_from_checkpoint(
            model, optimizer, lr_scheduler, scaler, str(checkpoint_path_to_attempt_restore), device,
            is_distributed, logger,
            load_weights_only=is_finetuning_from_external_ckpt
        )
        if is_master:
            if is_finetuning_from_external_ckpt:
                logger.info(f"Fine-tuning started with weights from: {checkpoint_path_to_attempt_restore}. Training starts at Epoch 1.")
            else:
                 logger.info(
                    f"Restored state from {checkpoint_path_to_attempt_restore}. Resuming from epoch {current_epoch_step + 1}. "
                    f"Last train loss: {last_avg_train_loss_epoch:.5f}. Scheduler restored: {scheduler_state_restored}"
                )
    elif startwith: # This case handles when `startwith` was provided but the file didn't exist.
        logger.warning(f"'startwith' checkpoint '{startwith}' not found. Starting fresh.")

    log_csv_path = output_dir_path / 'progress_log.csv'
    if is_master and log_csv_path.exists():
        try:
            loaded_progress_df = pd.read_csv(log_csv_path)
            if 'epoch' in loaded_progress_df.columns:
                 loaded_progress_df.set_index('epoch', inplace=True, drop=False)
            elif loaded_progress_df.index.name == 'epoch':
                 loaded_progress_df['epoch'] = loaded_progress_df.index
            
            if current_epoch_step > 0 :
                progress_df = loaded_progress_df[loaded_progress_df['epoch'] <= current_epoch_step].copy()
            else:
                progress_df = pd.DataFrame(columns=progress_log_columns)

            for metric_key in validation_metrics:
                col_name = f'validation_{metric_key}'
                if col_name in progress_df.columns:
                    metric_values = pd.to_numeric(progress_df[col_name], errors='coerce').dropna().tolist()
                    if metric_values:
                        val_metrics_history[metric_key] = metric_values
            logger.info(f"Loaded training progress from {log_csv_path} up to epoch {current_epoch_step}.")
        except Exception as e:
            logger.warning(f"Could not load or parse {log_csv_path}: {e}. Starting progress log fresh.", exc_info=True)
            progress_df = pd.DataFrame(columns=progress_log_columns)
            val_metrics_history = defaultdict(list)

    def _run_val_and_log_epoch_state(epoch_num_completed, avg_train_loss_this_epoch):
        nonlocal progress_df, val_metrics_history

        current_val_metrics = {}
        should_run_validation = (epoch_num_completed % validation_epochs == 0) or (epoch_num_completed == 0 and not val_metrics_history)

        if should_run_validation:
            if is_master: logger.info(f"--- Running Validation for Epoch {epoch_num_completed} ---")
            current_val_metrics = eval_epoch(
                model, val_loader, val_baseline_log_likelihood, device,
                metrics=validation_metrics, is_distributed=is_distributed,
                is_master=is_master, logger=logger
            )
            if is_master:
                log_str = f"Validation Epoch {epoch_num_completed} Results: "
                for k_metric, v_metric in current_val_metrics.items(): log_str += f"{k_metric}: {v_metric:.4f} | "
                logger.info(log_str.strip(" | "))
                for k_metric, v_metric in current_val_metrics.items():
                    if not np.isnan(v_metric): val_metrics_history[k_metric].append(v_metric)
        else:
            if is_master: logger.info(f"Skipping validation for epoch {epoch_num_completed} (validation_epochs={validation_epochs}).")
            for m_key in validation_metrics: current_val_metrics[m_key] = np.nan

        if is_master:
            if tb_writer:
                try:
                    if not np.isnan(avg_train_loss_this_epoch): tb_writer.add_scalar('Loss/Train_Epoch_Avg', avg_train_loss_this_epoch, epoch_num_completed)
                    current_lr_tb = optimizer.param_groups[0]['lr']
                    tb_writer.add_scalar('Meta/Learning_Rate', current_lr_tb, epoch_num_completed)
                    
                    unwrapped_model_tb = model.module if is_distributed else model
                    if hasattr(unwrapped_model_tb, 'finalizer') and unwrapped_model_tb.finalizer is not None:
                        finalizer_tb = unwrapped_model_tb.finalizer
                        if hasattr(finalizer_tb, 'gauss') and hasattr(finalizer_tb.gauss, 'sigma'):
                            tb_writer.add_scalar('Params/Finalizer_Sigma', finalizer_tb.gauss.sigma.item(), epoch_num_completed)
                        if hasattr(finalizer_tb, 'center_bias_weight'):
                            tb_writer.add_scalar('Params/Finalizer_CenterBiasWeight', finalizer_tb.center_bias_weight.item(), epoch_num_completed)

                    for k_metric, v_metric in current_val_metrics.items():
                        if not np.isnan(v_metric): tb_writer.add_scalar(f'Validation/{k_metric}', v_metric, epoch_num_completed)
                except Exception as e_tb:
                    logger.error(f"Error writing to TensorBoard for epoch {epoch_num_completed}: {e_tb}", exc_info=True)

            new_log_row_dict = {'epoch': epoch_num_completed,
                                'timestamp': datetime.utcnow().isoformat(),
                                'learning_rate': optimizer.param_groups[0]['lr'],
                                'train_loss': avg_train_loss_this_epoch}
            for k_metric, v_metric in current_val_metrics.items():
                new_log_row_dict[f'validation_{k_metric}'] = v_metric
            
            progress_df = progress_df[progress_df['epoch'] != epoch_num_completed]
            new_row_df_entry = pd.DataFrame([new_log_row_dict])
            progress_df = pd.concat([progress_df, new_row_df_entry], ignore_index=True).sort_values(by='epoch')
            
            try:
                with atomic_save(str(log_csv_path), text_mode=True, overwrite_part=True) as f_csv:
                    progress_df.to_csv(f_csv, index=False)
                logger.info(f"Progress log saved to {log_csv_path} for epoch {epoch_num_completed}.")
            except Exception as e_csv:
                logger.error(f"Failed to save progress log {log_csv_path}: {e_csv}", exc_info=True)

            logger.info(f"Epoch {epoch_num_completed} Summary (master):\n{progress_df[progress_df['epoch'] == epoch_num_completed].to_string()}")

            current_epoch_ckpt_path = output_dir_path / f'step-{epoch_num_completed:04d}.pth'
            save_training_state(model, optimizer, lr_scheduler, scaler,
                                epoch_num_completed, avg_train_loss_this_epoch,
                                str(current_epoch_ckpt_path), is_distributed, is_master, logger)

            best_val_epoch_num = -1
            if validation_metric_col_name := f'validation_{validation_metric}':
                if validation_metric_col_name in progress_df.columns and not progress_df[validation_metric_col_name].dropna().empty:
                    higher_is_better = validation_metric.upper() in ['IG', 'NSS', 'AUC', 'AUC_CPU', 'AUC_GPU']
                    valid_scores_series = pd.to_numeric(progress_df[validation_metric_col_name], errors='coerce').dropna()
                    
                    if not valid_scores_series.empty:
                        if higher_is_better:
                            best_score_so_far = valid_scores_series.max()
                        else:
                            best_score_so_far = valid_scores_series.min()
                        
                        best_epoch_candidates = progress_df.loc[pd.to_numeric(progress_df[validation_metric_col_name], errors='coerce') == best_score_so_far, 'epoch']
                        if not best_epoch_candidates.empty:
                            best_val_epoch_num = int(best_epoch_candidates.iloc[0])
                            logger.info(f"Best validation score ({validation_metric}: {best_score_so_far:.4f}) so far was at epoch {best_val_epoch_num}.")
            
            all_intermediate_ckpts = sorted(output_dir_path.glob('step-*.pth'))
            for ckpt_p in all_intermediate_ckpts:
                try:
                    ckpt_epoch_num = int(ckpt_p.stem.split('-')[1])
                    if ckpt_epoch_num != epoch_num_completed and ckpt_epoch_num != best_val_epoch_num:
                        ckpt_p.unlink()
                        logger.debug(f"Removed old intermediate checkpoint: {ckpt_p}")
                except (ValueError, IndexError, OSError) as e_rm:
                    logger.warning(f"Could not parse or remove old checkpoint {ckpt_p}: {e_rm}")

    needs_initial_evaluation_flag = False
    if is_master:
        if current_epoch_step == 0 or \
           (not progress_df.empty and current_epoch_step not in progress_df['epoch'].values) or \
           (progress_df.empty and current_epoch_step > 0) :
            logger.info(f"Running initial evaluation/log for (restored) epoch {current_epoch_step}.")
            needs_initial_evaluation_flag = True
            
    if is_distributed:
        initial_eval_tensor = torch.tensor(int(needs_initial_evaluation_flag), device=device, dtype=torch.int)
        dist.broadcast(initial_eval_tensor, src=0)
        needs_initial_evaluation_flag = bool(initial_eval_tensor.item())

    if needs_initial_evaluation_flag:
        _run_val_and_log_epoch_state(current_epoch_step, last_avg_train_loss_epoch)


    if current_epoch_step > 0 and scheduler_state_restored:
        if is_master: logger.info(f"LR scheduler state was restored. Current LR: {optimizer.param_groups[0]['lr']:.2e}")
    elif current_epoch_step > 0 and not scheduler_state_restored:
        if is_master: logger.warning(f"Resumed from epoch {current_epoch_step} but scheduler state NOT restored. Scheduler starts fresh.")

    if is_master: logger.info(f"--- Starting Training Loop from Epoch {current_epoch_step + 1} ---")
    
    epoch_to_run = current_epoch_step + 1

    while optimizer.param_groups[0]['lr'] >= minimum_learning_rate:
        if is_master: logger.info(f"--- Beginning Epoch {epoch_to_run} (LR: {optimizer.param_groups[0]['lr']:.2e}) ---")
        
        if is_distributed and train_sampler is not None and hasattr(train_sampler, 'set_epoch'):
            train_sampler.set_epoch(epoch_to_run)

        epoch_start_time_actual = datetime.now()
        
        avg_train_loss_this_epoch = train_epoch(
            model, train_loader, optimizer, device, scaler,
            gradient_accumulation_steps, is_distributed, is_master, logger
        )
        
        if is_master:
            epoch_duration_actual = datetime.now() - epoch_start_time_actual
            if np.isnan(avg_train_loss_this_epoch):
                logger.warning(f"Epoch {epoch_to_run} finished with NaN training loss. Duration: {epoch_duration_actual}")
            else:
                logger.info(f"Epoch {epoch_to_run} training finished. Avg Train Loss (Rank 0): {avg_train_loss_this_epoch:.5f}. Duration: {epoch_duration_actual}")

        _run_val_and_log_epoch_state(epoch_to_run, avg_train_loss_this_epoch)
        
        if not np.isnan(avg_train_loss_this_epoch):
            try:
                lr_scheduler.step()
            except Exception as e_lr_step:
                logger.error(f"Error stepping LR scheduler at end of epoch {epoch_to_run}: {e_lr_step}", exc_info=True)
        else:
            if is_master: logger.warning(f"Skipping LR scheduler step for epoch {epoch_to_run} due to invalid training loss.")

        epoch_to_run += 1

    if is_master:
        logger.info("Training loop finished (LR below minimum or other condition).")
        final_model_path = output_dir_path / 'final_model_at_lr_cutoff.pth'
        try:
            model_to_save_final = model.module if is_distributed else model
            torch.save(model_to_save_final.state_dict(), str(final_model_path))
            logger.info(f"Final model state_dict (at LR cutoff) saved to {final_model_path}")
        except Exception as e_final_save:
            logger.error(f"Failed to save final model state_dict: {e_final_save}", exc_info=True)

        best_val_epoch_num_final = -1
        best_val_score_final = np.nan
        validation_metric_col_name_final = f'validation_{validation_metric}'
        if validation_metric_col_name_final in progress_df.columns and not progress_df[validation_metric_col_name_final].dropna().empty:
            higher_is_better_final = validation_metric.upper() in ['IG', 'NSS', 'AUC', 'AUC_CPU', 'AUC_GPU']
            valid_scores_series_final = pd.to_numeric(progress_df[validation_metric_col_name_final], errors='coerce').dropna()
            if not valid_scores_series_final.empty:
                best_score_final = valid_scores_series_final.max() if higher_is_better_final else valid_scores_series_final.min()
                best_epoch_candidates_final = progress_df.loc[pd.to_numeric(progress_df[validation_metric_col_name_final], errors='coerce') == best_score_final, 'epoch']
                if not best_epoch_candidates_final.empty:
                    best_val_epoch_num_final = int(best_epoch_candidates_final.iloc[0])
                    best_val_score_final = best_score_final

        if best_val_epoch_num_final > 0:
            best_epoch_ckpt_path_final = output_dir_path / f'step-{best_val_epoch_num_final:04d}.pth'
            if best_epoch_ckpt_path_final.exists():
                # The new, predictable name for the best checkpoint
                final_best_val_path = output_dir_path / 'final_best_val.pth' 
                
                try:
                    # Simply copy the entire file
                    import shutil
                    shutil.copy(str(best_epoch_ckpt_path_final), str(final_best_val_path))
                    logger.info(f"Copied BEST validation checkpoint (Epoch {best_val_epoch_num_final}, {validation_metric}: {best_val_score_final:.4f}) to {final_best_val_path}")
                    
                    # Also save the model-only state for easy inference later
                    final_best_val_model_path = output_dir_path / 'final_best_val_model.pth'
                    best_ckpt_data = torch.load(best_epoch_ckpt_path_final, map_location='cpu')
                    model_state_dict_to_save = _extract_model_state_dict_from_checkpoint(best_ckpt_data, logger)
                    if model_state_dict_to_save:
                        torch.save(model_state_dict_to_save, str(final_best_val_model_path))
                        logger.info(f"Saved BEST validation model state_dict for inference to {final_best_val_model_path}")

                except Exception as e_best_save:
                    logger.error(f"Failed to copy/save best validation checkpoint from {best_epoch_ckpt_path_final}: {e_best_save}", exc_info=True)
            else:
                logger.warning(f"Best validation checkpoint file {best_epoch_ckpt_path_final} not found. Cannot save final best checkpoint.")
        else:
            logger.warning(f"No best validation epoch found in progress log for metric '{validation_metric}'.")

        if tb_writer:
            try:
                tb_writer.close()
            except Exception as e_tb_close:
                logger.error(f"Error closing TensorBoard writer: {e_tb_close}")
    
    if is_distributed:
        dist.barrier()
    logger.info(f"{current_rank_str} finished _train function.")