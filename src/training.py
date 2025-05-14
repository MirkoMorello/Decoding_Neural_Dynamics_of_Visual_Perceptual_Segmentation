# training.py
# flake8: noqa E501
# pylint: disable=not-callable
# E501: line too long

from collections import defaultdict
from datetime import datetime
import os
from collections import OrderedDict, defaultdict
from boltons.cacheutils import cached, LRU
from boltons.fileutils import atomic_save
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch import amp
from .metrics import log_likelihood, nss, auc
from .modules import DeepGazeII
import torchmetrics
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
import sys
import warnings
from .metrics import log_likelihood, nss, auc as auc_cpu_fn # Rename import to avoid conflict


baseline_performance = cached(LRU(max_size=3))(lambda model, *args, **kwargs: model.information_gain(*args, **kwargs))


def eval_epoch(model, dataset, baseline_information_gain, device, metrics=None, is_distributed=False, is_master=True, logger=None):
    
    """ Evaluates the model for one epoch on the validation set. Handles DDP aggregation. """
    model.eval()
    default_metrics = ['LL', 'IG', 'NSS', 'AUC_CPU']
    if metrics is None: metrics = default_metrics
    if 'IG' in metrics and 'LL' not in metrics: metrics.append('LL')
    if is_master: logger.debug(f"Evaluating metrics: {metrics}")

    total_metric_sums = {
        name: torch.tensor(0.0, device=device, dtype=torch.float64)
        for name in metrics if name in ['LL', 'NSS', 'AUC_CPU']
    }
    total_weight = torch.tensor(0.0, device=device, dtype=torch.float64)

    auroc_metric_gpu = None
    if 'AUC_GPU' in metrics:
        if is_master: logger.debug("Initializing GPU AUROC (max_fpr=1.0)")
        auroc_metric_gpu = torchmetrics.AUROC(task="binary", max_fpr=1.0).to(device)

    metric_functions_avg = {}
    if 'LL' in metrics: metric_functions_avg['LL'] = log_likelihood
    if 'NSS' in metrics: metric_functions_avg['NSS'] = nss
    if 'AUC_CPU' in metrics:
        if is_master: logger.debug("Will calculate CPU AUC using original function.")
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
                logger.error(f"Error moving batch data to device {device}: {e}")
                continue

            log_density = None
            try:
                underlying_model = model.module if is_distributed else model

                # No AMP during validation as inference might be sensitive
                if isinstance(underlying_model, DeepGazeII):
                    logger.debug("Using DeepGazeII forward.")
                    log_density = model(image, centerbias, **kwargs)

                elif getattr(underlying_model, 'scanpath_network', None) is None:
                    # Spatial-only: Call model's forward with only image and centerbias
                    logger.debug("Using spatial-only via model.forward(image, centerbias)")
                    log_density = model(image, centerbias)
                else:
                    # Full scanpath model: Call model's forward with all relevant arguments
                    logger.debug("Using full model forward with scanpath.")
                    log_density = model(
                        image, centerbias, x_hist=x_hist, y_hist=y_hist, durations=durations, **kwargs
                    )

            except torch.cuda.OutOfMemoryError as e:
                current_rank = dist.get_rank() if is_distributed else 0
                logger.error(
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
                logger.exception("Error during validation forward pass")
                continue

            if log_density is None:
                logger.error("log_density is None after forward pass, skipping batch.")
                continue

            try:
                if isinstance(fixation_mask, torch.sparse.Tensor):
                    target_mask_dense = fixation_mask.to_dense()
                else:
                    target_mask_dense = fixation_mask
                target_mask_int = target_mask_dense.long()
                target_mask_binary = (target_mask_int > 0).long() # Binary mask used for most metrics
            except Exception as e:
                logger.error(f"Error preparing target masks: {e}")
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


def train_epoch(model, dataset, optimizer, device, scaler, gradient_accumulation_steps=1, is_distributed=False, is_master=True, logger=None):
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
    if is_master: logger.debug(f"Estimated global batch size for logging: {global_batch_size_est}")

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
                logger.error(f"Error moving batch data to device {device}: {e}")
                continue

            log_density = None
            try:
                underlying_model = model.module if is_distributed else model

                # Use Automatic Mixed Precision (AMP)
                with amp.autocast('cuda', dtype=torch.float16): # Use float16 for speed/memory
                    if isinstance(underlying_model, DeepGazeII):
                        logger.debug("Using DeepGazeII forward.")
                        log_density = model(image, centerbias, **kwargs)
                    elif getattr(underlying_model, 'scanpath_network', None) is None:
                        logger.debug("Using spatial-only via model.forward(image, centerbias)")
                        log_density = model(image, centerbias)
                    else:
                        logger.debug("Using full model forward with scanpath.")
                        log_density = model(
                            image, centerbias, x_hist=x_hist, y_hist=y_hist, durations=durations, **kwargs
                        )

                    if log_density is None:
                        logger.error("log_density is None after forward pass, skipping batch.")
                        continue

                    loss = -log_likelihood(log_density, fixation_mask, weights=weights)
                    # Scale loss for gradient accumulation
                    loss = loss / gradient_accumulation_steps

                if torch.isnan(loss) or torch.isinf(loss):
                    logger.error(f"NaN or Inf loss detected! Skipping backward/step.")
                    # If NaN/Inf, reset gradients accumulated so far for this step
                    if not is_accumulation_step and not is_last_batch_in_epoch:
                        optimizer.zero_grad() # Reset gradients for the current accumulation cycle
                    continue

            except Exception as forward_loss_e:
                logger.exception("Error during training forward pass or loss calculation")
                if not is_accumulation_step and not is_last_batch_in_epoch:
                    optimizer.zero_grad() # Reset potentially corrupted gradients
                continue

            # Backward pass with GradScaler
            try:
                # scaler.scale(loss).backward() will sync grads if sync_context is nullcontext
                scaler.scale(loss).backward()
            except Exception as backward_e:
                logger.exception("Error during backward pass")
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
                logger.exception("Error during optimizer step or scaler update")
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


def restore_from_checkpoint(model, optimizer, scheduler, scaler, path, device, is_distributed=False, logger=None):
    """ Restores training state from a checkpoint file, handling DDP 'module.' prefix and GradScaler. """
    if not os.path.exists(path):
        logger.error(f"Checkpoint path not found: {path}. Cannot restore.")
        return 0, np.nan, False  # No checkpoint, start from scratch

    logger.info(f"Restoring checkpoint from: {path} (Distributed: {is_distributed})")
    try:
        # Load either a full checkpoint dict or a bare state_dict
        data = torch.load(path, map_location=device)
        if not isinstance(data, dict) or 'model' not in data:
            state_dict = data
        else:
            state_dict = data['model']
    except Exception as e:
        logger.exception(f"Failed to load checkpoint file {path}")
        return 0, np.nan, False

    # --- Restore Model State ---
    model_state_dict = state_dict
    adjusted_state_dict = OrderedDict()
    is_ddp_checkpoint = any(k.startswith('module.') for k in model_state_dict.keys())
    current_model_is_ddp = isinstance(model, DDP)

    if current_model_is_ddp:
        # Model is wrapped in DDP, ensure keys start with 'module.'
        if not is_ddp_checkpoint:
            logger.warning("Current model is DDP, but checkpoint is not. Adding 'module.' prefix.")
            for k, v in model_state_dict.items():
                adjusted_state_dict[f'module.{k}'] = v
        else:
            logger.debug("Current model and checkpoint are both DDP. Checking prefixes.")
            for k, v in model_state_dict.items():
                if not k.startswith('module.'):
                    logger.warning(f"Adding missing 'module.' prefix to key '{k}'.")
                    adjusted_state_dict[f'module.{k}'] = v
                else:
                    adjusted_state_dict[k] = v
    else:
        # Model is not DDP, strip 'module.' if present
        if is_ddp_checkpoint:
            logger.warning("Current model is not DDP, but checkpoint is. Removing 'module.' prefix.")
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
            logger.warning(f"Missing keys when loading model state: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys when loading model state: {unexpected_keys}")
    except Exception as load_err:
        logger.exception("Error loading model state_dict")
        return 0, np.nan, False

    # --- Restore Optimizer State ---
    if isinstance(data, dict) and 'optimizer' in data and optimizer is not None:
        try:
            optimizer.load_state_dict(data['optimizer'])
            logger.info("Optimizer state restored.")
            # Move optimizer tensors to correct device
            for state in optimizer.state.values():
                for key, val in state.items():
                    if isinstance(val, torch.Tensor):
                        state[key] = val.to(device)
        except Exception as e:
            logger.warning(f"Could not restore optimizer state: {e}. Starting with fresh optimizer.")
    elif optimizer is not None:
        logger.warning("Optimizer state not found in checkpoint. Starting with fresh optimizer.")

    # --- Restore Scheduler State ---
    scheduler_restored = False
    if isinstance(data, dict) and 'scheduler' in data and scheduler is not None:
        try:
            scheduler.load_state_dict(data['scheduler'])
            logger.info("Scheduler state restored.")
            scheduler_restored = True
        except Exception as e:
            logger.warning(f"Could not restore scheduler state: {e}. Starting with fresh scheduler.")
    elif scheduler is not None:
        logger.warning("Scheduler state not found in checkpoint. Starting with fresh scheduler.")

    # --- Restore GradScaler State (for AMP) ---
    if isinstance(data, dict) and 'grad_scaler' in data and scaler is not None:
        try:
            scaler.load_state_dict(data['grad_scaler'])
            logger.info("GradScaler state restored.")
        except Exception as e:
            logger.warning(f"Could not restore GradScaler state: {e}. Starting with fresh GradScaler.")
    elif scaler is not None:
        logger.warning("GradScaler state not found in checkpoint. Starting with fresh GradScaler.")

    # --- Warn about RNG state if present ---
    if isinstance(data, dict) and ('rng_state' in data or 'cuda_rng_state' in data):
        logger.warning("Checkpoint contains RNG state, but it is not being restored by default.")

    # --- Final bookkeeping ---
    step = data.get('step', 0) if isinstance(data, dict) else 0
    loss = data.get('loss', np.nan) if isinstance(data, dict) else np.nan
    if step > 0:
        logger.info(f"Restored to step {step} with loss {loss:.5f}")
    else:
        logger.info("No previous step/loss found. Starting from step 0.")

    return step, loss, scheduler_restored



def save_training_state(model, optimizer, scheduler, scaler, step, loss, path, is_distributed=False, is_master=True, logger=None):
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
        logger.debug(f"Saved checkpoint to {path} at step {step}")
    except Exception as e:
        logger.exception(f"Failed to save checkpoint {path} at step {step}")


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
        is_master=True,
        logger=None,):
    """ Main training loop function, adapted for DDP, AMP, Grad Accum. """

    output_dir_path = Path(this_directory)
    final_checkpoint_path = output_dir_path / 'final.pth'
    finished_training = False

    if is_master:
        if final_checkpoint_path.exists():
            logger.info(f"Final checkpoint {final_checkpoint_path} already exists. Training previously finished.")
            finished_training = True

    finished_flag = torch.tensor([int(finished_training)], device=device, dtype=torch.int)
    if is_distributed:
        dist.broadcast(finished_flag, src=0)
    if finished_flag.item() == 1:
        current_rank = dist.get_rank() if is_distributed else 0
        logger.info(f"Rank {current_rank} exiting: Training already finished.")
        return

    if is_master:
        try:
            output_dir_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.critical(f"Failed to create output directory {output_dir_path}: {e}")
            if is_distributed: dist.barrier()
            sys.exit(1)
    if device is None:
        raise ValueError("Device must be specified for _train function.")
    current_rank = dist.get_rank() if is_distributed else 0
    logger.info(f"Rank {current_rank} using device: {device}")

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
            logger.info(f"TensorBoard logs will be written to: {log_dir}")
            writer = SummaryWriter(log_dir, flush_secs=30)
        except Exception as e:
            logger.error(f"Failed to create TensorBoard writer: {e}. TensorBoard logging disabled.")

    checkpoint_to_load = None
    if startwith and os.path.exists(startwith):
        checkpoint_to_load = startwith
        if is_master: logger.info(f"Attempting to restore from specified checkpoint: {startwith}")
    else:
        step_files = sorted(output_dir_path.glob('step-*.pth'))
        if step_files:
            checkpoint_to_load = step_files[-1]
            if is_master: logger.info(f"No startwith specified, found latest checkpoint in output dir: {checkpoint_to_load}")

    if checkpoint_to_load:
        # <<< CHANGE: Pass scaler to restore_from_checkpoint >>>
        step, last_train_loss, scheduler_restored = restore_from_checkpoint(
            model, optimizer, lr_scheduler, scaler, checkpoint_to_load, device, is_distributed, logger
        )
        # <<< END CHANGE >>>
        if is_master: logger.info(f"Restored training state from {checkpoint_to_load} to step {step}.")
    elif startwith:
        logger.warning(f"Start checkpoint specified ({startwith}) but not found. Starting from scratch.")
    else:
        if is_master: logger.info("No checkpoint found or specified. Starting training from scratch.")

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

            logger.info(f"Loaded previous progress from {log_csv_path} up to step {step}")
        except Exception as e:
            logger.warning(f"Could not load or parse progress from {log_csv_path}: {e}. Starting log fresh.")
            progress_df = pd.DataFrame(columns=columns)
            val_metrics_history = defaultdict(list)

    def save_and_log_step(current_step, current_loss):
        nonlocal progress_df, val_metrics_history

        _val_metrics_epoch = {}
        run_validation_this_step = (current_step % validation_epochs == 0)

        if run_validation_this_step:
            if is_master: logger.info(f"Running validation for step {current_step}...")
            _val_metrics_epoch = eval_epoch(model, val_loader, val_baseline_log_likelihood, device, metrics=validation_metrics, is_distributed=is_distributed, is_master=is_master, logger=logger)
            if is_master: logger.info(f"Validation results step {current_step}: {_val_metrics_epoch}")
        else:
            if is_master: logger.info(f"Skipping validation for step {current_step}.")
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
                    logger.error(f"Error writing to TensorBoard: {tb_err}")

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
                logger.error(f"Error updating progress DataFrame: {df_err}")

            logger.info(f"Step {current_step} Summary:\n{progress_df.iloc[-1:].to_string()}")

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
                        logger.info(f"Best validation ({validation_metric}) score so far {best_val_score:.5f} occurred at step {best_val_step}")

            chkpt_path = output_dir_path / f'step-{current_step:04d}.pth'
            # Pass scaler to save its state
            save_training_state(model, optimizer, lr_scheduler, scaler, current_step, current_loss, str(chkpt_path), is_distributed, is_master, logger = logger)


            try:
                with atomic_save(str(log_csv_path), text_mode=True, overwrite_part=True) as f:
                    progress_df.to_csv(f, index=False)
            except Exception as e:
                logger.exception(f"Failed to save progress log {log_csv_path}")

            # Clean up old checkpoints, keeping current and best validation
            all_checkpoints = sorted(output_dir_path.glob('step-*.pth'))
            for cp_path in all_checkpoints:
                try:
                    cp_step = int(cp_path.stem.split('-')[1])
                    # Keep current step and best validation step (if valid)
                    if cp_step != current_step and (best_val_step == -1 or cp_step != best_val_step):
                        cp_path.unlink()
                        logger.debug(f"Removed old checkpoint: {cp_path}")
                except (ValueError, IndexError, OSError) as e:
                    logger.warning(f"Could not parse step or remove checkpoint {cp_path}: {e}")

    needs_initial_eval = False
    if is_master:
        if step == 0 or (step > 0 and step not in progress_df['epoch'].values):
            logger.info(f"Step {step} needs initial evaluation.")
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
            if is_master: logger.info(f"LR scheduler state restored for step {step}. Will step at end of next epoch.")
        except Exception as e:
            logger.error(f"Error checking scheduler after restore: {e}")
    elif step > 0 and not scheduler_restored:
        if is_master: logger.warning(f"Restored to step {step}, but scheduler state was not restored. Scheduler starts fresh.")


    if is_master: logger.info("Starting training loop...")
    while True:
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < minimum_learning_rate:
            if is_master: logger.info(f"Learning rate ({current_lr:.2e}) reached minimum ({minimum_learning_rate:.2e}). Stopping training.")
            break

        step += 1
        epoch_start_time = datetime.now()

        if is_distributed and hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(step)
            if hasattr(val_loader, 'sampler') and hasattr(val_loader.sampler, 'set_epoch'):
                val_loader.sampler.set_epoch(step) # Sync validation sampler too

        if is_master: logger.info(f"--- Starting Epoch {step} (LR: {current_lr:.2e}) ---")
        # Pass scaler and gradient_accumulation_steps to train_epoch
        last_train_loss = train_epoch(
            model, train_loader, optimizer, device, scaler,
            gradient_accumulation_steps, is_distributed, is_master, logger
        )
        if is_master:
            epoch_duration = datetime.now() - epoch_start_time
            if np.isnan(last_train_loss):
                logger.warning(f"Epoch {step} finished with NaN loss. Duration: {epoch_duration}")
            else:
                logger.info(f"Epoch {step} finished. Rank 0 Avg Train Loss: {last_train_loss:.5f}. Duration: {epoch_duration}")

        save_and_log_step(step, last_train_loss)

        try:
            # Step scheduler AFTER saving checkpoint for the current epoch
            lr_scheduler.step()
        except Exception as e:
            logger.error(f"Error stepping scheduler at end of step {step}: {e}")


    if is_master:
        logger.info("Training loop finished.")
        try:
            final_path = output_dir_path / 'final.pth'
            model_to_save = model.module if is_distributed else model
            # Save only the model state_dict for the final model
            torch.save(model_to_save.state_dict(), str(final_path))
            logger.info(f"Final model state dict saved to {final_path}")
        except Exception as e:
            logger.exception("Failed to save final model state dict")

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
                        logger.info(f"Saved best validation model state dict (Step {best_val_step}, Score: {best_val_score:.5f}) to {final_best_path}")
                    else:
                        logger.error(f"Best checkpoint {best_chkpt_path} does not contain 'model' key.")

                    # Optional: Copy full best checkpoint as well
                    # shutil.copyfile(str(best_chkpt_path), str(output_dir_path / 'full_best_val.pth'))
                    # logger.info(f"Copied full best validation checkpoint (Step {best_val_step}) to {output_dir_path / 'full_best_val.pth'}")

                except Exception as e:
                    logger.exception(f"Failed to save/copy best checkpoint model state {best_chkpt_path}")
            else:
                logger.warning(f"Best validation checkpoint file {best_chkpt_path} not found, cannot save best model separately.")

        # Optional cleanup of remaining step checkpoints
        # for step_file in output_dir_path.glob('step-*.pth'):
        #      try: step_file.unlink()
        #      except OSError as e: logger.error(f"Failed to remove final intermediate checkpoint {step_file}: {e}")

        if writer:
            writer.close()

    if is_distributed:
        # dist.barrier(device_ids=[device.index] if device.type == 'cuda' else None)
        dist.barrier() # Removed device_ids