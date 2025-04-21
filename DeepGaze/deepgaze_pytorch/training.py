# training.py
# flake8: noqa E501
# pylint: disable=not-callable
# E501: line too long

from collections import defaultdict
from datetime import datetime
import glob
import os
import tempfile
from torch.serialization import safe_globals

from boltons.cacheutils import cached, LRU
from boltons.fileutils import atomic_save, mkdir_p
from boltons.iterutils import windowed
from IPython import get_ipython
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysaliency
from pysaliency.filter_datasets import iterate_crossvalidation
from pysaliency.plotting import visualize_distribution
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

from .data import ImageDataset, FixationDataset, ImageDatasetSampler, FixationMaskTransform
#from .loading import import_class, build_model, DeepGazeCheckpointModel, SharedPyTorchModel, _get_from_config
from .metrics import log_likelihood, nss, auc
from .modules import DeepGazeII
import torchmetrics



baseline_performance = cached(LRU(max_size=3))(lambda model, *args, **kwargs: model.information_gain(*args, **kwargs))


import torch
import numpy as np
from tqdm import tqdm
import torchmetrics
import warnings # Import warnings

# Your existing metric functions (log_likelihood, nss) are assumed to be defined above

import torch
import numpy as np
from tqdm import tqdm
import torchmetrics # For GPU AUC
import warnings
# Import your ORIGINAL CPU-based AUC function along with others
from .metrics import log_likelihood, nss, auc as auc_cpu_fn # Rename import to avoid conflict

def eval_epoch(model, dataset, baseline_information_gain, device, metrics=None):
    model.eval()
    default_metrics = ['LL', 'IG', 'NSS', 'AUC_CPU'] # Default to CPU AUC
    if metrics is None:
        metrics = default_metrics
    else:
        # Ensure IG is handled correctly if requested but LL isn't
        if 'IG' in metrics and 'LL' not in metrics:
            metrics.append('LL') # Need LL to calculate IG
    print(f"DEBUG: Evaluating metrics: {metrics}")

    # --- Initialize Accumulators & Metrics ---
    total_metric_sums = {
        # Accumulators for metrics calculated via batch averaging
        metric_name: torch.tensor(0.0, device=device, dtype=torch.float64)
        for metric_name in metrics if metric_name in ['LL', 'NSS', 'AUC_CPU']
    }
    total_weight = torch.tensor(0.0, device=device, dtype=torch.float64)

    # Initialize GPU AUROC only if requested
    auroc_metric_gpu = None
    if 'AUC_GPU' in metrics:
        print("DEBUG: Initializing GPU AUROC (Exact Calculation - May OOM if batch size too large)")
        # Calculate EXACT AUC - NO thresholds argument
        auroc_metric_gpu = torchmetrics.AUROC(task="binary").to(device)

    # Dictionary for metrics computed via batch averaging (LL, NSS, CPU AUC)
    metric_functions_avg = {}
    if 'LL' in metrics: metric_functions_avg['LL'] = log_likelihood
    if 'NSS' in metrics: metric_functions_avg['NSS'] = nss
    if 'AUC_CPU' in metrics:
        print("DEBUG: Will calculate CPU AUC using original function.")
        metric_functions_avg['AUC_CPU'] = auc_cpu_fn # Use the imported original function

    # --- Validation Loop ---
    oom_error_gpu_auc = False # Flag to track if GPU AUC failed
    with torch.no_grad():
        pbar = tqdm(dataset, desc="Validating")
        for batch in pbar:
            # --- Move batch to GPU ---
            image = batch.pop('image').to(device)
            centerbias = batch.pop('centerbias').to(device)
            fixation_mask = batch.pop('fixation_mask').to(device)
            x_hist = batch.pop('x_hist', torch.tensor([])).to(device)
            y_hist = batch.pop('y_hist', torch.tensor([])).to(device)
            weights = batch.pop('weight').to(device)
            durations = batch.pop('durations', torch.tensor([])).to(device)
            kwargs = {k: v.to(device) for k, v in batch.items()}

            # --- Model Forward Pass ---
            try:
                if isinstance(model, DeepGazeII):
                    log_density = model(image, centerbias, **kwargs)
                else:
                    log_density = model(image, centerbias, x_hist=x_hist, y_hist=y_hist, durations=durations, **kwargs)
            except torch.cuda.OutOfMemoryError as e:
                print("\n\n!!! OOM ERROR during Model Forward Pass !!!")
                print(f"Batch Shape Image: {image.shape}")
                print(f"Error: {e}")
                print("Try reducing validation batch size significantly.")
                # Re-raise or handle as needed, maybe return NaNs for all metrics
                raise e # Or return {m: float('nan') for m in metrics}

            # --- Prep Targets (Dense & Binary) ---
            if isinstance(fixation_mask, torch.sparse.IntTensor):
                target_mask_int = fixation_mask.to_dense().long()
            else:
                target_mask_int = fixation_mask.long()
            target_mask_binary = (target_mask_int > 0).long()

            # --- Calculate Batch Weight ---
            batch_weight_sum = weights.sum()

            # --- Accumulate Batch-Averaged Metrics (LL, NSS, AUC_CPU) ---
            if batch_weight_sum > 0:
                total_weight += batch_weight_sum
                for metric_name, metric_fn in metric_functions_avg.items():
                    if metric_name in total_metric_sums:
                        try:
                            # Pass appropriate mask: binary float for LL/NSS (as before), original mask for CPU AUC
                            # The CPU AUC function expects the original format and handles its own conversions
                            mask_for_metric = target_mask_binary.float() if metric_name != 'AUC_CPU' else fixation_mask
                            metric_value_batch_avg = metric_fn(log_density, mask_for_metric, weights=weights)

                            if not isinstance(metric_value_batch_avg, torch.Tensor) or metric_value_batch_avg.ndim != 0:
                                 warnings.warn(f"Metric function {metric_name} did not return scalar tensor! Got: {metric_value_batch_avg}", RuntimeWarning)
                                 # Attempt to convert if possible, otherwise skip accumulation
                                 if isinstance(metric_value_batch_avg, (int, float)):
                                     metric_value_batch_avg = torch.tensor(metric_value_batch_avg, device=device)
                                 else: continue # Skip accumulation if conversion fails

                            total_metric_sums[metric_name] += metric_value_batch_avg * batch_weight_sum
                        except Exception as e:
                             warnings.warn(f"Error calculating metric {metric_name}: {e}. Skipping accumulation for this batch.", RuntimeWarning)


            # --- Update GPU AUC (if active and no previous OOM) ---
            if auroc_metric_gpu is not None and not oom_error_gpu_auc:
                try:
                    preds_for_auc = log_density
                    targets_flat = target_mask_binary.flatten()
                    preds_flat = preds_for_auc.flatten()
                    if torch.unique(targets_flat).numel() > 1:
                        auroc_metric_gpu.update(preds_flat, targets_flat)
                except torch.cuda.OutOfMemoryError as e:
                    warnings.warn("\n!!! OOM ERROR during GPU AUROC update !!!\nGPU AUC will be NaN. Reduce validation batch size if GPU AUC is desired.", RuntimeWarning)
                    oom_error_gpu_auc = True # Set flag to stop trying GPU AUC
                    # Release metric memory if possible
                    del auroc_metric_gpu
                    auroc_metric_gpu = None
                    torch.cuda.empty_cache()
                except Exception as e:
                     warnings.warn(f"Error updating GPU AUROC: {e}. GPU AUC may be incorrect.", RuntimeWarning)


            # --- Update progress bar ---
            desc = "Validating"
            if 'LL' in metrics and 'LL' in total_metric_sums and total_weight > 0:
                 current_avg_ll = (total_metric_sums['LL'] / total_weight).item()
                 desc += f' LL {current_avg_ll:.5f}'
            # Add CPU AUC to progress bar if desired (more stable)
            if 'AUC_CPU' in metrics and 'AUC_CPU' in total_metric_sums and total_weight > 0:
                 current_avg_auc_cpu = (total_metric_sums['AUC_CPU'] / total_weight).item()
                 desc += f' AUC_CPU {current_avg_auc_cpu:.5f}'
            pbar.set_description(desc)
            # ---

    # --- Final Calculation ---
    final_metrics = {}
    total_weight_cpu = total_weight.item()

    if total_weight_cpu > 0:
        # Calculate final averages for LL, NSS, AUC_CPU
        for metric_name, metric_sum in total_metric_sums.items():
             if metric_name in metrics: # Only compute if requested
                final_metrics[metric_name] = (metric_sum / total_weight).item()

        # Compute final GPU AUC (if requested and didn't fail)
        if 'AUC_GPU' in metrics:
            if auroc_metric_gpu is not None and not oom_error_gpu_auc:
                try:
                    if hasattr(auroc_metric_gpu, 'update_count') and auroc_metric_gpu.update_count > 0:
                         final_metrics['AUC_GPU'] = auroc_metric_gpu.compute().item()
                    elif hasattr(auroc_metric_gpu, 'preds') and len(auroc_metric_gpu.preds) > 0: # Fallback check
                         final_metrics['AUC_GPU'] = auroc_metric_gpu.compute().item()
                    else:
                         warnings.warn("GPU AUROC metric may not have been updated. Setting AUC_GPU to NaN.", RuntimeWarning)
                         final_metrics['AUC_GPU'] = float('nan')
                except Exception as e:
                    warnings.warn(f"Failed to compute final GPU AUC: {e}. Setting AUC_GPU to NaN.", RuntimeWarning)
                    final_metrics['AUC_GPU'] = float('nan')
                finally:
                    if auroc_metric_gpu: auroc_metric_gpu.reset()
            else:
                # OOM happened or wasn't requested
                final_metrics['AUC_GPU'] = float('nan')

    else: # Handle zero weight case
        for metric_name in metrics:
             if metric_name != 'IG': final_metrics[metric_name] = float('nan')

    # Ensure all requested metrics have a value (even if NaN)
    for metric_name in metrics:
        if metric_name != 'IG' and metric_name not in final_metrics:
             final_metrics[metric_name] = float('nan')


    # Calculate IG
    if 'IG' in metrics:
        ll_value = final_metrics.get('LL', float('nan'))
        if not np.isnan(ll_value):
            final_metrics['IG'] = ll_value - baseline_information_gain
        else:
            final_metrics['IG'] = float('nan')

    return final_metrics


def train_epoch(model, dataset, optimizer, device):
    model.train()
    losses = []
    batch_weights = []

    pbar = tqdm(dataset)
    for batch in pbar:
        optimizer.zero_grad()

        image = batch.pop('image').to(device)
        centerbias = batch.pop('centerbias').to(device)
        fixation_mask = batch.pop('fixation_mask').to(device)
        x_hist = batch.pop('x_hist', torch.tensor([])).to(device)
        y_hist = batch.pop('y_hist', torch.tensor([])).to(device)
        weights = batch.pop('weight').to(device)
        durations = batch.pop('durations', torch.tensor([])).to(device)

        kwargs = {}
        for key, value in dict(batch).items():
            kwargs[key] = value.to(device)

        if isinstance(model, DeepGazeII):
            log_density = model(image, centerbias, **kwargs)
        else:
            log_density = model(image, centerbias, x_hist=x_hist, y_hist=y_hist, durations=durations, **kwargs)

        loss = -log_likelihood(log_density, fixation_mask, weights=weights)
        losses.append(loss.detach().cpu().numpy())

        batch_weights.append(weights.detach().cpu().numpy().sum())

        pbar.set_description('{:.05f}'.format(np.average(losses, weights=batch_weights)))

        loss.backward()

        optimizer.step()

    return np.average(losses, weights=batch_weights)


def restore_from_checkpoint(model, optimizer, scheduler, path):
    print("Restoring from", path)
    data = torch.load(path, weights_only=False)
    if 'optimizer' in data:
        # checkpoint contains training progress
        model.load_state_dict(data['model'])
        optimizer.load_state_dict(data['optimizer'])
        scheduler.load_state_dict(data['scheduler'])
        torch.set_rng_state(data['rng_state'])
        return data['step'], data['loss']
    else:
        # checkpoint contains just a model
        missing_keys, unexpected_keys = model.load_state_dict(data, strict=False)
        if missing_keys:
            print("WARNING! missing keys", missing_keys)
        if unexpected_keys:
            print("WARNING! Unexpected keys", unexpected_keys)


def save_training_state(model, optimizer, scheduler, step, loss, path):
    data = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'rng_state': torch.get_rng_state(),
        'step': step,
        'loss': loss,
    }

    with atomic_save(path, text_mode=False, overwrite_part=True) as f:
        torch.save(data, f)




def _train(this_directory,
          model,
          train_loader, train_baseline_log_likelihood,
          val_loader, val_baseline_log_likelihood,
          optimizer, lr_scheduler,
          #optimizer_config, lr_scheduler_config,
          minimum_learning_rate,
          #initial_learning_rate, learning_rate_scheduler, learning_rate_decay, learning_rate_decay_epochs, learning_rate_backlook, learning_rate_reset_strategy, minimum_learning_rate,
          validation_metric='IG',
          validation_metrics=['IG', 'LL', 'AUC', 'NSS'],
          validation_epochs=1,
          startwith=None,
          device=None):
    mkdir_p(this_directory)

    if os.path.isfile(os.path.join(this_directory, 'final.pth')):
        print("Training Already finished")
        return

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    print("Using device", device)

    model.to(device)

    val_metrics = defaultdict(lambda: [])

    if startwith is not None:
        restore_from_checkpoint(model, optimizer, lr_scheduler, startwith)

    writer = SummaryWriter(os.path.join(this_directory, 'log'), flush_secs=30)

    columns = ['epoch', 'timestamp', 'learning_rate', 'loss']
    print("validation metrics", validation_metrics)
    for metric in validation_metrics:
        columns.append(f'validation_{metric}')

    progress = pd.DataFrame(columns=columns)

    step = 0
    last_loss = np.nan

    def save_step():

        save_training_state(
            model, optimizer, lr_scheduler, step, last_loss,
            '{}/step-{:04d}.pth'.format(this_directory, step),
        )

        #f = visualize(model, vis_data_loader)
        #display_if_in_IPython(f)

        #writer.add_figure('prediction', f, step)
        writer.add_scalar('training/loss', last_loss, step)
        writer.add_scalar('training/learning_rate', optimizer.state_dict()['param_groups'][0]['lr'], step)
        writer.add_scalar('parameters/sigma', model.finalizer.gauss.sigma.detach().cpu().numpy(), step)
        writer.add_scalar('parameters/center_bias_weight', model.finalizer.center_bias_weight.detach().cpu().numpy()[0], step)

        if step % validation_epochs == 0:
            _val_metrics = eval_epoch(model, val_loader, val_baseline_log_likelihood, device, metrics=validation_metrics)
        else:
            print("Skipping validation")
            _val_metrics = {}

        for key, value in _val_metrics.items():
            val_metrics[key].append(value)

        for key, value in _val_metrics.items():
            writer.add_scalar(f'validation/{key}', value, step)

        new_row = {
            'epoch': step,
            'timestamp': datetime.utcnow(),
            'learning_rate': optimizer.state_dict()['param_groups'][0]['lr'],
            'loss': last_loss,
            #'validation_ig': val_igs[-1]
        }
        for key, value in _val_metrics.items():
            new_row['validation_{}'.format(key)] = value

        progress.loc[step] = new_row

        print(progress.tail(n=2))
        print(progress[['validation_{}'.format(key) for key in val_metrics]].idxmax(axis=0))

        with atomic_save('{}/log.csv'.format(this_directory), text_mode=True, overwrite_part=True) as f:
            progress.to_csv(f)

        for old_step in range(1, step):
            # only check if we are computing validation metrics...
            if validation_metric in val_metrics and val_metrics[validation_metric] and old_step == np.argmax(val_metrics[validation_metric]):
                continue
            for filename in glob.glob('{}/step-{:04d}.pth'.format(this_directory, old_step)):
                print("removing", filename)
                os.remove(filename)

    old_checkpoints = sorted(glob.glob(os.path.join(this_directory, 'step-*.pth')))
    if old_checkpoints:
        last_checkpoint = old_checkpoints[-1]
        print("Found old checkpoint", last_checkpoint)
        step, last_loss = restore_from_checkpoint(model, optimizer, lr_scheduler, last_checkpoint)
        print("Setting step to", step)

    if step == 0:
        print("Beginning training")
        save_step()

    else:
        print("Continuing from step", step)
        progress = pd.read_csv(os.path.join(this_directory, 'log.csv'), index_col=0)
        val_metrics = {}
        for column_name in progress.columns:
            if column_name.startswith('validation_'):
                val_metrics[column_name.split('validation_', 1)[1]] = list(progress[column_name])

        if step not in progress.epoch.values:
            print("Epoch not yet evaluated, evaluating...")
            save_step()

        # We have to make one scheduler step here, since we make the
        # scheduler step _after_ saving the checkpoint
        lr_scheduler.step()

        print(progress)

    while optimizer.state_dict()['param_groups'][0]['lr'] >= minimum_learning_rate:
        step += 1
        last_loss = train_epoch(model, train_loader, optimizer, device)
        save_step()
        lr_scheduler.step()



    #if learning_rate_reset_strategy == 'validation':
     #   best_step = np.argmax(val_metrics[validation_metric])
     #   print("Best previous validation in step {}, saving as final result".format(best_step))
     #   restore_from_checkpoint(model, optimizer, scheduler, os.path.join(this_directory, 'step-{:04d}.pth'.format(best_step)))
    #else:
    #    print("Not resetting to best validation epoch")

    torch.save(model.state_dict(), '{}/final.pth'.format(this_directory))

    for filename in glob.glob(os.path.join(this_directory, 'step-*')):
        print("removing", filename)
        os.remove(filename)