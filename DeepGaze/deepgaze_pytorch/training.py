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


def eval_epoch(model, dataset, baseline_information_gain, device, metrics=None):
    model.eval()

    if metrics is None:
        metrics = ['LL', 'IG', 'NSS', 'AUC']

    # --- Initialize Accumulators and Metrics on GPU ---
    # Use float64 for sums to avoid precision loss over many batches
    total_metric_sums = {
        metric_name: torch.tensor(0.0, device=device, dtype=torch.float64)
        for metric_name in metrics if metric_name in ['LL', 'NSS'] # Only those we sum manually
    }
    total_weight = torch.tensor(0.0, device=device, dtype=torch.float64)

    # Initialize torchmetrics AUROC if AUC is requested
    auroc_metric = None
    if 'AUC' in metrics:
        # Assuming binary classification: each pixel is either fixation (1) or not (0)
        # Adjust task if your setup is different (e.g., multiclass)
        auroc_metric = torchmetrics.AUROC(task="binary",
                                          thresholds=500).to(device) # remember that threshold results in an oversimplified AUC

    # --- Metric Functions Dictionary (excluding the slow custom AUC) ---
    metric_functions = {
        'LL': log_likelihood,
        'NSS': nss,
        # 'AUC': auc, # Don't use the old one
    }

    with torch.no_grad():
        pbar = tqdm(dataset, desc="Validating")
        for batch in pbar:
            # --- Move batch to GPU ---
            image = batch.pop('image').to(device)
            centerbias = batch.pop('centerbias').to(device)
            fixation_mask = batch.pop('fixation_mask').to(device) # Keep on GPU
            x_hist = batch.pop('x_hist', torch.tensor([])).to(device)
            y_hist = batch.pop('y_hist', torch.tensor([])).to(device)
            weights = batch.pop('weight').to(device) # Keep weights on GPU
            durations = batch.pop('durations', torch.tensor([])).to(device)

            kwargs = {}
            for key, value in dict(batch).items():
                kwargs[key] = value.to(device)

            # --- Model Forward Pass ---
            if isinstance(model, DeepGazeII):
                log_density = model(image, centerbias, **kwargs)
            else:
                log_density = model(image, centerbias, x_hist=x_hist, y_hist=y_hist, durations=durations, **kwargs)

            # --- Ensure fixation_mask is dense for calculations ---
            # This conversion might add some overhead if masks are large and sparse,
            # but necessary for torchmetrics and easier for standard metrics too.
            if isinstance(fixation_mask, torch.sparse.IntTensor):
                # Ensure it's float for some operations and int/long for torchmetrics target
                target_mask_int = fixation_mask.to_dense().long() # Target for AUROC
            else:
                target_mask_int = fixation_mask.long() # Target for AUROC


            # --- Calculate Batch Weight ---
            # Use the original weights sum for proper averaging
            # Note: Your original weighting scheme in LL/NSS was different,
            # this uses the standard approach. Adjust if needed.
            batch_weight_sum = weights.sum()

            # --- Accumulate Metrics ---
            if batch_weight_sum > 0:
                total_weight += batch_weight_sum

                # LL and NSS (using your functions returning batch averages)
                for metric_name, metric_fn in metric_functions.items():
                    if metric_name in total_metric_sums:
                        # Assumes metric_fn returns the batch average score (scalar tensor)
                        # Multiply by batch_weight_sum to get the batch SUM for correct overall average
                        metric_value_batch_avg = metric_fn(log_density, target_mask_int.float(), weights=weights) # Pass dense float mask

                        if not isinstance(metric_value_batch_avg, torch.Tensor) or metric_value_batch_avg.ndim != 0:
                             raise ValueError(f"Metric function {metric_name} did not return a scalar tensor!")

                        total_metric_sums[metric_name] += metric_value_batch_avg * batch_weight_sum


                # AUC (using torchmetrics)
                if auroc_metric is not None:
                    # Predictions: Use log_density or density. Using density might be more standard for AUC.
                    preds = torch.exp(log_density) # Convert log-density to density
                    # Flatten preds and targets for AUROC
                    # Target mask needs to be integer type (e.g., long)
                    auroc_metric.update(preds.flatten(), target_mask_int.flatten())

            # --- Optional: Update progress bar (requires some CPU transfer) ---
            desc = "Validating"
            if 'LL' in total_metric_sums and total_weight > 0:
                current_avg_ll = (total_metric_sums['LL'] / total_weight).item()
                desc += f' LL {current_avg_ll:.5f}'
            if auroc_metric is not None:
                 # .compute() is more expensive, maybe show intermediate sum/count or skip for pbar
                 pass # Avoid computing AUC every batch for speed
            pbar.set_description(desc)
            # ---

    # --- Final Calculation (after loop) ---
    final_metrics = {}
    total_weight_cpu = total_weight.item() # Single CPU transfer

    if total_weight_cpu > 0:
        # Calculate final averages for LL, NSS
        for metric_name, metric_sum in total_metric_sums.items():
            final_metrics[metric_name] = (metric_sum / total_weight).item() # Single CPU transfer per metric

        # Compute final AUC
        if auroc_metric is not None:
            try:
                final_metrics['AUC'] = auroc_metric.compute().item() # Compute epoch AUC, transfer result
            except Exception as e:
                print(f"WARNING: Failed to compute AUC: {e}")
                final_metrics['AUC'] = float('nan')
            finally:
                 auroc_metric.reset() # Reset metric state for potential next use
        else:
            if 'AUC' in metrics: final_metrics['AUC'] = float('nan') # Mark as NaN if requested but not computed


    else: # Handle case with zero total weight
        for metric_name in metrics:
             if metric_name != 'IG': # IG is derived
                 final_metrics[metric_name] = float('nan')

    # Calculate Information Gain (IG)
    if 'IG' in metrics:
        if 'LL' in final_metrics and not np.isnan(final_metrics['LL']):
             # baseline_information_gain is already a CPU float
            final_metrics['IG'] = final_metrics['LL'] - baseline_information_gain
        else:
            final_metrics['IG'] = float('nan')
    # ---

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