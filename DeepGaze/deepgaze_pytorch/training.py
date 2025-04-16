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
    total_metric_sums = {
        metric_name: torch.tensor(0.0, device=device, dtype=torch.float64)
        for metric_name in metrics if metric_name in ['LL', 'NSS']
    }
    total_weight = torch.tensor(0.0, device=device, dtype=torch.float64)

    # Initialize torchmetrics AUROC - REMOVE thresholds argument
    auroc_metric = None
    if 'AUC' in metrics:
        # <<< DEFINE THRESHOLDS BASED ON OBSERVED log_density RANGE >>>
        # Get range from debug prints, maybe slightly wider.
        # These values might need tuning if the range changes significantly across epochs/datasets.
        min_log_density_observed = -26.0  # Lower bound (a bit below observed min -24.6)
        max_log_density_observed = -8.0   # Upper bound (a bit above observed max -9.6)
        num_thresholds_to_use = 500       # Keep the number reasonable

        # Create a tensor of thresholds linearly spaced within the observed range
        threshold_tensor = torch.linspace(
            min_log_density_observed,
            max_log_density_observed,
            steps=num_thresholds_to_use, # Use 'steps' argument
            device=device
        )

        print(f"Using {num_thresholds_to_use} AUROC thresholds between {min_log_density_observed:.2f} and {max_log_density_observed:.2f}") # Add print statement

        auroc_metric = torchmetrics.AUROC(
            task="binary",
            thresholds=threshold_tensor  # <<< PASS THE CUSTOM THRESHOLD TENSOR
        ).to(device)

    metric_functions = { 'LL': log_likelihood, 'NSS': nss }

    # <<< DEBUG FLAG >>>
    printed_debug_info = False

    # torch.cuda.empty_cache() # Optional: uncomment if you suspect fragmentation OOMs

    with torch.no_grad():
        pbar = tqdm(dataset, desc="Validating")
        for batch_idx, batch in enumerate(pbar): # Use enumerate for index
            # --- Move batch to GPU ---
            # (Same as before)
            image = batch.pop('image').to(device)
            centerbias = batch.pop('centerbias').to(device)
            fixation_mask = batch.pop('fixation_mask').to(device)
            x_hist = batch.pop('x_hist', torch.tensor([])).to(device)
            y_hist = batch.pop('y_hist', torch.tensor([])).to(device)
            weights = batch.pop('weight').to(device)
            durations = batch.pop('durations', torch.tensor([])).to(device)
            kwargs = {k: v.to(device) for k, v in batch.items()}

            # --- Model Forward Pass ---
            # (Same as before)
            if isinstance(model, DeepGazeII):
                log_density = model(image, centerbias, **kwargs)
            else:
                log_density = model(image, centerbias, x_hist=x_hist, y_hist=y_hist, durations=durations, **kwargs)

            # --- Ensure fixation_mask is dense ---
            # (Same as before)
            if isinstance(fixation_mask, torch.sparse.IntTensor):
                target_mask_int = fixation_mask.to_dense().long()
            else:
                target_mask_int = fixation_mask.long()

            # --- Calculate Batch Weight ---
            # (Same as before)
            batch_weight_sum = weights.sum()

            # --- Accumulate Metrics ---
            if batch_weight_sum > 0:
                total_weight += batch_weight_sum

                # LL and NSS
                # (Same as before)
                for metric_name, metric_fn in metric_functions.items():
                    if metric_name in total_metric_sums:
                        metric_value_batch_avg = metric_fn(log_density, target_mask_int.float(), weights=weights)
                        if not isinstance(metric_value_batch_avg, torch.Tensor) or metric_value_batch_avg.ndim != 0:
                             raise ValueError(f"Metric function {metric_name} did not return a scalar tensor!")
                        total_metric_sums[metric_name] += metric_value_batch_avg * batch_weight_sum

                # AUC (using torchmetrics)
                if auroc_metric is not None:
                    preds_for_auc = log_density
                    targets_flat = target_mask_int.flatten()
                    preds_flat = preds_for_auc.flatten()

                    # <<< ----- DEBUG PRINTS (Runs only for first batch) ----- >>>
                    if not printed_debug_info:
                        print("\n--- AUC DEBUG INFO (First Batch) ---")
                        print(f"Targets shape: {targets_flat.shape}")
                        print(f"Targets unique values: {torch.unique(targets_flat)}")
                        print(f"Targets sum: {targets_flat.sum().item()}")
                        num_positives = (targets_flat > 0).sum().item()
                        print(f"Number of positive targets (>0): {num_positives}")

                        print(f"\nPredictions shape: {preds_flat.shape}")
                        print(f"Predictions contain NaN: {torch.isnan(preds_flat).any().item()}")
                        print(f"Predictions contain Inf: {torch.isinf(preds_flat).any().item()}")
                        if not torch.isnan(preds_flat).any() and not torch.isinf(preds_flat).any():
                            print(f"Predictions min: {preds_flat.min().item():.4f}")
                            print(f"Predictions max: {preds_flat.max().item():.4f}")
                            print(f"Predictions mean: {preds_flat.mean().item():.4f}")
                            # Check predictions at positive locations
                            if num_positives > 0:
                                preds_at_positives = preds_flat[targets_flat > 0]
                                print(f"Predictions @ Positives min: {preds_at_positives.min().item():.4f}")
                                print(f"Predictions @ Positives max: {preds_at_positives.max().item():.4f}")
                                print(f"Predictions @ Positives mean: {preds_at_positives.mean().item():.4f}")
                        print("-------------------------------------\n")
                        printed_debug_info = True
                    # <<< ----- END DEBUG PRINTS ----- >>>

                    # Check if there are both positive and negative samples before updating
                    if torch.unique(targets_flat).numel() > 1:
                         auroc_metric.update(preds_flat, targets_flat)
                    else:
                         # If only one class present in the batch, AUC is not well-defined for this batch alone
                         # torchmetrics handles accumulation correctly, but we can print a warning
                         if not printed_debug_info: # Avoid repeating for every batch
                             print(f"WARNING: Skipping AUROC update for batch {batch_idx} because only one class is present.")


            # --- Optional: Update progress bar ---
            # (Same as before)
            desc = "Validating"
            if 'LL' in total_metric_sums and total_weight > 0:
                current_avg_ll = (total_metric_sums['LL'] / total_weight).item()
                desc += f' LL {current_avg_ll:.5f}'
            pbar.set_description(desc)

    # --- Final Calculation (after loop) ---
    # (Same as before - computes final metrics from accumulators)
    final_metrics = {}
    total_weight_cpu = total_weight.item()

    if total_weight_cpu > 0:
        for metric_name, metric_sum in total_metric_sums.items():
            final_metrics[metric_name] = (metric_sum / total_weight).item()

        if auroc_metric is not None:
            try:
                # Compute final AUC - ensure state has been updated at least once
                if auroc_metric.update_count > 0: # Check internal counter or similar state
                    final_metrics['AUC'] = auroc_metric.compute().item()
                else:
                    print("WARNING: AUROC metric was never updated (no valid batches found?). Setting AUC to NaN.")
                    final_metrics['AUC'] = float('nan')

            except Exception as e:
                # Catch potential errors during compute, e.g., if state is invalid
                print(f"WARNING: Failed to compute AUC: {e}")
                print("AUROC Metric State:", auroc_metric) # Print state for debugging
                final_metrics['AUC'] = float('nan')
            finally:
                 auroc_metric.reset()
        else:
            if 'AUC' in metrics: final_metrics['AUC'] = float('nan')

    else:
        for metric_name in metrics:
             if metric_name != 'IG':
                 final_metrics[metric_name] = float('nan')

    # Calculate Information Gain (IG)
    # (Same as before)
    if 'IG' in metrics:
        if 'LL' in final_metrics and not np.isnan(final_metrics['LL']):
            final_metrics['IG'] = final_metrics['LL'] - baseline_information_gain
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