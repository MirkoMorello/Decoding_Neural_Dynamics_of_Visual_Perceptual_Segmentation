#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation script for running inference-time control experiments on a trained model.
Supports multi-fold cross-validation analysis with parallel execution across multiple GPUs.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any
import copy
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# --- Add project root to path for imports ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
# ---

# Import necessary components from your project
from src.data import ImageDatasetWithSegmentation
from src.registry import DATA_REGISTRY, MODEL_REGISTRY
from src.training import eval_epoch, restore_from_checkpoint
from src.train import _load_cfg as load_training_cfg, _auto_import_modules

# --- Setup basic logging ---
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s][%(processName)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# --- Helper Classes ---
class _SingleGpuDdpContext:
    """A mock DDP context for single-GPU evaluation."""
    def __init__(self, device):
        self.rank, self.world, self.local_rank = 0, 1, 0
        self.enabled, self.is_master = False, True
        self.device = device
    def barrier(self): pass

class MismatchedMaskDataset(Dataset):
    """A Dataset wrapper that systematically mismatches segmentation masks."""
    def __init__(self, original_dataset: Dataset, seed: int = 42):
        self.original_dataset = original_dataset
        indices = np.arange(len(self.original_dataset))
        rng = np.random.default_rng(seed)
        self.permuted_indices = rng.permutation(indices)
        for i in range(len(indices)):
            if self.permuted_indices[i] == i:
                swap_with = (i + 1) % len(indices)
                p_i, p_swap = self.permuted_indices[i], self.permuted_indices[swap_with]
                self.permuted_indices[i], self.permuted_indices[swap_with] = p_swap, p_i
        logger.info(f"Initialized MismatchedMaskDataset with {len(self)} items.")

    def __len__(self) -> int:
        return len(self.original_dataset)

    def __getitem__(self, index: int) -> dict:
        original_sample = copy.deepcopy(self.original_dataset[index])
        mismatched_mask_index = self.permuted_indices[index]
        mismatched_sample = self.original_dataset[mismatched_mask_index]
        original_sample['segmentation_mask'] = mismatched_sample['segmentation_mask']
        return original_sample

# --- Core Evaluation Logic ---
def run_evaluation_scenario(title: str, loader: DataLoader, model: torch.nn.Module, baseline_ll: float,
                            device: torch.device) -> dict:
    """Runs the evaluation loop for a given scenario and returns metrics."""
    logger.info(f"--- Starting Evaluation Scenario: {title} ---")
    metrics = eval_epoch(
        model, tqdm(loader, desc=f"Evaluating ({title})", leave=False), baseline_ll, device,
        ['IG', 'LL', 'AUC_GPU', 'NSS'], is_master=True, logger=logger
    )
    logger.info(f"--- Results for {title} ---")
    for key, value in metrics.items(): logger.info(f"  {key}: {value:.4f}")
    return metrics

def run_all_scenarios_for_fold(model, val_loader, val_baseline_ll, settings, device):
    """Runs all three control experiments for a given model and dataset."""
    fold_results = []
    baseline_val_dataset = val_loader.dataset

    # Scenario 1: Baseline (Correct Masks)
    try:
        metrics = run_evaluation_scenario("Baseline (Correct Masks)", val_loader, model, val_baseline_ll, device)
        fold_results.append(("Baseline", metrics))
    except Exception as e:
        logger.error(f"Baseline evaluation failed: {e}", exc_info=True)

    # Scenario 2: Control 1 (Zero Masks)
    try:
        non_existent_dir = PROJECT_ROOT / "this_directory_will_never_exist_abc123"
        zero_mask_dataset = ImageDatasetWithSegmentation(
            stimuli=baseline_val_dataset.stimuli, fixations=baseline_val_dataset.fixations,
            centerbias_model=baseline_val_dataset.centerbias_model, transform=baseline_val_dataset.transform,
            segmentation_mask_dir=non_existent_dir, average="image"
        )
        zero_mask_loader = DataLoader(
            zero_mask_dataset, batch_size=settings['batch_size'], num_workers=0, pin_memory=True
        )
        metrics = run_evaluation_scenario("Control 1 (Zero Masks)", zero_mask_loader, model, val_baseline_ll, device)
        fold_results.append(("Control 1: Zero", metrics))
    except Exception as e:
        logger.error(f"Control 1 (Zero Masks) failed: {e}", exc_info=True)
    
    # Scenario 3: Control 2 (Mismatched Masks)
    try:
        mismatched_dataset = MismatchedMaskDataset(baseline_val_dataset, seed=settings['mismatched_mask_seed'])
        mismatched_loader = DataLoader(
            mismatched_dataset, batch_size=settings['batch_size'], num_workers=0, pin_memory=True
        )
        metrics = run_evaluation_scenario("Control 2 (Mismatched Masks)", mismatched_loader, model, val_baseline_ll, device)
        fold_results.append(("Control 2: Mismatched", metrics))
    except Exception as e:
        logger.error(f"Control 2 (Mismatched Masks) failed: {e}", exc_info=True)

    return fold_results

def summarize_cross_validation_results(
    all_results: list[tuple[int, list[tuple[str, dict]]]],
    output_dir: Path | None = None
):
    """
    Aggregates results from all folds, prints a summary table with mean and std,
    and saves the raw and summarized results to CSV files.
    """
    if not all_results:
        logger.warning("No results to summarize. This can happen if all checkpoints were missing.")
        return

    # 1. Prepare a detailed DataFrame with all raw results
    records = [
        {'Fold': fold_idx, 'Scenario': scenario_name, **metrics}
        for fold_idx, scenarios in all_results
        for scenario_name, metrics in scenarios
    ]
    df_raw = pd.DataFrame(records)
    if df_raw.empty:
        logger.warning("DataFrame of results is empty, cannot summarize.")
        return
        
    # 2. Calculate the summary statistics (mean and std)
    df_summary = df_raw.groupby('Scenario').agg(['mean', 'std']).drop(columns='Fold')
    
    # 3. Print the formatted summary table to the console
    logger.info("\n" + "="*80)
    logger.info(" " * 25 + "CROSS-VALIDATION SUMMARY")
    logger.info("="*80)
    
    scenarios = sorted(df_raw['Scenario'].unique())
    metrics = [col for col in df_raw.columns if col not in ['Fold', 'Scenario']]
    
    header = f"{'Scenario':<25}" + "".join([f"{key:>15}" for key in metrics])
    logger.info(header)
    logger.info("-" * len(header))
    
    for scenario in scenarios:
        row_str = f"{scenario:<25}"
        for metric in metrics:
            mean_val = df_summary.loc[scenario, (metric, 'mean')]
            std_val = df_summary.loc[scenario, (metric, 'std')]
            row_str += f"{mean_val:>8.4f} ± {std_val:<5.4f}"
        logger.info(row_str)
    logger.info("="*80)

    # 4. Save the results to files if an output directory is provided
    if output_dir:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save the raw, per-fold results
            raw_results_path = output_dir / "all_folds_raw_results.csv"
            df_raw.to_csv(raw_results_path, index=False)
            logger.info(f"Saved raw per-fold results to: {raw_results_path}")

            # Save the final summary table
            summary_results_path = output_dir / "summary_mean_std_results.csv"
            df_summary.to_csv(summary_results_path)
            logger.info(f"Saved summary (mean ± std) results to: {summary_results_path}")

        except Exception as e:
            logger.error(f"Failed to save result files: {e}", exc_info=True)


# --- Worker Function for a Single Fold ---
def run_evaluation_for_single_fold(args):
    """
    This function is executed in a separate process. It handles everything
    for evaluating one specific fold on one specific GPU.
    """
    eval_params, fold, device_id = args
    device = torch.device(f"cuda:{device_id}")
    
    logger.info(f"Starting evaluation for FOLD {fold} on device {device}.")

    settings = eval_params['settings']
    cv_config = eval_params['cross_validation']

    # 1. Construct Paths
    ckpt_template = cv_config['checkpoint_path_template']
    cfg_template = cv_config['training_config_template']
    checkpoint_path = PROJECT_ROOT / ckpt_template.format(fold=fold)
    training_config_path = PROJECT_ROOT / cfg_template.format(fold=fold)
    
    if not checkpoint_path.exists():
        logger.warning(f"Checkpoint for fold {fold} not found at {checkpoint_path}. Skipping.")
        return (fold, [])

    # 2. Load Config and Model
    _auto_import_modules()
    training_cfg = load_training_cfg(training_config_path)
    
    # IMPORTANT: Set num_workers from eval_config, NOT training_config for DataLoaders created by the registry
    training_cfg.num_workers = 0

    # Correctly set the fold number in the structured config
    if not hasattr(training_cfg.stage, 'extra'):
        setattr(training_cfg.stage, 'extra', {})
    training_cfg.stage.extra['fold'] = fold

    model_key = eval_params['model']['model_key']
    model = MODEL_REGISTRY[model_key](training_cfg)
    restore_from_checkpoint(model, path=checkpoint_path, device=device, logger=logger, load_weights_only=True)
    model.to(device)
    model.eval()

    # 3. Load Dataset
    ddp_ctx = _SingleGpuDdpContext(device)
    dataset_name = eval_params['dataset']['name']
    _, val_loader, baseline_ll = DATA_REGISTRY[dataset_name](training_cfg, ddp_ctx, logger)
    val_baseline_ll = baseline_ll['val']
    
    # 4. Run all scenarios
    fold_results = run_all_scenarios_for_fold(model, val_loader, val_baseline_ll, settings, device)
    
    logger.info(f"Finished evaluation for FOLD {fold} on device {device}.")
    return (fold, fold_results)

# --- Main Orchestrator Function ---
def main(eval_params: dict, cli_args: argparse.Namespace):
    cv_config = eval_params.get('cross_validation', {})
    if not (cv_config.get('enabled', False)):
        raise ValueError("Multi-GPU cross-validation is not enabled in the config. Please set 'cross_validation.enabled = true'.")

    gpu_ids = eval_params['settings'].get('gpus', [0])
    num_gpus = len(gpu_ids)
    num_folds = cv_config['num_folds']
    
    # Define a dedicated output directory for the results
    output_dir = PROJECT_ROOT / "evaluation_results" / cli_args.eval_config.stem
    logger.info(f"Results will be saved to: {output_dir}")
    
    logger.info(f"Starting cross-validation for {num_folds} folds using {num_gpus} GPU(s): {gpu_ids}")

    # Prepare a list of tasks.
    tasks = [
        (eval_params, fold_idx, gpu_ids[fold_idx % num_gpus])
        for fold_idx in range(num_folds)
    ]

    # Use a multiprocessing Pool to execute tasks in parallel ('spawn' is safest for CUDA).
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=num_gpus) as pool:
        all_fold_results = []
        for result in tqdm(pool.imap_unordered(run_evaluation_for_single_fold, tasks), total=len(tasks), desc="Evaluating Folds"):
            if result and result[1]: # Ensure result is not empty
                all_fold_results.append(result)

    # Sort results by fold index for consistent reporting
    all_fold_results.sort(key=lambda x: x[0])

    # Summarize final results and save to files
    summarize_cross_validation_results(all_fold_results, output_dir=output_dir)

def _deep_set_dict(d: dict, key_str: str, value: Any):
    """Helper to set a nested dictionary key using a dot-separated string."""
    keys = key_str.split('.')
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value

# --- CLI Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference-time control experiments via a YAML config.")
    parser.add_argument("--eval_config", type=Path, required=True, help="Path to the evaluation YAML config file.")
    parser.add_argument("overrides", nargs="*", help="Key=value pairs to override YAML settings (e.g., 'settings.gpus=[0]').")
    cli_args = parser.parse_args()
    try:
        with open(cli_args.eval_config, 'r') as f: eval_params = yaml.safe_load(f)
        logger.info(f"Loaded evaluation config from: {cli_args.eval_config}")
    except Exception as e:
        logger.error(f"Failed to load or parse evaluation config: {e}")
        sys.exit(1)
    
    for override in cli_args.overrides:
        try:
            key, value_str = override.split('=', 1)
            # Use yaml.safe_load to handle complex types like lists [0,1]
            try: value = yaml.safe_load(value_str)
            except (yaml.YAMLError, ValueError):
                value = value_str # Fallback to string if parsing fails
            
            _deep_set_dict(eval_params, key, value)
            logger.info(f"Overrode config: {key} = {value}")
        except Exception:
            logger.error(f"Invalid override format: '{override}'. Must be 'key.subkey=value'.")

    main(eval_params, cli_args)