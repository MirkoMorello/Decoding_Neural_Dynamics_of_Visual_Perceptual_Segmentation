#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation script for running inference-time control experiments on a trained model.
This version correctly uses the project's own data preparation pipeline from the DATA_REGISTRY.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# --- Add project root to path for imports ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
    format="[%(asctime)s][%(levelname)s] - %(message)s",
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
        original_sample = self.original_dataset[index]
        mismatched_mask_index = self.permuted_indices[index]
        mismatched_sample = self.original_dataset[mismatched_mask_index]
        original_sample['segmentation_mask'] = mismatched_sample['segmentation_mask']
        return original_sample

def run_evaluation_scenario(title: str, loader: DataLoader, model: torch.nn.Module, baseline_ll: float,
                            device: torch.device) -> dict:
    """Runs the evaluation loop for a given scenario and returns metrics."""
    logger.info(f"--- Starting Evaluation Scenario: {title} ---")
    metrics = eval_epoch(
        model, tqdm(loader, desc=f"Evaluating ({title})"), baseline_ll, device,
        ['IG', 'LL', 'AUC_GPU', 'NSS'], is_master=True, logger=logger)
    logger.info(f"--- Results for {title} ---")
    for key, value in metrics.items(): logger.info(f"  {key}: {value:.4f}")
    logger.info("-" * (len(title) + 20))
    return metrics

def main(eval_params: dict):
    """Main function to orchestrate the loading and evaluation process."""
    _auto_import_modules()
    
    settings = eval_params['settings']
    model_params = eval_params['model']
    dataset_params = eval_params['dataset']
    
    device = torch.device(settings['device'])
    logger.info(f"Using device: {device}")

    # --- 1. Load Model ---
    training_config_path = PROJECT_ROOT / model_params['training_config']
    checkpoint_path = PROJECT_ROOT / model_params['checkpoint']
    training_cfg = load_training_cfg(training_config_path)
    
    # Overwrite 'auto' num_workers from training config with integer from eval config
    if isinstance(training_cfg.num_workers, str):
        training_cfg.num_workers = settings['num_workers']

    model = MODEL_REGISTRY[model_params['model_key']](training_cfg)
    restore_from_checkpoint(model, path=checkpoint_path, device=device, logger=logger, load_weights_only=True)
    model.to(device)
    model.eval()

    # --- 2. Load Dataset using the project's own data preparation pipeline ---
    logger.info("Preparing dataset using the project's DATA_REGISTRY...")
    ddp_ctx = _SingleGpuDdpContext(device)
    _, val_loader, baseline_ll = DATA_REGISTRY[dataset_params['name']](training_cfg, ddp_ctx, logger)
    
    baseline_val_dataset = val_loader.dataset
    val_baseline_ll = baseline_ll['val']
    logger.info("Dataset preparation complete.")

    # --- 3. Run All Evaluation Scenarios ---
    all_results = []
    
    # Scenario 1: Baseline (Correct Masks)
    try:
        all_results.append(("Baseline", run_evaluation_scenario(
            "Baseline (Correct Masks)", val_loader, model, val_baseline_ll, device
        )))
    except Exception as e:
        logger.error(f"Baseline evaluation failed: {e}", exc_info=True)

    # Scenario 2: Control 1 (Zero Masks) - CORRECTED LOGIC
    try:
        non_existent_dir = PROJECT_ROOT / "this_directory_will_never_exist_abc123"
        
        # Re-create the dataset instance using components from the baseline validation dataset
        zero_mask_dataset = ImageDatasetWithSegmentation(
            stimuli=baseline_val_dataset.stimuli, 
            fixations=baseline_val_dataset.fixations, 
            centerbias_model=baseline_val_dataset.centerbias_model, 
            transform=baseline_val_dataset.transform,
            segmentation_mask_dir=non_existent_dir,
            average="image" # Assuming this was the setting during training
        )
        zero_mask_loader = DataLoader(
            zero_mask_dataset, batch_size=settings['batch_size'], num_workers=settings['num_workers'], pin_memory=True
        )
        all_results.append(("Control 1: Zero", run_evaluation_scenario(
            "Control 1 (Zero Masks)", zero_mask_loader, model, val_baseline_ll, device
        )))
    except Exception as e:
        logger.error(f"Control 1 (Zero Masks) failed: {e}", exc_info=True)
    
    # Scenario 3: Control 2 (Mismatched Masks)
    try:
        # The MismatchedMaskDataset wraps the baseline validation dataset directly
        mismatched_dataset = MismatchedMaskDataset(baseline_val_dataset, seed=settings['mismatched_mask_seed'])
        mismatched_loader = DataLoader(
            mismatched_dataset, batch_size=settings['batch_size'], num_workers=settings['num_workers'], pin_memory=True
        )
        all_results.append(("Control 2: Mismatched", run_evaluation_scenario(
            "Control 2 (Mismatched Masks)", mismatched_loader, model, val_baseline_ll, device
        )))
    except Exception as e:
        logger.error(f"Control 2 (Mismatched Masks) failed: {e}", exc_info=True)

    # --- 4. Print Final Summary Table ---
    if all_results:
        logger.info("\n" + "="*80)
        logger.info(" " * 28 + "EVALUATION SUMMARY")
        logger.info("="*80)
        header_keys = list(all_results[0][1].keys())
        header = f"{'Scenario':<25}" + "".join([f"{key:>12}" for key in header_keys])
        logger.info(header)
        logger.info("-" * len(header))
        for name, metrics in all_results:
            row = f"{name:<25}" + "".join([f"{metrics.get(key, float('nan')):>12.4f}" for key in header_keys])
            logger.info(row)
        logger.info("="*80)

# --- CLI Parsing (Unchanged) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference-time control experiments via a YAML config.")
    parser.add_argument("--eval_config", type=Path, required=True, help="Path to the evaluation YAML config file.")
    parser.add_argument("overrides", nargs="*", help="Key=value pairs to override YAML settings (e.g., 'settings.batch_size=4').")
    cli_args = parser.parse_args()
    try:
        with open(cli_args.eval_config, 'r') as f: eval_params = yaml.safe_load(f)
        logger.info(f"Loaded evaluation config from: {cli_args.eval_config}")
    except Exception as e:
        logger.error(f"Failed to load or parse evaluation config: {e}")
        sys.exit(1)
    for override in cli_args.overrides:
        try:
            key, value = override.split('=', 1)
            try: value = int(value)
            except ValueError:
                try: value = float(value)
                except ValueError:
                    if value.lower() == 'true': value = True
                    elif value.lower() == 'false': value = False
            _deep_set_dict(eval_params, key, value)
            logger.info(f"Overrode config: {key} = {value}")
        except Exception:
            logger.error(f"Invalid override format: '{override}'. Must be 'key=value'.")
    main(eval_params)