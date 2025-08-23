# Place this in the same file where you have @register_data("MIT1003")
# This is a complete replacement for your existing prepare_mit1003 function.

import torch
import pysaliency
from torch.utils.data import DataLoader
import cloudpickle as cpickle
from pathlib import Path
from boltons.fileutils import atomic_save

from src.registry import register_data
from src.data import (
    ImageDataset, ImageDatasetWithSegmentation, FixationDataset,
    FixationDatasetWithSegmentation, FixationMaskTransform,
    convert_stimuli, convert_fixation_trains, ImageDatasetSampler
)
from torch.utils.data.distributed import DistributedSampler

# Define a constant for the project root to resolve relative paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()


def _get_mit_data(cfg, ddp_ctx, logger):
    """
    Handles the acquisition and caching of MIT1003 stimuli and scanpaths.
    This version correctly separates the download and load steps for pysaliency.
    """
    mit_converted_path = cfg.paths["dataset_dir"] / "MIT1003_converted_cache"
    stimuli_cache = mit_converted_path / "stimuli.pkl"
    scanpaths_cache = mit_converted_path / "scanpaths.pkl"
    
    # --- 1. Master creates the cache directory if it doesn't exist ---
    if ddp_ctx.is_master:
        mit_converted_path.mkdir(parents=True, exist_ok=True)
    ddp_ctx.barrier()

    # --- 2. Check for stimuli cache ---
    if stimuli_cache.exists():
        if ddp_ctx.is_master:
            logger.info(f"Loading cached converted MIT1003 stimuli from {stimuli_cache}")
    else:
        # --- Stimuli cache is missing, must generate it ---
        if ddp_ctx.is_master:
            logger.info("Cached stimuli not found. Starting raw data acquisition and conversion.")
            
            logger.info(f"Ensuring raw MIT1003 data is downloaded to {cfg.paths['dataset_dir']}...")
            pysaliency.get_mit1003(location=str(cfg.paths["dataset_dir"]))
            logger.info("Raw data download/check complete.")
            
            # Step 2b: Now, load the raw data from the HDF5 file (which is guaranteed to exist).
            stimuli_orig, _ = pysaliency.external_datasets.mit.get_mit1003_with_initial_fixation(
                location=str(cfg.paths["dataset_dir"]),
                replace_initial_invalid_fixations=True
            )
            logger.info("Raw MIT1003 stimuli loaded. Converting...")
            
            stimuli_resized = convert_stimuli(stimuli_orig, mit_converted_path, ddp_ctx.is_master, ddp_ctx.enabled, ddp_ctx.device, logger)
            
            with atomic_save(str(stimuli_cache), text_mode=False, overwrite_part=True) as f:
                cpickle.dump(stimuli_resized, f)
            logger.info("Stimuli conversion complete and cached.")
            
        ddp_ctx.barrier() # All processes wait for master to finish writing the cache
    
    # --- 3. All processes now load the stimuli cache (guaranteed to exist) ---
    with open(stimuli_cache, "rb") as f:
        stimuli_resized = cpickle.load(f)

    # --- 4. Check for scanpaths cache ---
    if scanpaths_cache.exists():
        if ddp_ctx.is_master:
            logger.info(f"Loading cached converted MIT1003 scanpaths from {scanpaths_cache}")
    else:
        # --- Scanpaths cache is missing, must generate it ---
        if ddp_ctx.is_master:
            logger.info("Cached scanpaths not found. Starting raw data acquisition and conversion.")

            # We don't need the download call again here, because if we've reached this point,
            # the stimuli block above has already ensured the raw data exists.
            stimuli_orig, fixations_orig = pysaliency.external_datasets.mit.get_mit1003_with_initial_fixation(
                location=str(cfg.paths["dataset_dir"]),
                replace_initial_invalid_fixations=True
            )
            logger.info("Raw MIT1003 fixations loaded. Converting...")

            scanpaths_resized = convert_fixation_trains(stimuli_orig, fixations_orig, ddp_ctx.is_master, logger)
            
            with atomic_save(str(scanpaths_cache), text_mode=False, overwrite_part=True) as f:
                cpickle.dump(scanpaths_resized, f)
            logger.info("Scanpaths conversion complete and cached.")
        
        ddp_ctx.barrier()
        
    # --- 5. All processes now load the scanpaths cache (guaranteed to exist) ---
    with open(scanpaths_cache, "rb") as f:
        scanpaths_resized = cpickle.load(f)

    if ddp_ctx.is_master:
        logger.info("MIT1003 data preparation complete.")
        
    return stimuli_resized, scanpaths_resized

@register_data("MIT1003")
def prepare_mit1003(cfg, ddp_ctx, logger):
    """Prepares the MIT1003 dataset, handling data conversion and cross-validation folds."""
    extra = cfg.stage.extra
    fold = extra.get("fold")
    if fold is None:
        raise ValueError("MIT1003 stage requires 'fold' to be set in stage.extra config.")
    
    # 1. Get or convert data
    stimuli_resized, scanpaths_resized = _get_mit_data(cfg, ddp_ctx, logger)
    
    # 2. Split data and calculate fold-specific baselines
    stim_train, scan_train = pysaliency.dataset_config.train_split(stimuli_resized, scanpaths_resized, crossval_folds=10, fold_no=fold)
    stim_val, scan_val = pysaliency.dataset_config.validation_split(stimuli_resized, scanpaths_resized, crossval_folds=10, fold_no=fold)
    
    fix_train = scan_train
    fix_val = scan_val

    centerbias = pysaliency.baseline_utils.CrossvalidatedBaselineModel(
        stimuli_resized, 
        scanpaths_resized, 
        bandwidth=10**-1.6667673342543432, 
        eps=10**-14.884189168516073, 
        caching=False
    )

    train_ll, val_ll = None, None
    if ddp_ctx.is_master:
        baseline_cache_dir = cfg.paths["dataset_dir"] / "MIT1003_baseline_cache"
        baseline_cache_dir.mkdir(exist_ok=True, parents=True)
        train_ll_cache = baseline_cache_dir / f'train_ll_fold{fold}.pkl'
        val_ll_cache = baseline_cache_dir / f'val_ll_fold{fold}.pkl'
        
        try:
            with open(train_ll_cache, 'rb') as f: train_ll = cpickle.load(f)
        except (FileNotFoundError, EOFError):
            train_ll = centerbias.information_gain(stim_train, fix_train, average='image')
            with open(train_ll_cache, 'wb') as f: cpickle.dump(train_ll, f)
            
        try:
            with open(val_ll_cache, 'rb') as f: val_ll = cpickle.load(f)
        except (FileNotFoundError, EOFError):
            val_ll = centerbias.information_gain(stim_val, fix_val, average='image')
            with open(val_ll_cache, 'wb') as f: cpickle.dump(val_ll, f)

        logger.info(f"MIT1003 Fold {fold} Baselines - Train LL: {train_ll:.4f}, Val LL: {val_ll:.4f}")

    ll_bcast = [train_ll, val_ll]
    if ddp_ctx.enabled:
        torch.distributed.broadcast_object_list(ll_bcast, src=0)
    train_ll, val_ll = ll_bcast
    baseline_ll = {"train": train_ll, "val": val_ll}

    # --- Step 3. Create Datasets based on explicit stage configuration ---
    model_requires_segmentation = extra.get("requires_segmentation", False)
    is_scanpath_stage = extra.get("is_scanpath_stage", False)
    
    ds_kwargs = {
        "transform": FixationMaskTransform(sparse=False),
        "average": "image"
    }
    
    if model_requires_segmentation:
        mask_rel_path = extra.get("mask_dir")
        if not mask_rel_path:
            raise ValueError("MIT1003 with segmentation requires 'mask_dir' in stage.extra.")
        ds_kwargs["segmentation_mask_dir"] = PROJECT_ROOT / mask_rel_path
        logger.info(f"Using MIT1003 masks from: {ds_kwargs['segmentation_mask_dir']}")

    if is_scanpath_stage:
        included_fixations = extra.get("included_fixations")
        if not included_fixations:
            raise ValueError("'is_scanpath_stage' is true, but 'included_fixations' is not defined.")
        ds_kwargs["included_fixations"] = included_fixations
        ds_kwargs["allow_missing_fixations"] = True
        
        DatasetClass = FixationDatasetWithSegmentation if model_requires_segmentation else FixationDataset
        train_dataset = DatasetClass(stim_train, scan_train, centerbias, **ds_kwargs)
        val_dataset = DatasetClass(stim_val, scan_val, centerbias, **ds_kwargs)
    else: # Spatial stage
        DatasetClass = ImageDatasetWithSegmentation if model_requires_segmentation else ImageDataset
        train_dataset = DatasetClass(stim_train, fix_train, centerbias, **ds_kwargs)
        val_dataset = DatasetClass(stim_val, fix_val, centerbias, **ds_kwargs)

    # --- Step 4. Create DataLoaders ---
    logger.info("Creating DataLoaders to match the old script's behavior.")
    logger.warning("Using standard DistributedSampler for validation, which may be unstable.")

    # 4a. Create the TRAIN loader (This part is IDENTICAL to the old script, which used ImageDatasetSampler)
    train_batch_sampler = ImageDatasetSampler(
        data_source=train_dataset,
        batch_size=cfg.stage.batch_size,
        shuffle=True,
        drop_last=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0
    )

    # 4b. Create the VALIDATION loader (This now matches the old script's unstable method)
    if ddp_ctx.enabled:
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=ddp_ctx.world,  # Correct attribute
            rank=ddp_ctx.rank,          # Correct attribute
            shuffle=False,
            drop_last=False
        )
        # The dataloader needs shuffle=False because the sampler handles it
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.stage.batch_size,
            sampler=val_sampler,
            num_workers=cfg.num_workers,
            pin_memory=True,
            persistent_workers=cfg.num_workers > 0
        )
    else: # Single-GPU case
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.stage.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            persistent_workers=cfg.num_workers > 0
        )

    return train_loader, val_loader, baseline_ll