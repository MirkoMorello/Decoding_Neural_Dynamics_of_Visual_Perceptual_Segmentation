import torch
import pysaliency
from torch.utils.data import DataLoader, DistributedSampler, Subset
import cloudpickle as cpickle
from pathlib import Path
from boltons.fileutils import atomic_save

from src.registry import register_data
from src.data import (
    ImageDataset, ImageDatasetWithSegmentation, FixationDataset,
    FixationDatasetWithSegmentation, FixationMaskTransform,
    convert_stimuli, convert_fixation_trains, ImageDatasetSampler
)

# Define a constant for the project root to resolve relative paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

def _get_mit_data(cfg, ddp_ctx, logger):
    """
    Handles the acquisition and caching of MIT1003 stimuli and scanpaths.
    This centralized function prevents redundant downloads and conversions.
    """
    # Use a more specific cache directory to avoid conflicts
    mit_converted_path = cfg.paths["dataset_dir"] / "MIT1003_converted_cache"
    stimuli_cache = mit_converted_path / "stimuli.pkl"
    scanpaths_cache = mit_converted_path / "scanpaths.pkl"
    
    if ddp_ctx.is_master:
        mit_converted_path.mkdir(parents=True, exist_ok=True)
    ddp_ctx.barrier()
    
    # Check if both caches exist. If so, load from them.
    if stimuli_cache.exists() and scanpaths_cache.exists():
        if ddp_ctx.is_master:
            logger.info(f"Loading cached converted MIT1003 data from {mit_converted_path}")
        with open(stimuli_cache, "rb") as f:
            stimuli_resized = cpickle.load(f)
        with open(scanpaths_cache, "rb") as f:
            scanpaths_resized = cpickle.load(f)
        return stimuli_resized, scanpaths_resized

    # If caches don't exist, perform the conversion.
    if ddp_ctx.is_master:
        logger.info("No cached MIT1003 data found, starting conversion...")
    
    stimuli_orig, fixations_orig = pysaliency.external_datasets.mit.get_mit1003_with_initial_fixation(
        location=str(cfg.paths["dataset_dir"]),
        replace_initial_invalid_fixations=True
    )
    
    # All processes run conversion logic, but only master writes.
    stimuli_resized = convert_stimuli(stimuli_orig, mit_converted_path, ddp_ctx.is_master, ddp_ctx.enabled, ddp_ctx.device, logger)
    scanpaths_resized = convert_fixation_trains(stimuli_orig, fixations_orig, ddp_ctx.is_master, logger)
    
    if ddp_ctx.is_master:
        with atomic_save(str(stimuli_cache), text_mode=False, overwrite_part=True) as f:
            cpickle.dump(stimuli_resized, f)
        with atomic_save(str(scanpaths_cache), text_mode=False, overwrite_part=True) as f:
            cpickle.dump(scanpaths_resized, f)
    
    # All processes wait for the master to finish writing before proceeding.
    ddp_ctx.barrier()
    
    # Re-read from cache to ensure all processes have identical data.
    # This avoids potential minor discrepancies if conversion logic had non-deterministic elements.
    with open(stimuli_cache, "rb") as f:
        stimuli_resized = cpickle.load(f)
    with open(scanpaths_cache, "rb") as f:
        scanpaths_resized = cpickle.load(f)
        
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
    
    # Convert to fixations for baseline calculation
    fix_train = scan_train.to_fixations()
    fix_val = scan_val.to_fixations()

    # The baseline model should be trained on the full dataset for cross-validation
    centerbias = pysaliency.baseline_utils.CrossvalidatedBaselineModel(stimuli_resized, scanpaths_resized.to_fixations())

    train_ll, val_ll = None, None
    if ddp_ctx.is_master:
        # It's better to cache baselines per fold to avoid re-computation
        baseline_cache_dir = cfg.paths["train_dir"] / "MIT1003_baseline_cache"
        baseline_cache_dir.mkdir(exist_ok=True)
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

    # Broadcast baseline values to all processes
    ll_bcast = [train_ll, val_ll]
    if ddp_ctx.enabled:
        torch.distributed.broadcast_object_list(ll_bcast, src=0)
    train_ll, val_ll = ll_bcast
    baseline_ll = {"train": train_ll, "val": val_ll}

    # 3. Create Datasets based on explicit stage configuration
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

    # 4. Create DataLoaders with shape-aware sampling
    train_sampler = None # Will be used by _train for DDP set_epoch
    if ddp_ctx.enabled:
        # Create a sampler that deals with the whole dataset
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
        # Create a subset of the dataset for the current rank to use with the shape-aware sampler
        rank_subset_indices = list(iter(train_sampler))
        rank_train_dataset = Subset(train_dataset, rank_subset_indices)
        
        train_batch_sampler = ImageDatasetSampler(
            data_source=rank_train_dataset,
            batch_size=cfg.stage.batch_size,
            shuffle=True
        )
        train_loader = DataLoader(
            rank_train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=cfg.num_workers,
            pin_memory=True
        )
    else:
        # For single GPU, the shape-aware sampler works on the whole dataset
        train_batch_sampler = ImageDatasetSampler(train_dataset, batch_size=cfg.stage.batch_size, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, num_workers=cfg.num_workers, pin_memory=True)

    val_sampler = DistributedSampler(val_dataset, shuffle=False) if ddp_ctx.enabled else None
    val_loader = DataLoader(val_dataset, batch_size=cfg.stage.batch_size * 2, sampler=val_sampler, num_workers=cfg.num_workers, pin_memory=True, shuffle=False)

    return train_loader, val_loader, baseline_ll