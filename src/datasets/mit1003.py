# src/data/mit1003.py
import torch
import pysaliency
from torch.utils.data import DataLoader, DistributedSampler
import cloudpickle as cpickle
from pathlib import Path

from src.registry import register_data
from src.data import (
    ImageDataset, ImageDatasetWithSegmentation, FixationDataset,
    FixationMaskTransform, convert_stimuli, convert_fixation_trains, ImageDatasetSampler
)

@register_data("MIT1003")
def prepare_mit1003(cfg, ddp_ctx, logger):
    """Prepares the MIT1003 dataset, handling data conversion and cross-validation folds."""
    paths = cfg.paths
    extra = cfg.stage.extra or {}
    fold = extra.get("fold")
    if fold is None:
        raise ValueError("MIT1003 stage requires 'fold' to be set in stage.extra config.")
    
    # 1. Convert data if not already cached
    mit_converted_path = paths["train_dir"] / "MIT1003_converted"
    stimuli_cache = mit_converted_path / "stimuli.pkl"
    scanpaths_cache = mit_converted_path / "scanpaths.pkl"
    
    if ddp_ctx.is_master:
        mit_converted_path.mkdir(parents=True, exist_ok=True)
    ddp_ctx.barrier()
    
    # This logic should be run by all processes to load the data,
    # but only the master will write the cache.
    if stimuli_cache.exists() and scanpaths_cache.exists():
        logger.info(f"Loading cached converted MIT1003 data from {mit_converted_path}")
        with open(stimuli_cache, "rb") as f: stimuli_resized = cpickle.load(f)
        with open(scanpaths_cache, "rb") as f: scanpaths_resized = cpickle.load(f)
    else:
        logger.info("No cached MIT1003 data found, starting conversion...")
        stimuli_orig, fixations_orig = pysaliency.external_datasets.mit.get_mit1003_with_initial_fixation(location=str(paths["dataset_dir"]))
        stimuli_resized = convert_stimuli(stimuli_orig, mit_converted_path, ddp_ctx.is_master, ddp_ctx.enabled, ddp_ctx.device, logger)
        scanpaths_resized = convert_fixation_trains(stimuli_orig, fixations_orig, ddp_ctx.is_master, logger)
        if ddp_ctx.is_master:
            with open(stimuli_cache, "wb") as f: cpickle.dump(stimuli_resized, f)
            with open(scanpaths_cache, "wb") as f: cpickle.dump(scanpaths_resized, f)
    ddp_ctx.barrier()
    
    # 2. Split data and calculate baselines
    stim_train, scan_train = pysaliency.dataset_config.train_split(stimuli_resized, scanpaths_resized, crossval_folds=10, fold_no=fold)
    stim_val, scan_val = pysaliency.dataset_config.validation_split(stimuli_resized, scanpaths_resized, crossval_folds=10, fold_no=fold)
    
    fix_train = scan_train.to_fixations()
    fix_val = scan_val.to_fixations()

    centerbias = pysaliency.baseline_utils.CrossvalidatedBaselineModel(stimuli_resized, scanpaths_resized.to_fixations())

    train_ll, val_ll = None, None
    if ddp_ctx.is_master:
        train_ll = centerbias.information_gain(stim_train, fix_train, average='image')
        val_ll = centerbias.information_gain(stim_val, fix_val, average='image')
        logger.info(f"MIT1003 Fold {fold} Baselines - Train LL: {train_ll:.4f}, Val LL: {val_ll:.4f}")

    ll_bcast = [train_ll, val_ll]
    if ddp_ctx.enabled:
        torch.distributed.broadcast_object_list(ll_bcast, src=0)
    train_ll, val_ll = ll_bcast
    baseline_ll = {"train": train_ll, "val": val_ll}

    # 3. Create Datasets and DataLoaders based on stage type
    model_requires_segmentation = extra.get("requires_segmentation", False)
    ds_kwargs = {
        "transform": FixationMaskTransform(sparse=False),
        "average": "image"
    }
    if model_requires_segmentation:
        ds_kwargs["segmentation_mask_dir"] = paths.get("mit_all_mask_dir")

    is_scanpath_stage = "scanpath" in cfg.stage.name
    if is_scanpath_stage:
        train_dataset = FixationDataset(stim_train, scan_train, centerbias, included_fixations=[-1, -2, -3, -4], **ds_kwargs)
        val_dataset = FixationDataset(stim_val, scan_val, centerbias, included_fixations=[-1, -2, -3, -4], **ds_kwargs)
    else: # Spatial stage
        DatasetClass = ImageDatasetWithSegmentation if model_requires_segmentation else ImageDataset
        train_dataset = DatasetClass(stim_train, fix_train, centerbias, **ds_kwargs)
        val_dataset = DatasetClass(stim_val, fix_val, centerbias, **ds_kwargs)

    # MIT benefits from the shape-aware sampler
    if ddp_ctx.enabled:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
        train_loader = DataLoader(train_dataset, batch_size=cfg.stage.batch_size, sampler=train_sampler, num_workers=cfg.num_workers, pin_memory=True)
    else:
        # Use ImageDatasetSampler for single GPU, which handles batching itself
        train_batch_sampler = ImageDatasetSampler(train_dataset, batch_size=cfg.stage.batch_size, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, num_workers=cfg.num_workers, pin_memory=True)

    val_sampler = DistributedSampler(val_dataset, shuffle=False) if ddp_ctx.enabled else None
    val_loader = DataLoader(val_dataset, batch_size=cfg.stage.batch_size * 2, sampler=val_sampler, num_workers=cfg.num_workers, pin_memory=True)

    return train_loader, val_loader, baseline_ll