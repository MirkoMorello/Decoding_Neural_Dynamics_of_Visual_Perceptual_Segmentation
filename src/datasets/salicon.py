import torch
import pysaliency
from torch.utils.data import DataLoader, DistributedSampler
import cloudpickle as cpickle
from pathlib import Path
from boltons.fileutils import atomic_save # It's good practice to use atomic_save for caching

from src.registry import register_data
from src.data import ImageDataset, ImageDatasetWithSegmentation, FixationMaskTransform
from pysaliency.baseline_utils import BaselineModel

# Define a constant for the project root to resolve relative paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

@register_data("SALICON")
def prepare_salicon(cfg, ddp_ctx, logger):
    """
    Prepares the SALICON dataset, dataloaders, and baselines.
    This version is designed for the generic training script, uses centralized
    caching, and reads stage-specific parameters from the config.
    """
    global_paths = cfg.paths
    stage_extra = cfg.stage.extra

    salicon_loc = global_paths["dataset_dir"] / 'SALICON'
    
    # 1. Download data (master only)
    if ddp_ctx.is_master:
        logger.info(f"Checking for SALICON data at: {salicon_loc}")
        pysaliency.get_SALICON_train(location=str(salicon_loc.parent))
        pysaliency.get_SALICON_val(location=str(salicon_loc.parent))
    ddp_ctx.barrier()

    train_stim, train_fix = pysaliency.get_SALICON_train(location=str(salicon_loc.parent))
    val_stim, val_fix = pysaliency.get_SALICON_val(location=str(salicon_loc.parent))

    # 2. Calculate and broadcast baseline log-likelihood from a central cache
    train_ll, val_ll = None, None
    if ddp_ctx.is_master:
        baseline_cache_dir = global_paths["dataset_dir"] / "SALICON_baseline_cache"
        baseline_cache_dir.mkdir(parents=True, exist_ok=True)
        train_ll_cache = baseline_cache_dir / 'salicon_baseline_train_ll.pkl'
        val_ll_cache = baseline_cache_dir / 'salicon_baseline_val_ll.pkl'

        centerbias = BaselineModel(
            stimuli=train_stim, fixations=train_fix,
            bandwidth=0.0217, eps=2e-13, caching=False
        )
        try:
            with open(train_ll_cache, 'rb') as f: train_ll = cpickle.load(f)
        except (FileNotFoundError, EOFError):
            logger.info("Calculating SALICON train baseline LL...")
            train_ll = centerbias.information_gain(train_stim, train_fix, average='image')
            with atomic_save(train_ll_cache, text_mode=False, overwrite_part=True) as f:
                cpickle.dump(train_ll, f)
        try:
            with open(val_ll_cache, 'rb') as f: val_ll = cpickle.load(f)
        except (FileNotFoundError, EOFError):
            logger.info("Calculating SALICON validation baseline LL...")
            val_ll = centerbias.information_gain(val_stim, val_fix, average='image')
            with atomic_save(val_ll_cache, text_mode=False, overwrite_part=True) as f:
                cpickle.dump(val_ll, f)
        logger.info(f"SALICON Baselines - Train LL: {train_ll:.4f}, Val LL: {val_ll:.4f}")

    ll_bcast = [train_ll, val_ll]
    if ddp_ctx.enabled:
        torch.distributed.broadcast_object_list(ll_bcast, src=0)
    train_ll, val_ll = ll_bcast
    baseline_ll = {"train": train_ll, "val": val_ll}

    # 3. Create Dataset and DataLoader instances
    centerbias_for_loader = BaselineModel(
        stimuli=train_stim, fixations=train_fix,
        bandwidth=0.0217, eps=2e-13, caching=False
    )

    # ==================== START OF NEW LOGIC ====================
    model_requires_segmentation = stage_extra.get("requires_segmentation", False)
    
    # 1. Determine unique cache path based on mask usage
    train_cache_suffix, val_cache_suffix = "", ""
    train_mask_dir_rel, val_mask_dir_rel = None, None
    
    if model_requires_segmentation:
        train_mask_dir_rel = stage_extra.get('salicon_train_mask_dir')
        val_mask_dir_rel = stage_extra.get('salicon_val_mask_dir')

        # Create a unique suffix from the mask directory name itself
        train_cache_suffix = f"_masks_{Path(train_mask_dir_rel).name}" if train_mask_dir_rel else "_with_dummy_masks"
        val_cache_suffix = f"_masks_{Path(val_mask_dir_rel).name}" if val_mask_dir_rel else "_with_dummy_masks"

    lmdb_path_train = global_paths.get("lmdb_dir") / f"salicon_train{train_cache_suffix}" if global_paths.get("lmdb_dir") else None
    lmdb_path_val = global_paths.get("lmdb_dir") / f"salicon_val{val_cache_suffix}" if global_paths.get("lmdb_dir") else None

    # 2. Trigger DDP-safe cache creation on master process
    if ddp_ctx.is_master:
        if lmdb_path_train:
            train_mask_dir_abs = PROJECT_ROOT / train_mask_dir_rel if train_mask_dir_rel else None
            create_lmdb_cache_if_needed(
                stimuli=train_stim,
                centerbias_model=centerbias_for_loader,
                lmdb_path=lmdb_path_train,
                logger=logger,
                segmentation_mask_dir=train_mask_dir_abs
            )
        if lmdb_path_val:
            val_mask_dir_abs = PROJECT_ROOT / val_mask_dir_rel if val_mask_dir_rel else None
            create_lmdb_cache_if_needed(
                stimuli=val_stim,
                centerbias_model=centerbias_for_loader, # Note: using train centerbias for val cache is usually acceptable
                lmdb_path=lmdb_path_val,
                logger=logger,
                segmentation_mask_dir=val_mask_dir_abs
            )
    
    if ddp_ctx.enabled:
        ddp_ctx.barrier() # All processes wait for caches to be ready

    # 3. Instantiate Dataset classes
    DatasetClass = ImageDatasetWithSegmentation if model_requires_segmentation else ImageDataset
    
    train_ds_kwargs = {
        "transform": FixationMaskTransform(sparse=False),
        "average": "image",
        "lmdb_path": str(lmdb_path_train) if lmdb_path_train else None
    }
    val_ds_kwargs = {
        "transform": FixationMaskTransform(sparse=False),
        "average": "image",
        "lmdb_path": str(lmdb_path_val) if lmdb_path_val else None
    }

    if model_requires_segmentation:
        # Pass the original mask dir for the fallback logic inside ImageDatasetWithSegmentation
        train_ds_kwargs['segmentation_mask_dir'] = PROJECT_ROOT / train_mask_dir_rel if train_mask_dir_rel else None
        val_ds_kwargs['segmentation_mask_dir'] = PROJECT_ROOT / val_mask_dir_rel if val_mask_dir_rel else None
    
    train_dataset = DatasetClass(train_stim, train_fix, centerbias_for_loader, **train_ds_kwargs)
    val_dataset = DatasetClass(val_stim, val_fix, centerbias_for_loader, **val_ds_kwargs)
    # ===================== END OF NEW LOGIC =====================

    # 4. Create DataLoaders
    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True) if ddp_ctx.enabled else None
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.stage.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=cfg.num_workers > 0
    )

    val_sampler = DistributedSampler(val_dataset, shuffle=False) if ddp_ctx.enabled else None
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.stage.batch_size * 2,
        shuffle=False, sampler=val_sampler,
        num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=cfg.num_workers > 0
    )

    return train_loader, val_loader, baseline_ll