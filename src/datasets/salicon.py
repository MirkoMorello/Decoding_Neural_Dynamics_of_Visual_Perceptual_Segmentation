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

    # --- CHANGE 1: Define and use salicon_loc correctly ---
    # The root location for the raw SALICON dataset files.
    salicon_loc = global_paths["dataset_dir"] / 'SALICON'
    
    # 1. Download data (master only)
    if ddp_ctx.is_master:
        logger.info(f"Checking for SALICON data at: {salicon_loc}")
        # The location passed to get_SALICON should be the parent of the 'SALICON' directory
        pysaliency.get_SALICON_train(location=str(salicon_loc.parent))
        pysaliency.get_SALICON_val(location=str(salicon_loc.parent))
    ddp_ctx.barrier()

    # Load the data after ensuring it's present
    train_stim, train_fix = pysaliency.get_SALICON_train(location=str(salicon_loc.parent))
    val_stim, val_fix = pysaliency.get_SALICON_val(location=str(salicon_loc.parent))

    # 2. Calculate and broadcast baseline log-likelihood from a central cache
    train_ll, val_ll = None, None
    if ddp_ctx.is_master:
        # Define and use the central baseline cache directory ---
        baseline_cache_dir = global_paths["dataset_dir"] / "SALICON_baseline_cache"
        baseline_cache_dir.mkdir(parents=True, exist_ok=True)
        train_ll_cache = baseline_cache_dir / 'salicon_baseline_train_ll.pkl'
        val_ll_cache = baseline_cache_dir / 'salicon_baseline_val_ll.pkl'

        centerbias = BaselineModel(
            stimuli=train_stim, fixations=train_fix,
            bandwidth=0.0217, eps=2e-13, caching=False
        )

        try:
            with open(train_ll_cache, 'rb') as f:
                train_ll = cpickle.load(f)
        except (FileNotFoundError, EOFError):
            logger.info("Calculating SALICON train baseline LL...")
            train_ll = centerbias.information_gain(train_stim, train_fix, average='image')
            # Use atomic_save for safe writing in case of interruption
            with atomic_save(train_ll_cache, text_mode=False, overwrite_part=True) as f:
                cpickle.dump(train_ll, f)

        try:
            with open(val_ll_cache, 'rb') as f:
                val_ll = cpickle.load(f)
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

    lmdb_path_train = global_paths.get("lmdb_dir") / "salicon_train" if global_paths.get("lmdb_dir") else None
    lmdb_path_val = global_paths.get("lmdb_dir") / "salicon_val" if global_paths.get("lmdb_dir") else None
    
    # This barrier isn't strictly necessary here but doesn't hurt.
    # It ensures all processes wait for baselines before proceeding.
    ddp_ctx.barrier()

    model_requires_segmentation = stage_extra.get("requires_segmentation", False)
    DatasetClass = ImageDatasetWithSegmentation if model_requires_segmentation else ImageDataset

    ds_kwargs = {
        "transform": FixationMaskTransform(sparse=False),
        "average": "image",
    }
    
    if model_requires_segmentation:
        train_mask_rel_path = stage_extra.get('salicon_train_mask_dir')
        val_mask_rel_path = stage_extra.get('salicon_val_mask_dir')

        if not train_mask_rel_path or not val_mask_rel_path:
            raise ValueError("Config for SALICON with segmentation requires 'salicon_train_mask_dir' and 'salicon_val_mask_dir' in stage.extra")
        
        train_mask_abs_path = PROJECT_ROOT / train_mask_rel_path
        val_mask_abs_path = PROJECT_ROOT / val_mask_rel_path
        
        logger.info(f"Train masks path: {train_mask_abs_path}")
        logger.info(f"Validation masks path: {val_mask_abs_path}")

        train_dataset = DatasetClass(train_stim, train_fix, centerbias_for_loader,
                                     lmdb_path=str(lmdb_path_train) if lmdb_path_train else None,
                                     segmentation_mask_dir=train_mask_abs_path, **ds_kwargs)
        val_dataset = DatasetClass(val_stim, val_fix, centerbias_for_loader,
                                   lmdb_path=str(lmdb_path_val) if lmdb_path_val else None,
                                   segmentation_mask_dir=val_mask_abs_path, **ds_kwargs)
    else:
        train_dataset = DatasetClass(train_stim, train_fix, centerbias_for_loader,
                                     lmdb_path=str(lmdb_path_train) if lmdb_path_train else None,
                                     **ds_kwargs)
        val_dataset = DatasetClass(val_stim, val_fix, centerbias_for_loader,
                                   lmdb_path=str(lmdb_path_val) if lmdb_path_val else None,
                                   **ds_kwargs)

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
        val_dataset, batch_size=cfg.stage.batch_size,
        shuffle=False, sampler=val_sampler,
        num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=cfg.num_workers > 0
    )

    return train_loader, val_loader, baseline_ll