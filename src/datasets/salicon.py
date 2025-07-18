# src/datasets/salicon.py
import torch
import pysaliency
from torch.utils.data import DataLoader, DistributedSampler
import cloudpickle as cpickle
from pathlib import Path

from src.registry import register_data
# Import the Dataset classes from your data.py file
from src.data import ImageDataset, ImageDatasetWithSegmentation, FixationMaskTransform
# Import the BaselineModel class correctly
from pysaliency.baseline_utils import BaselineModel

@register_data("SALICON")
def prepare_salicon(cfg, ddp_ctx, logger):
    """
    Prepares the SALICON dataset, dataloaders, and baselines.
    This version relies on the ImageDataset class from src/data.py to handle
    automatic LMDB cache creation if a path is provided.
    """
    paths = cfg.paths
    
    # 1. Download data (master only)
    salicon_loc = paths["dataset_dir"] / 'SALICON'
    if ddp_ctx.is_master:
        logger.info(f"Checking for SALICON data at: {salicon_loc}")
        if not (salicon_loc / 'stimuli' / 'train').exists():
            pysaliency.get_SALICON_train(location=str(salicon_loc.parent))
        if not (salicon_loc / 'stimuli' / 'val').exists():
            pysaliency.get_SALICON_val(location=str(salicon_loc.parent))
    ddp_ctx.barrier() # All processes wait for download to complete
    
    train_stim, train_fix = pysaliency.get_SALICON_train(location=str(salicon_loc.parent))
    val_stim, val_fix = pysaliency.get_SALICON_val(location=str(salicon_loc.parent))

    # 2. Calculate and broadcast baseline log-likelihood
    train_ll, val_ll = None, None
    if ddp_ctx.is_master:
        train_ll_cache = paths["dataset_dir"] / 'salicon_baseline_train_ll.pkl'
        val_ll_cache = paths["dataset_dir"] / 'salicon_baseline_val_ll.pkl'
        
        # Instantiate BaselineModel with required arguments
        centerbias = BaselineModel(
            stimuli=train_stim, 
            fixations=train_fix, 
            bandwidth=0.0217, 
            eps=2e-13, 
            caching=False
        )
        
        try:
            with open(train_ll_cache, 'rb') as f:
                train_ll = cpickle.load(f)
        except FileNotFoundError:
            logger.info("Calculating SALICON train baseline LL...")
            train_ll = centerbias.information_gain(train_stim, train_fix, average='image')
            with open(train_ll_cache, 'wb') as f:
                cpickle.dump(train_ll, f)
        
        try:
            with open(val_ll_cache, 'rb') as f:
                val_ll = cpickle.load(f)
        except FileNotFoundError:
            logger.info("Calculating SALICON validation baseline LL...")
            val_ll = centerbias.information_gain(val_stim, val_fix, average='image')
            with open(val_ll_cache, 'wb') as f:
                cpickle.dump(val_ll, f)
                
        logger.info(f"SALICON Baselines - Train LL: {train_ll:.4f}, Val LL: {val_ll:.4f}")
            
    ll_bcast = [train_ll, val_ll]
    if ddp_ctx.enabled:
        torch.distributed.broadcast_object_list(ll_bcast, src=0)
    train_ll, val_ll = ll_bcast
    baseline_ll = {"train": train_ll, "val": val_ll}

    # 3. Create Dataset and DataLoader instances for all processes
    centerbias_for_loader = BaselineModel(
        stimuli=train_stim, 
        fixations=train_fix, 
        bandwidth=0.0217, 
        eps=2e-13, 
        caching=False
    )
    
    # Construct LMDB paths from the configuration
    lmdb_path_train = paths.get("lmdb_dir") / "salicon_train" if paths.get("lmdb_dir") else None
    lmdb_path_val = paths.get("lmdb_dir") / "salicon_val" if paths.get("lmdb_dir") else None
    
    # This barrier ensures all processes wait for the master to finish the baseline calculation
    # before they all try to initialize the Dataset, which might trigger cache creation.
    ddp_ctx.barrier()

    # Determine which Dataset class to use based on model requirements in the config
    model_requires_segmentation = cfg.stage.extra.get("requires_segmentation", False)
    DatasetClass = ImageDatasetWithSegmentation if model_requires_segmentation else ImageDataset
    
    # --- Training Dataset ---
    ds_kwargs_train = {
        "transform": FixationMaskTransform(sparse=False),
        "average": "image",
        "lmdb_path": str(lmdb_path_train) if lmdb_path_train else None
    }
    if model_requires_segmentation:
        ds_kwargs_train["segmentation_mask_dir"] = paths.get("salicon_train_mask_dir")
        # You can also pass the mask bank paths from cfg.paths here if needed by ImageDatasetWithSegmentation
        ds_kwargs_train["segmentation_mask_variable_payload_file"] = paths.get("train_mask_variable_payload_file")
        ds_kwargs_train["segmentation_mask_variable_header_file"] = paths.get("train_mask_variable_header_file")

    # The automatic caching will happen here, inside the constructor, on the master process
    # All other processes will wait due to DDP synchronization inside _export_dataset_to_lmdb
    train_dataset = DatasetClass(train_stim, train_fix, centerbias_for_loader, **ds_kwargs_train)
    
    # --- Validation Dataset ---
    ds_kwargs_val = {
        "transform": FixationMaskTransform(sparse=False),
        "average": "image",
        "lmdb_path": str(lmdb_path_val) if lmdb_path_val else None
    }
    if model_requires_segmentation:
        ds_kwargs_val["segmentation_mask_dir"] = paths.get("salicon_val_mask_dir")
        ds_kwargs_val["segmentation_mask_variable_payload_file"] = paths.get("val_mask_variable_payload_file")
        ds_kwargs_val["segmentation_mask_variable_header_file"] = paths.get("val_mask_variable_header_file")
    
    val_dataset = DatasetClass(val_stim, val_fix, centerbias_for_loader, **ds_kwargs_val)

    # --- DataLoaders ---
    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True) if ddp_ctx.enabled else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.stage.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0
    )
    
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if ddp_ctx.enabled else None
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.stage.batch_size * 2, # Use a larger batch size for validation
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0
    )

    return train_loader, val_loader, baseline_ll