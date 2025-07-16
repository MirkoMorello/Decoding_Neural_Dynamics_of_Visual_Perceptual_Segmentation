#!/usr/bin/env python
"""
Multi-GPU-ready training script for a hybrid DenseNet+DINO model using SPADE.
This version uses a self-contained model class and robust DDP setup to ensure
stability and prevent deadlocks, inspired by a proven working implementation.
"""
import os
import sys

# Add project root to sys.path for local module imports
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import yaml
import argparse
import logging
from pathlib import Path
import pickle  # <<< FIX: Import pickle for its exception type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import numpy as np
import math

import pysaliency
import pysaliency.external_datasets.mit
from pysaliency.dataset_config import train_split, validation_split
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from pysaliency.baseline_utils import BaselineModel, CrossvalidatedBaselineModel
import cloudpickle as cpickle

torch.set_float32_matmul_precision("medium") # 3090s support this, but not TITAN V

try:
    from torch_scatter import scatter_mean
except ImportError:
    print("torch_scatter not found. Please install it for efficient semantic map creation.")
    sys.exit(1)

try:
    from src.data import (
        ImageDatasetWithSegmentation, FixationMaskTransform,
        convert_stimuli, convert_fixation_trains
    )
    from DeepGaze.deepgaze_pytorch.features.densenet import RGBDenseNet201
    from src.dinov2_backbone import DinoV2Backbone
    from src.modules import FeatureExtractor, Finalizer, build_fixation_selection_network
    from src.layers import Bias
    from src.training import _train, restore_from_checkpoint
except ImportError as e:
    print(f"PYTHON IMPORT ERROR: {e}\n(sys.path: {sys.path})")
    sys.exit(1)

_logger = logging.getLogger("train_hybrid_gaze_spade")


import datetime

def init_distributed() -> tuple[torch.device, int, int, bool, bool]:
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_distributed = world_size > 1
    if is_distributed:
        # Set a generous timeout to accommodate the long baseline calculation on rank 0
        timeout = datetime.timedelta(hours=3) # 3 hours
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", timeout=timeout) # <--- THIS LINE IS MODIFIED
        
        device = torch.device(f"cuda:{local_rank}")
        is_master = rank == 0
    else:
        rank = 0; world_size = 1; is_master = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, rank, world_size, is_master, is_distributed


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

class SPADELayerNormDynamic(nn.Module):
    def __init__(self, norm_features, semantic_feature_channels, hidden_mlp_channels=128, eps=1e-12, kernel_size=3):
        super().__init__()
        self.norm_features = norm_features
        self.eps = eps
        self.semantic_feature_channels = semantic_feature_channels
        padding = kernel_size // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(self.semantic_feature_channels, hidden_mlp_channels, kernel_size=kernel_size, padding=padding, bias=True),
            nn.ReLU(inplace=True)
        )
        self.mlp_gamma = nn.Conv2d(hidden_mlp_channels, norm_features, kernel_size=kernel_size, padding=padding, bias=True)
        self.mlp_beta = nn.Conv2d(hidden_mlp_channels, norm_features, kernel_size=kernel_size, padding=padding, bias=True)

    def forward(self, x, painted_semantic_map):
        normalized_shape = (self.norm_features, x.size(2), x.size(3))
        normalized_x = F.layer_norm(x, normalized_shape, weight=None, bias=None, eps=self.eps)
        semantic_map_resized = F.interpolate(painted_semantic_map, size=x.size()[2:], mode='bilinear', align_corners=False)
        shared_features = self.mlp_shared(semantic_map_resized)
        gamma_map = self.mlp_gamma(shared_features)
        beta_map = self.mlp_beta(shared_features)
        return normalized_x * (1 + gamma_map) + beta_map

class SaliencyNetworkSPADEDynamic(nn.Module):
    def __init__(self, input_channels_main_path, semantic_feature_channels_for_spade):
        super().__init__()
        self.spade_ln0 = SPADELayerNormDynamic(input_channels_main_path, semantic_feature_channels_for_spade)
        self.conv0 = nn.Conv2d(input_channels_main_path, 8, (1, 1), bias=False)
        self.bias0 = Bias(8); self.softplus0 = nn.Softplus()
        self.spade_ln1 = SPADELayerNormDynamic(8, semantic_feature_channels_for_spade)
        self.conv1 = nn.Conv2d(8, 16, (1, 1), bias=False)
        self.bias1 = Bias(16); self.softplus1 = nn.Softplus()
        self.spade_ln2 = SPADELayerNormDynamic(16, semantic_feature_channels_for_spade)
        self.conv2 = nn.Conv2d(16, 1, (1, 1), bias=False)
        self.bias2 = Bias(1); self.softplus2 = nn.Softplus()

    def forward(self, x_main_path, painted_semantic_map):
        h = x_main_path
        h = self.spade_ln0(h, painted_semantic_map)
        h = self.conv0(h); h = self.bias0(h); h = self.softplus0(h)
        h = self.spade_ln1(h, painted_semantic_map)
        h = self.conv1(h); h = self.bias1(h); h = self.softplus1(h)
        h = self.spade_ln2(h, painted_semantic_map)
        h = self.conv2(h); h = self.bias2(h); h = self.softplus2(h)
        return h

class HybridSpadeSaliencyModel(nn.Module):
    def __init__(self, features_module, dino_features_module, saliency_network, fixation_selection_network,
                 finalizer, num_total_segments, downsample=1, readout_factor=4):
        super().__init__()
        self.features = features_module
        self.dino_features = dino_features_module
        self.saliency_network = saliency_network
        self.fixation_selection_network = fixation_selection_network
        self.finalizer = finalizer
        self.num_total_segments = num_total_segments
        self.downsample = downsample
        self.readout_factor = readout_factor

        for p in self.features.parameters(): p.requires_grad = False
        for p in self.dino_features.parameters(): p.requires_grad = False
        self.features.eval()
        self.dino_features.eval()

    def _create_painted_semantic_map_vectorized(self, F_dino_patches, raw_sam_pixel_segmap):
        B, C_dino, H_p, W_p = F_dino_patches.shape
        _, H_img, W_img = raw_sam_pixel_segmap.shape
        device = F_dino_patches.device
        segmap_at_feat_res = F.interpolate(raw_sam_pixel_segmap.unsqueeze(1).float(), size=(H_p, W_p), mode='nearest').long()
        flat_features = F_dino_patches.permute(0, 2, 3, 1).reshape(-1, C_dino)
        flat_segmap_at_feat_res = segmap_at_feat_res.view(-1)
        batch_idx_tensor = torch.arange(B, device=device, dtype=torch.long).view(B, 1).expand(-1, H_p * W_p).reshape(-1)
        global_segment_ids = batch_idx_tensor * self.num_total_segments + torch.clamp(flat_segmap_at_feat_res, 0, self.num_total_segments - 1)
        total_segments_in_batch = B * self.num_total_segments
        segment_avg_features = scatter_mean(src=flat_features, index=global_segment_ids, dim=0, dim_size=total_segments_in_batch)
        segment_avg_features = torch.nan_to_num(segment_avg_features, nan=0.0)
        flat_pixel_segmap = raw_sam_pixel_segmap.view(B, -1)
        batch_idx_pixel_tensor = torch.arange(B, device=device, dtype=torch.long).view(B, 1).expand(-1, H_img * W_img)
        global_pixel_ids = batch_idx_pixel_tensor.reshape(-1) * self.num_total_segments + torch.clamp(flat_pixel_segmap.view(-1), 0, self.num_total_segments - 1)
        painted_flat = segment_avg_features[global_pixel_ids]
        return painted_flat.view(B, H_img, W_img, C_dino).permute(0, 3, 1, 2)
    
    def forward(self, image, centerbias, segmentation_mask, **kwargs):
        if segmentation_mask is None: raise ValueError(f"{self.__class__.__name__} requires 'segmentation_mask'.")
        img_for_features = image
        if self.downsample != 1:
            img_for_features = F.interpolate(image, scale_factor=1.0 / self.downsample, mode='bilinear', align_corners=False)
        with torch.no_grad():
            dino_feature_maps = self.dino_features(img_for_features)
            semantic_dino_patches = dino_feature_maps[0]
            extracted_feature_maps = self.features(img_for_features)
        painted_map = self._create_painted_semantic_map_vectorized(semantic_dino_patches, segmentation_mask)
        readout_h = math.ceil(image.shape[2] / self.downsample / self.readout_factor)
        readout_w = math.ceil(image.shape[3] / self.downsample / self.readout_factor)
        processed_features_list = [F.interpolate(f, size=(readout_h, readout_w), mode='bilinear', align_corners=False) for f in extracted_feature_maps]
        concatenated_features = torch.cat(processed_features_list, dim=1)
        saliency_output = self.saliency_network(concatenated_features, painted_map)
        final_readout = self.fixation_selection_network((saliency_output, None))
        log_density = self.finalizer(final_readout, centerbias)
        return log_density


def salicon_pretrain(args, device, is_master, is_distributed, model):
    if is_master: _logger.info("--- Preparing SALICON Pretraining ---")
    
    salicon_loc = args.dataset_dir / 'SALICON'
    if is_master:
        if not (salicon_loc/'stimuli'/'train').exists(): pysaliency.get_SALICON_train(location=str(salicon_loc.parent))
        if not (salicon_loc/'stimuli'/'val').exists(): pysaliency.get_SALICON_val(location=str(salicon_loc.parent))
    if is_distributed: dist.barrier() # Barrier to ensure data is downloaded
    
    train_stim, train_fix = pysaliency.get_SALICON_train(location=str(salicon_loc.parent))
    val_stim, val_fix = pysaliency.get_SALICON_val(location=str(salicon_loc.parent))
    
    # These variables will be populated on rank 0, and remain None on other ranks
    train_ll, val_ll = None, None

    if is_master:
        centerbias = BaselineModel(train_stim, train_fix, bandwidth=0.0217, eps=2e-13, caching=False)
        train_ll_cache = args.train_dir / 'salicon_baseline_train_ll.pkl'
        val_ll_cache = args.train_dir / 'salicon_baseline_val_ll.pkl'
        
        _logger.info("Master processing baseline LLs...")
        try:
            train_ll = cpickle.load(open(train_ll_cache, 'rb'))
            _logger.info(f"Loaded train LL from cache: {train_ll_cache}")
        except Exception:
            _logger.info(f"Cache not found or invalid for train LL. Computing...")
            train_ll = centerbias.information_gain(train_stim, train_fix, verbose=True, average='image')
            with open(train_ll_cache, 'wb') as f: cpickle.dump(train_ll, f)

        try:
            val_ll = cpickle.load(open(val_ll_cache, 'rb'))
            _logger.info(f"Loaded val LL from cache: {val_ll_cache}")
        except Exception:
            _logger.info(f"Cache not found or invalid for val LL. Computing...")
            val_ll = centerbias.information_gain(val_stim, val_fix, verbose=True, average='image')
            with open(val_ll_cache, 'wb') as f: cpickle.dump(val_ll, f)
        _logger.info(f"Final LLs on master: Train={train_ll:.4f}, Val={val_ll:.4f}")

    if is_distributed:
        # Master sends the data it just computed/loaded.
        # Workers receive the data. They wait here until the master is done.
        ll_bcast = [train_ll, val_ll] 
        dist.broadcast_object_list(ll_bcast, src=0)
        train_ll, val_ll = ll_bcast
    
    # After this point, all processes have the correct values for train_ll and val_ll.
    if train_ll is None or val_ll is None:
        _logger.critical(f"Rank {dist.get_rank() if is_distributed else 0} has invalid LL values. Exiting.")
        sys.exit(1)

    # Re-create the centerbias object for the dataloader on all processes
    centerbias = BaselineModel(train_stim, train_fix, bandwidth=0.0217, eps=2e-13, caching=False)
    
    # ... The rest of the function for creating optimizer, datasets, etc. ...
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones)
    
    lmdb_path_train = args.train_dir / f'salicon_train_imgs_{args.dino_model_name}' if args.use_lmdb_images else None
    lmdb_path_val = args.train_dir / f'salicon_val_imgs_{args.dino_model_name}' if args.use_lmdb_images else None

    train_dataset = ImageDatasetWithSegmentation(
        train_stim, train_fix, centerbias,
        lmdb_path=lmdb_path_train, # <--- PASS THE ARGUMENT
        segmentation_mask_dir=args.salicon_train_mask_dir,
        transform=FixationMaskTransform(sparse=False),
        segmentation_mask_format='png',
        average="image"
    )
    val_dataset = ImageDatasetWithSegmentation(
        val_stim, val_fix, centerbias,
        lmdb_path=lmdb_path_val, # <--- PASS THE ARGUMENT
        segmentation_mask_dir=args.salicon_val_mask_dir,
        transform=FixationMaskTransform(sparse=False),
        segmentation_mask_format='png',
        average="image"
    )
    
    train_sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=True, drop_last=True) if is_distributed else None
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_sampler = torch.utils.data.DistributedSampler(val_dataset, shuffle=False, drop_last=False) if is_distributed else None
    validation_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        sampler=val_sampler, num_workers=args.num_workers, pin_memory=True
    )
        
    experiment_name = f"{args.stage}_{args.densenet_model_name}_dino_{args.dino_model_name}_lr{args.lr}"
    output_dir = args.train_dir / experiment_name
    
    _train(
    this_directory=str(output_dir), 
    model=model, 
    train_loader=train_loader, 
    train_baseline_log_likelihood=train_ll, 
    val_loader=validation_loader, 
    val_baseline_log_likelihood=val_ll, 
    optimizer=optimizer, 
    lr_scheduler=lr_scheduler, 
    gradient_accumulation_steps=args.gradient_accumulation_steps, 
    minimum_learning_rate=args.min_lr, 
    validation_metrics=['LL', 'IG', 'NSS', 'AUC_CPU'], 
    validation_epochs=args.validation_epochs, 
    startwith=args.resume_checkpoint, 
    device=device, 
    is_distributed=is_distributed, 
    is_master=is_master, 
    logger=_logger
)


def mit_finetune(args, device, is_master, is_distributed, model_cpu):
    # Apply the same logic to mit_finetune
    fold = args.fold
    if fold is None or not (0 <= fold < 10): _logger.critical("--fold (0-9) required for MIT stages."); sys.exit(1)
    if is_master: _logger.info(f"--- Preparing MIT Stage: {args.stage} (Fold {fold}) ---")
    
    mit_converted_data_path = args.train_dir / "MIT1003_converted_hybrid_gaze"
    mit_stimuli_cache_file = mit_converted_data_path / "stimuli.pkl"
    mit_stimuli_all, mit_fixations_all = None, None

    if mit_stimuli_cache_file.exists() and mit_stimuli_cache_file.stat().st_size > 0:
        if is_master: _logger.info(f"Loading pre-converted MIT data from cache: {mit_converted_data_path}")
        with open(mit_stimuli_cache_file, "rb") as f: mit_stimuli_all = cpickle.load(f)
        mit_stimuli_orig, mit_fixations_orig = pysaliency.external_datasets.mit.get_mit1003_with_initial_fixation(location=str(args.dataset_dir), replace_initial_invalid_fixations=True)
        mit_fixations_all = convert_fixation_trains(mit_stimuli_orig, mit_fixations_orig, is_master, _logger)
    else:
        if is_master: _logger.info("No valid MIT cache found. Starting data conversion...")
        mit_stimuli_orig, mit_fixations_orig = pysaliency.external_datasets.mit.get_mit1003_with_initial_fixation(location=str(args.dataset_dir), replace_initial_invalid_fixations=True)
        mit_stimuli_all = convert_stimuli(mit_stimuli_orig, mit_converted_data_path, is_master, is_distributed, device, _logger)
        mit_fixations_all = convert_fixation_trains(mit_stimuli_orig, mit_fixations_orig, is_master, _logger)
    
    if is_distributed: dist.barrier()
    if mit_stimuli_all is None or mit_fixations_all is None: _logger.critical("Failed to load or convert MIT data."); sys.exit(1)

    train_stim, train_fix = train_split(mit_stimuli_all, mit_fixations_all, crossval_folds=10, fold_no=fold)
    val_stim, val_fix = validation_split(mit_stimuli_all, mit_fixations_all, crossval_folds=10, fold_no=fold)
    centerbias = CrossvalidatedBaselineModel(mit_stimuli_all, mit_fixations_all, bandwidth=10**-1.6667673342543432, eps=10**-14.884189168516073, caching=False)

    train_ll_cache = args.train_dir / f'mit_baseline_train_ll_fold{fold}.pkl'
    val_ll_cache = args.train_dir / f'mit_baseline_val_ll_fold{fold}.pkl'
    train_ll, val_ll = None, None

    if is_master:
        _logger.info(f"Master processing baseline LLs for fold {fold}...")
        try:
            train_ll = cpickle.load(open(train_ll_cache, 'rb'))
            _logger.info(f"Loaded train LL from cache: {train_ll_cache}")
        except (FileNotFoundError, EOFError, pickle.UnpicklingError): # <<< FIX: Corrected exception type
            _logger.info(f"Cache not found or invalid for train LL (fold {fold}). Computing...")
            train_ll = centerbias.information_gain(train_stim, train_fix, verbose=True, average='image')
            with open(train_ll_cache, 'wb') as f: cpickle.dump(train_ll, f)
        
        try:
            val_ll = cpickle.load(open(val_ll_cache, 'rb'))
            _logger.info(f"Loaded val LL from cache: {val_ll_cache}")
        except (FileNotFoundError, EOFError, pickle.UnpicklingError): # <<< FIX: Corrected exception type
            _logger.info(f"Cache not found or invalid for val LL (fold {fold}). Computing...")
            val_ll = centerbias.information_gain(val_stim, val_fix, verbose=True, average='image')
            with open(val_ll_cache, 'wb') as f: cpickle.dump(val_ll, f)
        
        _logger.info(f"Final LLs on master for fold {fold}: Train={train_ll:.4f}, Val={val_ll:.4f}")
    
    if is_distributed:
        ll_bcast = [train_ll, val_ll]
        dist.broadcast_object_list(ll_bcast, src=0)
        train_ll, val_ll = ll_bcast

    if train_ll is None or val_ll is None: _logger.critical("MIT Baseline LLs invalid."); sys.exit(1)

    if args.salicon_checkpoint_path and args.salicon_checkpoint_path.exists():
        if is_master: _logger.info(f"Loading SALICON weights from {args.salicon_checkpoint_path}")
        restore_from_checkpoint(model_cpu, None, None, None, str(args.salicon_checkpoint_path), 'cpu', False, _logger)
    else:
        _logger.warning("No SALICON checkpoint provided for MIT fine-tuning. Head will be randomly initialized.")

    model = model_cpu.to(device)
    if is_distributed: model = DDP(model, device_ids=[device.index], find_unused_parameters=True)
    if args.use_torch_compile: model = torch.compile(model)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_mit_spatial)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones_mit_spatial)

    lmdb_path_train = args.train_dir / f'MIT1003_train_imgs_fold{fold}_{args.dino_model_name}' if args.use_lmdb_images else None
    lmdb_path_val = args.train_dir / f'MIT1003_val_imgs_fold{fold}_{args.dino_model_name}' if args.use_lmdb_images else None
    
    train_dataset = ImageDatasetWithSegmentation(
        train_stim, train_fix, centerbias,
        lmdb_path=lmdb_path_train, # <--- PASS THE ARGUMENT
        segmentation_mask_dir=args.mit_all_mask_dir,
        transform=FixationMaskTransform(sparse=False),
        segmentation_mask_format='png',
        average="image"
    )
    val_dataset = ImageDatasetWithSegmentation(
        val_stim, val_fix, centerbias,
        lmdb_path=lmdb_path_val, # <--- PASS THE ARGUMENT
        segmentation_mask_dir=args.mit_all_mask_dir,
        transform=FixationMaskTransform(sparse=False),
        segmentation_mask_format='png',
        average="image"
    )
    
    train_sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=True, drop_last=True) if is_distributed else None
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True, multiprocessing_context="spawn",
    )
    val_sampler = torch.utils.data.DistributedSampler(val_dataset, shuffle=False, drop_last=False) if is_distributed else None
    validation_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        sampler=val_sampler, num_workers=args.num_workers, pin_memory=True, multiprocessing_context="spawn",
    )
    experiment_name = f"{args.stage}_fold{fold}_{args.densenet_model_name}_dino_{args.dino_model_name}_k{args.num_total_segments}_lr{args.lr_mit_spatial}"
    output_dir = args.train_dir / experiment_name
    
    _train(
    this_directory=str(output_dir), 
    model=model, 
    train_loader=train_loader, 
    train_baseline_log_likelihood=train_ll, 
    val_loader=validation_loader, 
    val_baseline_log_likelihood=val_ll, 
    optimizer=optimizer, 
    lr_scheduler=lr_scheduler, 
    gradient_accumulation_steps=args.gradient_accumulation_steps, 
    minimum_learning_rate=args.min_lr, 
    validation_metrics=['LL', 'IG', 'NSS', 'AUC_CPU'], 
    validation_epochs=args.validation_epochs, 
    startwith=args.resume_checkpoint, 
    device=device, 
    is_distributed=is_distributed, 
    is_master=is_master, 
    logger=_logger
)


def main(args):
    device, rank, world, is_master, is_distributed = init_distributed()
    
    log_level = logging.INFO if is_master else logging.WARNING
    if args.log_level: log_level = getattr(logging, str(args.log_level).upper(), log_level)
    logging.basicConfig(level=log_level, format=f"%(asctime)s Rank{rank} %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", force=True)
    _logger.setLevel(log_level)

    if is_master:
        _logger.info("================== Effective Configuration ==================")
        cleaned_args = {k: v for k, v in vars(args).items() if not (k.startswith('dinov2_') or k == 'config_file')}
        for name, value in sorted(cleaned_args.items()): _logger.info(f"  {name}: {value}")
        _logger.info(f"  DDP Info: Rank {rank}/{world}, Master: {is_master}, Distributed: {is_distributed}, Device: {device}")
        _logger.info("===========================================================")

    for p in [args.dataset_dir, args.train_dir, args.lmdb_dir]:
        if is_master and p: p.mkdir(parents=True, exist_ok=True)
    if is_distributed: dist.barrier()

    if is_master: _logger.info(f"Initializing {args.densenet_model_name} backbone...")
    densenet_base = RGBDenseNet201()
    densenet_feature_nodes = ['1.features.denseblock4.denselayer32.norm1', '1.features.denseblock4.denselayer32.conv1', '1.features.denseblock4.denselayer31.conv2']
    features_module = FeatureExtractor(densenet_base, densenet_feature_nodes)
    
    if is_master: _logger.info(f"Initializing {args.dino_model_name} backbone...")
    dino_semantic_backbone = DinoV2Backbone(
        layers=[args.dino_semantic_layer_idx], model_name=args.dino_model_name,
        patch_size=args.dino_patch_size, freeze=True
    )
    
    features_module.to(device)
    dino_semantic_backbone.to(device)
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 256, 256).to(device)
        main_path_channels = sum(f.shape[1] for f in features_module(dummy_input))
        semantic_path_channels = dino_semantic_backbone(dummy_input)[0].shape[1]
    
    if is_master:
        _logger.info(f"Main path (DenseNet) concatenated channels: {main_path_channels}")
        _logger.info(f"Dynamic semantic path (DINO) channels: {semantic_path_channels}")

    saliency_net = SaliencyNetworkSPADEDynamic(main_path_channels, semantic_path_channels)
    fixsel_net = build_fixation_selection_network(scanpath_features=0)
    finalizer = Finalizer(sigma=args.finalizer_initial_sigma, learn_sigma=args.finalizer_learn_sigma, saliency_map_factor=4)
    
    model_cpu = HybridSpadeSaliencyModel(
        features_module=features_module.cpu(),
        dino_features_module=dino_semantic_backbone.cpu(),
        saliency_network=saliency_net,
        fixation_selection_network=fixsel_net,
        finalizer=finalizer,
        num_total_segments=args.num_total_segments
    )

    stage = args.stage
    if 'salicon_pretrain' in stage or 'salicon_pretraining' in stage:
        model = model_cpu.to(device)
        if is_distributed:
            model = DDP(model, device_ids=[device.index], find_unused_parameters=True)
        if args.use_torch_compile: 
            if is_master: _logger.info("Enabling torch.compile.")
            model = torch.compile(model)
        salicon_pretrain(args, device, is_master, is_distributed, model)
    elif 'mit_spatial' in stage:
        mit_finetune(args, device, is_master, is_distributed, model_cpu)
    else:
        _logger.critical(f"Unknown or unsupported stage: {stage}"); sys.exit(1)
        
    if is_master: _logger.info("Main training function finished. Cleaning up...")
    
    # 1. Explicitly delete large objects that might hold GPU tensors
    del features_module, dino_semantic_backbone, saliency_net, fixsel_net, finalizer, model_cpu
    
    # 2. Synchronize all processes before cleanup
    if is_distributed:
        dist.barrier()
        
    # 3. Empty the CUDA cache on all devices
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 4. Now, perform the final cleanup
    cleanup_distributed()
    if is_master: _logger.info("Cleanup complete. Script should now exit.")


if __name__ == "__main__":
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument('--config_file', type=str, default=None)
    _cfg_ns, _rem_args = _pre.parse_known_args()

    parser = argparse.ArgumentParser(parents=[_pre], description="Train Hybrid DenseNet+DINO SPADE model.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--stage', choices=['salicon_pretrain_hybrid', 'mit_spatial_hybrid'], help='Training stage to execute.')
    parser.add_argument('--log_level', type=str, default='INFO')
    parser.add_argument('--densenet_model_name', default='densenet201', choices=['densenet201'])
    parser.add_argument('--dino_model_name', default='dinov2_vitb14', choices=['dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'])
    parser.add_argument('--dino_patch_size', type=int, default=14)
    parser.add_argument('--dino_semantic_layer_idx', type=int, default=-1)
    parser.add_argument('--num_total_segments', type=int, default=16)
    parser.add_argument('--segmentation_mask_format', default='png', choices=['png', 'npy'])
    parser.add_argument('--salicon_train_mask_dir', type=str)
    parser.add_argument('--salicon_val_mask_dir', type=str)
    parser.add_argument('--mit_all_mask_dir', type=str)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lr_milestones', type=int, nargs='+', default=[20, 40, 55])
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--validation_epochs', type=int, default=1)
    parser.add_argument('--use_torch_compile', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--finalizer_initial_sigma', type=float, default=8.0)
    parser.add_argument('--finalizer_learn_sigma', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--num_workers', type=str, default='auto')
    parser.add_argument('--train_dir', type=str, default='./experiments_hybrid_gaze_spade')
    parser.add_argument('--dataset_dir', type=str, default='./data/pysaliency_datasets')
    parser.add_argument('--lmdb_dir', type=str, default='./data/lmdb_caches_hybrid')
    parser.add_argument('--fold', type=int)
    parser.add_argument('--resume_checkpoint', type=str)
    parser.add_argument('--salicon_checkpoint_path', type=str)
    parser.add_argument('--lr_mit_spatial', type=float, default=1e-4)
    parser.add_argument('--lr_milestones_mit_spatial', type=int, nargs='+', default=[10, 20])
    parser.add_argument('--use_lmdb_images', action=argparse.BooleanOptionalAction, default=True)

    if _cfg_ns.config_file:
        try:
            with open(_cfg_ns.config_file, 'r') as f: yaml_cfg = yaml.safe_load(f) or {}
            parser.set_defaults(**yaml_cfg)
        except Exception as e:
            logging.basicConfig()
            logging.getLogger().error(f"Could not read/parse YAML: {e}")

    final_args_ns = parser.parse_args(_rem_args)
    
    project_root = Path(__file__).resolve().parent.parent
    ws_env = int(os.environ.get("WORLD_SIZE", 1))
    if isinstance(final_args_ns.num_workers, str) and final_args_ns.num_workers.lower() == 'auto':
        try: cpu_c = len(os.sched_getaffinity(0))
        except AttributeError: cpu_c = os.cpu_count() or 1
        final_args_ns.num_workers = min(8, cpu_c // ws_env if ws_env > 0 else cpu_c)
    else: final_args_ns.num_workers = int(final_args_ns.num_workers)

    def resolve_path_arg(arg_value):
        if arg_value is None: return None
        path = Path(arg_value)
        return (project_root / path).resolve() if not path.is_absolute() else path.resolve()
    for arg_name, arg_value in vars(final_args_ns).items():
        if 'dir' in arg_name or 'path' in arg_name:
            setattr(final_args_ns, arg_name, resolve_path_arg(arg_value))
    
    try:
        main(final_args_ns)
    except KeyboardInterrupt: _logger.warning("Training interrupted by user (Ctrl+C)."); cleanup_distributed(); sys.exit(130)
    except Exception: _logger.critical("Unhandled exception during main execution:", exc_info=True); cleanup_distributed(); sys.exit(1)