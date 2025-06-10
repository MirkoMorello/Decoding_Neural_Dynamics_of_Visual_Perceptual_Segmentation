#!/usr/bin/env python
"""
Multi-GPU-ready training script for DeepGaze III with DenseNet + SPADE.
Includes MIT1003 data conversion for efficient, standardized batching.
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
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from DeepGaze.deepgaze_pytorch.features.densenet import RGBDenseNet201
import numpy as np
import math

import pysaliency
import pysaliency.external_datasets.mit
from pysaliency.dataset_config import train_split, validation_split
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from pysaliency.baseline_utils import BaselineModel, CrossvalidatedBaselineModel
import cloudpickle as cpickle
from boltons.fileutils import atomic_save

# --- Local Project Imports ---
try:
    from src.data import (
        ImageDatasetWithSegmentation,
        FixationMaskTransform,
        convert_stimuli, convert_fixation_trains
    )
    from src.modules import FeatureExtractor, Finalizer, build_fixation_selection_network
    from src.layers import Bias, LayerNorm, SelfAttention
    from src.training import _train, restore_from_checkpoint
except ImportError as e:
    print(f"PYTHON IMPORT ERROR: {e}\n(sys.path: {sys.path})")
    sys.exit(1)

_logger = logging.getLogger("train_densenet_spade_ddp")

def init_distributed() -> tuple[torch.device, int, int, bool, bool]:
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_distributed = world_size > 1
    if is_distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device(f"cuda:{local_rank}")
        is_master = rank == 0
    else:
        rank = 0; world_size = 1; is_master = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, rank, world_size, is_master, is_distributed

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

class SPADELayerNorm(nn.Module):
    def __init__(self, norm_features, segmap_input_channels, hidden_mlp_channels=128, eps=1e-12, kernel_size=3):
        super().__init__()
        self.norm_features = norm_features
        self.segmap_input_channels = segmap_input_channels
        padding = kernel_size // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(self.segmap_input_channels, hidden_mlp_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True)
        )
        self.mlp_gamma = nn.Conv2d(hidden_mlp_channels, norm_features, kernel_size=kernel_size, padding=padding)
        self.mlp_beta = nn.Conv2d(hidden_mlp_channels, norm_features, kernel_size=kernel_size, padding=padding)

    def forward(self, x, segmap_embedded):
        normalized_x = F.layer_norm(x, (self.norm_features, x.size(2), x.size(3)), eps=self.eps)
        segmap_resized = F.interpolate(segmap_embedded, size=x.size()[2:], mode='nearest')
        shared_features = self.mlp_shared(segmap_resized)
        gamma = self.mlp_gamma(shared_features)
        beta = self.mlp_beta(shared_features)
        return normalized_x * (1 + gamma) + beta

class SaliencyNetworkSPADE(nn.Module):
    def __init__(self, input_channels, num_total_segments=17, seg_embedding_dim=64, add_sa_head=False):
        super().__init__()
        self.seg_embedder = nn.Embedding(num_total_segments, seg_embedding_dim)
        self.add_sa_head = add_sa_head
        if self.add_sa_head:
            self.sa_head_module = SelfAttention(input_channels, key_channels=input_channels // 8)
            self.sa_ln_out = LayerNorm(input_channels)
            self.sa_softplus_out = nn.Softplus()
        self.spade_ln0 = SPADELayerNorm(input_channels, seg_embedding_dim)
        self.conv0 = nn.Conv2d(input_channels, 8, 1, bias=False)
        self.bias0 = Bias(8); self.softplus0 = nn.Softplus()
        self.spade_ln1 = SPADELayerNorm(8, seg_embedding_dim)
        self.conv1 = nn.Conv2d(8, 16, 1, bias=False)
        self.bias1 = Bias(16); self.softplus1 = nn.Softplus()
        self.spade_ln2 = SPADELayerNorm(16, seg_embedding_dim)
        self.conv2 = nn.Conv2d(16, 1, 1, bias=False)
        self.bias2 = Bias(1); self.softplus2 = nn.Softplus()

    def forward(self, x, raw_segmap_long):
        B, H, W = raw_segmap_long.shape
        clamped = torch.clamp(raw_segmap_long.view(B, -1), 0, self.seg_embedder.num_embeddings - 1)
        embedded = self.seg_embedder(clamped).view(B, H, W, -1).permute(0, 3, 1, 2)
        h = x
        if self.add_sa_head:
            h, _ = self.sa_head_module(h); h = self.sa_ln_out(h); h = self.sa_softplus_out(h)
        h = self.spade_ln0(h, embedded); h = self.conv0(h); h = self.bias0(h); h = self.softplus0(h)
        h = self.spade_ln1(h, embedded); h = self.conv1(h); h = self.bias1(h); h = self.softplus1(h)
        h = self.spade_ln2(h, embedded); h = self.conv2(h); h = self.bias2(h); h = self.softplus2(h)
        return h

class DeepGazeIIISpade(nn.Module):
    def __init__(self, features, saliency_network, scanpath_network, fixation_selection_network,
                 finalizer_learn_sigma, initial_sigma=8.0, downsample=1, readout_factor=4, saliency_map_factor=4):
        super().__init__()
        self.features = features
        self.saliency_network = saliency_network
        self.scanpath_network = scanpath_network
        self.fixation_selection_network = fixation_selection_network
        self.downsample = downsample
        self.readout_factor = readout_factor
        self.finalizer = Finalizer(sigma=initial_sigma, learn_sigma=finalizer_learn_sigma, saliency_map_factor=saliency_map_factor)

    def forward(self, image, centerbias, segmentation_mask, **kwargs):
        if segmentation_mask is None: raise ValueError("SaliencyNetworkSPADE requires a segmentation_mask.")
        img_for_features = F.interpolate(image, scale_factor=1.0/self.downsample, mode='bilinear', align_corners=False) if self.downsample != 1 else image
        extracted_maps = self.features(img_for_features)
        readout_h = math.ceil(image.shape[2] / self.downsample / self.readout_factor)
        readout_w = math.ceil(image.shape[3] / self.downsample / self.readout_factor)
        processed_features = [F.interpolate(f, size=(readout_h, readout_w), mode='bilinear', align_corners=False) for f in extracted_maps]
        concatenated = torch.cat(processed_features, dim=1)
        saliency_out = self.saliency_network(concatenated, segmentation_mask)
        final_readout = self.fixation_selection_network((saliency_out, None))
        return self.finalizer(final_readout, centerbias)

    def train(self, mode=True):
        if hasattr(self.features, 'eval'): self.features.eval()
        self.saliency_network.train(mode); self.fixation_selection_network.train(mode); self.finalizer.train(mode)
        super().train(mode)


def main(args: argparse.Namespace):
    device, rank, world, is_master, is_distributed = init_distributed()
    log_level = logging.INFO if is_master else logging.WARNING
    if args.log_level: log_level = getattr(logging, str(args.log_level).upper(), log_level)
    logging.basicConfig(level=log_level, format=f"%(asctime)s Rank{rank} %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", force=True)
    _logger.setLevel(log_level)

    if is_master:
        _logger.info("================== Effective Configuration ==================")
        for name, value in sorted(vars(args).items()): _logger.info(f"  {name}: {value}")
        _logger.info(f"  DDP Info: Rank {rank}/{world}, Master: {is_master}, Distributed: {is_distributed}, Device: {device}")
        _logger.info("===========================================================")

    for p in [args.dataset_dir, args.train_dir, args.lmdb_dir]:
        if is_master and p: p.mkdir(parents=True, exist_ok=True)
    if is_distributed: dist.barrier()

    if is_master: _logger.info(f"Initializing {args.densenet_model_name} backbone...")
    densenet_base = RGBDenseNet201()
    densenet_feature_nodes = ['1.features.denseblock4.denselayer32.norm1', '1.features.denseblock4.denselayer32.conv1', '1.features.denseblock4.denselayer31.conv2']
    features_module = FeatureExtractor(densenet_base, densenet_feature_nodes)
    for param in features_module.parameters(): param.requires_grad = False
    features_module.eval().to(device)
    
    with torch.no_grad():
        concatenated_channels = sum(f.shape[1] for f in features_module(torch.randn(1, 3, 256, 256).to(device)))
    if is_master: _logger.info(f"Determined concatenated input channels for SPADE head: {concatenated_channels}")

    if args.stage.startswith('salicon_pretrain'):
        if is_master: _logger.info(f"--- Preparing SALICON Pretraining with DenseNet+SPADE ---")
        experiment_name = f"{args.stage}"
        salicon_loc = args.dataset_dir / 'SALICON'
        if is_master:
            if not (salicon_loc/'stimuli'/'train').exists(): pysaliency.get_SALICON_train(location=str(salicon_loc.parent))
            if not (salicon_loc/'stimuli'/'val').exists(): pysaliency.get_SALICON_val(location=str(salicon_loc.parent))
        if is_distributed: dist.barrier()
        train_stim, train_fix = pysaliency.get_SALICON_train(location=str(salicon_loc.parent))
        val_stim, val_fix = pysaliency.get_SALICON_val(location=str(salicon_loc.parent))
        centerbias = BaselineModel(train_stim, train_fix, bandwidth=0.0217, eps=2e-13, caching=False)

        train_ll_cache = args.dataset_dir/f'salicon_baseline_train_ll_{args.densenet_model_name}.pkl'
        val_ll_cache = args.dataset_dir/f'salicon_baseline_val_ll_{args.densenet_model_name}.pkl'
        train_ll, val_ll = None, None
        if is_master:
            try: train_ll = cpickle.load(open(train_ll_cache, 'rb'))
            except: train_ll = centerbias.information_gain(train_stim, train_fix, verbose=True, average='image'); cpickle.dump(train_ll, open(train_ll_cache, 'wb'))
            try: val_ll = cpickle.load(open(val_ll_cache, 'rb'))
            except: val_ll = centerbias.information_gain(val_stim, val_fix, verbose=True, average='image'); cpickle.dump(val_ll, open(val_ll_cache, 'wb'))
        
        ll_bcast = [train_ll, val_ll]; dist.broadcast_object_list(ll_bcast, src=0) if is_distributed else None
        train_ll, val_ll = ll_bcast[0], ll_bcast[1]
        if np.isnan(train_ll) or np.isnan(val_ll): _logger.critical("NaN LLs received."); sys.exit(1)

        saliency_net = SaliencyNetworkSPADE(concatenated_channels, args.num_total_segments, args.seg_embedding_dim, args.add_sa_head)
        fixsel_net = build_fixation_selection_network(scanpath_features=0)
        model = DeepGazeIIISpade(features_module, saliency_net, None, fixsel_net, args.finalizer_learn_sigma, args.finalizer_initial_sigma).to(device)
        if is_distributed: model = DDP(model, device_ids=[device.index], find_unused_parameters=True)
        optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones)

        ds_kwargs_train = {"transform": FixationMaskTransform(sparse=False), "average": "image", "lmdb_path": str(args.lmdb_dir / f'SALICON_train_images_{args.densenet_model_name}') if args.use_lmdb_images else None, "segmentation_mask_dir": args.salicon_train_mask_dir, "segmentation_mask_format": args.segmentation_mask_format, "segmentation_mask_fixed_memmap_file": args.train_mask_memmap_file, "segmentation_mask_variable_payload_file": args.train_mask_variable_payload_file, "segmentation_mask_variable_header_file": args.train_mask_variable_header_file, "segmentation_mask_bank_dtype": args.segmentation_mask_bank_dtype}
        train_dataset = ImageDatasetWithSegmentation(train_stim, train_fix, centerbias, **ds_kwargs_train)
        ds_kwargs_val = {"transform": FixationMaskTransform(sparse=False), "average": "image", "lmdb_path": str(args.lmdb_dir / f'SALICON_val_images_{args.densenet_model_name}') if args.use_lmdb_images else None, "segmentation_mask_dir": args.salicon_val_mask_dir, "segmentation_mask_format": args.segmentation_mask_format, "segmentation_mask_fixed_memmap_file": args.val_mask_memmap_file, "segmentation_mask_variable_payload_file": args.val_mask_variable_payload_file, "segmentation_mask_variable_header_file": args.val_mask_variable_header_file, "segmentation_mask_bank_dtype": args.segmentation_mask_bank_dtype}
        val_dataset = ImageDatasetWithSegmentation(val_stim, val_fix, centerbias, **ds_kwargs_val)

        train_sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=True, drop_last=True) if is_distributed else None
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True, persistent_workers=args.num_workers > 0)
        val_sampler = torch.utils.data.DistributedSampler(val_dataset, shuffle=False, drop_last=False) if is_distributed else None
        validation_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, sampler=val_sampler, persistent_workers=args.num_workers > 0)
        
        output_dir = args.train_dir / experiment_name
        _train(this_directory=str(output_dir), model=model, train_loader=train_loader, train_baseline_log_likelihood=train_ll, val_loader=validation_loader, val_baseline_log_likelihood=val_ll, optimizer=optimizer, lr_scheduler=lr_scheduler, gradient_accumulation_steps=args.gradient_accumulation_steps, minimum_learning_rate=args.min_lr, validation_metrics=['LL', 'IG', 'NSS', 'AUC_CPU'], validation_epochs=args.validation_epochs, startwith=args.resume_checkpoint, device=device, is_distributed=is_distributed, is_master=is_master, logger=_logger)

    elif args.stage.startswith('mit_spatial'):
        fold = args.fold
        if fold is None or not (0 <= fold < 10): _logger.critical("--fold (0-9) required."); sys.exit(1)
        if is_master: _logger.info(f"--- Preparing MIT Stage: {args.stage} (Fold {fold}) ---")
        
        current_lr = args.lr_mit_spatial if args.lr_mit_spatial is not None else args.lr
        current_milestones = args.lr_milestones_mit_spatial if args.lr_milestones_mit_spatial is not None else args.lr_milestones
        experiment_name_mit = f"{args.stage}_fold{fold}_{args.densenet_model_name}_k{args.num_total_segments}_emb{args.seg_embedding_dim}_lr{current_lr}"
        
        salicon_checkpoint_path = args.salicon_checkpoint_path
        if not salicon_checkpoint_path:
            salicon_dir = args.train_dir / "salicon_pretrain_densenet_spade"
            for p in [salicon_dir / 'final_best_val.pth', salicon_dir / 'final.pth']:
                if p.exists(): salicon_checkpoint_path = p; break

        # --- ADDED: MIT Data Conversion and Caching ---
        mit_converted_data_path = args.train_dir / f"MIT1003_converted_deepgaze_{args.densenet_model_name}"
        mit_stimuli_cache_file = mit_converted_data_path / "stimuli.pkl"
        
        mit_stimuli_all, mit_fixations_all = None, None

        # Check for cache, otherwise run conversion
        if mit_stimuli_cache_file.exists() and mit_stimuli_cache_file.stat().st_size > 0:
            if is_master: _logger.info(f"Loading pre-converted MIT data from cache: {mit_converted_data_path}")
            with open(mit_stimuli_cache_file, "rb") as f: mit_stimuli_all = cpickle.load(f)
            mit_stimuli_orig, mit_fixations_orig = pysaliency.external_datasets.mit.get_mit1003_with_initial_fixation(location=str(args.dataset_dir), replace_initial_invalid_fixations=True)
            mit_fixations_all = convert_fixation_trains(mit_stimuli_orig, mit_fixations_orig, is_master, _logger)
        else:
            if is_master:
                _logger.info(f"No valid MIT cache found. Starting data conversion...")
                mit_converted_data_path.mkdir(parents=True, exist_ok=True)
            
            mit_stimuli_orig, mit_fixations_orig = pysaliency.external_datasets.mit.get_mit1003_with_initial_fixation(location=str(args.dataset_dir), replace_initial_invalid_fixations=True)
            mit_stimuli_all = convert_stimuli(mit_stimuli_orig, mit_converted_data_path, is_master, is_distributed, device, _logger)
            mit_fixations_all = convert_fixation_trains(mit_stimuli_orig, mit_fixations_orig, is_master, _logger)
        
        if is_distributed: dist.barrier()
        if mit_stimuli_all is None or mit_fixations_all is None:
            _logger.critical("Failed to load or convert MIT data. Exiting."); sys.exit(1)

        train_stim, train_fix = train_split(mit_stimuli_all, mit_fixations_all, crossval_folds=10, fold_no=fold)
        val_stim, val_fix = validation_split(mit_stimuli_all, mit_fixations_all, crossval_folds=10, fold_no=fold)
        centerbias = CrossvalidatedBaselineModel(mit_stimuli_all, mit_fixations_all, bandwidth=10**-1.6667673342543432, eps=10**-14.884189168516073, caching=False)
        
        # Use a new cache file for baseline LLs based on converted data
        train_ll_cache = args.dataset_dir / f'mit1003_baseline_train_ll_fold{fold}_{args.densenet_model_name}_converted.pkl'
        val_ll_cache = args.dataset_dir / f'mit1003_baseline_val_ll_fold{fold}_{args.densenet_model_name}_converted.pkl'
        train_ll, val_ll = None, None
        if is_master:
            try: train_ll = cpickle.load(open(train_ll_cache, 'rb'))
            except: train_ll = centerbias.information_gain(train_stim, train_fix, verbose=True, average='image'); cpickle.dump(train_ll, open(train_ll_cache, 'wb'))
            try: val_ll = cpickle.load(open(val_ll_cache, 'rb'))
            except: val_ll = centerbias.information_gain(val_stim, val_fix, verbose=True, average='image'); cpickle.dump(val_ll, open(val_ll_cache, 'wb'))
        
        ll_bcast = [train_ll, val_ll]; dist.broadcast_object_list(ll_bcast, src=0) if is_distributed else None
        train_ll, val_ll = ll_bcast[0], ll_bcast[1]
        if np.isnan(train_ll) or np.isnan(val_ll): _logger.critical("NaN LLs received for MIT."); sys.exit(1)

        saliency_net = SaliencyNetworkSPADE(concatenated_channels, args.num_total_segments, args.seg_embedding_dim, args.add_sa_head)
        fixsel_net = build_fixation_selection_network(scanpath_features=0)
        model_cpu = DeepGazeIIISpade(features_module.cpu(), saliency_net, None, fixsel_net, args.finalizer_learn_sigma, args.finalizer_initial_sigma)
        features_module.to(device)

        if salicon_checkpoint_path and salicon_checkpoint_path.exists():
            if is_master: _logger.info(f"Loading SALICON weights from {salicon_checkpoint_path}")
            restore_from_checkpoint(model=model_cpu, optimizer=None, scheduler=None, scaler=None, path=str(salicon_checkpoint_path), device='cpu', is_distributed=False, logger=_logger)
        
        model = model_cpu.to(device)
        if is_distributed: model = DDP(model, device_ids=[device.index], find_unused_parameters=True)
        optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=current_lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=current_milestones)

        ds_kwargs_mit = {"transform": FixationMaskTransform(sparse=False), "average": "image", "lmdb_path": str(args.lmdb_dir / f'MIT1003_images_fold{fold}_{args.densenet_model_name}') if args.use_lmdb_images else None, "segmentation_mask_dir": args.mit_all_mask_dir, "segmentation_mask_format": args.segmentation_mask_format, "segmentation_mask_fixed_memmap_file": args.mit_mask_fixed_memmap_file, "segmentation_mask_variable_payload_file": args.mit_mask_variable_payload_file, "segmentation_mask_variable_header_file": args.mit_mask_variable_header_file, "segmentation_mask_bank_dtype": args.segmentation_mask_bank_dtype}
        train_dataset = ImageDatasetWithSegmentation(train_stim, train_fix, centerbias, **ds_kwargs_mit)
        val_dataset = ImageDatasetWithSegmentation(val_stim, val_fix, centerbias, **ds_kwargs_mit)
        
        # Use standard DistributedSampler since images are now standard sizes
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=True, drop_last=True) if is_distributed else None
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True, persistent_workers=args.num_workers > 0)
        val_sampler = torch.utils.data.DistributedSampler(val_dataset, shuffle=False, drop_last=False) if is_distributed else None
        validation_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, sampler=val_sampler, persistent_workers=args.num_workers > 0)
        
        output_dir = args.train_dir / experiment_name_mit
        _train(this_directory=str(output_dir), model=model, train_loader=train_loader, train_baseline_log_likelihood=train_ll, val_loader=validation_loader, val_baseline_log_likelihood=val_ll, optimizer=optimizer, lr_scheduler=lr_scheduler, gradient_accumulation_steps=args.gradient_accumulation_steps, minimum_learning_rate=args.min_lr, validation_metrics=['LL', 'IG', 'NSS', 'AUC_CPU'], validation_epochs=args.validation_epochs, startwith=None, device=device, is_distributed=is_distributed, is_master=is_master, logger=_logger)
    else:
        _logger.critical(f"Unknown or unsupported stage: {args.stage}"); sys.exit(1)

    cleanup_distributed()

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    _module_logger = logging.getLogger(__name__)

    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument('--config_file', type=str, default=None, help="Path to YAML configuration file.")
    _cfg_namespace, _remaining_cli_args = _pre.parse_known_args()

    parser = argparse.ArgumentParser(parents=[_pre], description="Train DeepGazeIII with DenseNet+SPADE",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--stage', choices=['salicon_pretrain_densenet_spade', 'mit_spatial_densenet_spade'], help='Training stage to execute.')
    parser.add_argument('--densenet_model_name', default='densenet201', choices=['densenet161', 'densenet201'])
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    parser.add_argument('--num_total_segments', type=int, default=16)
    parser.add_argument('--seg_embedding_dim', type=int, default=64)
    parser.add_argument('--segmentation_mask_format', default='png', choices=['png', 'npy'])
    parser.add_argument('--segmentation_mask_bank_dtype', type=str, default='uint8', choices=['uint8', 'uint16'])
    parser.add_argument('--salicon_train_mask_dir', type=str, help="DIRECT path to SALICON training individual mask files.")
    parser.add_argument('--salicon_val_mask_dir', type=str, help="DIRECT path to SALICON validation individual mask files.")
    parser.add_argument('--mit_all_mask_dir', type=str, help="DIRECT path to ALL MIT1003 individual mask files.")
    parser.add_argument('--train_mask_memmap_file', type=str, help="Fixed-size bank for TRAIN SALICON masks.")
    parser.add_argument('--val_mask_memmap_file', type=str, help="Fixed-size bank for VAL SALICON masks.")
    parser.add_argument('--train_mask_variable_payload_file', type=str, help="Variable-size PAYLOAD for TRAIN SALICON.")
    parser.add_argument('--train_mask_variable_header_file', type=str, help="Variable-size HEADER for TRAIN SALICON.")
    parser.add_argument('--val_mask_variable_payload_file', type=str, help="Variable-size PAYLOAD for VAL SALICON.")
    parser.add_argument('--val_mask_variable_header_file', type=str, help="Variable-size HEADER for VAL SALICON.")
    parser.add_argument('--mit_mask_fixed_memmap_file', type=str, help="Fixed-size bank for ALL MIT1003 masks.")
    parser.add_argument('--mit_mask_variable_payload_file', type=str, help="Variable-size PAYLOAD for ALL MIT1003 masks.")
    parser.add_argument('--mit_mask_variable_header_file', type=str, help="Variable-size HEADER for ALL MIT1003 masks.")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-4, help="LR for SALICON pretraining.")
    parser.add_argument('--lr_milestones', type=int, nargs='+', default=[20, 35, 50]) 
    parser.add_argument('--min_lr', type=float, default=1e-6) 
    parser.add_argument('--validation_epochs', type=int, default=1)
    parser.add_argument('--resume_checkpoint', type=str, help='Path to checkpoint to resume SALICON stage from.')
    parser.add_argument('--finalizer_initial_sigma', type=float, default=8.0)
    parser.add_argument('--finalizer_learn_sigma', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--add_sa_head', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--num_workers', type=str, default='auto')
    parser.add_argument('--train_dir', type=str, default='./experiments_dg3_spade')
    parser.add_argument('--dataset_dir', type=str, default='./data/pysaliency_datasets')
    parser.add_argument('--lmdb_dir', type=str, default='./data/lmdb_caches')
    parser.add_argument('--use_lmdb_images', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--fold', type=int, help='MIT1003 fold (0-9).')
    parser.add_argument('--lr_mit_spatial', type=float, default=1e-4, help="LR for MIT spatial fine-tuning.")
    parser.add_argument('--lr_milestones_mit_spatial', type=int, nargs='+', default=[10, 20])
    parser.add_argument('--salicon_checkpoint_path', type=str, help='Path to SALICON pretrained checkpoint for MIT.')

    if _cfg_namespace.config_file:
        try:
            with open(_cfg_namespace.config_file, 'r') as f: yaml_cfg = yaml.safe_load(f) or {}
            parser.set_defaults(**{k: v for k,v in yaml_cfg.items() if hasattr(parser.parse_args([]), k)})
        except Exception as e: _module_logger.error(f"Could not read/parse YAML: {e}")

    final_args_ns = parser.parse_args(_remaining_cli_args)
    
    ws_env = int(os.environ.get("WORLD_SIZE", 1))
    if isinstance(final_args_ns.num_workers, str) and final_args_ns.num_workers.lower() == 'auto':
        try: cpu_c = len(os.sched_getaffinity(0))
        except AttributeError: cpu_c = os.cpu_count() or 1
        final_args_ns.num_workers = min(8, cpu_c // ws_env if ws_env > 0 else cpu_c)
    else: final_args_ns.num_workers = int(final_args_ns.num_workers)

    def resolve_path_arg(arg_value):
        if arg_value is None: return None
        path = Path(arg_value)
        return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    for arg_name, arg_value in vars(final_args_ns).items():
        if 'dir' in arg_name or 'file' in arg_name or 'path' in arg_name:
            setattr(final_args_ns, arg_name, resolve_path_arg(arg_value))
    
    try:
        main(final_args_ns) 
    except KeyboardInterrupt: _module_logger.warning("Training interrupted by user (Ctrl+C)."); cleanup_distributed(); sys.exit(130)
    except Exception: _module_logger.critical("Unhandled exception during main execution:", exc_info=True); cleanup_distributed(); sys.exit(1)