#!/usr/bin/env python
"""
Multi-GPU-ready training script for DeepGaze with a DenseNet backbone,
SPADE normalization, and DYNAMIC semantic embeddings derived from a DINOv2
backbone. This creates a hybrid model where the main features are from
DenseNet, but the semantic context for SPADE modulation comes from DINOv2.
(Version includes fixes for distributed deadlock and device placement).
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
    from src.metrics import log_likelihood, nss, auc as auc_cpu_fn
    from src.training import _train, restore_from_checkpoint
except ImportError as e:
    print(f"PYTHON IMPORT ERROR: {e}\n(sys.path: {sys.path})")
    sys.exit(1)

_logger = logging.getLogger("train_hybrid_gaze_spade")

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

class DeepGazeIIIDynamicDinoEmbedding(nn.Module):
    def __init__(self, features_module: FeatureExtractor, dino_features_module: DinoV2Backbone,
                 saliency_network: SaliencyNetworkSPADEDynamic, fixation_selection_network,
                 num_total_segments: int, finalizer_learn_sigma: bool, initial_sigma=8.0,
                 scanpath_network=None, downsample_input_to_backbone=1, readout_factor=4, saliency_map_factor_finalizer=4):
        super().__init__()
        self.features = features_module
        self.dino_features = dino_features_module
        self.saliency_network = saliency_network
        self.scanpath_network = scanpath_network
        self.fixation_selection_network = fixation_selection_network
        self.num_total_segments = num_total_segments
        self.downsample_input_to_backbone = downsample_input_to_backbone
        self.readout_factor = readout_factor
        self.finalizer = Finalizer(sigma=initial_sigma, learn_sigma=finalizer_learn_sigma, saliency_map_factor=saliency_map_factor_finalizer)

        if hasattr(self.dino_features, 'parameters'):
            for param in self.dino_features.parameters(): param.requires_grad = False
        if hasattr(self.dino_features, 'eval'): self.dino_features.eval()

    def _create_painted_semantic_map(self, F_dino_patches, raw_sam_pixel_segmap):
        B, C_dino, H_p, W_p = F_dino_patches.shape
        _, H_img, W_img = raw_sam_pixel_segmap.shape
        device = F_dino_patches.device; dtype = F_dino_patches.dtype
        painted_map_batch = torch.zeros(B, C_dino, H_img, W_img, device=device, dtype=dtype)
        for b_idx in range(B):
            img_patch_features = F_dino_patches[b_idx]
            img_sam_pixel_segmap = raw_sam_pixel_segmap[b_idx]
            segmap_at_patch_res = F.interpolate(img_sam_pixel_segmap.unsqueeze(0).unsqueeze(0).float(), size=(H_p, W_p), mode='nearest').squeeze(0).squeeze(0).long()
            flat_patch_features = img_patch_features.permute(1, 2, 0).reshape(H_p * W_p, C_dino)
            flat_patch_sam_ids = torch.clamp(segmap_at_patch_res.reshape(H_p * W_p), 0, self.num_total_segments - 1)
            segment_avg_dino_features = scatter_mean(src=flat_patch_features, index=flat_patch_sam_ids, dim=0, dim_size=self.num_total_segments)
            segment_avg_dino_features = torch.nan_to_num(segment_avg_dino_features, nan=0.0)
            clamped_pixel_sam_ids = torch.clamp(img_sam_pixel_segmap.long(), 0, self.num_total_segments - 1)
            painted_map_batch[b_idx] = segment_avg_dino_features[clamped_pixel_sam_ids].permute(2, 0, 1)
        return painted_map_batch

    def forward(self, image, centerbias, segmentation_mask, **kwargs):
        if segmentation_mask is None: raise ValueError(f"{self.__class__.__name__} requires 'segmentation_mask'.")
        
        img_for_features = image
        if self.downsample_input_to_backbone != 1:
            img_for_features = F.interpolate(image, scale_factor=1.0 / self.downsample_input_to_backbone, mode='bilinear', align_corners=False)

        with torch.no_grad():
            dino_feature_maps = self.dino_features(img_for_features)
            semantic_dino_patches = dino_feature_maps[0]
        
        painted_map = self._create_painted_semantic_map(semantic_dino_patches, segmentation_mask)
        extracted_feature_maps = self.features(img_for_features)
        readout_h = math.ceil(image.shape[2] / self.downsample_input_to_backbone / self.readout_factor)
        readout_w = math.ceil(image.shape[3] / self.downsample_input_to_backbone / self.readout_factor)
        
        processed_features_list = [F.interpolate(f, size=(readout_h, readout_w), mode='bilinear', align_corners=False) for f in extracted_feature_maps]
        concatenated_features = torch.cat(processed_features_list, dim=1)
        
        saliency_output = self.saliency_network(concatenated_features, painted_map)
        final_readout = self.fixation_selection_network((saliency_output, None))
        log_density = self.finalizer(final_readout, centerbias)
        return log_density

    def train(self, mode=True):
        if hasattr(self.features, 'eval'): self.features.eval()
        if hasattr(self.dino_features, 'eval'): self.dino_features.eval()
        self.saliency_network.train(mode); self.fixation_selection_network.train(mode); self.finalizer.train(mode)
        super().train(mode)

def salicon_pretrain(args, device, is_master, is_distributed, model):
    if is_master: _logger.info("--- Preparing SALICON Pretraining ---")
    
    salicon_loc = args.dataset_dir / 'SALICON'
    if is_master:
        if not (salicon_loc/'stimuli'/'train').exists(): pysaliency.get_SALICON_train(location=str(salicon_loc.parent))
        if not (salicon_loc/'stimuli'/'val').exists(): pysaliency.get_SALICON_val(location=str(salicon_loc.parent))
    if is_distributed: dist.barrier()
    
    train_stim, train_fix = pysaliency.get_SALICON_train(location=str(salicon_loc.parent))
    val_stim, val_fix = pysaliency.get_SALICON_val(location=str(salicon_loc.parent))
    centerbias = BaselineModel(train_stim, train_fix, bandwidth=0.0217, eps=2e-13, caching=False)

    train_ll, val_ll = None, None
    if is_master:
        _logger.info("Master computing baseline LLs... (this may take a while)")
        train_ll = centerbias.information_gain(train_stim, train_fix, verbose=True, average='image')
        val_ll = centerbias.information_gain(val_stim, val_fix, verbose=True, average='image')
        _logger.info(f"Master LLs computed: Train={train_ll:.4f}, Val={val_ll:.4f}")

    if is_distributed: dist.barrier()
    
    ll_bcast = [train_ll, val_ll]
    if is_distributed: dist.broadcast_object_list(ll_bcast, src=0)
    train_ll, val_ll = ll_bcast
    if train_ll is None or val_ll is None: _logger.critical("NaN LLs received."); sys.exit(1)

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones)
    
    ds_kwargs_train = {"transform": FixationMaskTransform(sparse=False), "average": "image", "segmentation_mask_dir": args.salicon_train_mask_dir, "segmentation_mask_format": args.segmentation_mask_format}
    ds_kwargs_val = {"transform": FixationMaskTransform(sparse=False), "average": "image", "segmentation_mask_dir": args.salicon_val_mask_dir, "segmentation_mask_format": args.segmentation_mask_format}
    train_dataset = ImageDatasetWithSegmentation(train_stim, train_fix, centerbias, **ds_kwargs_train)
    val_dataset = ImageDatasetWithSegmentation(val_stim, val_fix, centerbias, **ds_kwargs_val)
    train_sampler = (torch.utils.data.DistributedSampler(train_dataset, shuffle=True, drop_last=True) if is_distributed else None)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True, persistent_workers=args.num_workers > 0)
    val_sampler = (torch.utils.data.DistributedSampler(val_dataset, shuffle=False, drop_last=False) if is_distributed else None)
    validation_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, sampler=val_sampler, persistent_workers=args.num_workers > 0)
    
    experiment_name = f"{args.stage}_{args.densenet_model_name}_dino_{args.dino_model_name}_lr{args.lr}"
    output_dir = args.train_dir / experiment_name
    
    _train(str(output_dir), model, train_loader, train_ll, validation_loader, val_ll, optimizer, lr_scheduler, args.gradient_accumulation_steps, args.min_lr, ['LL', 'IG', 'NSS', 'AUC_CPU'], args.validation_epochs, args.resume_checkpoint, device, is_distributed, is_master, _logger)

def mit_finetune(args, device, is_master, is_distributed, model_cpu):
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

    train_ll, val_ll = None, None
    if is_master:
        _logger.info(f"Master computing baseline LLs for Fold {fold}...")
        train_ll = centerbias.information_gain(train_stim, train_fix, verbose=True, average='image')
        val_ll = centerbias.information_gain(val_stim, val_fix, verbose=True, average='image')
    
    if is_distributed: dist.barrier()
    
    ll_bcast = [train_ll, val_ll]
    if is_distributed: dist.broadcast_object_list(ll_bcast, src=0)
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

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr_mit_spatial)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones_mit_spatial)

    mit_ds_kwargs = {"transform": FixationMaskTransform(sparse=False), "average": "image", "segmentation_mask_dir": args.mit_all_mask_dir, "segmentation_mask_format": args.segmentation_mask_format}
    train_dataset = ImageDatasetWithSegmentation(train_stim, train_fix, centerbias, **mit_ds_kwargs)
    val_dataset = ImageDatasetWithSegmentation(val_stim, val_fix, centerbias, **mit_ds_kwargs)
    
    train_sampler = (torch.utils.data.DistributedSampler(train_dataset, shuffle=True, drop_last=True) if is_distributed else None)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True, persistent_workers=args.num_workers > 0)
    val_sampler = (torch.utils.data.DistributedSampler(val_dataset, shuffle=False, drop_last=False) if is_distributed else None)
    validation_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, sampler=val_sampler, persistent_workers=args.num_workers > 0)

    experiment_name = f"{args.stage}_fold{fold}_{args.densenet_model_name}_dino_{args.dino_model_name}_k{args.num_total_segments}_lr{args.lr_mit_spatial}"
    output_dir = args.train_dir / experiment_name
    
    _train(str(output_dir), model, train_loader, train_ll, validation_loader, val_ll, optimizer, lr_scheduler, args.gradient_accumulation_steps, args.min_lr, ['LL', 'IG', 'NSS', 'AUC_CPU'], args.validation_epochs, None, device, is_distributed, is_master, _logger)

def main(args):
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

    if is_master: _logger.info(f"Initializing {args.densenet_model_name} backbone for main saliency path...")
    densenet_base = RGBDenseNet201()
    densenet_feature_nodes = ['1.features.denseblock4.denselayer32.norm1', '1.features.denseblock4.denselayer32.conv1', '1.features.denseblock4.denselayer31.conv2']
    features_module = FeatureExtractor(densenet_base, densenet_feature_nodes)
    for param in features_module.parameters(): param.requires_grad = False
    features_module.to(device).eval()

    if is_master: _logger.info(f"Initializing {args.dino_model_name} backbone for semantic embeddings...")
    dino_semantic_backbone = DinoV2Backbone(
        layers=[args.dino_semantic_layer_idx],
        model_name=args.dino_model_name,
        patch_size=args.dino_patch_size,
        freeze=True
    )
    dino_semantic_backbone.to(device).eval()

    with torch.no_grad():
        # THE FIX: Move dummy_input to the correct device
        dummy_input = torch.randn(1, 3, 256, 256).to(device)
        dummy_out_densenet = features_module(dummy_input)
        main_path_channels = sum(f.shape[1] for f in dummy_out_densenet)
        dummy_out_dino = dino_semantic_backbone(dummy_input)
        semantic_path_channels = dummy_out_dino[0].shape[1]
    if is_master:
        _logger.info(f"Main path (DenseNet) concatenated channels: {main_path_channels}")
        _logger.info(f"Dynamic semantic path (DINO) channels: {semantic_path_channels}")

    saliency_net = SaliencyNetworkSPADEDynamic(main_path_channels, semantic_path_channels)
    fixsel_net = build_fixation_selection_network(scanpath_features=0)
    
    model_cpu = DeepGazeIIIDynamicDinoEmbedding(
        features_module.cpu(), dino_semantic_backbone.cpu(), saliency_net, fixsel_net, 
        args.num_total_segments, args.finalizer_learn_sigma, args.finalizer_initial_sigma
    )

    if args.stage.startswith('salicon_pretrain'):
        model = model_cpu.to(device)
        if is_distributed: model = DDP(model, device_ids=[device.index], find_unused_parameters=True)
        if args.use_torch_compile: model = torch.compile(model)
        salicon_pretrain(args, device, is_master, is_distributed, model)
    elif args.stage.startswith('mit_spatial'):
        mit_finetune(args, device, is_master, is_distributed, model_cpu)
    else:
        _logger.critical(f"Unknown or unsupported stage: {args.stage}"); sys.exit(1)
    
    cleanup_distributed()

if __name__ == "__main__":
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument('--config_file', type=str, default=None)
    _cfg_ns, _rem_args = _pre.parse_known_args()

    parser = argparse.ArgumentParser(parents=[_pre], description="Train Hybrid DenseNet+DINO SPADE model.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--stage', choices=['salicon_pretrain_hybrid', 'mit_spatial_hybrid'], help='Training stage to execute.')
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    parser.add_argument('--densenet_model_name', default='densenet201', choices=['densenet201'])
    parser.add_argument('--dino_model_name', default='dinov2_vitb14', choices=['dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'])
    parser.add_argument('--dino_patch_size', type=int, default=14)
    parser.add_argument('--dino_semantic_layer_idx', type=int, default=-1, help="Index of the DINOv2 layer to use for semantic features.")
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
    parser.add_argument('--use_torch_compile', action=argparse.BooleanOptionalAction, default=True)
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

    if _cfg_ns.config_file:
        try:
            with open(_cfg_ns.config_file, 'r') as f: yaml_cfg = yaml.safe_load(f) or {}
            parser.set_defaults(**yaml_cfg)
        except Exception as e: _logger.error(f"Could not read/parse YAML: {e}")

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