#!/usr/bin/env python
"""
Multi-GPU-ready training script for DinoGaze with DINOv2 backbone,
SPADE normalization, and dynamic semantic embeddings from SAM masks.
Semantic features for SPADE are derived on-the-fly from the DINOv2 backbone's
patch embeddings, guided by pre-loaded SAM segmentation masks, using torch_scatter.
Includes SALICON pretraining and MIT1003 fine-tuning stages.
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
import pickle # For pysaliency FileStimuli caching

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader # Added for MIT dataset loading
import numpy as np
import math

import pysaliency
import pysaliency.external_datasets.mit
from pysaliency.dataset_config import train_split, validation_split # For MIT splits
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from pysaliency.baseline_utils import BaselineModel, CrossvalidatedBaselineModel # Added Crossvalidated for MIT
import cloudpickle as cpickle
from tqdm import tqdm # For data conversion
from boltons.fileutils import atomic_save # For safe saving of converted data


try:
    from torch_scatter import scatter_mean
except ImportError:
    print("torch_scatter not found. Please install it for efficient semantic map creation.")
    print("See https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html")
    sys.exit(1)

# --- Local Project Imports ---
try:
    from src.data import (
        ImageDatasetWithSegmentation,
        FixationMaskTransform,
        convert_stimuli as convert_stimuli_mit, # Renaming to avoid conflict if original DGIII utils are also in src
        convert_fixation_trains as convert_fixation_trains_mit # Not used directly for spatial, but good to have
    )
    from src.dinov2_backbone import DinoV2Backbone
    from src.modules import Finalizer, encode_scanpath_features, build_fixation_selection_network
    from src.layers import Bias, LayerNorm # LayerNormMultiInput not used by this model's fixselnet
    from src.metrics import log_likelihood, nss, auc as auc_cpu_fn
    from src.training import _train
except ImportError as e:
    print(f"PYTHON IMPORT ERROR: {e}")
    print(f"Current sys.path: {sys.path}")
    print("Ensure 'src' is in sys.path and all required modules are present.")
    sys.exit(1)

_logger = logging.getLogger("train_dinogaze_spade_dynamic_sam_scatter")

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
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_master = True
    return device, rank, world_size, is_master, is_distributed

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

class SPADELayerNormDynamic(nn.Module):
    def __init__(self, norm_features, semantic_feature_channels,
                 hidden_mlp_channels=128, eps=1e-12, kernel_size=3):
        super().__init__()
        self.norm_features = norm_features
        self.eps = eps
        self.semantic_feature_channels = semantic_feature_channels
        padding = kernel_size // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(self.semantic_feature_channels, hidden_mlp_channels,
                      kernel_size=kernel_size, padding=padding, bias=True),
            nn.ReLU(inplace=True)
        )
        self.mlp_gamma = nn.Conv2d(hidden_mlp_channels, norm_features,
                                   kernel_size=kernel_size, padding=padding, bias=True)
        self.mlp_beta = nn.Conv2d(hidden_mlp_channels, norm_features,
                                  kernel_size=kernel_size, padding=padding, bias=True)

    def forward(self, x, painted_semantic_map):
        if x.shape[1] != self.norm_features:
            raise ValueError(f"Input x chans {x.shape[1]} != norm_features {self.norm_features}")
        normalized_shape = (self.norm_features, x.size(2), x.size(3))
        normalized_x = F.layer_norm(x, normalized_shape, weight=None, bias=None, eps=self.eps)
        if painted_semantic_map.ndim != 4:
             raise ValueError(f"Expected painted_semantic_map to be 4D, got {painted_semantic_map.ndim}")
        if painted_semantic_map.shape[1] != self.semantic_feature_channels:
             _logger.error(f"Painted map channels {painted_semantic_map.shape[1]} != expected {self.semantic_feature_channels}")
             raise ValueError("Painted semantic map channel mismatch")
        semantic_map_resized = F.interpolate(painted_semantic_map, size=x.size()[2:], mode='bilinear', align_corners=False)
        shared_features = self.mlp_shared(semantic_map_resized)
        gamma_map = self.mlp_gamma(shared_features)
        beta_map = self.mlp_beta(shared_features)
        out = normalized_x * (1 + gamma_map) + beta_map
        return out

    def extra_repr(self):
        return (f'norm_features={self.norm_features}, '
                f'semantic_feature_channels={self.semantic_feature_channels}, eps={self.eps}')

class SaliencyNetworkSPADEDynamic(nn.Module):
    def __init__(self, input_channels_main_path, semantic_feature_channels_for_spade):
        super().__init__()
        self.input_channels_main_path = input_channels_main_path
        self.semantic_feature_channels_for_spade = semantic_feature_channels_for_spade
        self.spade_ln0 = SPADELayerNormDynamic(input_channels_main_path, self.semantic_feature_channels_for_spade)
        self.conv0 = nn.Conv2d(input_channels_main_path, 8, (1, 1), bias=False)
        self.bias0 = Bias(8); self.softplus0 = nn.Softplus()
        self.spade_ln1 = SPADELayerNormDynamic(8, self.semantic_feature_channels_for_spade)
        self.conv1 = nn.Conv2d(8, 16, (1, 1), bias=False)
        self.bias1 = Bias(16); self.softplus1 = nn.Softplus()
        self.spade_ln2 = SPADELayerNormDynamic(16, self.semantic_feature_channels_for_spade)
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

class DinoGazeSpade(nn.Module):
    def __init__(self, features_module: DinoV2Backbone,
                 saliency_network: SaliencyNetworkSPADEDynamic,
                 fixation_selection_network, # For spatial only, this combines saliency and (None) scanpath
                 # Scanpath related (placeholders if we extend later)
                 scanpath_network=None, # Placeholder for a potential ScanpathNetworkSPADEDynamic
                 # DINO & SAM specific
                 dinov2_patch_size: int = 14,
                 semantic_feature_layer_idx: int = -1,
                 num_sam_segments: int = 64,
                 # General model structure
                 downsample_input_to_backbone=1,
                 readout_factor=7, # Default for ViT-B/L, ViT-G is 14
                 saliency_map_factor_finalizer=4,
                 initial_sigma=8.0,
                 finalizer_learn_sigma=True):
        super().__init__()

        self.features = features_module
        if hasattr(self.features, 'parameters'): # Should always be true for nn.Module
            for param in self.features.parameters():
                param.requires_grad = False # Freeze backbone by default
        if hasattr(self.features, 'eval'):
            self.features.eval()

        self.saliency_network = saliency_network
        self.scanpath_network = scanpath_network # Store it, even if None for now
        self.fixation_selection_network = fixation_selection_network

        self.dinov2_patch_size = dinov2_patch_size
        self.semantic_feature_layer_idx = semantic_feature_layer_idx
        self.num_sam_segments = num_sam_segments

        self.downsample_input_to_backbone = downsample_input_to_backbone
        self.readout_factor = readout_factor

        self.finalizer = Finalizer(
            sigma=initial_sigma,
            learn_sigma=finalizer_learn_sigma,
            saliency_map_factor=saliency_map_factor_finalizer,
        )
        _logger.info(f"DinoGazeSpade initialized. DINO patch size: {self.dinov2_patch_size}, "
                     f"Semantic features from DINO layer index: {self.semantic_feature_layer_idx}, "
                     f"Num SAM segments expected for painting: {self.num_sam_segments}, "
                     f"Scanpath Network: {'Active' if self.scanpath_network else 'Inactive'}")

    def _create_painted_semantic_map(self, F_semantic_patches, raw_sam_pixel_segmap):
        B, C_dino, H_p, W_p = F_semantic_patches.shape
        _, H_img, W_img = raw_sam_pixel_segmap.shape
        device = F_semantic_patches.device
        dtype = F_semantic_patches.dtype
        S_painted_map_full_res_batch = torch.zeros(B, C_dino, H_img, W_img, device=device, dtype=dtype)

        for b_idx in range(B):
            img_patch_features = F_semantic_patches[b_idx]
            img_sam_pixel_segmap = raw_sam_pixel_segmap[b_idx]
            segmap_at_patch_res_float = F.interpolate(
                img_sam_pixel_segmap.unsqueeze(0).unsqueeze(0).float(),
                size=(H_p, W_p), mode='nearest'
            )
            segmap_at_patch_res = segmap_at_patch_res_float.squeeze(0).squeeze(0).long()
            flat_patch_features = img_patch_features.permute(1, 2, 0).reshape(H_p * W_p, C_dino)
            flat_patch_sam_ids = segmap_at_patch_res.reshape(H_p * W_p)
            clamped_flat_patch_sam_ids = torch.clamp(flat_patch_sam_ids, 0, self.num_sam_segments - 1)
            segment_avg_dino_features = scatter_mean(
                src=flat_patch_features, index=clamped_flat_patch_sam_ids,
                dim=0, dim_size=self.num_sam_segments
            )
            segment_avg_dino_features = torch.nan_to_num(segment_avg_dino_features, nan=0.0)
            clamped_pixel_sam_ids = torch.clamp(img_sam_pixel_segmap.long(), 0, self.num_sam_segments - 1)
            painted_slice = segment_avg_dino_features[clamped_pixel_sam_ids]
            S_painted_map_full_res_batch[b_idx] = painted_slice.permute(2, 0, 1)
        return S_painted_map_full_res_batch

    def forward(self, image, centerbias, x_hist=None, y_hist=None, durations=None,
                segmentation_mask=None, **kwargs):
        raw_sam_pixel_segmap_internal = segmentation_mask
        is_spade_dynamic_saliency = isinstance(self.saliency_network, SaliencyNetworkSPADEDynamic)
        is_spade_dynamic_scanpath = self.scanpath_network and isinstance(self.scanpath_network, SaliencyNetworkSPADEDynamic) # Assuming scanpath would also be SPADE dynamic

        if raw_sam_pixel_segmap_internal is None and (is_spade_dynamic_saliency or is_spade_dynamic_scanpath):
            raise ValueError(f"{self.__class__.__name__} requires 'segmentation_mask' for SPADEDynamic components.")

        orig_img_shape_hw = image.shape[2:]
        if self.downsample_input_to_backbone != 1:
            img_for_features = F.interpolate(image, scale_factor=1.0 / self.downsample_input_to_backbone,
                                             recompute_scale_factor=False, mode='bilinear', align_corners=False)
        else:
            img_for_features = image

        extracted_feature_maps = self.features(img_for_features)
        readout_h = math.ceil(orig_img_shape_hw[0] / self.downsample_input_to_backbone / self.readout_factor)
        readout_w = math.ceil(orig_img_shape_hw[1] / self.downsample_input_to_backbone / self.readout_factor)
        readout_spatial_shape = (readout_h, readout_w)

        processed_features_list = []
        for feat_map in extracted_feature_maps:
            processed_features_list.append(
                F.interpolate(feat_map, size=readout_spatial_shape, mode='bilinear', align_corners=False)
            )
        concatenated_backbone_features = torch.cat(processed_features_list, dim=1)

        # Create painted map only if needed by any SPADE dynamic component
        S_painted_map_full_res = None
        if is_spade_dynamic_saliency or is_spade_dynamic_scanpath:
            F_semantic_patches_from_dino = extracted_feature_maps[self.semantic_feature_layer_idx]
            S_painted_map_full_res = self._create_painted_semantic_map(F_semantic_patches_from_dino, raw_sam_pixel_segmap_internal)

        # Saliency Path
        if is_spade_dynamic_saliency:
            saliency_path_output = self.saliency_network(concatenated_backbone_features, S_painted_map_full_res)
        else: # Non-SPADE saliency network (if this class were to support it)
            saliency_path_output = self.saliency_network(concatenated_backbone_features)


        # Scanpath Path (currently placeholder, assumes spatial only for this iteration)
        scanpath_path_output = None
        if self.scanpath_network is not None:
            _logger.warning("Scanpath network is defined but its forward pass is not fully implemented in this DinoGazeSpade version.")
            # Example if scanpath were also SPADE dynamic:
            # scanpath_history_features = encode_scanpath_features(...) # This would need to be defined or imported
            # scanpath_path_output = self.scanpath_network(scanpath_history_features, S_painted_map_full_res)
            pass

        combined_input_for_fixsel = (saliency_path_output, scanpath_path_output) # scanpath_path_output is None here
        final_readout_before_finalizer = self.fixation_selection_network(combined_input_for_fixsel)
        saliency_log_density = self.finalizer(final_readout_before_finalizer, centerbias)
        return saliency_log_density

    def train(self, mode=True):
        if hasattr(self.features, 'eval'): self.features.eval() # Backbone always eval
        self.saliency_network.train(mode=mode)
        if self.scanpath_network is not None:
            self.scanpath_network.train(mode=mode)
        self.fixation_selection_network.train(mode=mode)
        self.finalizer.train(mode=mode)
        super().train(mode)


def main(args: argparse.Namespace):
    device, rank, world_size, is_master, is_distributed = init_distributed()

    log_level = logging.INFO if is_master else logging.WARNING
    if hasattr(args, 'log_level') and args.log_level is not None:
        log_level = getattr(logging, str(args.log_level).upper(), log_level)
    logging.basicConfig(level=log_level, format=f"%(asctime)s Rank{rank} %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", force=True)
    _logger.setLevel(log_level)
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)

    if is_master:
        _logger.info("================== Effective Configuration ==================")
        for arg_name, arg_value in sorted(vars(args).items()): _logger.info(f"  {arg_name}: {arg_value}")
        _logger.info(f"  DDP Info: Rank {rank}/{world_size}, Master: {is_master}, Distributed: {is_distributed}, Device: {device}")
        _logger.info(f"  Torch: {torch.__version__}, CUDA: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
        _logger.info(f"  Torch Scatter available: {'Yes' if 'scatter_mean' in globals() else 'No (CRITICAL ERROR IF NO)'}")
        _logger.info("===========================================================")

    dataset_directory = Path(args.dataset_dir).resolve()
    train_output_base_dir = Path(args.train_dir).resolve()
    lmdb_image_cache_dir = Path(args.lmdb_dir).resolve()
    # SAM mask root dir for individual files (used as fallback or primary if banks not specified)
    segmentation_mask_root_dir = Path(args.segmentation_mask_dir).resolve() if args.segmentation_mask_dir else None


    if is_master:
        for p_path in [dataset_directory, train_output_base_dir, lmdb_image_cache_dir]:
            if p_path: p_path.mkdir(parents=True, exist_ok=True)
        if segmentation_mask_root_dir and not segmentation_mask_root_dir.exists():
             _logger.warning(f"Individual SAM mask root dir {segmentation_mask_root_dir} does not exist.")
    if is_distributed: dist.barrier()

    if is_master: _logger.info(f"Initializing {args.dinov2_model_name} backbone...")
    dino_backbone = DinoV2Backbone(
        layers=args.dinov2_layers_for_main_path, model_name=args.dinov2_model_name,
        patch_size=args.dinov2_patch_size, freeze=True # Backbone is always frozen for these experiments
    ).to(device)
    dino_backbone.eval() # Ensure backbone is in eval mode
    if is_master: _logger.info(f"{args.dinov2_model_name} (hooks: {args.dinov2_layers_for_main_path}) initialized and frozen.")

    C_dino_embed_dim = dino_backbone.num_channels
    main_path_concatenated_channels = len(args.dinov2_layers_for_main_path) * C_dino_embed_dim
    semantic_feature_channels_for_spade = C_dino_embed_dim # Semantic features are from one DINO layer

    if is_master:
        _logger.info(f"DINO Channels: Single layer output (C_dino) = {C_dino_embed_dim}")
        _logger.info(f"Main Path: Concatenated input channels to SPADE head = {main_path_concatenated_channels}")
        _logger.info(f"SPADE Modulation: Semantic feature channels (from one DINO layer) = {semantic_feature_channels_for_spade}")
        _logger.info(f"SPADE Modulation: Semantic features taken from DINO layer index {args.dinov2_semantic_feature_layer_idx} (actual positive index used in model)")


    # --- Stage Dispatch ---
    if args.stage == 'salicon_pretraining':
        if is_master: _logger.info(f"--- Preparing SALICON Pretraining with DINOv2+SPADE(DynamicSAM)+Scatter ---")
        current_lr = args.lr
        current_milestones = args.lr_milestones
        output_dir_stage_relative = f"{args.stage}_{args.dinov2_model_name}_spade_dynSAMscatter{args.num_total_sam_segments}_lr{args.lr}"
        
        salicon_loc = dataset_directory / 'SALICON'
        if is_master:
            try:
                if not (salicon_loc / 'stimuli' / 'train').exists(): pysaliency.get_SALICON_train(location=str(salicon_loc.parent))
                if not (salicon_loc / 'stimuli' / 'val').exists(): pysaliency.get_SALICON_val(location=str(salicon_loc.parent))
            except Exception as e: _logger.critical(f"Failed SALICON get: {e}"); dist.barrier(); sys.exit(1)
        if is_distributed: dist.barrier()
        SALICON_train_stimuli, SALICON_train_fixations = pysaliency.get_SALICON_train(location=str(salicon_loc.parent))
        SALICON_val_stimuli, SALICON_val_fixations = pysaliency.get_SALICON_val(location=str(salicon_loc.parent))

        SALICON_centerbias = BaselineModel(stimuli=SALICON_train_stimuli, fixations=SALICON_train_fixations, bandwidth=0.0217, eps=2e-13, caching=False)
        
        train_ll_cache_file = dataset_directory / f'salicon_baseline_train_ll_dinogaze_{args.dinov2_model_name}.pkl' # Make cache name specific
        val_ll_cache_file = dataset_directory / f'salicon_baseline_val_ll_dinogaze_{args.dinov2_model_name}.pkl'
        train_baseline_log_likelihood, val_baseline_log_likelihood = None, None

        if is_master: # Master computes or loads LLs
            # ... (same LL loading/computation logic as before) ...
            try:
                with open(train_ll_cache_file,'rb') as f: train_baseline_log_likelihood=cpickle.load(f)
                _logger.info(f"Loaded TRAIN LL from {train_ll_cache_file}")
            except:
                _logger.warning(f"TRAIN LL cache miss for {train_ll_cache_file}. Computing...");
                train_baseline_log_likelihood=SALICON_centerbias.information_gain(SALICON_train_stimuli, SALICON_train_fixations, verbose=True,average='image')
                with open(train_ll_cache_file,'wb') as f: cpickle.dump(train_baseline_log_likelihood,f)
            try:
                with open(val_ll_cache_file,'rb') as f: val_baseline_log_likelihood=cpickle.load(f)
                _logger.info(f"Loaded VAL LL from {val_ll_cache_file}")
            except:
                _logger.warning(f"VAL LL cache miss for {val_ll_cache_file}. Computing...");
                val_baseline_log_likelihood=SALICON_centerbias.information_gain(SALICON_val_stimuli, SALICON_val_fixations, verbose=True,average='image')
                with open(val_ll_cache_file,'wb') as f: cpickle.dump(val_baseline_log_likelihood,f)
            _logger.info(f"Master Baseline LLs - Train: {train_baseline_log_likelihood or float('nan'):.5f}, Val: {val_baseline_log_likelihood or float('nan'):.5f}")

        ll_bcast=[train_baseline_log_likelihood, val_baseline_log_likelihood] # Broadcast to other ranks
        if is_distributed:
            if is_master and (train_baseline_log_likelihood is None or val_baseline_log_likelihood is None): ll_bcast = [np.nan, np.nan]
            dist.broadcast_object_list(ll_bcast,src=0)
        train_baseline_log_likelihood,val_baseline_log_likelihood = ll_bcast
        if np.isnan(train_baseline_log_likelihood) or np.isnan(val_baseline_log_likelihood):
            _logger.critical(f"NaN LLs received/computed on rank {rank}. Exiting."); sys.exit(1)
        else: _logger.info(f"Rank {rank} Baseline LLs - Train: {train_baseline_log_likelihood:.5f}, Val: {val_baseline_log_likelihood:.5f}")

        # --- Model Instantiation for SALICON ---
        saliency_net_spade_dynamic = SaliencyNetworkSPADEDynamic(
            input_channels_main_path=main_path_concatenated_channels,
            semantic_feature_channels_for_spade=semantic_feature_channels_for_spade
        )
        # For spatial-only model, scanpath_features is 0
        fixsel_net = build_fixation_selection_network(scanpath_features=0)

        model = DinoGazeSpade(
            features_module=dino_backbone, # Already on device
            saliency_network=saliency_net_spade_dynamic,
            scanpath_network=None, # Explicitly None for SALICON pretraining (spatial only)
            fixation_selection_network=fixsel_net,
            dinov2_patch_size=args.dinov2_patch_size,
            semantic_feature_layer_idx=args.dinov2_semantic_feature_layer_idx,
            num_sam_segments=args.num_total_sam_segments,
            downsample_input_to_backbone=1, # Default, from DINOgaze
            readout_factor=args.dinov2_patch_size, # DINOgaze uses patch_size as readout_factor
            saliency_map_factor_finalizer=4, # Common value
            initial_sigma=args.finalizer_initial_sigma
        ).to(device)

        if is_master: _logger.info("DinoGazeSpade (Dynamic SAM) model built for SALICON.")
        if is_distributed:
            # find_unused_parameters=True because scanpath_network is None, so its params in fixsel_net won't get grads.
            # Or, if fixsel_net adapts to scanpath_features=0 and has no unused scanpath params, this can be False.
            # Given build_fixation_selection_network creates network based on scanpath_features, find_unused might be False if scanpath_features=0
            # To be safe for now, let's use True, but False might be possible if fixsel_net is lean.
            model = DDP(model, device_ids=[device.index], find_unused_parameters=True) # Scanpath is None
            if is_master: _logger.info("Wrapped model with DDP.")

        head_params = [p for p in model.parameters() if p.requires_grad]
        if not head_params: _logger.critical("No trainable parameters found!"); sys.exit(1)
        optimizer = optim.Adam(head_params, lr=current_lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=current_milestones)

        # --- Dataloaders for SALICON ---
        # Mask paths (individual file fallback)
        train_mask_individual_dir = segmentation_mask_root_dir / args.train_mask_subdir_name if segmentation_mask_root_dir and args.train_mask_subdir_name else None
        val_mask_individual_dir = segmentation_mask_root_dir / args.val_mask_subdir_name if segmentation_mask_root_dir and args.val_mask_subdir_name else None
        # Mask paths (banks)
        train_fixed_memmap = Path(args.train_mask_memmap_file).resolve() if args.train_mask_memmap_file else None
        val_fixed_memmap = Path(args.val_mask_memmap_file).resolve() if args.val_mask_memmap_file else None
        train_var_payload = Path(args.train_mask_variable_payload_file).resolve() if args.train_mask_variable_payload_file else None
        train_var_header = Path(args.train_mask_variable_header_file).resolve() if args.train_mask_variable_header_file else None
        val_var_payload = Path(args.val_mask_variable_payload_file).resolve() if args.val_mask_variable_payload_file else None
        val_var_header = Path(args.val_mask_variable_header_file).resolve() if args.val_mask_variable_header_file else None

        if is_master:
            _logger.info(f"Train SAM Mask Config: FallbackDir='{train_mask_individual_dir}', FixedBank='{train_fixed_memmap}', VarBankPayload='{train_var_payload}'")
            _logger.info(f"Val SAM Mask Config: FallbackDir='{val_mask_individual_dir}', FixedBank='{val_fixed_memmap}', VarBankPayload='{val_var_payload}'")

        dataset_kwargs_train = {
            "transform": FixationMaskTransform(sparse=False), "average": "image",
            "lmdb_path": str(lmdb_image_cache_dir / f'SALICON_train_imgs_dinogaze_{args.dinov2_model_name}') if args.use_lmdb_images else None,
            "segmentation_mask_dir": train_mask_individual_dir, # For individual SAM masks
            "segmentation_mask_format": args.segmentation_mask_format,
            "segmentation_mask_fixed_memmap_file": train_fixed_memmap, # For fixed banks
            "segmentation_mask_variable_payload_file": train_var_payload, # For variable banks
            "segmentation_mask_variable_header_file": train_var_header,
            "segmentation_mask_bank_dtype": args.segmentation_mask_bank_dtype
        }
        train_dataset = ImageDatasetWithSegmentation(stimuli=SALICON_train_stimuli, fixations=SALICON_train_fixations, centerbias_model=SALICON_centerbias, **dataset_kwargs_train)
        
        dataset_kwargs_val = { # Similar for validation
            "transform": FixationMaskTransform(sparse=False), "average": "image",
            "lmdb_path": str(lmdb_image_cache_dir / f'SALICON_val_imgs_dinogaze_{args.dinov2_model_name}') if args.use_lmdb_images else None,
            "segmentation_mask_dir": val_mask_individual_dir,
            "segmentation_mask_format": args.segmentation_mask_format,
            "segmentation_mask_fixed_memmap_file": val_fixed_memmap,
            "segmentation_mask_variable_payload_file": val_var_payload,
            "segmentation_mask_variable_header_file": val_var_header,
            "segmentation_mask_bank_dtype": args.segmentation_mask_bank_dtype
        }
        val_dataset = ImageDatasetWithSegmentation(stimuli=SALICON_val_stimuli, fixations=SALICON_val_fixations, centerbias_model=SALICON_centerbias, **dataset_kwargs_val)

        train_sampler = (torch.utils.data.DistributedSampler(train_dataset, shuffle=True, drop_last=True) if is_distributed else None)
        if train_sampler and hasattr(train_sampler, 'set_epoch'): train_sampler.set_epoch(0) # Initial epoch for DDP sampler

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                                 num_workers=args.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True,
                                                 persistent_workers=args.num_workers > 0)
        val_sampler = (torch.utils.data.DistributedSampler(val_dataset, shuffle=False, drop_last=False) if is_distributed else None)
        validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                                     num_workers=args.num_workers, pin_memory=True, sampler=val_sampler,
                                                     persistent_workers=args.num_workers > 0)
        if is_master: 
            _logger.info(f"Train Dataloader: {len(train_loader)} batches. Val Dataloader: {len(validation_loader)} batches.")
            # ... (sample batch inspection logic from original script) ...
            if not train_dataset or len(train_dataset) == 0: _logger.error("Train dataset is empty!"); sys.exit(1)
            else:
                try:
                    sample_batch = next(iter(train_loader))
                    _logger.info(f"Sample batch keys: {sample_batch.keys()}")
                    if 'segmentation_mask' not in sample_batch: _logger.error("CRITICAL: 'segmentation_mask' not in batch."); sys.exit(1)
                    if sample_batch['segmentation_mask'].max() >= args.num_total_sam_segments:
                         _logger.warning(f"WARNING: Max SAM ID in batch ({sample_batch['segmentation_mask'].max()}) >= num_total_sam_segments ({args.num_total_sam_segments}).")
                except Exception as e_batch: _logger.error(f"Error inspecting sample batch: {e_batch}", exc_info=True); sys.exit(1)


        output_dir_experiment = train_output_base_dir / output_dir_stage_relative
        if is_master: _logger.info(f"Experiment output to: {output_dir_experiment}")

        _train(
            this_directory=str(output_dir_experiment), model=model,
            train_loader=train_loader, train_baseline_log_likelihood=train_baseline_log_likelihood,
            val_loader=validation_loader, val_baseline_log_likelihood=val_baseline_log_likelihood,
            optimizer=optimizer, lr_scheduler=lr_scheduler,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            minimum_learning_rate=args.min_lr,
            validation_metrics=['LL', 'IG', 'NSS', 'AUC_CPU'], # Assuming AUC_CPU is desired
            validation_epochs=args.validation_epochs,
            startwith=args.resume_checkpoint, # General resume, _train handles specific step
            device=device, is_distributed=is_distributed, is_master=is_master, logger=_logger,
        )
        if is_master: _logger.info(f"--- Stage '{args.stage}' Finished ---")

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++ MIT STAGES FOR SPATIAL FINE-TUNING OF DinoGazeSpade +++
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    elif args.stage in ['mit_spatial']:
        fold = args.fold
        if fold is None or not (0 <= fold < 10):
            _logger.critical("--fold required for MIT stages and must be 0-9."); sys.exit(1)
        
        if is_master: _logger.info(f"--- Preparing MIT1003 Stage: {args.stage} (Fold {fold}) ---")
        current_lr = args.lr_mit_spatial # Use MIT specific LR
        current_milestones = args.lr_milestones_mit_spatial
        
        # Define previous stage dir for loading checkpoint
        # Assumes SALICON pretraining stage name is 'salicon_pretraining'
        prev_stage_name = f"salicon_pretraining_{args.dinov2_model_name}_spade_dynSAMscatter{args.num_total_sam_segments}_lr{args.lr}"
        prev_stage_checkpoint_dir = train_output_base_dir / prev_stage_name
        
        output_dir_stage_relative = f"{args.stage}_{args.dinov2_model_name}_fold{fold}_dynSAMscatter{args.num_total_sam_segments}_lr{current_lr}"
        output_dir_experiment = train_output_base_dir / output_dir_stage_relative

        # --- MIT Data Handling (Conversion, Splits, Baseline) ---
        mit_converted_stimuli_path = train_output_base_dir / f'MIT1003_converted_dinogaze_sam_{args.dinov2_model_name}'
        mit_stimuli_file_cache = mit_converted_stimuli_path / "stimuli.pkl"
        mit_scanpaths_file_cache = mit_converted_stimuli_path / "scanpaths.pkl" # Not used for spatial, but for completeness

        needs_conversion = True
        if mit_stimuli_file_cache.exists(): # Only check stimuli for spatial
            if is_master: _logger.info("Found cached converted MIT1003 stimuli.")
            needs_conversion = False
        
        if is_distributed:
            needs_conversion_tensor = torch.tensor(int(needs_conversion), device=device, dtype=torch.int)
            dist.broadcast(needs_conversion_tensor, src=0)
            needs_conversion = bool(needs_conversion_tensor.item())

        mit_stimuli_twosize, mit_scanpaths_twosize_all = None, None # scanpaths only for full dataset processing
        if needs_conversion:
            if is_master:
                _logger.info(f"Converting MIT1003 data. Original from: {dataset_directory}, Processed to: {mit_converted_stimuli_path}")
                mit_converted_stimuli_path.mkdir(parents=True, exist_ok=True)
                try:
                    pysaliency.external_datasets.mit.get_mit1003_with_initial_fixation(location=str(dataset_directory), replace_initial_invalid_fixations=True)
                except ImportError: _logger.error("pysaliency.external_datasets.mit module not found."); dist.barrier(); sys.exit(1)
                except Exception as e: _logger.critical(f"Failed to get original MIT1003: {e}"); dist.barrier(); sys.exit(1)
            if is_distributed: dist.barrier()
            mit_stimuli_orig, mit_scanpaths_orig = pysaliency.external_datasets.mit.get_mit1003_with_initial_fixation(location=str(dataset_directory), replace_initial_invalid_fixations=True)
            
            # Use the conversion functions from src.data
            mit_stimuli_twosize = convert_stimuli_mit(mit_stimuli_orig, mit_converted_stimuli_path, is_master, is_distributed, device, _logger)
            # mit_scanpaths_twosize_all = convert_fixation_trains_mit(mit_stimuli_orig, mit_scanpaths_orig, is_master, _logger) # If needed for scanpath later
            
            if mit_stimuli_twosize is None: _logger.critical("MIT1003 stimuli conversion failed."); sys.exit(1)
            if is_master:
                try:
                    with atomic_save(str(mit_stimuli_file_cache), text_mode=False, overwrite_part=True) as f: pickle.dump(mit_stimuli_twosize, f)
                    # if mit_scanpaths_twosize_all:
                    #    with atomic_save(str(mit_scanpaths_file_cache), text_mode=False, overwrite_part=True) as f: cpickle.dump(mit_scanpaths_twosize_all, f)
                    _logger.info(f"Saved converted MIT1003 stimuli to {mit_converted_stimuli_path}")
                except Exception as e: _logger.error(f"Failed to save converted MIT data: {e}")
            if is_distributed: dist.barrier()
        else: # Load from cache
            if is_master: _logger.info(f"Loading pre-converted MIT1003 stimuli from {mit_converted_stimuli_path}")
            try:
                with open(mit_stimuli_file_cache, "rb") as f: mit_stimuli_twosize = pickle.load(f)
                # if mit_scanpaths_file_cache.exists(): # Optionally load scanpaths if needed later
                #    with open(mit_scanpaths_file_cache, "rb") as f: mit_scanpaths_twosize_all = cpickle.load(f)
            except Exception as e: _logger.critical(f"Failed to load cached converted MIT stimuli: {e}"); sys.exit(1)
        
        # For spatial model, we need Fixations, not FixationTrains, for CrossvalidatedBaselineModel and ImageDataset.
        # If mit_scanpaths_twosize_all was loaded, convert. If not, need to load original scanpaths again.
        if not 'mit_scanpaths_orig' in locals(): # If not loaded during conversion step
            _, mit_scanpaths_orig = pysaliency.external_datasets.mit.get_mit1003_with_initial_fixation(location=str(dataset_directory), replace_initial_invalid_fixations=True)
        mit_fixations_flat_all = mit_scanpaths_orig.to_fixations() # Convert all original scanpaths to flat fixations

        if is_master: _logger.info("Initializing MIT1003 CrossvalidatedBaselineModel for centerbias...")
        MIT1003_centerbias = CrossvalidatedBaselineModel(
            mit_stimuli_twosize, # Use converted stimuli for sizes
            mit_fixations_flat_all, # Use flat fixations for baseline model training
            bandwidth=10**-1.6667673342543432, eps=10**-14.884189168516073, caching=False
        )

        MIT1003_stimuli_train, mit_fixations_train_flat = train_split(mit_stimuli_twosize, mit_fixations_flat_all, crossval_folds=10, fold_no=fold)
        MIT1003_stimuli_val, mit_fixations_val_flat = validation_split(mit_stimuli_twosize, mit_fixations_flat_all, crossval_folds=10, fold_no=fold)

        train_baseline_log_likelihood, val_baseline_log_likelihood = None, None
        if is_master: # Master computes or loads LLs for the fold
            _logger.info(f"Computing baseline LLs for MIT1003 Fold {fold}...")
            try:
                train_baseline_log_likelihood = MIT1003_centerbias.information_gain(MIT1003_stimuli_train, mit_fixations_train_flat, verbose=False, average='image')
                val_baseline_log_likelihood = MIT1003_centerbias.information_gain(MIT1003_stimuli_val, mit_fixations_val_flat, verbose=False, average='image')
                _logger.info(f"Fold {fold} Master Baseline LLs - Train: {train_baseline_log_likelihood:.5f}, Val: {val_baseline_log_likelihood:.5f}")
            except Exception as e:
                _logger.critical(f"Failed to compute MIT baseline LLs: {e}"); train_baseline_log_likelihood, val_baseline_log_likelihood = np.nan, np.nan
        
        ll_bcast_mit = [train_baseline_log_likelihood, val_baseline_log_likelihood]
        if is_distributed:
            if is_master and (train_baseline_log_likelihood is None or val_baseline_log_likelihood is None): ll_bcast_mit = [np.nan, np.nan]
            dist.broadcast_object_list(ll_bcast_mit, src=0)
        train_baseline_log_likelihood, val_baseline_log_likelihood = ll_bcast_mit
        if np.isnan(train_baseline_log_likelihood) or np.isnan(val_baseline_log_likelihood):
            _logger.critical(f"MIT Baseline LLs invalid on rank {rank}. Exiting."); sys.exit(1)
        _logger.info(f"Rank {rank} MIT Fold {fold} Baseline LLs - Train: {train_baseline_log_likelihood:.5f}, Val: {val_baseline_log_likelihood:.5f}")
        
        # --- Model Build & Load Checkpoint for MIT stage ---
        saliency_net_spade_dynamic_mit = SaliencyNetworkSPADEDynamic(
            input_channels_main_path=main_path_concatenated_channels,
            semantic_feature_channels_for_spade=semantic_feature_channels_for_spade
        )
        fixsel_net_mit = build_fixation_selection_network(scanpath_features=0) # Spatial only

        # Build model on CPU first for checkpoint loading
        model_cpu = DinoGazeSpade(
            features_module=dino_backbone.cpu(), # Move backbone to CPU temporarily
            saliency_network=saliency_net_spade_dynamic_mit,
            scanpath_network=None,
            fixation_selection_network=fixsel_net_mit,
            dinov2_patch_size=args.dinov2_patch_size,
            semantic_feature_layer_idx=args.dinov2_semantic_feature_layer_idx,
            num_sam_segments=args.num_total_sam_segments,
            readout_factor=args.dinov2_patch_size,
            initial_sigma=args.finalizer_initial_sigma
        )
        dino_backbone.to(device) # Move backbone back to device

        start_checkpoint_path_mit = None
        if args.resume_checkpoint_mit: # Specific resume for MIT stage
             start_checkpoint_path_mit = Path(args.resume_checkpoint_mit)
        elif prev_stage_checkpoint_dir:
            chkpt_options = [prev_stage_checkpoint_dir / 'final_best_val.pth', prev_stage_checkpoint_dir / 'final.pth']
            for p_opt in chkpt_options:
                if p_opt.exists(): start_checkpoint_path_mit = p_opt; break
        
        if start_checkpoint_path_mit and start_checkpoint_path_mit.exists():
            if is_master: _logger.info(f"Loading checkpoint for MIT model_cpu from: {start_checkpoint_path_mit}")
            state_dict = torch.load(start_checkpoint_path_mit, map_location='cpu', weights_only=False)
            if 'model_state_dict' in state_dict: state_dict = state_dict['model_state_dict']
            new_state_dict = OrderedDict()
            for k, v in state_dict.items(): new_state_dict[k[7:] if k.startswith('module.') else k] = v
            missing, unexpected = model_cpu.load_state_dict(new_state_dict, strict=False)
            if is_master:
                if missing: _logger.warning(f"MIT Load: Missing keys: {missing}")
                if unexpected: _logger.warning(f"MIT Load: Unexpected keys: {unexpected}")
        elif args.stage != 'salicon_pretraining': # Don't warn if SALICON stage has no explicit resume
            _logger.warning(f"No checkpoint found from previous stage ({prev_stage_checkpoint_dir}) or via --resume_checkpoint_mit. Starting MIT fine-tuning with fresh head weights.")
        
        model = model_cpu.to(device)

        if is_distributed:
            model = DDP(model, device_ids=[device.index], find_unused_parameters=True) # Scanpath is None
            if is_master: _logger.info("Wrapped MIT model with DDP.")

        head_params_mit = [p for p in model.parameters() if p.requires_grad]
        if not head_params_mit: _logger.critical("No trainable parameters found for MIT stage!"); sys.exit(1)
        optimizer = optim.Adam(head_params_mit, lr=current_lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=current_milestones)

        # --- Dataloaders for MIT Stage ---
        # Assume SAM masks for MIT are organized similarly or use specific bank files for MIT.
        # Update these paths in YAML or via CLI arguments (e.g., args.train_mask_subdir_name_mit)
        train_mask_individual_dir_mit = segmentation_mask_root_dir / args.train_mask_subdir_name_mit if segmentation_mask_root_dir and args.train_mask_subdir_name_mit else None
        val_mask_individual_dir_mit = segmentation_mask_root_dir / args.val_mask_subdir_name_mit if segmentation_mask_root_dir and args.val_mask_subdir_name_mit else None
        train_fixed_memmap_mit = Path(args.train_mask_memmap_file_mit).resolve() if args.train_mask_memmap_file_mit else None
        # ... and so on for other MIT mask bank paths ...
        # For simplicity, let's reuse the main mask args if specific MIT ones aren't defined,
        # but ideally, you'd have separate args for MIT SAM masks.
        # Here, I'll assume you'll configure the YAML/CLI to point to MIT SAM masks using the existing mask args.
        _logger.warning("Using SALICON mask configuration for MIT stage. Ensure these paths point to MIT SAM masks if different.")

        dataset_kwargs_train_mit = {
            "transform": FixationMaskTransform(sparse=False), "average": "image",
            "lmdb_path": str(lmdb_image_cache_dir / f'MIT1003_train_imgs_dinogaze_fold{fold}_{args.dinov2_model_name}') if args.use_lmdb_images else None,
            "segmentation_mask_dir": train_mask_individual_dir_mit or (segmentation_mask_root_dir / args.train_mask_subdir_name if segmentation_mask_root_dir else None),
            "segmentation_mask_format": args.segmentation_mask_format,
            "segmentation_mask_fixed_memmap_file": train_fixed_memmap_mit or (Path(args.train_mask_memmap_file).resolve() if args.train_mask_memmap_file else None),
            "segmentation_mask_variable_payload_file": Path(args.train_mask_variable_payload_file_mit).resolve() if args.train_mask_variable_payload_file_mit else (Path(args.train_mask_variable_payload_file).resolve() if args.train_mask_variable_payload_file else None),
            "segmentation_mask_variable_header_file": Path(args.train_mask_variable_header_file_mit).resolve() if args.train_mask_variable_header_file_mit else (Path(args.train_mask_variable_header_file).resolve() if args.train_mask_variable_header_file else None),
            "segmentation_mask_bank_dtype": args.segmentation_mask_bank_dtype
        }
        train_dataset_mit = ImageDatasetWithSegmentation(stimuli=MIT1003_stimuli_train, fixations=mit_fixations_train_flat, centerbias_model=MIT1003_centerbias, **dataset_kwargs_train_mit)
        
        dataset_kwargs_val_mit = { # Similar for validation
             "transform": FixationMaskTransform(sparse=False), "average": "image",
            "lmdb_path": str(lmdb_image_cache_dir / f'MIT1003_val_imgs_dinogaze_fold{fold}_{args.dinov2_model_name}') if args.use_lmdb_images else None,
            "segmentation_mask_dir": val_mask_individual_dir_mit or (segmentation_mask_root_dir / args.val_mask_subdir_name if segmentation_mask_root_dir else None),
            "segmentation_mask_format": args.segmentation_mask_format,
            "segmentation_mask_fixed_memmap_file": Path(args.val_mask_memmap_file_mit).resolve() if args.val_mask_memmap_file_mit else (Path(args.val_mask_memmap_file).resolve() if args.val_mask_memmap_file else None),
            "segmentation_mask_variable_payload_file": Path(args.val_mask_variable_payload_file_mit).resolve() if args.val_mask_variable_payload_file_mit else (Path(args.val_mask_variable_payload_file).resolve() if args.val_mask_variable_payload_file else None),
            "segmentation_mask_variable_header_file": Path(args.val_mask_variable_header_file_mit).resolve() if args.val_mask_variable_header_file_mit else (Path(args.val_mask_variable_header_file).resolve() if args.val_mask_variable_header_file else None),
            "segmentation_mask_bank_dtype": args.segmentation_mask_bank_dtype
        }
        val_dataset_mit = ImageDatasetWithSegmentation(stimuli=MIT1003_stimuli_val, fixations=mit_fixations_val_flat, centerbias_model=MIT1003_centerbias, **dataset_kwargs_val_mit)

        train_sampler_mit = (torch.utils.data.DistributedSampler(train_dataset_mit, shuffle=True, drop_last=True) if is_distributed else None)
        if train_sampler_mit and hasattr(train_sampler_mit, 'set_epoch'): train_sampler_mit.set_epoch(0)
        train_loader_mit = DataLoader(train_dataset_mit, batch_size=args.batch_size, shuffle=(train_sampler_mit is None),
                                   num_workers=args.num_workers, pin_memory=True, sampler=train_sampler_mit, drop_last=True,
                                   persistent_workers=args.num_workers > 0)
        
        val_sampler_mit = (torch.utils.data.DistributedSampler(val_dataset_mit, shuffle=False, drop_last=False) if is_distributed else None)
        validation_loader_mit = DataLoader(val_dataset_mit, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers, pin_memory=True, sampler=val_sampler_mit,
                                        persistent_workers=args.num_workers > 0)
        if is_master: 
            _logger.info(f"MIT Train Dataloader: {len(train_loader_mit)} batches. MIT Val Dataloader: {len(validation_loader_mit)} batches.")
            # ... (sample batch inspection for MIT) ...
        
        if is_master: _logger.info(f"Experiment output for MIT stage to: {output_dir_experiment}")
        _train(
            this_directory=str(output_dir_experiment), model=model,
            train_loader=train_loader_mit, train_baseline_log_likelihood=train_baseline_log_likelihood,
            val_loader=validation_loader_mit, val_baseline_log_likelihood=val_baseline_log_likelihood,
            optimizer=optimizer, lr_scheduler=lr_scheduler,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            minimum_learning_rate=args.min_lr, # Use general min_lr
            validation_metrics=['LL', 'IG', 'NSS', 'AUC_CPU'],
            validation_epochs=args.validation_epochs,
            startwith=None, # _train handles resumption from output_dir_experiment
            device=device, is_distributed=is_distributed, is_master=is_master, logger=_logger,
        )
        if is_master: _logger.info(f"--- MIT Stage '{args.stage}' (Fold {fold}) Finished ---")

    else:
        _logger.critical(f"Unknown or unsupported stage: {args.stage}"); sys.exit(1)

    cleanup_distributed()
    if is_master: _logger.info("Training script finished successfully.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    _module_logger = logging.getLogger(__name__)

    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument('--config_file', type=str, default=None)
    _cfg_namespace, _remaining_cli_args = _pre.parse_known_args()

    parser = argparse.ArgumentParser(parents=[_pre], description="Train DinoGaze+SPADE(DynamicSAM) with torch_scatter", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # --- Stage and General Config ---
    parser.add_argument('--stage', default='salicon_pretraining',
                        choices=['salicon_pretraining', 'mit_spatial'], # Added MIT stage
                        help='Training stage to execute.')
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])

    # --- DINOv2 Backbone ---
    parser.add_argument('--dinov2_model_name', default='dinov2_vitb14', choices=['dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'])
    parser.add_argument('--dinov2_patch_size', type=int, default=14)
    parser.add_argument('--dinov2_layers_for_main_path', type=int, nargs='+', default=[-3, -2, -1])
    parser.add_argument('--dinov2_semantic_feature_layer_idx', type=int, default=-1, help="Index from DINO layers for SPADE semantic features (0 to N-1, or negative from end).")

    # --- SAM Masks and SPADE ---
    parser.add_argument('--segmentation_mask_dir', type=str, help='Root directory for individual SAM masks (fallback).')
    parser.add_argument('--segmentation_mask_format', default='png', choices=['png', 'npy'])
    parser.add_argument('--num_total_sam_segments', type=int, default=64, help="Number of distinct SAM segment IDs expected (e.g., K if K-Means was used, or max ID + 1).")
    parser.add_argument('--segmentation_mask_bank_dtype', type=str, default='uint8', choices=['uint8', 'uint16'])

    # --- Training Hyperparameters (SALICON) ---
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4, help="LR for SALICON pretraining.")
    parser.add_argument('--lr_milestones', type=int, nargs='+', default=[20, 40, 55], help="LR milestones for SALICON.")
    parser.add_argument('--min_lr', type=float, default=1e-7, help="Minimum learning rate for any stage.")
    parser.add_argument('--validation_epochs', type=int, default=1)
    parser.add_argument('--resume_checkpoint', type=str, help='Path to checkpoint for SALICON stage resumption or explicit start.')
    parser.add_argument('--finalizer_initial_sigma', type=float, default=8.0)

    # --- Training Hyperparameters (MIT Fine-tuning) ---
    parser.add_argument('--fold', type=int, help='Cross-validation fold for MIT stages (0-9).', default=0)
    parser.add_argument('--lr_mit_spatial', type=float, default=5e-5, help="LR for MIT spatial fine-tuning.") # Often lower for fine-tuning
    parser.add_argument('--lr_milestones_mit_spatial', type=int, nargs='+', default=[10, 20], help="LR milestones for MIT spatial.")
    parser.add_argument('--resume_checkpoint_mit', type=str, help='Path to checkpoint for MIT stage resumption (overrides SALICON flow).')


    # --- Dataloading & System ---
    parser.add_argument('--num_workers', type=str, default='auto')
    parser.add_argument('--train_dir', type=str, default='./experiments_dinogaze_spade_sam_scatter_vitb14')
    parser.add_argument('--dataset_dir', type=str, default='./data/pysaliency_datasets')
    parser.add_argument('--lmdb_dir', type=str, default='./data/lmdb_caches_dinogaze_spade')
    parser.add_argument('--use_lmdb_images', action=argparse.BooleanOptionalAction, default=True)

    # --- SAM Mask Paths (SALICON) ---
    parser.add_argument('--train_mask_subdir_name', type=str, default='train_sam64', help="Subdir for SALICON train SAM masks (if using individual files).")
    parser.add_argument('--val_mask_subdir_name', type=str, default='val_sam64', help="Subdir for SALICON val SAM masks.")
    parser.add_argument('--train_mask_memmap_file', type=str, help="Fixed-size memmap bank (.npy) for TRAIN SALICON SAM masks.")
    parser.add_argument('--val_mask_memmap_file', type=str, help="Fixed-size memmap bank (.npy) for VAL SALICON SAM masks.")
    parser.add_argument('--train_mask_variable_payload_file', type=str, help="Variable-size mask PAYLOAD (.bin) for TRAIN SALICON SAM.")
    parser.add_argument('--train_mask_variable_header_file', type=str, help="Companion HEADER (.npy) for TRAIN SALICON SAM variable payload.")
    parser.add_argument('--val_mask_variable_payload_file', type=str, help="Variable-size mask PAYLOAD (.bin) for VAL SALICON SAM.")
    parser.add_argument('--val_mask_variable_header_file', type=str, help="Companion HEADER (.npy) for VAL SALICON SAM variable payload.")

    # --- SAM Mask Paths (MIT) ---
    parser.add_argument('--train_mask_subdir_name_mit', type=str, default='train_sam64_mit', help="Subdir for MIT train SAM masks.")
    parser.add_argument('--val_mask_subdir_name_mit', type=str, default='val_sam64_mit', help="Subdir for MIT val SAM masks.")
    parser.add_argument('--train_mask_memmap_file_mit', type=str, help="Fixed-size memmap bank for TRAIN MIT SAM masks.")
    parser.add_argument('--val_mask_memmap_file_mit', type=str, help="Fixed-size memmap bank for VAL MIT SAM masks.")
    parser.add_argument('--train_mask_variable_payload_file_mit', type=str, help="Variable PAYLOAD for TRAIN MIT SAM.")
    parser.add_argument('--train_mask_variable_header_file_mit', type=str, help="Variable HEADER for TRAIN MIT SAM.")
    parser.add_argument('--val_mask_variable_payload_file_mit', type=str, help="Variable PAYLOAD for VAL MIT SAM.")
    parser.add_argument('--val_mask_variable_header_file_mit', type=str, help="Variable HEADER for VAL MIT SAM.")


    if _cfg_namespace.config_file:
        try:
            with open(_cfg_namespace.config_file, 'r') as f: yaml_cfg = yaml.safe_load(f) or {}
            _module_logger.info(f"Loaded YAML config from: {_cfg_namespace.config_file}")
            parser.set_defaults(**yaml_cfg) # YAML values become defaults
        except Exception as e: _module_logger.warning(f"Could not read/parse YAML '{_cfg_namespace.config_file}': {e}")

    final_args_ns = parser.parse_args(_remaining_cli_args) # CLI overrides YAML

    # Post-processing num_workers
    ws_env = int(os.environ.get("WORLD_SIZE", 1))
    if isinstance(final_args_ns.num_workers, str) and final_args_ns.num_workers.lower() == 'auto':
        final_args_ns.num_workers = None
    if final_args_ns.num_workers is None:
        try: cpu_c = len(os.sched_getaffinity(0))
        except AttributeError: cpu_c = os.cpu_count() or 1
        final_args_ns.num_workers = max(0, cpu_c // ws_env if ws_env > 0 else cpu_c)
        if final_args_ns.num_workers == 0 and ws_env > 1 and cpu_c > ws_env: final_args_ns.num_workers = 1
    else:
        try: final_args_ns.num_workers = int(final_args_ns.num_workers)
        except ValueError: _module_logger.warning(f"Invalid num_workers='{final_args_ns.num_workers}', using 0."); final_args_ns.num_workers = 0
        if final_args_ns.num_workers < 0: _module_logger.warning(f"Negative num_workers='{final_args_ns.num_workers}', using 0."); final_args_ns.num_workers = 0

    # Validate mask source for the current stage
    is_mit_stage = 'mit_' in final_args_ns.stage
    current_mask_subdir_train = final_args_ns.train_mask_subdir_name_mit if is_mit_stage and final_args_ns.train_mask_subdir_name_mit else final_args_ns.train_mask_subdir_name
    current_mask_memmap_train = final_args_ns.train_mask_memmap_file_mit if is_mit_stage and final_args_ns.train_mask_memmap_file_mit else final_args_ns.train_mask_memmap_file
    current_mask_var_payload_train = final_args_ns.train_mask_variable_payload_file_mit if is_mit_stage and final_args_ns.train_mask_variable_payload_file_mit else final_args_ns.train_mask_variable_payload_file
    current_mask_var_header_train = final_args_ns.train_mask_variable_header_file_mit if is_mit_stage and final_args_ns.train_mask_variable_header_file_mit else final_args_ns.train_mask_variable_header_file
    
    has_fixed_train_bank = current_mask_memmap_train
    has_variable_train_bank = current_mask_var_payload_train and current_mask_var_header_train
    has_individual_masks_dir_cfg = final_args_ns.segmentation_mask_dir and current_mask_subdir_train
    
    if not (has_fixed_train_bank or has_variable_train_bank or has_individual_masks_dir_cfg):
        parser.error(f"A source for SAM segmentation masks is required for stage '{final_args_ns.stage}'. Checked individual dir, fixed bank, and variable bank options.")

    # Resolve dinov2_semantic_feature_layer_idx to be positive
    num_main_path_layers = len(final_args_ns.dinov2_layers_for_main_path)
    sem_idx = final_args_ns.dinov2_semantic_feature_layer_idx
    actual_sem_idx = sem_idx if sem_idx >= 0 else num_main_path_layers + sem_idx
    if not (0 <= actual_sem_idx < num_main_path_layers):
        parser.error(f"Invalid dinov2_semantic_feature_layer_idx ({sem_idx}). Resolved to {actual_sem_idx}, but must be between 0 and {num_main_path_layers-1}.")
    final_args_ns.dinov2_semantic_feature_layer_idx = actual_sem_idx


    try:
        main(final_args_ns)
    except KeyboardInterrupt: _logger.warning("Training interrupted by user (Ctrl+C)."); cleanup_distributed(); sys.exit(130)
    except Exception: _logger.critical("Unhandled exception during main execution:", exc_info=True); cleanup_distributed(); sys.exit(1)