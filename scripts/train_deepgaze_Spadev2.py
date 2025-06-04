#!/usr/bin/env python
"""
Multi-GPU-ready training script for DeepGaze III with DenseNet + SPADE. In this second version we avoid to use an embedding for segments, we use directly the segmentation mask as input to the SPADE layer.
Based on the DINOv2 training script structure.
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
from DeepGaze.deepgaze_pytorch.features.densenet import RGBDenseNet201
import numpy as np
import math

import pysaliency
import pysaliency.external_datasets.mit
from pysaliency.dataset_config import train_split, validation_split # For MIT splits
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from pysaliency.baseline_utils import BaselineModel, CrossvalidatedBaselineModel # Added CrossvalidatedBaselineModel
import cloudpickle as cpickle

# --- Local Project Imports ---
try:
    # Data handling classes (ensure ImageDatasetWithSegmentation is defined in data_utils.py)
    from src.data import (
        ImageDatasetWithSegmentation, # Key change
        ImageDataset, # If you kept the original name as this
        FixationMaskTransform,
        convert_stimuli, convert_fixation_trains # If used for MIT preprocessing
    )
    # Original DeepGaze modules and layers that we will reuse
    from src.modules import DeepGazeIII, FeatureExtractor, DeepGazeII, Finalizer, encode_scanpath_features, build_fixation_selection_network, build_scanpath_network
    from src.layers import (
        Bias, LayerNorm, LayerNormMultiInput,
        Conv2dMultiInput, FlexibleScanpathHistoryEncoding, SelfAttention
    )
    # Metrics and Training loop
    from src.metrics import log_likelihood, nss, auc as auc_cpu_fn
    from src.training import (_train) # Your _train function (needs modification for seg_mask)

except ImportError as e:
    actual_error_message = str(e)
    # Basic error printing if logger is not yet initialized
    print(f"PYTHON IMPORT ERROR: {actual_error_message}")
    print(f"Current sys.path: {sys.path}")
    print("Ensure 'src' is in sys.path, contains __init__.py, and all required .py files with correct internal imports.")
    print("Critical: ImageDatasetWithSegmentation must be in src/data_utils.py")
    print("Critical: Your _train function in src/training.py must be adapted to handle 'segmentation_mask'.")
    sys.exit(1)

# --- Logging Setup (Configured properly in main() after DDP init) ---
_logger = logging.getLogger("train_densenet_spade_ddp")

# --- Distributed Utils (Copied from DINOv2 script) ---
def init_distributed() -> tuple[torch.device, int, int, bool, bool]:
    """Initialise torch.distributed (NCCL) if environment variables are present."""
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
        is_master = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, rank, world_size, is_master, is_distributed

def cleanup_distributed():
    """Clean up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()

# ============================================================================
# == SPADE Layer and SPADE-Enhanced Network Definitions ==
# ============================================================================

class SPADELayerNorm(nn.Module):
    def __init__(self, norm_features, segmap_input_channels,
                 hidden_mlp_channels=128, eps=1e-12, kernel_size=3):
        super().__init__()
        self.norm_features = norm_features
        self.eps = eps
        self.segmap_input_channels = segmap_input_channels # This will now be num_total_segments
        padding = kernel_size // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(self.segmap_input_channels, hidden_mlp_channels,
                      kernel_size=kernel_size, padding=padding, bias=True),
            nn.ReLU(inplace=True)
        )
        self.mlp_gamma = nn.Conv2d(hidden_mlp_channels, norm_features,
                                   kernel_size=kernel_size, padding=padding, bias=True)
        self.mlp_beta = nn.Conv2d(hidden_mlp_channels, norm_features,
                                  kernel_size=kernel_size, padding=padding, bias=True)

    # Changed argument name from segmap_embedded to segmap_processed for clarity
    def forward(self, x, segmap_processed): 
        if x.shape[1] != self.norm_features:
            raise ValueError(f"Input x chans {x.shape[1]} != norm_features {self.norm_features}")

        normalized_shape = (self.norm_features, x.size(2), x.size(3))
        normalized_x = F.layer_norm(x, normalized_shape, weight=None, bias=None, eps=self.eps)

        # segmap_processed is now expected to be (B, num_total_segments, H_seg, W_seg) and float
        if segmap_processed.ndim != 4:
             raise ValueError(f"Expected segmap_processed to be 4D, got {segmap_processed.ndim}")
        if segmap_processed.shape[1] != self.segmap_input_channels:
             _logger.error(f"Processed segmap channels {segmap_processed.shape[1]} != expected {self.segmap_input_channels}")
             raise ValueError("Processed segmap channel mismatch")

        segmap_resized = F.interpolate(segmap_processed, size=x.size()[2:], mode='nearest')
        shared_features = self.mlp_shared(segmap_resized)
        gamma_map = self.mlp_gamma(shared_features)
        beta_map = self.mlp_beta(shared_features)
        out = normalized_x * (1 + gamma_map) + beta_map
        return out

    def extra_repr(self):
        return (f'norm_features={self.norm_features}, '
                f'segmap_input_channels={self.segmap_input_channels}, eps={self.eps}')


class SaliencyNetworkSPADE(nn.Module):
    # seg_embedding_dim is no longer directly used by SPADE layers here,
    # but kept in signature for consistency if other parts of a larger system expect it.
    def __init__(self, input_channels, num_total_segments, seg_embedding_dim, add_sa_head=False):
        super().__init__()
        self.input_channels = input_channels
        self.num_total_segments = num_total_segments # Max number of segment IDs (e.g., 0 to N-1)
        # self.seg_embedding_dim = seg_embedding_dim # Store if needed, but not used for SPADE input channels now
        self.add_sa_head = add_sa_head

        # --- MODIFICATION: Remove Embedding Layer ---
        # self.seg_embedder = nn.Embedding(self.num_total_segments, self.seg_embedding_dim)
        # --- END MODIFICATION ---

        current_channels = input_channels 

        if self.add_sa_head:
            self.sa_head_module = SelfAttention(current_channels, key_channels=current_channels // 8, return_attention=False)
            self.sa_ln_out = LayerNorm(current_channels) 
            self.sa_softplus_out = nn.Softplus()
            _logger.info(f"SaliencyNetworkSPADE: Added SA head. Input channels to SA: {current_channels}")

        # --- MODIFICATION: SPADE input channels change ---
        # The input channels to SPADE's MLP will now be num_total_segments (from one-hot encoding)
        spade_segmap_input_channels = self.num_total_segments
        # --- END MODIFICATION ---

        self.spade_ln0 = SPADELayerNorm(current_channels, spade_segmap_input_channels)
        self.conv0 = nn.Conv2d(current_channels, 8, (1, 1), bias=False)
        self.bias0 = Bias(8)
        self.softplus0 = nn.Softplus()

        self.spade_ln1 = SPADELayerNorm(8, spade_segmap_input_channels)
        self.conv1 = nn.Conv2d(8, 16, (1, 1), bias=False)
        self.bias1 = Bias(16)
        self.softplus1 = nn.Softplus()

        self.spade_ln2 = SPADELayerNorm(16, spade_segmap_input_channels)
        self.conv2 = nn.Conv2d(16, 1, (1, 1), bias=False)
        self.bias2 = Bias(1)
        self.softplus2 = nn.Softplus()

    def forward(self, x, raw_segmap_long): # raw_segmap expected as (B, H_img, W_img) LongTensor
        # --- MODIFICATION: Process raw_segmap_long to one-hot encoding ---
        if raw_segmap_long.dtype != torch.long:
            _logger.warning(f"raw_segmap_long dtype is {raw_segmap_long.dtype}, converting to long.")
            raw_segmap_long = raw_segmap_long.long()

        B, H_seg, W_seg = raw_segmap_long.shape
        
        # Clamp values to be safe for one_hot encoding (0 to num_total_segments-1)
        # F.one_hot expects class indices from 0 to num_classes-1.
        clamped_segmap_ids = torch.clamp(raw_segmap_long, 0, self.num_total_segments - 1)
        
        # One-hot encode the clamped segment IDs
        # Input: (B, H_seg, W_seg) with Long values in [0, num_total_segments-1]
        # Output: (B, H_seg, W_seg, num_total_segments) with boolean/int then float
        segmap_one_hot = F.one_hot(clamped_segmap_ids, num_classes=self.num_total_segments)
        
        # Permute to (B, num_total_segments, H_seg, W_seg) and convert to float for conv layers
        # This `segmap_processed_for_spade` will be passed to SPADELayerNorm
        segmap_processed_for_spade = segmap_one_hot.permute(0, 3, 1, 2).contiguous().float()
        # --- END MODIFICATION ---

        h = x
        if self.add_sa_head:
            h, _ = self.sa_head_module(h) 
            h = self.sa_ln_out(h)
            h = self.sa_softplus_out(h)

        h = self.spade_ln0(h, segmap_processed_for_spade) # Use the new processed segmap
        h = self.conv0(h)
        h = self.bias0(h)
        h = self.softplus0(h)

        h = self.spade_ln1(h, segmap_processed_for_spade) # Use the new processed segmap
        h = self.conv1(h)
        h = self.bias1(h)
        h = self.softplus1(h)

        h = self.spade_ln2(h, segmap_processed_for_spade) # Use the new processed segmap
        h = self.conv2(h)
        h = self.bias2(h)
        h = self.softplus2(h)
        return h

class DeepGazeIIISpade(nn.Module): # Renaming to avoid conflict with original
    def __init__(self, features, saliency_network, scanpath_network,
                 fixation_selection_network, downsample=2, readout_factor=2,
                 saliency_map_factor=2, included_fixations=-2, initial_sigma=8.0,
                 finalizer_learn_sigma=True): # Added finalizer_learn_sigma
        super().__init__()

        self.downsample = downsample
        self.readout_factor = readout_factor
        # self.saliency_map_factor = saliency_map_factor # This is used by Finalizer
        # self.included_fixations = included_fixations # Used by scanpath logic if any

        self.features = features # This is your DenseNet+FeatureExtractor or DinoV2Backbone
        # Freezing logic moved to main training script part
        # if hasattr(self.features, 'parameters'): # Freeze backbone if it has parameters
        #     for param in self.features.parameters():
        #         param.requires_grad = False
        # if hasattr(self.features, 'eval'):
        #     self.features.eval()

        self.saliency_network = saliency_network # This will be your SaliencyNetworkSPADE instance
        self.scanpath_network = scanpath_network
        self.fixation_selection_network = fixation_selection_network

        self.finalizer = Finalizer( # from deepgaze_pytorch.modules
            sigma=initial_sigma,
            learn_sigma=finalizer_learn_sigma, # Make this configurable
            saliency_map_factor=saliency_map_factor,
        )

    def forward(self, image, centerbias, x_hist=None, y_hist=None, durations=None,
                segmentation_mask=None, # NEW: For SPADE
                **kwargs): # For any other arguments from the batch

        orig_shape = image.shape # (B, C, H_orig, W_orig)

        # 1. Feature Extraction
        # Apply downsampling before feature extraction if specified
        if self.downsample != 1:
            img_for_features = F.interpolate(image, scale_factor=1.0 / self.downsample,
                                             recompute_scale_factor=False, mode='bilinear', align_corners=False)
        else:
            img_for_features = image

        # self.features is your FeatureExtractor(RGBDenseNet201(), ...) or DinoV2Backbone()
        # It should return a list of feature tensors
        extracted_feature_maps = self.features(img_for_features)

        # 2. Prepare features for readout network (resize & concatenate)
        # Target spatial dimensions for features entering the readout heads
        readout_h = math.ceil(orig_shape[2] / self.downsample / self.readout_factor)
        readout_w = math.ceil(orig_shape[3] / self.downsample / self.readout_factor)
        readout_spatial_shape = (readout_h, readout_w)

        processed_features_list = []
        for feat_map in extracted_feature_maps:
            processed_features_list.append(
                F.interpolate(feat_map, size=readout_spatial_shape, mode='bilinear', align_corners=False)
            )

        concatenated_backbone_features = torch.cat(processed_features_list, dim=1)

        # 3. Saliency Readout Head
        if isinstance(self.saliency_network, SaliencyNetworkSPADE):
            if segmentation_mask is None:
                 raise ValueError("SaliencyNetworkSPADE requires a segmentation_mask, but None was provided.")
            saliency_path_output = self.saliency_network(concatenated_backbone_features, segmentation_mask)
        else: # Original saliency network (doesn't take seg_mask)
            saliency_path_output = self.saliency_network(concatenated_backbone_features)

        # 4. Scanpath Readout Head (if used)
        scanpath_path_output = None
        if self.scanpath_network is not None:
            if x_hist is None or y_hist is None or x_hist.numel() == 0 or y_hist.numel() == 0:
                raise ValueError(
                    "Scanpath network is active, but x_hist or y_hist is None or empty. "
                    "Ensure dataloader provides scanpath history for this configuration."
                )
            scanpath_features_encoded = encode_scanpath_features(
                x_hist, y_hist,
                size=(orig_shape[2], orig_shape[3]),
                device=image.device
            )
            scanpath_features_resized = F.interpolate(scanpath_features_encoded,
                                                      size=readout_spatial_shape,
                                                      mode='bilinear', align_corners=False)
            scanpath_path_output = self.scanpath_network(scanpath_features_resized)

        # 5. Fixation Selection Network
        combined_input_for_fixsel = (saliency_path_output, scanpath_path_output)
        final_readout_before_finalizer = self.fixation_selection_network(combined_input_for_fixsel)

        # 6. Finalizer
        saliency_log_density = self.finalizer(final_readout_before_finalizer, centerbias)

        return saliency_log_density

    def train(self, mode=True): # From original DeepGazeIII
        # Backbone features are frozen by main script, their mode doesn't change from eval
        if hasattr(self.features, 'eval'):
            self.features.eval()

        self.saliency_network.train(mode=mode)
        if self.scanpath_network is not None:
            self.scanpath_network.train(mode=mode)
        self.fixation_selection_network.train(mode=mode)
        self.finalizer.train(mode=mode) # Finalizer has learnable params (sigma, center_bias_weight)
        super().train(mode)


# ============================================================================
# == MAIN FUNCTION ==
# ============================================================================
def main(args: argparse.Namespace): 
    device, rank, world, is_master, is_distributed = init_distributed()

    log_level = logging.INFO if is_master else logging.WARNING 
    if hasattr(args, 'log_level') and args.log_level is not None:
        try:
            log_level = getattr(logging, str(args.log_level).upper(), log_level)
        except AttributeError:
            print(f"Warning: Invalid log_level '{args.log_level}'. Using default.")

    logging.basicConfig(level=log_level, format=f"%(asctime)s Rank{rank} %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", force=True)
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING) 
    _logger.setLevel(log_level) 

    if is_master:
        _logger.info("================== Effective Configuration ==================")
        # Resolve paths to absolute for logging, if they are Path objects
        resolved_args_dict = {}
        for arg_name, arg_value in sorted(vars(args).items()):
            if isinstance(arg_value, Path):
                resolved_args_dict[arg_name] = str(arg_value.resolve())
            else:
                resolved_args_dict[arg_name] = arg_value
        
        for arg_name, arg_value in sorted(resolved_args_dict.items()):
             _logger.info(f"  {arg_name}: {arg_value}")
        _logger.info(f"  DDP Info: Rank {rank}/{world}, Master: {is_master}, Distributed: {is_distributed}, Device: {device}")
        _logger.info(f"  Torch: {torch.__version__}, CUDA: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
        _logger.info("===========================================================")


    # --- Path Setup ---
    dataset_directory = args.dataset_dir # Already a Path object from __main__
    train_output_base_dir = args.train_dir # Already a Path object
    lmdb_image_cache_dir = args.lmdb_dir # Already a Path object
    
    # REMOVED: segmentation_mask_root_dir related logic
    # segmentation_mask_root_dir = Path(args.segmentation_mask_dir).resolve() if args.segmentation_mask_dir else None

    if is_master:
        for p_obj in [dataset_directory, train_output_base_dir, lmdb_image_cache_dir]:
            if p_obj:  # Ensure it's not None
                try:
                    p_obj.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    _logger.error(f"Failed to create directory {p_obj}: {e}")
    # REMOVED: segmentation_mask_root_dir check
    # if segmentation_mask_root_dir and not segmentation_mask_root_dir.exists():
    #          _logger.warning(f"Individual mask root dir {segmentation_mask_root_dir} (for fallback/source) does not exist.")
    if is_distributed: dist.barrier()

    # --- Backbone Setup: DenseNet ---
    if is_master: _logger.info(f"Initializing {args.densenet_model_name} backbone...")
    if args.densenet_model_name == 'densenet201':
        densenet_base_model = RGBDenseNet201()
    else:
        _logger.critical(f"Unsupported densenet_model_name: {args.densenet_model_name}"); sys.exit(1)

    saliency_map_factor_densenet = 4 
    densenet_feature_nodes = [
        '1.features.denseblock4.denselayer32.norm1',
        '1.features.denseblock4.denselayer32.conv1',
        '1.features.denseblock4.denselayer31.conv2',
    ]
    features_module = FeatureExtractor(densenet_base_model, densenet_feature_nodes)

    for param in features_module.parameters(): param.requires_grad = False
    features_module.eval().to(device)
    if is_master: _logger.info(f"{args.densenet_model_name} with FeatureExtractor (hooks: {densenet_feature_nodes}) initialized and frozen on {device}.")

    concatenated_channels = 0
    dummy_h, dummy_w = 256, 256 
    try:
        salicon_check_path = dataset_directory / 'SALICON' / 'stimuli' / 'train'
        if salicon_check_path.exists():
            # Temporarily load one SALICON image to get dimensions
            # FIX: Changed 'SALICON'.parent to use dataset_directory directly
            temp_stim, _ = pysaliency.get_SALICON_train(location=str(dataset_directory)) 
            if temp_stim and temp_stim.sizes is not None and len(temp_stim.sizes) > 0:
                 h_s, w_s = temp_stim.sizes[0][:2]
                 min_dim_factor = 32 
                 dummy_h, dummy_w = max(h_s, min_dim_factor), max(w_s, min_dim_factor)
                 if is_master: _logger.info(f"Using dummy input size from SALICON sample: H={dummy_h}, W={dummy_w} for channel calculation.")
            del temp_stim 
    except Exception as e_sz_dummy:
        if is_master: _logger.warning(f"Could not get dataset size for dummy input ({e_sz_dummy}), using default H={dummy_h}, W={dummy_w}.")

    dummy_input = torch.randn(1, 3, dummy_h, dummy_w).to(device)
    with torch.no_grad():
        extracted_dummy_feature_maps = features_module(dummy_input)

    if not isinstance(extracted_dummy_feature_maps, list) or not all(isinstance(t, torch.Tensor) for t in extracted_dummy_feature_maps):
        _logger.critical(f"FeatureExtractor did not return list of tensors. Got: {type(extracted_dummy_feature_maps)}. Check FeatureExtractor/hooks."); sys.exit(1)

    concatenated_channels = sum(feat_map.shape[1] for feat_map in extracted_dummy_feature_maps)

    if concatenated_channels == 0:
        _logger.error(f"Dummy forward pass resulted in 0 concatenated_channels! Fallback to 2048. Investigate FeatureExtractor hooks: {densenet_feature_nodes}")
        concatenated_channels = 2048 
    if is_master:
        _logger.info(f"Determined concatenated input channels for SPADE head: {concatenated_channels}")
        for i, feat_map in enumerate(extracted_dummy_feature_maps):
            node_name = densenet_feature_nodes[i] if i < len(densenet_feature_nodes) else f"layer_{i}"
            _logger.info(f"  - Extracted dummy feature map {i} (from node '{node_name}') shape: {feat_map.shape}")
    del dummy_input, extracted_dummy_feature_maps 


    # --- Stage Dispatch ---
    if args.stage == 'salicon_pretrain_densenet_spade':
        if is_master: _logger.info(f"--- Preparing SALICON Pretraining with DenseNet+SPADE ---")
        current_lr = args.lr
        current_milestones = args.lr_milestones
        experiment_name = (f"{args.stage}")

        if is_master: _logger.info(f"Loading SALICON data from {dataset_directory}...")
        salicon_train_loc = dataset_directory / 'SALICON'
        salicon_val_loc = dataset_directory / 'SALICON'
        if is_master:
            try:
                if not (salicon_train_loc / 'stimuli' / 'train').exists(): pysaliency.get_SALICON_train(location=str(salicon_train_loc.parent))
                if not (salicon_val_loc / 'stimuli' / 'val').exists(): pysaliency.get_SALICON_val(location=str(salicon_val_loc.parent))
            except Exception as e: _logger.critical(f"Failed SALICON get: {e}"); dist.barrier(); sys.exit(1)
        if is_distributed: dist.barrier()
        SALICON_train_stimuli, SALICON_train_fixations = pysaliency.get_SALICON_train(location=str(salicon_train_loc.parent))
        SALICON_val_stimuli, SALICON_val_fixations = pysaliency.get_SALICON_val(location=str(salicon_val_loc.parent))
        if is_master: _logger.info("SALICON data loaded.")

        if is_master: _logger.info("Initializing SALICON BaselineModel...")
        SALICON_centerbias = BaselineModel(stimuli=SALICON_train_stimuli, fixations=SALICON_train_fixations, bandwidth=0.0217, eps=2e-13, caching=False)

        train_ll_cache_file = dataset_directory / f'salicon_baseline_train_ll_{args.densenet_model_name}.pkl'
        val_ll_cache_file = dataset_directory / f'salicon_baseline_val_ll_{args.densenet_model_name}.pkl'
        train_baseline_log_likelihood, val_baseline_log_likelihood = None, None
        if is_master:
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

        ll_bcast=[train_baseline_log_likelihood, val_baseline_log_likelihood]
        if is_distributed:
            if is_master and (train_baseline_log_likelihood is None or val_baseline_log_likelihood is None): ll_bcast = [np.nan, np.nan]
            dist.broadcast_object_list(ll_bcast,src=0)
        train_baseline_log_likelihood,val_baseline_log_likelihood = ll_bcast
        if np.isnan(train_baseline_log_likelihood) or np.isnan(val_baseline_log_likelihood): _logger.critical(f"NaN LLs received/computed on rank {rank}. Exiting."); sys.exit(1)
        else: _logger.info(f"Rank {rank} Baseline LLs - Train: {train_baseline_log_likelihood:.5f}, Val: {val_baseline_log_likelihood:.5f}")

        saliency_net_spade = SaliencyNetworkSPADE(
            input_channels=concatenated_channels,
            num_total_segments=args.num_total_segments,
            seg_embedding_dim=args.seg_embedding_dim,
            add_sa_head=args.add_sa_head
        )
        scanpath_net = None
        fixsel_net = build_fixation_selection_network(scanpath_features=0)

        model = DeepGazeIIISpade(
            features=features_module,
            saliency_network=saliency_net_spade,
            scanpath_network=scanpath_net,
            fixation_selection_network=fixsel_net,
            downsample=1, readout_factor=4,
            saliency_map_factor=saliency_map_factor_densenet,
            included_fixations=[], 
            initial_sigma=args.finalizer_initial_sigma,
            finalizer_learn_sigma=args.finalizer_learn_sigma
        ).to(device)

        if is_master: _logger.info("DeepGazeIII SPADE model built for SALICON.")
        if is_distributed:
            model = DDP(model, device_ids=[device.index], find_unused_parameters=True)
            if is_master: _logger.info("Wrapped model with DDP.")

        head_params = [p for p in model.parameters() if p.requires_grad]
        if not head_params: _logger.critical("No trainable parameters found!"); sys.exit(1)
        optimizer = optim.Adam(head_params, lr=current_lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=current_milestones)

        # --- Dataloaders for SALICON ---
        # FIX: Use direct path arguments from args
        train_mask_individual_dir = args.salicon_train_mask_dir # Already a Path object or None
        val_mask_individual_dir = args.salicon_val_mask_dir     # Already a Path object or None
        
        train_fixed_memmap = args.train_mask_memmap_file # Already a Path or None
        val_fixed_memmap = args.val_mask_memmap_file     # Already a Path or None
        train_var_payload = args.train_mask_variable_payload_file # Already a Path or None
        train_var_header = args.train_mask_variable_header_file   # Already a Path or None
        val_var_payload = args.val_mask_variable_payload_file     # Already a Path or None
        val_var_header = args.val_mask_variable_header_file       # Already a Path or None

        if is_master:
            _logger.info(f"Train Mask Config: FixedBank='{train_fixed_memmap}', VarBankPayload='{train_var_payload}', VarBankHeader='{train_var_header}', IndividualDir='{train_mask_individual_dir}'")
            _logger.info(f"Val Mask Config: FixedBank='{val_fixed_memmap}', VarBankPayload='{val_var_payload}', VarBankHeader='{val_var_header}', IndividualDir='{val_mask_individual_dir}'")

        dataset_common_kwargs_train = {
            "transform": FixationMaskTransform(sparse=False), "average": "image",
            "lmdb_path": str(lmdb_image_cache_dir / f'SALICON_train_images_{args.densenet_model_name}') if args.use_lmdb_images else None,
            "segmentation_mask_dir": train_mask_individual_dir, 
            "segmentation_mask_format": args.segmentation_mask_format,
            "segmentation_mask_fixed_memmap_file": train_fixed_memmap,
            "segmentation_mask_variable_payload_file": train_var_payload,
            "segmentation_mask_variable_header_file": train_var_header,
            "segmentation_mask_bank_dtype": args.segmentation_mask_bank_dtype
        }
        train_dataset = ImageDatasetWithSegmentation(stimuli=SALICON_train_stimuli, fixations=SALICON_train_fixations, centerbias_model=SALICON_centerbias, **dataset_common_kwargs_train)

        dataset_common_kwargs_val = {
            "transform": FixationMaskTransform(sparse=False), "average": "image",
            "lmdb_path": str(lmdb_image_cache_dir / f'SALICON_val_images_{args.densenet_model_name}') if args.use_lmdb_images else None,
            "segmentation_mask_dir": val_mask_individual_dir, 
            "segmentation_mask_format": args.segmentation_mask_format,
            "segmentation_mask_fixed_memmap_file": val_fixed_memmap,
            "segmentation_mask_variable_payload_file": val_var_payload,
            "segmentation_mask_variable_header_file": val_var_header,
            "segmentation_mask_bank_dtype": args.segmentation_mask_bank_dtype
        }
        val_dataset = ImageDatasetWithSegmentation(stimuli=SALICON_val_stimuli, fixations=SALICON_val_fixations, centerbias_model=SALICON_centerbias, **dataset_common_kwargs_val)

        train_sampler = (torch.utils.data.DistributedSampler(train_dataset, shuffle=True, drop_last=True) if is_distributed else None)
        if train_sampler and hasattr(train_sampler, 'set_epoch'): train_sampler.set_epoch(0) # Initial epoch set

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                                 num_workers=args.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True,
                                                 persistent_workers=args.num_workers > 0)
        if is_master: _logger.info(f"Train Dataloader: {len(train_loader)} batches.")

        val_sampler = (torch.utils.data.DistributedSampler(val_dataset, shuffle=False, drop_last=False) if is_distributed else None)
        # Note: val_sampler doesn't need set_epoch if shuffle=False

        validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                                     num_workers=args.num_workers, pin_memory=True, sampler=val_sampler,
                                                     persistent_workers=args.num_workers > 0)
        if is_master: _logger.info(f"Validation Dataloader: {len(validation_loader)} batches.")

        output_dir_experiment = train_output_base_dir / experiment_name
        if is_master: output_dir_experiment.mkdir(parents=True, exist_ok=True)
        if is_distributed: dist.barrier()

        _logger.info(f"Experiment output to: {output_dir_experiment}")

        _train(
            this_directory=str(output_dir_experiment), model=model,
            train_loader=train_loader, train_baseline_log_likelihood=train_baseline_log_likelihood,
            val_loader=validation_loader, val_baseline_log_likelihood=val_baseline_log_likelihood,
            optimizer=optimizer, lr_scheduler=lr_scheduler,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            minimum_learning_rate=args.min_lr,
            validation_metrics=['LL', 'IG', 'NSS', 'AUC_CPU'],
            validation_epochs=args.validation_epochs,
            startwith=args.resume_checkpoint, 
            device=device, is_distributed=is_distributed, is_master=is_master, logger=_logger,
        )
        if is_master: _logger.info(f"--- Stage '{args.stage}' Finished ---")

    elif args.stage == 'mit_spatial_densenet_spade':
        fold = args.fold
        if fold is None or not (0 <= fold < 10):
            _logger.critical("--fold (0-9) required for MIT stages and must be valid.");
            if is_distributed: dist.barrier(); sys.exit(1)
        if is_master: _logger.info(f"--- Preparing MIT1003 Stage: {args.stage} (Fold {fold}) ---")

        current_lr = args.lr_mit_spatial if args.lr_mit_spatial is not None else args.lr 
        current_milestones = args.lr_milestones_mit_spatial if args.lr_milestones_mit_spatial is not None else args.lr_milestones

        experiment_name_mit = f"{args.stage}_fold{fold}_{args.densenet_model_name}_k{args.num_total_segments}_emb{args.seg_embedding_dim}_lr{current_lr}"
        output_dir_experiment = train_output_base_dir / experiment_name_mit
        if is_master: output_dir_experiment.mkdir(parents=True, exist_ok=True)
        if is_distributed: dist.barrier()


        salicon_checkpoint_path_actual = None
        if args.salicon_checkpoint_path: # Already a Path object or None
            salicon_checkpoint_path_actual = args.salicon_checkpoint_path
        else:
            salicon_stage_output_dir = train_output_base_dir / "salicon_pretrain_densenet_spade"
            options = [salicon_stage_output_dir / 'final_best_val.pth', salicon_stage_output_dir / 'final.pth']
            for opt in options:
                if opt.exists():
                    salicon_checkpoint_path_actual = opt
                    if is_master: _logger.info(f"Auto-detected SALICON checkpoint: {salicon_checkpoint_path_actual}")
                    break
        if is_master and salicon_checkpoint_path_actual and not salicon_checkpoint_path_actual.exists():
            _logger.warning(f"Specified/inferred SALICON checkpoint {salicon_checkpoint_path_actual} not found.")
        elif is_master and not salicon_checkpoint_path_actual:
            _logger.warning(f"No SALICON checkpoint specified or auto-detected. MIT head will be randomly initialized.")

        if is_master: _logger.info(f"Loading MIT1003 data from {dataset_directory} for fold {fold}...")
        mit_stimuli_all, mit_fixations_all_scanpaths = pysaliency.external_datasets.mit.get_mit1003_with_initial_fixation(
            location=str(dataset_directory), replace_initial_invalid_fixations=True
        )
        if is_master: _logger.info("MIT1003 data loaded.")

        MIT_stimuli_train, MIT_fixations_train = train_split(mit_stimuli_all, mit_fixations_all_scanpaths, crossval_folds=10, fold_no=fold)
        MIT_stimuli_val, MIT_fixations_val = validation_split(mit_stimuli_all, mit_fixations_all_scanpaths, crossval_folds=10, fold_no=fold)

        if is_master: _logger.info("Initializing MIT1003 CrossvalidatedBaselineModel...")
        mit_bandwidth = 10**-1.6667673342543432
        mit_eps = 10**-14.884189168516073
        MIT_centerbias = CrossvalidatedBaselineModel(
            mit_stimuli_all, mit_fixations_all_scanpaths,
            bandwidth=mit_bandwidth, eps=mit_eps, caching=False
        )

        train_ll_cache_file = dataset_directory / f'mit1003_baseline_train_ll_fold{fold}_{args.densenet_model_name}.pkl'
        val_ll_cache_file = dataset_directory / f'mit1003_baseline_val_ll_fold{fold}_{args.densenet_model_name}.pkl'
        train_baseline_log_likelihood, val_baseline_log_likelihood = None, None

        if is_master:
            try:
                with open(train_ll_cache_file, 'rb') as f: train_baseline_log_likelihood = cpickle.load(f)
                _logger.info(f"Loaded MIT TRAIN LL from {train_ll_cache_file}")
            except:
                _logger.warning(f"MIT TRAIN LL cache miss for {train_ll_cache_file}. Computing...");
                train_baseline_log_likelihood = MIT_centerbias.information_gain(MIT_stimuli_train, MIT_fixations_train, verbose=True, average='image')
                with open(train_ll_cache_file, 'wb') as f: cpickle.dump(train_baseline_log_likelihood, f)
            try:
                with open(val_ll_cache_file, 'rb') as f: val_baseline_log_likelihood = cpickle.load(f)
                _logger.info(f"Loaded MIT VAL LL from {val_ll_cache_file}")
            except:
                _logger.warning(f"MIT VAL LL cache miss for {val_ll_cache_file}. Computing...");
                val_baseline_log_likelihood = MIT_centerbias.information_gain(MIT_stimuli_val, MIT_fixations_val, verbose=True, average='image')
                with open(val_ll_cache_file, 'wb') as f: cpickle.dump(val_baseline_log_likelihood, f)
            _logger.info(f"Master MIT Fold {fold} Baseline LLs - Train: {train_baseline_log_likelihood or float('nan'):.5f}, Val: {val_baseline_log_likelihood or float('nan'):.5f}")

        ll_bcast_mit = [train_baseline_log_likelihood, val_baseline_log_likelihood]
        if is_distributed:
            if is_master and (train_baseline_log_likelihood is None or val_baseline_log_likelihood is None or np.isnan(train_baseline_log_likelihood) or np.isnan(val_baseline_log_likelihood)): 
                ll_bcast_mit = [np.nan, np.nan] # Ensure NaN broadcast if master fails
            dist.broadcast_object_list(ll_bcast_mit, src=0)
        train_baseline_log_likelihood, val_baseline_log_likelihood = ll_bcast_mit
        if np.isnan(train_baseline_log_likelihood) or np.isnan(val_baseline_log_likelihood):
            _logger.critical(f"MIT Baseline LLs invalid on rank {rank}. Exiting."); dist.barrier(); sys.exit(1)
        _logger.info(f"Rank {rank} MIT Fold {fold} Baseline LLs - Train: {train_baseline_log_likelihood:.5f}, Val: {val_baseline_log_likelihood:.5f}")


        saliency_net_spade_mit = SaliencyNetworkSPADE(
            input_channels=concatenated_channels,
            num_total_segments=args.num_total_segments,
            seg_embedding_dim=args.seg_embedding_dim,
            add_sa_head=args.add_sa_head
        ) 
        scanpath_net_mit = None
        fixsel_net_mit = build_fixation_selection_network(scanpath_features=0) 

        model = DeepGazeIIISpade(
            features=features_module, 
            saliency_network=saliency_net_spade_mit, 
            scanpath_network=scanpath_net_mit,
            fixation_selection_network=fixsel_net_mit, 
            downsample=1, readout_factor=4, 
            saliency_map_factor=saliency_map_factor_densenet,
            included_fixations=[],
            initial_sigma=args.finalizer_initial_sigma, 
            finalizer_learn_sigma=args.finalizer_learn_sigma
        ).to(device) 

        if salicon_checkpoint_path_actual and salicon_checkpoint_path_actual.exists():
            if is_master: _logger.info(f"Loading SALICON checkpoint from {salicon_checkpoint_path_actual} for MIT model head.")
            # Ensure map_location handles device placement correctly, especially if resuming DDP model on non-DDP or vice-versa
            state_dict = torch.load(salicon_checkpoint_path_actual, map_location='cpu', weights_only=False) 
            if 'model_state_dict' in state_dict: state_dict = state_dict['model_state_dict'] 

            clean_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                clean_state_dict[name] = v
            
            missing_keys, unexpected_keys = model.load_state_dict(clean_state_dict, strict=False)
            if is_master:
                # Filter out feature keys from missing_keys as they are frozen and not expected to load if different
                missing_head_keys = [k for k in missing_keys if not k.startswith('features.')]
                if missing_head_keys:
                     _logger.warning(f"MIT Load: Missing HEAD keys: {missing_head_keys}")
                
                # Unexpected keys might indicate a mismatch if the SALICON model had more/different parts
                # than the current model structure (excluding features).
                unexpected_head_keys = [k for k in unexpected_keys if not k.startswith('features.')]
                if unexpected_head_keys: 
                     _logger.warning(f"MIT Load: Unexpected HEAD keys: {unexpected_head_keys}")
                _logger.info("SALICON head weights loaded into MIT model (or attempted).")
        else:
            if is_master: _logger.warning("SALICON checkpoint not found. MIT head starts with its default initialization.")

        if is_master: _logger.info(f"DeepGazeIII SPADE model for MIT Fold {fold} prepared.")
        if is_distributed:
            model = DDP(model, device_ids=[device.index], find_unused_parameters=True)
            if is_master: _logger.info("Wrapped MIT model with DDP.")

        head_params_mit = [p for p in model.parameters() if p.requires_grad]
        if not head_params_mit: _logger.critical("No trainable parameters found for MIT stage!"); sys.exit(1)
        optimizer = optim.Adam(head_params_mit, lr=current_lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=current_milestones)

        # --- Dataloaders for MIT ---
        # FIX: Use direct path argument from args
        mit_individual_mask_dir = args.mit_all_mask_dir # Already a Path or None
        
        mit_fixed_memmap = args.mit_mask_fixed_memmap_file         # Already a Path or None
        mit_var_payload = args.mit_mask_variable_payload_file     # Already a Path or None
        mit_var_header = args.mit_mask_variable_header_file       # Already a Path or None
        
        if is_master:
            _logger.info(f"MIT Mask Config (Fold {fold}): FixedBank='{mit_fixed_memmap}', VarBankPayload='{mit_var_payload}', VarBankHeader='{mit_var_header}', IndividualDir='{mit_individual_mask_dir}'")
            if mit_individual_mask_dir and not mit_individual_mask_dir.exists():
                _logger.warning(f"MIT individual mask directory does not exist: {mit_individual_mask_dir}")


        dataset_kwargs_mit_common = {
            "transform": FixationMaskTransform(sparse=False), "average": "image",
            "segmentation_mask_dir": mit_individual_mask_dir, 
            "segmentation_mask_format": args.segmentation_mask_format,
            "segmentation_mask_fixed_memmap_file": mit_fixed_memmap,
            "segmentation_mask_variable_payload_file": mit_var_payload,
            "segmentation_mask_variable_header_file": mit_var_header,
            "segmentation_mask_bank_dtype": args.segmentation_mask_bank_dtype,
        }

        train_dataset_mit = ImageDatasetWithSegmentation(
            stimuli=MIT_stimuli_train, fixations=MIT_fixations_train, centerbias_model=MIT_centerbias,
            lmdb_path=str(lmdb_image_cache_dir / f'MIT1003_train_images_fold{fold}_{args.densenet_model_name}') if args.use_lmdb_images else None,
            **dataset_kwargs_mit_common
        )
        val_dataset_mit = ImageDatasetWithSegmentation(
            stimuli=MIT_stimuli_val, fixations=MIT_fixations_val, centerbias_model=MIT_centerbias,
            lmdb_path=str(lmdb_image_cache_dir / f'MIT1003_val_images_fold{fold}_{args.densenet_model_name}') if args.use_lmdb_images else None,
            **dataset_kwargs_mit_common
        )

        train_sampler_mit = (torch.utils.data.DistributedSampler(train_dataset_mit, shuffle=True, drop_last=True) if is_distributed else None)
        if train_sampler_mit and hasattr(train_sampler_mit, 'set_epoch'): train_sampler_mit.set_epoch(0) # Initial epoch
        train_loader_mit = torch.utils.data.DataLoader(train_dataset_mit, batch_size=args.batch_size, shuffle=(train_sampler_mit is None),
                                                   num_workers=args.num_workers, pin_memory=True, sampler=train_sampler_mit, drop_last=True,
                                                   persistent_workers=args.num_workers > 0)
        if is_master: _logger.info(f"MIT Train Dataloader (Fold {fold}): {len(train_loader_mit)} batches.")

        val_sampler_mit = (torch.utils.data.DistributedSampler(val_dataset_mit, shuffle=False, drop_last=False) if is_distributed else None)
        validation_loader_mit = torch.utils.data.DataLoader(val_dataset_mit, batch_size=args.batch_size, shuffle=False,
                                                        num_workers=args.num_workers, pin_memory=True, sampler=val_sampler_mit,
                                                        persistent_workers=args.num_workers > 0)
        if is_master: _logger.info(f"MIT Validation Dataloader (Fold {fold}): {len(validation_loader_mit)} batches.")

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
            startwith=None, 
            device=device, is_distributed=is_distributed, is_master=is_master, logger=_logger
        )
        if is_master: _logger.info(f"--- MIT Stage '{args.stage}' (Fold {fold}) Finished ---")


    elif args.stage in ['mit_scanpath_frozen_densenet_spade', 'mit_scanpath_full_densenet_spade']:
        _logger.warning(f"MIT scanpath stage '{args.stage}' is a placeholder. Full implementation needed.")
        if is_master: _logger.info(f"--- Stage '{args.stage}' SKIPPED (Placeholder) ---")

    else:
        _logger.critical(f"Unknown or unsupported stage: {args.stage}"); sys.exit(1)

    cleanup_distributed()
    if is_master: _logger.info("Training script finished successfully.")


# ──────────────────────────────────────────────────────────────────────────────
#  Main entry-point – YAML + CLI handling
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    _module_logger = logging.getLogger(__name__)

    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument('--config_file', type=str, default=None, help="Path to YAML configuration file.")
    _cfg_namespace, _remaining_cli_args = _pre.parse_known_args()

    parser = argparse.ArgumentParser(parents=[_pre], description="Train DeepGazeIII with DenseNet+SPADE",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # --- Experiment Configuration ---
    parser.add_argument('--stage', choices=['salicon_pretrain_densenet_spade', 'mit_spatial_densenet_spade',
                                            'mit_scanpath_frozen_densenet_spade', 'mit_scanpath_full_densenet_spade'],
                        help='Training stage to execute.')
    parser.add_argument('--densenet_model_name', default='densenet201', choices=['densenet161', 'densenet201'])
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])

    # --- SPADE Specific Arguments ---
    parser.add_argument('--num_total_segments', type=int, default=16)
    parser.add_argument('--seg_embedding_dim', type=int, default=64)

    # --- Segmentation Mask Configuration (General) ---
    parser.add_argument('--segmentation_mask_format', default='png', choices=['png', 'npy'])
    parser.add_argument('--segmentation_mask_bank_dtype', type=str, default='uint8', choices=['uint8', 'uint16'])

    # --- DIRECT Paths to Individual Mask Folders ---
    parser.add_argument('--salicon_train_mask_dir', type=str, help="DIRECT path to SALICON training individual mask files.")
    parser.add_argument('--salicon_val_mask_dir', type=str, help="DIRECT path to SALICON validation individual mask files.")
    parser.add_argument('--mit_all_mask_dir', type=str, help="DIRECT path to ALL MIT1003 individual mask files.")

    # --- Mask Bank Paths ---
    parser.add_argument('--train_mask_memmap_file', type=str, help="Fixed-size bank for TRAIN SALICON masks.")
    parser.add_argument('--val_mask_memmap_file', type=str, help="Fixed-size bank for VAL SALICON masks.")
    parser.add_argument('--train_mask_variable_payload_file', type=str, help="Variable-size PAYLOAD for TRAIN SALICON.")
    parser.add_argument('--train_mask_variable_header_file', type=str, help="Variable-size HEADER for TRAIN SALICON.")
    parser.add_argument('--val_mask_variable_payload_file', type=str, help="Variable-size PAYLOAD for VAL SALICON.")
    parser.add_argument('--val_mask_variable_header_file', type=str, help="Variable-size HEADER for VAL SALICON.")
    parser.add_argument('--mit_mask_fixed_memmap_file', type=str, help="Fixed-size bank for ALL MIT1003 masks.")
    parser.add_argument('--mit_mask_variable_payload_file', type=str, help="Variable-size PAYLOAD for ALL MIT1003 masks.")
    parser.add_argument('--mit_mask_variable_header_file', type=str, help="Variable-size HEADER for ALL MIT1003 masks.")

    # --- Training Hyperparameters ---
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

    # --- Dataloading & System ---
    parser.add_argument('--num_workers', type=str, default='auto')
    parser.add_argument('--train_dir', type=str, default='./experiments_dg3_spade')
    parser.add_argument('--dataset_dir', type=str, default='./data/pysaliency_datasets')
    parser.add_argument('--lmdb_dir', type=str, default='./data/lmdb_caches')
    parser.add_argument('--use_lmdb_images', action=argparse.BooleanOptionalAction, default=True)

    # --- MIT Stage Specific Arguments ---
    parser.add_argument('--fold', type=int, help='MIT1003 fold (0-9).')
    parser.add_argument('--lr_mit_spatial', type=float, default=1e-4, help="LR for MIT spatial fine-tuning.")
    parser.add_argument('--lr_milestones_mit_spatial', type=int, nargs='+', default=[10, 20])
    parser.add_argument('--salicon_checkpoint_path', type=str, help='Path to SALICON pretrained checkpoint for MIT.')


    if _cfg_namespace.config_file:
        try:
            config_file_path = Path(_cfg_namespace.config_file)
            if not config_file_path.is_absolute():
                config_file_path = (PROJECT_ROOT / config_file_path).resolve()
            
            with open(config_file_path, 'r') as f:
                yaml_cfg = yaml.safe_load(f) or {}
            _module_logger.info(f"Loaded YAML config from: {config_file_path}")
            # Update parser defaults with YAML values, CLI will override them
            # Filter yaml_cfg to only include keys that are valid arguments
            valid_yaml_cfg = {k: v for k, v in yaml_cfg.items() if hasattr(parser.parse_args([]), k)}
            parser.set_defaults(**valid_yaml_cfg)

        except FileNotFoundError:
            _module_logger.warning(f"YAML config file not found: {config_file_path if 'config_file_path' in locals() else _cfg_namespace.config_file}")
        except Exception as e:
            _module_logger.error(f"Could not read/parse YAML '{config_file_path if 'config_file_path' in locals() else _cfg_namespace.config_file}': {e}")

    final_args_ns = parser.parse_args(_remaining_cli_args)
    
    world_size_env = int(os.environ.get("WORLD_SIZE", 1))
    if isinstance(final_args_ns.num_workers, str) and final_args_ns.num_workers.lower() == 'auto':
        final_args_ns.num_workers = None
    if final_args_ns.num_workers is None:
        try: cpu_count = len(os.sched_getaffinity(0))
        except AttributeError: cpu_count = os.cpu_count() or 1
        # Default to min(8, cpu_count_per_gpu) or something reasonable
        # Max 8 workers per GPU is often a good upper bound to prevent I/O bottlenecks or excessive memory.
        final_args_ns.num_workers = min(8, cpu_count // world_size_env if world_size_env > 0 else cpu_count)
        if final_args_ns.num_workers == 0 and cpu_count > 0 : final_args_ns.num_workers = 1 # Ensure at least 1 if cpus are available
    else:
        try: final_args_ns.num_workers = int(final_args_ns.num_workers)
        except ValueError: _module_logger.warning(f"Invalid num_workers='{final_args_ns.num_workers}', using 0."); final_args_ns.num_workers = 0
        if final_args_ns.num_workers < 0: _module_logger.warning(f"Negative num_workers='{final_args_ns.num_workers}', using 0."); final_args_ns.num_workers = 0


    if final_args_ns.stage is None: parser.error("--stage is required.")
    if final_args_ns.num_total_segments <= 0: parser.error("--num_total_segments must be > 0.")

    is_spade_stage = 'spade' in final_args_ns.stage
    if is_spade_stage:
        current_individual_mask_dir_arg = None
        current_fixed_bank_arg = None
        current_var_payload_arg = None
        current_var_header_arg = None
        mask_source_type_name = ""

        if final_args_ns.stage == 'salicon_pretrain_densenet_spade':
            current_individual_mask_dir_arg = final_args_ns.salicon_train_mask_dir 
            current_fixed_bank_arg = final_args_ns.train_mask_memmap_file
            current_var_payload_arg = final_args_ns.train_mask_variable_payload_file
            current_var_header_arg = final_args_ns.train_mask_variable_header_file
            mask_source_type_name = "SALICON Train"
        elif 'mit_' in final_args_ns.stage and 'spade' in final_args_ns.stage: # Be more specific for MIT SPADE stages
            if final_args_ns.fold is None or not (0 <= final_args_ns.fold < 10):
                parser.error("--fold (0-9) is required for MIT SPADE stages.")
            current_individual_mask_dir_arg = final_args_ns.mit_all_mask_dir
            current_fixed_bank_arg = final_args_ns.mit_mask_fixed_memmap_file
            current_var_payload_arg = final_args_ns.mit_mask_variable_payload_file
            current_var_header_arg = final_args_ns.mit_mask_variable_header_file
            mask_source_type_name = "MIT All"
        
        has_individual_masks = bool(current_individual_mask_dir_arg)
        has_fixed_bank = bool(current_fixed_bank_arg)
        has_variable_bank = bool(current_var_payload_arg and current_var_header_arg)

        if mask_source_type_name and not (has_individual_masks or has_fixed_bank or has_variable_bank): # Check if source type determined
            parser.error(f"For SPADE stage '{final_args_ns.stage}', a mask source for '{mask_source_type_name}' is required. "
                         f"Provide a direct individual mask directory OR bank files.")
    
    def resolve_path_arg(arg_value):
        if arg_value is None: return None
        path = Path(arg_value)
        return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()

    final_args_ns.dataset_dir = resolve_path_arg(final_args_ns.dataset_dir)
    final_args_ns.lmdb_dir = resolve_path_arg(final_args_ns.lmdb_dir)
    final_args_ns.train_dir = resolve_path_arg(final_args_ns.train_dir)
    
    final_args_ns.salicon_train_mask_dir = resolve_path_arg(final_args_ns.salicon_train_mask_dir)
    final_args_ns.salicon_val_mask_dir = resolve_path_arg(final_args_ns.salicon_val_mask_dir)
    final_args_ns.mit_all_mask_dir = resolve_path_arg(final_args_ns.mit_all_mask_dir)
    
    final_args_ns.train_mask_memmap_file = resolve_path_arg(final_args_ns.train_mask_memmap_file)
    final_args_ns.val_mask_memmap_file = resolve_path_arg(final_args_ns.val_mask_memmap_file)
    final_args_ns.train_mask_variable_payload_file = resolve_path_arg(final_args_ns.train_mask_variable_payload_file)
    final_args_ns.train_mask_variable_header_file = resolve_path_arg(final_args_ns.train_mask_variable_header_file)
    final_args_ns.val_mask_variable_payload_file = resolve_path_arg(final_args_ns.val_mask_variable_payload_file)
    final_args_ns.val_mask_variable_header_file = resolve_path_arg(final_args_ns.val_mask_variable_header_file)
    final_args_ns.mit_mask_fixed_memmap_file = resolve_path_arg(final_args_ns.mit_mask_fixed_memmap_file)
    final_args_ns.mit_mask_variable_payload_file = resolve_path_arg(final_args_ns.mit_mask_variable_payload_file)
    final_args_ns.mit_mask_variable_header_file = resolve_path_arg(final_args_ns.mit_mask_variable_header_file)
    
    final_args_ns.salicon_checkpoint_path = resolve_path_arg(final_args_ns.salicon_checkpoint_path)
    final_args_ns.resume_checkpoint = resolve_path_arg(final_args_ns.resume_checkpoint)

    try:
        main(final_args_ns) 
    except KeyboardInterrupt:
        _module_logger.warning("Training interrupted by user (Ctrl+C).")
        cleanup_distributed() 
        sys.exit(130)
    except Exception as e:
        _module_logger.critical("Unhandled exception during main execution:", exc_info=True)
        cleanup_distributed() 
        sys.exit(1)