#!/usr/bin/env python
"""
Multi-GPU-ready training script for DeepGaze III with DenseNet + SPADE.
Based on the DINOv2 training script structure.
"""
import os
import sys

# Add project root to sys.path for local module imports
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import yaml # <--- Add this import
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
from DeepGaze.deepgaze_pytorch.features.densenet import RGBDenseNet201 # IMPORTANT
import numpy as np
import math

import pysaliency
import pysaliency.external_datasets.mit # Ensure this can be imported
from PIL import Image
Image.MAX_IMAGE_PIXELS = None # Disable PIL decompression bomb check
from pysaliency.baseline_utils import BaselineModel, CrossvalidatedBaselineModel
import cloudpickle as cpickle

# --- Local Project Imports ---
try:
    # Data handling classes (ensure ImageDatasetWithSegmentation is defined in data_utils.py)
    from src.data import (
        ImageDatasetWithSegmentation, # Key change
        ImageDataset, # If you kept the original name as this
        FixationMaskTransform,
        # The prepare_... functions might need adaptation to pass segmentation_mask_dir
        # For now, we'll instantiate datasets directly in main for this script
        convert_stimuli, convert_fixation_trains # If used for MIT preprocessing
    )
    # Original DeepGaze modules and layers that we will reuse
    from src.modules import DeepGazeIII, FeatureExtractor, DeepGazeII, FeatureExtractor, Finalizer, encode_scanpath_features, build_fixation_selection_network, build_scanpath_network
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
        self.segmap_input_channels = segmap_input_channels
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

    def forward(self, x, segmap_embedded): # Expects embedded segmap
        if x.shape[1] != self.norm_features:
            raise ValueError(f"Input x chans {x.shape[1]} != norm_features {self.norm_features}")
        
        normalized_shape = (self.norm_features, x.size(2), x.size(3))
        normalized_x = F.layer_norm(x, normalized_shape, weight=None, bias=None, eps=self.eps)
        
        # segmap_embedded is already (B, embedding_dim, H_seg, W_seg) and float
        if segmap_embedded.ndim != 4:
             raise ValueError(f"Expected segmap_embedded to be 4D, got {segmap_embedded.ndim}")
        if segmap_embedded.shape[1] != self.segmap_input_channels:
             _logger.error(f"Embedded segmap channels {segmap_embedded.shape[1]} != expected {self.segmap_input_channels}")
             raise ValueError("Embedded segmap channel mismatch")

        segmap_resized = F.interpolate(segmap_embedded, size=x.size()[2:], mode='nearest')
        shared_features = self.mlp_shared(segmap_resized)
        gamma_map = self.mlp_gamma(shared_features)
        beta_map = self.mlp_beta(shared_features)
        out = normalized_x * (1 + gamma_map) + beta_map
        return out

    def extra_repr(self):
        return (f'norm_features={self.norm_features}, '
                f'segmap_input_channels={self.segmap_input_channels}, eps={self.eps}')


class SaliencyNetworkSPADE(nn.Module):
    def __init__(self, input_channels, num_total_segments=17, seg_embedding_dim=64, add_sa_head=False):
        super().__init__()
        self.input_channels = input_channels
        self.num_total_segments = num_total_segments
        self.seg_embedding_dim = seg_embedding_dim
        self.add_sa_head = add_sa_head # Note: SA logic not fully implemented here for brevity

        self.seg_embedder = nn.Embedding(self.num_total_segments, self.seg_embedding_dim)

        current_channels = input_channels # Will be updated if SA head is added and changes channels
        
        # Optional SelfAttention head (copied from original build_saliency_network)
        # This SA head operates on the original backbone features *before* SPADE kicks in.
        # Or, SPADE could be applied, then SA, then more SPADE. Order is experimental.
        # For now, let's assume SA is on the input features if enabled.
        if self.add_sa_head:
            self.sa_head_module = SelfAttention(current_channels, key_channels=current_channels // 8, return_attention=False)
            self.sa_ln_out = LayerNorm(current_channels) # Standard LayerNorm after SA
            self.sa_softplus_out = nn.Softplus()
            _logger.info(f"SaliencyNetworkSPADE: Added SA head. Input channels to SA: {current_channels}")
            # current_channels remains the same if SA outputs same number of channels

        # SPADE-based saliency readout
        self.spade_ln0 = SPADELayerNorm(current_channels, self.seg_embedding_dim) # segmap_input_channels is embedding_dim
        self.conv0 = nn.Conv2d(current_channels, 8, (1, 1), bias=False)
        self.bias0 = Bias(8)
        self.softplus0 = nn.Softplus()

        self.spade_ln1 = SPADELayerNorm(8, self.seg_embedding_dim)
        self.conv1 = nn.Conv2d(8, 16, (1, 1), bias=False)
        self.bias1 = Bias(16)
        self.softplus1 = nn.Softplus()

        self.spade_ln2 = SPADELayerNorm(16, self.seg_embedding_dim)
        self.conv2 = nn.Conv2d(16, 1, (1, 1), bias=False)
        self.bias2 = Bias(1)
        self.softplus2 = nn.Softplus() # Ensures non-negative before finalizer (original DeepGaze behavior)

    def forward(self, x, raw_segmap_long): # raw_segmap expected as (B, H_img, W_img) LongTensor
        # 1. Embed the raw integer segmentation map
        if raw_segmap_long.dtype != torch.long:
            _logger.warning(f"raw_segmap_long dtype is {raw_segmap_long.dtype}, converting to long.")
            raw_segmap_long = raw_segmap_long.long()
            
        B, H_seg, W_seg = raw_segmap_long.shape
        # Clamp values to be safe for embedding layer indices (0 to num_total_segments-1)
        clamped_segmap = torch.clamp(raw_segmap_long.view(B, -1), 0, self.num_total_segments - 1)
        embedded_segmap = self.seg_embedder(clamped_segmap) # (B, H_seg*W_seg, seg_embedding_dim)
        
        embedded_segmap = embedded_segmap.view(B, H_seg, W_seg, self.seg_embedding_dim)
        embedded_segmap = embedded_segmap.permute(0, 3, 1, 2).contiguous() # (B, seg_embedding_dim, H_seg, W_seg)

        # 2. Optional SelfAttention on input features (x)
        h = x
        if self.add_sa_head:
            h, _ = self.sa_head_module(h) # Assuming sa_head_module matches SelfAttention signature
            h = self.sa_ln_out(h)
            h = self.sa_softplus_out(h)

        # 3. Pass through SPADE-enhanced convolutional layers
        h = self.spade_ln0(h, embedded_segmap)
        h = self.conv0(h)
        h = self.bias0(h)
        h = self.softplus0(h)

        h = self.spade_ln1(h, embedded_segmap)
        h = self.conv1(h)
        h = self.bias1(h)
        h = self.softplus1(h)

        h = self.spade_ln2(h, embedded_segmap)
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
        if hasattr(self.features, 'parameters'): # Freeze backbone if it has parameters
            for param in self.features.parameters():
                param.requires_grad = False
        if hasattr(self.features, 'eval'):
            self.features.eval()

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
        # Here's the crucial part: check type and pass segmentation_mask
        if segmentation_mask is None and isinstance(self.saliency_network, SaliencyNetworkSPADE):
            # If SPADE network is used but no mask provided, create a dummy one or raise error
            # For now, let's assume dummy mask handling is inside SaliencyNetworkSPADE or SPADELayerNorm
            # or that the dataloader ALWAYS provides one (even if all zeros).
            # The SaliencyNetworkSPADE expects the raw_segmap_long.
            _logger = logging.getLogger(__name__) # Get a logger if not class member
            _logger.warning("SaliencyNetworkSPADE used but no segmentation_mask provided to DeepGazeIIISpade.forward(). This might error or lead to unexpected behavior.")
            # Create a dummy mask if absolutely necessary, but this should ideally be handled by dataloader
            # b_dummy, _, h_dummy, w_dummy = image.shape # Use original image shape for dummy mask
            # segmentation_mask = torch.zeros((b_dummy, h_dummy, w_dummy), dtype=torch.long, device=image.device)


        if isinstance(self.saliency_network, SaliencyNetworkSPADE):
            if segmentation_mask is None:
                 raise ValueError("SaliencyNetworkSPADE requires a segmentation_mask, but None was provided.")
            saliency_path_output = self.saliency_network(concatenated_backbone_features, segmentation_mask)
        else: # Original saliency network (doesn't take seg_mask)
            saliency_path_output = self.saliency_network(concatenated_backbone_features)

        # 4. Scanpath Readout Head (if used)
        scanpath_path_output = None
        if self.scanpath_network is not None:
            # Ensure x_hist and y_hist are not empty tensors if scanpath_network is active
            if x_hist.numel() == 0 or y_hist.numel() == 0: # Check if tensor is empty
                # This case implies scanpath_network is active, but dataloader didn't provide history
                # This should ideally not happen if dataset and model config are consistent.
                raise ValueError(
                    "Scanpath network is active, but x_hist or y_hist is an empty tensor. "
                    "Ensure dataloader provides scanpath history for this configuration."
                )
            # Proceed with encoding only if x_hist and y_hist are valid (non-empty)
            scanpath_features_encoded = encode_scanpath_features(
                x_hist, y_hist,
                size=(orig_shape[2], orig_shape[3]),
                device=image.device
            )
            # Resize scanpath features to match the readout spatial dimensions
            scanpath_features_resized = F.interpolate(scanpath_features_encoded,
                                                      size=readout_spatial_shape,
                                                      mode='bilinear', align_corners=False)
            
            # If scanpath_network also becomes SPADE-enhanced in the future:
            # if isinstance(self.scanpath_network, ScanpathNetworkSPADE): # Hypothetical
            #     scanpath_path_output = self.scanpath_network(scanpath_features_resized, segmentation_mask)
            # else: # Original scanpath network
            scanpath_path_output = self.scanpath_network(scanpath_features_resized)
        
        # 5. Fixation Selection Network (combines saliency and scanpath outputs)
        # The fixation_selection_network expects a tuple: (saliency_features, scanpath_features)
        # scanpath_path_output will be None if self.scanpath_network is None.
        # The original Conv2dMultiInput and LayerNormMultiInput handle None inputs in the tuple.
        combined_input_for_fixsel = (saliency_path_output, scanpath_path_output)
        final_readout_before_finalizer = self.fixation_selection_network(combined_input_for_fixsel)
        
        # 6. Finalizer (resizing, smoothing, centerbias, normalization)
        saliency_log_density = self.finalizer(final_readout_before_finalizer, centerbias)
        
        return saliency_log_density

    def train(self, mode=True): # From original DeepGazeIII
        # Backbone features are frozen, so their mode doesn't change from eval
        if hasattr(self.features, 'eval'): # Check if features module has eval method
            self.features.eval()
        
        self.saliency_network.train(mode=mode)
        if self.scanpath_network is not None:
            self.scanpath_network.train(mode=mode)
        self.fixation_selection_network.train(mode=mode)
        self.finalizer.train(mode=mode) # Finalizer has learnable params (sigma, center_bias_weight)




# ============================================================================
# == MAIN FUNCTION ==
# ============================================================================
def main(args):
    device, rank, world, is_master, is_distributed = init_distributed()

    # --- Logging Setup ---
    log_level = logging.INFO if is_master else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format=f"%(asctime)s Rank{rank} %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True # Override any root logger setup
    )
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)
    _logger.setLevel(log_level)

    # --- Log Arguments ---
    if is_master:
        _logger.info("==================================================")
        _logger.info(f"Starting DenseNet+SPADE Training: Stage '{args.stage}'")
        _logger.info(f"  Torch Version: {torch.__version__}, CUDA: {torch.cuda.is_available()} (Version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'})")
        _logger.info(f"  DDP: Rank {rank}/{world}, Master: {is_master}, Distributed: {is_distributed}, Device: {device}")
        _logger.info(f"  DenseNet Model: {args.densenet_model_name}")
        _logger.info(f"  Batch Size Per GPU: {args.batch_size}, Grad Accum: {args.gradient_accumulation_steps}")
        _logger.info(f"  Effective Global Batch Size: {args.batch_size * world * args.gradient_accumulation_steps}")
        _logger.info(f"  LR: {args.lr}, Min LR: {args.min_lr}")
        _logger.info(f"  Workers Per Rank: {args.num_workers}")
        _logger.info(f"  Output Base Dir: {args.train_dir}")
        _logger.info(f"  Dataset Dir: {args.dataset_dir}")
        _logger.info(f"  LMDB Cache Dir (Images): {args.lmdb_dir}")
        _logger.info(f"  Segmentation Mask Root Dir: {args.segmentation_mask_dir}")
        _logger.info(f"  Train Mask Subdir: {args.train_mask_subdir_name}, Val Mask Subdir: {args.val_mask_subdir_name}")
        _logger.info(f"  Mask Format: {args.segmentation_mask_format}")
        _logger.info(f"  Num Segments (for embedding): {args.num_total_segments}, Seg Embedding Dim: {args.seg_embedding_dim}")
        _logger.info(f"  Add SA Head to Saliency Net: {args.add_sa_head}")
        _logger.info("==================================================")

    # --- Path Setup ---
    dataset_directory = Path(args.dataset_dir).resolve()
    # train_directory is the base for all experiment outputs
    # Specific experiment output_dir will be created under this
    train_output_base_dir = Path(args.train_dir).resolve()
    lmdb_image_cache_dir = Path(args.lmdb_dir).resolve() # For image LMDBs
    segmentation_mask_root_dir = Path(args.segmentation_mask_dir).resolve()

    if is_master:
        for p in (dataset_directory, train_output_base_dir, lmdb_image_cache_dir, segmentation_mask_root_dir):
            try:
                p.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                _logger.critical(f"Failed to create directory {p}: {e}. Check permissions.")
                if is_distributed: dist.barrier()
                sys.exit(1)
    if is_distributed:
        dist.barrier() # Wait for master to create dirs

    # --- Backbone Setup: DenseNet ---
    if is_master: _logger.info(f"Initializing RGBDenseNet201 backbone...")
    densenet_base_model = RGBDenseNet201()
    saliency_map_factor_densenet = 4 # Standard, from your example
    features_module = FeatureExtractor(densenet_base_model, [
                '1.features.denseblock4.denselayer32.norm1',
                '1.features.denseblock4.denselayer32.conv1',
                '1.features.denseblock4.denselayer31.conv2',
            ])
    
    for param in features_module.parameters(): # Freezing the entire DenseNet
        param.requires_grad = False
    features_module.eval()
    features_module = features_module.to(device)
    if is_master: _logger.info("RGBDenseNet201 with FeatureExtractor initialized and frozen.")

    # --- Stage Dispatch (Focus on SALICON pretraining with DenseNet+SPADE) ---
    if args.stage == 'salicon_pretrain_densenet_spade':
        if is_master: _logger.info(f"--- Preparing SALICON Pretraining with DenseNet+SPADE ---")

        # Load SALICON data and compute/load baseline log-likelihoods
        if is_master: _logger.info(f"Loading SALICON data from {dataset_directory}...")
        salicon_train_loc = dataset_directory
        salicon_val_loc = dataset_directory
        # Ensure SALICON data is downloaded if not present (master only)
        if is_master:
            try:
                pysaliency.get_SALICON_train(location=salicon_train_loc)
                pysaliency.get_SALICON_val(location=salicon_val_loc)
            except Exception as e:
                _logger.critical(f"Failed to download/access SALICON data at {salicon_train_loc}: {e}")
                if is_distributed: dist.barrier(); sys.exit(1)
        if is_distributed: dist.barrier() # Wait for master to potentially download

        try:
            SALICON_train_stimuli, SALICON_train_fixations = pysaliency.get_SALICON_train(location=salicon_train_loc)
            SALICON_val_stimuli, SALICON_val_fixations = pysaliency.get_SALICON_val(location=salicon_val_loc)
        except Exception as e:
            _logger.critical(f"Failed to load SALICON metadata from {salicon_train_loc}: {e}")
            if is_distributed: dist.barrier(); sys.exit(1)
        if is_master: _logger.info("SALICON data loaded.")

        # SALICON Centerbias and Baseline LL
        if is_master: _logger.info("Initializing SALICON BaselineModel for centerbias...")
        # Bandwidth and eps from original DINOv2 script for SALICON
        SALICON_centerbias = BaselineModel(stimuli=SALICON_train_stimuli, fixations=SALICON_train_fixations,
                                           bandwidth=0.0217, eps=2e-13, caching=False) # Caching=False for BaselineModel itself
        
        # Paths for cached baseline log likelihoods
        train_ll_cache_file = dataset_directory / f'salicon_baseline_train_ll_{args.densenet_model_name}.pkl' # Make cache name specific
        val_ll_cache_file = dataset_directory / f'salicon_baseline_val_ll_{args.densenet_model_name}.pkl'
        train_baseline_log_likelihood, val_baseline_log_likelihood = None, None

        if is_master:
            # Try loading cached train LL
            try:
                with open(train_ll_cache_file, 'rb') as f: train_baseline_log_likelihood = cpickle.load(f)
                _logger.info(f"Loaded cached TRAIN baseline LL from: {train_ll_cache_file}")
            except Exception:
                _logger.warning(f"TRAIN LL cache miss or error. Computing...");
                train_baseline_log_likelihood = SALICON_centerbias.information_gain(SALICON_train_stimuli, SALICON_train_fixations, verbose=True, average='image')
                with open(train_ll_cache_file, 'wb') as f: cpickle.dump(train_baseline_log_likelihood, f)
                _logger.info(f"Saved computed TRAIN baseline LL to: {train_ll_cache_file}")
            # Try loading cached val LL
            try:
                with open(val_ll_cache_file, 'rb') as f: val_baseline_log_likelihood = cpickle.load(f)
                _logger.info(f"Loaded cached VALIDATION baseline LL from: {val_ll_cache_file}")
            except Exception:
                _logger.warning(f"VALIDATION LL cache miss or error. Computing...");
                val_baseline_log_likelihood = SALICON_centerbias.information_gain(SALICON_val_stimuli, SALICON_val_fixations, verbose=True, average='image')
                with open(val_ll_cache_file, 'wb') as f: cpickle.dump(val_baseline_log_likelihood, f)
                _logger.info(f"Saved computed VALIDATION baseline LL to: {val_ll_cache_file}")
            _logger.info(f"Master Baseline LLs - Train: {train_baseline_log_likelihood:.5f}, Val: {val_baseline_log_likelihood:.5f}")

        # Broadcast baseline LLs to other ranks
        ll_list_to_broadcast = [train_baseline_log_likelihood, val_baseline_log_likelihood]
        if is_distributed:
            # Ensure all workers have a value, even if master failed (e.g. assign NaN)
            if is_master and (train_baseline_log_likelihood is None or val_baseline_log_likelihood is None):
                _logger.error("Master failed to get baseline LLs, broadcasting NaNs.")
                ll_list_to_broadcast = [float('nan'), float('nan')]
            dist.broadcast_object_list(ll_list_to_broadcast, src=0)
        train_baseline_log_likelihood, val_baseline_log_likelihood = ll_list_to_broadcast
        
        if np.isnan(train_baseline_log_likelihood) or np.isnan(val_baseline_log_likelihood):
             _logger.critical(f"Baseline LLs invalid (NaN) on rank {rank}. Exiting.")
             if is_distributed: dist.barrier(); sys.exit(1)
        else:
            _logger.info(f"Rank {rank} Baseline LLs - Train: {train_baseline_log_likelihood:.5f}, Val: {val_baseline_log_likelihood:.5f}")


        # --- Build Model Components ---
        saliency_net_spade = SaliencyNetworkSPADE(
            input_channels=2048,
            num_total_segments=args.num_total_segments,
            seg_embedding_dim=args.seg_embedding_dim,
            add_sa_head=args.add_sa_head
        )
        # For this experiment, scanpath network is None, and fixation selection uses original layers
        scanpath_net = None #build_scanpath_network() # testing scanpaths without SPADE
        fixsel_net = build_fixation_selection_network(scanpath_features=0) # scanpath_features=0 as scanpath_net is None
        
        model = DeepGazeIIISpade(
            features=features_module, # Our DenseNetFeatureExtractor instance
            saliency_network=saliency_net_spade,
            scanpath_network=scanpath_net,
            fixation_selection_network=fixsel_net,
            downsample=1, # DenseNet features are already strided by backbone
            readout_factor=4,
            saliency_map_factor=saliency_map_factor_densenet, # Standard for DeepGaze finalizer
            included_fixations=[] # Spatial pretraining, no history
        ).to(device)

        if is_master: _logger.info("DeepGazeIII model with DenseNet backbone and SaliencyNetworkSPADE built.")

        if is_distributed:
            # find_unused_parameters=True is safer when starting with new architectures
            # or if parts of the model (like scanpath_network here) might be None.
            model = DDP(model, device_ids=[device.index], output_device=device.index,
                        broadcast_buffers=False, find_unused_parameters=True)
            if is_master: _logger.info("Wrapped model with DDP (find_unused_parameters=True).")

        # Optimizer: only head parameters (DenseNet is frozen)
        # All trainable parameters in `model` are part of the "head" (saliency, spade_mlps, fixsel, finalizer)
        head_params = [p for p in model.parameters() if p.requires_grad]
        if not head_params:
            _logger.critical("No trainable parameters found in the model! Check requires_grad settings.")
            if is_distributed: dist.barrier(); sys.exit(1)
            
        optimizer = optim.Adam(head_params, lr=args.lr)
        # Adjust milestones for DenseNet if needed, usually fewer epochs than full DINOv2 training
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones) # Use arg for milestones

        # --- Dataloaders with Segmentation Masks ---
        train_mask_dir_full = segmentation_mask_root_dir / args.train_mask_subdir_name
        val_mask_dir_full = segmentation_mask_root_dir / args.val_mask_subdir_name
        if is_master:
            _logger.info(f"Using TRAIN segmentation masks from: {train_mask_dir_full}")
            _logger.info(f"Using VAL segmentation masks from: {val_mask_dir_full}")
            if not train_mask_dir_full.exists(): _logger.error(f"Train mask directory NOT FOUND: {train_mask_dir_full}")
            if not val_mask_dir_full.exists(): _logger.error(f"Validation mask directory NOT FOUND: {val_mask_dir_full}")
            _logger.info(f"Segmentation mask caching: {'ENABLED' if args.cache_segmentation_masks else 'DISABLED'}")


        # LMDB path for images for SALICON (if you created one for DenseNet features)
        salicon_train_lmdb_path = lmdb_image_cache_dir / f'SALICON_train_images_{args.densenet_model_name}'
        salicon_val_lmdb_path = lmdb_image_cache_dir / f'SALICON_val_images_{args.densenet_model_name}'

        train_dataset = ImageDatasetWithSegmentation(
            stimuli=SALICON_train_stimuli,
            fixations=SALICON_train_fixations,
            centerbias_model=SALICON_centerbias,
            transform=FixationMaskTransform(sparse=False), # Dense target fixation map
            average="image", # As per original DeepGaze pretraining
            segmentation_mask_dir=train_mask_dir_full,
            segmentation_mask_format=args.segmentation_mask_format,
            lmdb_path=str(salicon_train_lmdb_path) if args.use_lmdb_images else None,
            # cached: Let ImageDataset decide or explicitly set False for large SALICON.
            # If lmdb_path is provided, ImageDataset sets its internal `self.cached` to False for image/CB.
            # cached=False, # Example: Explicitly disable parent image/CB caching for SALICON
            
            # --- THIS IS THE KEY CHANGE FOR MASK CACHING ---
            cached_masks=args.cache_segmentation_masks
        )
        train_sampler = (torch.utils.data.DistributedSampler(train_dataset, shuffle=True, drop_last=True)
                         if is_distributed else None)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True,
            persistent_workers=args.num_workers > 0
        )
        if is_master: _logger.info(f"Train Dataloader created. Num batches: {len(train_loader)}")

        val_dataset = ImageDatasetWithSegmentation(
            stimuli=SALICON_val_stimuli,
            fixations=SALICON_val_fixations,
            centerbias_model=SALICON_centerbias,
            transform=FixationMaskTransform(sparse=False),
            average="image",
            segmentation_mask_dir=val_mask_dir_full,
            segmentation_mask_format=args.segmentation_mask_format,
            lmdb_path=str(salicon_val_lmdb_path) if args.use_lmdb_images else None,
            # cached=False, # Example for validation set as well
            
            # --- THIS IS THE KEY CHANGE FOR MASK CACHING ---
            cached_masks=args.cache_segmentation_masks
        )
        val_sampler = (torch.utils.data.DistributedSampler(val_dataset, shuffle=False, drop_last=False)
                       if is_distributed else None)
        validation_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, 
            num_workers=args.num_workers, pin_memory=True, sampler=val_sampler,
            persistent_workers=args.num_workers > 0
        )
        if is_master: _logger.info(f"Validation Dataloader created. Num batches: {len(validation_loader)}")


        # --- Training ---
        # Construct a unique output directory name for this experiment
        experiment_name = (f"{args.stage}_{args.densenet_model_name}_spade_k{args.num_total_segments-1}_emb{args.seg_embedding_dim}"
                           f"_lr{args.lr}_bs{args.batch_size*world*args.gradient_accumulation_steps}")
        output_dir_experiment = train_output_base_dir / experiment_name
        if is_master: _logger.info(f"Experiment output directory: {output_dir_experiment}")
        
        # CRITICAL: Ensure your _train (from src.training) handles passing 'segmentation_mask' to model.forward()
        _train(
            this_directory=str(output_dir_experiment), model=model,
            train_loader=train_loader, train_baseline_log_likelihood=train_baseline_log_likelihood,
            val_loader=validation_loader, val_baseline_log_likelihood=val_baseline_log_likelihood,
            optimizer=optimizer, lr_scheduler=lr_scheduler,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            minimum_learning_rate=args.min_lr,
            validation_metrics=['LL', 'IG', 'NSS', 'AUC_CPU'], # Default metrics
            validation_epochs=args.validation_epochs,
            startwith=args.resume_checkpoint, # For resuming a specific experiment
            device=device,
            is_distributed=is_distributed, is_master=is_master,
            logger=_logger # Pass the logger instance
        )
        if is_master: _logger.info(f"--- Stage '{args.stage}' Finished ---")

    elif args.stage in ['mit_spatial_densenet_spade', 'mit_scanpath_frozen_densenet_spade', 'mit_scanpath_full_densenet_spade']:
        # Placeholder for MIT stages - would require similar adaptation:
        # - Correct prev_stage_dir to point to the SALICON DenseNet+SPADE output
        # - Adapt MIT data loading (stimuli, fixations, centerbias, baseline LLs)
        # - Adapt dataloaders to use ImageDatasetWithSegmentation or FixationDatasetWithSegmentation
        # - Pass correct segmentation_mask_dir for MIT folds
        # - Model building logic for different MIT stages (freezing, LRs)
        fold = args.fold
        if fold is None or not (0 <= fold < 10):
            _logger.critical("--fold required for MIT stages. Exiting.")
            if is_distributed: dist.barrier(); sys.exit(1)
        if is_master: _logger.info(f"--- Preparing MIT1003 Stage: {args.stage} (Fold {fold}) ---")
        _logger.warning(f"MIT stage '{args.stage}' with DenseNet+SPADE is not fully implemented in this script version.")
        _logger.warning("You will need to adapt data loading, checkpoint paths, and potentially model freezing logic.")
        # ... (Detailed MIT stage implementation would go here) ...

    else:
        _logger.critical(f"Unknown or unsupported stage for this script: {args.stage}")
        if is_distributed: dist.barrier()
        sys.exit(1)

    cleanup_distributed()
    if is_master:
        _logger.info("==================================================")
        _logger.info("Training script finished successfully.")
        _logger.info("==================================================")


# ============================================================================
# == CLI ARGUMENT PARSER & YAML Loading ==
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeepGazeIII with DenseNet+SPADE (Multi-GPU & YAML Config)")

    # --- Special Argument for YAML Config File ---
    parser.add_argument('--config_file', type=str, default=None,
                        help='Path to a YAML configuration file. CLI args will override YAML values.')

    # --- All your existing arguments ---
    # (Define them here as before, so they can act as defaults or overrides)
    # --- Experiment Configuration ---
    parser.add_argument('--stage', choices=['salicon_pretrain_densenet_spade',
                                 'mit_spatial_densenet_spade',
                                 'mit_scanpath_frozen_densenet_spade',
                                 'mit_scanpath_full_densenet_spade'],
                        help='Training stage to execute.')
    parser.add_argument('--densenet_model_name', choices=['densenet161', 'densenet201'],
                        help='DenseNet model variant.')

    # --- SPADE Specific Arguments ---
    parser.add_argument('--segmentation_mask_dir', type=str,
                        help='Root directory for segmentation masks.')
    parser.add_argument('--train_mask_subdir_name', type=str,
                        help='Subdirectory for training set masks.')
    parser.add_argument('--val_mask_subdir_name', type=str,
                        help='Subdirectory for validation set masks.')
    parser.add_argument('--segmentation_mask_format', choices=['png', 'npy'],
                        help='File format of segmentation masks.')
    parser.add_argument('--num_total_segments', type=int,
                        help='Total number of unique segment IDs (K for K-Means 0 to K-1).')
    parser.add_argument('--seg_embedding_dim', type=int, help='Dimension for segment embeddings.')

    # --- Training Hyperparameters ---
    parser.add_argument('--batch_size', type=int, help='Batch size *per GPU*.')
    parser.add_argument('--gradient_accumulation_steps', type=int, help='Gradient accumulation steps.')
    parser.add_argument('--lr', type=float, help='Initial learning rate.')
    parser.add_argument('--lr_milestones', type=int, nargs='+', help='Epochs for LR decay.')
    parser.add_argument('--min_lr', type=float, help='Minimum learning rate.')
    parser.add_argument('--fold', type=int, help='Cross-validation fold for MIT1003 (0-9).')
    parser.add_argument('--validation_epochs', type=int, help='Run validation every N epochs.')
    parser.add_argument('--resume_checkpoint', type=str, help='Path to resume checkpoint.')

    # --- Dataloading & System ---
    parser.add_argument('--num_workers', type=str, # Changed to str to handle 'auto' from YAML
                        help="Dataloader workers. 'auto' or integer. If not set, determined automatically.")
    parser.add_argument('--train_dir', help='Base directory for training outputs.')
    parser.add_argument('--dataset_dir', help='Directory for pysaliency datasets.')
    parser.add_argument('--lmdb_dir', help='Directory for LMDB image caches.')
    parser.add_argument('--use_lmdb_images', action=argparse.BooleanOptionalAction, help='Use LMDB for stimulus images.') # Allows --no-use-lmdb-images
    parser.add_argument('--cache_segmentation_masks', action=argparse.BooleanOptionalAction, help='Cache segmentation masks in RAM.')

    # --- Model Generic Arguments ---
    parser.add_argument('--add_sa_head', action=argparse.BooleanOptionalAction, help='Add SelfAttention to saliency network.')

    # --- Initial Parse for Config File Path ---
    # Parse known args first to get config_file path without erroring on other CLI args
    temp_args, remaining_argv = parser.parse_known_args()
    config_args = {}

    if temp_args.config_file:
        config_file_path = Path(temp_args.config_file)
        if config_file_path.is_file():
            logging.info(f"Loading config file: {config_file_path}")
            with open(config_file_path, 'r') as f:
                config_args = yaml.safe_load(f)
            if config_args is None: # Handle empty YAML file
                config_args = {}
        else:
            print(f"Warning: Specified config file not found: {config_file_path}. Using defaults/CLI args.")
            # Potentially exit if config file is required and not found
            # sys.exit(f"Error: Config file {config_file_path} not found.")

    # --- Set Defaults from YAML (or initial parser defaults if no YAML/key missing) ---
    # This allows CLI to override YAML values.
    # We effectively re-parse with YAML values as new defaults.
    # Create a new parser or update defaults of the existing one.
    # For simplicity here, we'll update the existing parser's defaults.
    # For every argument defined in the parser:
    for action in parser._actions:
        if action.dest in config_args and action.dest != "config_file": # Don't override config_file itself
            # Handle different action types correctly (store_true, etc.)
            if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction, argparse.BooleanOptionalAction)):
                # For boolean actions, the value from YAML should directly set the default
                # This is a bit tricky as `set_defaults` expects the default for store_true to be False
                # Let's directly update the default value
                parser.set_defaults(**{action.dest: config_args[action.dest]})
            elif action.default != config_args[action.dest]: # Only update if different
                parser.set_defaults(**{action.dest: config_args[action.dest]})

    # --- Final Parse with updated defaults and remaining CLI args ---
    # `remaining_argv` will be parsed on top of defaults (which now include YAML values)
    final_args = parser.parse_args(remaining_argv)


    # --- Automatic Worker Count ---
    world_size_env = int(os.environ.get("WORLD_SIZE", 1)) # Use a different var name
    # Handle 'auto' for num_workers from YAML/CLI
    if isinstance(final_args.num_workers, str) and final_args.num_workers.lower() == 'auto':
        final_args.num_workers = None # Let the original auto-detection logic handle it

    if final_args.num_workers is None: # If still None (was 'auto' or not set)
        try:
            cpu_count = len(os.sched_getaffinity(0))
        except AttributeError:
            cpu_count = os.cpu_count() or 1
        final_args.num_workers = max(0, cpu_count // world_size_env if world_size_env > 0 else cpu_count)
        if final_args.num_workers == 0 and world_size_env > 1:
            if cpu_count > world_size_env: final_args.num_workers = 1
    elif not isinstance(final_args.num_workers, int) or final_args.num_workers < 0:
        # If it came from YAML/CLI as a non-'auto' string or invalid int, default to 0
        print(f"Warning: Invalid num_workers value '{final_args.num_workers}'. Defaulting to 0.")
        final_args.num_workers = 0


    # --- Argument Validation (Example) ---
    if final_args.stage is None:
        parser.error("The --stage argument (or 'stage' in YAML) is required.")
    if final_args.segmentation_mask_dir is None and "spade" in final_args.stage:
        parser.error("--segmentation_mask_dir (or in YAML) is required for SPADE stages.")
    if final_args.num_total_segments is None or final_args.num_total_segments <= 0:
         parser.error("--num_total_segments (or in YAML) must be a positive integer.")


    try:
        main(final_args) # Pass the final Namespace object
    except KeyboardInterrupt:
        _logger.warning("Training interrupted by user (KeyboardInterrupt). Cleaning up...")
        cleanup_distributed()
        sys.exit(130)
    except Exception as e:
        _logger.critical("Unhandled exception during main execution:", exc_info=True)
        cleanup_distributed()
        sys.exit(1)