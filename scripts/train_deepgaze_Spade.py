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
from PIL import Image
Image.MAX_IMAGE_PIXELS = None 
from pysaliency.baseline_utils import BaselineModel, CrossvalidatedBaselineModel
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
def main(args: argparse.Namespace): # Expects Namespace, use args.attribute
    device, rank, world, is_master, is_distributed = init_distributed()

    # --- Logging Setup ---
    log_level = logging.INFO if is_master else logging.WARNING # Default to INFO for master
    # Override with log_level from args if provided in YAML/CLI, useful for debugging
    if hasattr(args, 'log_level') and args.log_level is not None:
        try:
            log_level = getattr(logging, str(args.log_level).upper(), log_level)
        except AttributeError:
            print(f"Warning: Invalid log_level '{args.log_level}'. Using default.")

    logging.basicConfig(level=log_level, format=f"%(asctime)s Rank{rank} %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", force=True)
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING) # Quieten PIL
    _logger.setLevel(log_level) # Set level for the module's global logger instance

    # --- Log Arguments ---
    if is_master:
        _logger.info("================== Effective Configuration ==================")
        for arg_name, arg_value in sorted(vars(args).items()):
            _logger.info(f"  {arg_name}: {arg_value}")
        _logger.info(f"  DDP Info: Rank {rank}/{world}, Master: {is_master}, Distributed: {is_distributed}, Device: {device}")
        _logger.info(f"  Torch: {torch.__version__}, CUDA: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
        _logger.info("===========================================================")

    # --- Path Setup ---
    dataset_directory = Path(args.dataset_dir).resolve()
    train_output_base_dir = Path(args.train_dir).resolve()
    lmdb_image_cache_dir = Path(args.lmdb_dir).resolve()
    # segmentation_mask_root_dir is for individual mask fallback/source
    segmentation_mask_root_dir = Path(args.segmentation_mask_dir).resolve() if args.segmentation_mask_dir else None

    if is_master:
        for p in [dataset_directory, train_output_base_dir, lmdb_image_cache_dir]: # Removed seg_mask_root_dir from mandatory creation
            if p: p.mkdir(parents=True, exist_ok=True)
        if segmentation_mask_root_dir and not segmentation_mask_root_dir.exists():
             _logger.warning(f"Individual mask root dir {segmentation_mask_root_dir} (for fallback/source) does not exist.")
    if is_distributed: dist.barrier()

    # --- Backbone Setup: DenseNet ---
    if is_master: _logger.info(f"Initializing {args.densenet_model_name} backbone...")
    if args.densenet_model_name == 'densenet201':
        densenet_base_model = RGBDenseNet201()
    # elif args.densenet_model_name == 'densenet161': # Add if you support it
    #     from DeepGaze.deepgaze_pytorch.features.densenet import RGBDenseNet161
    #     densenet_base_model = RGBDenseNet161()
    else:
        _logger.critical(f"Unsupported densenet_model_name: {args.densenet_model_name}"); sys.exit(1)

    saliency_map_factor_densenet = 4
    # These are the specific layers your FeatureExtractor will hook into.
    # Ensure these names are correct for the DenseNet version being used.
    densenet_feature_nodes = [
        '1.features.denseblock4.denselayer32.norm1',
        '1.features.denseblock4.denselayer32.conv1',
        '1.features.denseblock4.denselayer31.conv2',
    ]
    features_module = FeatureExtractor(densenet_base_model, densenet_feature_nodes)
    
    for param in features_module.parameters(): param.requires_grad = False
    features_module.eval().to(device)
    if is_master: _logger.info(f"{args.densenet_model_name} with FeatureExtractor (hooks: {densenet_feature_nodes}) initialized and frozen.")

    # --- Stage Dispatch ---
    if args.stage == 'salicon_pretrain_densenet_spade':
        if is_master: _logger.info(f"--- Preparing SALICON Pretraining with DenseNet+SPADE ---")

        # Load SALICON stimuli and fixations
        if is_master: _logger.info(f"Loading SALICON data from {dataset_directory}...")
        salicon_train_loc = dataset_directory / 'SALICON' # Assuming SALICON data is in a subdir
        salicon_val_loc = dataset_directory / 'SALICON'
        if is_master: # Download if not present
            try:
                if not (salicon_train_loc / 'stimuli' / 'train').exists(): pysaliency.get_SALICON_train(location=str(salicon_train_loc.parent))
                if not (salicon_val_loc / 'stimuli' / 'val').exists(): pysaliency.get_SALICON_val(location=str(salicon_val_loc.parent))
            except Exception as e: _logger.critical(f"Failed SALICON get: {e}"); dist.barrier(); sys.exit(1)
        if is_distributed: dist.barrier()
        SALICON_train_stimuli, SALICON_train_fixations = pysaliency.get_SALICON_train(location=str(salicon_train_loc.parent))
        SALICON_val_stimuli, SALICON_val_fixations = pysaliency.get_SALICON_val(location=str(salicon_val_loc.parent))
        if is_master: _logger.info("SALICON data loaded.")

        # SALICON Centerbias and Baseline LL
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
            if is_master and (train_baseline_log_likelihood is None or val_baseline_log_likelihood is None): ll_bcast = [float('nan'), float('nan')]
            dist.broadcast_object_list(ll_bcast,src=0)
        train_baseline_log_likelihood,val_baseline_log_likelihood = ll_bcast
        if np.isnan(train_baseline_log_likelihood) or np.isnan(val_baseline_log_likelihood): _logger.critical("NaN LLs received/computed on rank {rank}. Exiting."); sys.exit(1)
        else: _logger.info(f"Rank {rank} Baseline LLs - Train: {train_baseline_log_likelihood:.5f}, Val: {val_baseline_log_likelihood:.5f}")

        # --- Determine input channels for SaliencyNetworkSPADE via DUMMY FORWARD PASS ---
        concatenated_channels = 0
        dummy_h, dummy_w = 256, 256 # Default dummy size
        if SALICON_train_stimuli and SALICON_train_stimuli.sizes:
            try:
                h_s, w_s = SALICON_train_stimuli.sizes[0][:2]
                min_dim_factor = 32 # Heuristic for DenseNet based on total stride
                dummy_h, dummy_w = max(h_s, min_dim_factor), max(w_s, min_dim_factor)
                if is_master: _logger.info(f"Using dummy input size from dataset: H={dummy_h}, W={dummy_w} for channel calculation.")
            except Exception as e_sz:
                if is_master: _logger.warning(f"Could not get dataset size for dummy input ({e_sz}), using default H={dummy_h}, W={dummy_w}.")
        
        dummy_input = torch.randn(1, 3, dummy_h, dummy_w).to(device)
        with torch.no_grad():
            extracted_dummy_feature_maps = features_module(dummy_input) # List of tensors

        if not isinstance(extracted_dummy_feature_maps, list) or not all(isinstance(t, torch.Tensor) for t in extracted_dummy_feature_maps):
            _logger.critical(f"FeatureExtractor did not return list of tensors. Got: {type(extracted_dummy_feature_maps)}. Check FeatureExtractor/hooks."); sys.exit(1)
        
        concatenated_channels = sum(feat_map.shape[1] for feat_map in extracted_dummy_feature_maps)

        if concatenated_channels == 0: # Should not happen if hooks are correct
            _logger.error(f"Dummy forward pass resulted in 0 concatenated_channels! Fallback to 2048. Investigate FeatureExtractor hooks: {densenet_feature_nodes}")
            concatenated_channels = 2048 # Previous hardcoded fallback
        
        if is_master:
            _logger.info(f"Determined concatenated input channels for SPADE head: {concatenated_channels}")
            # Try to get return_nodes if FeatureExtractor stores them (torchvision's does)
            actual_return_nodes = getattr(features_module.feature_model if hasattr(features_module, 'feature_model') else features_module, 'return_nodes', densenet_feature_nodes)
            for i, feat_map in enumerate(extracted_dummy_feature_maps):
                node_name = actual_return_nodes[i] if i < len(actual_return_nodes) else f"layer_{i}"
                _logger.info(f"  - Extracted dummy feature map {i} (from node '{node_name}') shape: {feat_map.shape}")


        # --- Build Model Components ---
        saliency_net_spade = SaliencyNetworkSPADE(
            input_channels=concatenated_channels, # Use dynamically determined channels
            num_total_segments=args.num_total_segments,
            seg_embedding_dim=args.seg_embedding_dim,
            add_sa_head=args.add_sa_head
        )
        scanpath_net = None # For SALICON pretraining, no scanpath component
        fixsel_net = build_fixation_selection_network(scanpath_features=0) # scanpath_features=0 if scanpath_net is None

        model = DeepGazeIIISpade(
            features=features_module,
            saliency_network=saliency_net_spade,
            scanpath_network=scanpath_net,
            fixation_selection_network=fixsel_net,
            downsample=1, # DenseNet features are already appropriately strided
            readout_factor=4, # Factor to resize feature maps before readout heads
            saliency_map_factor=saliency_map_factor_densenet,
            included_fixations=[] # No scanpath history for spatial pretraining
        ).to(device)

        if is_master: _logger.info("DeepGazeIII SPADE model built.")
        if is_distributed:
            model = DDP(model, device_ids=[device.index], find_unused_parameters=True) # find_unused if scanpath_net can be None
            if is_master: _logger.info("Wrapped model with DDP.")

        head_params = [p for p in model.parameters() if p.requires_grad]
        if not head_params: _logger.critical("No trainable parameters found!"); sys.exit(1)
        optimizer = optim.Adam(head_params, lr=args.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones)

        # --- Dataloaders ---
        # Fallback individual mask dirs
        train_mask_individual_dir = segmentation_mask_root_dir / args.train_mask_subdir_name if segmentation_mask_root_dir and args.train_mask_subdir_name else None
        val_mask_individual_dir = segmentation_mask_root_dir / args.val_mask_subdir_name if segmentation_mask_root_dir and args.val_mask_subdir_name else None

        # Fixed-size memmap bank paths
        train_fixed_memmap = Path(args.train_mask_memmap_file).resolve() if args.train_mask_memmap_file else None
        val_fixed_memmap = Path(args.val_mask_memmap_file).resolve() if args.val_mask_memmap_file else None

        # Variable-size memmap bank paths
        train_var_payload = Path(args.train_mask_variable_payload_file).resolve() if args.train_mask_variable_payload_file else None
        train_var_header = Path(args.train_mask_variable_header_file).resolve() if args.train_mask_variable_header_file else None
        val_var_payload = Path(args.val_mask_variable_payload_file).resolve() if args.val_mask_variable_payload_file else None
        val_var_header = Path(args.val_mask_variable_header_file).resolve() if args.val_mask_variable_header_file else None

        if is_master:
            _logger.info(f"Train Mask Config: FixedBank='{train_fixed_memmap}', VarBankPayload='{train_var_payload}', VarBankHeader='{train_var_header}', FallbackDir='{train_mask_individual_dir}'")
            _logger.info(f"Val Mask Config: FixedBank='{val_fixed_memmap}', VarBankPayload='{val_var_payload}', VarBankHeader='{val_var_header}', FallbackDir='{val_mask_individual_dir}'")


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
        if train_sampler and hasattr(train_sampler, 'set_epoch'): train_sampler.set_epoch(0) # Initial epoch for sampler

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                                 num_workers=args.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True,
                                                 persistent_workers=args.num_workers > 0)
        if is_master: _logger.info(f"Train Dataloader: {len(train_loader)} batches.")

        val_sampler = (torch.utils.data.DistributedSampler(val_dataset, shuffle=False, drop_last=False) if is_distributed else None)
        # No need to set_epoch for val_sampler if shuffle=False usually

        validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                                     num_workers=args.num_workers, pin_memory=True, sampler=val_sampler,
                                                     persistent_workers=args.num_workers > 0)
        if is_master: _logger.info(f"Validation Dataloader: {len(validation_loader)} batches.")


        # --- Training ---
        #experiment_name = (f"{args.stage}_{args.densenet_model_name}_spade_k{args.num_total_segments}_emb{args.seg_embedding_dim}"
        #                   f"_lr{args.lr}_bs{args.batch_size*world*args.gradient_accumulation_steps}")
        experiment_name = (f"{args.stage}")
        output_dir_experiment = train_output_base_dir / experiment_name
        if is_master: _logger.info(f"Experiment output to: {output_dir_experiment}")

        # In scripts/train_deepgaze_Spade.py
# Ensure all arguments that _train expects are passed correctly.

        _train(
            this_directory=str(output_dir_experiment),
            model=model,
            # --- Explicitly use keywords for ALL subsequent arguments ---
            train_loader=train_loader,
            train_baseline_log_likelihood=train_baseline_log_likelihood,
            val_loader=validation_loader,
            val_baseline_log_likelihood=val_baseline_log_likelihood,
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
            logger=_logger,
        )
        if is_master: _logger.info(f"--- Stage '{args.stage}' Finished ---")

    elif args.stage in ['mit_spatial_densenet_spade', 'mit_scanpath_frozen_densenet_spade', 'mit_scanpath_full_densenet_spade']:
        fold = args.fold
        if fold is None or not (0 <= fold < 10): _logger.critical("--fold required for MIT stages and must be 0-9."); sys.exit(1)
        if is_master: _logger.info(f"--- Preparing MIT1003 Stage: {args.stage} (Fold {fold}) ---")
        _logger.warning(f"MIT stage '{args.stage}' is a placeholder. Full implementation needed for data loading, model setup, etc.")
        # TODO: Implement MIT data loading for the specific fold.
        # This will involve:
        # 1. Getting MIT stimuli and fixations for the current fold (e.g., from a pre-converted location).
        #    You might use your convert_stimuli_mit and convert_fixation_trains_mit here if running for the first time.
        #    `mit_data_path = dataset_directory / f'MIT1003_converted_fold{fold}'`
        # 2. Loading/computing MIT centerbias for the fold.
        # 3. Setting up paths for MIT mask banks (these will be variable-size).
        #    `train_var_payload = Path(args.train_mask_variable_payload_file_mit_fold{fold})` (need new args or pattern)
        # 4. Creating ImageDatasetWithSegmentation instances for MIT train/val splits of the fold.
        # 5. Potentially loading weights from the SALICON pretraining stage into the model.
        # 6. Adjusting optimizer/scheduler if fine-tuning with different LRs.
        # 7. Calling _train.

    else:
        _logger.critical(f"Unknown or unsupported stage: {args.stage}"); sys.exit(1)

    cleanup_distributed()
    if is_master: _logger.info("Training script finished successfully.")


# ──────────────────────────────────────────────────────────────────────────────
#  Main entry-point – YAML + CLI handling
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Basic logging config for startup messages before main's detailed logger
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    _module_logger = logging.getLogger(__name__) # For messages during arg parsing

    _pre = argparse.ArgumentParser(add_help=False) # Pre-parser for config_file
    _pre.add_argument('--config_file', type=str, default=None, help="Path to YAML configuration file.")
    _cfg_namespace, _remaining_cli_args = _pre.parse_known_args()

    # Full parser, inheriting --config_file
    parser = argparse.ArgumentParser(parents=[_pre], description="Train DeepGazeIII with DenseNet+SPADE", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # --- Define ALL arguments that can appear in YAML or CLI ---
    # Experiment Configuration
    parser.add_argument('--stage', choices=['salicon_pretrain_densenet_spade', 'mit_spatial_densenet_spade', 'mit_scanpath_frozen_densenet_spade', 'mit_scanpath_full_densenet_spade'], help='Training stage to execute.')
    parser.add_argument('--densenet_model_name', default='densenet201', choices=['densenet161', 'densenet201'], help='DenseNet variant.')
    parser.add_argument('--log_level', type=str, default=None, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Logging level for detailed logs.')

    # SPADE-specific & Individual Mask Fallback/Source
    parser.add_argument('--segmentation_mask_dir', type=str, help='Root directory for individual segmentation masks (fallback/source).')
    parser.add_argument('--train_mask_subdir_name', type=str, default='train', help='Subdirectory for training set individual masks.')
    parser.add_argument('--val_mask_subdir_name', type=str, default='val', help='Subdirectory for validation set individual masks.')
    parser.add_argument('--segmentation_mask_format', default='png', choices=['png', 'npy'], help='File format of individual masks.')
    parser.add_argument('--num_total_segments', type=int, default=16, help='Number of segment IDs (K for K-Means).')
    parser.add_argument('--seg_embedding_dim', type=int, default=64, help='Segment ID embedding dimension.')

    # Training hyper-parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size per GPU.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning-rate.')
    parser.add_argument('--lr_milestones', type=int, nargs='+', default=[15, 30, 45], help='Epochs for LR decay.')
    parser.add_argument('--min_lr', type=float, default=1e-7, help='Min learning rate.')
    parser.add_argument('--fold', type=int, help='MIT1003 fold (0-9), if applicable.')
    parser.add_argument('--validation_epochs', type=int, default=1, help='Validate every N epochs.')
    parser.add_argument('--resume_checkpoint', type=str, help='Path to checkpoint to resume from.')

    # Data-loading / system
    parser.add_argument('--num_workers', type=str, default='auto', help="'auto' or integer for dataloader workers per rank.")
    parser.add_argument('--train_dir', type=str, default='./experiments_dg3_spade', help="Base output directory for experiments.")
    parser.add_argument('--dataset_dir', type=str, default='./data/pysaliency_datasets', help="Pysaliency datasets directory.")
    parser.add_argument('--lmdb_dir', type=str, default='./data/lmdb_caches', help="LMDB image cache directory.")
    parser.add_argument('--use_lmdb_images', action=argparse.BooleanOptionalAction, default=True, help='Use LMDB for stimuli.')

    # Mask Bank Arguments
    parser.add_argument('--train_mask_memmap_file', type=str, help="Fixed-size memmap bank (.npy) for TRAIN masks.")
    parser.add_argument('--val_mask_memmap_file', type=str, help="Fixed-size memmap bank (.npy) for VAL masks.")
    parser.add_argument('--train_mask_variable_payload_file', type=str, help="Variable-size mask PAYLOAD (.bin) for TRAIN.")
    parser.add_argument('--train_mask_variable_header_file', type=str, help="Companion HEADER (.npy) for TRAIN variable payload.")
    parser.add_argument('--val_mask_variable_payload_file', type=str, help="Variable-size mask PAYLOAD (.bin) for VAL.")
    parser.add_argument('--val_mask_variable_header_file', type=str, help="Companion HEADER (.npy) for VAL variable payload.")
    parser.add_argument('--segmentation_mask_bank_dtype', type=str, default='uint8', choices=['uint8', 'uint16'], help="Dtype for mask bank storage.")

    # Model options
    parser.add_argument('--add_sa_head', action=argparse.BooleanOptionalAction, default=False, help='Add self-attention head to saliency network.')

    # Load YAML and set defaults
    if _cfg_namespace.config_file:
        try:
            with open(_cfg_namespace.config_file, 'r') as f: yaml_cfg = yaml.safe_load(f) or {}
            _module_logger.info(f"Loaded YAML config from: {_cfg_namespace.config_file}")
            parser.set_defaults(**yaml_cfg) # YAML values become defaults
        except FileNotFoundError: _module_logger.warning(f"YAML config file not found: {_cfg_namespace.config_file}")
        except Exception as e: _module_logger.warning(f"Could not read/parse YAML '{_cfg_namespace.config_file}': {e}")

    final_args_ns = parser.parse_args(_remaining_cli_args) # CLI overrides YAML-based defaults

    # Post-processing (num_workers)
    # ... (your existing robust num_workers post-processing logic) ...
    world_size_env = int(os.environ.get("WORLD_SIZE", 1))
    if isinstance(final_args_ns.num_workers, str) and final_args_ns.num_workers.lower() == 'auto':
        final_args_ns.num_workers = None
    if final_args_ns.num_workers is None: # If 'auto' or not set at all
        try: cpu_count = len(os.sched_getaffinity(0))
        except AttributeError: cpu_count = os.cpu_count() or 1
        final_args_ns.num_workers = max(0, cpu_count // world_size_env if world_size_env > 0 else cpu_count)
        if final_args_ns.num_workers == 0 and world_size_env > 1 and cpu_count > world_size_env: final_args_ns.num_workers = 1
    else: # If it was an int from YAML or CLI
        try: final_args_ns.num_workers = int(final_args_ns.num_workers)
        except ValueError: _module_logger.warning(f"Invalid num_workers='{final_args_ns.num_workers}', using 0."); final_args_ns.num_workers = 0
        if final_args_ns.num_workers < 0: _module_logger.warning(f"Negative num_workers='{final_args_ns.num_workers}', using 0."); final_args_ns.num_workers = 0


    # Basic validation
    if final_args_ns.stage is None: parser.error("--stage is required (in YAML or CLI).")
    is_spade_stage = 'spade' in final_args_ns.stage
    # Check if any mask source is provided if it's a SPADE stage
    has_fixed_train_bank = final_args_ns.train_mask_memmap_file
    has_variable_train_bank = final_args_ns.train_mask_variable_payload_file and final_args_ns.train_mask_variable_header_file
    has_individual_masks_dir_cfg = final_args_ns.segmentation_mask_dir and final_args_ns.train_mask_subdir_name # Check if configured

    if is_spade_stage and not (has_fixed_train_bank or has_variable_train_bank or has_individual_masks_dir_cfg):
        parser.error("For SPADE stages, a mask source is required: train_mask_memmap_file (fixed), "
                     "train_mask_variable_payload/header_file (variable), or segmentation_mask_dir + train_mask_subdir_name (individual fallback).")
    if final_args_ns.num_total_segments <= 0: # num_total_segments has a default now
        parser.error("--num_total_segments must be a positive integer.")

    try:
        main(final_args_ns)
    except KeyboardInterrupt: _logger.warning("Training interrupted by user (Ctrl+C)."); cleanup_distributed(); sys.exit(130)
    except Exception: _logger.critical("Unhandled exception during main execution:", exc_info=True); cleanup_distributed(); sys.exit(1)