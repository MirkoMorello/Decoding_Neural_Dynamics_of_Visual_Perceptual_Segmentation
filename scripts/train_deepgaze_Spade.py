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
        OriginalImageDataset, # If you kept the original name as this
        FixationMaskTransform,
        # The prepare_... functions might need adaptation to pass segmentation_mask_dir
        # For now, we'll instantiate datasets directly in main for this script
        convert_stimuli, convert_fixation_trains # If used for MIT preprocessing
    )
    # Original DeepGaze modules and layers that we will reuse
    from src.modules import DeepGazeIII, FeatureExtractor, DeepGazeII, FeatureExtractor, Finalizer
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

# --- Original Scanpath and Fixation Selection Networks (No SPADE here for now) ---
def build_scanpath_network_original():
    """ Builds the network processing scanpath history (original version). """
    return nn.Sequential(OrderedDict([
        ('encoding0', FlexibleScanpathHistoryEncoding(in_fixations=4, channels_per_fixation=3, out_channels=128, kernel_size=[1, 1], bias=True)),
        ('softplus0', nn.Softplus()),
        ('layernorm1', LayerNorm(128)),
        ('conv1', nn.Conv2d(128, 16, (1, 1), bias=False)),
        ('bias1', Bias(16)),
        ('softplus1', nn.Softplus()),
    ]))
    
def build_fixation_selection_network_original(scanpath_features=16):
    """ Builds the network combining saliency and scanpath features (original version). """
    saliency_channels = 1 # Output of saliency network's core path before combination
    in_channels_list = [saliency_channels, scanpath_features if scanpath_features > 0 else 0]
    # Filter out zero-channel inputs if any (original DeepGaze behavior)
    in_channels_list_filtered = [ch for ch in in_channels_list if ch > 0]

    return nn.Sequential(OrderedDict([
        # Note: LayerNormMultiInput and Conv2dMultiInput from deepgaze_pytorch handle list of features
        ('layernorm0', LayerNormMultiInput(in_channels_list_filtered)),
        ('conv0', Conv2dMultiInput(in_channels_list_filtered, 128, (1, 1), bias=False)),
        ('bias0', Bias(128)),
        ('softplus0', nn.Softplus()),
        ('layernorm1', LayerNorm(128)),
        ('conv1', nn.Conv2d(128, 16, (1, 1), bias=False)),
        ('bias1', Bias(16)),
        ('softplus1', nn.Softplus()),
        ('conv2', nn.Conv2d(16, 1, (1, 1), bias=False)), # Final output layer
    ]))



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
    densenet_base_model = RGBDenseNet201(pretrained=True if not args.densenet_weights_path else args.densenet_weights_path)
    densenet_target_layers_for_extraction = args.densenet_target_layers # From argparse
    C_in_densenet = args.densenet_C_in # Get from args, default to 2048
    readout_factor_densenet = args.densenet_readout_factor # Get from args, default to 4 (as per your example)
    saliency_map_factor_densenet = 4 # Standard, from your example
    features_module = FeatureExtractor(densenet_base_model, densenet_target_layers_for_extraction)
    
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
                train_baseline_log_likelihood = SALICON_centerbias.information_gain(SALICON_train_stimuli, SALICON_train_fixations, verbose=False, average='image')
                with open(train_ll_cache_file, 'wb') as f: cpickle.dump(train_baseline_log_likelihood, f)
                _logger.info(f"Saved computed TRAIN baseline LL to: {train_ll_cache_file}")
            # Try loading cached val LL
            try:
                with open(val_ll_cache_file, 'rb') as f: val_baseline_log_likelihood = cpickle.load(f)
                _logger.info(f"Loaded cached VALIDATION baseline LL from: {val_ll_cache_file}")
            except Exception:
                _logger.warning(f"VALIDATION LL cache miss or error. Computing...");
                val_baseline_log_likelihood = SALICON_centerbias.information_gain(SALICON_val_stimuli, SALICON_val_fixations, verbose=False, average='image')
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
            input_channels=C_in_densenet,
            num_total_segments=args.num_total_segments,
            seg_embedding_dim=args.seg_embedding_dim,
            add_sa_head=args.add_sa_head
        )
        # For this experiment, scanpath network is None, and fixation selection uses original layers
        scanpath_net = None # Or build_scanpath_network_original() if also testing scanpaths without SPADE
        fixsel_net = build_fixation_selection_network_original(scanpath_features=0) # scanpath_features=0 as scanpath_net is None
        
        model = DeepGazeIII(
            features=features_module, # Our DenseNetFeatureExtractor instance
            saliency_network=saliency_net_spade,
            scanpath_network=scanpath_net,
            fixation_selection_network=fixsel_net,
            downsample=1, # DenseNet features are already strided by backbone
            readout_factor=readout_factor_densenet,
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
            lmdb_path=str(salicon_train_lmdb_path) if args.use_lmdb_images else None, # Optional LMDB for images
            # cached = False # Usually false if using LMDB or large dataset
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
        )
        val_sampler = (torch.utils.data.DistributedSampler(val_dataset, shuffle=False, drop_last=False)
                       if is_distributed else None)
        validation_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, # Val batch size can often be larger
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
# == CLI ARGUMENT PARSER ==
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeepGazeIII with DenseNet Backbone and SPADE conditioning (Multi-GPU)")
    
    # --- Experiment Configuration ---
    parser.add_argument('--stage', default='salicon_pretrain_densenet_spade',
                        choices=['salicon_pretrain_densenet_spade',
                                 'mit_spatial_densenet_spade',
                                 'mit_scanpath_frozen_densenet_spade',
                                 'mit_scanpath_full_densenet_spade'],
                        help='Training stage to execute.')
    parser.add_argument('--densenet_model_name', default='densenet161', choices=['densenet161', 'densenet201'],
                        help='DenseNet model variant from torchvision.')
    
    # --- SPADE Specific Arguments ---
    parser.add_argument('--segmentation_mask_dir', required=True, type=str,
                        help='Root directory containing precomputed segmentation masks (e.g., ./masks/).')
    parser.add_argument('--train_mask_subdir_name', default='dinov2_kmeans_k16_salicon_train', type=str,
                        help='Subdirectory name under segmentation_mask_dir for training set masks.')
    parser.add_argument('--val_mask_subdir_name', default='dinov2_kmeans_k16_salicon_val', type=str,
                        help='Subdirectory name under segmentation_mask_dir for validation set masks.')
    parser.add_argument('--segmentation_mask_format', default='png', choices=['png', 'npy'],
                        help='File format of the saved segmentation masks.')
    parser.add_argument('--num_total_segments', type=int, default=17, # K_clusters + 1 if 0 is unused, or K if 0 is a segment
                        help='Total number of unique segment IDs for nn.Embedding. If K-Means gives 0 to K-1, this should be K.')
    parser.add_argument('--seg_embedding_dim', type=int, default=64,
                        help='Dimension for the learned segment ID embeddings.')

    # --- Training Hyperparameters ---
    parser.add_argument('--batch_size', type=int, default=16, # DenseNet might need smaller BS than DINOv2
                        help='Batch size *per GPU*. Effective batch size is batch_size * world_size * grad_accum.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of forward passes to accumulate gradients before optimizer step.')
    parser.add_argument('--lr', type=float, default=0.0005, help='Initial learning rate for the model head.')
    parser.add_argument('--lr_milestones', type=int, nargs='+', default=[15, 30, 45],
                        help='Epochs at which to decay learning rate for MultiStepLR.')
    parser.add_argument('--min_lr', type=float, default=1e-7, help='Minimum learning rate threshold for scheduler.')
    parser.add_argument('--fold', type=int, help='Cross-validation fold for MIT1003 stages (0-9).')
    parser.add_argument('--validation_epochs', type=int, default=1, help='Run validation every N epochs.')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='Path to a checkpoint to resume training from.')

    # --- Dataloading & System ---
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Dataloader workers per rank. Default: auto (cores // world_size). Set 0 for main process loading.')
    parser.add_argument('--train_dir', default='./experiments_densenet_spade',
                        help='Base directory for all training outputs of this script type.')
    parser.add_argument('--dataset_dir', default='../data/pysaliency_datasets', # Adjusted relative path
                        help='Directory for pysaliency datasets (SALICON, MIT original).')
    parser.add_argument('--lmdb_dir', default='../data/lmdb_cache_densenet', # Adjusted relative path
                        help='Directory for LMDB image data caches (if used).')
    parser.add_argument('--use_lmdb_images', action='store_true', help='Use LMDB for loading stimulus images.')
    
    # --- Model Generic Arguments ---
    parser.add_argument('--add_sa_head', action='store_true',
                        help='Add a SelfAttention layer at the beginning of the saliency network.')

    args = parser.parse_args()

    # --- Automatic Worker Count ---
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if args.num_workers is None:
        try:
            cpu_count = len(os.sched_getaffinity(0)) # More accurate on Linux
        except AttributeError:
            cpu_count = os.cpu_count() or 1 # Fallback
        args.num_workers = max(0, cpu_count // world_size if world_size > 0 else cpu_count)
        if args.num_workers == 0 and world_size > 1: # Ensure at least 1 worker if multiple GPUs and cpus allow
            if cpu_count > world_size : args.num_workers = 1
    elif args.num_workers < 0:
        args.num_workers = 0
    
    # --- Correct num_total_segments if K was specified ---
    # If user specifies --num_segments 16 (meaning K=16 clusters, IDs 0-15),
    # then nn.Embedding needs num_embeddings = 16.
    # If args.num_total_segments is meant to be K, then use that.
    # The current SaliencyNetworkSPADE uses num_total_segments directly for nn.Embedding.
    # So, if K-Means produced K=16 segments (IDs 0 to 15), num_total_segments should be 16.
    # My previous default of 17 was for K + 1 if 0 was a special background.
    # Let's assume num_total_segments is the actual count of unique IDs.
    # If K-Means gives k labels (0 to k-1), then num_total_segments should be k.
    # The generate_masks script used --num_segments for K.
    # So, if you ran generate_masks with --num_segments 16, then there are 16 unique labels (0-15).
    # Thus, args.num_total_segments should be 16 for nn.Embedding.
    # Let's clarify the argument:
    # parser.add_argument('--num_k_segments', type=int, default=16,
    #                     help='Number of K segments produced by K-Means (IDs 0 to K-1). nn.Embedding will use this value.')
    # And then in main(): args.num_total_segments = args.num_k_segments

    if args.num_total_segments <= 0:
        _logger.error("--num_total_segments must be positive.")
        sys.exit(1)


    try:
        main(args)
    except KeyboardInterrupt:
        _logger.warning("Training interrupted by user (KeyboardInterrupt). Cleaning up...")
        cleanup_distributed()
        sys.exit(130)
    except Exception as e:
        _logger.critical("Unhandled exception during main execution:", exc_info=True)
        cleanup_distributed()
        sys.exit(1)