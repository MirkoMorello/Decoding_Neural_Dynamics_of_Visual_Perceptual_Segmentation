#!/usr/bin/env python
"""
Multi-GPU-ready training script for DeepGaze III with a DINOv2 backbone.
Refactored for clarity, consistency, and simplified configuration.
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
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

import pysaliency
import pysaliency.external_datasets.mit
from pysaliency.dataset_config import train_split, validation_split
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from pysaliency.baseline_utils import BaselineModel, CrossvalidatedBaselineModel
import cloudpickle as cpickle

try:
    from src.data import (
        prepare_scanpath_dataset, prepare_spatial_dataset,
        convert_stimuli, convert_fixation_trains
    )
    from src.dinov2_backbone import DinoV2Backbone
    from src.dinogaze import (build_saliency_network, build_scanpath_network, build_fixation_selection_network)
    from src.modules import DeepGazeIII
    from src.metrics import log_likelihood, nss, auc as auc_cpu_fn
    from src.training import _train, restore_from_checkpoint
except ImportError as e:
    print(f"PYTHON IMPORT ERROR: {e}\n(sys.path: {sys.path})")
    sys.exit(1)

_logger = logging.getLogger("train_dinogaze")

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

def setup_model(args, device):
    """Initializes the DINOv2 backbone and the full DeepGazeIII model."""
    features = DinoV2Backbone(
        layers=args.layers,
        model_name=args.dino_model_name,
        freeze=True  # Always start frozen
    )

    if args.unfreeze_vit_layers:
        if args.is_master: _logger.info(f"Unfreezing DINOv2 layers: {args.unfreeze_vit_layers}")
        for name, param in features.backbone.named_parameters():
            if name.startswith('blocks.'):
                try:
                    block_index = int(name.split('.')[1])
                    if block_index in args.unfreeze_vit_layers:
                        param.requires_grad = True
                except (IndexError, ValueError):
                    pass

    readout_factor = 14 if 'vitg14' in args.dino_model_name else 7
    C_in = len(features.layers) * features.num_channels
    if args.is_master: _logger.info(f"Feature extractor initialized. Input channels to saliency network: {C_in}")

    saliency_net = build_saliency_network(C_in, add_sa_head=False) # Simplified for this version
    scanpath_net = build_scanpath_network() if args.stage == 'mit_scanpath' else None
    fixsel_net = build_fixation_selection_network(scanpath_features=16 if args.stage == 'mit_scanpath' else 0)

    model = DeepGazeIII(
        features=features,
        saliency_network=saliency_net,
        scanpath_network=scanpath_net,
        fixation_selection_network=fixsel_net,
        downsample=1,
        readout_factor=readout_factor,
        saliency_map_factor=4,
        included_fixations=[-1, -2, -3, -4] if args.stage == 'mit_scanpath' else []
    )
    return model

def salicon_pretrain(args, device, is_master, is_distributed):
    """Handles the SALICON pretraining stage."""
    if is_master: _logger.info("--- Preparing SALICON Pretraining Stage ---")
    
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
        train_ll = centerbias.information_gain(train_stim, train_fix, verbose=False, average='image')
        val_ll = centerbias.information_gain(val_stim, val_fix, verbose=False, average='image')
        _logger.info(f"Master Baseline LLs - Train: {train_ll:.5f}, Val: {val_ll:.5f}")
    
    ll_bcast = [train_ll, val_ll]
    if is_distributed: dist.broadcast_object_list(ll_bcast, src=0)
    train_ll, val_ll = ll_bcast
    if train_ll is None or val_ll is None: _logger.critical("NaN LLs received."); sys.exit(1)

    model = setup_model(args, device).to(device)
    if is_distributed:
        model = DDP(model, device_ids=[device.index], find_unused_parameters=bool(args.unfreeze_vit_layers))

    param_groups = [{'params': [p for n, p in model.named_parameters() if not n.startswith('module.features.backbone') and p.requires_grad]}]
    if args.unfreeze_vit_layers:
        param_groups.append({
            'params': [p for n, p in model.named_parameters() if n.startswith('module.features.backbone') and p.requires_grad],
            'lr': args.backbone_lr
        })
    optimizer = optim.Adam(param_groups, lr=args.lr)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones)

    train_loader = prepare_spatial_dataset(train_stim, train_fix, centerbias, args.batch_size, args.num_workers, is_distributed, is_master, device, args.lmdb_dir / f'SALICON_train_{args.dino_model_name}', _logger)
    validation_loader = prepare_spatial_dataset(val_stim, val_fix, centerbias, args.batch_size, args.num_workers, is_distributed, is_master, device, args.lmdb_dir / f'SALICON_val_{args.dino_model_name}', _logger)
    
    output_dir = args.train_dir / f"{args.stage}_{args.dino_model_name}"
    _train(str(output_dir), model, train_loader, train_ll, validation_loader, val_ll, optimizer, lr_scheduler, args.gradient_accumulation_steps, args.min_lr, ['LL', 'IG', 'NSS', 'AUC_CPU'], args.validation_epochs, args.resume_checkpoint, device, is_distributed, is_master, _logger)

def mit_finetune(args, device, is_master, is_distributed):
    """Handles all MIT fine-tuning stages (spatial and scanpath)."""
    fold = args.fold
    if fold is None or not (0 <= fold < 10): _logger.critical("--fold (0-9) required for MIT stages."); sys.exit(1)
    if is_master: _logger.info(f"--- Preparing MIT Stage: {args.stage} (Fold {fold}) ---")
    
    mit_converted_data_path = args.train_dir / f"MIT1003_converted_dinogaze_{args.dino_model_name}"
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
        train_ll = centerbias.information_gain(train_stim, train_fix, verbose=False, average='image')
        val_ll = centerbias.information_gain(val_stim, val_fix, verbose=False, average='image')
    ll_bcast = [train_ll, val_ll]; dist.broadcast_object_list(ll_bcast, src=0)
    train_ll, val_ll = ll_bcast
    if train_ll is None or val_ll is None: _logger.critical("MIT Baseline LLs invalid."); sys.exit(1)

    model_cpu = setup_model(args, 'cpu')
    if args.salicon_checkpoint_path and args.salicon_checkpoint_path.exists():
        if is_master: _logger.info(f"Loading SALICON weights from {args.salicon_checkpoint_path}")
        restore_from_checkpoint(model_cpu, None, None, None, str(args.salicon_checkpoint_path), 'cpu', False, _logger)
    else:
        _logger.warning("No SALICON checkpoint provided. Starting fine-tuning from scratch.")

    model = model_cpu.to(device)
    if is_distributed:
        model = DDP(model, device_ids=[device.index], find_unused_parameters=True) # Use True for safety in multi-stage
        
    param_groups = [{'params': [p for n, p in model.named_parameters() if not n.startswith('module.features.backbone') and p.requires_grad]}]
    if args.unfreeze_vit_layers:
        param_groups.append({
            'params': [p for n, p in model.named_parameters() if n.startswith('module.features.backbone') and p.requires_grad],
            'lr': args.backbone_lr
        })
    optimizer = optim.Adam(param_groups, lr=args.lr_mit)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones_mit)

    if args.stage == 'mit_spatial':
        train_loader = prepare_spatial_dataset(train_stim, train_fix, centerbias, args.batch_size, args.num_workers, is_distributed, is_master, device, args.lmdb_dir / f'MIT_train_spatial_{fold}', _logger)
        val_loader = prepare_spatial_dataset(val_stim, val_fix, centerbias, args.batch_size, args.num_workers, is_distributed, is_master, device, args.lmdb_dir / f'MIT_val_spatial_{fold}', _logger)
    else: # mit_scanpath
        train_loader = prepare_scanpath_dataset(train_stim, train_fix, centerbias, args.batch_size, args.num_workers, is_distributed, is_master, device, args.lmdb_dir / f'MIT_train_scanpath_{fold}', _logger)
        val_loader = prepare_scanpath_dataset(val_stim, val_fix, centerbias, args.batch_size, args.num_workers, is_distributed, is_master, device, args.lmdb_dir / f'MIT_val_scanpath_{fold}', _logger)

    output_dir = args.train_dir / f"{args.stage}_{args.dino_model_name}_fold{fold}"
    _train(str(output_dir), model, train_loader, train_ll, val_loader, val_ll, optimizer, lr_scheduler, args.gradient_accumulation_steps, args.min_lr, ['LL', 'IG', 'NSS', 'AUC_CPU'], args.validation_epochs, None, device, is_distributed, is_master, _logger)


def main(args):
    """Main function to set up DDP and dispatch to training stage handlers."""
    device, rank, world, is_master, is_distributed = init_distributed()
    args.is_master = is_master # Add to args for easy access in helper functions

    log_level = logging.INFO if is_master else logging.WARNING
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

    if args.stage == 'salicon_pretrain':
        salicon_pretrain(args, device, is_master, is_distributed)
    elif args.stage in ['mit_spatial', 'mit_scanpath']:
        mit_finetune(args, device, is_master, is_distributed)
    else:
        _logger.critical(f"Unknown stage: {args.stage}"); sys.exit(1)

    cleanup_distributed()

if __name__ == "__main__":
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument('--config_file', type=str, default=None)
    _cfg_ns, _rem_args = _pre.parse_known_args()

    parser = argparse.ArgumentParser(parents=[_pre], description="Train DeepGaze-III with DINOv2 backbone.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # --- Core arguments ---
    parser.add_argument('--stage', required=True, choices=['salicon_pretrain', 'mit_spatial', 'mit_scanpath'], help='Training stage to execute.')
    parser.add_argument('--dino_model_name', default='dinov2_vitl14', choices=['dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'], help='DINOv2 model variant.')
    parser.add_argument('--layers', type=int, nargs='+', default=[-3, -2, -1], help='Indices of transformer blocks to extract features from.')

    # --- Training Hyperparameters ---
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size per GPU.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate for the SALICON model head.')
    parser.add_argument('--lr_milestones', type=int, nargs='+', default=[15, 30, 45])
    parser.add_argument('--lr_mit', type=float, default=1e-4, help="LR for MIT fine-tuning stages.")
    parser.add_argument('--lr_milestones_mit', type=int, nargs='+', default=[5, 10, 15])
    parser.add_argument('--backbone_lr', type=float, default=1e-5, help='Learning rate for unfrozen backbone layers.')
    parser.add_argument('--min_lr', type=float, default=1e-7, help='Lower LR bound for schedulers.')
    parser.add_argument('--validation_epochs', type=int, default=1)
    
    # --- Fine-tuning & Checkpointing ---
    parser.add_argument('--fold', type=int, help='Cross-validation fold for MIT stages (0-9).')
    parser.add_argument('--resume_checkpoint', type=str, help='Path to checkpoint to resume SALICON stage from.')
    parser.add_argument('--salicon_checkpoint_path', type=str, help='Path to SALICON pretrained checkpoint for MIT stages.')
    parser.add_argument('--unfreeze_vit_layers', type=int, nargs='+', default=[], help='Indices of ViT blocks to unfreeze & fine-tune.')

    # --- Dataloading & System ---
    parser.add_argument('--num_workers', type=str, default='auto')
    parser.add_argument('--train_dir', default='./experiments_dinogaze')
    parser.add_argument('--dataset_dir', default='./data/pysaliency_datasets')
    parser.add_argument('--lmdb_dir', default='./data/lmdb_caches_dinogaze')

    if _cfg_ns.config_file:
        try:
            with open(_cfg_ns.config_file, 'r') as f: yaml_cfg = yaml.safe_load(f) or {}
            parser.set_defaults(**yaml_cfg)
        except Exception as e: print(f"Could not read/parse YAML: {e}")

    final_args_ns = parser.parse_args(_rem_args)

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