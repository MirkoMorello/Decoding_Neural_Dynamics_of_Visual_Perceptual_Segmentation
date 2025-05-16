#!/usr/bin/env python
"""
Multi‑GPU‑ready training script for DeepGaze III with DINOv2 backbone.
Launch with torchrun, e.g.:
    torchrun --standalone --nproc_per_node=4 train_dinogaze_ddp.py \
        --stage salicon_pretrain --batch_size 4 --num_workers 8 --model_name dinov2_vitl14 --lr 0.0005

The script falls back to single‑GPU/CPU when no distributed environment
variables are present, so you can still run it exactly like the original.
"""
import os
import sys

current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import logging
import pickle
import sys
from pathlib import Path
import pysaliency
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import pysaliency.external_datasets.mit
from PIL import Image
Image.MAX_IMAGE_PIXELS = None # Disable PIL decompression bomb check for large images
from pysaliency.baseline_utils import (BaselineModel,
                                    CrossvalidatedBaselineModel)
import cloudpickle as cpickle
import pysaliency
from tqdm import tqdm
from boltons.fileutils import atomic_save # Needed for copied _train


# -----------------------------------------------------------------------------
# UTILS – DISTRIBUTED SETUP ----------------------------------------------------
# -----------------------------------------------------------------------------

def init_distributed() -> tuple[torch.device, int, int, bool, bool]:
    """Initialise torch.distributed (NCCL) if environment variables are present.

    Returns
    -------
    device : torch.device     CUDA device for this rank (or cpu)
    rank   : int              Global rank (0 for single‑GPU case)
    world  : int              World size (1 for single‑GPU case)
    is_master : bool          True on rank 0 – use for logging / checkpointing
    is_distributed : bool     True if world_size > 1
    """
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_distributed = world_size > 1

    if is_distributed:
        # ---- torchrun sets these automatically --------------------------------
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        # Correct device after setting cuda device
        device = torch.device(f"cuda:{local_rank}")
        is_master = rank == 0
        # Logging configured later after rank is known
    else:
        # -------- single‑GPU / CPU fallback ------------------------------------
        rank = 0
        world_size = 1
        is_master = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return device, rank, world_size, is_master, is_distributed


def cleanup_distributed():
    """ Clean up the distributed environment. """
    if dist.is_initialized():
        dist.destroy_process_group()


try:
    from src.data import ( 
        FixationDataset, FixationMaskTransform,
        ImageDataset, ImageDatasetSampler,
        prepare_scanpath_dataset, prepare_spatial_dataset,
        convert_stimuli, convert_fixation_trains)
    from src.dinov2_backbone import DinoV2Backbone
    from src.dinogaze import (build_saliency_network, build_scanpath_network, build_fixation_selection_network)
    from src.modules import DeepGazeIII
    from src.metrics import log_likelihood, nss, auc as auc_cpu_fn # CHANGED
    from src.training import (_train)
except ImportError as e:
    actual_error_message = str(e) # Get the real Python error
    logger_instance_name = "_logger"
    if logger_instance_name in locals() and locals()[logger_instance_name] is not None:
        locals()[logger_instance_name].critical(f"PYTHON IMPORT ERROR: {actual_error_message}") # Print the real error
        locals()[logger_instance_name].critical(f"Current sys.path: {sys.path}")
        locals()[logger_instance_name].critical("Ensure 'src' is in sys.path and contains __init__.py and all required .py files with correct internal imports.")
    else:
        print(f"PYTHON IMPORT ERROR: {actual_error_message}") # Print the real error
        print(f"Current sys.path: {sys.path}")
        print("Ensure 'src' is in sys.path and contains __init__.py and all required .py files with correct internal imports.")
    sys.exit(1)

# -----------------------------------------------------------------------------
# LOGGING SETUP – Configured properly in main() after DDP init
# -----------------------------------------------------------------------------
_logger = logging.getLogger("train_dinogaze_ddp") # Get logger instance


# -----------------------------------------------------------------------------
# MAIN EXECUTION BLOCK
# -----------------------------------------------------------------------------

def main(args):
    """ Main function to parse arguments, set up DDP, and run training stages. """
    device, rank, world, is_master, is_distributed = init_distributed()

    log_level = logging.INFO if is_master else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format=f"%(asctime)s Rank{rank} %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True # Override any root logger setup
    )
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING) # Quieten PIL logs
    _logger.setLevel(log_level) # Ensure our logger uses the correct level

    if is_master:
        _logger.info("==================================================")
        _logger.info(f"Starting Training Run: Stage '{args.stage}'")
        _logger.info("==================================================")
        _logger.info(f"Torch Version: {torch.__version__}")
        _logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available(): _logger.info(f"CUDA Version: {torch.version.cuda}")
        _logger.info(f"DDP Initialised: Rank {rank}/{world} | Master: {is_master} | Distributed: {is_distributed} | Device: {device}")
        _logger.info(f"Model: {args.model_name} | Layers: {args.layers}")
        _logger.info(f"Batch Size Per GPU: {args.batch_size}")
        _logger.info(f"Gradient Accumulation Steps: {args.gradient_accumulation_steps}")
        effective_batch_size = args.batch_size * world * args.gradient_accumulation_steps
        _logger.info(f"Effective Global Batch Size: {effective_batch_size}")
        _logger.info(f"LR: {args.lr} | Min LR: {args.min_lr}")
        _logger.info(f"Workers Per Rank: {args.num_workers}")
        _logger.info(f"Train Dir: {args.train_dir}")
        _logger.info(f"Dataset Dir: {args.dataset_dir}")
        _logger.info(f"LMDB Dir: {args.lmdb_dir}")
        _logger.info(f"Add SA Head: {args.add_sa_head}")
        _logger.info(f"Unfreeze ViT Layers: {args.unfreeze_vit_layers if args.unfreeze_vit_layers else 'None (All Frozen)'}")
        if args.unfreeze_vit_layers:
            _logger.info(f"Backbone LR (if unfrozen): {args.backbone_lr}")
        if args.stage in ['mit_spatial', 'mit_scanpath_frozen', 'mit_scanpath_full']:
            if args.fold is None:
                _logger.critical("--fold parameter is required for MIT stages.")
                if is_distributed: dist.barrier()
                sys.exit(1)
            _logger.info(f"MIT Fold: {args.fold}")
        if args.num_workers == 0 and is_master:
            _logger.warning("Number of dataloader workers is 0. Data loading might be a bottleneck.")
        if args.gradient_accumulation_steps > 1 and is_master:
            _logger.info("Gradient accumulation enabled. Optimizer step occurs every "
                        f"{args.gradient_accumulation_steps} micro-batches.")

    dataset_directory = Path(args.dataset_dir).resolve()
    train_directory = Path(args.train_dir).resolve()
    lmdb_directory = Path(args.lmdb_dir).resolve()
    if is_master:
        for p in (dataset_directory, train_directory, lmdb_directory):
            try: p.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                _logger.critical(f"Failed to create directory {p}: {e}. Check permissions.")
                if is_distributed: dist.barrier()
                sys.exit(1)
    if is_distributed:
        dist.barrier() # Wait for master

    try:
        features = DinoV2Backbone(
            layers=args.layers,
            model_name=args.model_name,
            freeze=True # Start frozen
        )
        features = features.to(device)

    except Exception as e:
        _logger.critical(f"Failed to initialize DinoV2Backbone: {e}")
        if is_distributed: dist.barrier()
        sys.exit(1)

    if args.model_name in ['dinov2_vitg14']:
        readout_factor = 14
        if is_master: _logger.info(f"Using readout_factor={readout_factor} for {args.model_name}")
    else:
        readout_factor = 7
        if is_master: _logger.info(f"Using default readout_factor={readout_factor} for {args.model_name}")

    unfrozen_params_exist = False
    if args.unfreeze_vit_layers:
        if is_master: _logger.info(f"Attempting to unfreeze DINOv2 layers: {args.unfreeze_vit_layers}")
        unfrozen_count = 0
        total_backbone_params = sum(p.numel() for p in features.backbone.parameters())
        unfrozen_backbone_params_count = 0

        for name, param in features.backbone.named_parameters():
            should_unfreeze = False
            if name.startswith('blocks.'):
                try:
                    block_index = int(name.split('.')[1])
                    if block_index in args.unfreeze_vit_layers:
                        should_unfreeze = True
                except (IndexError, ValueError):
                    pass

            if name == 'norm' and hasattr(features.backbone, 'norm') and features.backbone.norm is not None:
                if args.unfreeze_vit_layers and max(args.unfreeze_vit_layers) == (len(features.backbone.blocks) - 1):
                    should_unfreeze = True
                    if is_master: _logger.info(f"  - Also unfreezing final norm layer.")

            if should_unfreeze:
                param.requires_grad = True
                unfrozen_count += 1
                unfrozen_backbone_params_count += param.numel()
                if is_master: _logger.debug(f"  - Unfroze parameter: {name} (Size: {param.numel()})")
            else:
                param.requires_grad = False

        if is_master:
            _logger.info(f"Successfully unfroze {unfrozen_count} parameter tensors in DINOv2 backbone.")
            _logger.info(f"Total backbone params: {total_backbone_params:,} | Unfrozen backbone params: {unfrozen_backbone_params_count:,}")
            if unfrozen_count > 0: unfrozen_params_exist = True
            elif args.unfreeze_vit_layers: _logger.warning("Requested unfreezing, but no matching parameters found.")
    else:
        if is_master: _logger.info("Keeping DINOv2 backbone fully frozen.")
    
    _logger.info(f"Effective saliency stride = "
             f"{readout_factor * 1} px")

    C_in = len(features.layers) * features.num_channels
    if is_master: _logger.info(f"Feature extractor initialized. Input channels to saliency network: {C_in}")

    # ======================= STAGE DISPATCH ============================
    if args.stage == 'salicon_pretrain':
        if is_master: _logger.info("--- Preparing SALICON Pretraining Stage ---")

        if is_master: _logger.info(f"Loading SALICON data from {dataset_directory}...")
        salicon_train_loc = dataset_directory
        salicon_val_loc = dataset_directory
        if is_master:
            try:
                pysaliency.get_SALICON_train(location=salicon_train_loc)
                pysaliency.get_SALICON_val(location=salicon_val_loc)
            except Exception as e:
                _logger.critical(f"Failed to download/access SALICON data: {e}")
                if is_distributed: dist.barrier(); sys.exit(1)
        if is_distributed:
            dist.barrier()
        try:
            SALICON_train_stimuli, SALICON_train_fixations = pysaliency.get_SALICON_train(location=salicon_train_loc)
            SALICON_val_stimuli, SALICON_val_fixations = pysaliency.get_SALICON_val(location=salicon_val_loc)
        except Exception as e:
            _logger.critical(f"Failed to load SALICON metadata: {e}")
            if is_distributed: dist.barrier(); sys.exit(1)
        if is_master: _logger.info("SALICON data loaded.")

        if is_master: _logger.info("Initializing SALICON BaselineModel...")
        SALICON_centerbias = BaselineModel(stimuli=SALICON_train_stimuli, fixations=SALICON_train_fixations, bandwidth=0.0217, eps=2e-13, caching=False)
        train_ll_cache_file = dataset_directory / 'salicon_baseline_train_ll.pkl'
        val_ll_cache_file = dataset_directory / 'salicon_baseline_val_ll.pkl'
        train_baseline_log_likelihood, val_baseline_log_likelihood = None, None

        if is_master:
            try:
                with open(train_ll_cache_file, 'rb') as f: train_baseline_log_likelihood = cpickle.load(f)
                _logger.info(f"Loaded cached train baseline LL from: {train_ll_cache_file}")
            except Exception as e:
                _logger.warning(f"Train LL cache fail ({e}). Computing...");
                try:
                    train_baseline_log_likelihood = SALICON_centerbias.information_gain(SALICON_train_stimuli, SALICON_train_fixations, verbose=False, average='image')
                    with open(train_ll_cache_file, 'wb') as f: cpikle.dump(train_baseline_log_likelihood, f)
                    _logger.info(f"Saved computed train baseline LL to: {train_ll_cache_file}")
                except Exception as compute_e:
                    _logger.error(f"Error computing/saving train LL cache: {compute_e}")
                    train_baseline_log_likelihood = -999.9
            try:
                with open(val_ll_cache_file, 'rb') as f: val_baseline_log_likelihood = cpickle.load(f)
                _logger.info(f"Loaded cached validation baseline LL from: {val_ll_cache_file}")
            except Exception as e:
                _logger.warning(f"Val LL cache fail ({e}). Computing...");
                try:
                    val_baseline_log_likelihood = SALICON_centerbias.information_gain(SALICON_val_stimuli, SALICON_val_fixations, verbose=False, average='image')
                    with open(val_ll_cache_file, 'wb') as f: cpikle.dump(val_baseline_log_likelihood, f)
                    _logger.info(f"Saved computed validation baseline LL to: {val_ll_cache_file}")
                except Exception as compute_e:
                    _logger.error(f"Error computing/saving val LL cache: {compute_e}")
                    val_baseline_log_likelihood = -999.9

            if train_baseline_log_likelihood == -999.9 or val_baseline_log_likelihood == -999.9:
                _logger.critical("Failed to obtain baseline LLs.")
                if train_baseline_log_likelihood is None: train_baseline_log_likelihood = -999.9
                if val_baseline_log_likelihood is None: val_baseline_log_likelihood = -999.9
            else:
                _logger.info(f"Master Baseline LLs - Train: {train_baseline_log_likelihood:.5f}, Val: {val_baseline_log_likelihood:.5f}")

        ll_list_to_broadcast = [train_baseline_log_likelihood, val_baseline_log_likelihood]
        if is_distributed: dist.broadcast_object_list(ll_list_to_broadcast, src=0)
        train_baseline_log_likelihood, val_baseline_log_likelihood = ll_list_to_broadcast
        if train_baseline_log_likelihood == -999.9 or val_baseline_log_likelihood == -999.9:
            _logger.critical(f"Baseline LLs invalid on rank {rank}. Exiting.")
            if is_distributed: dist.barrier(); sys.exit(1)
        
        _logger.info(f"Building saliency network with {args.C_in} input channels. Add SA Head: {args.add_sa_head}")
        _logger.info(f"Building fixation selection network with {args.scanpath_features} scanpath features.")
        
        model = DeepGazeIII(
            features=features, # features already on device
            saliency_network=build_saliency_network(C_in, add_sa_head=args.add_sa_head),
            scanpath_network=None,
            fixation_selection_network=build_fixation_selection_network(scanpath_features=0),
            downsample=1,
            readout_factor=readout_factor,
            saliency_map_factor=4,
            included_fixations=[]
        ).to(device)

        if is_distributed:
            ddp_find_unused = unfrozen_params_exist or args.add_sa_head
            model = DDP(model, device_ids=[device.index], output_device=device.index, broadcast_buffers=False, find_unused_parameters=ddp_find_unused)
            if is_master: _logger.info(f"Wrapped model with DDP (find_unused_parameters={ddp_find_unused}).")

        head_params = [p for n, p in model.named_parameters() if not n.startswith('features.backbone') and p.requires_grad]
        param_groups = [{'params': head_params, 'lr': args.lr}]
        if unfrozen_params_exist:
            backbone_params = [p for n, p in model.named_parameters() if n.startswith('features.backbone') and p.requires_grad]
            if backbone_params:
                param_groups.append({'params': backbone_params, 'lr': args.backbone_lr})
                if is_master: _logger.info(f"Created separate parameter group for unfrozen backbone layers with LR={args.backbone_lr}")
            elif is_master and args.unfreeze_vit_layers:
                _logger.warning("Backbone unfreezing requested, but no unfrozen backbone parameters found for optimizer group.")
        optimizer = optim.Adam(param_groups, lr=args.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 45, 60])

        train_loader = prepare_spatial_dataset(SALICON_train_stimuli, SALICON_train_fixations, SALICON_centerbias, args.batch_size, args.num_workers, is_distributed, is_master, device, lmdb_directory / 'SALICON_train', _logger)
        validation_loader = prepare_spatial_dataset(SALICON_val_stimuli, SALICON_val_fixations, SALICON_centerbias, args.batch_size, args.num_workers, is_distributed, is_master, device, lmdb_directory / 'SALICON_val', _logger)

        output_dir = train_directory / 'salicon_pretraining'
        if is_master: _logger.info(f"Starting training {args.stage}, output: {output_dir}")
        _train(
            this_directory=str(output_dir), model=model,
            train_loader=train_loader, train_baseline_log_likelihood=train_baseline_log_likelihood,
            val_loader=validation_loader, val_baseline_log_likelihood=val_baseline_log_likelihood,
            optimizer=optimizer, lr_scheduler=lr_scheduler,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            minimum_learning_rate=args.min_lr,
            validation_metrics=['LL', 'IG', 'NSS', 'AUC_CPU'],
            startwith=None,
            device=device,
            is_distributed=is_distributed, is_master=is_master
        )
        if is_master: _logger.info("--- SALICON Pretraining Finished ---")

    elif args.stage in ['mit_spatial', 'mit_scanpath_frozen', 'mit_scanpath_full']:
        fold = args.fold
        if fold is None or not (0 <= fold < 10):
            _logger.critical("--fold required for MIT stages. Exiting.")
            if is_distributed: dist.barrier(); sys.exit(1)
        if is_master: _logger.info(f"--- Preparing MIT1003 Stage: {args.stage} (Fold {fold}) ---")

        # --- Determine Stage Specific Config ---
        output_dir_base = train_directory / args.stage / f'crossval-10-{fold}'

        if args.stage == 'mit_spatial':
            stage_downsample = 1
            stage_scanpath_features = 0
            stage_included_fixations = []
            prev_stage_dir = train_directory / 'salicon_pretraining'
            stage_lr = args.lr
            scheduler_milestones = [5, 10, 15, 20]
            ddp_find_unused = unfrozen_params_exist or args.add_sa_head
            apply_specific_freezing = False
        elif args.stage == 'mit_scanpath_frozen':
            stage_downsample = 1
            stage_scanpath_features = 16
            stage_included_fixations = [-1, -2, -3, -4]
            prev_stage_dir = train_directory / 'mit_spatial' / f'crossval-10-{fold}'
            stage_lr = args.lr
            scheduler_milestones = [10, 20, 30]
            ddp_find_unused = True
            apply_specific_freezing = True
        elif args.stage == 'mit_scanpath_full':
            stage_downsample = 1
            stage_scanpath_features = 16
            stage_included_fixations = [-1, -2, -3, -4]
            prev_stage_dir = train_directory / 'mit_scanpath_frozen' / f'crossval-10-{fold}'
            recommended_lr = 1e-5
            stage_lr = recommended_lr if args.lr != recommended_lr else args.lr
            if stage_lr != args.lr and is_master:
                _logger.warning(f"Overriding LR for 'mit_scanpath_full' to {stage_lr} (was {args.lr})")
            scheduler_milestones = [5, 10, 15]
            ddp_find_unused = unfrozen_params_exist or args.add_sa_head
            apply_specific_freezing = False
        else:
            _logger.critical(f"Invalid MIT stage configuration for {args.stage}")
            if is_distributed: dist.barrier(); sys.exit(1)

        mit_converted_stimuli_path = train_directory / 'MIT1003_twosize'
        mit_converted_stimuli_file = mit_converted_stimuli_path / 'stimuli.json'
        scanpath_cache_file = mit_converted_stimuli_path / 'scanpaths_twosize.pkl'
        needs_conversion = False
        if is_master:
            if not mit_converted_stimuli_file.exists() or not scanpath_cache_file.exists():
                needs_conversion = True
                _logger.warning(f"Converted MIT1003 data incomplete ({mit_converted_stimuli_file}, {scanpath_cache_file}). Will convert...")
        if is_distributed:
            needs_conversion_tensor = torch.tensor(int(needs_conversion), device=device, dtype=torch.int)
            dist.broadcast(needs_conversion_tensor, src=0)
            needs_conversion = bool(needs_conversion_tensor.item())

        mit_stimuli_twosize, mit_scanpaths_twosize = None, None
        if needs_conversion:
            if is_master:
                _logger.info(f"Loading original MIT1003 from {dataset_directory}...")
                try:
                    pysaliency.external_datasets.mit.get_mit1003_with_initial_fixation(location=dataset_directory, replace_initial_invalid_fixations=True)
                except ImportError:
                    _logger.error("pysaliency.external_datasets.mit not found. Cannot get dataset.")
                    if is_distributed: dist.barrier(); sys.exit(1)
                except Exception as e: _logger.critical(f"Failed to get original MIT1003: {e}"); dist.barrier(); sys.exit(1)
            if is_distributed:
                dist.barrier()
            try:
                mit_stimuli_orig, mit_scanpaths_orig = pysaliency.external_datasets.mit.get_mit1003_with_initial_fixation(location=dataset_directory, replace_initial_invalid_fixations=True)
            except ImportError:
                _logger.error("pysaliency.external_datasets.mit not found. Cannot load dataset.")
                if is_distributed: dist.barrier(); sys.exit(1)
            except Exception as e: _logger.critical(f"Failed to load original MIT1003 meta: {e}"); dist.barrier(); sys.exit(1)

            mit_stimuli_twosize = convert_stimuli(mit_stimuli_orig, mit_converted_stimuli_path, is_master, is_distributed, device, _logger)
            mit_scanpaths_twosize = convert_fixation_trains(mit_stimuli_orig, mit_scanpaths_orig, is_master, _logger)
            if mit_stimuli_twosize is None or mit_scanpaths_twosize is None: _logger.critical("MIT1003 conversion failed."); dist.barrier(); sys.exit(1)
            if is_master:
                try:
                    with atomic_save(str(scanpath_cache_file), text_mode=False, overwrite_part=True) as f: pickle.dump(mit_scanpaths_twosize, f)
                    _logger.info(f"Saved converted scanpaths to {scanpath_cache_file}")
                except Exception as e: _logger.error(f"Failed to save scanpath cache: {e}")
            if is_distributed:
                dist.barrier()
        else:
            if is_master: _logger.info(f"Loading pre-converted MIT1003 from {mit_converted_stimuli_path}...")
            try: 
                stimuli_pkl = mit_converted_stimuli_path / "stimuli.pkl"
                with open(stimuli_pkl, "rb") as f:
                    mit_stimuli_twosize = pickle.load(f)      # ← this is a FileStimuli object
            except Exception as e: _logger.critical(f"Failed load stimuli JSON: {e}"); dist.barrier(); sys.exit(1)
            try:
                with open(scanpath_cache_file, 'rb') as f: mit_scanpaths_twosize = cpickle.load(f)
                if is_master: _logger.info(f"Loaded converted scanpaths from {scanpath_cache_file}.")
            except Exception as e: _logger.critical(f"Error loading scanpath cache: {e}"); dist.barrier(); sys.exit(1)

        mit_fixations_twosize = mit_scanpaths_twosize[mit_scanpaths_twosize.scanpath_history_length > 0]
        if is_master: _logger.info("Initializing MIT1003 Baseline...")
        MIT1003_centerbias = CrossvalidatedBaselineModel(mit_stimuli_twosize, mit_fixations_twosize, bandwidth=10**-1.6667673342543432, eps=10**-14.884189168516073, caching=False)
        MIT1003_stimuli_train, MIT1003_fixations_train = pysaliency.dataset_config.train_split(mit_stimuli_twosize, mit_fixations_twosize, crossval_folds=10, fold_no=fold)
        MIT1003_stimuli_val, MIT1003_fixations_val = pysaliency.dataset_config.validation_split(mit_stimuli_twosize, mit_fixations_twosize, crossval_folds=10, fold_no=fold)

        train_baseline_log_likelihood, val_baseline_log_likelihood = None, None
        if is_master:
            _logger.info(f"Computing baseline LLs for Fold {fold}...")
            try:
                train_baseline_log_likelihood = MIT1003_centerbias.information_gain(MIT1003_stimuli_train, MIT1003_fixations_train, verbose=False, average='image')
                val_baseline_log_likelihood = MIT1003_centerbias.information_gain(MIT1003_stimuli_val, MIT1003_fixations_val, verbose=False, average='image')
                _logger.info(f"Fold {fold} Master Baseline LLs - Train: {train_baseline_log_likelihood:.5f}, Val: {val_baseline_log_likelihood:.5f}")
            except Exception as e: _logger.critical(f"Failed compute baseline LLs: {e}"); train_baseline_log_likelihood, val_baseline_log_likelihood = -999.9, -999.9
        ll_list_to_broadcast = [train_baseline_log_likelihood, val_baseline_log_likelihood]
        if is_distributed: dist.broadcast_object_list(ll_list_to_broadcast, src=0)
        train_baseline_log_likelihood, val_baseline_log_likelihood = ll_list_to_broadcast
        if train_baseline_log_likelihood == -999.9 or val_baseline_log_likelihood == -999.9: _logger.critical(f"Baseline LLs invalid on rank {rank}."); dist.barrier(); sys.exit(1)

        _logger.info("Building base model on CPU for state dict loading...")
        saliency_net = build_saliency_network(C_in, add_sa_head=args.add_sa_head)
        scanpath_net = build_scanpath_network() if stage_scanpath_features > 0 else None
        fixsel_net = build_fixation_selection_network(scanpath_features=stage_scanpath_features)

        base_model = DeepGazeIII(
            features=features.cpu(),
            saliency_network=saliency_net,
            scanpath_network=scanpath_net,
            fixation_selection_network=fixsel_net,
            downsample=stage_downsample,
            readout_factor=readout_factor,
            saliency_map_factor=4,
            included_fixations=stage_included_fixations
        )

        start_state_dict_path = prev_stage_dir / 'final_best_val.pth'
        if not start_state_dict_path.exists():
            start_state_dict_path = prev_stage_dir / 'final.pth'
            if is_master: _logger.warning(f"'final_best_val.pth' not found in {prev_stage_dir}, falling back to 'final.pth'")

        if not start_state_dict_path.exists():
            _logger.critical(f"Required start checkpoint not found: {start_state_dict_path}")
            if is_distributed: dist.barrier(); sys.exit(1)
        else:
            _logger.info(f"Loading state_dict from {start_state_dict_path} (via CPU)")
            try:
                previous_stage_state_dict = torch.load(start_state_dict_path, map_location="cpu", weights_only=False)
                missing, unexpected = base_model.load_state_dict(previous_stage_state_dict, strict=False)
                if is_master:
                    _logger.warning(f"Loaded previous stage state dict. Missing keys: {missing}")
                    _logger.warning(f"Loaded previous stage state dict. Unexpected keys: {unexpected}")
                    saliency_keys_missing = any(k not in previous_stage_state_dict for k in base_model.saliency_network.state_dict().keys())
                    fixsel_keys_missing = any(k not in previous_stage_state_dict for k in base_model.fixation_selection_network.state_dict().keys())
                    scanpath_keys_missing = (base_model.scanpath_network is not None and
                                            any(k not in previous_stage_state_dict for k in base_model.scanpath_network.state_dict().keys()))
                    _logger.info(f"Saliency Network keys potentially missing from loaded dict: {saliency_keys_missing}")
                    _logger.info(f"Fixation Selection Network keys potentially missing from loaded dict: {fixsel_keys_missing}")
                    if base_model.scanpath_network is not None:
                        _logger.info(f"Scanpath Network keys potentially missing from loaded dict: {scanpath_keys_missing}")
                _logger.info("Successfully loaded previous stage state_dict into base CPU model.")
            except Exception as e:
                _logger.error(f"Failed to load state_dict from {start_state_dict_path}: {e}. Starting with potentially random weights in head.")

        model = base_model.to(device)

        if apply_specific_freezing: # True only for mit_scanpath_frozen
            frozen_scopes = [
                "saliency_network.layernorm0", "saliency_network.conv0", "saliency_network.bias0",
                "saliency_network.layernorm1", "saliency_network.conv1", "saliency_network.bias1",
            ]
            if is_master: _logger.info(f"Freezing parameters explicitly for {args.stage}: {frozen_scopes}")
            for name, param in model.named_parameters():
                if any(name.startswith(scope) for scope in frozen_scopes):
                    param.requires_grad = False

        elif args.stage == 'mit_scanpath_full': # Specific logic only for this stage
            if is_master: _logger.info(f"Ensuring all head parameters are unfrozen for {args.stage}.")
            for name, param in model.named_parameters():
                if not name.startswith('features.backbone'):
                    param.requires_grad = True

        # --- Common training setup for all MIT stages ---
        if is_distributed:
            model = DDP(model, device_ids=[device.index], output_device=device.index, broadcast_buffers=False, find_unused_parameters=ddp_find_unused)
            if is_master: _logger.info(f"Wrapped DDP {args.stage} (find_unused={ddp_find_unused}).")

        head_params = [p for n, p in model.named_parameters() if not n.startswith('features.backbone') and p.requires_grad]
        param_groups = [{'params': head_params, 'lr': stage_lr}]
        if unfrozen_params_exist:
            backbone_params = [p for n, p in model.named_parameters() if n.startswith('features.backbone') and p.requires_grad]
            if backbone_params:
                param_groups.append({'params': backbone_params, 'lr': args.backbone_lr})
                if is_master: _logger.info(f"Using separate param group for backbone (LR={args.backbone_lr})")
        optimizer = optim.Adam(param_groups, lr=stage_lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_milestones)

        if args.stage == 'mit_spatial':
            train_loader = prepare_spatial_dataset(MIT1003_stimuli_train, MIT1003_fixations_train, MIT1003_centerbias, args.batch_size, args.num_workers, is_distributed, is_master, device, lmdb_directory / f'MIT1003_train_spatial_{fold}', _logger)
            validation_loader = prepare_spatial_dataset(MIT1003_stimuli_val, MIT1003_fixations_val, MIT1003_centerbias, args.batch_size, args.num_workers, is_distributed, is_master, device, lmdb_directory / f'MIT1003_val_spatial_{fold}', _logger)
        else: # mit_scanpath_frozen or mit_scanpath_full
            train_loader = prepare_scanpath_dataset(MIT1003_stimuli_train, MIT1003_fixations_train, MIT1003_centerbias, args.batch_size, args.num_workers, is_distributed, is_master, device, lmdb_directory / f'MIT1003_train_scanpath_{fold}', _logger)
            validation_loader = prepare_scanpath_dataset(MIT1003_stimuli_val, MIT1003_fixations_val, MIT1003_centerbias, args.batch_size, args.num_workers, is_distributed, is_master, device, lmdb_directory / f'MIT1003_val_scanpath_{fold}', _logger)

        output_dir = output_dir_base
        # Clean intermediate checkpoints only if starting a *new* fine-tuning stage from a *previous* stage's checkpoint
        if is_master and args.stage in ['mit_scanpath_frozen', 'mit_scanpath_full']: # Apply to stages that fine-tune from previous
            intermediate_checkpoints = list(output_dir.glob('step-*.pth'))
            previous_checkpoint_exists = start_state_dict_path and start_state_dict_path.exists()
            # Only clean if we successfully loaded a *previous* stage and found *current* stage intermediates
            if previous_checkpoint_exists and intermediate_checkpoints:
                _logger.warning(f"Starting fine-tuning stage {args.stage} from {start_state_dict_path}, "
                                f"but intermediate checkpoints found in {output_dir}. "
                                f"Removing them to ensure fresh optimizer/scheduler state.")
                for cp in intermediate_checkpoints:
                    try: cp.unlink()
                    except OSError as e: _logger.error(f"Failed to remove {cp}: {e}")
            elif not previous_checkpoint_exists:
                _logger.error(f"Previous stage checkpoint {start_state_dict_path} not found, cannot start fine-tuning!")
                if is_distributed: dist.barrier(); sys.exit(1)

        if is_master: _logger.info(f"Starting training {args.stage} (Fold {fold}), output: {output_dir}")
        _train(
            this_directory=str(output_dir), model=model,
            train_loader=train_loader, train_baseline_log_likelihood=train_baseline_log_likelihood,
            val_loader=validation_loader, val_baseline_log_likelihood=val_baseline_log_likelihood,
            optimizer=optimizer, lr_scheduler=lr_scheduler,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            minimum_learning_rate=args.min_lr,
            device=device,
            startwith=None, # _train handles resumption from within output_dir
            validation_metrics=['LL', 'IG', 'NSS', 'AUC_CPU'],
            is_distributed=is_distributed, is_master=is_master,
            logger=_logger,
        )
        if is_master: _logger.info(f"--- Stage {args.stage} Finished (Fold {fold}) ---")

    else:
        _logger.critical(f"Unknown stage: {args.stage}");
        if is_distributed: dist.barrier();
        sys.exit(1)

    cleanup_distributed()
    if is_master:
        _logger.info("==================================================")
        _logger.info("Training script finished successfully.")
        _logger.info("==================================================")



# ──────────────────────────────────────────────────────────────────────────────
#  Main entry-point  –  YAML config *plus* CLI  (CLI has top priority)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ------------------------------------------------------------------
    # 1) Tiny “pre-parser” – only to capture --config_file so we can
    #    load the YAML *before* we declare real defaults.
    # ------------------------------------------------------------------
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Optional YAML file whose values become defaults; "
             "explicit CLI flags still override."
    )
    _cfg, _remaining_cli = _pre.parse_known_args()

    # ------------------------------------------------------------------
    # 2) Full argument parser – exactly the flags you had before.
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        parents=[_pre],      # inherit --config_file
        description="Train DeepGaze-III with DINOv2 backbone (multi-GPU via torchrun)"
    )

    # ── Core arguments ────────────────────────────────────────────────
    parser.add_argument('--stage', required=True,
        choices=['salicon_pretrain',
                 'mit_spatial',
                 'mit_scanpath_frozen',
                 'mit_scanpath_full'],
        help='Training stage to execute.')
    parser.add_argument('--model_name',
        default='dinov2_vitg14',
        choices=['dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'],
        help='DINOv2 model variant.')
    parser.add_argument('--layers', type=int, nargs='+',
        default=[-3, -2, -1],
        help='Indices of transformer blocks to extract features from.')

    # ── Training hyper-parameters ─────────────────────────────────────
    parser.add_argument('--batch_size', type=int, default=4,
        help='Batch size *per GPU*. Effective global batch size is '
             'batch_size × world_size × gradient_accumulation_steps.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-4,
        help='Initial learning rate for the model head.')
    parser.add_argument('--backbone_lr', type=float, default=1e-5,
        help='Learning rate for unfrozen backbone layers.')
    parser.add_argument('--min_lr', type=float, default=1e-7,
        help='Lower LR bound for schedulers.')
    parser.add_argument('--fold', type=int,
        help='Cross-validation fold (MIT stages only, 0-9).')

    # ── Dataloading & system ──────────────────────────────────────────
    parser.add_argument('--num_workers', type=int, default=None,
        help="Dataloader workers per rank.  None/'auto' ⇒ cpu_cores // WORLD_SIZE.")
    parser.add_argument('--train_dir', default='./train_dinogaze_vitg')
    parser.add_argument('--dataset_dir', default='../data/pysaliency_datasets')
    parser.add_argument('--lmdb_dir', default='../data/lmdb_cache_dinogaze_vitg')

    # ── Model tweaks / ablations ──────────────────────────────────────
    parser.add_argument('--add_sa_head', action='store_true',
        help='Add a Self-Attention layer before the saliency network.')
    parser.add_argument('--unfreeze_vit_layers', type=int, nargs='+', default=[],
        help='Indices of ViT blocks to unfreeze & fine-tune.')

    # ------------------------------------------------------------------
    # 3) If a YAML file was given, load it and set its keys as defaults.
    # ------------------------------------------------------------------
    if _cfg.config_file:
        try:
            with open(_cfg.config_file, 'r') as yml:
                yaml_cfg = yaml.safe_load(yml) or {}
            print(f"Loaded configuration from: {_cfg.config_file}")
            parser.set_defaults(**yaml_cfg)
        except Exception as e:
            print(f"⚠️  Could not read YAML '{_cfg.config_file}': {e} – "
                  "continuing with built-in defaults.")

    # ------------------------------------------------------------------
    # 4) Final parse – CLI overrides YAML which overrides built-ins.
    # ------------------------------------------------------------------
    args = parser.parse_args(_remaining_cli)

    # ------------------------------------------------------------------
    # 5) Automatic worker-count logic (unchanged from your original).
    # ------------------------------------------------------------------
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if args.num_workers is None:  # treat None or 'auto'
        try:
            cpu_count = len(os.sched_getaffinity(0))
        except AttributeError:
            cpu_count = os.cpu_count() or 1
        args.num_workers = max(0, cpu_count // world_size)
    elif args.num_workers < 0:
        args.num_workers = 0  # negative ⇒ 0 workers

    # ------------------------------------------------------------------
    # 6) Stage-specific LR warning (kept exactly as you had it).
    # ------------------------------------------------------------------
    if args.stage == 'mit_scanpath_full' and args.lr > 1e-5:
        print(f"WARNING: Recommended head LR for 'mit_scanpath_full' is 1e-5, "
              f"but got {args.lr}. Using provided value.")

    # ------------------------------------------------------------------
    # 7) Standard try/except wrapper around main().
    # ------------------------------------------------------------------
    try:
        main(args)
    except KeyboardInterrupt:
        try:
            _logger.warning("Training interrupted by user (Ctrl-C). Cleaning up …")
        except NameError:
            print("Training interrupted by user (Ctrl-C). Cleaning up …")
        cleanup_distributed()
        sys.exit(130)
    except Exception:
        try:
            _logger.critical("Unhandled exception during main execution:", exc_info=True)
        except NameError:
            import traceback; traceback.print_exc()
        cleanup_distributed()
        sys.exit(1)
