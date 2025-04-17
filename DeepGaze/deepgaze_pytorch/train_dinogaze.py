# train_dinogaze.py
import argparse
import logging
import os
import pickle
import shutil
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import pysaliency
import torch
import torch.nn as nn
import torch.optim as optim
from imageio.v3 import imread, imwrite
from PIL import Image
from pysaliency.baseline_utils import (BaselineModel,
                                       CrossvalidatedBaselineModel)
from tqdm import tqdm

# Make sure the deepgaze_pytorch directory is in the Python path
# Adjust this relative path if your script is located elsewhere
script_dir = Path(__file__).parent.resolve()
sys.path.append(str(script_dir / '../')) # Assuming deepgaze_pytorch is one level up

# --- DeepGaze III Imports ---
# (Ensure these paths are correct relative to where you place train_dinogaze.py)
try:
    from deepgaze_pytorch.data import (FixationDataset, FixationMaskTransform,
                                     ImageDataset, ImageDatasetSampler)
    from deepgaze_pytorch.dinov2_backbone import DinoV2Backbone
    from deepgaze_pytorch.layers import (Bias, Conv2dMultiInput,
                                       FlexibleScanpathHistoryEncoding,
                                       LayerNorm, LayerNormMultiInput)
    from deepgaze_pytorch.modules import DeepGazeIII, FeatureExtractor
    from deepgaze_pytorch.training import _train
    # Import the original DenseNet only if needed for comparison or specific stages
    # from deepgaze_pytorch.features.densenet import RGBDenseNet201
except ImportError as e:
    print(f"Error importing DeepGaze modules: {e}")
    print("Please ensure 'deepgaze_pytorch' directory is accessible.")
    sys.exit(1)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Model Building Functions (Copied from notebook) ---
def build_saliency_network(input_channels):
    logging.info(f"Building saliency network with {input_channels} input channels.")
    # Reduced complexity slightly given potentially richer ViT features
    return nn.Sequential(OrderedDict([
        ('layernorm0', LayerNorm(input_channels)),
        ('conv0', nn.Conv2d(input_channels, 64, (1, 1), bias=False)), # Increased capacity slightly
        ('bias0', Bias(64)),
        ('softplus0', nn.Softplus()),

        ('layernorm1', LayerNorm(64)),
        ('conv1', nn.Conv2d(64, 32, (1, 1), bias=False)), # Increased capacity slightly
        ('bias1', Bias(32)),
        ('softplus1', nn.Softplus()),

        ('layernorm2', LayerNorm(32)),
        ('conv2', nn.Conv2d(32, 1, (1, 1), bias=False)),
        ('bias2', Bias(1)),
        ('softplus2', nn.Softplus()),
    ]))

def build_scanpath_network():
    return nn.Sequential(OrderedDict([
        ('encoding0', FlexibleScanpathHistoryEncoding(in_fixations=4, channels_per_fixation=3, out_channels=128, kernel_size=[1, 1], bias=True)),
        ('softplus0', nn.Softplus()),
        ('layernorm1', LayerNorm(128)),
        ('conv1', nn.Conv2d(128, 16, (1, 1), bias=False)),
        ('bias1', Bias(16)),
        ('softplus1', nn.Softplus()),
    ]))

def build_fixation_selection_network(scanpath_features=16):
    logging.info(f"Building fixation selection network with scanpath features={scanpath_features}")
    # Adjust input layer based on whether scanpath features are present
    saliency_channels = 1 # Output of saliency network's core path before combination
    in_channels_list = [saliency_channels, scanpath_features if scanpath_features > 0 else 0]
    return nn.Sequential(OrderedDict([
        ('layernorm0', LayerNormMultiInput(in_channels_list)),
        ('conv0', Conv2dMultiInput(in_channels_list, 128, (1, 1), bias=False)),
        ('bias0', Bias(128)),
        ('softplus0', nn.Softplus()),
        ('layernorm1', LayerNorm(128)),
        ('conv1', nn.Conv2d(128, 16, (1, 1), bias=False)),
        ('bias1', Bias(16)),
        ('softplus1', nn.Softplus()),
        ('conv2', nn.Conv2d(16, 1, (1, 1), bias=False)),
    ]))

# --- Data Preparation Functions (Copied from notebook) ---
def prepare_spatial_dataset(stimuli, fixations, centerbias, batch_size, num_workers, path=None):
    lmdb_path = str(path) if path else None
    if lmdb_path:
        path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Using LMDB cache for spatial dataset at: {lmdb_path}")

    dataset = ImageDataset(
        stimuli=stimuli, fixations=fixations, centerbias_model=centerbias,
        transform=FixationMaskTransform(sparse=False), average='image', lmdb_path=lmdb_path
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=ImageDatasetSampler(dataset, batch_size=batch_size),
        pin_memory=True, num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False, prefetch_factor=2 if num_workers > 0 else None
    )
    return loader

def prepare_scanpath_dataset(stimuli, fixations, centerbias, batch_size, num_workers, path=None):
    lmdb_path = str(path) if path else None
    if lmdb_path:
        path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Using LMDB cache for scanpath dataset at: {lmdb_path}")

    dataset = FixationDataset(
        stimuli=stimuli, fixations=fixations, centerbias_model=centerbias,
        included_fixations=[-1, -2, -3, -4], allow_missing_fixations=True,
        transform=FixationMaskTransform(sparse=False), average='image', lmdb_path=lmdb_path
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=ImageDatasetSampler(dataset, batch_size=batch_size),
        pin_memory=True, num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False, prefetch_factor=2 if num_workers > 0 else None
    )
    return loader

# --- MIT Data Conversion Functions (Copied from notebook) ---
def convert_stimulus(input_image):
    size = input_image.shape[:2]
    new_size = (768, 1024) if size[0] < size[1] else (1024, 768)
    new_size_pil = tuple(list(new_size)[::-1]) # pillow uses width, height
    return np.array(Image.fromarray(input_image).resize(new_size_pil, Image.BILINEAR))

def convert_stimuli(stimuli, new_location: Path):
    assert isinstance(stimuli, pysaliency.FileStimuli)
    new_stimuli_location = new_location / 'stimuli'
    new_stimuli_location.mkdir(parents=True, exist_ok=True)
    new_filenames = []
    logging.info(f"Converting stimuli resolution and saving to {new_stimuli_location}...")
    for filename in tqdm(stimuli.filenames, desc="Converting Stimuli"):
        stimulus = imread(filename)
        # Ensure 3 channels if grayscale
        if stimulus.ndim == 2:
            stimulus = np.stack([stimulus]*3, axis=-1)
        elif stimulus.shape[2] == 1:
             stimulus = np.concatenate([stimulus]*3, axis=-1)
        elif stimulus.shape[2] == 4: # Handle RGBA
             stimulus = stimulus[:,:,:3]

        if stimulus.shape[2] != 3:
             logging.warning(f"Skipping stimulus {filename} with unexpected shape {stimulus.shape}")
             continue # Or handle differently

        new_stimulus = convert_stimulus(stimulus)
        basename = os.path.basename(filename)
        new_filename = new_stimuli_location / basename
        if new_stimulus.shape != stimulus.shape: # Only write if changed
            try:
                imwrite(new_filename, new_stimulus)
            except Exception as e:
                 logging.error(f"Failed to write {new_filename}: {e}")
                 continue # Skip this image if writing fails
        else:
            shutil.copy(filename, new_filename)
        new_filenames.append(new_filename)
    return pysaliency.FileStimuli(new_filenames)


def convert_fixation_trains(stimuli, fixations):
    logging.info("Converting fixation coordinates...")
    train_xs = fixations.train_xs.copy()
    train_ys = fixations.train_ys.copy()
    shapes_cache = stimuli.shapes # Cache shapes for efficiency
    for i in tqdm(range(len(train_xs)), desc="Converting Fixations"):
        n = fixations.train_ns[i]
        size = shapes_cache[n][:2] # Use cached shape
        new_size = (768, 1024) if size[0] < size[1] else (1024, 768)
        x_factor = new_size[1] / size[1]
        y_factor = new_size[0] / size[0]
        train_xs[i] *= x_factor
        train_ys[i] *= y_factor
    return pysaliency.FixationTrains(
        train_xs=train_xs, train_ys=train_ys, train_ts=fixations.train_ts.copy(),
        train_ns=fixations.train_ns.copy(), train_subjects=fixations.train_subjects.copy(),
        attributes={k: getattr(fixations, k).copy() for k in fixations.__attributes__ if k not in ['subjects', 'scanpath_index']}
    )


# --- Main Execution ---
def main(args):
    """ Main training function """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    logging.info(f"Running stage: {args.stage}")
    logging.info(f"Using DINOv2 model: {args.model_name}")
    logging.info(f"Extracting from layers: {args.layers}")
    logging.info(f"Using batch size: {args.batch_size}")

    dataset_directory = Path(args.dataset_dir).resolve()
    train_directory = Path(args.train_dir).resolve()
    lmdb_directory = Path(args.lmdb_dir).resolve()

    dataset_directory.mkdir(parents=True, exist_ok=True)
    train_directory.mkdir(parents=True, exist_ok=True)
    lmdb_directory.mkdir(parents=True, exist_ok=True)

    # --- Feature Extractor ---
    features = DinoV2Backbone(
        layers=args.layers,
        model_name=args.model_name,
        freeze=True # Keep frozen for all stages initially
    )
    C_in = len(features.layers) * features.num_channels
    logging.info(f"Feature extractor initialized. Input channels to saliency network: {C_in}")

    # --- Stage Specific Logic ---

    if args.stage == 'salicon_pretrain':
        logging.info("--- Starting SALICON Pretraining ---")
        # Load Data
        logging.info(f"Loading SALICON data from {dataset_directory}...")
        SALICON_train_stimuli, SALICON_train_fixations = pysaliency.get_SALICON_train(location=dataset_directory)
        SALICON_val_stimuli, SALICON_val_fixations = pysaliency.get_SALICON_val(location=dataset_directory)
        logging.info("SALICON data loaded.")

        # Baseline Model & LLs (Cached)
        logging.info("Initializing/Loading SALICON BaselineModel...")
        SALICON_centerbias = BaselineModel(stimuli=SALICON_train_stimuli, fixations=SALICON_train_fixations, bandwidth=0.0217, eps=2e-13, caching=False)
        train_ll_cache_file = dataset_directory / 'salicon_baseline_train_ll.pkl'
        val_ll_cache_file = dataset_directory / 'salicon_baseline_val_ll.pkl'

        try:
            with open(train_ll_cache_file, 'rb') as f: train_baseline_log_likelihood = pickle.load(f)
            logging.info(f"Loaded cached train baseline LL from: {train_ll_cache_file}")
        except Exception as e:
            logging.warning(f"Cache not found or invalid ({e}). Computing train baseline LL...")
            train_baseline_log_likelihood = SALICON_centerbias.information_gain(SALICON_train_stimuli, SALICON_train_fixations, verbose=True, average='image')
            try:
                with open(train_ll_cache_file, 'wb') as f: pickle.dump(train_baseline_log_likelihood, f)
                logging.info(f"Saved train baseline LL to: {train_ll_cache_file}")
            except Exception as save_e: logging.error(f"Error saving cache file {train_ll_cache_file}: {save_e}")

        try:
            with open(val_ll_cache_file, 'rb') as f: val_baseline_log_likelihood = pickle.load(f)
            logging.info(f"Loaded cached validation baseline LL from: {val_ll_cache_file}")
        except Exception as e:
            logging.warning(f"Cache not found or invalid ({e}). Computing validation baseline LL...")
            val_baseline_log_likelihood = SALICON_centerbias.information_gain(SALICON_val_stimuli, SALICON_val_fixations, verbose=True, average='image')
            try:
                with open(val_ll_cache_file, 'wb') as f: pickle.dump(val_baseline_log_likelihood, f)
                logging.info(f"Saved validation baseline LL to: {val_ll_cache_file}")
            except Exception as save_e: logging.error(f"Error saving cache file {val_ll_cache_file}: {save_e}")

        logging.info(f"Final Train Baseline Log Likelihood: {train_baseline_log_likelihood}")
        logging.info(f"Final Validation Baseline Log Likelihood: {val_baseline_log_likelihood}")

        # Model Definition (Spatial Only)
        model = DeepGazeIII(
            features=features,
            saliency_network=build_saliency_network(C_in),
            scanpath_network=None,
            fixation_selection_network=build_fixation_selection_network(scanpath_features=0),
            downsample=1, readout_factor=14, saliency_map_factor=4, included_fixations=[]
        ).to(device)

        # Optimizer & Scheduler
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        # Adjusted milestones for potentially longer pretraining
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90, 120, 150, 180])

        # DataLoaders
        train_loader = prepare_spatial_dataset(
            SALICON_train_stimuli, SALICON_train_fixations, SALICON_centerbias,
            args.batch_size, args.num_workers, path=lmdb_directory / 'SALICON_train'
        )
        validation_loader = prepare_spatial_dataset(
            SALICON_val_stimuli, SALICON_val_fixations, SALICON_centerbias,
            args.batch_size, args.num_workers, path=lmdb_directory / 'SALICON_val'
        )

        # Training
        output_dir = train_directory / 'salicon_pretraining'
        logging.info(f"Starting training, outputting to: {output_dir}")
        _train(
            output_dir, model,
            train_loader, train_baseline_log_likelihood,
            validation_loader, val_baseline_log_likelihood,
            optimizer, lr_scheduler,
            minimum_learning_rate=args.min_lr, device=device,
            validation_metrics=['LL', 'IG', 'NSS', 'AUC_CPU'] # Use CPU AUC for stability
        )
        logging.info("--- SALICON Pretraining Finished ---")

    elif args.stage in ['mit_spatial', 'mit_scanpath_frozen', 'mit_scanpath_full']:
        fold = args.fold
        if fold is None or not (0 <= fold < 10):
            raise ValueError("A valid --fold (0-9) is required for MIT1003 stages.")
        logging.info(f"--- Starting MIT1003 Stage: {args.stage} (Fold {fold}) ---")

        # Load/Convert MIT Data (Run conversion if needed)
        mit_converted_stimuli_path = train_directory / 'MIT1003_twosize'
        mit_converted_stimuli_file = mit_converted_stimuli_path / 'stimuli.json' # pysaliency saves metadata here

        if not mit_converted_stimuli_file.exists():
             logging.warning(f"Converted MIT1003 stimuli not found at {mit_converted_stimuli_path}. Converting now...")
             mit_stimuli_orig, mit_scanpaths_orig = pysaliency.external_datasets.mit.get_mit1003_with_initial_fixation(
                 location=dataset_directory, replace_initial_invalid_fixations=True
             )
             mit_stimuli_twosize = convert_stimuli(mit_stimuli_orig, mit_converted_stimuli_path)
             mit_scanpaths_twosize = convert_fixation_trains(mit_stimuli_twosize, mit_scanpaths_orig)
             # Save scanpaths for faster loading next time
             scanpath_cache_file = mit_converted_stimuli_path / 'scanpaths_twosize.pkl'
             with open(scanpath_cache_file, 'wb') as f:
                 pickle.dump(mit_scanpaths_twosize, f)
             logging.info(f"Saved converted scanpaths to {scanpath_cache_file}")
        else:
             logging.info(f"Loading converted MIT1003 stimuli from {mit_converted_stimuli_path}")
             mit_stimuli_twosize = pysaliency.read_json(mit_converted_stimuli_file)
             scanpath_cache_file = mit_converted_stimuli_path / 'scanpaths_twosize.pkl'
             try:
                 with open(scanpath_cache_file, 'rb') as f:
                     mit_scanpaths_twosize = pickle.load(f)
                 logging.info(f"Loaded converted scanpaths from {scanpath_cache_file}")
             except FileNotFoundError:
                 logging.error(f"Scanpath cache file {scanpath_cache_file} not found. Please run conversion again.")
                 sys.exit(1)


        mit_fixations_twosize = mit_scanpaths_twosize[mit_scanpaths_twosize.lengths > 0]

        # Baseline Model & LLs
        logging.info("Initializing MIT1003 Crossvalidated BaselineModel...")
        MIT1003_centerbias = CrossvalidatedBaselineModel(
            mit_stimuli_twosize, mit_fixations_twosize,
            bandwidth=10**-1.6667673342543432, eps=10**-14.884189168516073, caching=False
        )

        MIT1003_stimuli_train, MIT1003_fixations_train = pysaliency.dataset_config.train_split(mit_stimuli_twosize, mit_fixations_twosize, crossval_folds=10, fold_no=fold)
        MIT1003_stimuli_val, MIT1003_fixations_val = pysaliency.dataset_config.validation_split(mit_stimuli_twosize, mit_fixations_twosize, crossval_folds=10, fold_no=fold)

        logging.info("Computing baseline log likelihoods for the current fold...")
        # Compute LLs on the fly for the specific fold - caching this per fold is complex
        train_baseline_log_likelihood = MIT1003_centerbias.information_gain(MIT1003_stimuli_train, MIT1003_fixations_train, verbose=False, average='image')
        val_baseline_log_likelihood = MIT1003_centerbias.information_gain(MIT1003_stimuli_val, MIT1003_fixations_val, verbose=False, average='image')
        logging.info(f"Fold {fold} Train Baseline LL: {train_baseline_log_likelihood}")
        logging.info(f"Fold {fold} Validation Baseline LL: {val_baseline_log_likelihood}")

        # --- Stage-Specific Model/Training Setup ---
        if args.stage == 'mit_spatial':
            # Model Definition (Spatial Only)
            model = DeepGazeIII(
                features=features, saliency_network=build_saliency_network(C_in), scanpath_network=None,
                fixation_selection_network=build_fixation_selection_network(scanpath_features=0),
                downsample=1, readout_factor=14, saliency_map_factor=4, included_fixations=[]
            ).to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.lr) # Use provided LR
             # Shorter schedule for fine-tuning
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15, 20])

            train_loader = prepare_spatial_dataset(
                MIT1003_stimuli_train, MIT1003_fixations_train, MIT1003_centerbias,
                args.batch_size, args.num_workers, path=lmdb_directory / f'MIT1003_train_spatial_{fold}'
            )
            validation_loader = prepare_spatial_dataset(
                MIT1003_stimuli_val, MIT1003_fixations_val, MIT1003_centerbias,
                args.batch_size, args.num_workers, path=lmdb_directory / f'MIT1003_val_spatial_{fold}'
            )
            start_checkpoint = train_directory / 'salicon_pretraining' / 'final.pth'
            if not start_checkpoint.exists():
                logging.error(f"SALICON pretraining checkpoint not found at {start_checkpoint}. Run pretraining first.")
                sys.exit(1)

            output_dir = train_directory / 'mit_spatial' / f'crossval-10-{fold}'
            logging.info(f"Starting spatial fine-tuning, outputting to: {output_dir}")
            _train(
                output_dir, model, train_loader, train_baseline_log_likelihood,
                validation_loader, val_baseline_log_likelihood, optimizer, lr_scheduler,
                minimum_learning_rate=args.min_lr, device=device, startwith=start_checkpoint,
                validation_metrics=['LL', 'IG', 'NSS', 'AUC_CPU']
            )
            logging.info(f"--- MIT Spatial Fine-tuning Finished (Fold {fold}) ---")

        elif args.stage == 'mit_scanpath_frozen':
            # Model Definition (Full, with Scanpath)
            model = DeepGazeIII(
                features=features, saliency_network=build_saliency_network(C_in),
                scanpath_network=build_scanpath_network(),
                fixation_selection_network=build_fixation_selection_network(scanpath_features=16), # Default 16 channels from scanpath net
                downsample=1, readout_factor=14, saliency_map_factor=4,
                included_fixations=[-1, -2, -3, -4] # Include history
            ).to(device)

            # Freeze early saliency layers
            frozen_scopes = [
                "saliency_network.layernorm0", "saliency_network.conv0", "saliency_network.bias0",
                "saliency_network.layernorm1", "saliency_network.conv1", "saliency_network.bias1",
            ]
            logging.info("Freezing parameters in scopes: {}".format(', '.join(frozen_scopes)))
            for scope in frozen_scopes:
                for name, param in model.named_parameters():
                    if name.startswith(scope):
                        param.requires_grad = False
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr) # Only optimize non-frozen
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30]) # Adjust schedule

            train_loader = prepare_scanpath_dataset(
                 MIT1003_stimuli_train, MIT1003_fixations_train, MIT1003_centerbias,
                 args.batch_size, args.num_workers, path=lmdb_directory / f'MIT1003_train_scanpath_{fold}'
            )
            validation_loader = prepare_scanpath_dataset(
                 MIT1003_stimuli_val, MIT1003_fixations_val, MIT1003_centerbias,
                 args.batch_size, args.num_workers, path=lmdb_directory / f'MIT1003_val_scanpath_{fold}'
            )
            start_checkpoint = train_directory / 'mit_spatial' / f'crossval-10-{fold}' / 'final.pth'
            if not start_checkpoint.exists():
                 logging.error(f"MIT spatial checkpoint not found at {start_checkpoint}. Run spatial tuning first.")
                 sys.exit(1)

            output_dir = train_directory / 'mit_scanpath_frozen' / f'crossval-10-{fold}'
            logging.info(f"Starting frozen scanpath training, outputting to: {output_dir}")
            _train(
                output_dir, model, train_loader, train_baseline_log_likelihood,
                validation_loader, val_baseline_log_likelihood, optimizer, lr_scheduler,
                minimum_learning_rate=args.min_lr, device=device, startwith=start_checkpoint,
                validation_metrics=['LL', 'IG', 'NSS', 'AUC_CPU']
            )
            logging.info(f"--- MIT Frozen Scanpath Training Finished (Fold {fold}) ---")

        elif args.stage == 'mit_scanpath_full':
             # Model Definition (Full, all trainable)
            model = DeepGazeIII(
                features=features, saliency_network=build_saliency_network(C_in),
                scanpath_network=build_scanpath_network(),
                fixation_selection_network=build_fixation_selection_network(scanpath_features=16),
                downsample=1, readout_factor=14, saliency_map_factor=4,
                included_fixations=[-1, -2, -3, -4]
            ).to(device)

            # Very low LR for final fine-tuning
            optimizer = optim.Adam(model.parameters(), lr=args.lr) # Use low LR passed via args
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15]) # Short schedule

            train_loader = prepare_scanpath_dataset(
                 MIT1003_stimuli_train, MIT1003_fixations_train, MIT1003_centerbias,
                 args.batch_size, args.num_workers, path=lmdb_directory / f'MIT1003_train_scanpath_{fold}' # Can reuse LMDB
            )
            validation_loader = prepare_scanpath_dataset(
                 MIT1003_stimuli_val, MIT1003_fixations_val, MIT1003_centerbias,
                 args.batch_size, args.num_workers, path=lmdb_directory / f'MIT1003_val_scanpath_{fold}' # Can reuse LMDB
            )
            start_checkpoint = train_directory / 'mit_scanpath_frozen' / f'crossval-10-{fold}' / 'final.pth'
            if not start_checkpoint.exists():
                 logging.error(f"MIT frozen scanpath checkpoint not found at {start_checkpoint}. Run frozen scanpath training first.")
                 sys.exit(1)

            output_dir = train_directory / 'mit_scanpath_full' / f'crossval-10-{fold}'
            logging.info(f"Starting full scanpath fine-tuning, outputting to: {output_dir}")
            _train(
                output_dir, model, train_loader, train_baseline_log_likelihood,
                validation_loader, val_baseline_log_likelihood, optimizer, lr_scheduler,
                minimum_learning_rate=args.min_lr, device=device, startwith=start_checkpoint,
                validation_metrics=['LL', 'IG', 'NSS', 'AUC_CPU']
            )
            logging.info(f"--- MIT Full Scanpath Fine-tuning Finished (Fold {fold}) ---")

    else:
        raise ValueError(f"Unknown stage: {args.stage}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeepGazeIII with DINOv2 Backbone")
    parser.add_argument('--stage', required=True, choices=['salicon_pretrain', 'mit_spatial', 'mit_scanpath_frozen', 'mit_scanpath_full'], help='Training stage to execute.')
    parser.add_argument('--model_name', default='dinov2_vitg14', choices=['dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'], help='DINOv2 model variant.')
    parser.add_argument('--layers', type=int, nargs='+', default=[-3, -2, -1], help='Indices of transformer layers to extract features from.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size. Reduce significantly for ViT-G/14!') # Default low for ViT-G
    parser.add_argument('--lr', type=float, default=0.0005, help='Initial learning rate.') # Slightly lower default LR
    parser.add_argument('--min_lr', type=float, default=1e-7, help='Minimum learning rate for scheduler.')
    parser.add_argument('--fold', type=int, help='Cross-validation fold for MIT1003 stages (0-9).')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of dataloader workers. Defaults to cpu_count(). Set to 0 to disable multiprocessing.')
    parser.add_argument('--train_dir', default='./train_dinogaze_vitg', help='Base directory for training outputs (checkpoints, logs).')
    parser.add_argument('--dataset_dir', default='./pysaliency_datasets', help='Directory to store/cache datasets.')
    parser.add_argument('--lmdb_dir', default='./lmdb_cache_dinogaze_vitg', help='Directory for LMDB data caches.')

    args = parser.parse_args()

    if args.num_workers is None:
        args.num_workers = os.cpu_count()
        logging.info(f"Using default number of workers: {args.num_workers}")
    elif args.num_workers < 0:
         logging.warning("num_workers cannot be negative. Setting to 0.")
         args.num_workers = 0

    # Adjust default LR for final scanpath stage if needed
    if args.stage == 'mit_scanpath_full' and args.lr > 1e-5:
         logging.warning(f"Setting LR for full scanpath stage to 1e-5 (was {args.lr})")
         args.lr = 1e-5 # Override default if too high for this stage

    main(args)