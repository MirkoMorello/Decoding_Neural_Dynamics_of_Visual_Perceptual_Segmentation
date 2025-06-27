# data.py
from collections import Counter
import io
import os
import pickle
import random
import shutil
from pathlib import Path
from boltons.iterutils import chunked
import lmdb
import numpy as np
from PIL import Image
import pysaliency
from pysaliency.utils import remove_trailing_nans
import torch
from tqdm import tqdm
import torch.distributed as dist
from imageio.v3 import imread, imwrite
from torch.utils.data import Sampler, RandomSampler, BatchSampler
import numpy as np
import torch
import cloudpickle as cpickle
import json
import logging

logger = logging.getLogger(__name__)


def ensure_color_image(image):
    if len(image.shape) == 2:
        return np.dstack([image, image, image])
    return image


def x_y_to_sparse_indices(xs, ys):
    # Converts list of x and y coordinates into indices and values for sparse mask
    x_inds = []
    y_inds = []
    values = []
    pair_inds = {}

    for x, y in zip(xs, ys):
        key = (x, y)
        if key not in pair_inds:
            x_inds.append(x)
            y_inds.append(y)
            pair_inds[key] = len(x_inds) - 1
            values.append(1)
        else:
            values[pair_inds[key]] += 1

    return np.array([y_inds, x_inds]), values


class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        stimuli: pysaliency.Stimuli,
        fixations: pysaliency.Fixations,
        centerbias_model: pysaliency.Model | None = None,
        lmdb_path: str | Path | None = None,
        transform=None,
        cached: bool | None = None,
        average: str = 'fixation'
    ):
        self.stimuli = stimuli
        self.fixations = fixations
        self.centerbias_model = centerbias_model
        self.lmdb_path = Path(lmdb_path) if lmdb_path else None
        self.transform = transform
        self.average = average

        if cached is None:
            self.cached = len(self.stimuli) < 100
        else:
            self.cached = cached
        self.cache_fixation_data = self.cached

        self.lmdb_env = None # Initialize to None

        if self.lmdb_path:
            if self.centerbias_model is None: # Condition from your _export_dataset_to_lmdb path
                logger.warning(f"ImageDataset: LMDB path '{self.lmdb_path}' provided, but centerbias_model is None. "
                               "Cannot create/validate LMDB for image/centerbias. LMDB will NOT be used.")
            elif not isinstance(self.stimuli, pysaliency.FileStimuli):
                logger.warning(f"ImageDataset: LMDB path '{self.lmdb_path}' provided, but stimuli object is not FileStimuli. "
                               "Cannot get filenames for LMDB creation. LMDB will NOT be used.")
            else:
                # Always call _export_dataset_to_lmdb if lmdb_path and requirements are met.
                # This function will handle checking for existing valid DB or regenerating it.
                # It should typically only run on master process if in DDP.
                # We'll assume the main script handles calling this on master only, or _export handles DDP.
                # For now, let's assume it's called by each process but only master writes.
                # A dist.barrier() might be needed in main script after dataset init if master creates it.
                
                # --- LMDB EXPORT CALL ---
                # This call needs to be robust, e.g., only master rank writes.
                # For simplicity in the Dataset class, we just call it.
                # The _export_dataset_to_lmdb function itself should be DDP-aware if necessary,
                # or the calling script (main.py) should manage DDP for this setup step.
                # Let's assume for now that if it's called by multiple DDP processes,
                # lmdb.open with subdir=True and file locks will mostly handle it,
                # or the "check if valid and complete" part prevents redundant writes.
                # The current _export_dataset_to_lmdb is mostly designed for single-process execution.
                # A truly DDP-safe version would have master create, others wait.
                
                # Simplification: _export_dataset_to_lmdb will check existence.
                # If you are in DDP, this export should ideally be done by rank 0 before other ranks try to open.
                # The main training script should orchestrate this.
                # However, the current _export_dataset_to_lmdb includes a check for valid and complete DB.
                
                # Check if master and if export needed (this is a simplified check)
                is_master_process = True # Placeholder, this should be passed or determined
                if 'RANK' in os.environ: # Basic DDP check
                    is_master_process = int(os.environ.get("RANK",0)) == 0

                if is_master_process: # Only master attempts to create/validate fully
                    logger.info(f"ImageDataset (Master): Ensuring LMDB at {self.lmdb_path} is up-to-date...")
                    _export_dataset_to_lmdb(self.stimuli, self.centerbias_model, str(self.lmdb_path))
                
                # All processes wait if in DDP, ensuring master finishes DB creation/check
                if 'WORLD_SIZE' in os.environ and int(os.environ.get("WORLD_SIZE", 1)) > 1:
                    if dist.is_initialized(): # Check if DDP is actually active
                        logger.debug(f"ImageDataset (Rank {os.environ.get('RANK')}): Waiting at barrier for LMDB.")
                        dist.barrier()
                    else: # DDP env vars set, but not initialized (e.g. before DDP group init)
                        # This state is tricky. For now, we proceed.
                        pass


                # After potential creation/validation, try to open read-only
                lmdb_data_file = self.lmdb_path / "data.mdb"
                if lmdb_data_file.exists() and self.lmdb_path.is_dir():
                    try:
                        self.lmdb_env = lmdb.open(str(self.lmdb_path), subdir=True,
                                                readonly=True, lock=False, readahead=False, meminit=False)
                        logger.info(f"ImageDataset: Successfully opened LMDB: {self.lmdb_path}")
                        self.cached = False
                        self.cache_fixation_data = True
                    except lmdb.Error as e:
                        logger.error(f"ImageDataset: Failed to open LMDB at {self.lmdb_path} even after export attempt: {e}. LMDB will NOT be used.")
                        self.lmdb_env = None # Ensure it's None on failure
                else:
                    logger.warning(f"ImageDataset: LMDB data file {lmdb_data_file} still not found after export attempt. LMDB will NOT be used.")
                    self.lmdb_env = None
        # else: self.lmdb_env remains None (initialized above)

        if self.cached:
            self._cache = {}
            logger.info("ImageDataset: Main data RAM caching (image, CB, fix_coords) ENABLED.")
        else:
            logger.info("ImageDataset: Main data RAM caching DISABLED (due to LMDB or config).")

        if self.cache_fixation_data:
            logger.info("ImageDataset: Populating fixation coordinate RAM cache (_xs_cache, _ys_cache)...")
            self._xs_cache = {}
            self._ys_cache = {}
            for x_coord, y_coord, n_idx in zip(self.fixations.x_int, self.fixations.y_int, tqdm(self.fixations.n, desc="Caching fixation coords")):
                self._xs_cache.setdefault(n_idx, []).append(x_coord)
                self._ys_cache.setdefault(n_idx, []).append(y_coord)
            for key_idx in list(self._xs_cache.keys()): self._xs_cache[key_idx] = np.array(self._xs_cache[key_idx], dtype=int)
            for key_idx in list(self._ys_cache.keys()): self._ys_cache[key_idx] = np.array(self._ys_cache[key_idx], dtype=int)
            logger.info("ImageDataset: Fixation coordinate RAM cache populated.")
        else:
            self._xs_cache, self._ys_cache = None, None
            logger.info("ImageDataset: Fixation coordinate RAM caching DISABLED.")

    # ... (rest of ImageDataset: get_shapes, _get_stimulus_as_array, _get_image_data, __getitem__, __len__) ...
    # (These methods from your previous full data.py are fine)
    def get_shapes(self):
        return list(self.stimuli.sizes) # [(H, W), ...] or [(H, W, C), ...]

    def _get_stimulus_as_array(self, n_idx: int) -> np.ndarray:
        """Helper to get stimulus n_idx as a NumPy array from various Stimuli types."""
        stimulus_data = self.stimuli.stimuli[n_idx] # From pysaliency.Stimuli
        if isinstance(stimulus_data, Image.Image):
            return np.array(stimulus_data)
        elif isinstance(stimulus_data, np.ndarray):
            return stimulus_data
        elif isinstance(stimulus_data, (str, Path)): # If FileStimuli returns path
             return np.array(Image.open(stimulus_data))
        else:
            raise TypeError(f"Unexpected stimulus data type for stimuli[{n_idx}]: {type(stimulus_data)}")


    def _get_image_data(self, n_idx: int) -> tuple[np.ndarray, np.ndarray | None]:
        if self.lmdb_env:
            return _get_image_data_from_lmdb(self.lmdb_env, n_idx) # Returns (image_chw_f32, cb_hw_f32)
        else:
            image_arr_orig_channels = self._get_stimulus_as_array(n_idx)
            centerbias_prediction = None
            if self.centerbias_model:
                centerbias_prediction = self.centerbias_model.log_density(image_arr_orig_channels)
            else: # Create a dummy zero centerbias if model is None
                h, w = image_arr_orig_channels.shape[:2]
                centerbias_prediction = np.zeros((h,w), dtype=np.float32)


            image_rgb = ensure_color_image(image_arr_orig_channels).astype(np.float32)
            image_chw = image_rgb.transpose(2, 0, 1) # HWC to CHW
            return image_chw, centerbias_prediction

    def __getitem__(self, key: int):
        if not self.cached or key not in self._cache:
            image_chw_f32, centerbias_hw_f32 = self._get_image_data(key)
            
            xs_arr, ys_arr = None, None
            if self.cache_fixation_data and self._xs_cache is not None:
                if key in self._xs_cache:
                    if self.cached:
                        xs_arr = self._xs_cache.pop(key, np.array([], dtype=int))
                        ys_arr = self._ys_cache.pop(key, np.array([], dtype=int))
                    else:
                        xs_arr = self._xs_cache.get(key, np.array([], dtype=int))
                        ys_arr = self._ys_cache.get(key, np.array([], dtype=int))
            
            if xs_arr is None: 
                inds = self.fixations.n == key
                xs_arr = np.array(self.fixations.x_int[inds], dtype=int)
                ys_arr = np.array(self.fixations.y_int[inds], dtype=int)

            data = {
                "image": image_chw_f32, 
                "x": xs_arr,
                "y": ys_arr,
                "centerbias": centerbias_hw_f32.astype(np.float32), 
            }
            data['weight'] = 1.0 if self.average == 'image' else float(len(xs_arr)) if len(xs_arr) > 0 else 1.0

            if self.cached: self._cache[key] = data
        else:
            data = self._cache[key]

        item_to_return = dict(data) if self.transform is not None else data
        if self.transform:
            return self.transform(item_to_return)
        return item_to_return

    def __len__(self):
        return len(self.stimuli)



class ImageDatasetWithSegmentation(ImageDataset):
    def __init__(self, *args, # Args for parent ImageDataset (stimuli, fixations, etc.)
                 segmentation_mask_dir: str | Path | None = None, # DIRECT path to folder of mask files FOR THIS DATASET
                 segmentation_mask_format: str = "png",
                 segmentation_mask_fixed_memmap_file: str | Path | None = None, # Bank specific to this dataset
                 segmentation_mask_variable_payload_file: str | Path | None = None, # Bank specific to this dataset
                 segmentation_mask_variable_header_file: str | Path | None = None, # Bank specific to this dataset
                 segmentation_mask_bank_dtype: str = "uint8",
                 **kwargs): # Kwargs for parent ImageDataset

        super().__init__(*args, **kwargs) # Call parent init

        # This is the DIRECT path to the folder containing mask image files for THIS specific dataset.
        # (e.g., "./masks/salicon/train_masks/" or "./masks/mit1003/all_masks/")
        self.individual_mask_files_dir = Path(segmentation_mask_dir).resolve() if segmentation_mask_dir else None
        
        self.segmentation_mask_format = segmentation_mask_format.lower()
        self._mask_bank_dtype_np = np.dtype(segmentation_mask_bank_dtype)

        self.mask_fixed_mmap_bank = None
        self.mask_fixed_mmap_shape = None
        self.mask_variable_payload_mmap = None
        self.mask_variable_header_data = None

        # --- Initialize Fixed-Size Memmap Bank ---
        if segmentation_mask_fixed_memmap_file:
            fixed_path = Path(segmentation_mask_fixed_memmap_file).resolve()
            if fixed_path.exists():
                try:
                    mmap = np.memmap(fixed_path, mode='r', dtype=self._mask_bank_dtype_np)
                    if mmap.ndim == 3 and mmap.shape[0] == len(self.stimuli):
                        self.mask_fixed_mmap_bank = mmap
                        self.mask_fixed_mmap_shape = mmap.shape
                        logger.info(f"Successfully memory-mapped FIXED-SIZE masks from: {fixed_path} shape {mmap.shape}")
                    else:
                        logger.error(f"Fixed mask memmap shape {mmap.shape} incompatible with stimuli count {len(self.stimuli)} from {fixed_path}. Disabling.")
                except Exception as e: logger.error(f"Error memory-mapping fixed-size masks from {fixed_path}: {e}")
            else: logger.warning(f"Fixed-size mask memmap file not found: {fixed_path}")

        # --- Initialize Variable-Size Memmap Bank ---
        if not self.mask_fixed_mmap_bank and segmentation_mask_variable_payload_file and segmentation_mask_variable_header_file:
            payload_path = Path(segmentation_mask_variable_payload_file).resolve()
            header_path = Path(segmentation_mask_variable_header_file).resolve()
            if payload_path.exists() and header_path.exists():
                try:
                    header = np.load(header_path)
                    if header.ndim == 2 and header.shape[1] == 3 and header.shape[0] == len(self.stimuli):
                        self.mask_variable_header_data = header
                        self.mask_variable_payload_mmap = np.memmap(payload_path, dtype=self._mask_bank_dtype_np, mode='r')
                        logger.info(f"Successfully set up VARIABLE-SIZE mask loading: Payload '{payload_path}', Header '{header_path}'")
                    else:
                        logger.error(f"Variable mask header {header_path} shape {header.shape} incompatible with stimuli count {len(self.stimuli)}. Disabling.")
                except Exception as e: logger.error(f"Error setting up variable-size mask loading (Header: {header_path}, Payload: {payload_path}): {e}")
            else: logger.warning(f"Variable-size mask payload or header file not found: {payload_path}, {header_path}")

        # --- Determine Final Mask Loading Strategy ---
        self.mask_loading_strategy = "dummy"
        if self.mask_fixed_mmap_bank is not None:
            self.mask_loading_strategy = "fixed_bank"
            logger.info(f"Using FIXED-SIZE memmap bank for segmentation masks.")
        elif self.mask_variable_payload_mmap is not None and self.mask_variable_header_data is not None:
            self.mask_loading_strategy = "variable_bank"
            logger.info(f"Using VARIABLE-SIZE memmap bank for segmentation masks.")
        elif self.individual_mask_files_dir: # Check the direct path provided
            logger.info(f"Attempting individual files. Checking direct mask path: '{self.individual_mask_files_dir}'")
            if self.individual_mask_files_dir.exists() and self.individual_mask_files_dir.is_dir():
                self.mask_loading_strategy = "individual_files"
                logger.info(f"SUCCESS: Using individual mask files from {self.individual_mask_files_dir}.")
                try:
                    first_few = list(self.individual_mask_files_dir.glob(f'*.{self.segmentation_mask_format}'))[:5]
                    if first_few: logger.info(f"  Found mask files, e.g.: {[f.name for f in first_few]}")
                    else: logger.warning(f"  Directory {self.individual_mask_files_dir} exists, but no masks ('*.{self.segmentation_mask_format}') found directly within it.")
                except Exception as e_glob: logger.warning(f"  Error listing mask files in {self.individual_mask_files_dir}: {e_glob}")
            else:
                logger.warning(f"FAILURE: Directory for individual masks ('{self.individual_mask_files_dir}') does not exist or is not a directory.")
        
        if self.mask_loading_strategy == "dummy":
             logger.warning("ImageDatasetWithSegmentation: No valid mask source found/configured. Dummy (all-zero) masks will be used.")


    def __getitem__(self, key: int):
            """
            Retrieves a complete data sample. This version uses the public .filenames
            attribute from the FileStimuli object to reliably get the stimulus ID.
            """
            # 1. Get base data from parent. This loads the resized image from the cache path.
            try:
                original_item_from_parent = super().__getitem__(key)
            except IndexError:
                logger.error(f"Index {key} out of bounds in parent ImageDataset. Skipping.")
                return {
                    'image': torch.zeros(3, 224, 224, dtype=torch.float),
                    'centerbias': torch.zeros(224, 224, dtype=torch.float),
                    'fixations': torch.zeros(0, 2, dtype=torch.float),
                    'weight': torch.tensor(1.0, dtype=torch.float32),
                    'segmentation_mask': torch.zeros(224, 224, dtype=torch.long)
                }

            # 2. Prepare the final dictionary, converting numpy arrays to tensors.
            final_item = {}
            for k_orig, v_orig in original_item_from_parent.items():
                if isinstance(v_orig, np.ndarray):
                    final_item[k_orig] = torch.from_numpy(v_orig.copy())
                else:
                    final_item[k_orig] = v_orig

            if 'weight' in final_item and not isinstance(final_item['weight'], torch.Tensor):
                final_item['weight'] = torch.tensor(final_item['weight'], dtype=torch.float32)
            elif 'weight' not in final_item:
                final_item['weight'] = torch.tensor(1.0, dtype=torch.float32)

            # 3. Load the corresponding segmentation mask.
            mask_tensor_to_add = None
            stimulus_id_str = f"key:{key}" # Default for logging

            if self.mask_loading_strategy == "individual_files":
                try:
                    # --- THIS IS THE FINAL FIX ---
                    # Since self.stimuli is a pysaliency.FileStimuli object, it has a
                    # .filenames attribute, which is a list of the full paths to the cached images.
                    cached_image_filename = self.stimuli.filenames[key]
                    # We get the base name without the extension (e.g., 'i12345').
                    stimulus_id_str = Path(cached_image_filename).stem
                    # -----------------------------

                    mask_fname = f"{stimulus_id_str}.{self.segmentation_mask_format}"
                    mask_path_full = self.individual_mask_files_dir / mask_fname

                    if mask_path_full.exists():
                        if self.segmentation_mask_format == "png":
                            with Image.open(mask_path_full) as mask_pil:
                                if mask_pil.mode not in ['L', 'I', 'P', '1']:
                                    mask_pil = mask_pil.convert('L')
                                mask_arr = np.array(mask_pil)
                        elif self.segmentation_mask_format == "npy":
                            mask_arr = np.load(mask_path_full)
                        else:
                            raise ValueError(f"Unsupported mask format: {self.segmentation_mask_format}")
                        mask_tensor_to_add = torch.from_numpy(mask_arr.astype(np.int64))
                    else:
                        logger.debug(f"Individual mask file not found: {mask_path_full}")
                except (AttributeError, IndexError) as e:
                    logger.error(f"Failed to get mask filename for key {key}: {e}. Ensure self.stimuli is a FileStimuli object. Using dummy mask.")
                except Exception as e:
                    logger.error(f"General error loading mask for key {key}: {e}. Using dummy mask.")

            # ... (bank loading logic remains the same) ...
            elif self.mask_loading_strategy in ["fixed_bank", "variable_bank"]:
                try:
                    # This part is fine, as it uses the integer key
                    if self.mask_loading_strategy == "fixed_bank":
                        mask_np_view = self.mask_fixed_mmap_bank[key]
                        mask_tensor_to_add = torch.from_numpy(np.array(mask_np_view).astype(np.int64))
                    else: # variable_bank
                        offset, H, W = self.mask_variable_header_data[key]
                        if H > 0 and W > 0:
                            num_elements = H * W
                            mask_1d_view = self.mask_variable_payload_mmap[offset : offset + num_elements]
                            mask_tensor_to_add = torch.from_numpy(np.array(mask_1d_view).reshape(H, W).astype(np.int64))
                except Exception as e:
                    logger.error(f"Error reading from mask bank for key {key}: {e}. Using dummy mask.")

            # 4. Fallback to dummy mask.
            if mask_tensor_to_add is None:
                if self.mask_loading_strategy != 'dummy':
                    logger.error(f"Mask for {stimulus_id_str} failed to load via '{self.mask_loading_strategy}' strategy. Creating dummy mask.")
                
                if 'image' in final_item and isinstance(final_item['image'], torch.Tensor):
                    img_h, img_w = final_item['image'].shape[1], final_item['image'].shape[2]
                else:
                    img_h, img_w = 224, 224
                    logger.critical(f"Image tensor missing for key {key} when creating dummy mask!")
                
                mask_tensor_to_add = torch.zeros((img_h, img_w), dtype=torch.long)

            # 5. Add the final mask and return.
            final_item['segmentation_mask'] = mask_tensor_to_add
            return final_item


class FixationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        stimuli, fixations,
        centerbias_model=None,
        lmdb_path=None,
        transform=None,
        included_fixations=-2,
        allow_missing_fixations=False,
        average='fixation',
        cache_image_data=False,
    ):
        self.stimuli = stimuli
        self.fixations = fixations
        self.centerbias_model = centerbias_model
        self.lmdb_path = lmdb_path

        if lmdb_path is not None:
            _export_dataset_to_lmdb(stimuli, centerbias_model, lmdb_path)
            self.lmdb_env = lmdb.open(lmdb_path, subdir=os.path.isdir(lmdb_path),
                readonly=True, lock=False,
                readahead=False, meminit=False
            )
            cache_image_data=False
        else:
            self.lmdb_env = None

        self.transform = transform
        self.average = average

        self._shapes = None

        if isinstance(included_fixations, int):
            if included_fixations < 0:
                included_fixations = [-1 - i for i in range(-included_fixations)]
            else:
                raise NotImplementedError()

        self.included_fixations = included_fixations
        self.allow_missing_fixations = allow_missing_fixations
        self.fixation_counts = Counter(fixations.n)

        self.cache_image_data = cache_image_data

        if self.cache_image_data:
            self.image_data_cache = {}

            print("Populating image cache")
            for n in tqdm(range(len(self.stimuli))):
                self.image_data_cache[n] = self._get_image_data(n)

    def get_shapes(self):
        if self._shapes is None:
            shapes = list(self.stimuli.sizes)
            self._shapes = [shapes[n] for n in self.fixations.n]

        return self._shapes

    def _get_image_data(self, n):
        if self.lmdb_path:
            return _get_image_data_from_lmdb(self.lmdb_env, n)
        image = np.array(self.stimuli.stimuli[n])
        centerbias_prediction = self.centerbias_model.log_density(image)

        image = ensure_color_image(image).astype(np.float32)
        image = image.transpose(2, 0, 1)

        return image, centerbias_prediction

    def __getitem__(self, key):
        n = self.fixations.n[key]

        if self.cache_image_data:
            image, centerbias_prediction = self.image_data_cache[n]
        else:
            image, centerbias_prediction = self._get_image_data(n)

        centerbias_prediction = centerbias_prediction.astype(np.float32)

        x_hist = remove_trailing_nans(self.fixations.x_hist[key])
        y_hist = remove_trailing_nans(self.fixations.y_hist[key])

        if self.allow_missing_fixations:
            _x_hist = []
            _y_hist = []
            for fixation_index in self.included_fixations:
                if fixation_index < -len(x_hist):
                    _x_hist.append(np.nan)
                    _y_hist.append(np.nan)
                else:
                    _x_hist.append(x_hist[fixation_index])
                    _y_hist.append(y_hist[fixation_index])
            x_hist = np.array(_x_hist)
            y_hist = np.array(_y_hist)
        else:
            print("Not missing")
            x_hist = x_hist[self.included_fixations]
            y_hist = y_hist[self.included_fixations]

        data = {
            "image": image,
            "x": np.array([self.fixations.x_int[key]], dtype=int),
            "y": np.array([self.fixations.y_int[key]], dtype=int),
            "x_hist": x_hist,
            "y_hist": y_hist,
            "centerbias": centerbias_prediction,
        }

        if self.average == 'image':
            data['weight'] = 1.0 / self.fixation_counts[n]
        else:
            data['weight'] = 1.0

        if self.transform is not None:
            return self.transform(data)

        return data

    def __len__(self):
        return len(self.fixations)


class FixationMaskTransform(object):
    def __init__(self, sparse=True):
        super().__init__()
        self.sparse = sparse

    def __call__(self, item):
        shape = torch.Size([item['image'].shape[1], item['image'].shape[2]])
        x = item.pop('x')
        y = item.pop('y')

        # inds, values = x_y_to_sparse_indices(x, y)
        inds = np.array([y, x])
        values = np.ones(len(y), dtype=int)

        mask = torch.sparse_coo_tensor(torch.tensor(inds), torch.tensor(values), shape, dtype=torch.int)
        mask = mask.coalesce()
        # sparse tensors don't work with workers...
        if not self.sparse:
            mask = mask.to_dense()

        item['fixation_mask'] = mask

        return item


class ImageDatasetSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, batch_size=1, ratio_used=1.0, shuffle=True):
        # It's good practice to call super().__init__ if inheriting from torch.utils.data.Sampler,
        # though often not strictly necessary if you override __iter__ and __len__.
        # super().__init__(data_source) # Optional, can be added

        self.data_source = data_source # Keep a reference if needed by __iter__ or __len__ later
                                      # In your current code, it's only used for get_shapes initially.
        self.batch_size = batch_size
        self.ratio_used = ratio_used
        self.shuffle = shuffle

        # --- MODIFICATION FOR Subset compatibility ---
        if isinstance(data_source, torch.utils.data.Subset):
            # If data_source is a Subset, get shapes from the underlying dataset
            underlying_dataset = data_source.dataset
            if hasattr(underlying_dataset, 'get_shapes') and callable(getattr(underlying_dataset, 'get_shapes')):
                shapes = underlying_dataset.get_shapes()
                # The indices k below will be relative to the underlying_dataset.
                # We need to ensure that the indices we store in shape_indices are
                # the indices *within the Subset*.
                # So, we first get shapes for ALL items in the underlying dataset,
                # then filter to only include items that are part of the current Subset.

                # Get all shapes from the original full dataset
                all_original_shapes = underlying_dataset.get_shapes()

                # `data_source.indices` gives the indices in the `underlying_dataset`
                # that form this `Subset`.
                subset_original_indices = data_source.indices

                # Get shapes only for the items in the current subset
                # And map original indices to subset_indices (0 to len(subset)-1)
                # shapes_for_subset will be a list of shapes, corresponding to items 0..len(subset)-1
                shapes_for_subset = [all_original_shapes[original_idx] for original_idx in subset_original_indices]

                # The indices `k` used below to populate `shape_indices` should now be
                # relative to the subset (0 to len(subset)-1)
                # So, `shapes` should be `shapes_for_subset`
                shapes_to_process = shapes_for_subset
                num_items_to_process = len(shapes_to_process)

            else:
                raise AttributeError(f"The underlying dataset of type {type(underlying_dataset)} "
                                     "for the Subset does not have a 'get_shapes' method.")
        elif hasattr(data_source, 'get_shapes') and callable(getattr(data_source, 'get_shapes')):
            # This is the original path, data_source is the full dataset
            shapes_to_process = data_source.get_shapes()
            num_items_to_process = len(shapes_to_process) # or len(data_source)
        else:
            raise AttributeError(f"Data source of type {type(data_source)} "
                                 "does not have a 'get_shapes' method.")
        # --- END MODIFICATION ---

        unique_shapes = sorted(set(shapes_to_process))
        shape_indices = [[] for _ in unique_shapes] # Corrected: use _ if shape var not used

        # The indices 'k' here must be local to the data_source being processed
        # (either the full dataset or the subset).
        # `enumerate` will give k from 0 to len(shapes_to_process)-1, which is correct.
        for k, shape_val in enumerate(shapes_to_process): # Renamed shape to shape_val to avoid conflict
            shape_indices[unique_shapes.index(shape_val)].append(k)

        if self.shuffle:
            for indices_list in shape_indices: # Renamed indices to indices_list
                random.shuffle(indices_list)

        # The `chunked` function will take these local indices (0 to len(data_source)-1)
        # and form batches. These indices are what DataLoader expects for data_source.__getitem__().
        self.batches = sum([chunked(indices_list, size=self.batch_size) for indices_list in shape_indices], [])
        # Important: After sum(), self.batches is a list of batches, where each batch is a list of indices.
        # Example: [[0, 1, 2], [10, 11, 12], [5, 6], ...]

    def __iter__(self):
        # The batches in self.batches already contain the correct indices (relative to
        # data_source, which is the Subset in DDP case or full dataset otherwise).
        # So, this part of the code doesn't need to change.

        batch_indices_to_iterate = list(range(len(self.batches))) # Create a list of batch indices

        if self.shuffle:
            # Shuffle the order of the batches themselves
            random.shuffle(batch_indices_to_iterate) # Use random.shuffle for list of indices

        if self.ratio_used < 1.0:
            num_batches_to_use = int(self.ratio_used * len(batch_indices_to_iterate))
            batch_indices_to_iterate = batch_indices_to_iterate[:num_batches_to_use]

        # Yield one batch (which is a list of item indices) at a time
        for i in batch_indices_to_iterate:
            yield self.batches[i]

    def __len__(self):
        # This returns the number of batches.
        return int(self.ratio_used * len(self.batches))


def _export_dataset_to_lmdb(stimuli: pysaliency.FileStimuli, centerbias_model: pysaliency.Model, lmdb_path, write_frequency=100):
    """
    Checks if a valid LMDB database exists; if not, generates it. KISS version.

    1. Checks if lmdb_path is a directory containing a valid LMDB with the correct item count.
    2. If valid, prints a message and returns immediately.
    3. If not valid (doesn't exist, not a dir, wrong count, corrupted),
       it removes the existing path (if any) and generates the database from scratch.
    """
    lmdb_path_str = str(os.path.expanduser(lmdb_path))
    expected_count = len(stimuli)
    is_valid_and_complete = False # Flag to track if we should skip generation

    # --- Phase 1: Check for existing valid LMDB ---
    if os.path.isdir(lmdb_path_str):
        db_check = None
        try:
            # Attempt to open read-only to check metadata
            db_check = lmdb.open(lmdb_path_str, subdir=True, readonly=True, lock=False)
            with db_check.begin() as txn_check:
                len_bytes = txn_check.get(b'__len__')
                if len_bytes is not None:
                    try:
                        retrieved_count = int(len_bytes.decode('utf-8'))
                        if retrieved_count == expected_count:
                            # Valid and complete! Set the flag.
                            is_valid_and_complete = True
                        else:
                            print(f"LMDB check: Count mismatch ({retrieved_count} vs {expected_count}). Needs regeneration.")
                    except (ValueError, UnicodeDecodeError):
                         print(f"LMDB check: Error decoding count. Needs regeneration.")
                else:
                     print("LMDB check: '__len__' key missing. Needs regeneration.")

            db_check.close() # Close the read-only handle

        except lmdb.Error as e:
            # Error opening/reading the existing DB
            print(f"LMDB check: Could not open/read existing DB ({e}). Needs regeneration.")
            if db_check: # Ensure handle is closed even on error
                 try: db_check.close()
                 except lmdb.Error: pass # Ignore close errors if already bad
        # No need for further exception handling here, is_valid_and_complete remains False

    # --- Phase 2: Return if valid, otherwise proceed to generate ---
    if is_valid_and_complete:
        print(f"Valid LMDB found at {lmdb_path_str} with {expected_count} items. Skipping generation.")
        return # <<< EXIT EARLY

    # --- Phase 3: Cleanup and Generation (Only runs if not returned early) ---
    print(f"Regenerating LMDB at {lmdb_path_str}")

    # --- Cleanup existing path ---
    if os.path.lexists(lmdb_path_str): # Use lexists to handle broken symlinks too
        print(f"Removing existing path: {lmdb_path_str}")
        try:
            if os.path.isdir(lmdb_path_str) and not os.path.islink(lmdb_path_str):
                shutil.rmtree(lmdb_path_str)
                # recreate the directory so LMDB can open it
                os.makedirs(lmdb_path_str, exist_ok=True)

            else: # It's a file or a symlink
                 os.remove(lmdb_path_str)
        except OSError as e:
            print(f"Warning: Failed to remove existing path {lmdb_path_str}: {e}. Generation might fail.")
            # Depending on the error, you might want to raise it here
            # raise e

    # --- Generate the database ---
    os.makedirs(lmdb_path_str, exist_ok=True)
    db_write = None
    try:
        # Open for writing. subdir=True creates the directory if needed.
        db_write = lmdb.open(lmdb_path_str, subdir=True,
                           map_size = 32 * 1024**3, readonly=False, # Adjust map_size if needed
                           meminit=False, map_async=True)

        actual_written_count = 0
        txn_write = db_write.begin(write=True)
        try: # Wrap write loop for transaction handling
            for idx, stimulus in enumerate(tqdm(stimuli, desc="Writing LMDB entries")):
                key = u'{}'.format(idx).encode('ascii')
                stimulus_filename = stimuli.filenames[idx] # Assumes FileStimuli
                centerbias = centerbias_model.log_density(stimulus) # Calculate centerbias
                encoded_data = _encode_filestimulus_item(stimulus_filename, centerbias)

                if encoded_data is not None: # Handle potential encoding errors
                    txn_write.put(key, encoded_data)
                    actual_written_count += 1

                # Commit periodically
                if actual_written_count > 0 and actual_written_count % write_frequency == 0:
                    txn_write.commit()
                    txn_write = db_write.begin(write=True)

            # Commit the final transaction
            txn_write.commit()
            txn_write = None # Mark transaction as finished

            # Write metadata AFTER successful data commit
            print(f"Writing metadata: count = {actual_written_count}")
            with db_write.begin(write=True) as txn_meta:
                 len_bytes_write = str(actual_written_count).encode('utf-8')
                 txn_meta.put(b'__len__', len_bytes_write)

        finally: # Ensure transaction is aborted if loop failed
            if txn_write:
                print("Aborting write transaction due to error.")
                txn_write.abort()

    except Exception as e:
         print(f"!!! ERROR during LMDB generation: {e}")
         # Optionally re-raise
         # raise e
    finally: # Ensure database is closed
        if db_write:
            print("Closing database.")
            db_write.close()


def _encode_filestimulus_item(filename, centerbias):
    with open(filename, 'rb') as f:
        image_bytes = f.read()

    buffer = io.BytesIO()
    pickle.dump({'image': image_bytes, 'centerbias': centerbias}, buffer)
    buffer.seek(0)
    return buffer.read()


def _get_image_data_from_lmdb(lmdb_env, n):
    key = '{}'.format(n).encode('ascii')
    with lmdb_env.begin(write=False) as txn:
        byteflow = txn.get(key)
    data = pickle.loads(byteflow)
    buffer = io.BytesIO(data['image'])
    buffer.seek(0)
    image = np.array(Image.open(buffer).convert('RGB'))
    centerbias_prediction = data['centerbias']
    image = image.transpose(2, 0, 1)

    return image, centerbias_prediction



def prepare_spatial_dataset(
    stimuli,
    fixations,
    centerbias,
    batch_size: int,        # This is the PER-GPU batch size
    num_workers: int,
    is_distributed: bool,
    is_master: bool,
    device: torch.device,
    path: Path | None = None,
    current_epoch: int = 0, # Added: current epoch for DistributedSampler
):
    """
    Prepares the DataLoader for spatial (image-based) saliency prediction,
    handling DDP with shape-aware batching correctly.
    """
    # ----------  LMDB bookkeeping  ----------
    lmdb_path_str = str(path) if path else None
    if lmdb_path_str:
        if is_master:
            try:
                if path: path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Using LMDB cache for spatial dataset at: {lmdb_path_str}")
            except OSError as e:
                logger.error(f"Failed to create LMDB directory {path}: {e}")
                lmdb_path_str = None # Fallback
        if is_distributed:
            # dist.barrier(device_ids=[device.index] if device.type == 'cuda' else None)
            dist.barrier() # Simplified barrier call

    # ----------  build Full Dataset ----------
    full_dataset = ImageDataset(
        stimuli=stimuli,
        fixations=fixations,
        centerbias_model=centerbias,
        transform=FixationMaskTransform(sparse=False), # Use dense masks
        average="image", # Average fixations per image
        lmdb_path=lmdb_path_str,
    )

    loader: torch.utils.data.DataLoader # Type hint

    # ----------  choose samplers and build DataLoader ----------
    if is_distributed:
        # DDP Path:
        # 1. Create a DistributedSampler for the full dataset.
        distributed_sampler = torch.utils.data.DistributedSampler(
            full_dataset,
            shuffle=True,
            drop_last=True
        )
        # CRITICAL: Set the epoch for the DistributedSampler
        distributed_sampler.set_epoch(current_epoch)

        # 2. Create a Subset of the full_dataset for the current rank.
        rank_subset_indices = list(iter(distributed_sampler))
        rank_dataset_subset = torch.utils.data.Subset(full_dataset, rank_subset_indices)

        # 3. Use ImageDatasetSampler on this rank-specific subset.
        shape_aware_batch_sampler = ImageDatasetSampler(
            data_source=rank_dataset_subset, # Operates on the subset for this rank
            batch_size=batch_size,           # Per-GPU batch size
            shuffle=True                     # Shuffle items within the subset before batching by shape
        )
        # Optional: if ImageDatasetSampler has its own epoch setting
        # if hasattr(shape_aware_batch_sampler, 'set_epoch'):
        #     shape_aware_batch_sampler.set_epoch(current_epoch)

        # 4. Create the DataLoader for DDP
        loader = torch.utils.data.DataLoader(
            rank_dataset_subset,
            batch_sampler=shape_aware_batch_sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )
    else:
        # Non-Distributed Path (Single GPU or CPU):
        # Use ImageDatasetSampler on the full dataset directly.
        batch_sampler_single_gpu = ImageDatasetSampler(
            full_dataset,
            batch_size=batch_size,
            shuffle=True # Assuming this enables shuffling within ImageDatasetSampler
        )
        # Optional: if ImageDatasetSampler has its own epoch setting
        # if hasattr(batch_sampler_single_gpu, 'set_epoch'):
        #    batch_sampler_single_gpu.set_epoch(current_epoch)

        loader = torch.utils.data.DataLoader(
            full_dataset,
            batch_sampler=batch_sampler_single_gpu,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )

    return loader


def prepare_scanpath_dataset(
    stimuli,
    fixations,
    centerbias,
    batch_size: int,        # This is the PER-GPU batch size
    num_workers: int,
    is_distributed: bool,
    is_master: bool,
    device: torch.device,   # device is kept for consistency, used by barrier
    path: Path | None = None,
    current_epoch: int = 0, # Added: current epoch for DistributedSampler
    logger = None,       # Added: logger for error messages
):
    """
    Prepares the DataLoader for scanpath prediction (fixation-based),
    handling DDP with shape-aware batching correctly.
    """
    lmdb_path_str = str(path) if path else None
    if lmdb_path_str:
        if is_master:
            try:
                if path: path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Using LMDB cache for scanpath dataset at: {lmdb_path_str}")
            except OSError as e:
                logger.error(f"Failed to create LMDB directory {path}: {e}")
                lmdb_path_str = None # Fallback
        if is_distributed:
            # The original barrier call didn't use device_ids if device was CPU.
            # dist.barrier(device_ids=[device.index] if device.type == 'cuda' else None)
            dist.barrier() # Simplified barrier call as in your provided code

    # 1. Create the full dataset instance
    full_dataset = FixationDataset(
        stimuli=stimuli,
        fixations=fixations,
        centerbias_model=centerbias,
        included_fixations=[-1, -2, -3, -4],
        allow_missing_fixations=True,
        transform=FixationMaskTransform(sparse=False),
        average="image",
        lmdb_path=lmdb_path_str,
    )

    loader: torch.utils.data.DataLoader # Type hint for clarity

    if is_distributed:
        # DDP Path:
        # 2. Create a DistributedSampler for the full dataset.
        #    drop_last=True is important for consistent batch counts across GPUs when using Subset.
        distributed_sampler = torch.utils.data.DistributedSampler(
            full_dataset,
            shuffle=True,       # Shuffles the global list of indices
            drop_last=True      # Ensures all GPUs get the same number of samples
        )
        # CRITICAL: Set the epoch for the DistributedSampler for proper shuffling each epoch
        distributed_sampler.set_epoch(current_epoch)

        # 3. Create a Subset of the full_dataset for the current rank.
        #    iter(distributed_sampler) yields the indices assigned to the current rank.
        rank_subset_indices = list(iter(distributed_sampler))
        rank_dataset_subset = torch.utils.data.Subset(full_dataset, rank_subset_indices)

        # 4. Use ImageDatasetSampler on this rank-specific subset.
        #    It will perform shape-aware batching ONLY on the data for this GPU.
        #    Set shuffle=True if ImageDatasetSampler should shuffle items within the
        #    rank's subset before forming shape-compatible batches.
        shape_aware_batch_sampler = ImageDatasetSampler(
            data_source=rank_dataset_subset, # Operates on the subset for this rank
            batch_size=batch_size,           # Per-GPU batch size
            shuffle=True                     # Shuffle items within the subset before batching by shape
        )
        # Optional: if ImageDatasetSampler itself has epoch-aware internal shuffling
        # if hasattr(shape_aware_batch_sampler, 'set_epoch'):
        #     shape_aware_batch_sampler.set_epoch(current_epoch)

        # 5. Create the DataLoader for DDP
        loader = torch.utils.data.DataLoader(
            rank_dataset_subset,        # Use the subset for this rank
            batch_sampler=shape_aware_batch_sampler, # Custom batch sampler
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
            # drop_last is effectively handled by ImageDatasetSampler and DistributedSampler(drop_last=True)
        )
    else:
        # Non-Distributed Path (Single GPU or CPU):
        # Use ImageDatasetSampler on the full dataset directly, as in your "before" code.
        # This assumes ImageDatasetSampler's shuffle=True handles epoch shuffling correctly
        # for the non-DDP case, or that you call set_epoch on it if available.
        batch_sampler_single_gpu = ImageDatasetSampler(
            full_dataset,
            batch_size=batch_size,
            shuffle=True  # Assuming this enables shuffling within ImageDatasetSampler
        )
        # Optional: if ImageDatasetSampler itself has epoch-aware internal shuffling
        # if hasattr(batch_sampler_single_gpu, 'set_epoch'):
        #    batch_sampler_single_gpu.set_epoch(current_epoch)

        loader = torch.utils.data.DataLoader(
            full_dataset,
            batch_sampler=batch_sampler_single_gpu,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )

    return loader

# -----------------------------------------------------------------------------
# MIT Data Conversion Functions (Copied from original with DDP modifications)
# -----------------------------------------------------------------------------
# ── shared size helper ─────────────────────────────────────────────
def target_size(h_orig, w_orig):          # returns (width, height)
    if h_orig < w_orig:                   # landscape
        return 1024, 768
    else:                                 # portrait or square
        return 768, 1024



def convert_stimulus(input_image):
    h, w = input_image.shape[:2]
    new_w, new_h = target_size(h, w)      # <- use helper
    return np.array(
        Image.fromarray(input_image).resize((new_w, new_h), Image.BILINEAR)
    )


def convert_stimuli(stimuli, new_location: Path, is_master: bool, is_distributed: bool, device: torch.device, logger = None):
    """ Converts all stimuli in a FileStimuli object to standard sizes and saves them. """
    assert isinstance(stimuli, pysaliency.FileStimuli)
    new_stimuli_location = new_location / 'stimuli'
    if is_master:
        try:
            new_stimuli_location.mkdir(parents=True, exist_ok=True)
            logger.info(f"Converting stimuli resolution and saving to {new_stimuli_location}...")
        except OSError as e:
            logger.critical(f"Failed to create stimuli conversion directory {new_stimuli_location}: {e}")
            return None

    new_filenames = []
    filenames_iterable = tqdm(stimuli.filenames, desc="Converting Stimuli", disable=not is_master)
    conversion_errors = 0

    for filename in filenames_iterable:
        try:
            stimulus = imread(filename)
            if stimulus.ndim == 2:
                stimulus = np.stack([stimulus]*3, axis=-1)
            elif stimulus.shape[2] == 1:
                stimulus = np.concatenate([stimulus]*3, axis=-1)
            elif stimulus.shape[2] == 4:
                stimulus = stimulus[:,:,:3]
            if stimulus.shape[2] != 3:
                if is_master: logger.warning(f"Skipping stimulus {filename} with unexpected shape {stimulus.shape}")
                conversion_errors += 1
                continue
            new_stimulus = convert_stimulus(stimulus)
            basename = os.path.basename(filename)
            new_filename = new_stimuli_location / basename
            if is_master:
                # Store original shape in the JSON metadata if using FileStimuli's save method
                # Or, handle metadata storage separately if needed.
                if new_stimulus.shape != stimulus.shape or not new_filename.exists():
                    try: imwrite(new_filename, new_stimulus)
                    except Exception as e: logger.error(f"Failed to write {new_filename}: {e}"); conversion_errors +=1; continue
                elif not new_filename.exists():
                    try: shutil.copy(filename, new_filename)
                    except Exception as e: logger.error(f"Failed to copy {filename} to {new_filename}: {e}"); conversion_errors += 1; continue
            new_filenames.append(str(new_filename.resolve()))   # <- absolute
        except Exception as read_err:
            if is_master: logger.exception(f"Failed to read or process stimulus {filename}")
            conversion_errors += 1
            continue

    if is_master and conversion_errors > 0:
        logger.warning(f"Encountered {conversion_errors} errors during stimuli conversion.")

    # synchronize if distributed
    if is_distributed:
        dist.barrier()

    # persist cache and metadata
    if is_master:
        with open(new_location / "stimuli.pkl", "wb") as f:
            cpickle.dump(pysaliency.FileStimuli(new_filenames), f, protocol=pickle.HIGHEST_PROTOCOL)
        meta = {
            "filenames": new_filenames,
            "shapes":    [list(s) for s in pysaliency.FileStimuli(new_filenames).shapes],
        }
        with open(new_location / "stimuli.json", "w") as f:
            json.dump(meta, f, indent=2)

    # return a FileStimuli made from the *string* paths
    return pysaliency.FileStimuli(new_filenames)




def convert_fixation_trains(stimuli: pysaliency.FileStimuli,
                            fixations: pysaliency.FixationTrains,
                            is_master: bool,
                            logger) -> pysaliency.ScanpathFixations:
    """
    Rescale MIT‑1003 FixationTrains to 1024×768 / 768×1024 and return
    a pysaliency.ScanpathFixations object **with correct history & time‑stamps**.
    """

    # -------------- 1. pre‑compute per‑stimulus scale factors --------------
    orig_h = np.array([h for h, w, _ in stimuli.shapes])
    orig_w = np.array([w for h, w, _ in stimuli.shapes])
    tgt_w  = np.where(orig_h < orig_w, 1024, 768)
    tgt_h  = np.where(orig_h < orig_w,  768,1024)
    sx     = tgt_w / orig_w
    sy     = tgt_h / orig_h

    # -------------- 2. walk every scan‑path -------------------------------
    new_xs, new_ys, new_ts  = [], [], []
    new_x_hist, new_y_hist, new_dur = [], [], []

    skipped = 0
    for xs, ys, ts, ns in zip(fixations.train_xs,
                              fixations.train_ys,
                              fixations.train_ts,
                              fixations.train_ns):

        f_sx, f_sy = sx[ns], sy[ns]                       # scalar
        xs_ = np.clip(xs * f_sx, 0, tgt_w[ns]-1e-6)
        ys_ = np.clip(ys * f_sy, 0, tgt_h[ns]-1e-6)

        new_xs.append(xs_)
        new_ys.append(ys_)

        # ---------- timestamps ----------
        if ts is None or len(ts)==0:
            new_ts.append(np.full_like(xs_, np.nan))
        else:
            new_ts.append(np.asarray(ts, dtype=float))

        # ---------- scan‑path history ----------
        # history arrays are (n_fixations, 4, 2) in MIT1003; scale both coordinates
        if hasattr(fixations, "x_hist"):
            hist_x = np.clip(fixations.x_hist[skipped] * f_sx, 0, tgt_w[ns]-1e-6)
            hist_y = np.clip(fixations.y_hist[skipped] * f_sy, 0, tgt_h[ns]-1e-6)
            new_x_hist.append(hist_x)
            new_y_hist.append(hist_y)

        if hasattr(fixations, "durations"):
            new_dur.append(fixations.durations[skipped].copy())

        skipped += 1

    # -------------- 3. assemble pysaliency.Scanpaths -----------------------
    scanpaths = pysaliency.Scanpaths(xs=new_xs, ys=new_ys, ts=new_ts,
                                     n=fixations.train_ns,
                                     scanpath_attributes={"subject": fixations.train_subjects})

    # add optional arrays if we have them
    if new_x_hist:  scanpaths.x_hist = new_x_hist
    if new_y_hist:  scanpaths.y_hist = new_y_hist
    if new_dur:     scanpaths.durations = new_dur

    # -------------- 4. sanity‑check & return ------------------------------
    xx = np.concatenate(new_xs)
    yy = np.concatenate(new_ys)
    assert (xx >= 0).all() and (yy >= 0).all(), "negative coords after scaling!"
    assert (xx < tgt_w.max()).all() and (yy < tgt_h.max()).all(), "coords out of bounds!"

    if is_master:
        logger.info(f"✅ converted {len(scanpaths)} scan‑paths "
                     f"({xx.size:,} fixations total)")

    return pysaliency.ScanpathFixations(scanpaths=scanpaths)
