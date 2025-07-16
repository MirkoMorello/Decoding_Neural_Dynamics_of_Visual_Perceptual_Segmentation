# data.py
from collections import Counter
import io
import os
import pickle
import random
import shutil
from pathlib import Path
from boltons.iterutils import chunked
from typing   import Optional, Tuple, List
import lmdb
import numpy as np
from PIL import Image, ImageFile
import pysaliency
from pysaliency.utils import remove_trailing_nans
import torch
from tqdm import tqdm
import torch.distributed as dist
from imageio.v3 import imread, imwrite
from torch.utils.data import Sampler, RandomSampler, BatchSampler
import numpy as np
import torch
from boltons.fileutils import atomic_save
import cloudpickle as cpickle
import json
import logging

logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None        # avoid DecompressionBomb warnings
ImageFile.LOAD_TRUNCATED_IMAGES = True   # skip truncated-file errors

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


def decode_and_prepare_image(path_or_bytes: str | Path | bytes) -> np.ndarray:
    """
    Carica un'immagine, la converte in RGB e la restituisce come array uint8 [0, 255]
    nel formato (C, H, W).

    Returns:
        np.ndarray: L'immagine come array NumPy (C, H, W) di tipo uint8.
    """
    if isinstance(path_or_bytes, (str, Path)):
        img_pil = Image.open(path_or_bytes)
    else:
        img_pil = Image.open(io.BytesIO(path_or_bytes))

    img_pil = img_pil.convert("RGB")
    img_arr_uint8 = np.array(img_pil)
    return img_arr_uint8.transpose(2, 0, 1)

def safe_precompute_sizes(stimuli: pysaliency.FileStimuli) -> None:
    """
    Touch every stimulus once so that `stimuli.sizes` is fully populated.
    If some files are unreadable we log them, and—when the installed
    pysaliency exposes `keep_only`—we drop them to avoid crashes later.
    On older versions we just warn.
    """
    bad_items = []
    for i, fname in enumerate(stimuli.filenames):
        try:
            _ = stimuli.sizes[i]          # forces a lazy read
        except Exception as e:
            print(f"[WARN] could not read {fname}: {e}")
            bad_items.append(i)

    if not bad_items:
        return                            # nothing to do

    keep = [j for j in range(len(stimuli)) if j not in bad_items]

    if hasattr(stimuli, "keep_only"):
        stimuli.keep_only(keep)           # newest pysaliency
        print(f"[INFO] removed {len(bad_items)} corrupted stimuli")
    else:
        print(
            f"[INFO] pysaliency < 0.3: cannot drop {len(bad_items)} bad images "
            "(no keep_only); they will stay in the list – be aware that any "
            "access to them (e.g. during batching) will still raise."
        )

class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        stimuli,
        fixations,
        centerbias_model=None,
        *,                 # keyword-only after this
        lmdb_path=None,
        transform=None,
        cached=None,
        average="fixation",
    ):
        logger.info("⏳ ImageDataset initialising…")

        self.stimuli          = stimuli
        self.fixations        = fixations
        self.centerbias_model = centerbias_model
        self.transform        = transform
        self.average          = average

        # ----- LMDB bookkeeping (lazy-open later in __getitem__) -------
        self.lmdb_path = (
            Path(lmdb_path).expanduser().resolve() if lmdb_path else None
        )
        self.lmdb_env  = None            # each worker opens its own handle
        if self.lmdb_path is not None:
            logger.info(f"  • LMDB path set to {self.lmdb_path} (lazy open)")
        else:
            logger.info("  • LMDB disabled (direct file reads)")

        # ----- decide RAM caching of full samples ----------------------
        self.cached = bool(cached) if cached is not None else (
            self.lmdb_path is None and len(self.stimuli) < 100
        )
        if self.cached:
            self._cache = {}
            logger.info(f"  • RAM caching ENABLED for {len(self.stimuli)} items")
        else:
            logger.info("  • RAM caching DISABLED")

        # ----- pre-cache fixation coordinates --------------------------
        logger.info("  • Building fixation-coordinate caches …")
        self._xs_cache, self._ys_cache = {}, {}
        for x, y, n in zip(self.fixations.x_int,
                           self.fixations.y_int,
                           self.fixations.n):
            self._xs_cache.setdefault(n, []).append(x)
            self._ys_cache.setdefault(n, []).append(y)
        for k in list(self._xs_cache):
            self._xs_cache[k] = np.asarray(self._xs_cache[k], dtype=int)
            self._ys_cache[k] = np.asarray(self._ys_cache[k], dtype=int)
        logger.debug("    fixation-coord caches built")

        logger.info("✅ ImageDataset ready")

    def get_shapes(self):
        return list(self.stimuli.sizes) # [(H, W), ...] or [(H, W, C), ...]

    def __len__(self):
        return len(self.stimuli)

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
                return _get_image_data_from_lmdb(self.lmdb_env, n_idx)
            else:
                stimulus_path = self.stimuli.filenames[n_idx]
                image_chw_uint8 = decode_and_prepare_image(stimulus_path)
                centerbias_prediction = None
                if self.centerbias_model:
                    image_for_cb = np.array(Image.open(stimulus_path).convert('RGB'))
                    centerbias_prediction = self.centerbias_model.log_density(image_for_cb)
                else: 
                    h, w = image_chw_uint8.shape[1:]
                    centerbias_prediction = np.zeros((h, w), dtype=np.float32)

                return image_chw_uint8, centerbias_prediction.astype(np.float32)
            
    def _open_lmdb_env(self) -> lmdb.Environment:
        return lmdb.open(
            str(self.lmdb_path), subdir=True,
            readonly=True, lock=False,
            readahead=False, meminit=False
        )

    def __getitem__(self, idx: int):
        # 1.  ensure LMDB handle for THIS worker
        if self.lmdb_env is None and self.lmdb_path is not None:
            logger.info(f"[PID {os.getpid()}] opening LMDB → {self.lmdb_path}")
            self.lmdb_env = self._open_lmdb_env()

        # 2.  cache fast-path
        if self.cached and idx in self._cache:
            logger.debug(f"cache hit item {idx}")
            sample = self._cache[idx]
        else:
            # 3-a. image + center-bias
            image, cb = self._get_image_data(idx)

            # 3-b. fixation coordinates
            xs = self._xs_cache.get(idx, np.empty(0, dtype=int))
            ys = self._ys_cache.get(idx, np.empty(0, dtype=int))

            # 3-c. assemble
            sample = dict(
                image      = image,
                x          = xs,
                y          = ys,
                centerbias = cb.astype(np.float32),
                weight     = 1.0 if self.average == "image" else float(len(xs)),
            )
            if self.cached:
                self._cache[idx] = sample
                logger.debug(f"stored item {idx} in RAM cache")

        # 4.  optional transform
        if self.transform is not None:
            sample = self.transform(sample)
        return sample



class ImageDatasetWithSegmentation(ImageDataset):
    def __init__(
        self,
        stimuli,
        fixations,
        centerbias_model=None,
        *,                       # ------- keyword-only after this
        lmdb_path=None,
        transform=None,
        cached=None,
        average="fixation",
        # ------------- segmentation-specific arguments -----------------
        segmentation_mask_dir: str | Path | None = None,
        segmentation_mask_format: str = "png",
        segmentation_mask_fixed_memmap_file: str | Path | None = None,
        segmentation_mask_variable_payload_file: str | Path | None = None,
        segmentation_mask_variable_header_file: str | Path | None = None,
        segmentation_mask_bank_dtype: str = "uint8",
    ):
        logger.info("⏳ ImageDatasetWithSegmentation initialising…")

        # ------------------------------------------------------------------
        # 1.  *Base* dataset initialisation  (no segmentation kwargs)
        # ------------------------------------------------------------------
        super().__init__(
            stimuli,
            fixations,
            centerbias_model=centerbias_model,
            lmdb_path=lmdb_path,
            transform=transform,
            cached=cached,
            average=average,
        )
        logger.debug("✓ Base ImageDataset init completed")

        # ------------------------------------------------------------------
        # 2.  Book-keeping for mask handling
        # ------------------------------------------------------------------
        self.segmentation_mask_format  = segmentation_mask_format.lower()
        self._mask_bank_dtype_np       = np.dtype(segmentation_mask_bank_dtype)

        # paths
        self.individual_mask_files_dir = (
            Path(segmentation_mask_dir).expanduser().resolve()
            if segmentation_mask_dir else None
        )

        # holders for the various strategies
        self.mask_fixed_mmap_bank       = None
        self.mask_variable_payload_mmap = None
        self.mask_variable_header_data  = None
        self.mask_loading_strategy      = "dummy"   # provisional

        # ------------------------------------------------------------------
        # 3.  Try fixed-size mem-map bank
        # ------------------------------------------------------------------
        if segmentation_mask_fixed_memmap_file:
            fpath = Path(segmentation_mask_fixed_memmap_file).expanduser().resolve()
            if fpath.exists():
                try:
                    mm = np.memmap(fpath, mode="r", dtype=self._mask_bank_dtype_np)
                    if mm.ndim == 3 and mm.shape[0] == len(self.stimuli):
                        self.mask_fixed_mmap_bank = mm
                        self.mask_loading_strategy = "fixed_bank"
                        logger.info(f"✓ Fixed-size mask bank found: {fpath}  shape={mm.shape}")
                    else:
                        logger.error(
                            "❌ Fixed-bank shape mismatch "
                            f"(got {mm.shape}, expected n={len(self.stimuli)}) – ignored"
                        )
                except Exception as e:
                    logger.exception(f"❌ Could not mem-map fixed bank {fpath}: {e}")
            else:
                logger.warning(f"⚠ fixed-size mem-map file not found: {fpath}")

        # ------------------------------------------------------------------
        # 4.  Try variable-size mem-map bank
        # ------------------------------------------------------------------
        if (self.mask_loading_strategy == "dummy"
            and segmentation_mask_variable_payload_file
            and segmentation_mask_variable_header_file):

            payload = Path(segmentation_mask_variable_payload_file).expanduser().resolve()
            header  = Path(segmentation_mask_variable_header_file ).expanduser().resolve()

            if payload.exists() and header.exists():
                try:
                    hdr = np.load(header)
                    if hdr.ndim == 2 and hdr.shape[1] == 3 and hdr.shape[0] == len(self.stimuli):
                        self.mask_variable_header_data  = hdr
                        self.mask_variable_payload_mmap = np.memmap(
                            payload, dtype=self._mask_bank_dtype_np, mode="r"
                        )
                        self.mask_loading_strategy = "variable_bank"
                        logger.info(f"✓ Variable-size mask bank enabled (payload={payload})")
                    else:
                        logger.error(
                            f"❌ Variable bank header shape {hdr.shape} "
                            f"incompatible with stimuli count {len(self.stimuli)} – ignored"
                        )
                except Exception as e:
                    logger.exception(f"❌ Error loading variable bank: {e}")
            else:
                logger.warning(f"⚠ Variable bank files missing: {payload} / {header}")

        # ------------------------------------------------------------------
        # 5.  Fallback to individual mask files
        # ------------------------------------------------------------------
        if (self.mask_loading_strategy == "dummy"
            and self.individual_mask_files_dir
            and self.individual_mask_files_dir.is_dir()):
            self.mask_loading_strategy = "individual_files"
            try:
                sample_files = list(
                    self.individual_mask_files_dir.glob(f"*.{self.segmentation_mask_format}")
                )[:5]
                logger.info(
                    " Using individual mask files "
                    f"in {self.individual_mask_files_dir} "
                    f"(example: {[f.name for f in sample_files]})"
                )
            except Exception as e:
                logger.warning(f"⚠ Could not list mask dir {self.individual_mask_files_dir}: {e}")

        # ------------------------------------------------------------------
        # 6.  Final status
        # ------------------------------------------------------------------
        if self.mask_loading_strategy == "dummy":
            logger.warning("🚧 No valid mask source found — dummy zero masks will be used.")

        logger.info(f"✅ ImageDatasetWithSegmentation ready (strategy = {self.mask_loading_strategy})")


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
                    cached_image_filename = self.stimuli.filenames[key]
                    stimulus_id_str = Path(cached_image_filename).stem
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
    def __len__(self):
        return super().__len__()

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

    def _get_image_data(self, n_idx: int) -> tuple[np.ndarray, np.ndarray | None]:
        if self.lmdb_env:
            return _get_image_data_from_lmdb(self.lmdb_env, n_idx)
        else:

            stimulus_path = self.stimuli.filenames[n_idx]
            image_chw_f32 = decode_and_prepare_image(stimulus_path)
            centerbias_prediction = None
            if self.centerbias_model:
                image_for_cb = np.array(Image.open(stimulus_path).convert('RGB'))
                centerbias_prediction = self.centerbias_model.log_density(image_for_cb)
            else:
                h, w = image_chw_f32.shape[1:]
                centerbias_prediction = np.zeros((h, w), dtype=np.float32)

            return image_chw_f32, centerbias_prediction.astype(np.float32)

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
    
    



class FixationDatasetWithSegmentation(FixationDataset):
    """
    Extends FixationDataset to add support for loading segmentation masks.

    This class inherits all scanpath functionality from its parent and adds
    the mask-loading logic, mirroring the design of ImageDatasetWithSegmentation.
    """
    def __init__(
        self,
        *args, # Pass positional args (stimuli, fixations, etc.) to the parent
        segmentation_mask_dir: str | Path,
        segmentation_mask_format: str = "png",
        **kwargs # Pass keyword args to the parent
    ):
        # Call the parent's __init__ method with all the original arguments
        super().__init__(*args, **kwargs)

        # Now, set up the new segmentation-specific attributes
        self.individual_mask_files_dir = Path(segmentation_mask_dir).resolve()
        self.segmentation_mask_format = segmentation_mask_format.lower()
        
        self.use_segmentation_masks = False
        if self.individual_mask_files_dir.exists() and self.individual_mask_files_dir.is_dir():
            self.use_segmentation_masks = True
            logger.info(f"FixationDatasetWithSegmentation: Mask loading ENABLED from {self.individual_mask_files_dir}.")
        else:
            # Raise an error because if this class is used, masks are expected.
            raise FileNotFoundError(f"Segmentation mask directory not found or is not a directory: {self.individual_mask_files_dir}")

    def __getitem__(self, key: int):
        # 1. Get the complete data dictionary from the parent class.
        # This will include 'image', 'x_hist', 'y_hist', 'centerbias', etc.
        data = super().__getitem__(key)

        # The parent may have already applied a transform. We need to handle this.
        # The ideal place to add the mask is *before* the transform.
        # A small refactor of the parent could help, but we can work around it.
        # Let's assume the parent doesn't return tensors yet, or we convert back.
        # Your current parent `FixationDataset` returns a dict of numpy arrays
        # before the transform, so this is fine.

        # 2. Load and add the segmentation mask
        mask_tensor_to_add = None
        stimulus_idx = self.fixations.n[key]

        try:
            stimulus_filename_stem = Path(self.stimuli.filenames[stimulus_idx]).stem
            mask_fname = f"{stimulus_filename_stem}.{self.segmentation_mask_format}"
            mask_path_full = self.individual_mask_files_dir / mask_fname

            if mask_path_full.exists():
                if self.segmentation_mask_format == "png":
                    mask_pil = Image.open(mask_path_full).convert('L')
                    mask_arr = np.array(mask_pil)
                elif self.segmentation_mask_format == "npy":
                    mask_arr = np.load(mask_path_full)
                # No need for torch.from_numpy here, the final transform will handle it
                mask_tensor_to_add = mask_arr.astype(np.int64)
            else:
                logger.warning(f"Mask file not found for stimulus {stimulus_filename_stem}, creating dummy mask. Path: {mask_path_full}")

        except Exception as e:
            logger.error(f"Error loading mask for stimulus index {stimulus_idx}: {e}")

        if mask_tensor_to_add is None:
            # Create a dummy numpy array if loading failed
            img_h, img_w = data['image'].shape[1], data['image'].shape[2]
            mask_tensor_to_add = np.zeros((img_h, img_w), dtype=np.int64)
        
        data['segmentation_mask'] = mask_tensor_to_add

        # 3. The parent class's __getitem__ already handles the transform at the end,
        # so we don't need to call it again. Our job is just to add the mask
        # to the dictionary before it's returned.
        # Wait, the parent already calls the transform. This is a problem.
        # We need to re-structure slightly.

        # Let's try a better approach that doesn't break the transform logic.
        return self._get_item_with_mask(key)


    def _create_scanpath_tensor(self, x_hist, y_hist, durations,
                                target_shape_hw, device=torch.device('cpu')):

        B, num_hist_fix = 1, x_hist.shape[0]     # dataset still returns one sample
        H, W = target_shape_hw

        xs_grid = torch.linspace(0, W - 1, W, device=device)
        ys_grid = torch.linspace(0, H - 1, H, device=device)
        grid_y, grid_x = torch.meshgrid(ys_grid, xs_grid, indexing='ij')

        grid_x = grid_x.expand(num_hist_fix, H, W)
        grid_y = grid_y.expand(num_hist_fix, H, W)

        dx = grid_x - torch.from_numpy(x_hist).view(num_hist_fix, 1, 1).to(device)
        dy = grid_y - torch.from_numpy(y_hist).view(num_hist_fix, 1, 1).to(device)
        dist = torch.sqrt(dx ** 2 + dy ** 2)     # <── new third channel
        # (duration is *not* used by the current head)

        stacked = torch.stack([dx, dy, dist], dim=1)   # (N, 3, H, W)
        return stacked.view(num_hist_fix * 3, H, W).float()
    
    def _load_or_dummy_mask(
        self,
        stimulus_idx: int,
        img_shape_hw: tuple[int, int],   # (H, W) – for fallback shape
    ) -> np.ndarray:
        """
        Returns the segmentation mask for `stimulus_idx` as a **numpy int64
        array** (H, W).  If the file is missing or corrupt, a zero mask with the
        requested image size is returned.
        """
        if not self.use_segmentation_masks:
            return np.zeros(img_shape_hw, dtype=np.int64)

        try:
            # e.g.  "train_23.jpg" → "train_23.png"
            stem = Path(self.stimuli.filenames[stimulus_idx]).stem
            mask_path = self.individual_mask_files_dir / f"{stem}.{self.segmentation_mask_format}"

            if mask_path.exists():
                if self.segmentation_mask_format == "png":
                    mask = np.array(Image.open(mask_path).convert("L"), dtype=np.int64)
                elif self.segmentation_mask_format == "npy":
                    mask = np.load(mask_path).astype(np.int64)
                else:
                    raise ValueError(f"Unsupported mask format '{self.segmentation_mask_format}'")
                return mask

            logger.warning(f"Mask file not found: {mask_path}")

        except Exception as e:
            logger.error(f"Error reading mask for stimulus {stimulus_idx}: {e}")

        # Fallback – all zeros, same size as image
        return np.zeros(img_shape_hw, dtype=np.int64)

    def _get_item_with_mask(self, key: int):
        stim_idx = self.fixations.n[key]

        # ---------- A. image & center-bias ----------
        image, centerbias = self._get_image_data(stim_idx)         # (3,H,W) / (H,W)

        # ---------- B. raw histories (length = 4) ----------
        x_hist_raw = remove_trailing_nans(self.fixations.x_hist[key])
        y_hist_raw = remove_trailing_nans(self.fixations.y_hist[key])
        dur_raw    = np.nan_to_num(remove_trailing_nans(self.fixations.t_hist[key]),
                                   nan=0.0)

        x_hist, y_hist, durations = [], [], []
        for i in self.included_fixations:          # [-1,-2,-3,-4]
            if i < -len(x_hist_raw):
                x_hist.append(np.nan);  y_hist.append(np.nan); durations.append(0.0)
            else:
                x_hist.append(x_hist_raw[i]); y_hist.append(y_hist_raw[i]); durations.append(dur_raw[i])

        x_hist = np.asarray(x_hist, dtype=np.float32)
        y_hist = np.asarray(y_hist, dtype=np.float32)

        # ---------- C. segmentation mask ----------
        mask_np = self._load_or_dummy_mask(stim_idx, image.shape[1:])   # helper below

        # ---------- D. package ----------
        sample = dict(
            image            = image,                                   # (3,H,W)  float32
            x                = np.asarray([self.fixations.x_int[key]], dtype=np.int32),
            y                = np.asarray([self.fixations.y_int[key]], dtype=np.int32),
            x_hist           = x_hist,                                  # 4-vector  float32
            y_hist           = y_hist,                                  # 4-vector  float32
            centerbias       = centerbias,                              # (H,W)    float32
            segmentation_mask= mask_np,                                 # (H,W)    int64
            weight           = (1.0 / self.fixation_counts[stim_idx]
                                if self.average == "image" else 1.0)
        )

        if self.transform is not None:
            sample = self.transform(sample)        # keeps x_hist / y_hist untouched
        return sample
    
    
    
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


class ImageDatasetSampler(torch.utils.data.Sampler[list[int]]):
    def __init__(
        self,
        data_source: torch.utils.data.Dataset,
        batch_size: int = 1,
        ratio_used: float = 1.0,
        shuffle: bool = True,
    ):
        self.data_source = data_source
        self.batch_size  = batch_size
        self.ratio_used  = ratio_used
        self.shuffle     = shuffle

        rng = random.Random()                       # independent RNG

        # -------- 1. resolve shapes for *local* indices ---------------
        if isinstance(data_source, torch.utils.data.Subset):
            parent_shapes = data_source.dataset.get_shapes()
            shapes = [parent_shapes[i] for i in data_source.indices]
        else:
            shapes = data_source.get_shapes()

        # group local indices by shape (H,W)
        shape_buckets: dict[tuple[int, int], list[int]] = {}
        for local_idx, shp in enumerate(shapes):
            shape_buckets.setdefault(tuple(shp[:2]), []).append(local_idx)

        # -------- 2. build batches, never mixing shapes ---------------
        self.batches: list[list[int]] = []
        for idx_list in shape_buckets.values():
            if shuffle:
                rng.shuffle(idx_list)

            for i in range(0, len(idx_list), batch_size):
                chunk = idx_list[i : i + batch_size]
                if len(chunk) == batch_size or (
                    len(chunk) and ratio_used == 1.0
                ):
                    self.batches.append(chunk)

        if shuffle:
            rng.shuffle(self.batches)

        # -------- 3. optionally down-sample epoch --------------------
        if ratio_used < 1.0:
            keep = int(len(self.batches) * ratio_used)
            self.batches = self.batches[:keep]

    # Sampler interface ----------------------------------------------
    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)


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
    """Codifica l'immagine PRE-ELABORATA e il centerbias."""
    # Usa la nostra nuova funzione canonica per ottenere l'array corretto
    image_chw_f32 = decode_and_prepare_image(filename)

    # Serializza l'array NumPy e il centerbias
    # Pickle è semplice e funziona bene qui.
    payload = pickle.dumps({
        'image': image_chw_f32,
        'centerbias': centerbias.astype(np.float32) # Assicura che anche il CB sia float32
    })
    return payload


def _get_image_data_from_lmdb(lmdb_env, n):
    """Legge l'immagine e il centerbias PRE-ELABORATI da LMDB."""
    key = '{}'.format(n).encode('ascii')
    with lmdb_env.begin(write=False) as txn:
        byteflow = txn.get(key)
    
    data = pickle.loads(byteflow)
    
    # Non c'è più bisogno di decodificare o convertire, è già pronto
    image_chw_f32 = data['image']
    centerbias_hw_f32 = data['centerbias']
    
    return image_chw_f32, centerbias_hw_f32



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


def _target_size(h_orig: int, w_orig: int) -> Tuple[int, int]:
    if h_orig < w_orig:               # landscape
        return 1024, 768              #  (new_w, new_h)
    else:                             # portrait or square
        return 768, 1024

def _resize_np(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    new_w, new_h = _target_size(h, w)
    return np.array(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))

def convert_stimuli(
    stimuli: pysaliency.FileStimuli,
    new_location: Path,
    is_master: bool,
    is_distributed: bool,
    device: torch.device,
    logger: Optional[logging.Logger] = None,
    *,
    save_format: str = "jpeg",        # "jpeg"  or  "png"
) -> pysaliency.FileStimuli:
    """
    • Resizes every image in `stimuli` to 1024×768 / 768×1024.
    • Saves them to  `<new_location>/stimuli/`  with the chosen format.
    • Returns a new `FileStimuli` with **absolute paths** to the resized files.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    save_format = save_format.lower()
    if save_format not in ("jpeg", "jpg", "png"):
        raise ValueError("save_format must be 'jpeg' or 'png'")

    new_stimuli_location = new_location / "stimuli"

    # ------------------------------------------------------------------ #
    # 1. Create target directory (only on master)                        #
    # ------------------------------------------------------------------ #
    if is_master:
        new_stimuli_location.mkdir(parents=True, exist_ok=True)
        logger.info(f"Converting stimuli and saving to {new_stimuli_location}")

    # make sure other ranks wait until directory exists
    if is_distributed:
        torch.distributed.barrier()

    # ------------------------------------------------------------------ #
    # 2. Iterate over original images                                    #
    # ------------------------------------------------------------------ #
    new_filenames: List[str] = []
    io_errors = 0

    for fname_in in tqdm(stimuli.filenames,
                         desc="Converting stimuli",
                         disable=not is_master):

        try:
            img_np = np.asarray(Image.open(fname_in).convert("RGB"))
        except Exception as e:
            if is_master:
                logger.warning(f"[READ-FAIL] {fname_in} – {e}")
            io_errors += 1
            continue

        img_resized = _resize_np(img_np)

        # -----------------------  output name  ------------------------- #
        stem = Path(fname_in).stem
        suffix = ".jpg" if save_format in ("jpeg", "jpg") else ".png"
        fname_out = new_stimuli_location / f"{stem}{suffix}"

        # -----------------------  write file (master only)  ------------ #
        if is_master and not fname_out.exists():
            try:
                if save_format in ("jpeg", "jpg"):
                    Image.fromarray(img_resized).save(
                        fname_out, "JPEG", quality=93, optimize=True)
                else:                   # png
                    Image.fromarray(img_resized).save(fname_out, "PNG")
            except Exception as e:
                logger.warning(f"[WRITE-FAIL] {fname_out} – {e}")
                io_errors += 1
                continue

        new_filenames.append(str(fname_out.resolve()))

    # gather counts across ranks
    if is_distributed:
        io_errors_tensor = torch.tensor(io_errors, device=device)
        torch.distributed.all_reduce(io_errors_tensor, op=torch.distributed.ReduceOp.SUM)
        io_errors = int(io_errors_tensor.item())

    if is_master and io_errors:
        logger.warning(f"⚠️  {io_errors} images failed during read/write")

    # ------------------------------------------------------------------ #
    # 3. Save cache files (master)                                       #
    # ------------------------------------------------------------------ #
    if is_master:
        with atomic_save(str(new_location / "stimuli.pkl"),
                 text_mode=False, overwrite_part=True) as f:
            cpickle.dump(pysaliency.FileStimuli(new_filenames), f)

        meta = {
            "filenames": new_filenames,
            "shapes":    [list(s) for s in pysaliency.FileStimuli(new_filenames).shapes],
            "save_format": save_format
        }
        with open(new_location / "stimuli.json", "w") as f:
            json.dump(meta, f, indent=2)

    # make sure every rank sees the cache files
    if is_distributed:
        torch.distributed.barrier()

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