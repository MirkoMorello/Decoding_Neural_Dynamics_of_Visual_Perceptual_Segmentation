from collections import Counter
import io
import os
import pickle
import random
import shutil
import numpy

from boltons.iterutils import chunked
import lmdb
import numpy as np
from PIL import Image
import pysaliency
from pysaliency.datasets import create_subset
from pysaliency.utils import remove_trailing_nans
import torch
from tqdm import tqdm


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
        stimuli,
        fixations,
        centerbias_model=None,
        lmdb_path=None,
        transform=None,
        cached=None,
        average='fixation'
    ):
        self.stimuli = stimuli
        self.fixations = fixations
        self.centerbias_model = centerbias_model
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.average = average

        # cache only short dataset
        if cached is None:
            cached = len(self.stimuli) < 100

        cache_fixation_data = cached

        if lmdb_path is not None:
            _export_dataset_to_lmdb(stimuli, centerbias_model, lmdb_path)
            self.lmdb_env = lmdb.open(lmdb_path, subdir=os.path.isdir(lmdb_path),
                readonly=True, lock=False,
                readahead=False, meminit=False
            )
            cached = False
            cache_fixation_data = True
        else:
            self.lmdb_env = None

        self.cached = cached
        if cached:
            self._cache = {}
        self.cache_fixation_data = cache_fixation_data
        if cache_fixation_data:
            print("Populating fixations cache")
            self._xs_cache = {}
            self._ys_cache = {}

            for x, y, n in zip(self.fixations.x_int, self.fixations.y_int, tqdm(self.fixations.n)):
                self._xs_cache.setdefault(n, []).append(x)
                self._ys_cache.setdefault(n, []).append(y)

            for key in list(self._xs_cache):
                self._xs_cache[key] = np.array(self._xs_cache[key], dtype=int)
            for key in list(self._ys_cache):
                self._ys_cache[key] = np.array(self._ys_cache[key], dtype=int)

    def get_shapes(self):
        return list(self.stimuli.sizes)

    def _get_image_data(self, n):
        if self.lmdb_env:
            image, centerbias_prediction = _get_image_data_from_lmdb(self.lmdb_env, n)
        else:
            image = np.array(self.stimuli.stimuli[n])
            centerbias_prediction = self.centerbias_model.log_density(image)

            image = ensure_color_image(image).astype(np.float32)
            image = image.transpose(2, 0, 1)

        return image, centerbias_prediction

    def __getitem__(self, key):
        if not self.cached or key not in self._cache:

            image, centerbias_prediction = self._get_image_data(key)
            centerbias_prediction = centerbias_prediction.astype(np.float32)

            if self.cache_fixation_data and self.cached:
                xs = self._xs_cache.pop(key)
                ys = self._ys_cache.pop(key)
            elif self.cache_fixation_data and not self.cached:
                xs = self._xs_cache[key]
                ys = self._ys_cache[key]
            else:
                inds = self.fixations.n == key
                xs = np.array(self.fixations.x_int[inds], dtype=int)
                ys = np.array(self.fixations.y_int[inds], dtype=int)

            data = {
                "image": image,
                "x": xs,
                "y": ys,
                "centerbias": centerbias_prediction,
            }

            if self.average == 'image':
                data['weight'] = 1.0
            else:
                data['weight'] = float(len(xs))

            if self.cached:
                self._cache[key] = data
        else:
            data = self._cache[key]

        if self.transform is not None:
            return self.transform(dict(data))

        return data

    def __len__(self):
        return len(self.stimuli)


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
        self.ratio_used = ratio_used
        self.shuffle = shuffle

        shapes = data_source.get_shapes()
        unique_shapes = sorted(set(shapes))

        shape_indices = [[] for shape in unique_shapes]

        for k, shape in enumerate(shapes):
            shape_indices[unique_shapes.index(shape)].append(k)

        if self.shuffle:
            for indices in shape_indices:
                random.shuffle(indices)

        self.batches = sum([chunked(indices, size=batch_size) for indices in shape_indices], [])

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(len(self.batches))
        else:
            indices = range(len(self.batches))

        if self.ratio_used < 1.0:
            indices = indices[:int(self.ratio_used * len(indices))]

        return iter(self.batches[i] for i in indices)

    def __len__(self):
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
                           map_size = 8 * 1024**3, readonly=False, # Adjust map_size if needed
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