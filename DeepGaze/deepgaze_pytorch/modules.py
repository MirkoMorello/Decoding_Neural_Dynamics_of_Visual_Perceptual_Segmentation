# modules.py
import functools
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import GaussianFilterNd


def encode_scanpath_features(
        x_hist,                   # tensor (B, N)  or  (N,)
        y_hist,                   # tensor (B, N)  or  (N,)
        size,                     # (H, W) image / feature-map size
        *,                        # force kwargs after this point
        durations=None,           # tensor (B, N) or (N,) — optional
        device=None,
        include_x=True,
        include_y=True,
        include_duration=False):
    """
    Build a (B, C, H, W) tensor with ∆x, ∆y, distance (and optionally duration)
    for every fixation in the history.

    C =   3  if include_duration == False   (dx, dy, dist)
        or 4  if include_duration == True   (dx, dy, dist, dur)

    Parameters
    ----------
    x_hist, y_hist : torch.Tensor or np.ndarray
        History of fixations normalised to the same resolution as `size`.
        Shape either (N,)  or  (B, N).  Will be cast to float32.
    size : tuple
        (height, width) of the target map.
    durations : torch.Tensor or np.ndarray, optional
        Dwell times for the same fixations, shape (N,) or (B, N).
    """

    # ───── basic checks ──────────────────────────────────────────────
    assert include_x and include_y, \
        "encode_scanpath_features currently always expects include_x and include_y = True"
    if include_duration and durations is None:
        raise ValueError("include_duration=True but `durations` is None")

    H, W = int(size[0]), int(size[1])

    # ───── ensure torch tensors, 2-D shape (B, N) ────────────────────
    def _to_2d_tensor(arr, name):
        if not torch.is_tensor(arr):
            arr = torch.as_tensor(arr)
        if arr.dim() == 1:                # (N,)  →  (1, N)
            arr = arr.unsqueeze(0)
        elif arr.dim() != 2:
            raise ValueError(f"{name} must be 1-D or 2-D, got shape {arr.shape}")
        return arr.float()

    x_hist = _to_2d_tensor(x_hist, "x_hist").to(device)
    y_hist = _to_2d_tensor(y_hist, "y_hist").to(device)
    if include_duration:
        durations = _to_2d_tensor(durations, "durations").to(device)

    B, N = x_hist.shape

    # ───── coordinate grid ───────────────────────────────────────────
    ys = torch.linspace(0, H - 1, H, dtype=torch.float32, device=device)
    xs = torch.linspace(0, W - 1, W, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')           # (H, W)

    grid_x = grid_x.unsqueeze(0).unsqueeze(0)                        # (1,1,H,W)
    grid_y = grid_y.unsqueeze(0).unsqueeze(0)

    # repeat for B and N
    grid_x = grid_x.expand(B, N, -1, -1)                             # (B,N,H,W)
    grid_y = grid_y.expand(B, N, -1, -1)

    # ───── broadcast fixation coordinates ────────────────────────────
    x_hist = x_hist.view(B, N, 1, 1)
    y_hist = y_hist.view(B, N, 1, 1)

    dx = grid_x - x_hist                                             # (B,N,H,W)
    dy = grid_y - y_hist
    dist = torch.sqrt(dx ** 2 + dy ** 2)

    feats = [dx, dy, dist]
    if include_duration:
        dur = durations.view(B, N, 1, 1).expand_as(dx)
        feats.append(dur)

    feat_tensor = torch.cat(feats, dim=1)                            # (B, N*C_i, H, W)
    feat_tensor = feat_tensor.view(B, N * len(feats), H, W)          # collapse N into channel

    return feat_tensor

class FeatureExtractor(torch.nn.Module):
    def __init__(self, features, targets):
        super().__init__()
        self.features = features
        self.targets = targets
        #print("Targets are {}".format(targets))
        self.outputs = {}

        for target in targets:
            layer = dict([*self.features.named_modules()])[target]
            layer.register_forward_hook(self.save_outputs_hook(target))

    def save_outputs_hook(self, layer_id: str):
        def fn(_, __, output):
            self.outputs[layer_id] = output.clone()
        return fn

    def forward(self, x):

        self.outputs.clear()
        self.features(x)
        return [self.outputs[target] for target in self.targets]


def upscale(tensor, size):
    tensor_size = torch.tensor(tensor.shape[2:]).type(torch.float32)
    target_size = torch.tensor(size).type(torch.float32)
    factors = torch.ceil(target_size / tensor_size)
    factor = torch.max(factors).type(torch.int64).to(tensor.device)
    assert factor >= 1

    tensor = torch.repeat_interleave(tensor, factor, dim=2)
    tensor = torch.repeat_interleave(tensor, factor, dim=3)

    tensor = tensor[:, :, :size[0], :size[1]]

    return tensor


class Finalizer(nn.Module):
    """Transforms a readout into a gaze prediction

    A readout network returns a single, spatial map of probable gaze locations.
    This module bundles the common processing steps necessary to transform this into
    the predicted gaze distribution:

     - resizing to the stimulus size
     - smoothing of the prediction using a gaussian filter
     - removing of channel and time dimension
     - weighted addition of the center bias
     - normalization
    """

    def __init__(
        self,
        sigma,
        kernel_size=None,
        learn_sigma=False,
        center_bias_weight=1.0,
        learn_center_bias_weight=True,
        saliency_map_factor=4,
    ):
        """Creates a new finalizer

        Args:
            size (tuple): target size for the predictions
            sigma (float): standard deviation of the gaussian kernel used for smoothing
            kernel_size (int, optional): size of the gaussian kernel
            learn_sigma (bool, optional): If True, the standard deviation of the gaussian kernel will
                be learned (default: False)
            center_bias (string or tensor): the center bias
            center_bias_weight (float, optional): initial weight of the center bias
            learn_center_bias_weight (bool, optional): If True, the center bias weight will be
                learned (default: True)
        """
        super(Finalizer, self).__init__()

        self.saliency_map_factor = saliency_map_factor

        self.gauss = GaussianFilterNd([2, 3], sigma, truncate=3, trainable=learn_sigma)
        self.center_bias_weight = nn.Parameter(torch.Tensor([center_bias_weight]), requires_grad=learn_center_bias_weight)

    def forward(self, readout, centerbias):
        """Applies the finalization steps to the given readout"""

        downscaled_centerbias = F.interpolate(
            centerbias.view(centerbias.shape[0], 1, centerbias.shape[1], centerbias.shape[2]),
            scale_factor=1 / self.saliency_map_factor,
            recompute_scale_factor=False,
        )[:, 0, :, :]

        out = F.interpolate(
            readout,
            size=[downscaled_centerbias.shape[1], downscaled_centerbias.shape[2]]
        )

        # apply gaussian filter
        out = self.gauss(out)

        # remove channel dimension
        out = out[:, 0, :, :]

        # add to center bias
        out = out + self.center_bias_weight * downscaled_centerbias

        out = F.interpolate(out[:, np.newaxis, :, :], size=[centerbias.shape[1], centerbias.shape[2]])[:, 0, :, :]

        # normalize
        out = out - out.logsumexp(dim=(1, 2), keepdim=True)

        return out


class DeepGazeII(torch.nn.Module):
    def __init__(self, features, readout_network, downsample=2, readout_factor=16, saliency_map_factor=2, initial_sigma=8.0):
        super().__init__()

        self.readout_factor = readout_factor
        self.saliency_map_factor = saliency_map_factor

        self.features = features

        for param in self.features.parameters():
            param.requires_grad = False
        self.features.eval()

        self.readout_network = readout_network
        self.finalizer = Finalizer(
            sigma=initial_sigma,
            learn_sigma=True,
            saliency_map_factor=self.saliency_map_factor,
        )
        self.downsample = downsample

    def forward(self, x, centerbias):
        orig_shape = x.shape
        x = F.interpolate(
            x,
            scale_factor=1 / self.downsample,
            recompute_scale_factor=False,
        )
        x = self.features(x)

        readout_shape = [math.ceil(orig_shape[2] / self.downsample / self.readout_factor), math.ceil(orig_shape[3] / self.downsample / self.readout_factor)]
        x = [F.interpolate(item, readout_shape) for item in x]

        x = torch.cat(x, dim=1)
        x = self.readout_network(x)
        x = self.finalizer(x, centerbias)

        return x

    def train(self, mode=True):
        self.features.eval()
        self.readout_network.train(mode=mode)
        self.finalizer.train(mode=mode)


class DeepGazeIII(torch.nn.Module):
    def __init__(self, features, saliency_network, scanpath_network, fixation_selection_network, downsample=2, readout_factor=2, saliency_map_factor=2, included_fixations=-2, initial_sigma=8.0):
        super().__init__()

        self.downsample = downsample
        self.readout_factor = readout_factor
        self.saliency_map_factor = saliency_map_factor
        self.included_fixations = included_fixations

        self.features = features

        for param in self.features.parameters():
            param.requires_grad = False
        self.features.eval()

        self.saliency_network = saliency_network
        self.scanpath_network = scanpath_network
        self.fixation_selection_network = fixation_selection_network

        self.finalizer = Finalizer(
            sigma=initial_sigma,
            learn_sigma=True,
            saliency_map_factor=self.saliency_map_factor,
        )

    def forward(self, x, centerbias, x_hist=None, y_hist=None, durations=None):
        orig_shape = x.shape
        x = F.interpolate(x, scale_factor=1 / self.downsample)
        x = self.features(x)

        readout_shape = [math.ceil(orig_shape[2] / self.downsample / self.readout_factor), math.ceil(orig_shape[3] / self.downsample / self.readout_factor)]
        x = [F.interpolate(item, readout_shape) for item in x]

        x = torch.cat(x, dim=1)
        x = self.saliency_network(x)

        if self.scanpath_network is not None:
            scanpath_features = encode_scanpath_features(x_hist, y_hist, size=(orig_shape[2], orig_shape[3]), device=x.device)
            #scanpath_features = F.interpolate(scanpath_features, scale_factor=1 / self.downsample / self.readout_factor)
            scanpath_features = F.interpolate(scanpath_features, readout_shape)
            y = self.scanpath_network(scanpath_features)
        else:
            y = None

        x = self.fixation_selection_network((x, y))

        x = self.finalizer(x, centerbias)

        return x

    def train(self, mode=True):
        self.features.eval()
        self.saliency_network.train(mode=mode)
        if self.scanpath_network is not None:
            self.scanpath_network.train(mode=mode)
        self.fixation_selection_network.train(mode=mode)
        self.finalizer.train(mode=mode)


class DeepGazeIIIMixture(torch.nn.Module):
    def __init__(self, features, saliency_networks, scanpath_networks, fixation_selection_networks, finalizers, downsample=2, readout_factor=2, saliency_map_factor=2, included_fixations=-2, initial_sigma=8.0):
        super().__init__()

        self.downsample = downsample
        self.readout_factor = readout_factor
        self.saliency_map_factor = saliency_map_factor
        self.included_fixations = included_fixations

        self.features = features

        for param in self.features.parameters():
            param.requires_grad = False
        self.features.eval()

        self.saliency_networks = torch.nn.ModuleList(saliency_networks)
        self.scanpath_networks = torch.nn.ModuleList(scanpath_networks)
        self.fixation_selection_networks = torch.nn.ModuleList(fixation_selection_networks)
        self.finalizers = torch.nn.ModuleList(finalizers)

    def forward(self, x, centerbias, x_hist=None, y_hist=None, durations=None):
        orig_shape = x.shape
        x = F.interpolate(
            x,
            scale_factor=1 / self.downsample,
            recompute_scale_factor=False,
        )
        x = self.features(x)

        readout_shape = [math.ceil(orig_shape[2] / self.downsample / self.readout_factor), math.ceil(orig_shape[3] / self.downsample / self.readout_factor)]
        x = [F.interpolate(item, readout_shape) for item in x]

        x = torch.cat(x, dim=1)

        predictions = []

        readout_input = x

        for saliency_network, scanpath_network, fixation_selection_network, finalizer in zip(
            self.saliency_networks, self.scanpath_networks, self.fixation_selection_networks, self.finalizers
        ):

            x = saliency_network(readout_input)

            if scanpath_network is not None:
                scanpath_features = encode_scanpath_features(x_hist, y_hist, size=(orig_shape[2], orig_shape[3]), device=x.device)
                scanpath_features = F.interpolate(scanpath_features, readout_shape)
                y = scanpath_network(scanpath_features)
            else:
                y = None

            x = fixation_selection_network((x, y))

            x = finalizer(x, centerbias)

            predictions.append(x[:, np.newaxis, :, :])

        predictions = torch.cat(predictions, dim=1) - np.log(len(self.saliency_networks))

        prediction = predictions.logsumexp(dim=(1), keepdim=True)

        return prediction


class MixtureModel(torch.nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, *args, **kwargs):
        predictions = [model.forward(*args, **kwargs) for model in self.models]
        predictions = torch.cat(predictions, dim=1)
        predictions -= np.log(len(self.models))
        prediction = predictions.logsumexp(dim=(1), keepdim=True)

        return prediction
