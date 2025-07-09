# modules.py
import functools
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import GaussianFilterNd, Bias, LayerNorm, LayerNormMultiInput, Conv2dMultiInput, SelfAttention, FlexibleScanpathHistoryEncoding
from collections import OrderedDict


def encode_scanpath_features(
    x_hist: torch.Tensor,
    y_hist: torch.Tensor,
    *,
    size: tuple[int, int],                 # (H, W)
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Returns a tensor (B, 3·N_fix, H, W) with
        dx, dy, distance  for each of the N_fix history entries.

    Any slot whose (x_hist, y_hist) is NaN is interpreted as “no fixation yet”
    and its three channels are set to **0**.
    """
    if device is None:
        device = x_hist.device

    B, N_fix = x_hist.shape
    H, W     = size

    # --------- 1. handle NaNs (= missing fixations) ----------
    valid = (~torch.isnan(x_hist)) & (~torch.isnan(y_hist))          # (B, N_fix)
    #   replace NaNs with zeros so arithmetic below is finite
    x_hist_f = torch.where(valid, x_hist, torch.zeros_like(x_hist))
    y_hist_f = torch.where(valid, y_hist, torch.zeros_like(y_hist))

    # --------- 2. build coordinate grids ---------------------
    xs = torch.arange(W, device=device, dtype=torch.float32)
    ys = torch.arange(H, device=device, dtype=torch.float32)
    YS, XS = torch.meshgrid(ys, xs, indexing='ij')                   # (H, W)

    XS = XS.expand(B, N_fix, H, W).clone()
    YS = YS.expand(B, N_fix, H, W).clone()

    XS -= x_hist_f.unsqueeze(-1).unsqueeze(-1)
    YS -= y_hist_f.unsqueeze(-1).unsqueeze(-1)
    DIST = torch.sqrt(XS**2 + YS**2)

    # --------- 3. apply the valid-mask ------------------------
    #         invalid slots → 0 in all three channels
    mask = valid.unsqueeze(-1).unsqueeze(-1)                         # (B,N_fix,1,1)
    XS   = XS   * mask
    YS   = YS   * mask
    DIST = DIST * mask

    # --------- 4. interleave dx,dy,dist exactly as before -----
    out = torch.cat((XS, YS, DIST), dim=1)                           # (B,3N_fix,H,W)
    return out

def build_saliency_network(input_channels, add_sa_head=False):
    """ Builds the saliency prediction head network. """
    layers = OrderedDict()

    if add_sa_head:
        # Add SelfAttention as the first layer
        # NOTE: SelfAttention outputs the same number of channels as input by default.
        layers['sa_head'] = SelfAttention(input_channels, key_channels=input_channels // 8, return_attention=False)
        layers['layernorm_sa_out'] = LayerNorm(input_channels) # Normalize SA output
        layers['softplus_sa_out'] = nn.Softplus() # Add non-linearity after SA

    # Reduced complexity slightly given potentially richer ViT features
    layers['layernorm0'] = LayerNorm(input_channels)
    layers['conv0'] = nn.Conv2d(input_channels, 8, (1, 1), bias=False) # Increased capacity slightly
    layers['bias0'] = Bias(8)
    layers['softplus0'] = nn.Softplus()

    layers['layernorm1'] = LayerNorm(8)
    layers['conv1'] = nn.Conv2d(8, 16, (1, 1), bias=False) # Increased capacity slightly
    layers['bias1'] = Bias(16)
    layers['softplus1'] = nn.Softplus()

    layers['layernorm2'] = LayerNorm(16)
    layers['conv2'] = nn.Conv2d(16, 1, (1, 1), bias=False)
    layers['bias2'] = Bias(1)
    layers['softplus2'] = nn.Softplus() # Ensures non-negative output before final log-likelihood

    return nn.Sequential(layers)


def build_scanpath_network():
    """ Builds the network processing scanpath history. """
    return nn.Sequential(OrderedDict([
        ('encoding0', FlexibleScanpathHistoryEncoding(in_fixations=4, channels_per_fixation=3, out_channels=128, kernel_size=[1, 1], bias=True)),
        ('softplus0', nn.Softplus()),
        ('layernorm1', LayerNorm(128)),
        ('conv1', nn.Conv2d(128, 16, (1, 1), bias=False)), # Output 16 channels
        ('bias1', Bias(16)),
        ('softplus1', nn.Softplus()),
    ]))


    
def build_fixation_selection_network(scanpath_features=16):
    """ Builds the network combining saliency and scanpath features for fixation selection. """
    saliency_channels = 1 # Output of saliency network's core path before combination
    in_channels_list = [saliency_channels, scanpath_features if scanpath_features > 0 else 0]

    return nn.Sequential(OrderedDict([
        ('layernorm0', LayerNormMultiInput(in_channels_list)), # Now gets [1, 0] or [1, 16]
        ('conv0', Conv2dMultiInput(in_channels_list, 128, (1, 1), bias=False)), # Now gets [1, 0] or [1, 16]
        ('bias0', Bias(128)),
        ('softplus0', nn.Softplus()),
        ('layernorm1', LayerNorm(128)),
        ('conv1', nn.Conv2d(128, 16, (1, 1), bias=False)),
        ('bias1', Bias(16)),
        ('softplus1', nn.Softplus()),
        ('conv2', nn.Conv2d(16, 1, (1, 1), bias=False)), # Final output layer
    ]))

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

    def forward(self, x, centerbias, x_hist=None, y_hist=None, durations=None, **kwargs):
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

    def forward(self, x, centerbias, x_hist=None, y_hist=None, durations=None, **kwargs):
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
