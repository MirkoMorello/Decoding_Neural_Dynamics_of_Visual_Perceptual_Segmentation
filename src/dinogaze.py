# dinogaze.py

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from torch.utils import model_zoo

from .features.densenet import RGBDenseNet201
from .modules import FeatureExtractor, Finalizer, DeepGazeIIIMixture
from .layers import FlexibleScanpathHistoryEncoding

from .layers import (
    Conv2dMultiInput,
    LayerNorm,
    LayerNormMultiInput,
    Bias,
    SelfAttention,
)


def build_saliency_network(input_channels, add_sa_head=False):
    """ Builds the saliency prediction head network. """
    # Using _logger.info here is fine, it will only log on master
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


class DeepGazeIII(DeepGazeIIIMixture):
    """DeepGazeIII model

    :note
    See Kümmerer, M., Bethge, M., & Wallis, T.S.A. (2022). DeepGaze III: Modeling free-viewing human scanpaths with deep learning. Journal of Vision 2022, https://doi.org/10.1167/jov.22.5.7
    """
    def __init__(self, pretrained=True):
        features = RGBDenseNet201()

        feature_extractor = FeatureExtractor(features, [
            '1.features.denseblock4.denselayer32.norm1',
            '1.features.denseblock4.denselayer32.conv1',
            '1.features.denseblock4.denselayer31.conv2',
        ])

        saliency_networks = []
        scanpath_networks = []
        fixation_selection_networks = []
        finalizers = []
        for component in range(10):
            saliency_network = build_saliency_network(2048)
            scanpath_network = build_scanpath_network()
            fixation_selection_network = build_fixation_selection_network()

            saliency_networks.append(saliency_network)
            scanpath_networks.append(scanpath_network)
            fixation_selection_networks.append(fixation_selection_network)
            finalizers.append(Finalizer(sigma=8.0, learn_sigma=True, saliency_map_factor=4))

        super().__init__(
            features=feature_extractor,
            saliency_networks=saliency_networks,
            scanpath_networks=scanpath_networks,
            fixation_selection_networks=fixation_selection_networks,
            finalizers=finalizers,
            downsample=2,
            readout_factor=4,
            saliency_map_factor=4,
            included_fixations=[-1, -2, -3, -4]
        )

        if pretrained:
            self.load_state_dict(model_zoo.load_url('https://github.com/matthias-k/DeepGaze/releases/download/v1.1.0/deepgaze3.pth', map_location=torch.device('cpu')))