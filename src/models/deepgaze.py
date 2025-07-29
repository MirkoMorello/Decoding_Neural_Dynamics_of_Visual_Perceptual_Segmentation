"""
Model definition and builder for the original DeepGazeIII model.

This model uses a frozen DenseNet-201 backbone and a series of custom
convolutional heads for saliency and scanpath processing. It does not use
any form of SPADE normalization.
"""
from collections import OrderedDict
import logging
import torch
import torch.nn as nn

from src.registry import register_model
from src.modules import FeatureExtractor, DeepGazeIII
from src.layers import (
    Bias, LayerNorm, LayerNormMultiInput,
    Conv2dMultiInput, FlexibleScanpathHistoryEncoding
)

try:
    from DeepGaze.deepgaze_pytorch.features.densenet import RGBDenseNet201
except ImportError:
    raise ImportError("The 'DeepGaze' library is required for the DeepGazeIII model.")

logger = logging.getLogger(__name__)

# =============================================================================
# 1. COMPONENT BUILDER FUNCTIONS
# =============================================================================

def _build_saliency_network(input_channels: int) -> nn.Module:
    """Builds the standard saliency head for DeepGazeIII."""
    return nn.Sequential(OrderedDict([
        ('layernorm0', LayerNorm(input_channels)),
        ('conv0', nn.Conv2d(input_channels, 8, (1, 1), bias=False)),
        ('bias0', Bias(8)),
        ('softplus0', nn.Softplus()),
        ('layernorm1', LayerNorm(8)),
        ('conv1', nn.Conv2d(8, 16, (1, 1), bias=False)),
        ('bias1', Bias(16)),
        ('softplus1', nn.Softplus()),
        ('layernorm2', LayerNorm(16)),
        ('conv2', nn.Conv2d(16, 1, (1, 1), bias=False)),
        ('bias2', Bias(1)),
        ('softplus2', nn.Softplus()),
    ]))

def _build_scanpath_network(in_fixations: int) -> nn.Module:
    """Builds the scanpath processing head, configurable by history length."""
    return nn.Sequential(OrderedDict([
        ('encoding0', FlexibleScanpathHistoryEncoding(
            in_fixations=in_fixations, channels_per_fixation=3,
            out_channels=128, kernel_size=(1, 1), bias=True)
        ),
        ('softplus0', nn.Softplus()),
        ('layernorm1', LayerNorm(128)),
        ('conv1', nn.Conv2d(128, 16, (1, 1), bias=False)),
        ('bias1', Bias(16)),
        ('softplus1', nn.Softplus()),
    ]))

def _build_fixation_selection_network(scanpath_features: int) -> nn.Module:
    """Builds the network that combines saliency and scanpath features."""
    saliency_channels = 1
    in_channels_list = [saliency_channels]
    if scanpath_features > 0:
        in_channels_list.append(scanpath_features)

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


# =============================================================================
# 2. MAIN MODEL BUILDER FUNCTION
# =============================================================================

@register_model("deepgaze3")
def build(cfg):
    """
    Builds the complete original DeepGazeIII model from a configuration object.
    Architecture and freezing are controlled by explicit flags in `cfg.stage.extra`.
    """
    extra = cfg.stage.extra
    logger.info("Building DeepGazeIII model with configuration: %s", extra)
    
    # 1. Build the frozen DenseNet-201 backbone
    densenet_base = RGBDenseNet201()
    densenet_feature_nodes = [
        '1.features.denseblock4.denselayer32.norm1',
        '1.features.denseblock4.denselayer32.conv1',
        '1.features.denseblock4.denselayer31.conv2',
    ]
    features_module = FeatureExtractor(densenet_base, densenet_feature_nodes)
    for param in features_module.parameters():
        param.requires_grad = False
    features_module.eval()
    
    # 2. Build the network heads based on explicit config flags
    saliency_net = _build_saliency_network(input_channels=2048)

    scanpath_net = None
    fixsel_net = None
    included_fixations = extra.get("included_fixations") # Get history list

    if extra.get("is_scanpath_stage", False):
        if not included_fixations:
            raise ValueError("'is_scanpath_stage' is true, but 'included_fixations' is not defined in config.stage.extra.")
        
        logger.info("  - Building in SCANPATH mode.")
        scanpath_net = _build_scanpath_network(in_fixations=len(included_fixations))
        fixsel_net = _build_fixation_selection_network(scanpath_features=16)
    else:
        logger.info("  - Building in SPATIAL-ONLY mode.")
        fixsel_net = _build_fixation_selection_network(scanpath_features=0)
        if included_fixations:
            logger.warning("  - 'included_fixations' is set but 'is_scanpath_stage' is false. History will not be used.")
            included_fixations = None # Ensure it's None for the model

    # 3. Assemble the final DeepGazeIII model
    model = DeepGazeIII(
        features=features_module,
        saliency_network=saliency_net,
        scanpath_network=scanpath_net,
        fixation_selection_network=fixsel_net,
        downsample=extra.get("downsample", 1.0),
        readout_factor=4,
        saliency_map_factor=4,
        included_fixations=included_fixations
    )

    # 4. Apply stage-specific freezing based on an explicit config flag
    if extra.get("freeze_saliency_network", False):
        logger.info("  - Freezing saliency network as per 'freeze_saliency_network: true'.")
        frozen_scopes = [
            "saliency_network.layernorm0", "saliency_network.conv0", "saliency_network.bias0",
            "saliency_network.layernorm1", "saliency_network.conv1", "saliency_network.bias1",
        ]
        for name, param in model.named_parameters():
            if any(name.startswith(scope) for scope in frozen_scopes):
                param.requires_grad = False
    else:
        # For all other cases, ensure all head parameters are trainable by default.
        logger.info("  - All head parameters are trainable (backbone remains frozen).")
        for name, param in model.named_parameters():
            if not name.startswith('features.'):
                param.requires_grad = True
    
    return model