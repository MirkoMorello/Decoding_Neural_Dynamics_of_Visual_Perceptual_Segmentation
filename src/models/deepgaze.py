"""
Model definition and builder for the original DeepGazeIII model.

This file defines a self-contained `DeepGazeIII_modular` class that
overrides the forward pass of the original DeepGazeIII to handle
both spatial-only and scanpath modes correctly within a modular
training script.
"""
from collections import OrderedDict
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.registry import register_model # TEST
from src.modules import (
    FeatureExtractor, Finalizer, encode_scanpath_features,
    build_saliency_network, build_scanpath_network, build_fixation_selection_network
)

try:
    from DeepGaze.deepgaze_pytorch.features.densenet import RGBDenseNet201
except ImportError:
    raise ImportError("The 'DeepGaze' library is required for the DeepGazeIII model.")

logger = logging.getLogger(__name__)

# =============================================================================
# 1. SELF-CONTAINED MODEL DEFINITION FOR THIS BUILDER
# =============================================================================

class DeepGazeIII_modular(nn.Module):
    """
    A modular-aware implementation of DeepGazeIII.

    This version is specifically designed to work with the modular builder,
    which constructs different network heads for spatial vs. scanpath stages.
    The forward pass is modified to handle these different configurations gracefully.
    """
    def __init__(self,
                 features: nn.Module,
                 saliency_network: nn.Module,
                 scanpath_network: nn.Module | None,
                 fixation_selection_network: nn.Module,
                 downsample: float,
                 readout_factor: int,
                 saliency_map_factor: int,
                 included_fixations: list[int] | None):
        super().__init__()

        self.downsample = downsample
        self.readout_factor = readout_factor
        self.included_fixations = included_fixations

        self.features = features
        for param in self.features.parameters():
            param.requires_grad = False
        self.features.eval()

        self.saliency_network = saliency_network
        self.scanpath_network = scanpath_network
        self.fixation_selection_network = fixation_selection_network

        self.finalizer = Finalizer(
            sigma=8.0,  # Default from original, can be configured if needed
            learn_sigma=True,
            saliency_map_factor=saliency_map_factor,
        )

    def forward(self, x, centerbias, x_hist=None, y_hist=None, durations=None, **kwargs):
        orig_shape_hw = x.shape[2:]

        # --- 1. Downsample and Feature Extraction ---
        x = F.interpolate(x, scale_factor=1 / self.downsample, recompute_scale_factor=False)
        features_list = self.features(x)
        
        readout_h = math.ceil(orig_shape_hw[0] / self.downsample / self.readout_factor)
        readout_w = math.ceil(orig_shape_hw[1] / self.downsample / self.readout_factor)
        readout_shape = (readout_h, readout_w)

        interpolated_features = [F.interpolate(f, size=readout_shape, mode='bilinear', align_corners=False) for f in features_list]
        concatenated_features = torch.cat(interpolated_features, dim=1)

        # --- 2. Saliency Path ---
        saliency_features = self.saliency_network(concatenated_features)

        # --- 3. Conditional Scanpath and Fixation Selection (THE FIX) ---
        if self.scanpath_network is not None:
            # --- SCANPATH MODE ---
            if x_hist is None or y_hist is None or x_hist.numel() == 0:
                B, _, H, W = saliency_features.shape
                scanpath_features = torch.zeros(B, 16, H, W, device=saliency_features.device)
            else:
                scanpath_encoding = encode_scanpath_features(x_hist, y_hist, size=orig_shape_hw, device=x.device)
                scanpath_encoding = F.interpolate(scanpath_encoding, size=readout_shape, mode='bilinear', align_corners=False)
                scanpath_features = self.scanpath_network(scanpath_encoding)
            
            combined_input = (saliency_features, scanpath_features)
            readout = self.fixation_selection_network(combined_input)
        else:
            # --- SPATIAL-ONLY MODE ---
            readout = self.fixation_selection_network((saliency_features, None))


        # --- 4. Finalizer ---
        log_density = self.finalizer(readout, centerbias)
        return log_density

    def train(self, mode=True):
        # Override train() to ensure the backbone stays in eval mode.
        super().train(mode)
        self.features.eval()


# =============================================================================
# 3. MAIN MODEL BUILDER FUNCTION (REGISTERED)
# =============================================================================

@register_model("deepgaze3")
def build(cfg):
    """
    Builds the complete original DeepGazeIII model from a configuration object.
    This now uses the `DeepGazeIII_modular` class with the corrected forward pass.
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
    
    # 2. Build the network heads based on explicit config flags
    # Input channels from these DenseNet hooks is 2048.
    saliency_net = build_saliency_network(input_channels=2048)

    scanpath_net = None
    fixsel_net = None
    included_fixations = extra.get("included_fixations") # Get history list

    if extra.get("is_scanpath_stage", False):
        if not included_fixations:
            raise ValueError("'is_scanpath_stage' is true, but 'included_fixations' is not defined in config.stage.extra.")
        
        logger.info("  - Building in SCANPATH mode.")
        scanpath_net = build_scanpath_network()
        fixsel_net = build_fixation_selection_network(scanpath_features=16)
    else:
        logger.info("  - Building in SPATIAL-ONLY mode.")
        fixsel_net = build_fixation_selection_network(scanpath_features=0)
        if included_fixations:
            logger.warning("  - 'included_fixations' is set but 'is_scanpath_stage' is false. History will not be used.")
            included_fixations = None # Ensure it's None for the model

    # 3. Assemble the final model using our new, self-contained class
    model = DeepGazeIII_modular(
        features=features_module,
        saliency_network=saliency_net,
        scanpath_network=scanpath_net,
        fixation_selection_network=fixsel_net,
        downsample=extra.get("downsample", 2.0), # Default is 2 in notebook
        readout_factor=4, # From notebook
        saliency_map_factor=4, # From notebook
        included_fixations=included_fixations
    )

    # 4. Apply stage-specific freezing based on an explicit config flag
    if extra.get("freeze_saliency_network", False):
        logger.info("  - Freezing saliency network as per 'freeze_saliency_network: true'.")
        # Freeze all parameters of the saliency_network
        for param in model.saliency_network.parameters():
            param.requires_grad = False
    
    # The default behavior is that all non-backbone parameters are trainable.
    # We don't need an explicit 'else' block for this.
    
    return model