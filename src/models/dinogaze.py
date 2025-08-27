"""
Model definition and builder for a standard DinoGaze model.

This model uses a DINOv2 backbone, as seen in the training script
`scripts/train_dinogaze_ddp.py`. It defines a self-contained `DinoGaze`
class that is compatible with the modular training framework, handling
both spatial-only and scanpath modes.
"""
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.registry import register_model
from src.dinov2_backbone import DinoV2Backbone
from src.modules import (
    Finalizer, encode_scanpath_features,
    build_saliency_network, build_scanpath_network, build_fixation_selection_network
)

logger = logging.getLogger(__name__)

# =============================================================================
# 1. SELF-CONTAINED MODEL DEFINITION
# =============================================================================

class DinoGaze(nn.Module):
    """
    A modular-aware implementation of the standard DinoGaze model.

    This version is designed to work with a modular builder, which constructs
    different network heads for spatial vs. scanpath stages. The forward pass
    handles these configurations gracefully.
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
        # Backbone is frozen by default in the builder.
        # The train() override ensures it stays in eval mode.
        self.features.eval()

        self.saliency_network = saliency_network
        self.scanpath_network = scanpath_network
        self.fixation_selection_network = fixation_selection_network

        self.finalizer = Finalizer(
            sigma=8.0,
            learn_sigma=True,
            saliency_map_factor=saliency_map_factor,
        )

    def forward(self, x, centerbias, x_hist=None, y_hist=None, **kwargs):
        orig_shape_hw = x.shape[2:]

        # --- 1. Feature Extraction ---
        if self.downsample != 1:
            logger.warning(f"DinoGaze expects downsample=1, but got {self.downsample}. Input will be resized.")
            x_in = F.interpolate(x, scale_factor=1 / self.downsample, recompute_scale_factor=False)
        else:
            x_in = x

        features_list = self.features(x_in)

        readout_h = math.ceil(orig_shape_hw[0] / self.downsample / self.readout_factor)
        readout_w = math.ceil(orig_shape_hw[1] / self.downsample / self.readout_factor)
        readout_shape = (readout_h, readout_w)

        interpolated_features = [F.interpolate(f, size=readout_shape, mode='bilinear', align_corners=False) for f in features_list]
        concatenated_features = torch.cat(interpolated_features, dim=1)

        # --- 2. Saliency Path ---
        saliency_features = self.saliency_network(concatenated_features)

        # --- 3. Scanpath and Fixation Selection ---
        scanpath_features = None
        if self.scanpath_network is not None:
            if x_hist is not None and y_hist is not None and x_hist.numel() > 0:
                scanpath_encoding = encode_scanpath_features(x_hist, y_hist, size=orig_shape_hw, device=x.device)
                scanpath_encoding = F.interpolate(scanpath_encoding, size=readout_shape, mode='bilinear', align_corners=False)
                scanpath_features = self.scanpath_network(scanpath_encoding)
            else:
                B, _, H, W = saliency_features.shape
                scanpath_features = torch.zeros(B, 16, H, W, device=saliency_features.device)

        combined_input = (saliency_features, scanpath_features)
        readout = self.fixation_selection_network(combined_input)

        # --- 4. Finalizer ---
        log_density = self.finalizer(readout, centerbias)
        return log_density

    def train(self, mode=True):
        super().train(mode)
        self.features.eval()


# =============================================================================
# 2. MAIN MODEL BUILDER FUNCTION (REGISTERED)
# =============================================================================

@register_model("dinogaze")
def build(cfg):
    """
    Builds the complete DinoGaze model from a configuration object,
    based on the implementation in `scripts/train_dinogaze_ddp.py`.
    """
    extra = cfg.stage.extra
    logger.info("Building DinoGaze model with configuration: %s", extra)

    # 1. Build the DINOv2 backbone
    dino_model_name = extra.get("dino_model_name", "dinov2_vitl14")
    features_module = DinoV2Backbone(
        layers=extra.get("dino_layers_for_main_path", [-3, -2, -1]),
        model_name=dino_model_name,
        freeze=True
    )

    unfreeze_layers = extra.get("unfreeze_vit_layers", [])
    if unfreeze_layers:
        logger.info(f"Unfreezing DINOv2 layers: {unfreeze_layers}")
        for name, param in features_module.backbone.named_parameters():
            if name.startswith('blocks.'):
                try:
                    block_index = int(name.split('.')[1])
                    if block_index in unfreeze_layers:
                        param.requires_grad = True
                except (IndexError, ValueError):
                    pass

    # 2. Build network heads based on config flags
    C_in = len(features_module.layers) * features_module.num_channels
    saliency_net = build_saliency_network(C_in, add_sa_head=False)

    scanpath_net = None
    included_fixations = extra.get("included_fixations")

    if extra.get("is_scanpath_stage", False):
        if not included_fixations:
            raise ValueError("'is_scanpath_stage' is true, but 'included_fixations' is not defined.")
        logger.info("  - Building in SCANPATH mode.")
        scanpath_net = build_scanpath_network()
        fixsel_net = build_fixation_selection_network(scanpath_features=16)
    else:
        logger.info("  - Building in SPATIAL-ONLY mode.")
        fixsel_net = build_fixation_selection_network(scanpath_features=0)
        if included_fixations:
            logger.warning("  - 'included_fixations' is set but 'is_scanpath_stage' is false. History will not be used.")
            included_fixations = None

    # 3. Assemble the final model
    model = DinoGaze(
        features=features_module,
        saliency_network=saliency_net,
        scanpath_network=scanpath_net,
        fixation_selection_network=fixsel_net,
        downsample=extra.get("downsample", 1.0),
        readout_factor=extra.get("dino_patch_size", 14),
        saliency_map_factor=extra.get("saliency_map_factor", 4),
        included_fixations=included_fixations
    )

    if extra.get("freeze_saliency_network", False):
        logger.info("  - Freezing saliency network as per 'freeze_saliency_network: true'.")
        for param in model.saliency_network.parameters():
            param.requires_grad = False

    return model
