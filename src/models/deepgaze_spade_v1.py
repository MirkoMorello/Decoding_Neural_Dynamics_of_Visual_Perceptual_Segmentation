"""
Model definition and builder for DeepGazeIII with DenseNet and STATIC SPADE.

This model uses a DenseNet-201 backbone for the main saliency pathway.
SPADE modulation is achieved via a learnable nn.Embedding layer.
This modular version correctly handles both spatial and scanpath stages and uses
a learnable scanpath encoding via FlexibleScanpathHistoryEncoding.
"""
import torch
import torch.nn as nn
import math
from torch.nn import functional as F

from src.registry import register_model
from src.modules import (
    FeatureExtractor, Finalizer, build_scanpath_network,
    build_fixation_selection_network, encode_scanpath_features
)
from src.models.common.spade_layers import SaliencyNetworkSPADE

try:
    from DeepGaze.deepgaze_pytorch.features.densenet import RGBDenseNet201
except ImportError:
    raise ImportError("The 'DeepGaze' library is required for this model.")


class DeepgazeSpadeV1(nn.Module):
    """
    Modular implementation of DeepGazeIII with DenseNet and static SPADE.
    """
    def __init__(self, features, saliency_network, scanpath_network,
                 fixation_selection_network, downsample, readout_factor,
                 saliency_map_factor, initial_sigma, finalizer_learn_sigma,
                 included_fixations):
        super().__init__()
        self.features = features
        for p in self.features.parameters(): p.requires_grad = False
        self.features.eval()

        self.saliency_network = saliency_network
        self.scanpath_network = scanpath_network
        self.fixation_selection_network = fixation_selection_network
        
        self.downsample = downsample
        self.readout_factor = readout_factor
        self.included_fixations = included_fixations

        self.finalizer = Finalizer(
            sigma=initial_sigma, learn_sigma=finalizer_learn_sigma,
            saliency_map_factor=saliency_map_factor
        )

    def forward(self, image, centerbias, **kwargs):
        segmentation_mask = kwargs.get('segmentation_mask')
        if segmentation_mask is None:
            raise ValueError(f"{self.__class__.__name__} requires 'segmentation_mask'.")

        orig_shape_hw = image.shape[2:]
        img_for_features = F.interpolate(image, scale_factor=1.0/self.downsample, mode='bilinear', align_corners=False) if self.downsample != 1 else image
        
        extracted_maps = self.features(img_for_features)
        
        readout_h = math.ceil(orig_shape_hw[0] / self.downsample / self.readout_factor)
        readout_w = math.ceil(orig_shape_hw[1] / self.downsample / self.readout_factor)
        readout_shape = (readout_h, readout_w)
        
        processed_features = [F.interpolate(f, size=readout_shape, mode='bilinear', align_corners=False) for f in extracted_maps]
        concatenated = torch.cat(processed_features, dim=1)

        saliency_out = self.saliency_network(concatenated, segmentation_mask)
        
        if self.scanpath_network is not None:
            # --- SCANPATH MODE ---
            x_hist, y_hist = kwargs.get('x_hist'), kwargs.get('y_hist')
            if x_hist is None or y_hist is None or x_hist.numel() == 0:
                B, _, H, W = saliency_out.shape
                scanpath_out = torch.zeros(B, 16, H, W, device=saliency_out.device)
            else:
                scanpath_encoding = encode_scanpath_features(x_hist, y_hist, size=orig_shape_hw, device=image.device)
                scanpath_encoding = F.interpolate(scanpath_encoding, size=readout_shape, mode='bilinear', align_corners=False)
                scanpath_out = self.scanpath_network(scanpath_encoding)

            final_readout = self.fixation_selection_network((saliency_out, scanpath_out))
        else:
            # --- SPATIAL-ONLY MODE ---
            final_readout = self.fixation_selection_network((saliency_out,))
        
        return self.finalizer(final_readout, centerbias)

    def train(self, mode=True):
        super().train(mode)
        self.features.eval()

@register_model("densenet_spade_static")
def build(cfg):
    """Builds DeepGazeIII with DenseNet and STATIC SPADE."""
    extra = cfg.stage.extra
    
    densenet_base = RGBDenseNet201()
    densenet_feature_nodes = [
        '1.features.denseblock4.denselayer32.norm1',
        '1.features.denseblock4.denselayer32.conv1',
        '1.features.denseblock4.denselayer31.conv2'
    ]
    features_module = FeatureExtractor(densenet_base, densenet_feature_nodes)
    main_ch = 2048

    saliency_net = SaliencyNetworkSPADE(
        input_channels=main_ch,
        num_total_segments=extra.get("num_total_segments", 64),
        seg_embedding_dim=extra.get("seg_embedding_dim", 128)
    )
    
    if extra.get("is_scanpath_stage", False):
        # This correctly builds the modern scanpath network
        scanpath_net = build_scanpath_network()
        fixsel_net = build_fixation_selection_network(scanpath_features=16)
        included_fixations = extra.get("included_fixations")
        if not included_fixations:
            raise ValueError("'is_scanpath_stage' requires 'included_fixations'.")
    else:
        scanpath_net = None
        fixsel_net = build_fixation_selection_network(scanpath_features=0)
        included_fixations = None
    
    model = DeepgazeSpadeV1(
        features=features_module,
        saliency_network=saliency_net,
        scanpath_network=scanpath_net,
        fixation_selection_network=fixsel_net,
        downsample=extra.get("downsample", 1.0),
        readout_factor=extra.get("readout_factor", 4),
        saliency_map_factor=extra.get("saliency_map_factor", 4),
        initial_sigma=extra.get("finalizer_initial_sigma", 8.0),
        finalizer_learn_sigma=extra.get("finalizer_learn_sigma", True),
        included_fixations=included_fixations
    )
    
    if extra.get("freeze_saliency_network", False):
        for param in model.saliency_network.parameters():
            param.requires_grad = False
            
    return model