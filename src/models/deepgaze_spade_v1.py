"""
This file contains the implementation of the DeepGazeIII model with SPADE layers.
It uses a DenseNet-201 backbone and supports both spatial and scanpath-based saliency prediction.
It uses Densenet features for the main saliency pathway and a separate semantic map for SPADE modulation.
This version uses no semantic painting, the embedding is learned through time.
"""

# src/models/densenet_spade.py
import torch
import torch.nn as nn
import math
from torch.nn import functional as F

from src.registry import register_model
from src.modules import ( 
    Finalizer, build_scanpath_network, encode_scanpath_features,
    build_fixation_selection_network, FeatureExtractor
)
from src.models.common.spade_layers import SaliencyNetworkSPADE
from DeepGaze.deepgaze_pytorch.features.densenet import RGBDenseNet201

class DeepGazeIIISpade(nn.Module):
    def __init__(self, features, saliency_network, scanpath_network, fixation_selection_network,
                 finalizer_learn_sigma, initial_sigma=8.0, downsample=1, readout_factor=4, saliency_map_factor=4):
        super().__init__()
        self.features = features
        self.saliency_network = saliency_network
        self.scanpath_network = scanpath_network
        self.fixation_selection_network = fixation_selection_network
        self.downsample = downsample
        self.readout_factor = readout_factor
        self.finalizer = Finalizer(
            sigma=initial_sigma, learn_sigma=finalizer_learn_sigma,
            saliency_map_factor=saliency_map_factor
        )

    def forward(self, image, centerbias, segmentation_mask, **kwargs):
        """
        Corrected forward pass that handles both spatial and scanpath stages.
        """
        if segmentation_mask is None:
            raise ValueError("SaliencyNetworkSPADE requires a segmentation_mask.")

        # --- 1. Backbone Feature Extraction ---
        img_for_features = F.interpolate(image, scale_factor=1.0/self.downsample, mode='bilinear', align_corners=False) if self.downsample != 1 else image
        extracted_maps = self.features(img_for_features)
        readout_h = math.ceil(image.shape[2] / self.downsample / self.readout_factor)
        readout_w = math.ceil(image.shape[3] / self.downsample / self.readout_factor)
        processed_features = [F.interpolate(f, size=(readout_h, readout_w), mode='bililinear', align_corners=False) for f in extracted_maps]
        concatenated = torch.cat(processed_features, dim=1)

        # --- 2. Saliency Head ---
        saliency_out = self.saliency_network(concatenated, segmentation_mask)
        
        # --- 3. Scanpath Head (Handles both cases) ---
        # Get scanpath history from kwargs if available
        x_hist = kwargs.get('x_hist')
        y_hist = kwargs.get('y_hist')

        # Check if we are in a scanpath stage AND have history to process
        if self.scanpath_network is not None and x_hist is not None and y_hist is not None and x_hist.numel() > 0:
            scanpath_features = encode_scanpath_features(x_hist, y_hist, size=image.shape[2:])
            scanpath_features_resized = F.interpolate(scanpath_features, size=(readout_h, readout_w), mode='bilinear', align_corners=False)
            scanpath_out = self.scanpath_network(scanpath_features_resized)
        else:
            # SHIM: For spatial-only stages or the first fixation, create a zero tensor
            # with the expected shape for the scanpath path.
            # The scanpath network output is 16 channels.
            B, _, H, W = saliency_out.shape
            device = saliency_out.device
            scanpath_out = torch.zeros(B, 16, H, W, device=device)

        # --- 4. Fixation Selection Head ---
        # This now correctly receives both saliency and scanpath (or zero-shim) features.
        final_readout = self.fixation_selection_network((saliency_out, scanpath_out))
        
        # --- 5. Finalizer ---
        return self.finalizer(final_readout, centerbias)

    def train(self, mode=True):
        if hasattr(self.features, 'eval'):
            self.features.eval()
        self.saliency_network.train(mode)
        if self.scanpath_network:
            self.scanpath_network.train(mode)
        self.fixation_selection_network.train(mode)
        self.finalizer.train(mode)
        super().train(mode)


# --- Builder Functions (Unchanged, they are correct) ---

def _build_densenet_backbone(device='cpu'):
    densenet_base = RGBDenseNet201()
    densenet_feature_nodes = [
        '1.features.denseblock4.denselayer32.norm1',
        '1.features.denseblock4.denselayer32.conv1',
        '1.features.denseblock4.denselayer31.conv2'
    ]
    features_module = FeatureExtractor(densenet_base, densenet_feature_nodes)
    for param in features_module.parameters():
        param.requires_grad = False
    features_module.eval()
    
    with torch.no_grad():
        dummy_out = features_module(torch.randn(1, 3, 256, 256).to(device))
        main_path_channels = sum(f.shape[1] for f in dummy_out)
    return features_module, main_path_channels

@register_model("deepgaze_spade_v1")
def build(cfg):
    """Constructs DeepGazeIII with DenseNet and SPADE."""
    extra = cfg.stage.extra
    features, main_ch = _build_densenet_backbone()
    
    saliency_net = SaliencyNetworkSPADE(
        input_channels=main_ch,
        num_total_segments=extra.get("num_total_segments", 17),
        seg_embedding_dim=extra.get("seg_embedding_dim", 64)
    )
    
    if "scanpath" in cfg.stage.name:
        scanpath_net = build_scanpath_network()
        fixsel_net = build_fixation_selection_network(scanpath_features=16)
    else:
        scanpath_net = None
        fixsel_net = build_fixation_selection_network(scanpath_features=0)
    
    model = DeepGazeIIISpade(
        features=features,
        saliency_network=saliency_net,
        scanpath_network=scanpath_net,
        fixation_selection_network=fixsel_net,
        finalizer_learn_sigma=extra.get("finalizer_learn_sigma", True),
        initial_sigma=extra.get("initial_sigma", 8.0)
    )
    return model