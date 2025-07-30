# src/models/densenet_spade_dynamic.py
"""
Model definition and builder for DeepGazeIIIDynamicEmbedding.

This model uses a frozen DenseNet-201 backbone for two purposes:
1.  Extracting multi-level features for the main saliency pathway.
2.  Extracting deep features from a specific layer to create a dynamic
    semantic map for SPADE modulation.

This version is fully general and supports both spatial-only and
scanpath-based saliency prediction.
"""
import torch
import torch.nn as nn
import math
from torch.nn import functional as F

from src.registry import register_model
from src.modules import (
    FeatureExtractor, Finalizer, build_fixation_selection_network,
    build_scanpath_network, encode_scanpath_features
)
from src.models.common.spade_layers import SaliencyNetworkSPADEDynamic

# Import the DenseNet backbone
try:
    from DeepGaze.deepgaze_pytorch.features.densenet import RGBDenseNet201
except ImportError:
    raise ImportError("The 'DeepGaze' library is required.")

# Import torch_scatter, which is essential for this model
try:
    from torch_scatter import scatter_mean
except ImportError:
    raise ImportError("torch_scatter is required for DeepGazeIIIDynamicEmbedding.")


# =============================================================================
# 1. COMPLETE MODEL CLASS DEFINITION
# =============================================================================

class DeepgazeSpadeV2(nn.Module):
    """
    The complete DeepGazeIII model with a DenseNet backbone and dynamic SPADE.
    This version supports both spatial and scanpath training stages.
    """
    def __init__(self, features_module: FeatureExtractor,
                 saliency_network: SaliencyNetworkSPADEDynamic,
                 fixation_selection_network: nn.Module,
                 scanpath_network: nn.Module | None,
                 finalizer: Finalizer,
                 semantic_feature_layer_idx: int,
                 num_total_segments: int,
                 downsample: int = 1,
                 readout_factor: int = 4):
        super().__init__()
        self.features = features_module
        self.saliency_network = saliency_network
        self.scanpath_network = scanpath_network
        self.fixation_selection_network = fixation_selection_network
        self.finalizer = finalizer
        self.semantic_feature_layer_idx = semantic_feature_layer_idx
        self.num_total_segments = num_total_segments
        self.downsample = downsample
        self.readout_factor = readout_factor

        # Ensure the backbone is frozen and in evaluation mode
        for p in self.features.parameters():
            p.requires_grad = False
        self.features.eval()

    def _create_painted_semantic_map(self, F_semantic_features, raw_sam_pixel_segmap):
        """
        Creates the dynamic semantic map using features from the DenseNet backbone.
        This is the non-vectorized (per-image loop) version from your script.
        """
        B, C_feat, H_feat, W_feat = F_semantic_features.shape
        _, H_img, W_img = raw_sam_pixel_segmap.shape
        device = F_semantic_features.device
        dtype = F_semantic_features.dtype
        painted_map_batch = torch.zeros(B, C_feat, H_img, W_img, device=device, dtype=dtype)
        
        for b_idx in range(B):
            img_sam_pixel_segmap = raw_sam_pixel_segmap[b_idx]
            segmap_at_feat_res = F.interpolate(
                img_sam_pixel_segmap.unsqueeze(0).unsqueeze(0).float(),
                size=(H_feat, W_feat),
                mode='nearest'
            ).squeeze(0).squeeze(0).long()
            
            flat_features = F_semantic_features[b_idx].permute(1, 2, 0).reshape(H_feat * W_feat, C_feat)
            flat_sam_ids = torch.clamp(segmap_at_feat_res.reshape(H_feat * W_feat), 0, self.num_total_segments - 1)
            
            segment_avg_features = scatter_mean(
                src=flat_features, index=flat_sam_ids, dim=0,
                dim_size=self.num_total_segments
            )
            segment_avg_features = torch.nan_to_num(segment_avg_features, nan=0.0)
            
            clamped_pixel_sam_ids = torch.clamp(img_sam_pixel_segmap.long(), 0, self.num_total_segments - 1)
            painted_map_batch[b_idx] = segment_avg_features[clamped_pixel_sam_ids].permute(2, 0, 1)
            
        return painted_map_batch

    def forward(self, image, centerbias, segmentation_mask, **kwargs):
        if segmentation_mask is None:
            raise ValueError(f"{self.__class__.__name__} requires 'segmentation_mask'.")

        # 1. Pre-process and extract features
        img_for_features = F.interpolate(image, scale_factor=1.0/self.downsample, mode='bilinear', align_corners=False) if self.downsample != 1 else image
        
        with torch.no_grad():
            extracted_feature_maps = self.features(img_for_features)
        
        # 2. Prepare features for main saliency path and semantic path
        readout_h = math.ceil(image.shape[2] / self.downsample / self.readout_factor)
        readout_w = math.ceil(image.shape[3] / self.downsample / self.readout_factor)
        processed_features_list = [F.interpolate(f, size=(readout_h, readout_w), mode='bilinear', align_corners=False) for f in extracted_feature_maps]
        concatenated_features = torch.cat(processed_features_list, dim=1)
        
        # Select the deep feature map for semantic modulation
        semantic_feature_map = extracted_feature_maps[self.semantic_feature_layer_idx]
        
        # 3. Create the dynamic semantic map
        painted_map = self._create_painted_semantic_map(semantic_feature_map, segmentation_mask)
        
        # 4. Saliency Head
        saliency_output = self.saliency_network(concatenated_features, painted_map)
        
        # 5. Scanpath Head (Handles both cases)
        x_hist, y_hist = kwargs.get('x_hist'), kwargs.get('y_hist')

        if self.scanpath_network is not None and x_hist is not None and y_hist is not None and x_hist.numel() > 0:
            scanpath_features_encoded = encode_scanpath_features(x_hist, y_hist, size=image.shape[2:])
            scanpath_features_resized = F.interpolate(scanpath_features_encoded, size=(readout_h, readout_w), mode='bilinear', align_corners=False)
            scanpath_out = self.scanpath_network(scanpath_features_resized)
        else:
            # SHIM: For spatial-only stages, create a zero tensor
            B, _, H, W = saliency_output.shape
            scanpath_out = torch.zeros(B, 16, H, W, device=saliency_output.device) # Assuming 16 output channels for scanpath

        # 6. Fixation Selection (receives both streams)
        final_readout = self.fixation_selection_network((saliency_output, scanpath_out))
        
        # 7. Finalizer
        log_density = self.finalizer(final_readout, centerbias)
        return log_density

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self.features, 'eval'):
            self.features.eval()

# =============================================================================
# 2. MODEL BUILDER FUNCTION
# =============================================================================

def _build_densenet_backbone_and_channels():
    """Creates the backbone and probes its output channel dimensions."""
    densenet_base = RGBDenseNet201()
    densenet_feature_nodes = [
        '1.features.denseblock4.denselayer32.norm1',
        '1.features.denseblock4.denselayer32.conv1',
        '1.features.denseblock4.denselayer31.conv2'
    ]
    features_module = FeatureExtractor(densenet_base, densenet_feature_nodes)
    
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224)
        dummy_output = features_module(dummy_input)
        main_path_channels = sum(f.shape[1] for f in dummy_output)
        
    return features_module, main_path_channels, dummy_output

@register_model("deepgaze_spade_v2")
def build(cfg):
    """
    Builds the complete DeepGazeIIIDynamicEmbedding model from a configuration object.
    """
    extra_params = cfg.stage.extra or {}
    
    # 1. Build the backbone and get channel info
    features, main_path_channels, dummy_output = _build_densenet_backbone_and_channels()
    
    # Determine the number of channels for the semantic path
    sem_idx = extra_params.get("densenet_semantic_feature_layer_idx", -1)
    semantic_path_channels = dummy_output[sem_idx].shape[1]
    
    # 2. Build the network heads
    saliency_net = SaliencyNetworkSPADEDynamic(
        input_channels_main_path=main_path_channels,
        semantic_feature_channels_for_spade=semantic_path_channels
    )
    
    scanpath_net = None
    if "scanpath" in cfg.stage.name:
        scanpath_net = build_scanpath_network()
        fixsel_net = build_fixation_selection_network(scanpath_features=16)
    else:
        fixsel_net = build_fixation_selection_network(scanpath_features=0)
        
    finalizer = Finalizer(
        sigma=extra_params.get("finalizer_initial_sigma", 8.0),
        learn_sigma=extra_params.get("finalizer_learn_sigma", True)
    )
    
    # 3. Assemble the final model
    model = DeepgazeSpadeV2(
        features_module=features,
        saliency_network=saliency_net,
        fixation_selection_network=fixsel_net,
        scanpath_network=scanpath_net,
        finalizer=finalizer,
        semantic_feature_layer_idx=sem_idx,
        num_total_segments=extra_params.get("num_total_segments", 16)
    )
    
    return model