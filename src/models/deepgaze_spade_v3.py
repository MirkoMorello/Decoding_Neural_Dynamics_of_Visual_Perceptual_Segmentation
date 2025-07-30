# src/models/hybrid_spade.py
"""
Model definition and builder for the DeepgazeSpadeHybrid.

This model uses a hybrid feature extraction strategy:
1.  A DenseNet-201 backbone provides the main feature maps for the saliency head.
2.  A DINOv2 backbone provides features used to create a *dynamic* semantic
    map for SPADE modulation.
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
from src.dinov2_backbone import DinoV2Backbone
from src.models.common.spade_layers import (
    SaliencyNetworkSPADEDynamic,
    FixationSelectionNetworkSPADEDynamic 
)

# Import the DenseNet backbone
try:
    from DeepGaze.deepgaze_pytorch.features.densenet import RGBDenseNet201
except ImportError:
    raise ImportError("The 'DeepGaze' library is required.")

# Import torch_scatter, which is essential for this model
try:
    from torch_scatter import scatter_mean
except ImportError:
    raise ImportError("torch_scatter is required for DeepgazeSpadeHybrid.")


# =============================================================================
# 1. COMPLETE MODEL CLASS DEFINITION
# =============================================================================

class DeepgazeSpadeV3(nn.Module):
    """
    A saliency model combining DenseNet features with DINO-based dynamic SPADE.
    This version supports both spatial and scanpath training stages.
    """
    def __init__(self, densenet_features: FeatureExtractor,
                 dino_features: DinoV2Backbone,
                 saliency_network: SaliencyNetworkSPADEDynamic,
                 fixation_selection_network: nn.Module, # This will now be FixationSelectionNetworkSPADEDynamic
                 scanpath_network: nn.Module | None,
                 finalizer: Finalizer,
                 num_total_segments: int,
                 downsample: int = 1,
                 readout_factor: int = 4):
        super().__init__()
        self.densenet_features = densenet_features
        self.dino_features = dino_features
        self.saliency_network = saliency_network
        self.scanpath_network = scanpath_network
        self.fixation_selection_network = fixation_selection_network
        self.finalizer = finalizer
        self.num_total_segments = num_total_segments
        self.downsample = downsample
        self.readout_factor = readout_factor

        # Ensure both backbones are frozen and in evaluation mode
        for p in self.densenet_features.parameters():
            p.requires_grad = False
        for p in self.dino_features.parameters():
            p.requires_grad = False
        self.densenet_features.eval()
        self.dino_features.eval()

    def _create_painted_semantic_map_vectorized(self, F_dino_patches, raw_sam_pixel_segmap):
        B, C_dino, H_p, W_p = F_dino_patches.shape
        _, H_img, W_img = raw_sam_pixel_segmap.shape
        device = F_dino_patches.device
        segmap_at_feat_res = F.interpolate(raw_sam_pixel_segmap.unsqueeze(1).float(), size=(H_p, W_p), mode='nearest').long()
        flat_features = F_dino_patches.permute(0, 2, 3, 1).reshape(-1, C_dino)
        flat_segmap_at_feat_res = segmap_at_feat_res.view(-1)
        batch_idx_tensor = torch.arange(B, device=device, dtype=torch.long).view(B, 1).expand(-1, H_p * W_p).reshape(-1)
        global_segment_ids = batch_idx_tensor * self.num_total_segments + torch.clamp(flat_segmap_at_feat_res, 0, self.num_total_segments - 1)
        total_segments_in_batch = B * self.num_total_segments
        segment_avg_features = scatter_mean(src=flat_features, index=global_segment_ids, dim=0, dim_size=total_segments_in_batch)
        segment_avg_features = torch.nan_to_num(segment_avg_features, nan=0.0)
        flat_pixel_segmap = raw_sam_pixel_segmap.view(B, -1)
        batch_idx_pixel_tensor = torch.arange(B, device=device, dtype=torch.long).view(B, 1).expand(-1, H_img * W_img)
        global_pixel_ids = batch_idx_pixel_tensor.reshape(-1) * self.num_total_segments + torch.clamp(flat_pixel_segmap.view(-1), 0, self.num_total_segments - 1)
        painted_flat = segment_avg_features[global_pixel_ids]
        return painted_flat.view(B, H_img, W_img, C_dino).permute(0, 3, 1, 2)

    def forward(self, image, centerbias, segmentation_mask, **kwargs):
        """
        Corrected forward pass that handles both spatial and scanpath stages.
        """
        if segmentation_mask is None:
            raise ValueError(f"{self.__class__.__name__} requires a 'segmentation_mask'.")

        # 1. Pre-process image if downsampling is needed
        img_for_features = F.interpolate(image, scale_factor=1.0 / self.downsample, mode='bilinear', align_corners=False) if self.downsample != 1 else image

        # 2. Extract features from BOTH backbones
        with torch.no_grad():
            dino_feature_maps = self.dino_features(img_for_features)
            semantic_dino_patches = dino_feature_maps[0]
            densenet_feature_maps = self.densenet_features(img_for_features)

        # 3. Create the dynamic semantic map using DINO features
        painted_map = self._create_painted_semantic_map_vectorized(semantic_dino_patches, segmentation_mask)
        
        # 4. Prepare DenseNet features for the saliency head
        readout_h = math.ceil(image.shape[2] / self.downsample / self.readout_factor)
        readout_w = math.ceil(image.shape[3] / self.downsample / self.readout_factor)
        processed_features_list = [F.interpolate(f, size=(readout_h, readout_w), mode='bilinear', align_corners=False) for f in densenet_feature_maps]
        concatenated_features = torch.cat(processed_features_list, dim=1)
        
        # 5. Saliency Head
        saliency_output = self.saliency_network(concatenated_features, painted_map)

        # 6. Scanpath Head (Handles both cases)
        x_hist = kwargs.get('x_hist')
        y_hist = kwargs.get('y_hist')

        if self.scanpath_network is not None and x_hist is not None and y_hist is not None and x_hist.numel() > 0:
            scanpath_features_encoded = encode_scanpath_features(x_hist, y_hist, size=image.shape[2:])
            scanpath_features_resized = F.interpolate(scanpath_features_encoded, size=(readout_h, readout_w), mode='bilinear', align_corners=False)
            # The scanpath network itself is not SPADE-modulated in this design
            scanpath_out = self.scanpath_network(scanpath_features_resized)
        else:
            # SHIM: For spatial-only stages, create a zero tensor
            B, _, H, W = saliency_output.shape
            device = saliency_output.device
            scanpath_out = torch.zeros(B, 16, H, W, device=device) # Assuming scanpath head outputs 16 channels

        # 7. Final Fixation Selection (SPADE-modulated)
        final_readout = self.fixation_selection_network((saliency_output, scanpath_out), painted_map)
        
        # 8. Finalizer
        log_density = self.finalizer(final_readout, centerbias)
        
        return log_density


# =============================================================================
# 2. MODEL BUILDER FUNCTION
# =============================================================================

@register_model("deepgaze_spade_v3")
def build(cfg):
    """
    Builds the complete DeepgazeSpadeHybrid from a configuration object.
    """
    extra_params = cfg.stage.extra or {}
    
    # 1. Build DenseNet backbone for main features
    densenet_base = RGBDenseNet201()
    densenet_feature_nodes = [
        '1.features.denseblock4.denselayer32.norm1',
        '1.features.denseblock4.denselayer32.conv1',
        '1.features.denseblock4.denselayer31.conv2'
    ]
    densenet_features = FeatureExtractor(densenet_base, densenet_feature_nodes)
    
    # 2. Build DINOv2 backbone for semantic features
    dino_semantic_features = DinoV2Backbone(
        layers=[extra_params.get("dino_semantic_layer_idx", -1)],
        model_name=extra_params.get("dino_model_name", "dinov2_vitb14"),
        patch_size=extra_params.get("dino_patch_size", 14),
        freeze=True
    )
    
    # 3. Probe channel counts
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224)
        densenet_dummy_out = densenet_features(dummy_input)
        dino_dummy_out = dino_semantic_features(dummy_input)
        main_path_channels = sum(f.shape[1] for f in densenet_dummy_out)
        semantic_path_channels = dino_dummy_out[0].shape[1]

    # 4. Build the network heads based on stage type
    saliency_net = SaliencyNetworkSPADEDynamic(
        input_channels_main_path=main_path_channels,
        semantic_feature_channels_for_spade=semantic_path_channels
    )
    
    scanpath_net = None
    if "scanpath" in cfg.stage.name:
        scanpath_net = build_scanpath_network()
        # For scanpath stages, the fixation selection network is also SPADE-modulated
        fixsel_net = FixationSelectionNetworkSPADEDynamic(
            saliency_channels=1,
            scanpath_channels=16,
            semantic_feature_channels_for_spade=semantic_path_channels
        )
    else:
        # For spatial stages, use a simpler fixation selection network
        # Or reuse the SPADE one, as it will get a zero-tensor for the scanpath
        fixsel_net = FixationSelectionNetworkSPADEDynamic(
            saliency_channels=1,
            scanpath_channels=16, # The shim tensor will have 16 channels
            semantic_feature_channels_for_spade=semantic_path_channels
        )
        
    finalizer = Finalizer(
        sigma=extra_params.get("finalizer_initial_sigma", 8.0),
        learn_sigma=extra_params.get("finalizer_learn_sigma", True)
    )
    
    # 5. Assemble the final model
    model = DeepgazeSpadeV3(
        densenet_features=densenet_features,
        dino_features=dino_semantic_features,
        saliency_network=saliency_net,
        scanpath_network=scanpath_net,
        fixation_selection_network=fixsel_net,
        finalizer=finalizer,
        num_total_segments=extra_params.get("num_total_segments", 16)
    )
    
    return model