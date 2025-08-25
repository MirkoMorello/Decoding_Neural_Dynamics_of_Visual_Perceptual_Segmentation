"""
Model definition and builder for DeepGazeIII with Dynamic SPADE on DenseNet features.

This model uses a frozen DenseNet-201 backbone for two purposes:
1.  Extracting multi-level features for the main saliency pathway.
2.  Extracting a deep feature from a specific layer to create a dynamic
    semantic map for SPADE modulation via feature painting.
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

try:
    from DeepGaze.deepgaze_pytorch.features.densenet import RGBDenseNet201
except ImportError:
    raise ImportError("The 'DeepGaze' library is required for this model.")

try:
    from torch_scatter import scatter_mean
except ImportError:
    raise ImportError("torch_scatter is required for the dynamic SPADE model.")

class DeepgazeSpadeV2(nn.Module):
    def __init__(self, features_module: FeatureExtractor,
                 saliency_network: SaliencyNetworkSPADEDynamic,
                 fixation_selection_network: nn.Module,
                 scanpath_network: nn.Module | None,
                 semantic_feature_layer_idx: int,
                 num_total_segments: int,
                 downsample: float,
                 readout_factor: int,
                 saliency_map_factor: int,
                 initial_sigma: float,
                 finalizer_learn_sigma: bool,
                 included_fixations: list[int] | None):
        super().__init__()
        self.features = features_module
        self.saliency_network = saliency_network
        self.scanpath_network = scanpath_network
        self.fixation_selection_network = fixation_selection_network
        self.semantic_feature_layer_idx = semantic_feature_layer_idx
        self.num_total_segments = num_total_segments
        self.downsample = downsample
        self.readout_factor = readout_factor
        self.included_fixations = included_fixations

        self.finalizer = Finalizer(
            sigma=initial_sigma,
            learn_sigma=finalizer_learn_sigma,
            saliency_map_factor=saliency_map_factor
        )

        for p in self.features.parameters():
            p.requires_grad = False
        self.features.eval()

    def _create_painted_semantic_map_vectorized(self, F_semantic_patches, raw_sam_pixel_segmap):
        B, C_dino, H_p, W_p = F_semantic_patches.shape
        _, H_img, W_img = raw_sam_pixel_segmap.shape
        device = F_semantic_patches.device
        segmap_at_feat_res = F.interpolate(raw_sam_pixel_segmap.unsqueeze(1).float(), size=(H_p, W_p), mode='nearest').long()
        flat_features = F_semantic_patches.permute(0, 2, 3, 1).reshape(-1, C_dino)
        flat_segmap_at_feat_res = segmap_at_feat_res.view(-1)
        batch_idx_tensor = torch.arange(B, device=device, dtype=torch.long).view(B, 1).expand(-1, H_p * W_p).reshape(-1)
        global_segment_ids = batch_idx_tensor * self.num_total_segments + torch.clamp(flat_segmap_at_feat_res, 0, self.num_total_segments - 1)
        segment_avg_features = scatter_mean(src=flat_features, index=global_segment_ids, dim=0, dim_size=B * self.num_total_segments)
        segment_avg_features = torch.nan_to_num(segment_avg_features, nan=0.0)
        flat_pixel_segmap = raw_sam_pixel_segmap.view(B, -1)
        batch_idx_pixel_tensor = torch.arange(B, device=device, dtype=torch.long).view(B, 1).expand(-1, H_img * W_img)
        global_pixel_ids = batch_idx_pixel_tensor.reshape(-1) * self.num_total_segments + torch.clamp(flat_pixel_segmap.view(-1), 0, self.num_total_segments - 1)
        painted_flat = segment_avg_features[global_pixel_ids]
        return painted_flat.view(B, H_img, W_img, C_dino).permute(0, 3, 1, 2)

    def forward(self, image, centerbias, **kwargs):
        segmentation_mask = kwargs.get('segmentation_mask')
        if segmentation_mask is None:
            raise ValueError(f"{self.__class__.__name__} requires 'segmentation_mask'.")

        orig_shape_hw = image.shape[2:]
        img_for_features = F.interpolate(image, scale_factor=1.0/self.downsample, mode='bilinear', align_corners=False) if self.downsample != 1 else image
        
        with torch.no_grad():
            extracted_feature_maps = self.features(img_for_features)
        
        readout_h = math.ceil(orig_shape_hw[0] / self.downsample / self.readout_factor)
        readout_w = math.ceil(orig_shape_hw[1] / self.downsample / self.readout_factor)
        readout_shape = (readout_h, readout_w)
        
        processed_features_list = [F.interpolate(f, size=readout_shape, mode='bilinear', align_corners=False) for f in extracted_feature_maps]
        concatenated_features = torch.cat(processed_features_list, dim=1)
        
        semantic_feature_map = extracted_feature_maps[self.semantic_feature_layer_idx]
        
        painted_map = self._create_painted_semantic_map_vectorized(semantic_feature_map, segmentation_mask)
        
        saliency_output = self.saliency_network(concatenated_features, painted_map)
        
        if self.scanpath_network is not None:
            x_hist, y_hist = kwargs.get('x_hist'), kwargs.get('y_hist')
            if x_hist is None or y_hist is None or x_hist.numel() == 0:
                B, _, H, W = saliency_output.shape
                scanpath_out = torch.zeros(B, 16, H, W, device=saliency_output.device)
            else:
                scanpath_features_encoded = encode_scanpath_features(x_hist, y_hist, size=orig_shape_hw)
                scanpath_features_resized = F.interpolate(scanpath_features_encoded, size=readout_shape, mode='bilinear', align_corners=False)
                scanpath_out = self.scanpath_network(scanpath_features_resized)
            final_readout = self.fixation_selection_network((saliency_output, scanpath_out))
        else:
            final_readout = self.fixation_selection_network((saliency_output,))
        
        return self.finalizer(final_readout, centerbias)

    def train(self, mode=True):
        super().train(mode)
        self.features.eval()

@register_model("densenet_spade_dynamic")
def build(cfg):
    extra = cfg.stage.extra
    
    densenet_base = RGBDenseNet201()
    densenet_feature_nodes = [
        '1.features.denseblock4.denselayer32.norm1',
        '1.features.denseblock4.denselayer32.conv1',
        '1.features.denseblock4.denselayer31.conv2'
    ]
    features_module = FeatureExtractor(densenet_base, densenet_feature_nodes)
    
    main_path_channels = 2048
    semantic_path_channels = 128
    sem_idx = extra.get("densenet_semantic_feature_layer_idx", -1)

    saliency_net = SaliencyNetworkSPADEDynamic(
        input_channels_main_path=main_path_channels,
        semantic_feature_channels_for_spade=semantic_path_channels
    )
    
    if extra.get("is_scanpath_stage", False):
        scanpath_net = build_scanpath_network()
        fixsel_net = build_fixation_selection_network(scanpath_features=16)
        included_fixations = extra.get("included_fixations")
        if not included_fixations:
            raise ValueError("'is_scanpath_stage' requires 'included_fixations'.")
    else:
        scanpath_net = None
        fixsel_net = build_fixation_selection_network(scanpath_features=0)
        included_fixations = None
        
    model = DeepgazeSpadeV2(
        features_module=features_module,
        saliency_network=saliency_net,
        fixation_selection_network=fixsel_net,
        scanpath_network=scanpath_net,
        semantic_feature_layer_idx=sem_idx,
        num_total_segments=extra.get("num_total_segments", 64),
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