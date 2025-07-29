import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from src.registry import register_model
from src.dinov2_backbone import DinoV2Backbone
# --- CORRECTED IMPORTS ---
# Import the standard, NON-SPADE scanpath builder
from src.modules import Finalizer, build_scanpath_network, encode_scanpath_features
# Import the SPADE-enabled saliency and fixation selection networks
from src.models.common.spade_layers import SaliencyNetworkSPADEDynamic, FixationSelectionNetworkSPADEDynamic

try:
    from torch_scatter import scatter_mean
except ImportError:
    raise ImportError("torch_scatter is required for the DinoGazeSpade model.")

logger = logging.getLogger(__name__)

class DinoGazeSpadeV2(nn.Module):
    """
    DinoGazeSpade v2: A hybrid architecture where the saliency and fixation
    selection heads are SPADE-modulated, but the scanpath head is a standard
    network without SPADE.
    """
    def __init__(self,
                 features_module: DinoV2Backbone,
                 saliency_network: SaliencyNetworkSPADEDynamic,
                 fixation_selection_network: FixationSelectionNetworkSPADEDynamic,
                 scanpath_network: nn.Module | None, # Can be a standard nn.Module
                 semantic_feature_layer_idx: int,
                 num_total_segments: int,
                 readout_factor: int,
                 initial_sigma: float,
                 finalizer_learn_sigma: bool,
                 included_fixations: list[int] | None = None):
        super().__init__()
        self.features = features_module
        for param in self.features.parameters():
            param.requires_grad = False
        self.features.eval()

        self.saliency_network = saliency_network
        self.fixation_selection_network = fixation_selection_network
        self.scanpath_network = scanpath_network

        self.semantic_feature_layer_idx = semantic_feature_layer_idx
        self.num_total_segments = num_total_segments
        self.readout_factor = readout_factor
        self.included_fixations = included_fixations

        self.finalizer = Finalizer(
            sigma=initial_sigma,
            learn_sigma=finalizer_learn_sigma,
        )

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
            raise ValueError(f"{self.__class__.__name__} requires 'segmentation_mask' in its input.")

        orig_img_shape_hw = image.shape[2:]
        with torch.no_grad():
            extracted_feature_maps = self.features(image)

        readout_h = math.ceil(orig_img_shape_hw[0] / self.readout_factor)
        readout_w = math.ceil(orig_img_shape_hw[1] / self.readout_factor)
        readout_spatial_shape = (readout_h, readout_w)

        processed_features_list = [
            F.interpolate(feat_map, size=readout_spatial_shape, mode='bilinear', align_corners=False)
            for feat_map in extracted_feature_maps
        ]
        concatenated_backbone_features = torch.cat(processed_features_list, dim=1)

        F_semantic_patches_from_dino = extracted_feature_maps[self.semantic_feature_layer_idx]
        S_painted_map_full_res = self._create_painted_semantic_map_vectorized(F_semantic_patches_from_dino, segmentation_mask)

        # Saliency network IS SPADE-aware: pass two arguments
        saliency_path_output = self.saliency_network(concatenated_backbone_features, S_painted_map_full_res)

        B, _, H, W = saliency_path_output.shape
        device = saliency_path_output.device
        x_hist, y_hist = kwargs.get('x_hist'), kwargs.get('y_hist')

        if self.scanpath_network is not None and x_hist is not None and y_hist is not None and x_hist.numel() > 0:
            scanpath_input_tensor = encode_scanpath_features(x_hist, y_hist, size=orig_img_shape_hw, device=device)
            scanpath_input_tensor = F.interpolate(scanpath_input_tensor, size=(H, W), mode='bilinear', align_corners=False)
            # --- CORRECTED CALL: Standard scanpath network takes ONE argument ---
            scanpath_path_output = self.scanpath_network(scanpath_input_tensor)
        else:
            scanpath_output_channels = 16
            scanpath_path_output = torch.zeros(B, scanpath_output_channels, H, W, device=device)

        combined_input_for_fixsel = (saliency_path_output, scanpath_path_output)
        # Fixation selection network IS SPADE-aware: pass two arguments
        final_readout_before_finalizer = self.fixation_selection_network(combined_input_for_fixsel, S_painted_map_full_res)

        saliency_log_density = self.finalizer(final_readout_before_finalizer, centerbias)
        return saliency_log_density


@register_model("dinogaze_spade_v2")
def build(cfg):
    """Builds the DinoGazeSpade v2 model (hybrid SPADE architecture)."""
    extra = cfg.stage.extra
    logger.info("Building DinoGazeSpade v2 (hybrid-SPADE) with configuration: %s", extra)

    backbone = DinoV2Backbone(
        layers=extra.get("dino_layers_for_main_path", [-3, -2, -1]),
        model_name=extra.get("dino_model_name", "dinov2_vitl14"),
        freeze=True
    )
    main_ch = len(backbone.layers) * backbone.num_channels
    sem_ch = backbone.num_channels

    # Saliency and Fixation Selection networks are SPADE-aware
    saliency_net = SaliencyNetworkSPADEDynamic(main_ch, sem_ch)

    is_scanpath_stage = extra.get("is_scanpath_stage", False)
    if is_scanpath_stage:
        included_fixations = extra.get("included_fixations")
        if not included_fixations:
            raise ValueError("'is_scanpath_stage' is true, but 'included_fixations' is not defined.")
        logger.info("  - Building in SCANPATH mode.")
        # --- CORRECTED BUILD: Use the standard, NON-SPADE scanpath network ---
        scanpath_net = build_scanpath_network()
        scanpath_output_channels = 16
    else:
        logger.info("  - Building in SPATIAL-ONLY mode.")
        scanpath_net = None
        included_fixations = None
        scanpath_output_channels = 0

    fixsel_net = FixationSelectionNetworkSPADEDynamic(
        saliency_channels=1,
        scanpath_channels=scanpath_output_channels,
        semantic_feature_channels_for_spade=sem_ch
    )

    model = DinoGazeSpadeV2(
        features_module=backbone,
        saliency_network=saliency_net,
        fixation_selection_network=fixsel_net,
        scanpath_network=scanpath_net,
        semantic_feature_layer_idx=extra.get("dino_semantic_feature_layer_idx", -1),
        num_total_segments=extra.get("num_total_segments", 64),
        readout_factor=extra.get("dino_patch_size", 14),
        initial_sigma=extra.get("finalizer_initial_sigma", 8.0),
        finalizer_learn_sigma=extra.get("finalizer_learn_sigma", True),
        included_fixations=included_fixations
    )

    if extra.get("freeze_saliency_network", False):
        for param in model.saliency_network.parameters():
            param.requires_grad = False
        logger.info("  - Saliency network is FROZEN as per config.")

    return model