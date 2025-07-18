import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.registry import register_model
from src.dinov2_backbone import DinoV2Backbone
from src.modules import Finalizer, build_scanpath_network, encode_scanpath_features, build_fixation_selection_network
from src.models.common.spade_layers import SaliencyNetworkSPADEDynamic, SPADELayerNormDynamic

try:
    from torch_scatter import scatter_mean
except ImportError:
    raise ImportError("torch_scatter è necessario per il modello DinoGazeSpade.")

class DinoGazeSpade(nn.Module):
    """Modello completo DinoGaze con SPADE dinamico."""
    def __init__(self, features_module, saliency_network, fixation_selection_network, **kwargs):
        super().__init__()
        self.features = features_module
        for param in self.features.parameters():
            param.requires_grad = False
        self.features.eval()
        
        self.saliency_network = saliency_network
        self.fixation_selection_network = fixation_selection_network
        self.scanpath_network = kwargs.get("scanpath_network")
        self.semantic_feature_layer_idx = kwargs.get("semantic_feature_layer_idx")
        self.num_total_segments = kwargs.get("num_total_segments")
        self.readout_factor = kwargs.get("readout_factor", 7)
        self.finalizer = Finalizer(
            sigma=kwargs.get("initial_sigma", 8.0),
            learn_sigma=kwargs.get("finalizer_learn_sigma", True),
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

    def forward(self, image, centerbias, scanpath_features=None, x_hist=None, y_hist=None, durations=None, **kwargs):
            """
            Processes an input image and scanpath history to produce a saliency log-density map.

            This robust version ensures architectural consistency across all training stages
            (spatial-only and scanpath) by always providing a valid tensor for the scanpath
            path, enabling the use of a consistent fixation selection network.
            """
            segmentation_mask = kwargs.get('segmentation_mask', None)
            if segmentation_mask is None:
                raise ValueError(f"{self.__class__.__name__} requires 'segmentation_mask' in its input.")

            # --- 1. Backbone Feature Extraction ---
            orig_img_shape_hw = image.shape[2:]
            with torch.no_grad():
                extracted_feature_maps = self.features(image)

            # --- 2. Prepare Features and Semantic Map ---
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

            # --- 3. Saliency Head Processing (SPADE-modulated) ---
            saliency_path_output = self.saliency_network(concatenated_backbone_features, S_painted_map_full_res)
            
            # --- 4. Unified Scanpath Head Processing (SPADE-modulated) ---
            # This block now handles all cases: spatial stages, first fixation, and subsequent fixations.
            
            # Get shape and device from a tensor that is always present
            B, _, H, W = saliency_path_output.shape
            device = saliency_path_output.device
            
            # The output of the scanpath network is hardcoded to 16 channels. This should be a class property
            # or derived from the network itself, but for now, this is consistent with the architecture.
            scanpath_output_channels = 16 

            # Check if we are in a scanpath stage AND have a valid history to process
            if self.scanpath_network is not None and x_hist is not None and y_hist is not None and x_hist.numel() > 0:
                # History exists: encode it and pass it through the scanpath network
                scanpath_input_tensor = encode_scanpath_features(
                    x_hist, y_hist,
                    size=orig_img_shape_hw,
                    device=device
                )
                scanpath_input_tensor = F.interpolate(
                    scanpath_input_tensor, 
                    size=(H, W), # Use the determined readout spatial shape
                    mode='bilinear', 
                    align_corners=False
                )
                # Pass both the scanpath features and the semantic map for SPADE modulation
                scanpath_path_output = self.scanpath_network(scanpath_input_tensor)

            else:
                # SHIM: Handles spatial-only stages (where self.scanpath_network is None) OR
                # the first fixation of a scanpath sequence (where x_hist is empty).
                # Create a zero-tensor with the expected shape for the scanpath path.
                scanpath_path_output = torch.zeros(B, scanpath_output_channels, H, W, device=device)
            
            # --- 5. Final Fixation Selection (SPADE-modulated) ---
            # The input is now always a tuple of two valid tensors, making the architecture consistent.
            combined_input_for_fixsel = (saliency_path_output, scanpath_path_output)
            
            # The fixation selection network also receives the semantic map for its own SPADE layers.
            final_readout_before_finalizer = self.fixation_selection_network(combined_input_for_fixsel, S_painted_map_full_res)
            
            # --- 6. Finalizer and Log-Density Calculation ---
            saliency_log_density = self.finalizer(final_readout_before_finalizer, centerbias)
            
            return saliency_log_density


# --- Funzione Builder ---

@register_model("dinogaze_spade_v1")
def build(cfg):
    """Builds the DinoGazeSpade model with SPADE dynamic layers."""
    extra = cfg.stage.extra
    
    backbone = DinoV2Backbone(
        layers=extra.get("dino_layers", [-3, -2, -1]),
        model_name=extra.get("dino_model_name", "dinov2_vitl14"),
        freeze=True
    )
    main_ch = len(backbone.layers) * backbone.num_channels
    sem_ch = backbone.num_channels
    
    saliency_net = SaliencyNetworkSPADEDynamic(main_ch, sem_ch)
    
    scanpath_net = None
    if "scanpath" in cfg.stage.name:
        scanpath_net = build_scanpath_network()
        fixsel_net = build_fixation_selection_network(scanpath_features=16)
    else:
        fixsel_net = build_fixation_selection_network(scanpath_features=0)
    
    model = DinoGazeSpade(
        features_module=backbone,
        saliency_network=saliency_net,
        fixation_selection_network=fixsel_net,
        scanpath_network=scanpath_net,
        semantic_feature_layer_idx=extra.get("semantic_feature_layer_idx", -1),
        num_total_segments=extra.get("num_total_segments", 64),
        readout_factor=14, # o dalla config
        initial_sigma=extra.get("initial_sigma", 8.0),
        finalizer_learn_sigma=extra.get("learn_sigma", True)
    )
    
    if "frozen" in cfg.stage.name and hasattr(model, "saliency_network"):
        for param in model.saliency_network.parameters():
            param.requires_grad = False
            
    return model