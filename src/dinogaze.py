# dinogaze.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from collections import OrderedDict
from torch.utils import model_zoo
from .features.densenet import RGBDenseNet201
from .modules import FeatureExtractor, Finalizer, DeepGazeIIIMixture
from .layers import FlexibleScanpathHistoryEncoding
from __future__ import annotations
from typing import Iterable, List, Tuple
from .layers import (
    Conv2dMultiInput,
    LayerNorm,
    LayerNormMultiInput,
    Bias,
    SelfAttention,
)


def build_saliency_network(input_channels, add_sa_head=False):
    """ Builds the saliency prediction head network. """
    layers = OrderedDict()

    if add_sa_head:
        # Add SelfAttention as the first layer
        # NOTE: SelfAttention outputs the same number of channels as input by default.
        layers['sa_head'] = SelfAttention(input_channels, key_channels=input_channels // 8, return_attention=False)
        layers['layernorm_sa_out'] = LayerNorm(input_channels) # Normalize SA output
        layers['softplus_sa_out'] = nn.Softplus() # Add non-linearity after SA

    # Reduced complexity slightly given potentially richer ViT features
    layers['layernorm0'] = LayerNorm(input_channels)
    layers['conv0'] = nn.Conv2d(input_channels, 8, (1, 1), bias=False) # Increased capacity slightly
    layers['bias0'] = Bias(8)
    layers['softplus0'] = nn.Softplus()

    layers['layernorm1'] = LayerNorm(8)
    layers['conv1'] = nn.Conv2d(8, 16, (1, 1), bias=False) # Increased capacity slightly
    layers['bias1'] = Bias(16)
    layers['softplus1'] = nn.Softplus()

    layers['layernorm2'] = LayerNorm(16)
    layers['conv2'] = nn.Conv2d(16, 1, (1, 1), bias=False)
    layers['bias2'] = Bias(1)
    layers['softplus2'] = nn.Softplus() # Ensures non-negative output before final log-likelihood

    return nn.Sequential(layers)


def build_scanpath_network():
    """ Builds the network processing scanpath history. """
    return nn.Sequential(OrderedDict([
        ('encoding0', FlexibleScanpathHistoryEncoding(in_fixations=4, channels_per_fixation=3, out_channels=128, kernel_size=[1, 1], bias=True)),
        ('softplus0', nn.Softplus()),
        ('layernorm1', LayerNorm(128)),
        ('conv1', nn.Conv2d(128, 16, (1, 1), bias=False)), # Output 16 channels
        ('bias1', Bias(16)),
        ('softplus1', nn.Softplus()),
    ]))


    
def build_fixation_selection_network(scanpath_features=16):
    """ Builds the network combining saliency and scanpath features for fixation selection. """
    saliency_channels = 1 # Output of saliency network's core path before combination
    in_channels_list = [saliency_channels, scanpath_features if scanpath_features > 0 else 0]

    return nn.Sequential(OrderedDict([
        ('layernorm0', LayerNormMultiInput(in_channels_list)), # Now gets [1, 0] or [1, 16]
        ('conv0', Conv2dMultiInput(in_channels_list, 128, (1, 1), bias=False)), # Now gets [1, 0] or [1, 16]
        ('bias0', Bias(128)),
        ('softplus0', nn.Softplus()),
        ('layernorm1', LayerNorm(128)),
        ('conv1', nn.Conv2d(128, 16, (1, 1), bias=False)),
        ('bias1', Bias(16)),
        ('softplus1', nn.Softplus()),
        ('conv2', nn.Conv2d(16, 1, (1, 1), bias=False)), # Final output layer
    ]))


# --- YOUR NEW DINOV2-BASED MODEL CLASS ---
class Dinogaze(DeepGazeIIIMixture):
    def __init__(self,
                 dino_model_name: str = "dinov2_vitl14",
                 dino_layers: Iterable[int] | None = None,
                 freeze_dino: bool = True,
                 # SPADE related arguments
                 use_spade_saliency: bool = False,
                 use_spade_scanpath: bool = False, # Add if you plan to use SPADE in scanpath too
                 segmap_channels: int = 1, # e.g., 1 for raw mask, K for one-hot K classes
                 # DeepGazeIII parameters
                 num_mixture_components: int = 1, # Start with 1 component for DINOv2, easier to debug
                 downsample: int = 1, # DINOv2 is already strided
                 readout_factor: int = 14, # From DinoV2Backbone patch_size
                 saliency_map_factor: int = 4,
                 included_fixations: List[int] | None = None,
                 initial_sigma: float = 8.0,
                 # pretrained_weights_path: str | None = None # For loading your own DINO-based pretrained heads
                 **kwargs): # To catch any other args from DeepGazeIIIMixture

        module_logger.info(f"Initializing Dinogaze model with DINOv2: {dino_model_name}")

        actual_backbone = DinoV2Backbone(
            layers=dino_layers,
            model_name=dino_model_name,
            freeze=freeze_dino,
            patch_size=readout_factor # Ensure patch_size matches readout_factor
        )
        C_in_dino = len(actual_backbone.layers) * actual_backbone.num_channels
        module_logger.info(f"DINOv2 feature extractor C_in: {C_in_dino}")

        saliency_nets = []
        scanpath_nets = []
        fixsel_nets = []
        finalizers_list = []

        # For your experiment, you likely want the *same* head architecture for all components if num_mixture_components > 1
        # Or just one component to start.
        for _ in range(num_mixture_components):
            s_net = build_saliency_network(
                C_in_dino,
                add_sa_head=kwargs.get('add_sa_head', False), # Get from args if passed via main script
                use_spade=use_spade_saliency,
                segmap_channels_for_spade=segmap_channels
            )
            # Decide if scanpath is needed for this experiment
            sp_net = None
            fs_net = None
            if included_fixations and len(included_fixations) > 0: # Or some other condition
                sp_net = build_scanpath_network(
                    use_spade=use_spade_scanpath,
                    segmap_channels_for_spade=segmap_channels
                )
                # scanpath_output_features would be e.g., 16 from build_dinov2_scanpath_network
                fs_net = build_fixation_selection_network(scanpath_features=16) # Assuming 16 output channels
            else: # Spatial only model
                # Fixation selection network for spatial-only (no scanpath input)
                fs_net = build_fixation_selection_network(scanpath_features=0)


            saliency_nets.append(s_net)
            scanpath_nets.append(sp_net) # Will be None if not created
            fixsel_nets.append(fs_net)
            finalizers_list.append(Finalizer(sigma=initial_sigma, learn_sigma=True, saliency_map_factor=saliency_map_factor))

        super().__init__(
            features=actual_backbone, # Correctly pass the backbone here
            saliency_networks=saliency_nets,
            scanpath_networks=scanpath_nets,
            fixation_selection_networks=fixsel_nets,
            finalizers=finalizers_list,
            downsample=downsample,
            readout_factor=readout_factor, # This should be 1 if features already outputs at readout scale
            saliency_map_factor=saliency_map_factor,
            included_fixations=included_fixations if included_fixations is not None else []
        )

        # No model_zoo.load_url for deepgaze3.pth here, as that's for DenseNet.
        # if pretrained_weights_path:
        #     module_logger.info(f"Loading custom pretrained weights from: {pretrained_weights_path}")
        #     self.load_state_dict(torch.load(pretrained_weights_path, map_location='cpu'))

    # If your SPADE layers need the segmap, DeepGazeIIIMixture's forward needs adaptation,
    # or your build_dinov2_saliency_network etc. must return custom nn.Modules
    # that handle the segmap input internally.
    # The simplest way for DeepGazeIIIMixture to work is if saliency_network(x) is callable.
    # So, the saliency_network itself needs to be a module that accepts (x, segmap)
    # if it uses SPADE. This means nn.Sequential for SPADE heads is tricky.
    #
    # Let's assume you will make custom nn.Module wrappers for SPADE heads.
    # Example SaliencyHeadSPADE (goes in this file or src.custom_heads.py):
    # class SaliencyHeadSPADE(nn.Module):
    #     def __init__(self, input_channels, use_spade, segmap_channels):
    #         super().__init__()
    #         # Define layers, using SPADELayerNorm or LayerNorm
    #         # self.norm0 = SPADELayerNorm(input_channels, segmap_channels) if use_spade else LayerNorm(input_channels)
    #         # self.conv0 = ...
    #         self.actual_sequential_part = build_dinov2_saliency_network(...) # build_... would create the nn.Sequential
    #         self.use_spade = use_spade # Store this
    #
    #     def forward(self, x, segmap=None): # segmap is optional
    #         # if self.use_spade and segmap is None:
    #         #     raise ValueError("SPADE enabled but no segmap provided to SaliencyHeadSPADE")
    #         #
    #         # How to pass segmap to internal SPADELayerNorms if using nn.Sequential?
    #         # -> This is why nn.Sequential is hard with SPADE.
    #         # -> Each block with SPADE needs its own forward(x, segmap).
    #         #
    #         # For now, DeepGazeIIIMixture calls saliency_network(readout_input)
    #         # So, if saliency_network is one of your build_dinov2_saliency_network outputs
    #         # (which is an nn.Sequential), it CANNOT take a segmap.
    #
    # This is a key architectural point. The DeepGazeIIIMixture expects saliency_networks
    # to be callable as `saliency_network(features_output)`.
    # To use SPADE, you either:
    # 1. Modify DeepGazeIIIMixture.forward to also pass `segmap`.
    # 2. Make each `saliency_network` in the list a custom module that somehow gets `segmap`
    #    (e.g., via a forward hook, or if `segmap` is part of `readout_input` tuple).
    #    This is complex.
    # 3. **Simplest for now**: The `Dinogaze` class itself overrides `forward` from `DeepGazeIIIMixture`
    #    to handle the `segmap`.
    # Let's try overriding forward in Dinogaze:

    def forward(self, x_img, centerbias, x_hist=None, y_hist=None, durations=None, segmentation_mask=None, **kwargs): # Added segmentation_mask
        orig_shape = x_img.shape
        # Feature extraction
        img_features_scaled = F.interpolate(x_img, scale_factor=1 / self.downsample, recompute_scale_factor=False)
        # self.features is DinoV2Backbone instance
        extracted_block_features = self.features(img_features_scaled) # List of (B, C, Hf, Wf)

        readout_shape = [
            math.ceil(orig_shape[2] / self.downsample / self.readout_factor),
            math.ceil(orig_shape[3] / self.downsample / self.readout_factor)
        ]
        # Interpolate each feature block to the target readout_shape
        processed_features = [F.interpolate(item, readout_shape) for item in extracted_block_features]
        concatenated_features = torch.cat(processed_features, dim=1) # (B, C_total, Hr, Wr)

        predictions = []
        # Iterate through the mixture components (saliency_networks, scanpath_networks, etc.)
        for s_net, sp_net, fs_net, fin_module in zip(
            self.saliency_networks, self.scanpath_networks, self.fixation_selection_networks, self.finalizers
        ):
            # Saliency branch
            # HERE: s_net needs to accept (concatenated_features, segmentation_mask) if it uses SPADE
            # This requires s_net to NOT be a simple nn.Sequential if SPADE is internal to it.
            # OR, build_dinov2_saliency_network returns a custom module.
            # For now, let's assume s_net is a custom module:
            if hasattr(s_net, 'use_spade') and s_net.use_spade: # Hypothetical attribute
                 saliency_readout = s_net(concatenated_features, segmentation_mask)
            else:
                 saliency_readout = s_net(concatenated_features)


            # Scanpath branch
            scanpath_readout = None
            if sp_net is not None and x_hist is not None and y_hist is not None :
                scanpath_hist_features = encode_scanpath_features(x_hist, y_hist, size=(orig_shape[2], orig_shape[3]), device=x_img.device)
                scanpath_hist_features_scaled = F.interpolate(scanpath_hist_features, readout_shape)
                # Similar logic for sp_net and segmask
                if hasattr(sp_net, 'use_spade') and sp_net.use_spade:
                    scanpath_readout = sp_net(scanpath_hist_features_scaled, segmentation_mask)
                else:
                    scanpath_readout = sp_net(scanpath_hist_features_scaled)
            
            # Fixation selection
            # fs_net combines saliency_readout and scanpath_readout
            # If fs_net also uses SPADE on its inputs, it needs modification.
            # Assuming fs_net does not use SPADE for now for simplicity.
            combined_input_for_fixsel = (saliency_readout, scanpath_readout)
            final_component_readout = fs_net(combined_input_for_fixsel)

            # Finalizer
            prediction = fin_module(final_component_readout, centerbias)
            predictions.append(prediction[:, np.newaxis, :, :]) # Add mixture dim

        if not predictions:
            # This should not happen if num_mixture_components >= 1
            raise ValueError("No predictions generated by mixture components.")

        predictions_cat = torch.cat(predictions, dim=1)
        
        if self.num_mixture_components > 1:
            log_predictions = predictions_cat - math.log(self.num_mixture_components)
            final_prediction = log_predictions.logsumexp(dim=1) # Remove keepdim=True if shape is already (B,H,W)
        else:
            final_prediction = predictions_cat.squeeze(1) # Remove mixture dim if only 1 component

        return final_prediction