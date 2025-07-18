"""Contiene le definizioni dei layer SPADE riutilizzabili."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers import Bias, FlexibleScanpathHistoryEncoding # Assumiamo che Bias sia in src/layers

try:
    from torch_scatter import scatter_mean
except ImportError:
    print("Attention: torch_scatter not found. Necessary for SPADELayerNormDynamic.")
    scatter_mean = None

# --- Version SPADE with Learned Embedding (for masks with integer IDs) ---

class SPADELayerNorm(nn.Module):
    """LayerNorm modulated by a segmentation map with learned embeddings."""
    def __init__(self, norm_features, segmap_input_channels, hidden_mlp_channels=128, eps=1e-12, kernel_size=3):
        super().__init__()
        self.norm_features = norm_features
        self.segmap_input_channels = segmap_input_channels
        padding = kernel_size // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(self.segmap_input_channels, hidden_mlp_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True)
        )
        self.mlp_gamma = nn.Conv2d(hidden_mlp_channels, norm_features, kernel_size=kernel_size, padding=padding)
        self.mlp_beta = nn.Conv2d(hidden_mlp_channels, norm_features, kernel_size=kernel_size, padding=padding)

    def forward(self, x, segmap_embedded):   
        normalized_x = F.layer_norm(x, (self.norm_features, x.size(2), x.size(3)), eps=1e-12)
        segmap_resized = F.interpolate(segmap_embedded, size=x.size()[2:], mode='nearest')
        shared_features = self.mlp_shared(segmap_resized)
        gamma = self.mlp_gamma(shared_features)
        beta = self.mlp_beta(shared_features)
        return normalized_x * (1 + gamma) + beta

class SaliencyNetworkSPADE(nn.Module):
    """Saliency Network with SPADE using learned embeddings."""
    def __init__(self, input_channels, num_total_segments=17, seg_embedding_dim=64):
        super().__init__()
        self.seg_embedder = nn.Embedding(num_total_segments, seg_embedding_dim)
        self.spade_ln0 = SPADELayerNorm(input_channels, seg_embedding_dim)
        self.conv0 = nn.Conv2d(input_channels, 8, 1, bias=False)
        self.bias0 = Bias(8); self.softplus0 = nn.Softplus()
        self.spade_ln1 = SPADELayerNorm(8, seg_embedding_dim)
        self.conv1 = nn.Conv2d(8, 16, 1, bias=False)
        self.bias1 = Bias(16); self.softplus1 = nn.Softplus()
        self.spade_ln2 = SPADELayerNorm(16, seg_embedding_dim)
        self.conv2 = nn.Conv2d(16, 1, 1, bias=False)
        self.bias2 = Bias(1); self.softplus2 = nn.Softplus()

    def forward(self, x, raw_segmap_long):
        B, H, W = raw_segmap_long.shape
        clamped = torch.clamp(raw_segmap_long.view(B, -1), 0, self.seg_embedder.num_embeddings - 1)
        embedded = self.seg_embedder(clamped).view(B, H, W, -1).permute(0, 3, 1, 2)
        h = self.spade_ln0(x, embedded)
        h = self.conv0(h); h = self.bias0(h); h = self.softplus0(h)
        h = self.spade_ln1(h, embedded)
        h = self.conv1(h); h = self.bias1(h); h = self.softplus1(h)
        h = self.spade_ln2(h, embedded)
        h = self.conv2(h); h = self.bias2(h); h = self.softplus2(h)
        return h

# ---  SPADE with dynamic Embedding (Calculated with the feature of the backbone) ---

class SPADELayerNormDynamic(nn.Module):
    """LayerNorm modulated from a dynamically calculated semantic map."""
    def __init__(self, norm_features, semantic_feature_channels, hidden_mlp_channels=128, eps=1e-12, kernel_size=3):
        super().__init__()
        self.norm_features = norm_features
        self.eps = eps
        self.semantic_feature_channels = semantic_feature_channels
        padding = kernel_size // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(self.semantic_feature_channels, hidden_mlp_channels, kernel_size=kernel_size, padding=padding, bias=True),
            nn.ReLU(inplace=True)
        )
        self.mlp_gamma = nn.Conv2d(hidden_mlp_channels, norm_features, kernel_size=kernel_size, padding=padding, bias=True)
        self.mlp_beta = nn.Conv2d(hidden_mlp_channels, norm_features, kernel_size=kernel_size, padding=padding, bias=True)

    def forward(self, x, painted_semantic_map):
        normalized_shape = (self.norm_features, x.size(2), x.size(3))
        normalized_x = F.layer_norm(x, normalized_shape, weight=None, bias=None, eps=self.eps)
        semantic_map_resized = F.interpolate(painted_semantic_map, size=x.size()[2:], mode='bilinear', align_corners=False)
        shared_features = self.mlp_shared(semantic_map_resized)
        gamma_map = self.mlp_gamma(shared_features)
        beta_map = self.mlp_beta(shared_features)
        out = normalized_x * (1 + gamma_map) + beta_map
        return out

class SaliencyNetworkSPADEDynamic(nn.Module):
    """Saliency Network with SPADE using dynamically calculated semantic maps."""
    def __init__(self, input_channels_main_path, semantic_feature_channels_for_spade):
        super().__init__()
        self.spade_ln0 = SPADELayerNormDynamic(input_channels_main_path, semantic_feature_channels_for_spade)
        self.conv0 = nn.Conv2d(input_channels_main_path, 8, (1, 1), bias=False)
        self.bias0 = Bias(8); self.softplus0 = nn.Softplus()
        self.spade_ln1 = SPADELayerNormDynamic(8, semantic_feature_channels_for_spade)
        self.conv1 = nn.Conv2d(8, 16, (1, 1), bias=False)
        self.bias1 = Bias(16); self.softplus1 = nn.Softplus()
        self.spade_ln2 = SPADELayerNormDynamic(16, semantic_feature_channels_for_spade)
        self.conv2 = nn.Conv2d(16, 1, (1, 1), bias=False)
        self.bias2 = Bias(1); self.softplus2 = nn.Softplus()

    def forward(self, x_main_path, painted_semantic_map):
        h = x_main_path
        h = self.spade_ln0(h, painted_semantic_map)
        h = self.conv0(h); h = self.bias0(h); h = self.softplus0(h)
        h = self.spade_ln1(h, painted_semantic_map)
        h = self.conv1(h); h = self.bias1(h); h = self.softplus1(h)
        h = self.spade_ln2(h, painted_semantic_map)
        h = self.conv2(h); h = self.bias2(h); h = self.softplus2(h)
        return h
    

class FixationSelectionNetworkSPADEDynamic(nn.Module):
    """
    Builds the final network that combines saliency and scanpath features,
    and then applies SPADE modulation to the merged representation.
    """
    def __init__(self, saliency_channels, scanpath_channels, semantic_feature_channels_for_spade):
        super().__init__()
        # Total channels after concatenating the two input streams
        concatenated_channels = saliency_channels + scanpath_channels
        
        # First, a convolution to merge the concatenated features and project them.
        # This allows the model to learn a good initial combination.
        self.conv_merge = nn.Conv2d(concatenated_channels, 128, (1, 1), bias=False)
        self.bias_merge = Bias(128)
        self.softplus_merge = nn.Softplus()

        # Now, apply SPADE modulation to the merged representation
        self.spade_ln1 = SPADELayerNormDynamic(128, semantic_feature_channels_for_spade)
        
        # Subsequent layers process the modulated features
        self.conv1 = nn.Conv2d(128, 16, (1, 1), bias=False)
        self.bias1 = Bias(16)
        self.softplus1 = nn.Softplus()
        
        self.spade_ln2 = SPADELayerNormDynamic(16, semantic_feature_channels_for_spade)
        
        self.conv2 = nn.Conv2d(16, 1, (1, 1), bias=False) # Final output layer

    def forward(self, input_tuple, painted_semantic_map):
        """
        Args:
            input_tuple (tuple): A tuple containing (saliency_output, scanpath_output).
            painted_semantic_map (torch.Tensor): The semantic map for SPADE modulation.
        """
        saliency_out, scanpath_out = input_tuple
        
        # Concatenate the two feature streams along the channel dimension
        h = torch.cat([saliency_out, scanpath_out], dim=1)
        
        # Process through the network
        h = self.conv_merge(h)
        h = self.bias_merge(h)
        h = self.softplus_merge(h)
        
        h = self.spade_ln1(h, painted_semantic_map)
        
        h = self.conv1(h)
        h = self.bias1(h)
        h = self.softplus1(h)

        h = self.spade_ln2(h, painted_semantic_map)

        h = self.conv2(h)
        
        return h



class ScanpathNetworkSPADEDynamic(nn.Module):
    """
    Encodes the (dx, dy, dist) scan-path tensor and modulates each stage
    with the painted semantic map via SPADE-style LayerNorm.
    Produces 16 feature channels expected by the fixation-selection head.
    """
    def __init__(
        self,
        in_fixations: int = 4,
        channels_per_fixation: int = 3,
        semantic_feature_channels: int = 1024,   # 1024 for DINOv2 ViT-L/14
        hidden_mlp_channels: int = 128,
    ):
        super().__init__()

        # 0️⃣  initial encoding of scan-path history
        self.encoding = FlexibleScanpathHistoryEncoding(
            in_fixations=in_fixations,
            channels_per_fixation=channels_per_fixation,
            out_channels=hidden_mlp_channels,
            kernel_size=(1, 1),
            bias=True,
        )
        self.softplus_enc = nn.Softplus()

        # 1️⃣  SPADE block 1
        self.spade0 = SPADELayerNormDynamic(
            norm_features=hidden_mlp_channels,
            semantic_feature_channels=semantic_feature_channels,
        )
        self.conv0   = nn.Conv2d(hidden_mlp_channels, 64, 1, bias=False)
        self.bias0   = Bias(64);  self.softplus0 = nn.Softplus()

        # 2️⃣  SPADE block 2
        self.spade1 = SPADELayerNormDynamic(64, semantic_feature_channels)
        self.conv1   = nn.Conv2d(64, 32, 1, bias=False)
        self.bias1   = Bias(32);  self.softplus1 = nn.Softplus()

        # 3️⃣  SPADE block 3
        self.spade2 = SPADELayerNormDynamic(32, semantic_feature_channels)
        self.conv2   = nn.Conv2d(32, 16, 1, bias=False)   # <- 16 channels
        self.bias2   = Bias(16);  self.softplus2 = nn.Softplus()

    def forward(self, scanpath_tensor, painted_semantic_map):
        """
        scanpath_tensor        (B, 3·N_fix, H, W)
        painted_semantic_map   (B, C_sem,  H, W) – same spatial dims
        """
        h = self.encoding(scanpath_tensor)
        h = self.softplus_enc(h)

        h = self.spade0(h, painted_semantic_map)
        h = self.conv0(h); h = self.bias0(h); h = self.softplus0(h)

        h = self.spade1(h, painted_semantic_map)
        h = self.conv1(h); h = self.bias1(h); h = self.softplus1(h)

        h = self.spade2(h, painted_semantic_map)
        h = self.conv2(h); h = self.bias2(h); h = self.softplus2(h)
        return h



# ----------------------------------------------------------------------
# Convenience factory
# ----------------------------------------------------------------------
def build_scanpath_network_spade_dynamic(
    *,
    in_fixations: int                = 4,
    channels_per_fixation: int       = 3,
    semantic_feature_channels: int   = None,
    backbone=None,                   # pass DinoV2Backbone to auto-derive dim
):
    if semantic_feature_channels is None:
        assert backbone is not None, (
            "Provide either `semantic_feature_channels` or the backbone."
        )
        semantic_feature_channels = backbone.num_channels  # 768/1024/1536
    return ScanpathNetworkSPADEDynamic(
        in_fixations=in_fixations,
        channels_per_fixation=channels_per_fixation,
        semantic_feature_channels=semantic_feature_channels,
    )