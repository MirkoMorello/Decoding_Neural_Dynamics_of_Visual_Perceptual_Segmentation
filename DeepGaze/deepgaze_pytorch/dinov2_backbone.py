# -*- coding: utf-8 -*-
"""dinov2_backbone.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Drop‑in replacement for the `RGBDenseNet201` feature extractor used in
DeepGaze III.  It wraps the *DINO v2 ViT‑B/14* backbone and returns a list
of spatial feature‑maps so that the rest of the DeepGaze code base can
stay unchanged.

**Key design choices**
---------------------
1. **Frozen backbone** – the transformer weights are kept fixed exactly
   like in the original DenseNet setup.  Only the read‑out head and the
   downstream saliency / scan‑path modules are trained.
2. **Multi‑layer hooks** – you can specify which transformer blocks you
   want to tap.  By default we use the last three blocks (`[-3, -2, -1]`)
   because that empirically gives a nice semantic / spatial mix.
3. **Token → map reshape** – class tokens are discarded and the remaining
   patch tokens are reshaped into a `(B, C, H_patch, W_patch)` tensor that
   matches what the CNN backbone delivered before.
4. **Automatic padding** – the ViT requires the spatial dimensions to be
   divisible by the patch‑size (14).  We therefore *zero‑pad* on the fly
   and crop back afterwards; this keeps everything fully differentiable.

Usage example
-------------
```python
from dinov2_backbone import DinoV2Backbone
from deepgaze_pytorch.modules import DeepGazeIII
from deepgaze_pytorch.layers import build_saliency_network, build_fixation_selection_network

# === feature extractor ===
features = DinoV2Backbone(layers=[-3, -2, -1])          # 3 layers × 768 ch

# === channel count ===
C_in = 768 * 3                                           # 2304

model = DeepGazeIII(
    features=features,
    saliency_network=build_saliency_network(C_in),
    scanpath_network=None,
    fixation_selection_network=build_fixation_selection_network(scanpath_features=0),
    downsample=1,               # keep full resolution, ViT already strided
    readout_factor=14,          # patch‑stride of ViT‑B/14
    saliency_map_factor=4,
    included_fixations=[],
)
```

"""
from __future__ import annotations

from typing import Iterable, List, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DinoV2Backbone"]


class DinoV2Backbone(nn.Module):
    """Wrap *facebookresearch/dinov2* ViT‑B/14 so that it behaves like a CNN.

    Parameters
    ----------
    layers : Iterable[int]
        Indices of the transformer blocks to extract (same semantics as
        `get_intermediate_layers` – negative values count from the end).
    model_name : str, default ``"dinov2_vitb14"``
        Any identifier accepted by ``torch.hub.load('facebookresearch/dinov2', ...)``.
    freeze : bool, default ``True``
        If *True* (recommended) the backbone weights are frozen.
    patch_size : int, default ``14``
        Patch stride of the ViT variant you load.  Only change this if you
        load a different DINO v2 model (e.g. ViT‑L/16 ⇒ 16).
    """

    def __init__(
        self,
        layers: Iterable[int] | None = None,
        *,
        model_name: str = "dinov2_vitb14",
        #model_name = "dinov2_vitg14",
        freeze: bool = True,
        patch_size: int = 14,
    ) -> None:
        super().__init__()

        if layers is None:
            layers = (-3, -2, -1)
        self.layers: Tuple[int, ...] = tuple(layers)
        self.patch_size: int = patch_size

        # --- load backbone --------------------------------------------------
        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)

        # freeze params ------------------------------------------------------
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()

        # number of channels the ViT emits per token
        self.num_channels: int = self.backbone.embed_dim  # 768 for ViT‑B/14

    # ---------------------------------------------------------------------
    # helpers
    # ---------------------------------------------------------------------
    def _pad_to_multiple(self, x: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        B,C,h,w = x.shape
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        if pad_h or pad_w:
            # pad = (left, right, top, bottom)
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, (h, w)

    def _tokens_to_map(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        tokens: (B, N+1, C).  We drop CLS and reshape the remaining N patch tokens into
        a (B, C, H_patch, W_patch) map—automatically padding or trimming to fit.
        """
        # drop cls token
        tokens = tokens[:, 1:, :]  # (B, N, C)
        B, N, C = tokens.shape

        # compute how many patches we need in each dim
        H_p = math.ceil(h / self.patch_size)
        W_p = math.ceil(w / self.patch_size)
        expected_N = H_p * W_p

        # if we have too many tokens, trim; if too few, pad with zeros
        if N > expected_N:
            tokens = tokens[:, :expected_N, :]
        elif N < expected_N:
            pad = tokens.new_zeros((B, expected_N - N, C))
            tokens = torch.cat([tokens, pad], dim=1)

        # permute & reshape
        feat = tokens.permute(0, 2, 1).contiguous()  # (B, C, N)
        feat = feat.view(B, C, H_p, W_p)
        return feat

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return one (B, C, H_patch, W_patch) map per block in self.layers."""
        # 1) pad to patch‐multiple & remember original size
        x, (orig_h, orig_w) = self._pad_to_multiple(x, self.patch_size)
        H_pad, W_pad = x.shape[-2:]

        # 2) cast uint8→float32/255 else float32
        if x.dtype == torch.uint8:
            x = x.float().div_(255.0)
        else:
            x = x.float()

        # 3) grab tokens from the last n-needed layers
        with torch.set_grad_enabled(self.backbone.training):
            n_needed = max(abs(i) for i in self.layers)
            tokens = self.backbone.get_intermediate_layers(x, n=n_needed)
            # tokens is a list/tuple indexed by 0…n_needed-1

        # 4) for each requested layer, reshape to map & crop to original patches
        feats: List[torch.Tensor] = []
        for idx in self.layers:
            layer_tokens = tokens[idx]  # (B, N+1, C)
            feat_map = self._tokens_to_map(layer_tokens, H_pad, W_pad)
            # now crop any extra patch‐rows/cols beyond original
            H_crop = math.ceil(orig_h / self.patch_size)
            W_crop = math.ceil(orig_w / self.patch_size)
            feat_map = feat_map[:, :, :H_crop, :W_crop]
            feats.append(feat_map)

        return feats

