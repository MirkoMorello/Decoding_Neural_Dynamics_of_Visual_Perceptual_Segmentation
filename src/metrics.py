# metrics.py

import numpy as np
from pysaliency.roc import general_roc
from pysaliency.numba_utils import auc_for_one_positive
import torch


def _general_auc(positives, negatives):
    if len(positives) == 1:
        return auc_for_one_positive(positives[0], negatives)
    else:
        return general_roc(positives, negatives)[0]


def log_likelihood(log_density, fixation_mask, weights=None):
    #if weights is None:
    #    weights = torch.ones(log_density.shape[0])

    weights = len(weights) * weights.view(-1, 1, 1) / weights.sum()

    if isinstance(fixation_mask, torch.sparse.IntTensor):
        dense_mask = fixation_mask.to_dense()
    else:
        dense_mask = fixation_mask
    fixation_count = dense_mask.sum(dim=(-1, -2), keepdim=True)
    ll = torch.mean(
        weights * torch.sum(log_density * dense_mask, dim=(-1, -2), keepdim=True) / fixation_count
    )
    return (ll + torch.log(torch.tensor(log_density.shape[-1] * log_density.shape[-2]))) / torch.log(torch.tensor(2.0))


def nss(log_density, fixation_mask, weights=None):
    weights = len(weights) * weights.view(-1, 1, 1) / weights.sum()
    if isinstance(fixation_mask, torch.sparse.IntTensor):
        dense_mask = fixation_mask.to_dense()
    else:
        dense_mask = fixation_mask

    fixation_count = dense_mask.sum(dim=(-1, -2), keepdim=True)

    density = torch.exp(log_density)
    mean, std = torch.std_mean(density, dim=(-1, -2), keepdim=True)
    saliency_map = (density - mean) / std

    nss = torch.mean(
        weights * torch.sum(saliency_map * dense_mask, dim=(-1, -2), keepdim=True) / fixation_count
    )
    return nss


def auc(log_density, fixation_mask, weights=None):
    weights = len(weights) * weights / weights.sum()

    # TODO: This doesn't account for multiple fixations in the same location!
    def image_auc(log_density, fixation_mask):
        if isinstance(fixation_mask, torch.sparse.IntTensor):
            dense_mask = fixation_mask.to_dense()
        else:
            dense_mask = fixation_mask

        positives = torch.masked_select(log_density, dense_mask.type(torch.bool)).detach().cpu().numpy().astype(np.float64)
        negatives = log_density.flatten().detach().cpu().numpy().astype(np.float64)

        auc = _general_auc(positives, negatives)

        return torch.tensor(auc)

    return torch.mean(weights.cpu() * torch.tensor([
        image_auc(log_density[i], fixation_mask[i]) for i in range(log_density.shape[0])
    ]))


def auc_gpu(log_density, fixation_mask, weights=None):
    weights = len(weights) * weights / weights.sum()

    if isinstance(fixation_mask, torch.sparse.IntTensor):
        dense_mask = fixation_mask.to_dense()
    else:
        dense_mask = fixation_mask
    dense_mask = dense_mask.bool()

    batch_size = log_density.shape[0]
    aucs = torch.zeros(batch_size, device=log_density.device, dtype=torch.float64)

    for i in range(batch_size):
        image_log_density = log_density[i]
        image_mask = dense_mask[i]

        positives = torch.masked_select(image_log_density, image_mask)
        
        if positives.numel() == 0:
            # No fixations, AUC is undefined. Returning NaN, which is consistent
            # with what would likely happen in the original code.
            aucs[i] = torch.tensor(float('nan'), device=log_density.device, dtype=torch.float64)
            continue

        negatives = image_log_density.flatten()
        
        # The original implementation's _general_auc is equivalent to averaging
        # the AUC for each positive against all other values. This is equivalent
        # to the average rank percentile of positive samples among all samples.
        
        # Sort all pixel values
        sorted_negatives = torch.sort(negatives)[0]
        
        # Find rank of each positive
        # number of values less than p
        ranks_lower = torch.searchsorted(sorted_negatives, positives, right=False)
        # number of values less than or equal to p
        ranks_upper = torch.searchsorted(sorted_negatives, positives, right=True)
        
        # rank = num_less + 0.5 * num_equal
        ranks = ranks_lower.to(torch.float64) + 0.5 * (ranks_upper - ranks_lower).to(torch.float64)
        
        # Average rank percentile
        auc_val = torch.mean(ranks) / negatives.numel()
        aucs[i] = auc_val

    # torch.mean of a tensor with NaNs is NaN, which is consistent with original behavior.
    return torch.mean(weights.to(aucs.device) * aucs)