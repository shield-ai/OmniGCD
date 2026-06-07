"""Loss functions for GCDformer training."""

from __future__ import annotations


def pairwise_contrastive_loss(embeddings, labels, margin: float = 1.0):
    """Contrastive loss from the OmniGCD paper.

    Same-label pairs are pulled together with squared Euclidean distance.
    Different-label pairs are pushed apart until they are at least ``margin``
    apart.
    """
    import torch
    import torch.nn.functional as F

    distances = torch.cdist(embeddings, embeddings, p=2)
    same = labels.unsqueeze(2).eq(labels.unsqueeze(1)).float()
    eye = torch.eye(labels.shape[1], device=labels.device).unsqueeze(0)
    valid = 1.0 - eye

    positive = same * distances.pow(2)
    negative = (1.0 - same) * F.relu(margin - distances).pow(2)
    loss = (positive + negative) * valid
    return loss.sum() / valid.sum().clamp_min(1.0) / labels.shape[0]
