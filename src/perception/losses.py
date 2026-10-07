"""Loss masking preserves RGB examples and safely ignores invalid pseudo labels."""
import torch
from torch.nn import functional as F


def masked_smooth_l1(prediction, target, mask):
    """Mean over valid coordinates; all-invalid batches give differentiable zero.

    Invalid targets may be NaN. They are replaced before evaluating the loss,
    because multiplying a NaN loss by zero would not remove the NaN.
    """
    mask = mask.bool()
    while mask.ndim < target.ndim:
        mask = mask.unsqueeze(-1)
    mask = torch.broadcast_to(mask, target.shape) & torch.isfinite(target)
    safe_target = torch.where(mask, target, prediction.detach())
    terms = F.smooth_l1_loss(prediction, safe_target, reduction="none", beta=0.1)
    return torch.where(mask, terms, torch.zeros_like(terms)).sum() / mask.sum().clamp_min(1)


def visual_losses(feature, teacher_feature, object_prediction, hand_prediction,
                  object_target, hand_target, object_mask, hand_mask,
                  object_enabled=False, hand_enabled=False,
                  object_weight=1.0, hand_weight=1.0):
    # Squared distance between unit vectors is 2 - 2 cosine similarity.
    consistency = (F.normalize(feature, dim=-1) - F.normalize(teacher_feature, dim=-1)).square().sum(-1).mean()
    object_loss = masked_smooth_l1(object_prediction, object_target, object_mask)
    hand_loss = masked_smooth_l1(hand_prediction.reshape(-1, 21, 2), hand_target.reshape(-1, 21, 2), hand_mask)
    total = consistency
    if object_enabled:
        total = total + object_weight * object_loss
    if hand_enabled:
        total = total + hand_weight * hand_loss
    return total, {"consistency": consistency, "object": object_loss, "hand": hand_loss}
