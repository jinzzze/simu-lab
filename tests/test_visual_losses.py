"""Tests target experimental validity boundaries, not training accuracy."""
import numpy as np
import pytest
import torch
from src.perception.losses import masked_smooth_l1, visual_losses
from src.perception.training import GROUPS, batch_stream, photometric_augment, validate_scene_splits


def test_invalid_nan_labels_are_zero_without_nan_gradients():
    prediction = torch.ones(3, 4, requires_grad=True)
    loss = masked_smooth_l1(prediction, torch.full((3, 4), float("nan")), torch.zeros(3, dtype=torch.bool))
    loss.backward()
    assert loss.item() == 0
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))


def test_partial_hand_mask_ignores_invalid_coordinates():
    prediction = torch.zeros(1, 2, 2, requires_grad=True)
    target = torch.tensor([[[0.2, 0.4], [float("nan"), float("nan")]]])
    loss = masked_smooth_l1(prediction, target, torch.tensor([[True, False]]))
    assert loss.item() == pytest.approx(0.25)
    loss.backward()
    assert prediction.grad[0, 1].abs().sum().item() == 0


@pytest.mark.parametrize("group", list(GROUPS))
def test_only_declared_auxiliary_heads_receive_gradients(group):
    feature = torch.randn(2, 8, requires_grad=True)
    teacher = torch.randn(2, 8)
    obj = torch.zeros(2, 4, requires_grad=True)
    hand = torch.zeros(2, 42, requires_grad=True)
    object_on, hand_on = GROUPS[group]
    total, parts = visual_losses(feature, teacher, obj, hand, torch.ones_like(obj), torch.ones_like(hand),
                                torch.ones(2, dtype=torch.bool), torch.ones(2, 21, dtype=torch.bool), object_on, hand_on)
    total.backward()
    assert feature.grad.abs().sum() > 0
    assert (obj.grad is not None) == object_on
    assert (hand.grad is not None) == hand_on
    assert total.detach().item() == pytest.approx((parts["consistency"] + object_on * parts["object"] + hand_on * parts["hand"]).detach().item())


def test_group_pairing_has_identical_batch_and_augmentation_stream():
    a = list(batch_stream(np.arange(7), 4, 9, 71))
    b = list(batch_stream(np.arange(7), 4, 9, 71))
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    images = torch.rand(4, 3, 16, 16)
    x = photometric_augment(images, torch.Generator().manual_seed(72))
    y = photometric_augment(images, torch.Generator().manual_seed(72))
    assert torch.equal(x, y)


def test_scene_seed_cannot_cross_simulation_splits():
    with pytest.raises(ValueError, match="leaks"):
        validate_scene_splits(np.array([1, 2, 1]), np.array(["train", "val", "test"]))
    validate_scene_splits(np.array([1, 2, 3]), np.array(["train", "val", "test"]))
