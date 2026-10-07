"""The deployment predictor accepts RGB only, never simulator object state."""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18


def make_encoder(pretrained=True):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Identity()
    return model


def normalize_images(images):
    """Float RGB BCHW in [0, 1] -> ImageNet normalization."""
    mean = images.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = images.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return (images - mean) / std


class AuxiliaryModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder = make_encoder(pretrained)
        # All groups instantiate both heads, including A; only losses differ.
        self.object_head = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 4))
        self.hand_head = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 42))

    def forward(self, normalized_images):
        feature = self.encoder(normalized_images)
        return feature, self.object_head(feature).sigmoid(), self.hand_head(feature).sigmoid()


class StateModel(nn.Module):
    def __init__(self, target_mean, target_scale):
        super().__init__()
        self.encoder = make_encoder(pretrained=False)
        # Preserve image location: pooled features discard the spatial layout
        # needed to localize a 20 mm block in a full-frame camera observation.
        self.xy_head = nn.Sequential(nn.Conv2d(512, 32, 1), nn.ReLU(), nn.Flatten(),
                                     nn.Linear(32 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 2))
        self.register_buffer("target_mean", torch.as_tensor(target_mean, dtype=torch.float32))
        self.register_buffer("target_scale", torch.as_tensor(target_scale, dtype=torch.float32))

    def forward(self, normalized_images):
        enc = self.encoder
        feature = enc.maxpool(enc.relu(enc.bn1(enc.conv1(normalized_images))))
        feature = enc.layer4(enc.layer3(enc.layer2(enc.layer1(feature))))
        return self.xy_head(feature)

    def predict_xy_m(self, normalized_images):
        return self(normalized_images) * self.target_scale + self.target_mean


class RGBStatePredictor:
    """Callable HWC RGB uint8 -> world XY in metres, shape (2,).

    Resizing is the same full-frame 224x224 operation used during training.
    Fixed object height and task geometry belong to the common controller;
    this model does not estimate airborne Z, object yaw, or a hand pose.
    """
    def __init__(self, model, device):
        self.model = model.to(device).eval()
        self.device = torch.device(device)

    @torch.inference_mode()
    def __call__(self, rgb_image):
        image = np.asarray(rgb_image)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("Expected HWC RGB uint8 image (not BGR or simulator state)")
        # OpenCV INTER_AREA is used by both dataset construction and deployment.
        import cv2
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div(255).unsqueeze(0)
        return self.model.predict_xy_m(normalize_images(tensor.to(self.device)))[0].cpu().numpy()


def load_state_predictor(checkpoint, device=None):
    """Load only project-produced, trusted local checkpoint files."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    if payload.get("kind") != "visual_state_xy_v1":
        raise ValueError("Checkpoint is not a visual_state_xy_v1 deployment model")
    model = StateModel(payload["target_mean"], payload["target_scale"])
    model.load_state_dict(payload["model_state"])
    return RGBStatePredictor(model, device)
