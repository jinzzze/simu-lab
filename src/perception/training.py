"""Shared deterministic sampling, validation and provenance helpers."""
import hashlib
import json
import os
import random
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch


GROUPS = {"A": (False, False), "B": (True, False), "C": (False, True), "D": (True, True)}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def provenance(root, data_path, config):
    files = sorted((root / "src/perception").glob("*.py"))
    files += [root / "scripts/train_visual_ablation.py", root / "scripts/adapt_visual_state.py"]
    return {"data_path": str(data_path.resolve()), "data_sha256": sha256_file(data_path),
            "code_sha256": {str(p.relative_to(root)): sha256_file(p) for p in files if p.exists()},
            "config": config, "torch": str(torch.__version__), "numpy": str(np.__version__),
            "cuda": str(torch.version.cuda),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def load_arrays(path, real=True):
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    images = arrays["images"]
    n = len(images)
    if n == 0 or images.dtype != np.uint8 or images.shape[1:] != (224, 224, 3):
        raise ValueError("images must be non-empty uint8 [N,224,224,3]")
    if real:
        expected = {"object_targets": (n, 4), "object_mask": (n,), "hand_targets": (n, 42),
                    "hand_mask": (n, 21), "episode_ids": (n,), "frame_indices": (n,)}
        for name, shape in expected.items():
            if name not in arrays or arrays[name].shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        keys = list(zip(arrays["episode_ids"].tolist(), arrays["frame_indices"].tolist()))
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate episode/frame entries would reweight real data")
    else:
        if arrays["target_xy_m"].shape != (n, 2) or not np.isfinite(arrays["target_xy_m"]).all():
            raise ValueError("target_xy_m must be finite [N,2]")
        validate_scene_splits(arrays["scene_seeds"], arrays["split"])
        if len(arrays["split"]) != n or len(arrays["scene_seeds"]) != n:
            raise ValueError("Scene and split array lengths must match images")
    return arrays


def validate_scene_splits(seeds, splits):
    if not {"train", "val"}.issubset(set(splits.tolist())):
        raise ValueError("Separate train and val scenes required")
    if not set(splits.tolist()).issubset({"train", "val", "test"}):
        raise ValueError("Unknown simulation split")
    seen = {}
    for seed, split in zip(seeds.tolist(), splits.tolist()):
        if seed in seen and seen[seed] != split:
            raise ValueError(f"Scene seed {seed} leaks across {seen[seed]} and {split}")
        seen[seed] = split


def batch_stream(indices, batch_size, steps, seed):
    generator = torch.Generator().manual_seed(seed)
    indices = torch.as_tensor(indices, dtype=torch.long)
    if len(indices) == 0:
        raise ValueError("Empty training set")
    # Cycle permutations; even small datasets keep the exact batch budget.
    pending = torch.empty(0, dtype=torch.long)
    for _ in range(steps):
        while len(pending) < batch_size:
            pending = torch.cat((pending, indices[torch.randperm(len(indices), generator=generator)]))
        yield pending[:batch_size]
        pending = pending[batch_size:]


def image_batch(images, indices, device):
    return torch.from_numpy(images[np.asarray(indices)]).permute(0, 3, 1, 2).float().div(255).to(device)


def photometric_augment(images, generator, strength=0.1):
    """Full-frame per-image brightness/contrast/saturation; no geometry changes."""
    values = 1 + (torch.rand((len(images), 3, 1, 1), generator=generator) * 2 - 1) * strength
    values = values.to(images.device)
    brightness, contrast, saturation = values[:, 0:1], values[:, 1:2], values[:, 2:3]
    changed = images * brightness
    mean = changed.mean(dim=(1, 2, 3), keepdim=True)
    changed = (changed - mean) * contrast + mean
    gray = changed.mean(dim=1, keepdim=True)
    return ((changed - gray) * saturation + gray).clamp(0, 1)
