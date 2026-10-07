"""Matched A/B/C/D real RGB pretraining. Run from the project environment."""
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import yaml
from src.perception.losses import visual_losses
from src.perception.models import AuxiliaryModel, make_encoder, normalize_images
from src.perception.training import (GROUPS, batch_stream, image_batch, load_arrays,
    photometric_augment, provenance, seed_everything, state_digest, write_json)


def train_group(arrays, teacher_features, group, seed, config, metadata, output_root, device):
    out = output_root / f"{group}_seed{seed}"
    if (out / "checkpoint.pt").exists():
        raise FileExistsError(f"Refusing to overwrite completed run: {out}")
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    model = AuxiliaryModel(pretrained=True).to(device)
    initial_hash = state_digest(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    generator = torch.Generator().manual_seed(seed + 200)
    arrays_t = {name: torch.from_numpy(arrays[name]).to(device) for name in
                ("object_targets", "hand_targets", "object_mask", "hand_mask")}
    batch_digest = hashlib.sha256()
    started = time.perf_counter()
    object_on, hand_on = GROUPS[group]
    manifest = {**metadata, "kind": "visual_pretraining_v1", "group": group, "seed": seed,
                "initial_model_sha256": initial_hash, "started_utc": datetime.now(timezone.utc).isoformat(),
                "rgb_frames": len(arrays["images"]), "episodes": sorted(set(arrays["episode_ids"].tolist())),
                "object_valid_frames": int(arrays["object_mask"].sum()),
                "hand_valid_landmarks": int(arrays["hand_mask"].sum()),
                "enabled": {"object": object_on, "hand": hand_on},
                "evaluation_status": "development pretraining only; no real holdout performance"}
    write_json(out / "manifest.json", manifest)
    model.train()
    with (out / "losses.jsonl").open("w", encoding="utf-8") as log:
        for step, indices in enumerate(batch_stream(np.arange(len(arrays["images"])), config["batch_size"], config["steps"], seed + 100), 1):
            batch_digest.update(indices.numpy().tobytes())
            x = image_batch(arrays["images"], indices, device)
            x = photometric_augment(x, generator, config["photometric_strength"])
            feature, objects, hands = model(normalize_images(x))
            ids = indices.to(device)
            loss, parts = visual_losses(feature, teacher_features[indices].to(device), objects, hands,
                arrays_t["object_targets"][ids], arrays_t["hand_targets"][ids], arrays_t["object_mask"][ids],
                arrays_t["hand_mask"][ids], object_on, hand_on, config["object_weight"], config["hand_weight"])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training loss at {group} step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            row = {"step": step, "total": float(loss.detach()), **{key: float(val.detach()) for key, val in parts.items()},
                   "object_valid": int(arrays_t["object_mask"][ids].sum()),
                   "hand_valid_points": int(arrays_t["hand_mask"][ids].sum()), "grad_norm": float(grad_norm)}
            log.write(json.dumps(row, allow_nan=False) + "\n")
            if step == 1 or step % 40 == 0 or step == config["steps"]:
                log.flush()
                print(f"pretrain {group} seed={seed} step={step}/{config['steps']} total={row['total']:.6f} "
                      f"base={row['consistency']:.6f} obj={row['object']:.6f} hand={row['hand']:.6f}", flush=True)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    manifest.update({"elapsed_seconds": time.perf_counter() - started,
                     "batch_sequence_sha256": batch_digest.hexdigest(),
                     "augmentation_final_rng_sha256": hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest(),
                     "final_losses": row, "status": "completed", "steps_completed": config["steps"]})
    model = model.cpu()
    checkpoint_temp = out / "checkpoint.pt.tmp"
    torch.save({"kind": "visual_pretraining_v1", "encoder_state": model.encoder.state_dict(),
                "auxiliary_state": {"object": model.object_head.state_dict(), "hand": model.hand_head.state_dict()},
                "manifest": manifest}, checkpoint_temp)
    checkpoint_temp.replace(out / "checkpoint.pt")
    write_json(out / "manifest.json", manifest)
    del model, optimizer, arrays_t
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/pilot_pretraining_v1.npz")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/visual_training.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/runs/visual_pretraining_v1")
    parser.add_argument("--groups", nargs="+", choices=list(GROUPS), default=list(GROUPS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    all_config = json.loads(args.config.read_text(encoding="utf-8"))
    config = dict(all_config["pretraining"])
    if args.steps is not None:
        config["steps"] = args.steps
    if config["steps"] < 1:
        raise ValueError("steps must be positive")
    actual_groups = yaml.safe_load((ROOT / "configs/experiment_groups.yaml").read_text(encoding="utf-8"))["groups"]
    for group, (obj, hand) in GROUPS.items():
        if actual_groups[group] != {"object_supervision": obj, "hand_supervision": hand}:
            raise ValueError("Group protocol and code disagree")
    arrays = load_arrays(args.data, real=True)
    metadata = provenance(ROOT, args.data, config)
    metadata["real_data_status"] = all_config["real_data_status"]
    seed_everything(0)
    teacher = make_encoder(pretrained=True).to(args.device).eval()
    teacher.requires_grad_(False)
    metadata["teacher_state_sha256"] = state_digest(teacher.state_dict())
    with torch.inference_mode():
        features = torch.cat([teacher(normalize_images(image_batch(arrays["images"], np.arange(start, min(start + 32, len(arrays["images"]))), args.device))).cpu()
                              for start in range(0, len(arrays["images"]), 32)])
    # Clone outside inference mode so the target can participate in autograd loss operations.
    features = features.clone()
    del teacher
    results = []
    for seed in args.seeds:
        for group in args.groups:
            results.append(train_group(arrays, features, group, seed, config, metadata, args.output, args.device))
    write_json(args.output / f"summary_{'_'.join(args.groups)}_{'_'.join(map(str, args.seeds))}.json", results)


if __name__ == "__main__":
    main()
