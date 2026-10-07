"""Fit identical new state heads to matched simulation images and XY labels."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.nn import functional as F
from src.perception.models import StateModel, normalize_images
from src.perception.training import (GROUPS, batch_stream, image_batch, load_arrays,
    photometric_augment, provenance, seed_everything, sha256_file, state_digest, write_json)


@torch.inference_mode()
def evaluate(model, arrays, indices, device):
    model.eval()
    predictions = []
    for start in range(0, len(indices), 32):
        predictions.append(model.predict_xy_m(normalize_images(image_batch(arrays["images"], indices[start:start + 32], device))).cpu().numpy())
    predicted = np.concatenate(predictions)
    truth = arrays["target_xy_m"][indices]
    errors = np.linalg.norm(predicted - truth, axis=1) * 1000
    return {"n_frames": len(indices), "n_scenes": len(np.unique(arrays["scene_seeds"][indices])),
            "xy_mean_error_mm": float(errors.mean()), "xy_median_error_mm": float(np.median(errors)),
            "xy_p90_error_mm": float(np.percentile(errors, 90)),
            "xy_rmse_mm": float(np.sqrt(np.mean(np.sum((predicted - truth) ** 2, axis=1))) * 1000)}, predicted


def adapt_group(arrays, group, seed, config, metadata, pretrain_root, output_root, device):
    source = pretrain_root / f"{group}_seed{seed}" / "checkpoint.pt"
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if payload.get("kind") != "visual_pretraining_v1" or payload["manifest"]["group"] != group or payload["manifest"]["seed"] != seed:
        raise ValueError(f"Wrong pretraining checkpoint: {source}")
    out = output_root / f"{group}_seed{seed}"
    if (out / "checkpoint.pt").exists():
        raise FileExistsError(f"Refusing to overwrite completed run: {out}")
    out.mkdir(parents=True, exist_ok=True)
    train = np.flatnonzero(arrays["split"] == "train")
    target_mean = arrays["target_xy_m"][train].mean(axis=0)
    target_scale = np.maximum(arrays["target_xy_m"][train].std(axis=0), 0.01)
    seed_everything(seed + 500)
    model = StateModel(target_mean, target_scale)
    # The original auxiliary heads are intentionally never loaded.
    head_initial_hash = state_digest(model.xy_head.state_dict())
    model.encoder.load_state_dict(payload["encoder_state"])
    model.to(device)
    if not config["encoder_finetune"]:
        model.encoder.requires_grad_(False)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=config["learning_rate"], weight_decay=config["weight_decay"])
    targets = torch.from_numpy((arrays["target_xy_m"] - target_mean) / target_scale).to(device)
    generator = torch.Generator().manual_seed(seed + 700)
    batch_digest = hashlib.sha256()
    manifest = {**metadata, "kind": "visual_state_xy_v1", "group": group, "seed": seed,
                "pretraining_path": str(source.resolve()), "pretraining_sha256": sha256_file(source),
                "head_initial_sha256": head_initial_hash, "target_mean": target_mean.tolist(),
                "target_scale": target_scale.tolist(), "selection": "fixed final step; no best-checkpoint selection",
                "runtime_inputs": "RGB only; no simulator object state", "output": "XY metres", "train_frames": len(train)}
    write_json(out / "manifest.json", manifest)
    started = time.perf_counter()
    model.train()
    if not config["encoder_finetune"]:
        model.encoder.eval()
    with (out / "losses.jsonl").open("w", encoding="utf-8") as log:
        for step, indices in enumerate(batch_stream(train, config["batch_size"], config["steps"], seed + 600), 1):
            batch_digest.update(indices.numpy().tobytes())
            x = photometric_augment(image_batch(arrays["images"], indices, device), generator, config["photometric_strength"])
            predicted = model(normalize_images(x))
            loss = F.smooth_l1_loss(predicted, targets[indices.to(device)], beta=0.5)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite adaptation loss at {group} step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            row = {"step": step, "xy_scaled_smooth_l1": float(loss.detach()), "grad_norm": float(grad_norm)}
            log.write(json.dumps(row, allow_nan=False) + "\n")
            if step == 1 or step % 50 == 0 or step == config["steps"]:
                log.flush()
                print(f"adapt {group} seed={seed} step={step}/{config['steps']} loss={row['xy_scaled_smooth_l1']:.6f}", flush=True)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    # Neither validation nor test has been evaluated or used during optimization.
    metrics = {}
    predictions = {}
    for split in ("train", "val", "test"):
        ids = np.flatnonzero(arrays["split"] == split)
        if len(ids):
            metrics[split], predicted = evaluate(model, arrays, ids, device)
            predictions[f"{split}_indices"] = ids
            predictions[f"{split}_xy_m"] = predicted
            predictions[f"{split}_true_xy_m"] = arrays["target_xy_m"][ids]
            predictions[f"{split}_scene_seeds"] = arrays["scene_seeds"][ids]
    manifest.update({"status": "completed", "elapsed_seconds": elapsed,
                     "batch_sequence_sha256": batch_digest.hexdigest(),
                     "augmentation_final_rng_sha256": hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest(),
                     "steps_completed": config["steps"], "final_training_loss": row, "metrics": metrics})
    checkpoint_temp = out / "checkpoint.pt.tmp"
    torch.save({"kind": "visual_state_xy_v1", "model_state": model.cpu().state_dict(),
                "target_mean": target_mean.tolist(), "target_scale": target_scale.tolist(), "manifest": manifest}, checkpoint_temp)
    checkpoint_temp.replace(out / "checkpoint.pt")
    np.savez_compressed(out / "predictions.npz", **predictions)
    write_json(out / "manifest.json", manifest)
    print(json.dumps({"group": group, "seed": seed, "metrics": metrics}), flush=True)
    del model, optimizer, targets, payload
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/visual_training.json")
    parser.add_argument("--pretraining", type=Path, default=ROOT / "artifacts/runs/visual_pretraining_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/runs/visual_state_v1")
    parser.add_argument("--groups", nargs="+", choices=list(GROUPS), default=list(GROUPS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = dict(json.loads(args.config.read_text(encoding="utf-8"))["adaptation"])
    if args.steps is not None:
        config["steps"] = args.steps
    if config["steps"] < 1:
        raise ValueError("steps must be positive")
    arrays = load_arrays(args.data, real=False)
    metadata = provenance(ROOT, args.data, config)
    results = []
    for seed in args.seeds:
        for group in args.groups:
            results.append(adapt_group(arrays, group, seed, config, metadata, args.pretraining, args.output, args.device))
    write_json(args.output / f"summary_{'_'.join(args.groups)}_{'_'.join(map(str, args.seeds))}.json", results)


if __name__ == "__main__":
    main()
