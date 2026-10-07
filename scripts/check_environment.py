"""Run P1 hardware/library smoke checks; these are not research results."""
from __future__ import annotations

import hashlib
import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "diagnostics"
OUT.mkdir(parents=True, exist_ok=True)


def simulation_check():
    import gymnasium as gym
    import imageio.v3 as iio
    import numpy as np
    import panda_gym  # registers environments
    import pybullet

    env = gym.make("PandaPush-v3", render_mode="rgb_array", render_width=512, render_height=512)
    try:
        obs, info = env.reset(seed=7)
        start = env.unwrapped.robot.get_ee_position().copy()
        frame = env.render()
        if frame is None or frame.ndim != 3 or frame.std() < 1:
            raise RuntimeError("Simulator returned an empty or constant RGB frame")
        iio.imwrite(OUT / "panda_initial.png", frame)
        for _ in range(12):
            obs, reward, terminated, truncated, info = env.step(np.array([0.15, 0., 0.], dtype=np.float32))
            if terminated or truncated:
                break
        end = env.unwrapped.robot.get_ee_position().copy()
        distance = float(np.linalg.norm(end-start))
        if not np.isfinite(distance) or distance < 0.001:
            raise RuntimeError(f"End-effector did not move as expected: {distance}")
        iio.imwrite(OUT / "panda_after_motion.png", env.render())
        return {"environment": "PandaPush-v3", "pybullet_api": pybullet.getAPIVersion(),
                "frame_shape": list(frame.shape), "action_shape": list(env.action_space.shape),
                "end_effector_motion_m": distance, "observation_keys": list(obs),
                "note": "Tests rendering and commanded arm motion only; no learned policy or task success claim."}
    finally:
        env.close()


def cuda_check():
    import numpy as np
    import torch
    import torchvision
    from torchvision.models import resnet18, ResNet18_Weights

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable to project PyTorch")
    torch.manual_seed(7)
    x = torch.randn(64, 64)
    ref = x @ x.T
    actual = (x.cuda() @ x.cuda().T).cpu()
    torch.testing.assert_close(actual, ref, rtol=1e-4, atol=1e-4)
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval().cuda()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 224, 224, device="cuda"))
    if list(output.shape) != [1, 1000] or not torch.isfinite(output).all():
        raise RuntimeError("ResNet-18 GPU inference returned invalid output")
    props = torch.cuda.get_device_properties(0)
    result = {"torch": torch.__version__, "torchvision": torchvision.__version__,
              "cuda_runtime": torch.version.cuda, "gpu": props.name,
              "vram_gib": round(props.total_memory/(1024**3), 2),
              "resnet18_weights": "IMAGENET1K_V1", "output_shape": list(output.shape),
              "max_matmul_error": float((actual-ref).abs().max()),
              "torch_cache": os.environ.get("TORCH_HOME")}
    del model, output
    torch.cuda.empty_cache()
    return result


def vision_check():
    import cv2
    import mediapipe as mp
    import numpy as np
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(dictionary, 0, 160)
    canvas = np.full((240, 240), 255, np.uint8)
    canvas[40:200, 40:200] = marker
    _, ids, _ = cv2.aruco.ArucoDetector(dictionary).detectMarkers(canvas)
    if ids is None or 0 not in ids.ravel():
        raise RuntimeError("OpenCV did not recover the generated calibration marker")
    model_path = ROOT / "artifacts" / "models" / "hand_landmarker.task"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE, num_hands=2)
    with vision.HandLandmarker.create_from_options(options) as detector:
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                          data=np.zeros((256, 256, 3), dtype=np.uint8)))
    if result.hand_landmarks:
        raise RuntimeError("Hand detector hallucinated a hand on a blank test image")
    return {"opencv": cv2.__version__, "mediapipe": metadata.version("mediapipe"),
            "aruco_roundtrip_id": 0, "blank_frame_hand_count": len(result.hand_landmarks),
            "hand_model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "note": "Model loads and processes an image; hand accuracy on the applicant's videos is still unverified."}


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT / 'environment_current')
    parser.add_argument('--check', choices=['simulation', 'cuda', 'vision'])
    args = parser.parse_args()
    OUT = args.output.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    functions = {'simulation': simulation_check, 'cuda': cuda_check, 'vision': vision_check}
    if args.check:
        start = time.monotonic()
        try:
            detail = functions[args.check]()
            result = {'status': 'passed', 'seconds': round(time.monotonic()-start, 2), **detail}
        except Exception as exc:
            result = {'status': 'failed', 'error': str(exc), 'traceback': traceback.format_exc()}
        (OUT / f'{args.check}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        return 0 if result['status'] == 'passed' else 1
    import psutil
    packages = ["numpy", "pybullet", "panda-gym", "gymnasium", "mediapipe",
                "opencv-contrib-python", "torch", "torchvision"]
    versions = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    report = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "project_root": str(ROOT), "python_executable": sys.executable,
              "python": sys.version, "platform": platform.platform(),
              "logical_cpus": os.cpu_count(), "ram_gib": round(psutil.virtual_memory().total/1024**3, 2),
              "disk_free_gib": round(shutil.disk_usage(ROOT).free/1024**3, 2),
              "packages": versions, "checks": {}}
    report['native_library_isolation'] = 'separate processes; no duplicate OpenMP suppression'
    for name in functions:
        # Environment diagnosis itself must not mix the incompatible runtimes.
        with (OUT / f'{name}.log').open('w', encoding='utf-8') as log:
            child = subprocess.run([sys.executable, str(Path(__file__)), '--check', name, '--output', str(OUT)],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        result_path = OUT / f'{name}.json'
        if child.returncode == 0 and result_path.exists():
            report['checks'][name] = json.loads(result_path.read_text(encoding='utf-8'))
        else:
            report['checks'][name] = {'status': 'failed', 'returncode': child.returncode, 'log': str(OUT / f'{name}.log')}
        print(f"{report['checks'][name]['status'].upper()}: {name}", flush=True)
        (OUT/"environment.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    report["all_passed"]=all(c["status"] == "passed" for c in report["checks"].values())
    (OUT/"environment.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({"all_passed":report["all_passed"],"report":str(OUT/"environment.json")},ensure_ascii=False))
    return 0 if report["all_passed"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
