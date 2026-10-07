"""Run explicitly privileged control diagnostics, never a learned-policy result."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import imageio.v2 as imageio
import numpy as np
from src.sim import GraspEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--output", default="artifacts/diagnostics/grasp_oracle")
    parser.add_argument("--record-first", action="store_true")
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    env = GraspEnv()
    results = []
    try:
        for i in range(args.episodes):
            seed = args.start_seed + i
            observation = env.reset(seed)
            if i == 0:
                imageio.imwrite(output / "initial_overhead.png", observation["rgb"])
            # Explicit oracle accessor. All learned policies must pass predicted XY instead.
            initial_xy = env.get_object_xy_for_labels()
            result = env.execute_grasp(initial_xy, record=(i == 0 and args.record_first))
            result["initial_object_xy_m"] = initial_xy.tolist()
            results.append(result)
            print(json.dumps({"seed": seed, "success": result["success"],
                              "clearance": result["max_clearance_m"],
                              "final": result["final_object_xyz_m"],
                              "flags": result["final_flags"]}), flush=True)
            if i == 0:
                imageio.imwrite(output / "final_overhead.png", env.render())
                (output / "first_trace.json").write_text(json.dumps(env.trace, indent=2), encoding="utf-8")
                if args.record_first:
                    imageio.mimwrite(output / "oracle_demo.mp4", env.frames, fps=1 / (2 * env.sim.dt),
                                     codec="libx264", quality=8, macro_block_size=16)
            summary = {"kind": "oracle_development_diagnostic", "config": env.config,
                       "episodes": len(results), "successes": sum(r["success"] for r in results),
                       "success_rate": float(np.mean([r["success"] for r in results])),
                       "not_a_research_result": True, "results": results}
            (output / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    finally:
        env.close()

if __name__ == "__main__":
    main()
