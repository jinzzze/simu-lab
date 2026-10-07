"""Refresh project documentation after all independent DP results are available."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]


def section_replace(text,start,end,content):
    if start in text:
        a=text.index(start);b=text.index(end,a)
        return text[:a]+content+'\n\n'+text[b:]
    return text.replace(end,content+'\n\n'+end,1)


def main():
    result=json.loads((ROOT/'artifacts/reports/diffusion_policy_v1/results.json').read_text())
    assert result['status']=='completed' and result['audit_passed']
    agg=result['aggregate'];rows=result['per_seed']
    table='\n'.join(f'| {r["seed"]} | {r["successes"]}/{r["n"]} | {r["qualified_lifts"]}/{r["n"]} | {r["validation"]["action_xyz_mean_error_mm"]:.3f} mm |' for r in rows)
    section=f'''## Independent Diffusion Policy extension

A user-requested DP-D group now learns future robot commands from **128 additional simulated expert trajectories**, with 16 complete validation trajectories. The frozen D_seed7 model supplies the initial RGB estimate; the action model also sees robot proprioception, the previous effective command and a robot clock. A compact conditional temporal U-Net learns action diffusion; 20-step DDIM predicts 16 commands and executes eight before replanning. Runtime does not invoke the scripted grasp-stage controller.

| Policy seed | Strict success | Qualified lifts | Held-out action XYZ error |
|---|---:|---:|---:|
{table}

Combined: **{agg['successes']}/{agg['executions']} ({agg['success_rate']*100:.2f}%)**, sharing 64 initial scenes. The existing D_seed7 fixed-controller reference is 64/64 on these same images. All three policy seeds use the same frozen visual checkpoint, fixed 5,000 training steps and final EMA weights; no best-seed selection. These extra action labels and different control/training budgets prevent a causal, budget-matched comparison with A/B/C/D.

This is initial-image state-conditioned, clock-conditioned action diffusion, not an end-to-end RGB-history policy or an exact reproduction of the published benchmark setup. Its robot action labels come from simulation, not the phone videos. All predeclared test scenes are retained; two test initial images also occur in training/validation, so the report includes a separate descriptive sensitivity check excluding those images. Existing experiment results remain unchanged.

[Implementation and limits](docs/DIFFUSION_POLICY.md) · [Measured DP report](artifacts/reports/diffusion_policy_v1/REPORT_ZH.md) · [All metrics](artifacts/reports/diffusion_policy_v1/results.json) · [First-scene video, seed 7](artifacts/runs/diffusion_policy_grasp_v1/DP_seed7/first_scene_demo.mp4)
'''
    path=ROOT/'README.md';text=path.read_text(encoding='utf-8-sig')
    text=section_replace(text,'## Independent Diffusion Policy extension','## Method',section.strip())
    text=text.replace('Video is sampled as frames; temporal action learning is not implemented.',
        'Personal video is sampled as frames; the separate DP extension learns temporal robot command chunks from simulated expert demonstrations, not human action sequences.')
    text=text.replace('There is still no VLA, learned action policy, demonstrated planning benefit or broad dynamics generalization.',
        'There is still no VLA, demonstrated world-model planning benefit or broad dynamics generalization. The separate DP extension learns actions from additional simulated robot demonstrations.')
    path.write_text(text,encoding='utf-8')
    path=ROOT/'docs/CURRENT_TASK.md';text=path.read_text(encoding='utf-8-sig')
    brief=f'''## 独立 Diffusion Policy 组

DP-D 已完成 128 条仿真专家训练轨迹、16 条验证轨迹以及三种子固定预算训练。沿用 D_seed7 初始 RGB 估计，另加机器人自身状态、上一有效动作与时钟，学习 16 步动作并滚动执行 8 步；部署不使用原抓取阶段状态机。

三个策略种子在共享 64 场景上共严格成功 {agg['successes']}/{agg['executions']}（{agg['success_rate']*100:.2f}%）。相同视觉模型配固定控制器的既有参照为 64/64。新增动作监督和控制器使其不属于 A/B/C/D 等预算消融；全部结果及重复图像敏感性检查见 [Diffusion Policy 报告](../artifacts/reports/diffusion_policy_v1/REPORT_ZH.md)与[实现说明](DIFFUSION_POLICY.md)。'''
    text=section_replace(text,'## 独立 Diffusion Policy 组','## 尚需补充',brief)
    path.write_text(text,encoding='utf-8')
    path=ROOT/'docs/CHALLENGE_ALIGNMENT.md';text=path.read_text(encoding='utf-8-sig')
    addition=f'- Independent DP-D learned-action extension: 128 simulated expert training trajectories, 16 validation trajectories, three final EMA policy seeds and {agg["successes"]}/{agg["executions"]} strict successes on 64 shared test scenes. Same frozen D7 visual model; additional action labels and different training/control budgets, so this is not a matched auxiliary-supervision comparison. See [implementation and caveats](DIFFUSION_POLICY.md).\n'
    if addition not in text:text=text.replace('- Source and data provenance,',addition+'- Source and data provenance,')
    text=text.replace('There is no VLA, temporal imitation, long-horizon learned dynamics validation or demonstrated planning benefit.',
        'There is no VLA, imitation of human action sequences, long-horizon learned dynamics validation or demonstrated world-model planning benefit. A separate DP-D policy now imitates simulated expert robot command sequences; its results and extra supervision are reported independently.')
    path.write_text(text,encoding='utf-8')
    path=ROOT/'docs/REPRODUCIBILITY.md';text=path.read_text(encoding='utf-8-sig')
    if '## Diffusion Policy extension' not in text:
        text+='''\n## Diffusion Policy extension

The separately trained action policy has a fixed plan in `configs/diffusion_policy.json`. Existing demonstrations and final checkpoints are local. `collect_diffusion_demonstrations.py` and `train_diffusion_policy.py` protect existing output directories. `evaluate_diffusion_policy.py` checks source/config/checkpoint identity before resuming missing scene rows; it saves every action chunk and effective command. `analyze_diffusion_policy.py` requires all three complete runs and audits paired scene images and data lineage.

The two additional test suites, `tests/test_diffusion_policy.py` and `tests/test_policy_environment.py`, must be run in separate Python processes due to native-library isolation. `record_diffusion_failures.py` replays the first failed recorded action stream per seed, verifies the physical outcome and does not create additional independent evaluation samples. The stored `--limit` evaluation prefix is diagnostic only and is excluded from the full report.

See [DP-D scope and commands](DIFFUSION_POLICY.md). Recompute its report with `.\\.env\\python.exe scripts\\analyze_diffusion_policy.py`, then rebuild the combined local page with `.\\.env\\python.exe scripts\\build_project_report.py`.
'''
    path.write_text(text,encoding='utf-8')
    path=ROOT/'docs/DIFFUSION_POLICY.md';text=path.read_text(encoding='utf-8-sig')
    old='三种子模型已训练；物理评测结果以'
    new=f'三种子模型及 192 次物理评测已完成，严格成功 {agg["successes"]}/{agg["executions"]}（{agg["success_rate"]*100:.2f}%）；逐种子结果与局限见'
    text=text.replace(old,new)
    path.write_text(text,encoding='utf-8')
    print('Updated README and four scope/reproduction documents from verified results')


if __name__=='__main__':main()
