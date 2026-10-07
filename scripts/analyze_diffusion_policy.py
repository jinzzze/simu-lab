"""Audit and report the independent DP-D extension; no selective partial summaries."""
from pathlib import Path
import hashlib,json
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/reports/diffusion_policy_v1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def main():
    cfg=read(ROOT/'configs/diffusion_policy.json')
    collected=read(ROOT/'data/manifests/diffusion_policy_v1.json')
    checks=[]
    def check(name,passed):
        checks.append({'check':name,'passed':bool(passed)})
        if not passed:raise AssertionError(name)
    check('collection completed and plan unchanged',collected['status']=='completed' and collected['plan']==cfg)
    for path,expected in collected['source_sha256'].items():check('collection source '+path,sha(ROOT/path)==expected)
    all_scene_ids=set();hashes={}
    for split in cfg['splits']:
        folder=ROOT/'data/processed/diffusion_policy_v1'
        check(split+' dataset SHA',sha(folder/f'{split}.npz')==collected['splits'][split]['dataset_sha256'])
        check(split+' metadata SHA',sha(folder/f'{split}_episodes.json')==collected['splits'][split]['episodes_sha256'])
        records=read(folder/f'{split}_episodes.json')
        hashes[split]={r['initial_rgb_sha256'] for r in records}
        with np.load(folder/f'{split}.npz',allow_pickle=False) as data:
            scenes=set(data['scene_seeds'].tolist())
            check(split+' scene split disjoint',not(all_scene_ids&scenes));all_scene_ids|=scenes
            check(split+' observation previous command is causal',np.allclose(data['observations'][:,1:,6:10],data['actions'][:,:-1],atol=1e-7))
            expected_xy=np.asarray([r['initial_visual_xy_m'] for r in records],np.float32)
            check(split+' only frozen vision provides initial XY',np.allclose(data['observations'][:,:,:2],expected_xy[:,None,:],atol=1e-7))
            check(split+' clock is pre-action timestep',np.allclose(data['observations'][:,:,10],np.arange(cfg['episode_steps'])[None,:]/cfg['episode_steps'],atol=1e-7))
    baseline_path=ROOT/'artifacts/runs/visual_grasp_v1/D_seed7/episodes.jsonl'
    baseline=[json.loads(x) for x in baseline_path.read_text().splitlines()]
    base={r['scene_seed']:r for r in baseline}
    scenes=list(range(cfg['evaluation']['start'],cfg['evaluation']['start']+cfg['evaluation']['count']))
    check('baseline identical full scenes',list(base)==scenes)
    check('DP train/validation scenes disjoint from test IDs',not all_scene_ids.intersection(scenes))
    test_hashes={r['initial_rgb_sha256'] for r in baseline}
    overlap_scenes=[r['scene_seed'] for r in baseline if r['initial_rgb_sha256'] in hashes['train']|hashes['val']]
    per_seed=[];all_rows=[];curves=[];total_seconds=0
    for seed in cfg['training']['seeds']:
        train_dir=ROOT/f'artifacts/runs/diffusion_policy_v1/DP_seed{seed}'
        eval_dir=ROOT/f'artifacts/runs/diffusion_policy_grasp_v1/DP_seed{seed}'
        train,measured=read(train_dir/'manifest.json'),read(eval_dir/'manifest.json')
        check(f'{seed} complete training/evaluation',train['status']==measured['status']=='completed')
        check(f'{seed} exact plan and final budget',train['config']==measured['plan']==cfg and train['steps_completed']==cfg['training']['steps'])
        check(f'{seed} checkpoint identity',sha(train_dir/'checkpoint.pt')==train['checkpoint_sha256']==measured['checkpoint_sha256'])
        check(f'{seed} data lineage',train['dataset_manifest_sha256']==sha(ROOT/'data/manifests/diffusion_policy_v1.json'))
        check(f'{seed} fixed visual checkpoint',measured['visual_checkpoint_sha256']==collected['visual_checkpoint_sha256']==sha(ROOT/cfg['visual_checkpoint']))
        for owner in (train,measured):
            for path,expected in owner['source_sha256'].items():check(f'{seed} source {path}',sha(ROOT/path)==expected)
        rows=[json.loads(x) for x in (eval_dir/'episodes.jsonl').read_text().splitlines()]
        check(f'{seed} full ordered paired evaluation',[r['scene_seed'] for r in rows]==scenes)
        check(f'{seed} episodes SHA',sha(eval_dir/'episodes.jsonl')==measured['episodes_sha256'])
        for row in rows:
            scene=row['scene_seed'];trace=eval_dir/f'scene_{scene}_actions.npz'
            check(f'{seed}/{scene} initial image identical to D7 baseline',row['initial_rgb_sha256']==base[scene]['initial_rgb_sha256'])
            check(f'{seed}/{scene} action trace',sha(trace)==row['action_trace_sha256'])
            check(f'{seed}/{scene} same sampling RNG across seeds',row['sampling_seed']==cfg['evaluation']['sampling_seed']+scene)
            check(f'{seed}/{scene} actual physical execution',row['object_attached_or_teleported'] is False and abs(row['simulated_duration_s']-cfg['episode_steps']*cfg['control_dt_s'])<1e-8)
        success=sum(r['success'] for r in rows)
        entry={'seed':seed,'successes':success,'n':len(rows),'success_rate':success/len(rows),
               'qualified_lifts':sum(r['lifted'] for r in rows),'validation':train['validation'],
               'training_seconds':train['elapsed_seconds'],
               'mean_chunk_inference_ms':1000*float(np.mean([r['inference_mean_seconds'] for r in rows])),
               'clipped_command_count':sum(r['clipped_actions'] for r in rows),
               'failures_without_lift':sum(not r['success'] and not r['lifted'] for r in rows),
               'failures_after_lift':sum(not r['success'] and r['lifted'] for r in rows),
               'terminal_flag_failures':{key:sum(not r['success'] and not r['final_flags'][key] for r in rows) for key in rows[0]['final_flags']}}
        per_seed.append(entry);all_rows.extend(rows);total_seconds+=train['elapsed_seconds']
        curves.append([json.loads(x)['noise_mse'] for x in (train_dir/'losses.jsonl').read_text().splitlines()])
    OUT.mkdir(parents=True,exist_ok=True)
    result={'status':'completed','plan':cfg,'collection':collected['splits'],'per_seed':per_seed,
            'aggregate':{'successes':sum(r['success'] for r in all_rows),'executions':len(all_rows),
                'success_rate':float(np.mean([r['success'] for r in all_rows])),
                'qualified_lifts':sum(r['lifted'] for r in all_rows),'distinct_test_scenes':len(scenes),
                'training_seconds_total':total_seconds},
            'fixed_controller_reference':{'group':'D_seed7','successes':sum(r['success'] for r in baseline),'n':len(baseline),
                'simulated_duration_range_s':[min(r['outcome']['simulated_duration_s'] for r in baseline),max(r['outcome']['simulated_duration_s'] for r in baseline)],
                'episodes_sha256':sha(baseline_path),'scope':'Existing 64 executions with the same frozen visual model; no added reruns'},
            'overlap_scene_ids':overlap_scenes,
            'sensitivity_excluding_identical_initial_images':{
                'scope':'Post-evaluation descriptive sensitivity check, not a replacement for the predeclared full test',
                'per_seed':[{'seed':s, 'n':sum(r['policy_seed']==s and r['scene_seed'] not in overlap_scenes for r in all_rows),
                             'successes':sum(r['policy_seed']==s and r['scene_seed'] not in overlap_scenes and r['success'] for r in all_rows)} for s in cfg['training']['seeds']]},
            'rendered_image_overlap':{'train_val':len(hashes['train']&hashes['val']),
                'train_test':len(hashes['train']&test_hashes),'val_test':len(hashes['val']&test_hashes)},
            'limitations':['Extra expert action labels and larger optimizer budget: not matched to A/B/C/D.',
                'One frozen D7 initial-frame visual estimator for three policy seeds; not three independent perception models.',
                'Clock-conditioned robot-state action diffusion; not end-to-end RGB-history training.',
                'Reused development test scenes with fixed appearance/physics; no new real-world generalization evidence.'],
            'analysis_source_sha256':sha(Path(__file__)),'audit_passed':all(c['passed'] for c in checks)}
    write(OUT/'results.json',result);write(OUT/'audit.json',{'passed':True,'checks':checks})
    (OUT/'episodes.jsonl').write_text('\n'.join(json.dumps(r) for r in all_rows)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    axes[0].bar(['D7 fixed\ncontroller']+[f'DP seed {r["seed"]}' for r in per_seed],
                [100*sum(r['success'] for r in baseline)/len(baseline)]+[r['success_rate']*100 for r in per_seed],
                color=['#84949f','#087b80','#087b80','#087b80'])
    axes[0].set(ylabel='Strict physical success (%)',ylim=(0,110),title='Same 64 initial scenes; different action supervision')
    for i,(seed,curve) in enumerate(zip(cfg['training']['seeds'],curves)):
        axes[1].plot(np.arange(100,len(curve)+1),np.convolve(curve,np.ones(100)/100,'valid'),label=f'seed {seed}')
    axes[1].set(xlabel='Optimizer step',ylabel='Noise prediction MSE',title='100-step trailing mean');axes[1].legend()
    fig.savefig(OUT/'results.png',dpi=180);plt.close(fig)
    media=[]
    for seed in cfg['training']['seeds']:
        source=ROOT/f'artifacts/runs/diffusion_policy_grasp_v1/DP_seed{seed}/first_scene_demo.mp4'
        cap=cv2.VideoCapture(str(source));frames=[];fps=cap.get(cv2.CAP_PROP_FPS)
        while True:
            ok,frame=cap.read()
            if not ok:break
            frames.append(frame)
        cap.release();assert len(frames)>0
        chosen=[frames[i] for i in np.linspace(0,len(frames)-1,6).round().astype(int)]
        cv2.imwrite(str(OUT/f'DP_seed{seed}_contact_sheet.jpg'),np.vstack([np.hstack(chosen[:3]),np.hstack(chosen[3:])]))
        media.append({'seed':seed,'scene':scenes[0],'frames':len(frames),'fps':fps,'source_sha256':sha(source)})
    write(OUT/'media_validation.json',media)
    lines=['# Diffusion Policy：独立追加实验','','## 方法与输入边界','',
        'DP-D 使用已有 D_seed7 模型从初始 RGB 估计方块 XY，冻结该视觉模块。条件包括该初始估计、机器人末端位置、夹爪宽度、上一动作和运行时钟。策略学习 16 步末端 XYZ＋夹爪命令，每次执行前 8 步后重新读取机器人状态预测。部署没有调用原来的抓取阶段状态机，真值只参与评分。','',
        '这是参考 Diffusion Policy 条件动作扩散思想的紧凑实现：FiLM 时序 U-Net、100 步余弦噪声训练、20 步确定性 DDIM、最终 EMA 权重。不是官方全部配置的复刻，也不是端到端多帧 RGB 策略。','',
        '新增 128 条训练和 16 条验证仿真专家轨迹（均实际成功），每条 320 个控制步。专家使用仿真真实初始位置生成动作；个人视频只用于上游视觉预训练，没有从手部关键点伪造机器人动作。三个策略种子各固定训练 5,000 步。','',
        '## 实际物理评测','','| 策略 | 严格成功 | 合格抬升 | 验证动作 XYZ 误差 mm |', '|---|---:|---:|---:|',
        f'| 原 D7＋固定控制器（既有参照） | {result["fixed_controller_reference"]["successes"]}/64 | — | — |']
    for r in per_seed:lines.append(f'| DP 种子 {r["seed"]} | {r["successes"]}/{r["n"]} | {r["qualified_lifts"]}/{r["n"]} | {r["validation"]["action_xyz_mean_error_mm"]:.3f} |')
    total=result['aggregate']
    lines+=['',f'合计 {total["successes"]}/{total["executions"]}，成功率 {total["success_rate"]*100:.2f}%。192 次执行重复使用 64 个场景；三个策略种子使用同一个视觉模型。没有选择最好检查点、种子或测试子集。','',
        '![成功率和训练曲线](results.png)','','## 如何解读','',
        'DP 每回合固定执行 12.8 秒，原固定控制器在完成阶段后结束（具体时长范围见 results.json），执行时长也不相同。与原方法的控制器及动作监督不同；不能把成功率差异归因于辅助标签或扩散算法本身。固定控制器针对这个简单任务已经非常有效。仅离线动作误差或训练损失较低不保证闭环抓取成功；偏差可能在连续执行中累积。','',
        f'数据与输入审计 {len(checks)} 项通过。初始测试图像与 D7 参照逐场景完全一致。跨集合完全相同的渲染图像数量：{result["rendered_image_overlap"]}；独立 scene seed 不保证像素唯一。','',
        '## 相同初始图像的追加敏感性检查','',
        f'训练/验证与测试存在相同像素的初始图像；对应测试场景为 {overlap_scenes}。它们使用不同的 scene seed，但不属于视觉独立样本。主表保留所有预定场景，没有根据结果删除。额外排除这些初始图像后，各种子成功数/次数：'+str(result['sensitivity_excluding_identical_initial_images']['per_seed'])+'。这只是事后描述性核查。','',
        '## 证据和复现','','[固定方案](../../../configs/diffusion_policy.json) · [完整指标](results.json) · [审计](audit.json) · [逐回合记录](episodes.jsonl)','',
        '采集/训练入口保护已有数据与模型，不能直接覆盖重跑。评测仅在模型、配置、源代码一致时续跑尚未完成的场景。报告可从完整工件重新生成。','',
        '```powershell', '. .\\scripts\\Enter-Project.ps1', '.\\.env\\python.exe scripts\\analyze_diffusion_policy.py', '```','',
        '实现入口：scripts/collect_diffusion_demonstrations.py、train_diffusion_policy.py、evaluate_diffusion_policy.py。原 A/B/C/D/R 与世界模型工件保持不变。','',
        '[方法来源：Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) · [官方实现](https://github.com/real-stanford/diffusion_policy)','']
    failure_lines=['## 失败分解','','| 策略种子 | 未合格抬升 | 抬升后仍失败 |','|---|---:|---:|']
    for r in per_seed:
        failure_lines.append(f'| {r["seed"]} | {r["failures_without_lift"]} | {r["failures_after_lift"]} |')
    failure_lines += ['', '终态各失败标志为非互斥计数，见 results.json；例如未完全越线、物体翻倒可能同时发生。首次失败回放从已保存动作重放，不再次抽样扩散动作，也不增加独立评测样本。', '', '[失败回放核对](failure_review/summary.json)', '']
    lines += failure_lines
    (OUT/'REPORT_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'aggregate':total,'per_seed':per_seed,'audit_checks':len(checks)},indent=2),flush=True)


if __name__=='__main__':main()
