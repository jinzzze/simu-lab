"""Summarize only completed, audited learned-policy rollouts; no model selection."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import argparse, hashlib, json, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT=Path(__file__).resolve().parent
CONDITIONS=('simulation_only','video_guided')
LABELS={'simulation_only':'仿真对照','video_guided':'视频引导'}

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def category(row):
    if row['success']:return 'success'
    if not row['lifted']:return 'no_qualified_lift'
    if not row['final_flags']['inside_goal']:return 'lifted_final_footprint_outside_goal'
    if not row['final_flags']['released']:return 'inside_goal_not_released'
    if not row['final_flags']['withdrawn']:return 'inside_goal_not_withdrawn'
    return 'inside_goal_other_or_insufficient_stability'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--suite',default='validation');parser.add_argument('--analysis-revision',default='analysis_v1')
    args=parser.parse_args();cfg=read(ROOT/args.config);rev=cfg['revision']
    audit=ROOT/f'artifacts/reports/{rev}/{args.suite}_validation.json'
    assert audit.exists(),'Run full source/causal/scoring validator first'
    receipt=read(audit);assert receipt['passed'] and receipt['revision']==rev and receipt['suite']==args.suite
    evalroot=ROOT/f'artifacts/runs/{rev}_evaluation/{args.suite}'
    out=ROOT/f'artifacts/reports/{rev}/{args.suite}_{args.analysis_revision}'
    assert not out.exists(),'Preserve previous analysis';out.mkdir(parents=True)
    source={str(audit.relative_to(ROOT)):sha(audit),args.config:sha(ROOT/args.config),'analyze_paired_policy.py':sha(__file__)}
    rows=[];models={};by={};scenes=None
    for condition in CONDITIONS:
        for seed in cfg['training']['seeds']:
            folder=evalroot/f'{condition}_seed{seed}';meta=read(folder/'manifest.json')
            assert meta['status']=='completed' and meta['config']==cfg
            erows=[json.loads(x) for x in (folder/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
            assert len(erows)==meta['count_planned'] and sha(folder/'episodes.jsonl')==meta['episodes_sha256']
            ids=[r['scene_seed'] for r in erows]
            if scenes is None:scenes=ids
            assert scenes==ids
            trainfile=ROOT/f'artifacts/runs/{rev}/{condition}_seed{seed}/manifest.json'
            training=read(trainfile);assert training['status']=='completed'
            for path in [folder/'manifest.json',folder/'episodes.jsonl',trainfile]:source[str(path.relative_to(ROOT))]=sha(path)
            models[f'{condition}_seed{seed}']={'successes':sum(r['success'] for r in erows),'episodes':len(erows),
                'lifted':sum(r['lifted'] for r in erows),'teacher_forced_error':training['validation'],
                'categories':dict(Counter(category(r) for r in erows)),
                'valid_action_targets_seen':training.get('valid_action_targets_seen'),
                'limited_commands':sum(r['limited_commands'] for r in erows)}
            for row in erows:
                row=dict(row);row['failure_category']=category(row)
                row['missing_final_flags']=[key for key,value in row['final_flags'].items() if not value]
                row['final_xy_error_mm']=float(np.linalg.norm(np.array(row['final_object_xyz_m'][:2])-row['goal_xy_m'])*1000)
                rows.append(row);by[(condition,seed,row['scene_seed'])]=row
    assert len(rows)==receipt['executed_rollouts_checked']
    summaries={}
    for condition in CONDITIONS:
        group=[r for r in rows if r['condition']==condition]
        summaries[condition]={'successes':sum(r['success'] for r in group),'episodes':len(group),'qualified_lifts':sum(r['lifted'] for r in group),
            'failure_categories':dict(Counter(category(r) for r in group)),
            'missing_final_flags_including_no_lift':dict(Counter(key for r in group if not r['success'] for key in r['missing_final_flags'])),
            'goals':{f'G{i}':{'successes':sum(r['success'] for r in group if r['goal_id']==f'G{i}'),
                'episodes':sum(r['goal_id']==f'G{i}' for r in group)} for i in range(1,5)}}
    pairs=Counter()
    for seed in cfg['training']['seeds']:
        for scene in scenes:
            a=by[('simulation_only',seed,scene)];b=by[('video_guided',seed,scene)]
            assert a['initial_rgb_sha256']==b['initial_rgb_sha256']
            pairs[{(False,False):'neither',(True,False):'simulation_only',(False,True):'video_guided',(True,True):'both'}[(a['success'],b['success'])]]+=1
    result={'revision':rev,'suite':args.suite,'analysis_utc':datetime.now(timezone.utc).isoformat(),
        'scene_ids':scenes,'independent_initial_scenes':len(scenes),'training_seeds':cfg['training']['seeds'],
        'groups':summaries,'per_model':models,'matched_pair_success_table':dict(pairs),
        'paired_difference_video_minus_simulation_count':summaries['video_guided']['successes']-summaries['simulation_only']['successes'],
        'statistics_scope':'descriptive only; seeds reuse scenes; no independent-binomial CI or significance claim',
        'failure_category_scope':'ordered observed outcomes, not inferred physical root causes',
        'validation_scope':'shared development scenes' if args.suite=='validation' else 'new simulator scenes for this frozen revision; no real robot or human heldout evaluation',
        'latency_scope':'other training/evaluation processes may overlap; timings are not hardware benchmarks',
        'source_sha256':source,'episodes':rows}
    write(out/'results.json',result)
    # The rows show all seeds and all scenes; no selected best model.
    matrix=np.array([[2 if by[(c,s,scene)]['success'] else (1 if by[(c,s,scene)]['lifted'] else 0) for scene in scenes] for c in CONDITIONS for s in cfg['training']['seeds']])
    fig,ax=plt.subplots(figsize=(max(8,len(scenes)*.6),4.3));cmap=matplotlib.colors.ListedColormap(['#e7e7e7','#f0b44c','#3a9971'])
    ax.imshow(matrix,cmap=cmap,vmin=0,vmax=2,aspect='auto');ax.set_xticks(range(len(scenes)),[str(s) for s in scenes],rotation=60)
    ax.set_yticks(range(6),[f'{c} / seed {s}' for c in CONDITIONS for s in cfg['training']['seeds']])
    for y in range(6):
        for x in range(len(scenes)):ax.text(x,y,['no lift','lift','success'][matrix[y,x]],ha='center',va='center',fontsize=7)
    ax.set_title(f'{rev} / {args.suite}: all retained executions');fig.tight_layout();fig.savefig(out/'all_outcomes.png',dpi=160);plt.close(fig)
    # Fixed first scene, seed7: post-action truth is used only in this diagnostic plot.
    fixed=cfg['training']['seeds'][0];scene=scenes[0];fig,axes=plt.subplots(2,3,figsize=(14,7))
    for line,condition in enumerate(CONDITIONS):
        folder=evalroot/f'{condition}_seed{fixed}';trace_path=folder/f'scene_{scene}_scoring.json'
        trace=read(trace_path);source[str(trace_path.relative_to(ROOT))]=sha(trace_path)
        obj=np.array([t['object_xyz_m'] for t in trace])*1000;ee=np.array([t['ee_xyz_m'] for t in trace])*1000
        t=np.array([x['t_s'] for x in trace]);goal=np.array(by[(condition,fixed,scene)]['goal_xy_m'])*1000
        ax=axes[line,0];ax.plot(ee[:,0],ee[:,1],label='EE');ax.plot(obj[:,0],obj[:,1],label='object scoring truth')
        ax.add_patch(Rectangle(goal-20,40,40,fill=False,color='#286abc'));ax.scatter(*goal,marker='+',color='#286abc');ax.set_aspect('equal',adjustable='datalim')
        ax.set_xlabel('world X / mm');ax.set_ylabel('world Y / mm');ax.legend(fontsize=7);ax.set_title(condition)
        axes[line,1].plot(t,ee[:,2],label='EE Z');axes[line,1].plot(t,obj[:,2],label='object center Z');axes[line,1].set_xlabel('simulation time / s');axes[line,1].set_ylabel('Z / mm');axes[line,1].legend(fontsize=8)
        axes[line,2].plot(t,np.array([x['finger_width_m'] for x in trace])*1000,label='measured width')
        axes[line,2].plot(t,np.array([x['clearance_m'] for x in trace])*1000,label='object bottom clearance')
        axes[line,2].set_xlabel('simulation time / s');axes[line,2].set_ylabel('mm');axes[line,2].legend(fontsize=8)
    fig.suptitle(f'Fixed first scene {scene}, seed {fixed}; object truth is diagnostic only');fig.tight_layout();fig.savefig(out/'first_scene_diagnostic.png',dpi=150);plt.close(fig)
    write(out/'results.json',result)
    comparison='没有测到视频引导的成功数优势。' if result['paired_difference_video_minus_simulation_count']<=0 else '视频引导成功数较高；这只是本版本仿真描述结果，不能推出真实迁移或显著收益。'
    text=[f'# {rev}：{args.suite} 实际执行结果','',f'仿真对照 {summaries["simulation_only"]["successes"]}/{summaries["simulation_only"]["episodes"]}，视频引导 {summaries["video_guided"]["successes"]}/{summaries["video_guided"]["episodes"]}。{comparison}',
          '',f'每组 {len(cfg["training"]["seeds"])} 个固定最终 EMA 模型复用相同 {len(scenes)} 个初始场景。没有挑选种子、检查点或成功场景；失败保留。validation 是开发集。其他 suite 是本版本预先冻结的新仿真场景，均不是独立真实视频/真实机器人测试。',
          '', '| 条件 | 种子 | 严格成功 | 合格抬升 |','|---|---:|---:|---:|']
    for condition in CONDITIONS:
        for seed in cfg['training']['seeds']:
            m=models[f'{condition}_seed{seed}'];text.append(f'| {LABELS[condition]} | {seed} | {m["successes"]}/{m["episodes"]} | {m["lifted"]}/{m["episodes"]} |')
    text+=['','![全部结果](all_outcomes.png)','','## 数据和执行边界','',
        '两组各使用 16 条匹配的机器人训练示范；视频引导组的时长和归一化搬运进度由本人拍摄的 S01/train 视频离线约束。机械臂轨迹、高度和动作标签仍由工程轨迹与仿真执行产生。没有恢复人类三维动作，也没有把手机画面直接作为这些策略的视觉训练输入。',
        '', '在线输入是当前/历史 RGB、机器人自身状态、事前目标 XY 和上一有效命令。循环版本另有因果学习记忆，可能编码时间信息；没有外部绝对时钟、专家阶段或物体真值。部署仅模型动作加统一 IK/限幅，不用评分纠正动作或提前终止。',
        '', '成功保持原严格阈值：双指接触合格抬升、完整物体轮廓进入指定 40 mm 方框、落桌、正立、松爪、撤离和稳定。P1→G1 至 P4→G4 的起点与目标相关，不能声称任意目标泛化。',
        '', 'v1a/v2 使用 6000 步扩散训练；v3 使用 2500 步整轨迹循环 BC。每个版本内部两组配对，跨架构的训练目标数量与算力不同，不能当作等预算架构消融。',
        '', '## 失败诊断','', '下面是按固定顺序归类的观测结果，不是接触根因鉴定；未形成抬升的回合也可能同时未入框。']
    names={'success':'严格成功','no_qualified_lift':'无合格抬升','lifted_final_footprint_outside_goal':'已抬升但最终轮廓未入框','inside_goal_not_released':'在框内但未松爪','inside_goal_not_withdrawn':'在框内但未撤离','inside_goal_other_or_insufficient_stability':'在框内但其他条件/稳定不足'}
    text+=['','| 观测结果 | 仿真对照 | 视频引导 |','|---|---:|---:|']
    for name,label in names.items():text.append(f'| {label} | {summaries[CONDITIONS[0]]["failure_categories"].get(name,0)} | {summaries[CONDITIONS[1]]["failure_categories"].get(name,0)} |')
    text+=['',f'匹配成功表：双方成功 {pairs["both"]}，仅仿真对照成功 {pairs["simulation_only"]}，仅视频引导成功 {pairs["video_guided"]}，双方失败 {pairs["neither"]}。重复场景与三个种子不足以支持独立二项式显著性结论。',
        '', '![固定首场景诊断](first_scene_diagnostic.png)', '',f'上图固定 seed {fixed}、首场景 {scene}；物体坐标来自动作后的评分日志，仅用于诊断。小教师强制动作误差不等于闭环任务成功。',
        '', 'S02/S03/S04 未拍摄，个人视频严格成功标签与真实机器人标定未确认。需要本人协作的步骤已记录并跳过。未提交申请或发布新实验。',
        '', '[全部数值、逐回合结果和来源 SHA256](results.json) · [独立输入/评分核验](../'+args.suite+'_validation.json)']
    (out/'REPORT_ZH.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    shutil.copy2(__file__,out/'analyze_paired_policy.py')
    write(out/'completion_receipt.json',{'analysis_source_sha256':sha(__file__),'input_sha256':source,'output_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()},'created_utc':datetime.now(timezone.utc).isoformat()})
    print(json.dumps({'report':str(out/'REPORT_ZH.md'),'groups':summaries,'matched_success':dict(pairs)},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
