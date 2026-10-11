"""Training-set mechanism diagnostic on matched physical starts and changed goals."""
import json
import numpy as np
import torch
from common import ROOT,OLD,read,write,sha,now,check_sources
from paired_recurrent_goal_policy import GoalRelativeRecurrentPolicy


def first(entry):
    path=ROOT/entry['file'];assert sha(path)==entry['dataset_sha256']
    with np.load(path,allow_pickle=False) as d:
        return d['rgb'][0:1].copy(),d['observations'][0:1].copy(),d['actions'][0].copy()


def main():
    out=ROOT/'artifacts/initial_goal_coupling_v1';out.mkdir(parents=True,exist_ok=False)
    index=read(ROOT/'artifacts/robot_demos_v1/training_index.json');cfg=read(ROOT/'configs/policy_v1.json')
    torch.set_num_threads(4)
    pairs=[]
    for a,b in zip(index['train_groups']['paired_24'],index['train_groups']['coverage_24']):
        assert a['scene_seed']==b['scene_seed'] and a['start_id']==b['start_id']
        if a['goal_id']==b['goal_id']:
            continue
        aa,bb=first(a),first(b)
        assert np.array_equal(aa[1][:,:4],bb[1][:,:4]) and np.array_equal(aa[1][:,8:],bb[1][:,8:])
        assert np.array_equal(aa[2],bb[2]),'Initial teacher target must be independent of destination'
        pairs.append((a,b,aa,bb))
    assert len(pairs)==16
    rows=[];sources={}
    for condition in cfg['conditions']:
        for seed in cfg['training']['seeds']:
            folder=ROOT/f'artifacts/models/{condition}_seed{seed}';m=read(folder/'manifest.json')
            assert m['status']=='completed';check_sources(m['source_sha256'])
            cp=folder/'checkpoint.pt';assert sha(cp)==m['checkpoint_sha256'];policy=GoalRelativeRecurrentPolicy(cp)
            sources[str(cp.relative_to(ROOT)).replace('\\','/')]=sha(cp)
            for a,b,aa,bb in pairs:
                predictions=[]
                for rgb,obs,_ in (aa,bb):
                    policy.reset(0);predictions.append(policy.predict(rgb,obs)[0])
                delta=(predictions[1]-predictions[0])[:2];goal_delta=(bb[1]-aa[1])[0,4:6]
                rows.append(dict(condition=condition,seed=seed,scene_seed=a['scene_seed'],start_id=a['start_id'],
                    first_goal_id=a['goal_id'],second_goal_id=b['goal_id'],
                    decoded_first_command_change_mm=float(np.linalg.norm(delta)*1000),
                    goal_shift_mm=float(np.linalg.norm(goal_delta)*1000),
                    decoder_goal_translation_compensation_fraction=float(1-delta@goal_delta/(goal_delta@goal_delta)),
                    teacher_first_command_change_mm=0.))
            del policy
    summary={c:dict(comparisons=sum(r['condition']==c for r in rows),
        median_first_command_change_mm=float(np.median([r['decoded_first_command_change_mm'] for r in rows if r['condition']==c])),
        per_seed_median_mm={str(s):float(np.median([r['decoded_first_command_change_mm'] for r in rows if r['condition']==c and r['seed']==s])) for s in cfg['training']['seeds']}) for c in cfg['conditions']}
    result=dict(created_utc=now(),summary=summary,comparisons=rows,checkpoint_sha256=sources,
        training_index_sha256=sha(ROOT/'artifacts/robot_demos_v1/training_index.json'),code_sha256=sha(ROOT/'diagnose_initial_goal_coupling.py'),
        scope='Post-hoc training-set mechanism diagnostic;16 matched starts each given2 consistent numeric/visual goals, repeated for3 seeds. No new closed-loop episodes or model selection.',
        interpretation='Teacher first action is exactly equal for each paired input. Smaller learned initial-command change supports reduced destination/start coupling on these observed inputs; does not establish unseen-task success.')
    write(out/'results.json',result)
    lines=['# 初始抓取命令对目标变化的敏感性','',
        '从本轮训练数据取16组相同物体/机器人初态，分别提供两个一致的数值目标和视觉高亮。教师的首个命令逐值相同；模型每次清空循环状态。', '',
        '| 训练方式 | 首个解码XY命令变化中位数 | seed7 / seed17 / seed27 中位数 |', '|---|---:|---|']
    for c,s in summary.items():
        values=' / '.join(f'{v:.2f}' for v in s['per_seed_median_mm'].values())
        lines.append(f"| {c} | {s['median_first_command_change_mm']:.2f} mm | {values} mm |")
    lines += ['', '这是训练输入上的事后机制诊断，不是新增独立测试，也不单独证明泛化。数值和视觉目标同时改变，不能把作用唯一归给某个输入。所有闭环主结果仍以冻结评测为准。',
        '', '[逐输入结果及检查点哈希](results.json)']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    main()
