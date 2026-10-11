"""Verify paired optimization budgets and that all24 episodes reached training."""
import json
from collections import Counter
from common import ROOT,read,write,sha,now,check_sources,verify_training_pairs


def main():
    out=ROOT/'artifacts/training_audit_v1';out.mkdir(parents=True,exist_ok=False)
    cfg=read(ROOT/'configs/policy_v1.json');index=read(ROOT/cfg['training_index'])
    assert index['dataset_manifest_sha256']==sha(ROOT/cfg['dataset_manifest'])
    models={};runs=[]
    for condition in cfg['conditions']:
        entries=index['train_groups'][condition];verify_training_pairs(entries)
        assert len(entries)==24
        for field,prefix in [('start_id','P'),('goal_id','G')]:
            assert Counter(r[field] for r in entries)=={f'{prefix}{i}':6 for i in range(1,5)}
        for seed in cfg['training']['seeds']:
            folder=ROOT/f'artifacts/models/{condition}_seed{seed}';m=read(folder/'manifest.json')
            assert m['status']=='completed' and m['config']==cfg
            assert sha(folder/'checkpoint.pt')==m['checkpoint_sha256'];check_sources(m['source_sha256'])
            log=[json.loads(x) for x in (folder/'losses.jsonl').read_text(encoding='utf-8').splitlines()]
            assert [x['step'] for x in log]==list(range(1,2501))
            counts=m['episode_sampling_counts']
            assert len(counts)==24 and min(counts)>0 and sum(counts)==5000
            expected=sum(n*r['steps'] for n,r in zip(counts,entries))
            assert expected==m['valid_action_targets_seen']
            models[(condition,seed)]=m
            runs.append(dict(condition=condition,seed=seed,parameters=m['parameters'],updates=len(log),
                episode_draws=sum(counts),minimum_episode_draws=min(counts),
                supervised_action_targets=expected,wall_s=m['wall_s'],manifest_sha256=sha(folder/'manifest.json')))
    pairs=[]
    for seed in cfg['training']['seeds']:
        a,b=[models[(c,seed)] for c in cfg['conditions']]
        for key in ('initial_weights_sha256','episode_stream_sha256','augmentation_stream_sha256','steps_completed','parameters'):
            assert a[key]==b[key],key
        pairs.append(dict(seed=seed,initial_weights_matched=True,sampled_episode_indices_matched=True,augmentation_parameters_matched=True,
                          target_counts=[a['valid_action_targets_seen'],b['valid_action_targets_seen']]))
    result=dict(status='passed',created_utc=now(),runs=runs,paired_checks=pairs,all24_episodes_sampled=True,
        code_sha256=sha(ROOT/'audit_training.py'),scope='Matched models, updates, episode samples and augmentations. Frame-target counts and measured compute differ with trajectory length; no equal-compute claim.')
    write(out/'receipt.json',result)
    lines=['# 训练预算与配对审计','',
        '六个模型均完成2500次更新；24条轨迹均实际被采样，未遗漏第17–24条。两组每个P和G各出现6次。',
        '三个种子的完整初始权重、轨迹采样序列和实际增强参数序列分别配对一致。', '',
        '| 条件 | 种子 | 更新数 | 轨迹抽样次数 | 有效动作目标数 | 训练秒数 |','|---|---:|---:|---:|---:|---:|']
    for r in runs:
        lines.append(f"| {r['condition']} | {r['seed']} | {r['updates']} | {r['episode_draws']} | {r['supervised_action_targets']} | {r['wall_s']:.1f} |")
    lines+=['','有效目标数按日志抽样次数×每条真实长度复算，和训练记录一致。不同轨迹长度造成帧数及计算量差异，不能称为完全等算力。','', '[完整审计收据](receipt.json)']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
