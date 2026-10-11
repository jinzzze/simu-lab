"""Post-hoc sensitivity to duplicate initial pixels; retain the primary results."""
import numpy as np
from common import ROOT,read,write,sha,now


def totals(rows):
    return dict(executions=len(rows),successes=sum(r['success'] for r in rows),lifts=sum(r['lifted'] for r in rows),
        physical_starts=len({r['scene_seed'] for r in rows}))


def main():
    source=ROOT/'artifacts/results_v1/results.json';result=read(source);assert result['status']=='completed'
    out=ROOT/'artifacts/initial_overlap_audit_v1';out.mkdir(parents=True,exist_ok=False)
    index=read(ROOT/'artifacts/robot_demos_v1/training_index.json')
    references=index['train_groups']['geometry_only_24']+index['shared_validation']
    cases=result['study']['test_cases'];overlaps=result['test_initial_pixel_overlap']
    excluded={r['case_id'] for r in overlaps if r['matches_training'] or r['matches_development']}
    affected={r['scene_seed'] for r in cases if r['case_id'] in excluded}
    matches=[]
    rows=result['all_cases']
    for case in cases:
        if case['case_id'] not in excluded:continue
        test=next(r for r in rows if r['suite']=='test' and r['case_id']==case['case_id'] and r['condition']=='geometry_only_24' and r['policy_seed']==7)
        trace=read(ROOT/f"artifacts/evaluation/test/geometry_only_24_seed7/{case['case_id']}_scoring.json")
        xy=np.array(trace[0]['object_xyz_m'])[:2]
        for ref in references:
            if ref['initial_rgb_sha256']==test['initial_rgb_sha256']:
                matches.append(dict(test_case=case['case_id'],test_scene_seed=case['scene_seed'],reference_episode=ref['episode_id'],
                    reference_split=ref['split'],initial_rgb_sha256=test['initial_rgb_sha256'],
                    xy_distance_from_reference_mm=float(np.linalg.norm(xy-ref['initial_truth_xy_m'])*1000),
                    metric_note='Test scoring pose after first action versus reference pre-action pose; diagnostic only'))
    summaries={}
    for name,accept in [('all_predeclared',lambda r:True),('exclude_4_matching_task_images',lambda r:r['case_id'] not in excluded),
                        ('exclude_all_goals_for_2_affected_starts',lambda r:r['scene_seed'] not in affected)]:
        summaries[name]={}
        for condition in result['study']['conditions']:
            kept=[r for r in rows if r['suite']=='test' and r['condition']==condition and accept(r)]
            summaries[name][condition]=dict(all=totals(kept),heldout=totals([r for r in kept if r['heldout_pair']]),
                per_seed={str(s):totals([r for r in kept if r['policy_seed']==s]) for s in result['study']['seeds']})
    receipt=dict(status='completed',created_utc=now(),primary_results_sha256=sha(source),
        exact_pixel_matches=matches,excluded_case_ids=sorted(excluded),affected_scene_seeds=sorted(affected),summaries=summaries,
        policy_runs_added=0,primary_results_modified=False,post_hoc=True,code_sha256=sha(ROOT/'audit_initial_overlap.py'),
        scope='Descriptive duplicate-pixel sensitivity, not a replacement benchmark or new independent confirmation. Scene seeds are disjoint but rendered pixels can repeat.')
    write(out/'receipt.json',receipt)
    lines=['# 初始图像重合与敏感性检查','',
        '训练、验证、测试的随机种子互不相同，但 4 个测试任务的初始 RGB 与已有训练/验证图像完全相同，涉及 2 个测试物理起点。随机种子不同不保证渲染像素独立。', '',
        '以下两种排除规则都是发现重合后的补充分析。原始 264 次执行和原始结果全部保留；没有补跑、替换或挑选新场景。', '',
        '|范围|共同几何|额外视频时序/进度|每组物理起点数|','|---|---:|---:|---:|']
    labels=['原始全部测试','排除 4 个相同图像任务','排除 2 个相关物理起点的所有目标']
    for label,groups in zip(labels,summaries.values()):
        cells=[f"{r['all']['successes']}/{r['all']['executions']} ({100*r['all']['successes']/r['all']['executions']:.1f}%)" for r in groups.values()]
        lines.append('|'+label+'|'+'|'.join(cells)+f"|{next(iter(groups.values()))['all']['physical_starts']}|")
    lines+=['','|测试任务|重合来源|用途|评分位置与来源初态差距|','|---|---|---|---:|']
    for m in matches:lines.append(f"|{m['test_case']}|{m['reference_episode']}|{m['reference_split']}|{m['xy_distance_from_reference_mm']:.3f} mm|")
    lines+=['','坐标距离仅用于检查：测试位置取第一次命令后的评分记录，来源位置取初态；不把它解释成定位误差。',
        '', '当前证据不支持“完全未见初始图像”的主张。排除后样本更小，各模型种子与目标之间仍共享物理起点；不由这些数值推断统计显著性。']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(__import__('json').dumps(summaries,ensure_ascii=False))


if __name__=='__main__':main()
