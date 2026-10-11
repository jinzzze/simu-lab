"""Apply the frozen development gate; expand fixed seeds only when it passes."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time
from common import ROOT,PROJECT,read,write,sha,now,check_sources


def run(script,args,logname):
    path=ROOT/'artifacts'/logname
    with path.open('x',encoding='utf-8') as log:
        p=subprocess.run([sys.executable,'-B','-u',str(ROOT/script)]+args,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,
                         creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if p.returncode:
        raise RuntimeError(f'{script} failed; see {path}')


def main():
    plan=read(ROOT/'configs/waypoint_pilot_study_v1.json')
    out=ROOT/'artifacts/waypoint_development_v1';out.mkdir(parents=True,exist_ok=False)
    meta=dict(status='waiting_for_seed7',created_utc=now(),plan_sha256=sha(ROOT/'configs/waypoint_pilot_study_v1.json'))
    write(out/'manifest.json',meta)
    path=ROOT/'artifacts/waypoint_pilot_evaluation/validation/waypoint_24_seed7/manifest.json'
    deadline=time.monotonic()+2400
    try:
        while True:
            if time.monotonic()>deadline:
                raise TimeoutError('Pilot did not complete within40 minutes')
            try:
                current=read(path)
                if current['status']=='failed':
                    raise RuntimeError(current.get('error','Pilot failed'))
                if current['status']=='completed':
                    break
            except (FileNotFoundError,json.JSONDecodeError):
                pass
            for partial in (ROOT/'artifacts/waypoint_demos_v1/manifest.json',ROOT/'artifacts/models/waypoint_24_seed7/manifest.json'):
                try:
                    if read(partial)['status']=='failed':
                        raise RuntimeError('Earlier pilot stage failed: '+str(partial))
                except (FileNotFoundError,json.JSONDecodeError):
                    pass
            time.sleep(5)
        assert current['executions']==12
        gate=current['successes']>=6 and current['lifts']>=8
        meta.update(status='expanding' if gate else 'completed_gate_not_met',pilot_successes=current['successes'],pilot_lifts=current['lifts'],gate_passed=gate)
        write(out/'manifest.json',meta)
        seeds=[7]
        if gate:
            for seed in plan['additional_seeds']:
                run('train_waypoint.py',['--condition','waypoint_24','--seed',str(seed)],f'waypoint_training_seed{seed}.log')
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(run,'evaluate_waypoint.py',['--condition','waypoint_24','--seed',str(s),'--suite','validation'],f'waypoint_evaluation_seed{s}.log') for s in plan['additional_seeds']]
                for f in futures:
                    f.result()
            seeds += plan['additional_seeds']
        results=[]
        for seed in seeds:
            folder=ROOT/f'artifacts/models/waypoint_24_seed{seed}';m=read(folder/'manifest.json')
            baseline=read(ROOT/f'artifacts/models/coverage_24_seed{seed}/manifest.json')
            for key in ('initial_weights_sha256','episode_stream_sha256','augmentation_stream_sha256','valid_action_targets_seen','steps_completed','parameters'):
                assert m[key]==baseline[key],key
            check_sources(m['source_sha256'])
            eval_folder=ROOT/f'artifacts/waypoint_pilot_evaluation/validation/waypoint_24_seed{seed}'
            score=read(eval_folder/'manifest.json');check_sources(score['source_sha256'])
            assert score['status']=='completed' and score['episodes_sha256']==sha(eval_folder/'episodes.json')
            rows=read(eval_folder/'episodes.json')
            originals=read(ROOT/f'artifacts/evaluation/validation/coverage_24_seed{seed}/episodes.json')
            assert len(rows)==len(originals)==12
            for a,b in zip(rows,originals):
                assert a['case_id']==b['case_id'] and a['initial_rgb_sha256']==b['initial_rgb_sha256']
            results.append(dict(seed=seed,requested_waypoint=dict(successes=score['successes'],lifts=score['lifts'],n=12),
                effective_command_baseline=dict(successes=sum(x['success'] for x in originals),lifts=sum(x['lifted'] for x in originals),n=12),
                matched_initial_weights_sampling_augmentation_and_frame_budget=True))
        meta.update(status='completed',completed_utc=now(),results=results,
            scope='Development only; either predeclared seed7 pilot or all3 seeds after its gate. No held-out confirmation or personal-video contribution claim.',
            code_sha256=sha(ROOT/'follow_waypoint_pilot.py'))
        write(out/'manifest.json',meta)
        lines=['# 模仿教师原始目标点：独立开发试验','',
            '本试验保持原24条覆盖示范的RGB、状态及过去实际命令不变，只把训练目标换成教师原始的限幅前目标点。每条标签通过重新执行教师并逐值比较实际命令、输入历史和完整评分轨迹获得。',
            '运行时仍由学习模型预测目标点，共用原8mm安全限幅器；不输入教师阶段或物体真值。','',
            f"固定seed7开发门槛：至少6/12严格成功且8/12合格抬升。实际 {current['successes']}/12 成功、{current['lifts']}/12 抬升；门槛通过：{gate}。",'',
            '| 种子 | 原有效命令目标：成功 / 抬升 | 请求目标点：成功 / 抬升 |','|---|---:|---:|']
        for r in results:
            a,b=r['effective_command_baseline'],r['requested_waypoint']
            lines.append(f"| {r['seed']} | {a['successes']}/12 · {a['lifts']}/12 | {b['successes']}/12 · {b['lifts']}/12 |")
        lines += ['', '这是复用开发任务的探索结果。只有满足预先门槛才扩展到另外两个固定种子；不存在挑选表现最好的种子。后续泛化确认必须另冻结新场景。',
                  '', '[方案](../../configs/waypoint_pilot_study_v1.json) · [完整收据](manifest.json)']
        (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        print(json.dumps(meta,ensure_ascii=False),flush=True)
    except Exception as exc:
        meta.update(status='failed',error=str(exc));write(out/'manifest.json',meta);raise


if __name__=='__main__':
    main()
