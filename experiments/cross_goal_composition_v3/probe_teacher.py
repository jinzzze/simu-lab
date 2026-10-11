"""Matched development probe on allowed training pairs, never held-out pairs."""
import numpy as np
from common import ROOT,read,write,now,snapshot,source_paths,check_sources,sha
from paired_sim import PairedGoalEnv
from teacher import execute


def main():
    out=ROOT/'artifacts/teacher_development_v1';out.mkdir(parents=True,exist_ok=False)
    plan=dict(created_utc=now(),pairs=read(ROOT/'protocol.json')['training_pairs'],
              scene_base=69000,jitter_m=.003,variants=['legacy','quintic_0p10'],
              new_training=False,scope='development teacher feasibility; privileged initial XY; no held-out combinations',
              selection_rule='Use quintic0.10 only if all12 lift, at least11 succeed and maximum length <=480; else retain and revise before main training.')
    write(out/'plan.json',plan)
    hashes=snapshot(out,source_paths()+[ROOT/'probe_teacher.py'])
    rows=[];env=PairedGoalEnv()
    try:
        for variant in plan['variants']:
            for p,g in plan['pairs']:
                scene=plan['scene_base']+int(p[1])-1
                obs=env.reset_task(scene,p,g)
                initial=env.get_object_xy_for_labels()
                row=env.execute_expert(initial,record=False) if variant=='legacy' else execute(env,initial,record=False)
                row.update(variant=variant,case=p+'_'+g,initial_rgb_sha256=__import__('hashlib').sha256(obs['rgb'].tobytes()).hexdigest())
                trace_path=out/f'{variant}_{p}_{g}_scoring.json';write(trace_path,env.trace)
                row['scoring_sha256']=sha(trace_path);rows.append(row);write(out/'episodes.json',rows)
                print(__import__('json').dumps({'variant':variant,'pair':p+'_'+g,'success':row['success'],'lifted':row['lifted'],'steps':row['steps']}),flush=True)
    finally:
        env.close()
    check_sources(hashes)
    summary={v:dict(n=sum(r['variant']==v for r in rows),successes=sum(r['variant']==v and r['success'] for r in rows),
                    lifts=sum(r['variant']==v and r['lifted'] for r in rows),max_steps=max(r['steps'] for r in rows if r['variant']==v)) for v in plan['variants']}
    a=summary['quintic_0p10'];passed=a['lifts']==12 and a['successes']>=11 and a['max_steps']<=480
    write(out/'manifest.json',dict(status='completed',plan=plan,source_sha256=hashes,summary=summary,
                                  predeclared_gate_passed=passed,episodes_sha256=sha(out/'episodes.json')))
    print(__import__('json').dumps({'summary':summary,'gate_passed':passed}),flush=True)


if __name__=='__main__':
    main()
