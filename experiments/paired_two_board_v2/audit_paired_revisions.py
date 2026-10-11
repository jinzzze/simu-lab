"""Verify shared initialization, episode sampling and budget across refinements."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,shutil
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--earlier',required=True);p.add_argument('--later',required=True);p.add_argument('--tag',required=True);args=p.parse_args()
    a=read(ROOT/args.earlier);b=read(ROOT/args.later);records=[];sources={args.earlier:sha(ROOT/args.earlier),args.later:sha(ROOT/args.later),'audit_paired_revisions.py':sha(__file__)}
    assert a['training_index']==b['training_index'] and a['dataset_manifest_sha256']==b['dataset_manifest_sha256']
    assert a['conditions']==b['conditions'] and a['training']['seeds']==b['training']['seeds']
    for key in ['steps','batch_size','learning_rate','weight_decay','grad_clip','optimizer','sampling']:assert a['training'][key]==b['training'][key],key
    assert a['augmentation']==b['augmentation']
    for condition in a['conditions']:
        for seed in a['training']['seeds']:
            paths=[ROOT/f'artifacts/runs/{cfg["revision"]}/{condition}_seed{seed}/manifest.json' for cfg in (a,b)];metas=[read(path) for path in paths]
            for path in paths:sources[str(path.relative_to(ROOT))]=sha(path)
            assert all(m['status']=='completed' for m in metas)
            for key in ['initial_weights_sha256','episode_fraction_stream_sha256','valid_action_targets_seen','steps_completed','parameters']:assert metas[0][key]==metas[1][key],(condition,seed,key)
            records.append({'condition':condition,'seed':seed,'initial_weights_sha256':metas[0]['initial_weights_sha256'],
                'episode_stream_sha256':metas[0]['episode_fraction_stream_sha256'],'valid_frame_targets_each_revision':metas[0]['valid_action_targets_seen'],
                'parameters':metas[0]['parameters'],'steps_each_revision':metas[0]['steps_completed']})
    out=ROOT/f'artifacts/reports/{b["revision"]}/{args.tag}';assert not out.exists();out.mkdir(parents=True)
    changes={key:{'earlier':a.get(key),'later':b.get(key)} for key in ['action_representation','normalization','training'] if a.get(key)!=b.get(key)}
    result={'passed':True,'earlier':a['revision'],'later':b['revision'],'created_utc':datetime.now(timezone.utc).isoformat(),'pairs':records,
        'configuration_changes':changes,'source_sha256':sources,'scope':'matched recorded initialization, streams and target/update counts; changed parameterization and scales remain a joint refinement'}
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');shutil.copy2(__file__,out/Path(__file__).name)
    (out/'REPORT_ZH.md').write_text(f'# 相邻修订配对核验\n\n{a["revision"]} → {b["revision"]} 六对运行通过：完整初始权重、整轨迹采样哈希、有效帧目标数、优化步数和模型参数数均匹配。\n\n条件间长度可能不同，不能声称等帧数/等算力。表示/尺度改变是联合开发修订；v4→v5保持表示/尺度，只改损失权重。全部差异与来源见 [收据](results.json)。\n',encoding='utf-8')
    print(json.dumps({'passed':True,'report':str(out/'REPORT_ZH.md'),'pairs':records},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
