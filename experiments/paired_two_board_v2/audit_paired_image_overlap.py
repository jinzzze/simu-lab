"""Describe exact RGB overlap without retroactively deleting benchmark cases."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,shutil
import numpy as np
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rgb_sha(rgb):return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()

def collect(entries):
    all_images={};first_images={};source={}
    for entry in entries:
        path=ROOT/entry['file'];digest=sha(path);assert digest==entry['dataset_sha256'];source[entry['file']]=digest
        with np.load(path,allow_pickle=False) as data:
            images=data['rgb']
            for tick,frame in enumerate(images):
                key=rgb_sha(frame);all_images.setdefault(key,[]).append([entry['episode_id'],tick])
                if tick==0:first_images.setdefault(key,[]).append(entry['episode_id'])
    return all_images,first_images,source

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--suite',default='validation')
    parser.add_argument('--tag',default='image_overlap_v1');args=parser.parse_args();cfg=read(ROOT/args.config)
    index=read(ROOT/cfg['training_index']);out=ROOT/f'artifacts/reports/{cfg["revision"]}/{args.suite}_{args.tag}'
    assert not out.exists();out.mkdir(parents=True)
    val_all,val_first,source=collect(index['shared_validation']);groups={}
    for condition in cfg['conditions']:
        train_all,train_first,training_source=collect(index['train_groups'][condition]);source.update(training_source)
        cases=[]
        for seed in cfg['training']['seeds']:
            folder=ROOT/f'artifacts/runs/{cfg["revision"]}_evaluation/{args.suite}/{condition}_seed{seed}'
            meta=read(folder/'manifest.json');assert meta['status']=='completed'
            rows=[json.loads(x) for x in (folder/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
            source[str((folder/'episodes.jsonl').relative_to(ROOT))]=sha(folder/'episodes.jsonl')
            for row in rows:
                digest=row['initial_rgb_sha256'];data_path=folder/f'scene_{row["scene_seed"]}_inputs_actions.npz'
                assert sha(data_path)==row['dataset_sha256']
                with np.load(data_path,allow_pickle=False) as data:
                    keys={rgb_sha(frame) for frame in data['policy_input_rgb'].reshape(-1,224,224,3)}
                    assert rgb_sha(data['policy_input_rgb'][0,0])==digest
                cases.append({'policy_seed':seed,'scene_seed':row['scene_seed'],'success':row['success'],
                    'initial_rgb_sha256':digest,'training_initial_matches':train_first.get(digest,[]),
                    'training_any_frame_matches':train_all.get(digest,[]),
                    'shared_teacher_validation_initial_matches':val_first.get(digest,[]),
                    'unique_policy_input_rgb':len(keys),'unique_inputs_matching_training':len(keys & train_all.keys()),
                    'unique_inputs_matching_shared_teacher_validation':len(keys & val_all.keys())})
        groups[condition]={'cases':cases,'initial_images_matching_training':sum(bool(c['training_any_frame_matches']) for c in cases),
            'initial_images_matching_teacher_validation':sum(bool(c['shared_teacher_validation_initial_matches']) for c in cases),
            'unique_initial_images':len({c['initial_rgb_sha256'] for c in cases}),'executions':len(cases),
            'training_unique_rgb':len(train_all),'training_initial_unique_rgb':len(train_first)}
    result={'revision':cfg['revision'],'suite':args.suite,'created_utc':datetime.now(timezone.utc).isoformat(),'groups':groups,
        'source_sha256':source,'analysis_sha256':sha(__file__),
        'scope':'exact pixel equality, not perceptual similarity; reused common states may render identically across distinct physics seeds; all original results retained',
        'selection_policy':'no primary-case removal; any overlap exclusions would be labelled posthoc sensitivity only'}
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# 输入图像完全重复核查','', '该核查比较完整原始 224×224 RGB 像素，不依据相似度筛选回合，也不改变主结果。训练图像来自各自条件的所有示范帧；共同教师验证帧另行标记。',
        '', '| 条件 | 执行数 | 独特初始 RGB | 初图匹配任一训练帧 | 初图匹配共同教师验证 |','|---|---:|---:|---:|---:|']
    for c,g in groups.items():lines.append(f'| {c} | {g["executions"]} | {g["unique_initial_images"]} | {g["initial_images_matching_training"]} | {g["initial_images_matching_teacher_validation"]} |')
    lines+=['', '多个模型复用同一场景，初图重复是设计的一部分。不同物理种子也可能因像素量化或共同姿态产生相同图像；场景编号不同不保证像素不同。validation 复用相同教师验证场景，属于开发集。新场景只可称为新的仿真初始场景，不能替代独立真实会话或证明任意目标泛化。',
        '', '[逐回合匹配位置、来源与统计](results.json)']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');shutil.copy2(__file__,out/Path(__file__).name)
    print(json.dumps({'report':str(out/'REPORT_ZH.md'),'summary':{c:{k:v for k,v in g.items() if k!='cases'} for c,g in groups.items()}},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
