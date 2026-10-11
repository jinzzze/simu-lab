"""Create a separate capture protocol; preserve all legacy data and plans."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'paired_two_board_v2'
HELDOUT=[('P1','G2'),('P2','G3'),('P3','G4'),('P4','G1')]
PAIRED=[(f'P{i}',f'G{i}') for i in range(1,5)]
CROSS=[(f'P{i}',f'G{j}') for i in range(1,5) for j in range(1,5) if i!=j and (f'P{i}',f'G{j}') not in HELDOUT]
TRAIN=sorted(PAIRED+CROSS)


def main():
    for path in ('configs','source_snapshot','data/raw/X01_train','data/raw/X02_val','data/raw/X03_test',
                 'data/processed','artifacts','receipts'):
        (ROOT/path).mkdir(parents=True,exist_ok=True)
    if (ROOT/'protocol.json').exists():
        raise FileExistsError('Refuse to replace a frozen capture protocol')
    now=datetime.now(timezone.utc).isoformat()
    source_hashes={}
    for name in ('protocol.json','capture_checklist.csv','goal_catalog.json','layout_measurements.json','configs/sim_task_v1.json','configs/sim_physics_v1.json'):
        src=OLD/name
        target=ROOT/'source_snapshot'/('legacy_'+name.replace('/','__'))
        shutil.copy2(src,target)
        source_hashes[name]=hashlib.sha256(src.read_bytes()).hexdigest()
    sessions=[dict(session_id='X01',split='train',folder='X01_train',count=16,recipe='8 new training combinations, 2 attempts each'),
              dict(session_id='X02',split='val',folder='X02_val',count=12,recipe='12 training-covered combinations, 1 new attempt each'),
              dict(session_id='X03',split='test',folder='X03_test',count=16,recipe='4 held-out combinations x3, plus 4 paired controls x1')]
    protocol=dict(revision='cross-goal-composition-v3',created_utc=now,
        authorization='Applicant authorized work independent of new recordings; collection list and point numbering agreed in chat.',
        numbering='Each board: top-left1, top-right2, bottom-left3, bottom-right4; red P board left, blue G board right in landscape view.',
        dimensions_mm=dict(grid_width=100,grid_height=130,nearest_grid_edge_gap=130,grid_center_gap=230),
        training_pairs=TRAIN,heldout_pairs=HELDOUT,paired_controls=PAIRED,
        previous_personal_data=dict(experiment='../paired_two_board_v2',session='S01',split='train',received=16,
                                   role='Shared target geometry is available; new cross-pair timing data is pending. Original raw files remain in v2.'),
        new_sessions=sessions,planned_new_clips=44,received_new_clips=0,
        split_rules=['Assign complete real recording sessions before extraction; derivatives inherit split.',
                     'Held-out P/G pairs prohibited in every new training source, including synthetic robot demonstrations and human timing profiles.',
                     'X03 sealed from model/threshold selection; analyze only after model and scoring freeze.',
                     'Prior v6 all16-pair diagnostic is development evidence; new tests require fresh scene IDs and no selection on test outcomes.',
                     'Video clips are applicant demonstrations, not robot success trials.',
                     'Retain failed attempts; technical exclusions require recorded reasons.'],
        success='Same strict virtual40mm footprint/lift/release/withdraw/stability criterion as v2; uncertain human labels remain uncertain.',
        missing=['all44 new recordings','actual capture timestamps','applicant outcome adjudication','real board-to-robot registration'],
        legacy_source_sha256=source_hashes)
    (ROOT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    rows=[]
    specs=[('X01','train','X01_train',CROSS*2),('X02','val','X02_val',TRAIN),('X03','test','X03_test',HELDOUT*3+PAIRED)]
    for session,split,folder,pairs in specs:
        repeats={}
        for index,(p,g) in enumerate(pairs,1):
            repeats[(p,g)]=repeats.get((p,g),0)+1
            repeat=repeats[(p,g)]
            rows.append(dict(episode_id=f'{session}_E{index:02}',session_id=session,split=split,start_id=p,goal_id=g,
                repeat=repeat,planned_video_path=f'data/raw/{folder}/{split}_{p}_{g}_{repeat:02}.mp4',
                capture_status='not_collected',captured_at='',video_sha256='',actual_outcome='unknown',
                technical_qc='pending',notes=''))
    with (ROOT/'capture_checklist.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    guide=['# 新交叉组合采集清单','', '旧 S01 的16条保留在 paired_two_board_v2。本目录是独立新协议；旧 S02/S03/S04 清单不被改写。', '',
           '编号：每张纸左上1、右上2、左下3、右下4；横向画面左纸红 P，右纸蓝 G。', '',
           '| 批次 | 用途 | 内容 | 条数 | 上传目录 |','|---|---|---|---:|---|',
           '| X01 | 训练 | P1→G3/G4、P2→G1/G4、P3→G1/G2、P4→G2/G3；各2次 | 16 | data/raw/X01_train |',
           '| X02 | 验证 | 上述8组合 + 四个同编号组合，各1次 | 12 | data/raw/X02_val |',
           '| X03 | 封存测试 | P1→G2、P2→G3、P3→G4、P4→G1各3次 + 四个同编号各1次 | 16 | data/raw/X03_test |','',
           '不同批次为真实独立录制会话，记录带时区的拍摄日期和机位/光照变化；同一场固定手机和纸。不能把一个连续场次拆成训练和测试。',
           '每条开头静止1秒，自然抓起、离桌搬运、以指定蓝点为中心放稳、松手撤开约6cm、结尾静止2秒。整片保存，失败保留。',
           '两网格各100×130mm，最近网格边缘空隙130mm。完整露出标定码；每场需要重新确认几何，不能套用旧相机像素位置。',
           '文件名与逐条编号见 [CSV清单](capture_checklist.csv)。传入原片后另外登记原名、哈希与采集日期；不从最终落点倒推指令。',
           'X03 在训练/规则冻结前不提取轨迹、不调参；四个留出组合也不能生成训练用机器人示范。',
           '', '当前三批目录为空；创建目录不代表视频已经收集。']
    (ROOT/'CAPTURE_GUIDE_ZH.md').write_text('\n'.join(guide)+'\n',encoding='utf-8')
    for _,split,folder,_ in specs:
        (ROOT/f'data/raw/{folder}/UPLOAD_HERE.txt').write_text(f'Put original {split} clips here. See ../../../CAPTURE_GUIDE_ZH.md. Do not overwrite originals.\n',encoding='utf-8')
    (ROOT/'.gitignore').write_text('data/raw/**\n!data/raw/**/UPLOAD_HERE.txt\ndata/processed/\nartifacts/\nreceipts/\n__pycache__/\n',encoding='utf-8')
    print(json.dumps({'root':str(ROOT),'planned_new_clips':len(rows),'training_pairs':TRAIN,'heldout_pairs':HELDOUT},ensure_ascii=False))


if __name__=='__main__':
    main()
