"""Plan and verify reversible test-video names from the completed visual review."""
import argparse
from collections import Counter
from pathlib import Path
import cv2
from common import ROOT, read, write, sha, now

PAIRS = {
    'T01':('P2','G3'), 'T02':('P2','G3'), 'T03':('P4','G4'), 'T04':('P1','G2'),
    'T05':('P4','G1'), 'T06':('P1','G2'), 'T07':('P3','G4'), 'T08':('P2','G2'),
    'T09':('P1','G1'), 'T10':('P2','G3'), 'T11':('P4','G1'), 'T12':('P3','G4'),
    'T13':('P4','G1'), 'T14':('P3','G4'), 'T15':('P1','G2'), 'T16':('P3','G3')
}
EXPECTED = {('P1','G2'):3, ('P2','G3'):3, ('P3','G4'):3, ('P4','G1'):3,
            ('P1','G1'):1, ('P2','G2'):1, ('P3','G3'):1, ('P4','G4'):1}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify-renamed',action='store_true')
    args=parser.parse_args()
    out=ROOT/'data/processed/test_intake/x03_naming_v1'
    source=out/'manifest.json';data=read(source)
    raw=(ROOT/'data/raw/X03_test').resolve()
    plan_path=out/'rename_plan.json'
    if not args.verify_renamed:
        assert not plan_path.exists()
        assert data['all_decode_passed'] and data['unique_raw_files']==data['unique_decoded_streams']==16
        assert not any(r['matches_registered_train_val_file'] or r['matches_x01_x02_decoded_stream'] for r in data['recordings'])
        assert Counter(PAIRS.values())==Counter(EXPECTED)
        count=Counter();mapping=[]
        detector=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
        for row in data['recordings']:
            pair=PAIRS[row['review_id']];count[pair]+=1
            name=f'test_{pair[0]}_{pair[1]}_{count[pair]:02d}'+Path(row['original_name']).suffix.lower()
            old=(raw/row['original_name']).resolve();new=(raw/name).resolve()
            assert old.parent==new.parent==raw and old.exists() and not new.exists()
            assert sha(old)==row['sha256']
            marker_rows=[]
            for sample in (row['frames'][0],row['frames'][-1]):
                frame=cv2.imread(str(ROOT/sample['path']));width=frame.shape[1];detected=[]
                for side,lo,hi in (('left',0,width//2),('right',width//2,width)):
                    _,ids,_=detector.detectMarkers(frame[:,lo:hi])
                    if ids is not None:
                        detected.extend((side,int(x)) for x in ids.ravel() if x in (0,2))
                marker_rows.append({'frame':sample['frame'],'expected_marker_count':len(set(detected)),
                                    'both_boards_detected':set(detected)=={('left',0),('left',2),('right',0),('right',2)}})
            mapping.append({'review_id':row['review_id'],'original_name':row['original_name'],'new_name':name,
                'sha256':row['sha256'],'observed_start_id':pair[0],'observed_destination_id':pair[1],
                'repeat_index':count[pair],'recording_order_known':False,
                'prescribed_goal_independently_confirmed':False,
                'label_source':'User-requested visual content naming; initial source and apparent placement region reviewed at 5 sampled frames. Does not establish original intention or strict physical success.',
                'first_last_marker_checks':marker_rows,
                'sampled_frame_indices':[f['frame'] for f in row['frames']],
                'strict_physical_success':None,'duration_s':row['duration_s'],'shape':row['shape'],'fps':row['fps']})
        assert len({r['new_name'] for r in mapping})==16
        write(plan_path,{'status':'ready_to_rename','created_utc':now(),'session_id':'X03','split':'test',
            'raw_directory':str(raw),'source_manifest_sha256':sha(source),'mapping':mapping,
            'file_bytes_must_remain_identical':True,'sampled_frames_visually_inspected':80,
            'training_or_tuning_allowed':False,'script_sha256':sha(ROOT/'rename_x03_review.py')})
        print({'plan':str(plan_path),'planned_renames':16,
               'all_first_last_markers_detected':all(m['both_boards_detected'] for r in mapping for m in r['first_last_marker_checks'])})
        return
    plan=read(plan_path)
    assert plan['source_manifest_sha256']==sha(source)
    for item in plan['mapping']:
        target=(raw/item['new_name']).resolve()
        assert target.parent==raw and target.exists() and sha(target)==item['sha256']
        assert not (raw/item['original_name']).exists()
    videos={p.name for p in raw.iterdir() if p.suffix.lower() in ('.mp4','.mov','.m4v')}
    assert videos=={r['new_name'] for r in plan['mapping']}
    receipt=out/'rename_receipt.json';assert not receipt.exists()
    write(receipt,{'status':'completed','completed_utc':now(),'files_renamed':16,
        'all_video_hashes_unchanged':True,'rename_plan_sha256':sha(plan_path),
        'original_names_preserved_in':str(plan_path.relative_to(ROOT)),
        'test_content_opened_for_user_authorized_naming':True,'training_or_tuning_performed':False,
        'capture_metadata_confirmed':False,'per_file_intended_goal_confirmation_pending':True})
    lines=['# X03 测试视频命名与检查','',
        '根据用户要求查看16条原片的80张抽样帧，核对起点与表观放置目标，完成文件命名。此操作只做测试素材整理，不用于训练、调参或成功率统计。',
        '全部16条完整解码，960×544，29–30fps，约6–7秒。文件哈希和完整解码像素序列各自唯一；与登记的S01/X01/X02文件哈希无重复，与X01/X02完整解码像素序列无完全一致。',
        '15条为约30fps；ec2e50bb开头原片为29fps，仍完整可读，按实际帧率处理。全部文件保留，不按结果筛选。',
        '同组01–03按原文件名字典序分配，仅为文件编号，实际拍摄顺序未知。重命名前后逐条SHA256一致。',
        '命名反映目视识别的起点和放置位置；原始指令、拍摄时间和独立会话信息仍待申请人确认。没有把落点一致自动标注为严格物理成功。',
        '本次是申请人明确要求的测试内容命名检查，已记录接触测试内容的事实；不能再描述为从未查看过的素材。测试目录、检查图和映射仍与训练/验证处理入口隔离。', '',
        '| 组合 | 条数 |','|---|---:|']
    lines += [f'| {p}→{g} | {n} |' for (p,g),n in sorted(EXPECTED.items())]
    lines += ['', '## 原名映射', '', '| 原名 | 新名 |', '|---|---|']
    lines += [f"| {r['original_name']} | {r['new_name']} |" for r in plan['mapping']]
    lines += ['', '[机器可读映射](rename_plan.json) · [逐条内容与解码检查](manifest.json) · [重命名收据](rename_receipt.json)', '']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    write(ROOT/'receipts/x03_intake_latest.json',{'created_utc':now(),'received':16,'named':16,'split':'test',
        'report':str((out/'REPORT_ZH.md').relative_to(ROOT)),
        'report_sha256':sha(out/'REPORT_ZH.md'),'receipt':str(receipt.relative_to(ROOT)),
        'receipt_sha256':sha(receipt),'training_allowed':False})
    print({'status':'completed','renamed':16,'all_video_hashes_unchanged':True,'report':str(out/'REPORT_ZH.md')})


if __name__=='__main__':
    main()
