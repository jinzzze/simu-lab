"""Audit all new train/development derivatives without opening any X03 files."""
from collections import Counter
from pathlib import Path
import json
import numpy as np
from common import ROOT,PROJECT,read,write,sha,now


def main():
    output=ROOT/'artifacts/reports/new_video_labels_audit_v1';output.mkdir(parents=True,exist_ok=False)
    results=[];all_files={};recipes=[]
    for session,split,n in [('X01','train',16),('X02','val',12)]:
        summary_path=ROOT/f'artifacts/reports/{session}_trajectories_v1/summary.json';summary=read(summary_path)
        review_path=summary_path.parent/'coarse_visual_review_v1.json';review=read(review_path)
        calibration_path=ROOT/f'data/processed/{session}_calibration_v2/manifest.json';calibration=read(calibration_path)
        assert summary['status']==calibration['status']==review['status']=='completed'
        assert len(summary['episodes'])==review['reviewed_episodes']==calibration['calibrations_completed']==n
        recipes.append((summary['source_hashes'],summary['config']))
        for path,digest in summary['source_hashes'].items():assert sha(PROJECT/path)==digest
        counts=Counter();hands=invalid=total=0;episodes=[]
        for e in summary['episodes']:
            eid=e['episode_id'];out=ROOT/f'data/processed/{session}_trajectories_v1'
            rows_path=out/f'{eid}.frames.jsonl';rows=[json.loads(line) for line in rows_path.read_text().splitlines()]
            rpath=out/f'{eid}.reviewed.json';r=read(rpath)
            assert len(rows)==e['frames']==e['annotated_output_frames_verified']
            assert r['frames_sha256']==sha(rows_path)
            assert r['review_sheet_sha256']==sha(summary_path.parent/f'{eid}_review.jpg')
            for path,digest in e['files'].items():assert sha(ROOT/path)==digest;all_files[path]=digest
            all_files[str(rpath.relative_to(ROOT))]=sha(rpath)
            for index,row in enumerate(rows):
                assert row['session_id']==session and row['split']==split and row['frame_index']==index
                assert row['source_sha256']==e['source_sha256']
                assert all(row[k] is None for k in ('object_on_table','object_grid_uv','object_robot_xyz','robot_action'))
                o=row['object'];assert not o['depends_on_hand_labels'] and not o['temporally_filled']
                assert o['auxiliary_mask_candidate']==(o['status']=='visible_candidate')
                valid_reference=all(q['valid'] for q in row['qc']['reference'].values())
                if o['auxiliary_mask_candidate']:
                    assert valid_reference and np.isfinite(o['center_px']).all() and len(o['polygon_px'])==4
                else:assert o['center_px'] is None and o['bbox_xywh_px'] is None
                if index:assert row['timestamp_s']>rows[index-1]['timestamp_s']
            local=Counter(x['object']['status'] for x in rows);counts.update(local);total+=len(rows)
            hands+=e['hand_frames'];invalid+=e['reference_invalid_frames']
            episodes.append({'episode_id':eid,'frames':len(rows),'object_candidates':local['visible_candidate'],
                'reference_invalid_frames':e['reference_invalid_frames'],'hand_frames':e['hand_frames']})
        assert total==summary['total_frames']
        for f in (summary_path,review_path,calibration_path):all_files[str(f.relative_to(ROOT))]=sha(f)
        results.append({'session':session,'split':split,'episodes':n,'frames':total,'object_counts':dict(counts),
            'object_candidate_fraction':counts['visible_candidate']/total,'hand_frames':hands,'reference_invalid_frames':invalid,
            'max_calibration_fit_rms_px':calibration['max_fit_rms_px'],'max_endpoint_grid_drift_px':calibration['max_endpoint_drift_px'],
            'visual_reviewed_frames':review['overview_frames_inspected'],'rows':episodes})
    assert recipes[0]==recipes[1],'Recipe changed between train and validation'
    result={'status':'passed','created_utc':now(),'sessions':results,'files_sha256':all_files,
        'test_files_opened':False,'same_frozen_train_val_recipe':True,'strict_success_labels':0,'robot_action_labels':0,
        'limitations':['Coverage is not accuracy; no independent manual center/physical-event ground truth.',
            'Reference drift mask conservatively removes candidates; not every missing frame is an occlusion.',
            'Coarse brackets were reviewed by Codex using sampled frames, not applicant frame-by-frame annotations.',
            'Timestamps use frame_index / mean FPS, not recovered individual frame PTS.']}
    write(output/'audit.json',result)
    lines=['# 新视频标定与候选标签检查','',f"审计通过：{sum(r['episodes'] for r in results)} 条、{sum(r['frames'] for r in results)} 帧。训练与验证使用完全相同、在 X01 上冻结的检测程序。",'',
        '|会话|用途|视频/帧|物体候选帧|手部输出帧|参照失效帧|标定拟合 RMS 最大值|', '|---|---|---|---|---|---|---|']
    for r in results:lines.append(f"|{r['session']}|{r['split']}|{r['episodes']} / {r['frames']}|{r['object_counts']['visible_candidate']} ({100*r['object_candidate_fraction']:.1f}%)|{r['hand_frames']}|{r['reference_invalid_frames']}|{r['max_calibration_fit_rms_px']:.2f} px|")
    lines+=['','上述比例是**候选覆盖率**，不是准确率或机器人成功率。标定拟合残差不是实际毫米精度；纸张起伏和镜头误差未独立测量。',
        '', 'Codex 实际检查了每条 24 张总览抽帧，共 672 帧，保存 56 个搬运开始/放置粗时间区间。可见框没有明显错认其他物体；尚未开展独立人工像素误差审计。',
        '', '验证机位变化和搬运遮挡会造成明显缺失；无效物体坐标保持 null，没有插值补写。真实离桌高度、物理接触、完整底部入框和严格成功仍未知。',
        '', '24 条训练引导来源固定为 S01_E01–E08 与全部 X01；X02 只做开发检查。新对照位于 ../personal_video_timing_v4，X03 没有进入本次标签开发或训练。',
        '', '首轮标定失败记录 X01_calibration_v1 保留。第二轮修复手机 MP4 末帧随机跳转问题，并在 X01 首帧上确定蓝色/深色阈值；没有借助 X02 或 X03 重新选阈值。']
    (output/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print({k:v for k,v in result.items() if k not in ('files_sha256','sessions')})
    for r in results:print({k:v for k,v in r.items() if k!='rows'})


if __name__=='__main__':main()
