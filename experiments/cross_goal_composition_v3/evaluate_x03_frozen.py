"""Separate held-out video-label coverage audit; no training or recipe changes."""
from collections import Counter
from pathlib import Path
import json
import shutil
import time
import traceback
import cv2
import numpy as np
from common import ROOT,OLD,PROJECT,read,write,sha,now
import calibrate_recordings as calibration
import extract_recordings as extraction


def main():
    cv2.setNumThreads(1)
    v4=ROOT.parent/'personal_video_timing_v4'
    freeze_path=v4/'artifacts/final_evaluation_freeze_v1.json'
    freeze=read(freeze_path)
    assert freeze['policy_sha256']==sha(v4/'configs/policy_v1.json')
    for name,digest in freeze['checkpoints'].items():
        assert sha(v4/f'artifacts/models/{name}/checkpoint.pt')==digest
    frozen=read(ROOT/'artifacts/reports/X01_trajectories_v1/summary.json')
    assert frozen['source_hashes']==read(ROOT/'artifacts/reports/X02_trajectories_v1/summary.json')['source_hashes']
    for key,digest in frozen['source_hashes'].items():assert sha(PROJECT/key)==digest
    cfg=frozen['config']
    prior=ROOT/'data/processed/test_intake/x03_naming_v1'
    intake=read(prior/'manifest.json');plan=read(prior/'rename_plan.json');renamed=read(prior/'rename_receipt.json')
    assert renamed['status']=='completed' and renamed['rename_plan_sha256']==sha(prior/'rename_plan.json')
    assert plan['source_manifest_sha256']==sha(prior/'manifest.json')
    by_id={r['review_id']:r for r in intake['recordings']}
    items=[]
    for m in sorted(plan['mapping'],key=lambda x:x['new_name']):
        source=by_id[m['review_id']]
        assert source['sha256']==m['sha256']
        item=dict(episode_id=f"X03_{m['review_id']}",session_id='X03',split='test',
            relative_video_path='data/raw/X03_test/'+m['new_name'],sha256=m['sha256'],
            intended_start_id=m['observed_start_id'],intended_goal_id=m['observed_destination_id'],
            label_origin='User-confirmed naming mapping; original command sheet unavailable',
            qc={'fps':source['fps'],'decoded_frames':source['decoded_frames']})
        assert sha(ROOT/item['relative_video_path'])==item['sha256'];items.append(item)
    out=ROOT/'data/processed/X03_frozen_labels_v1';report=ROOT/'artifacts/reports/X03_frozen_labels_v1'
    out.mkdir(parents=True,exist_ok=False);report.mkdir(parents=True,exist_ok=False)
    snap=report/'source_snapshot';snap.mkdir()
    sources=dict(frozen['source_hashes'])
    for file in [Path(__file__),PROJECT/'assets/calibration/board_geometry.json',freeze_path]:
        sources[str(file.relative_to(PROJECT)).replace('\\','/')]=sha(file)
    for name in sources:
        if name.endswith('.task'):continue
        dest=snap/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(PROJECT/name,dest)
    write(report/'frozen_train_config.json',cfg)
    write(report/'normalized_test_intake.json',dict(session='X03',split='test',recordings=items,
        historical_naming_qc_access=True,training_allowed=False))
    geometry=read(PROJECT/'assets/calibration/board_geometry.json')
    module=calibration.legacy_module();detector=calibration.BoardDetector()
    settings={k:cfg[k] for k in ('num_hands','min_hand_detection_confidence','min_hand_presence_confidence')}
    options=extraction.vision.HandLandmarkerOptions(
        base_options=extraction.python.BaseOptions(model_asset_path=str(PROJECT/'artifacts/models/hand_landmarker.task')),
        running_mode=extraction.vision.RunningMode.IMAGE,**settings)
    records=[];errors=[];started=time.monotonic()
    with extraction.vision.HandLandmarker.create_from_options(options) as hand:
        for item in items:
            try:
                first,last=calibration.endpoints(ROOT/item['relative_video_path'],item['qc']['decoded_frames'])
                boards={b:module.calibrate(first,b,geometry,detector) for b in ('source_left','target_right')}
                goals=calibration.measure_blue_goals(first,boards['target_right'],module)
                final={b:module.calibrate(last,b,geometry,detector) for b in boards}
                drift={b:float(np.max(np.linalg.norm(np.array(boards[b]['grid_quad_pixels'])-final[b]['grid_quad_pixels'],axis=1))) for b in boards}
                cal=dict(calibration_id=item['episode_id']+'_frozen_test_v1',episode_id=item['episode_id'],
                    session_id='X03',split='test',raw_relative_path=item['relative_video_path'],raw_sha256=item['sha256'],
                    first_frame_index=0,boards=boards,goals=goals,prescribed_start_id=item['intended_start_id'],
                    prescribed_goal_id=item['intended_goal_id'],robot_registration=None,strict_outcome=None,
                    max_endpoint_drift_pixels=drift,scope='Frozen train recipe applied to test; no metric accuracy claim')
                calpath=out/(item['episode_id']+'.calibration.json');write(calpath,cal)
                result=extraction.process(item,cal,cfg,out,report,hand,detector)
                labels=[json.loads(x) for x in (out/(item['episode_id']+'.frames.jsonl')).read_text(encoding='utf-8').splitlines()]
                assert len(labels)==item['qc']['decoded_frames']
                for i,r in enumerate(labels):
                    assert r['split']=='test' and r['frame_index']==i and r['source_sha256']==item['sha256']
                    assert not r['training_ready'] and r['object_robot_xyz'] is None and r['robot_action'] is None
                    assert r['object_on_table'] is None and r['object_grid_uv'] is None
                    if not r['object']['auxiliary_mask_candidate']:assert r['object']['center_px'] is None
                result.update(calibration_sha256=sha(calpath),visible_object_frames=sum(r['object']['auxiliary_mask_candidate'] for r in labels),
                    max_fit_rms_px=max(b['corner_fit_rms_pixels'] for b in boards.values()),
                    max_endpoint_drift_px=max(drift.values()))
                records.append(result)
                print({k:result[k] for k in ('episode_id','frames','visible_object_frames','reference_invalid_frames')},flush=True)
            except Exception as exc:
                error=dict(episode_id=item['episode_id'],error=str(exc),traceback=traceback.format_exc())
                errors.append(error);print(error,flush=True)
    for name,digest in sources.items():assert sha(PROJECT/name)==digest
    total=sum(r['qc']['decoded_frames'] for r in items);visible=sum(r['visible_object_frames'] for r in records)
    result=dict(status='completed' if not errors else 'completed_with_errors',created_utc=now(),session_id='X03',split='test',
        source_hashes=sources,config=cfg,model_freeze_sha256=sha(freeze_path),
        received_clips=len(items),processed_clips=len(records),received_frames=total,
        processed_frames=sum(r['frames'] for r in records),visible_object_frames=visible,coverage_over_all_received_frames=visible/total,
        reference_invalid_frames=sum(r['reference_invalid_frames'] for r in records),episodes=records,errors=errors,
        elapsed_s=time.monotonic()-started,training_or_tuning_performed=False,
        prior_naming_qc_access=True,physical_success_labels=0,independent_accuracy_measured=False,
        scope='Separate frozen candidate-coverage audit after all policy checkpoints are fixed. No recipe retuning or train-profile modification. Events remain unreviewed.')
    write(out/'manifest.json',result);write(report/'summary.json',result)
    lines=['# X03 冻结后候选覆盖检查','',
        f"接收 {len(items)} 条、{total} 帧；成功处理 {len(records)} 条。可用物体候选 {visible}/{total} 帧（{100*visible/total:.1f}%），分母包含处理失败文件。",'',
        '沿用 X01 冻结并用于 X02 的完整检测函数、阈值和手部模型。所有六个策略检查点已固定后才运行本检查；结果不回流修改训练或测试判据。', '',
        '这是候选覆盖率，不能当作定位准确率、任务成功率或真实机器人测试。粗事件仍是未审阅候选。人类物理动作、空中 XYZ、严格成功均保持未知。', '',
        'X03 先前按照申请人要求用于文件命名质检；不能称为从未查看。文件命名对应关系由申请人确认，独立原始指令表不可得。', '',
        '|视频|帧数|物体候选帧|参照失效帧|标定拟合 RMS|','|---|---:|---:|---:|---:|']
    for r in records:lines.append(f"|{r['episode_id']}|{r['frames']}|{r['visible_object_frames']}|{r['reference_invalid_frames']}|{r['max_fit_rms_px']:.2f} px|")
    lines+=['','未完成/失败记录：'+str([{k:v for k,v in e.items() if k!='traceback'} for e in errors]),
        '', '抽帧图已生成，但程序运行本身不代表有人查看过图。像素误差及物理事件真值仍需独立标注。']
    (report/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print({k:result[k] for k in ('status','processed_clips','received_frames','visible_object_frames','coverage_over_all_received_frames')},flush=True)


if __name__=='__main__':main()
