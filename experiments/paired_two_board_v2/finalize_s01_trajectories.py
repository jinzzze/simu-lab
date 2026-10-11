"""Record Codex visual coarse review separately from automatic proposals.

Review bounds below refer to tile indices in each 24-frame overview. They are
search brackets, not certified single-frame event times or physical contact.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
import cv2
import numpy as np

ROOT=Path('E:/HumanoidRobotLearning/experiments/paired_two_board_v2')
REV='S01_trajectories_v1'
REPORT=ROOT/'artifacts/reports'/REV
OUT=ROOT/'data/processed'/REV
NAMES=['contact','lift','transport','placement','release','withdraw']
# The 24-tile overviews were inspected by Codex, not by a human annotator.
BOUNDS={
    'S01_E01':[[6,7],[9,10],[9,10],[15,17],[16,17],[17,18]],
    'S01_E02':[[9,10],[11,12],[11,12],[16,18],[17,18],[18,19]],
    'S01_E03':[[7,8],[10,11],[10,11],[14,16],[15,16],[16,17]],
    'S01_E04':[[8,9],[11,12],[11,12],[15,18],[17,18],[18,19]],
    'S01_E05':[[7,8],[9,10],[9,10],[12,14],[14,15],[15,16]],
    'S01_E06':[[10,11],[12,13],[12,13],[15,18],[17,18],[18,19]],
    'S01_E07':[[5,7],[9,10],[9,10],[14,17],[17,18],[18,19]],
    'S01_E08':[[6,7],[9,10],[9,10],[14,17],[17,18],[18,19]],
    'S01_E09':[[8,9],[10,11],[10,11],[14,16],[16,17],[17,18]],
    'S01_E10':[[9,10],[10,12],[11,12],[15,17],[17,18],[18,19]],
    'S01_E11':[[8,9],[10,12],[11,12],[15,17],[17,18],[18,19]],
    'S01_E12':[[9,10],[10,11],[10,12],[16,18],[18,19],[19,20]],
    'S01_E13':[[9,10],[12,14],[12,14],[16,18],[18,19],[19,20]],
    'S01_E14':[[7,8],[10,11],[10,11],[15,16],[16,17],[17,18]],
    'S01_E15':[[8,9],[11,13],[12,13],[15,17],[17,18],[18,19]],
    'S01_E16':[[8,9],[11,13],[12,13],[16,18],[18,19],[19,20]],
}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    summary=json.loads((REPORT/'summary.json').read_text(encoding='utf-8'))
    assert len(BOUNDS)==len(summary['episodes'])==16,'Every episode must have an actual visual review before finalization'
    if (REPORT/'coarse_visual_review.json').exists():raise ValueError('Review revision already exists')
    created=datetime.now(timezone.utc).isoformat()
    reviewed=[]
    for episode in summary['episodes']:
        name=episode['episode_id'];indices=episode['review_frame_indices']
        intake=json.loads((ROOT/f'data/processed/S01_intake_v1/{name}.json').read_text(encoding='utf-8'))
        cal=json.loads((ROOT/f'data/processed/S01_calibration_v2/{name}.json').read_text(encoding='utf-8'))
        rows=[json.loads(line) for line in (ROOT/episode['frame_labels']).read_text(encoding='utf-8').splitlines()]
        fps=intake['source']['fps'];events=[]
        for event,(a,b) in zip(NAMES,BOUNDS[name]):
            assert 0<=a<=b<24
            lo,hi=indices[a],indices[b]
            events.append({'name':event,'frame_interval':[lo,hi],'time_interval_s':[lo/fps,hi/fps],
                           'interval_semantics':'coarse visually inspected search bracket; exact physical onset remains unknown',
                           'annotation_source':'Codex_visual_inspection','reviewer':'Codex; not human annotation',
                           'visibility':'uncertain' if event in ('contact','lift','placement') else 'visual_evidence',
                           'qualified_physical_event':None,'evidence_frame_indices':[lo,hi]})
        obj=rows[-1]['object'];containment=None;margin=None;poly_mm=None
        if obj['auxiliary_mask_candidate']:
            H=np.array(cal['end_frame_diagnostic']['boards']['target_right']['image_to_grid_mm_homography'])
            poly_mm=cv2.perspectiveTransform(np.array(obj['polygon_px'],np.float64)[None],H)[0]
            goal=np.array(cal['goals'][cal['prescribed_goal_id']]['center_grid_mm'])
            margin=float(np.min(20-np.abs(poly_mm-goal)))
            containment='inside_proxy' if margin>2.5 else 'outside_proxy' if margin < -2.5 else 'boundary_uncertain'
        record={'episode_id':name,'session_id':'S01','split':'train','source_sha256':episode['source_sha256'],
                'created_utc':created,'review_code_sha256':sha(__file__),
                'reviewer':'Codex visual inspection of 24 evenly spaced annotated raw-frame tiles per clip; no human ground-truth annotation',
                'review_sheet':str((REPORT/f'{name}_review.jpg').relative_to(ROOT)).replace('\\','/'),
                'review_sheet_sha256':sha(REPORT/f'{name}_review.jpg'),'review_frame_indices':indices,
                'events':events,'visual_transfer_observed':True,'visual_release_and_withdraw_observed':True,
                'apparent_final_region_diagnostic':{
                    'goal_id':cal['prescribed_goal_id'],'goal_definition_frame_index':0,
                    'status':containment or 'unavailable','visible_dark_rectangle_plane_projection_mm':None if poly_mm is None else poly_mm.tolist(),
                    'minimum_proxy_margin_mm':margin,'review_band_mm':2.5,
                    'scope':'final visible dark rectangle projected to paper plane; not measured physical base footprint',
                    'projection_limit':'object height, shadow, segmentation, camera distortion and paper warp not bounded'},
                'final_image_stationary_run_s':episode['final_image_stationary_run_s'],
                'clear_end_after_reviewed_withdraw_bracket_s':max(0,(len(rows)-1-events[-1]['frame_interval'][1])/fps),
                'physical_qualified_lift_height_m':None,'physical_base_footprint_containment':None,'strict_success':None,
                'allowed_use':'offline event/trajectory search constraints; do not use reviewed future events or final diagnostics as current policy observations'}
        write(OUT/f'{name}.reviewed.json',record);reviewed.append(record)
    review={'reviewed_episodes':16,'sampled_raw_frames_inspected':16*24,'coarse_event_brackets':16*6,
            'annotation_source':'Codex visual inspection; not applicant/human annotations','episodes':reviewed,
            'strict_success_labels':0,'training_ready':False,'created_utc':created}
    write(REPORT/'coarse_visual_review.json',review)
    counts=summary['object_status_counts'];visible=counts.get('visible_candidate',0)
    text=f'''# S01 轨迹候选与事件核查

2026-10-08。16 条原片、{summary['total_frames']} 帧均完成处理，原片保持不变；会话继续整体属于 train。16 条标注视频逐帧解码验证，分辨率与帧数完整。另保存每帧物体候选、独立手部输出、缺失掩码、事件检索候选与核查记录。

物体有效候选 {visible}/{summary['total_frames']} 帧（{100*visible/summary['total_frames']:.1f}%）；手部模型有输出 {summary['hand_frames']}/{summary['total_frames']} 帧。**这是候选覆盖率，不是准确率或任务成功率。** 无法检测的 RGB 保留，物体标签置空，不插值。

Codex 核查每条 24 张均匀抽帧，共 384 帧，观察到同编号搬运、释放和撤手过程。另行保存 96 个粗略事件检索区间：contact/lift/transport/placement/release/withdraw。它们是视觉搜索区间；接触、离桌瞬间与高度不具备物理认证，不能冒称人工精确标签。

自动事件提案保存在 events.json，保持未核查状态；核查后的粗区间另存在 reviewed.json，不覆盖前者。例如部分自动释放提案早于可见松手过程，不能直接进入机器人动作监督。

## 每条记录

|片段|物体候选帧/总帧|手部输出帧|末尾图像中心稳定时长(s)|末帧区域代理检查|
|---|---:|---:|---:|---|
'''
    for e,r in zip(summary['episodes'],reviewed):
        text+=f'|{e["episode_id"]}|{e["object_status_counts"].get("visible_candidate",0)}/{e["frames"]}|{e["hand_frames"]}|{e["final_image_stationary_run_s"]:.2f}|{r["apparent_final_region_diagnostic"]["status"]}|\n'
    text+='''

## 使用与限制

绿色框是当前帧可见深色部分；黄色框是首帧预定的 40×40 mm 目标区域；紫色轨迹遇缺失断开。静止时偶尔未检出可能由阈值或参照失效引起，未宣称缺失等于遮挡。

末帧代理检查将可见深色旋转框投影到纸面；inside_proxy 只表示这个代理框在目标区域内。物体上表面、阴影、未知高度与纸张起伏使它不能替代真实底部脚印判据。2.5 mm 是诊断分界带，不是已验证精度界限。末尾图像中心稳定也不等于物理接触、松手和完整入框的稳定计时。因此严格 success 全部保持 null。

手部点由项目原有 MediaPipe HandLandmarker 生成；记录模型 hash 和版本。inside_image_mask 是有效数值/画面内掩码，不是逐关节可见性真值。物体检测不读取手部、过去或未来位置。未知桌面位置、空中 XYZ 和机器人动作全部 null。

输出：data/processed/S01_trajectories_v1/{S01_Exx.frames.jsonl, S01_Exx.events.json, S01_Exx.reviewed.json}。视频和总览在本报告目录。配置与源代码快照在 config.json/source_snapshot；覆盖同名版本被禁止。两条试运行记录保存在 S01_trajectory_pilot_v1，不重复算新增样本。

复现校验（E:/HumanoidRobotLearning）：

```powershell
. ./scripts/Enter-Project.ps1
./.env/python.exe experiments/paired_two_board_v2/validate_s01_trajectories.py
```

下一阶段：在仿真中实现四个目标与前置 goal_xy 条件，利用这些视频的可见路径、时间区间与目标定义约束机器人示范生成；有效 XYZ/夹爪动作必须来自实际执行的机器人示范，不能由人手二维点伪造。对照组与视频约束组保持机器人示范数量、目标覆盖、模型和训练预算一致。新模型泛化评估仍需独立 S02/S03/S04；新策略尚未训练。
'''
    (REPORT/'REPORT_ZH.md').write_text(text,encoding='utf-8')
    receipt=ROOT/'receipts/S01_trajectories_v1';receipt.mkdir()
    for name in ['README.md','LABEL_FORMAT.md','provenance.json']:
        shutil.copy2(ROOT/name,receipt/name)
    with (ROOT/'README.md').open('a',encoding='utf-8') as f:
        f.write('\n\n## 2026-10-08 逐帧轨迹与事件候选\n\nS01 全部 16 条、2604 帧完成独立物体/手部候选提取与带标注视频。Codex 核查 384 抽帧，保存 96 个粗事件检索区间。见 [轨迹报告](artifacts/reports/S01_trajectories_v1/REPORT_ZH.md) 和 TRAJECTORY_LABELS.md。缺失不插值，严格成功/物理 XYZ/机器人动作仍未知。\n')
    with (ROOT/'LABEL_FORMAT.md').open('a',encoding='utf-8') as f:
        f.write('\n\n## 独立逐帧候选与视觉核查\n\nS01_trajectories_v1 是独立候选格式，详见 TRAJECTORY_LABELS.md。不能把它套成已有 episode.schema.json 的完整 labeled 状态。接收标签仍为 received；拍摄时间未知不补写。events.json 为自动检索提案，reviewed.json 为 Codex 粗视觉检索区间，均不是人工精确动作真值。\n')
    provenance=json.loads((ROOT/'provenance.json').read_text(encoding='utf-8'))
    provenance['trajectory_candidates']={'revision':REV,'session':'S01','split':'train','frames':2604,'episodes':16,
                                       'source_snapshot':str((REPORT/'source_snapshot').relative_to(ROOT)).replace('\\','/'),
                                       'review':'384 evenly spaced raw frame tiles inspected by Codex; 96 coarse event search brackets',
                                       'strict_success_labels':0,'robot_action_labels':0,'metric_xyz_labels':0}
    write(ROOT/'provenance.json',provenance)
    hashes={}
    for p in [ROOT/'extract_s01_trajectories.py',ROOT/'finalize_s01_trajectories.py',ROOT/'validate_s01_trajectories.py',ROOT/'TRAJECTORY_LABELS.md',ROOT/'README.md',ROOT/'LABEL_FORMAT.md',ROOT/'provenance.json']+list(OUT.glob('*'))+list(REPORT.glob('*.json'))+[REPORT/'REPORT_ZH.md']:
        hashes[str(p.relative_to(ROOT)).replace('\\','/')]=sha(p)
    write(receipt/'hashes.json',hashes)
    print(json.dumps({'reviewed_episodes':16,'sampled_frames':384,'coarse_event_brackets':96,
                      'object_candidate_frames':visible,'total_frames':2604,
                      'final_proxy_status_counts':{s:sum(r['apparent_final_region_diagnostic']['status']==s for r in reviewed) for s in ('inside_proxy','outside_proxy','boundary_uncertain','unavailable')},
                      'strict_success_labels':0},indent=2))


if __name__=='__main__':main()
