"""Quantify measured support for old S01 timing/progress guides; no retraining."""
import json
import numpy as np
from common import ROOT,OLD,read,write,sha,now


def main():
    out=ROOT/'artifacts/s01_profile_support_v1';out.mkdir(parents=True,exist_ok=False)
    manifest=read(OLD/'artifacts/reports/paired_robot_demos_v1/manifest.json')
    rows=[];sources={}
    for row in manifest['episodes']:
        if row['condition']!='video_guided':
            continue
        path=(OLD/row['file']).with_suffix('.metadata.json')
        assert sha(path)==row['metadata_sha256']
        meta=read(path);profile=meta['human_profile'];eid=profile['human_episode_id']
        raw=OLD/f'data/processed/S01_trajectories_v1/{eid}.frames.jsonl'
        frame_rows=[json.loads(x) for x in raw.read_text(encoding='utf-8').splitlines()]
        times={r['frame_index']:r['timestamp_s'] for r in frame_rows}
        valid=np.array([times[i] for i in profile['valid_source_frame_indices']])
        start,end=profile['coarse_time_interval_s']
        dt=float(np.median(np.diff(sorted(times.values()))))
        eligible=[t for t in times.values() if start<t<end]
        boundaries=np.r_[start,valid,end]
        gaps=np.maximum(np.diff(boundaries)-dt,0)
        knots=np.array(profile['progress_knots'])
        plateau_fraction=float(np.diff(knots[:,0])[np.diff(knots[:,1])<=1e-6].sum())
        record=dict(episode=eid,scene_seed=row['scene_seed'],goal=row['goal_id'],
            duration_s=profile['duration_s'],valid_center_frames=len(valid),eligible_transport_frames=len(eligible),
            visible_center_fraction=len(valid)/len(eligible),longest_interval_without_valid_center_s=float(max(gaps)),
            fixed_or_monotone_plateau_time_fraction=plateau_fraction,
            teacher_success=meta['outcome']['success'],teacher_steps=row['steps'],
            missing_human_xyz_not_filled=not profile['human_missing_labels_interpolated'],
            robot_setpoints_engineered_between_knots=profile['desired_robot_setpoints_interpolated'])
        rows.append(record);sources[str(path.relative_to(OLD)).replace('\\','/')]=sha(path)
        sources[str(raw.relative_to(OLD)).replace('\\','/')]=sha(raw)
    assert len(rows)==16
    s=dict(episodes=16,teacher_successes=sum(r['teacher_success'] for r in rows),
        median_visible_center_fraction=float(np.median([r['visible_center_fraction'] for r in rows])),
        min_visible_center_fraction=float(min(r['visible_center_fraction'] for r in rows)),
        longest_missing_interval_s=float(max(r['longest_interval_without_valid_center_s'] for r in rows)),
        episodes_with_gap_over_half_second=sum(r['longest_interval_without_valid_center_s']>.5 for r in rows),
        median_duration_s=float(np.median([r['duration_s'] for r in rows])))
    write(out/'results.json',dict(created_utc=now(),summary=s,episodes=rows,source_sha256=sources,code_sha256=sha(ROOT/'audit_old_video_profiles.py'),
        scope='Descriptive post-hoc support audit of training-only S01 guides; no thresholds or outcomes changed. Does not establish cause of policy performance differences.'))
    lines=['# 现有个人视频时序约束的观测支持','',
        f"16条训练视频生成的引导教师均成功（{s['teacher_successes']}/16）。搬运时间区间内，有效可见物体中心的帧占比中位数为 {100*s['median_visible_center_fraction']:.1f}%，最低 {100*s['min_visible_center_fraction']:.1f}%。",'',
        f"有 {s['episodes_with_gap_over_half_second']}/16 条存在超过0.5秒的连续无有效中心区间；最长约 {s['longest_missing_interval_s']:.2f} 秒。",'',
        '这些间隙没有被补成人类三维动作真值；离线生成器在已观察到的二维进度节点之间构造机器人设定点。它们因此包含工程插值与单目投影假设。',
        '该审计解释了视频约束的观测局限，不能证明遮挡造成了v6负结果。两组教师均成功，也不能保证不同时间分布同样容易被循环策略学习。',
        '后续个人视频对照应报告有效中心覆盖和时间区间不确定性，保持缺失标签为缺失。不能为了得到更好的结果事后只挑最清楚/最成功片段。','',
        '| 视频 | 有效帧/区间帧 | 最长无有效中心间隙(s) | 引导时长(s) |',
        '|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['episode']} | {r['valid_center_frames']}/{r['eligible_transport_frames']} | {r['longest_interval_without_valid_center_s']:.2f} | {r['duration_s']:.2f} |")
    lines += ['', '[逐片数据、定义和来源哈希](results.json)']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(s))


if __name__=='__main__':
    main()
