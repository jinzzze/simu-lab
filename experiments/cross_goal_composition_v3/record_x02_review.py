"""Save the completed 60-frame visual intake review without changing raw data."""
from collections import Counter
from common import ROOT, read, write, sha, now


def main():
    intake_path = ROOT/'data/processed/intake/x02_intake_v1/manifest.json'
    visual_path = ROOT/'data/processed/intake/x02_visual_review_v1/manifest.json'
    intake, visual = read(intake_path), read(visual_path)
    assert intake['session_id'] == 'X02' and intake['split'] == 'val'
    assert intake['received_mapped'] == 12 and len(visual['recordings']) == 12
    assert intake['all_planned_decode_passed'] and visual['x01_x02_decoded_stream_overlap'] == 0
    out = visual_path.parent
    review_path = out/'review.json'
    assert not review_path.exists()
    records = []
    for raw, samples in zip(intake['recordings'], visual['recordings']):
        assert raw['episode_id'] == samples['episode_id']
        assert sha(ROOT/raw['relative_video_path']) == raw['sha256'] == samples['raw_sha256']
        assert len(samples['frames']) == 5
        for frame in (samples['frames'][0], samples['frames'][-1]):
            valid = {(m['board_roi'], m['id']) for m in frame['aruco_candidates'] if m['id'] in (0, 2)}
            assert valid == {('left',0), ('left',2), ('right',0), ('right',2)}
        records.append({
            'episode_id': raw['episode_id'], 'filename': samples['filename'],
            'raw_sha256': raw['sha256'], 'prescribed_start_id': raw['intended_start_id'],
            'prescribed_goal_id': raw['intended_goal_id'],
            'instruction_source': 'Predeclared capture list and applicant-confirmed filename mapping; not inferred from final placement.',
            'visual_review_source': 'Codex inspection of all three contact sheets, five frames per recording.',
            'initial_object_region_matches_named_start': True,
            'apparent_final_object_region_matches_named_goal': True,
            'pre_action_four_goal_marks_visible': True,
            'first_and_last_expected_board_markers_detected': True,
            'final_sample_shows_hand_released': True,
            'reviewed_frame_indices': [f['frame'] for f in samples['frames']],
            'strict_physical_success': None,
            'metric_goal_containment': None,
            'qualified_lift_height_m': None,
            'exact_release_time_s': None,
            'required_initial_and_final_hold_duration_verified': False,
            'notes': 'Moving shadows and hand/object occlusion require uncertainty masks; event boundaries need finer temporal annotation.'
        })
    counts = Counter((r['intended_start_id'], r['intended_goal_id']) for r in intake['recordings'])
    assert len(counts) == 12 and set(counts.values()) == {1}
    unexpected = []
    for row in visual['recordings']:
        for frame in row['frames']:
            for marker in frame['aruco_candidates']:
                if marker['id'] not in (0, 2):
                    unexpected.append({'episode_id': row['episode_id'], 'frame':frame['frame'],
                                       'id':marker['id'], 'use_for_calibration':False})
    report = {
        'status':'coarse_visual_intake_passed', 'created_utc':now(),
        'session_id':'X02', 'split':'val', 'received':12, 'sampled_frames_inspected':60,
        'technical_intake_sha256':sha(intake_path), 'visual_frames_manifest_sha256':sha(visual_path),
        'training_visual_manifest_sha256':visual['training_visual_manifest_sha256'], 'contact_sheet_sha256':{p:sha(ROOT/p) for p in visual['contact_sheets']},
        'records':records, 'unexpected_marker_candidates_excluded':unexpected,
        'pair_counts':{p+'_'+g:n for (p,g),n in counts.items()},
        'all_raw_hashes_unchanged':True, 'all_decoded_pixel_streams_distinct':True,
        'ready_for_calibration_and_annotation':True, 'ready_for_profile_training':False,
        'new_data_used_in_training':False, 'training_use_allowed':False, 'x01_x02_exact_decoded_stream_duplicates':0, 'test_videos_opened':False,
        'capture_metadata':{'captured_at':None,'same_recording_session_confirmed':None,
                            'independent_of_X01_recording_session_confirmed':None,
                            'source':'Applicant response pending; filesystem times not used.'},
        'code_sha256':sha(ROOT/'record_x02_review.py')
    }
    write(review_path, report)
    durations = [r['qc']['duration_s'] for r in intake['recordings']]
    text = [
        '# X02 新验证视频检查结果', '',
        '12条已接收；命名和预定组合全部匹配，十二种组合各一条，无缺失或未映射视频。全部通过完整解码。',
        '每条文件哈希及完整解码后的像素序列均不同；与已登记旧数据/其他批次未发现文件哈希重复。',
        f"分辨率均为960×544，约30fps；时长{min(durations):.2f}–{max(durations):.2f}秒，共{sum(r['qc']['decoded_frames'] for r in intake['recordings'])}帧。", '',
        '与已登记的旧S01 16条和新X01 16条均无文件哈希重复；与X01的完整解码像素序列也无完全一致。此检查不能单独证明拍摄时段独立，也不排除经过裁剪或重新编码的重复来源。',
        '这批分辨率与X01的1280×720不同，画面构图也有变化；后续按片标定，不直接套用旧像素坐标。',
        '已目视检查每条的首帧、四分之一、中间、四分之三和末帧，共60个抽样帧。起点及最终所在目标的粗位置均与文件名对应；指令本身来自事前清单和申请人确认的命名，不从落点反推。',
        '全部首尾帧能检测到左右纸板各自的0、2号标定码。重复ID按左右纸板分开；其他ID的偶发候选不参与标定。',
        '各条末帧可见物体已放下、手已松开。搬运中有手部遮挡和移动阴影，后续保留不确定性掩码，不把不可见轨迹补成真实测量。', '',
        '**当前可进入验证集标定和标签提取。X02只供开发验证，不进入训练。**',
        '本检查没有认证物理抬升高度、40mm目标区域的完整包含、精确释放时间，或每条开头1秒/结尾2秒的静止时长。上述项目需要更细的时间与几何标注。',
        '实际拍摄日期、时间及时区，以及是否为独立于X01的录制会话，等待申请人补充；复制时间不作为拍摄日期。', '',
        '| 起点 | 目标 | 接收数量 |', '|---|---|---:|'
    ]
    text += [f'| {p} | {g} | {n} |' for (p,g),n in counts.items()]
    text += ['', '[技术检查](../x02_intake_v1/manifest.json) · [逐条目视检查](review.json)', '',
             '下一步：按每条视频重新确认板面几何与蓝点中心，细化动作时间区间；验证视频与X01训练视频始终分开。', '']
    report_path = out/'REPORT_ZH.md'
    report_path.write_text('\n'.join(text),encoding='utf-8')
    write(ROOT/'receipts/x02_intake_latest.json', {
        'created_utc':now(),'technical_intake':str(intake_path.relative_to(ROOT)),
        'visual_review':str(review_path.relative_to(ROOT)), 'visual_review_sha256':sha(review_path),
        'report':str(report_path.relative_to(ROOT)), 'report_sha256':sha(report_path),
        'received':12,'ready_for_calibration_and_annotation':True,'ready_for_profile_training':False
    })
    print({'status':report['status'],'received':12,'sampled_frames_inspected':60,
           'unexpected_marker_candidates_excluded':len(unexpected),'report':str(report_path)})


if __name__ == '__main__':
    main()

