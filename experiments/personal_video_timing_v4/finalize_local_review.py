"""Record actual Codex visual inspection and refresh current entry points."""
from common import ROOT,VIDEOS,PROJECT,read,write,sha,now


def main():
    files=['artifacts/figures/method_v1/method.png','artifacts/intervention_audit_v1/intervention.png',
        'artifacts/figures/personal_timing_v2/results.png','artifacts/visual_review_v1/geometry_dev_replay_v1.jpg',
        'artifacts/visual_review_v1/timing_dev_replay_v1.jpg','artifacts/demo_v1/review.jpg']
    write(ROOT/'artifacts/visual_review_v1/reviewed.json',dict(status='reviewed',created_utc=now(),reviewer='Codex',
        reviewed_files={name:sha(ROOT/name) for name in files},
        figures='Text, legends, task layout, counts and overlap caveat readable; no obvious overlap/clipping in rendered PNGs.',
        replay='24 uniformly sampled frames across two fixed development replays; both show grasp/lift/transfer, followed by out-of-goal placement consistent with failed strict scoring.',
        demo='10 sampled frames covering all seven sections, plus the selected test outcomes verified against recorded episode metadata. One fixed first test example is not presented as the aggregate outcome.',
        scope='Actual visual review of these rendered views, not continuous human video annotation or independent physical truth.'))
    x03=read(VIDEOS/'artifacts/reports/X03_frozen_labels_v1/summary.json');assert x03['status']=='completed'
    progress=ROOT/'PROGRESS_ZH.md';text=progress.read_text(encoding='utf-8')
    text+='\nX03 冻结后独立检查已完成：16 条、3,205 帧，2,637 帧有物体候选（82.3%）。该结果没有回流到策略训练；粗事件仍未独立标注，真实成功未知。\n'
    text+='\n92 秒英文演示已导出并完整解码验证 2,300 帧，包含个人原片、预先固定的首个测试案例、全种子成绩、失败、图像重合及复现范围。图表与演示抽帧由 Codex 实际目视检查。\n'
    progress.write_text(text,encoding='utf-8')
    root_readme=PROJECT/'README.md';text=root_readme.read_text(encoding='utf-8')
    start='<!-- paired-policy-current:start -->';end='<!-- paired-policy-current:end -->'
    assert start in text and end in text
    before,rest=text.split(start,1);_,after=rest.split(end,1)
    note='''
> **Latest local extension (2026-10-10):** The personal-video timing study is complete: six CNN–GRU models and 264 scored executions. Shared geometry achieved 7/96 test placements; added personal timing/progress achieved 11/96. Success remains low; four initial test images overlap training/development, with sensitivity results reported. See [current experiment](experiments/personal_video_timing_v4/README.md), [92-second demonstration](experiments/personal_video_timing_v4/artifacts/demo_v1/evidence_walkthrough.mp4), and [Chinese progress](experiments/personal_video_timing_v4/PROGRESS_ZH.md). Two fixed cases passed actual same-host reproduction from a separate directory/environment. These local results are not yet part of public release v1.0.0. The earlier studies below retain their original scope.
'''
    root_readme.write_text(before+start+note+end+after,encoding='utf-8')
    roadmap=VIDEOS/'SUBMISSION_ROADMAP_20261010_ZH.md';text=roadmap.read_text(encoding='utf-8')
    text=text.replace('新视频仍未完成几何标定、动作时间标签和新个人数据对照训练；旧报告中的“44条待采集”是上传前状态。',
        'X01/X02 已完成标定、候选轨迹与粗时间检查；新个人数据对照已完成 6 模型、264 次执行及固定案例复现。X03 在模型冻结后完成独立候选覆盖检查。旧报告中的“44条待采集”是上传前状态。')
    text=text.replace('- X01、X02、X03的真实拍摄日期、大致时间与时区；是否是独立录制会话、机位或光照如何改变。',
        '- 采集时间与机位变化已由申请人补充并保存；X03“一小时后”的参照仍略有歧义，记录为约 21:00–21:30 BST，未编造精确时间；该细节不阻塞当前实验。')
    text+='\n## 当前执行结果补记\n\n新个人视频实验见 [personal_video_timing_v4](../personal_video_timing_v4/README.md)：基线 7/96，视频组 11/96；排除重复初始图像后 5/84 对 8/84。总体成功率仍低，不能将 +4.2 个百分点写成稳定或显著提升。论文图、92 秒演示、英文项目草稿和同机复现已完成；新公开版本和正式申请仍未提交。\n'
    roadmap.write_text(text,encoding='utf-8')
    next_steps='''# 下一轮优先事项

本轮新增视频训练对照已完成。当前最主要的技术问题是学习策略的抓取稳定性，不是缺少视频文件：测试中基线 75/96、视频组 69/96 未完成合格抬升；失败案例第一次闭合时的 XY 偏差中位数约 25.5 mm、27.2 mm。教师示范两组均有 24/24 合格抬升。

1. 在开发集定位抓取偏差：区分目标输入对抓取位置的耦合、训练时动作拟合与实际闭环状态累积误差。先检查现有输入/动作轨迹，必要的输入干预单独标为诊断。
2. 预先限制最多两种小幅修订，以开发集闭环合格抬升和严格放置共同评估；保留原 v4 对照，不用已查看的 v4 测试成绩选模型。是否采用目标无关的抓取分支，需要诊断证据后再确定。
3. 若开发改动有一致改善，再冻结配置和选型规则，使用与全部已有数据按初始像素及物理位置审计的新测试清单。保留全部种子、失败和实际预算；不追补最有利场景。
4. 对外材料优先保证完整、清楚、可运行。当前负面结果与限制保留；不要为更好看的成功率隐藏初始图像重合或将教师成功率作为策略成绩。
5. 申请人需要审阅英文陈述和演示，并提供实际 CV、姓名、邮箱、教育与工作授权信息。新公开 release 与申请提交是之后的独立操作。

当前不要求继续补拍。独立人工标签误差评估仍可增加可信度，但真实空中 XYZ、接触和成功未知的部分保持未知。
'''
    (ROOT/'NEXT_STEPS_ZH.md').write_text(next_steps,encoding='utf-8')
    write(ROOT/'receipts/local_review_v1.json',dict(created_utc=now(),status='passed',
        visual_review_sha256=sha(ROOT/'artifacts/visual_review_v1/reviewed.json'),
        x03_report_sha256=sha(VIDEOS/'artifacts/reports/X03_frozen_labels_v1/summary.json'),
        result_sha256=sha(ROOT/'artifacts/results_v1/results.json'),published=False,application_submitted=False))
    print('Visual review, progress, roadmap and project entry point updated.')


if __name__=='__main__':main()
