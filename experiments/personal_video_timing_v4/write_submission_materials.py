"""Local application narrative using audited numbers, with authorship review pending."""
from common import ROOT,read,write,sha,now


def main():
    result=read(ROOT/'artifacts/results_v1/results.json');assert result['status']=='completed'
    conditions=result['study']['conditions'];a,b=[result['totals']['test'][c] for c in conditions]
    def count(row):return f"{row['successes']}/{row['executions']}"
    difference=100*(b['all']['successes']/b['all']['executions']-a['all']['successes']/a['all']['executions'])
    sensitivity=read(ROOT/'artifacts/initial_overlap_audit_v1/receipt.json')['summaries']
    task_sensitivity=sensitivity['exclude_4_matching_task_images']
    start_sensitivity=sensitivity['exclude_all_goals_for_2_affected_starts']
    narrative=f'''# Project description — applicant review draft

I built a reproducible tabletop robot-learning study using personally recorded phone videos and a contact-based Panda simulation. The current task is goal-conditioned pick-and-place between four source and four target locations. Success requires a qualified lift, release, withdrawal and stable placement of the entire block footprint inside a 40 mm goal square.

The latest experiment uses 24 personal training clips: eight earlier same-number demonstrations and 16 new cross-goal demonstrations. Additional X02 and X03 recording blocks are reserved for development and held-out video checks. I recorded them on the same day with small camera-angle changes; they are not diverse independent environments. Video processing retains missing object labels and produces coarse transfer-time and dimensionless image-progress constraints, without claiming recovered human 3D actions.

I froze a controlled comparison between shared video-derived goal geometry and the same geometry plus video-derived transport timing/progress. Each group has 24 physically executed robot demonstrations covering the same 12 source–goal combinations. Four combinations are withheld. Both groups use the same small CNN–GRU, paired initializations, sampling and augmentations, 2,500 updates, and all three fixed-final EMA checkpoints. Trajectory lengths differ, so this is not an equal-frame or equal-compute comparison. Robot action supervision is recorded from simulation commands. The learned policy receives current RGB, robot proprioception, the preprovided goal and the previous command; simulator object truth and scripted phases are excluded at deployment.

Across the frozen simulation test tasks, shared geometry scored {count(a['all'])} strict successes and additional video timing/progress scored {count(b['all'])}. The observed pooled difference is {difference:+.1f} percentage points. Held-out combinations scored {count(a['heldout'])} versus {count(b['heldout'])}. Each group has 32 tasks repeated for three model seeds, reusing eight physical initial positions. I report all seeds and failed cases; these counts do not establish statistical significance or broad real-world transfer. The personal-data intervention demonstrably changes the training command labels, which is separate from whether it improves policy performance.

An audit found four initial test-task images identical to training/development images, involving two physical starts despite disjoint random seeds. The original outcomes are retained. Post-hoc exclusion of the four tasks gives {count(task_sensitivity[conditions[0]]['all'])} versus {count(task_sensitivity[conditions[1]]['all'])}; excluding every goal associated with the two affected starts gives {count(start_sensitivity[conditions[0]]['all'])} versus {count(start_sensitivity[conditions[1]]['all'])}. These are descriptive sensitivity checks, not a replacement test or independent confirmation.

Earlier experiments are preserved separately: a visual auxiliary-supervision ablation, an action-conditioned fixed-skill outcome predictor, and a compact Diffusion Policy extension. Their controllers, data budgets and tasks differ from this experiment, so their success rates are not pooled. The project does not claim language-conditioned VLA training, evaluated world-model planning or real-robot validation.

The local deliverable includes frozen source/configuration snapshots, raw-data and checkpoint hashes, causal input/action traces, scientific figures, failure diagnostics and executable fixed-case reproduction commands. Public release v1.0.0 contains the earlier study; these new local results need a new public release before this description is submitted. I used Codex assistance for implementation, experiment execution, candidate-label inspection and writing. I personally supplied the recordings and will review the claims and explain the technical decisions before submission.

Applicant identity, education, CV and UK work-authorization details remain for the applicant to provide accurately. This document has not been submitted to the employer.
'''
    (ROOT/'APPLICATION_NARRATIVE_EN.md').write_text(narrative,encoding='utf-8')
    notes='''# 面试讲解提纲

先用一句话描述任务：给机械臂指定一个 G 目标，它需要从当前 P 位置抓起方块，完整放入目标框并撤离。

1. **个人视频发挥什么作用？** 两组共同使用首帧目标中心标定。视频组额外使用训练片段的粗搬运时间和二维进度，改变仿真教师实际执行的动作标签。学习策略不在运行时调用人类轨迹。
2. **为什么不是直接做人类动作克隆？** 单目手机视频没有可靠的机器人关节动作、空中公制 XYZ 或接触真值。把二维手部位置直接称为机器人动作会越过证据范围。
3. **为什么加入交叉组合？** 仅有 P1→G1 等同编号任务时，起点与目标强相关，模型容易把目标变化错误地当作抓取位置变化。12 个训练组合和 4 个留出组合用于检验这一问题。
4. **对照公平到什么程度？** 初始场景、组合覆盖、模型、初始化、轨迹抽样、增强和更新次数配对。两组轨迹长度不同，因此实际动作监督帧数和算力不同，报告真实预算。
5. **成功标准是什么？** 评分使用真实物理接触、合格抬升、释放、撤离、完整底面入框和稳定时间。物体位置真值用于离线教师初态与评分，不供学习策略在线纠偏。
6. **局限和失败如何解释？** 对照的实际成功数、每个种子、未抬升与抬升后放置失败见 RESULTS_EN.md。不能只展示成功视频，也不能把教师成功率当学习策略成功率。
7. **怎样证明别人能运行？** 固定案例回放比较完整输入/动作和评分轨迹；目录迁移与已有独立环境复跑的收据单独报告。该检查不等于在陌生电脑上从零全量重训。
8. **你本人和工具各做了什么？** 本人采集真实视频、确认任务布局与命名、审阅最终主张；Codex 协助实现、实验、候选标签检查和文档。不要把 Codex 抽帧检查写成独立人工标注。

这些问题需要申请人用自己的话解释，不能仅背诵结果。旧世界模型是固定技能的动作条件结果预测器；当前新策略是 CNN–GRU 行为克隆。两者是独立实验，不要声称已整合成经过评测的世界模型规划器。
'''
    (ROOT/'INTERVIEW_NOTES_ZH.md').write_text(notes,encoding='utf-8')
    write(ROOT/'receipts/submission_draft_v2.json',dict(created_utc=now(),results_sha256=sha(ROOT/'artifacts/results_v1/results.json'),
        sensitivity_sha256=sha(ROOT/'artifacts/initial_overlap_audit_v1/receipt.json'),
        narrative_sha256=sha(ROOT/'APPLICATION_NARRATIVE_EN.md'),code_sha256=sha(ROOT/'write_submission_materials.py'),
        public_release_updated=False,application_submitted=False,applicant_review_pending=True))
    print('Local English project narrative and Chinese interview notes saved.')


if __name__=='__main__':main()
