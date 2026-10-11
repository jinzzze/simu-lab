"""English/Chinese reports and editable scientific figure from audited results."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from common import ROOT,read,write,sha,now


def main():
    result=read(ROOT/'artifacts/results_v1/results.json');assert result['status']=='completed'
    teacher=read(ROOT/'artifacts/robot_demos_v1/manifest.json');budget=read(ROOT/'artifacts/training_audit_v1/receipt.json')
    sensitivity=read(ROOT/'artifacts/initial_overlap_audit_v1/receipt.json')
    out=ROOT/'artifacts/figures/personal_timing_v2';out.mkdir(parents=True,exist_ok=False)
    conditions=result['study']['conditions'];labels=['Shared geometry','+ Personal timing / progress'];colors=['#737f8a','#007d96']
    keys=['same_number','trained_cross','heldout','all'];groups=['Same-number\n4 pairs','Seen cross-goal\n8 pairs','Held-out\n4 pairs','All pairs']
    plt.rcParams.update({'font.family':'Arial','font.size':9,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(12,5.1),facecolor='white')
    gs=fig.add_gridspec(1,3,width_ratios=[1.9,1,1],left=.06,right=.98,top=.74,bottom=.28,wspace=.42)
    ax=fig.add_subplot(gs[0])
    for ci,c in enumerate(conditions):
        totals=result['totals']['test'][c];xs=np.arange(4)+(ci-.5)*.34
        vals=[100*totals[k]['successes']/totals[k]['executions'] for k in keys]
        ax.bar(xs,vals,.3,color=colors[ci],label=labels[ci],zorder=2)
        for x,k in zip(xs,keys):
            v=totals[k];ax.text(x,107+ci*8,f"{v['successes']}/{v['executions']}",ha='center',fontsize=8,color=colors[ci])
            for j,s in enumerate(('7','17','27')):
                m=totals['per_seed'][s][k]
                ax.scatter(x+(j-1)*.045,100*m['successes']/m['executions'],s=15,facecolor='white',edgecolor='#1c2930',linewidth=.6,zorder=3)
    ax.set_xticks(range(4),groups);ax.set_yticks([0,25,50,75,100]);ax.set_ylim(-3,130);ax.set_ylabel('Strict placement success (%)')
    ax.grid(axis='y',color='#e0e5e8');ax.set_axisbelow(True);ax.tick_params(axis='x',length=0,pad=8)
    ax.set_title('a   All frozen test outcomes',loc='left',fontweight='bold',pad=15)
    ax.legend(frameon=False,loc='upper left',bbox_to_anchor=(-.08,-.28),fontsize=8,ncol=2)
    for ci,c in enumerate(conditions):
        hm=fig.add_subplot(gs[ci+1]);matrix=np.zeros((4,4),int)
        for r in result['all_cases']:
            if r['suite']=='test' and r['condition']==c:matrix[int(r['start_id'][1])-1,int(r['goal_id'][1])-1]+=int(r['success'])
        im=hm.pcolormesh(np.arange(5)-.5,np.arange(5)-.5,matrix/6,cmap='Blues',vmin=0,vmax=1)
        hm.set_xlim(-.5,3.5);hm.set_ylim(3.5,-.5);hm.set_aspect('equal')
        for i in range(4):
            for j in range(4):
                hm.text(j,i,f'{matrix[i,j]}/6',ha='center',va='center',color='white' if matrix[i,j]>=4 else '#153341')
                if [f'P{i+1}',f'G{j+1}'] in result['study']['training_pairs']:
                    hm.add_patch(Rectangle((j-.46,i-.46),.92,.92,fill=False,edgecolor='#222',lw=1.05))
        hm.set_xticks(range(4),['G1','G2','G3','G4']);hm.set_yticks(range(4),['P1','P2','P3','P4']);hm.tick_params(length=0)
        hm.set_xlabel('Prescribed goal');hm.set_title(['b   Shared geometry','c   + Video timing / progress'][ci],loc='left',pad=15,fontweight='bold')
    cax=fig.add_axes([.67,.135,.24,.025]);cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[0,.5,1]);cb.solids.set_rasterized(False)
    cb.ax.set_xticklabels(['0%','50%','100%']);cb.outline.set_visible(False)
    fig.text(.06,.95,'Personal-video contribution to goal-conditioned pick-and-place',fontsize=14,fontweight='bold')
    fig.text(.06,.87,'Same 12 training pairs and 24 robot demonstrations per condition · same CNN–GRU · 2,500 updates × 3 seeds',fontsize=9,color='#4b5960')
    fig.text(.06,.060,'Dots: model seeds. Cell: two starts × three seeds. Outlines: training pairs. Eight physical test starts are reused.',fontsize=8,color='#4b5960')
    fig.text(.06,.025,'Four initial test images repeat training/development pixels; see the separate sensitivity analysis. No real-robot evaluation.',fontsize=8,color='#4b5960')
    fig.savefig(out/'results.svg',facecolor='white');fig.savefig(out/'results.png',dpi=300,facecolor='white');plt.close(fig)
    caption=('Both conditions share the same training-only goal geometry, 12 training combinations and 24 paired robot initial scenes. '
        'The video condition additionally uses coarse transfer time and dimensionless image-progress constraints from S01(8) and X01(16). '
        'Human metric actions are not recovered. Robot actions come from executed teacher commands. All teacher failures are retained. '
        'Learner, initialization, episode/augmentation streams and update counts match; frame-target counts and compute differ. '
        'All three fixed-final EMA models are evaluated on 32 tasks, reusing eight physical starts across goals and seeds. '
        'Dots are observed seed rates, not confidence intervals. Four initial test-task images exactly repeat training/development pixels, involving two physical starts. '
        'The full predeclared results are retained and duplicate-exclusion sensitivities are reported separately. This is an exploratory simulation-only ablation informed by prior experiments.')
    (out/'CAPTION_EN.md').write_text(caption+'\n',encoding='utf-8')
    rows=['# Personal video timing: audited results','',caption,'',
        '|Test category|Shared geometry|+ Video timing / progress|','|---|---:|---:|']
    zh=['# 新个人视频对照结果','', '两组均覆盖 12 个 P/G 组合、各 24 条机器人示范。共同目标几何来自同一批个人训练视频；第二组额外加入个人视频的粗搬运时间与二维进度约束。', '',
        '|测试类别|共同几何|额外视频时序/进度|','|---|---:|---:|']
    for k,label,zlabel in zip(keys,['Same-number','Seen cross-goal','Held-out combinations','All combinations'],['同编号','已见交叉组合','留出组合','全部组合']):
        vals=[]
        for c in conditions:
            r=result['totals']['test'][c][k];vals.append(f"{r['successes']}/{r['executions']} ({100*r['successes']/r['executions']:.1f}%)")
        rows.append('|'+label+'|'+'|'.join(vals)+'|');zh.append('|'+zlabel+'|'+'|'.join(vals)+'|')
    rows+=['','|Condition|Seed|Validation strict|Test strict|Test lifts|Mean final center distance (mm)|','|---|---:|---:|---:|---:|---:|']
    for c,label in zip(conditions,labels):
        for seed in ('7','17','27'):
            v=result['totals']['validation'][c]['per_seed'][seed]['all'];t=result['totals']['test'][c]['per_seed'][seed]['all']
            rows.append(f"|{label}|{seed}|{v['successes']}/{v['executions']}|{t['successes']}/{t['executions']}|{t['lifts']}/{t['executions']}|{t['final_center_distance_mean_mm']:.1f}|")
    a,b=[result['totals']['test'][c]['all'] for c in conditions];delta=100*(b['successes']/b['executions']-a['successes']/a['executions'])
    rows+=['',f'Observed pooled difference (video − geometry): {delta:+.1f} percentage points. This does not establish a generalizable or statistically significant benefit.',
        '', 'Teacher quality (not learned-policy results): '+str(teacher['summary']),
        '', 'Data, causal input histories, current-goal consistency, all seeds, and paired optimization streams passed the stored audits.',
        '', 'The full result contains failure flags and first-closure alignment diagnostics. Final center distance is simulator scoring truth, never a runtime observation.',
        '', '![Audited results](artifacts/figures/personal_timing_v2/results.png)']
    rows+=['','## Training budget actually consumed','',
        '|Condition|Seed|Updates|Episode draws|Valid action targets|Training seconds|',
        '|---|---:|---:|---:|---:|---:|']
    for r in budget['runs']:
        rows.append(f"|{r['condition']}|{r['seed']}|{r['updates']}|{r['episode_draws']}|{r['supervised_action_targets']}|{r['wall_s']:.1f}|")
    rows+=['','## Failure diagnostics','',
        'Flags may co-occur; their counts are not disjoint failure categories. First-closure alignment uses simulator truth for diagnosis only.', '',
        '|Condition|Failed executions|No qualified lift|Lift but strict placement fails|Median first-closure XY error among failures|',
        '|---|---:|---:|---:|---:|']
    for c in conditions:
        failed=[r for r in result['failure_diagnostics'] if r['suite']=='test' and r['condition']==c]
        errors=[r['first_closed_xy_error_mm'] for r in failed if r['first_closed_xy_error_mm'] is not None]
        median=f'{np.median(errors):.1f} mm' if errors else 'not available'
        rows.append(f"|{c}|{len(failed)}|{sum(not r['lifted'] for r in failed)}|{sum(r['lifted'] for r in failed)}|{median}|")
    rows+=['','Initial-scene overlap audit: '+str(result['overlap_summary']),
        '', '[Method diagram](artifacts/figures/method_v1/method.svg) · [Actual video-to-command intervention](artifacts/intervention_audit_v1/REPORT_EN.md) · [Reproduction](REPRODUCTION_ZH.md)']
    rows+=['','## Duplicate-pixel sensitivity (post hoc)','',
        '|Exclusion rule|Shared geometry|+ Video timing / progress|','|---|---:|---:|']
    for key,label in [('exclude_4_matching_task_images','Four identical initial task images'),('exclude_all_goals_for_2_affected_starts','All goals for two affected physical starts')]:
        groups=sensitivity['summaries'][key]
        values=[f"{groups[c]['all']['successes']}/{groups[c]['all']['executions']}" for c in conditions]
        rows.append('|'+label+'|'+'|'.join(values)+'|')
    rows+=['','These descriptive exclusions were introduced after the overlap audit, not before the experiment. They add no executions and do not replace the primary table. See [matching cases and all seeds](artifacts/initial_overlap_audit_v1/REPORT_ZH.md).']
    zh+=['',f'视频组相对共同几何组的合并成功率差为 {delta:+.1f} 个百分点。该数值是本轮观测差异，不能直接推断统计显著性或更广泛泛化收益。',
        '', '三种子全部保留；每组 96 次测试执行由 32 个任务重复三种子构成，只有 8 个物理初始位置，不能当作 96 个独立环境。',
        '', '示范质量：'+str(teacher['summary']),
        '', '真实视频共 28 条、5,460 帧完成候选提取和审计；Codex 检查 672 张抽帧。X01 物体候选覆盖率 89.6%，X02 为 80.6%。这是覆盖率而非准确率。',
        '', '动作监督全部来自仿真实际执行命令。人类空中三维轨迹、接触真值和真实严格成功仍未知；缺失标签没有插值补造。',
        '', '模型、初始化、轨迹采样、增强、更新次数已配对核验。轨迹长度不同使动作目标数量和运行时间不同，不声称完全等算力。',
        '', '初始图像检查发现 4 个测试任务与训练/验证图像完全相同，涉及 2 个物理起点。排除这 4 个任务后为 5/84 对 8/84；排除这 2 个起点的全部目标后为 4/72 对 7/72。两者均为事后敏感性分析，原始成绩保留，不声称全新图像泛化。',
        '', '两组固定案例均已实跑复现；独立目录与此前独立安装的 Python 环境复跑时，全部输入/动作数组及评分轨迹完全一致。只证明同机这两个案例，不等于全量重训或另一台电脑的验证。',
        '', '英文申请陈述和中文面试提纲已生成，仍需发布候选、申请人信息和最终审阅/提交。X03 未用于本次开发或训练；其独立冻结后候选检查单独保存，先前命名质检接触记录保留。',
        '', '详细三种子表见 RESULTS_EN.md；最新版图和可编辑 SVG 位于 artifacts/figures/personal_timing_v2。']
    (ROOT/'RESULTS_EN.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    (ROOT/'PROGRESS_ZH.md').write_text('\n'.join(zh)+'\n',encoding='utf-8')
    write(out/'manifest.json',dict(created_utc=now(),results_sha256=sha(ROOT/'artifacts/results_v1/results.json'),
        outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()},code_sha256=sha(ROOT/'report_results.py')))
    print(dict(figure=str(out/'results.png'),pooled_difference_percentage_points=delta),flush=True)


if __name__=='__main__':main()
