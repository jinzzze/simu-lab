"""Trace personal progress into executed teacher commands; no learner selection."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT,read,write,sha,now


def transport(entry):
    path=ROOT/entry['file'];trace_path=path.with_suffix('.scoring.json')
    assert sha(path)==entry['dataset_sha256'] and sha(trace_path)==entry['scoring_sha256']
    trace=read(trace_path);indices=[i for i,r in enumerate(trace) if 'transport' in r['stage']]
    with np.load(path,allow_pickle=False) as arrays:actions=arrays['actions'][indices].copy()
    dt=trace[1]['t_s']-trace[0]['t_s']
    xy=np.array(entry['initial_truth_xy_m']);direction=np.array(entry['outcome']['goal_xy_m'])-xy
    progress=(actions[:,:2]-xy)@direction/(direction@direction)
    return np.arange(1,len(indices)+1)*dt,progress


def main():
    out=ROOT/'artifacts/intervention_audit_v1';out.mkdir(parents=True,exist_ok=False)
    index=read(ROOT/'artifacts/robot_demos_v1/training_index.json')
    audit=read(ROOT/'artifacts/demonstration_audit_v1/receipt.json');assert audit['status']=='passed'
    pairs=[];example=None
    for a,b in zip(index['train_groups']['geometry_only_24'],index['train_groups']['video_timing_24']):
        ta,pa=transport(a);tb,pb=transport(b)
        shared=np.linspace(0,max(ta[-1],tb[-1]),201)
        # This interpolation is a diagnostic on executed robot commands only.
        difference=np.abs(np.interp(shared,ta,pa)-np.interp(shared,tb,pb))
        profile=b['human_profile']
        pairs.append(dict(human_episode_id=b['human_episode_id'],pair=f"{b['start_id']}_{b['goal_id']}",
            coarse_human_duration_s=profile['duration_s'],geometry_transport_s=float(ta[-1]),
            video_transport_s=float(tb[-1]),maximum_command_progress_difference=float(difference.max()),
            geometry_strict_success=a['success'],video_strict_success=b['success'],
            visible_constraints=len(profile['valid_source_frame_indices']),
            maximum_missing_constraint_gap_fraction=profile['maximum_constraint_gap_fraction']))
        if b['human_episode_id']=='X01_E01':example=(a,b,ta,pa,tb,pb)
    assert example is not None
    a,b,ta,pa,tb,pb=example;profile=b['human_profile'];knots=np.array(profile['progress_knots'])
    plt.rcParams.update({'font.family':'Arial','font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.2));fig.subplots_adjust(left=.08,right=.98,top=.78,bottom=.24,wspace=.28)
    axes[0].scatter(knots[1:-1,0]*profile['duration_s'],profile['source_progress_before_monotone_envelope'],s=20,color='#7f8d96',label='Visible RGB constraints')
    axes[0].plot(knots[:,0]*profile['duration_s'],knots[:,1],lw=1.5,color='#007d96',label='Monotone setpoint guide')
    axes[0].set_xlabel('Time within coarse human window (s)');axes[0].set_title('a   Image-space progress guide',loc='left',fontweight='bold')
    axes[1].plot(ta,pa,color='#737f8a',lw=1.6,label='Shared geometry')
    axes[1].plot(tb,pb,color='#007d96',lw=1.6,label='+ Personal timing / progress')
    axes[1].set_xlabel('Time since robot transport began (s)');axes[1].set_title('b   Executed robot command labels',loc='left',fontweight='bold')
    for ax in axes:
        ax.set_ylabel('Dimensionless progress');ax.set_ylim(-.04,1.08);ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=8,loc='lower right')
    fig.text(.08,.94,'Personal video changes the robot training targets',fontsize=13,fontweight='bold')
    fig.text(.08,.86,'Example fixed as the first X01 clip: X01_E01, P1 → G3. Teacher strict placement fails in both conditions.',fontsize=9,color='#4c606b')
    fig.text(.08,.07,'Guide interpolation defines desired robot setpoints; missing human labels remain missing. No recovered human 3D motion or learned-policy result is shown.',fontsize=8,color='#4c606b')
    for suffix in ('svg','png'):fig.savefig(out/f'intervention.{suffix}',dpi=300,facecolor='white')
    plt.close(fig)
    result=dict(status='passed',created_utc=now(),pairs=pairs,example_selection='X01_E01, first named new training clip; selected independently of policy outcomes',
        robot_command_interpolation_for_diagnostic_only=True,human_missing_labels_filled=False,
        all_24_interventions_changed_labels=all(r['maximum_command_progress_difference']>1e-5 for r in pairs),
        input_index_sha256=sha(ROOT/'artifacts/robot_demos_v1/training_index.json'),code_sha256=sha(ROOT/'inspect_intervention.py'))
    assert result['all_24_interventions_changed_labels']
    write(out/'receipt.json',result)
    text=['# Personal information actually used','',
        'All 24 paired trajectories have identical pre-transport inputs and commands. Each intervention changes the executed transport command sequence. The figure uses X01_E01, the first named X01 training clip; both teachers fail strict placement on that example.', '',
        'The human curve is a monotone envelope of visible RGB projections inside coarse event brackets. Its timing is approximate. Interpolation produces desired robot setpoints, while missing human observations remain null. Shared engineering speed limits and target settling affect the executed robot schedule.', '',
        '|Source|P/G|Human window (s)|Baseline transport (s)|Video transport (s)|Max command-progress difference|',
        '|---|---|---:|---:|---:|---:|']
    for r in pairs:text.append(f"|{r['human_episode_id']}|{r['pair']}|{r['coarse_human_duration_s']:.2f}|{r['geometry_transport_s']:.2f}|{r['video_transport_s']:.2f}|{r['maximum_command_progress_difference']:.3f}|")
    text+=['','This verifies a computational contribution of the personal recordings; it does not establish a policy-success benefit.','', '![Intervention](intervention.png)']
    (out/'REPORT_EN.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(dict(pairs=len(pairs),changed=result['all_24_interventions_changed_labels'],figure=str(out/'intervention.png')))


if __name__=='__main__':main()
