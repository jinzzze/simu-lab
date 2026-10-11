"""Vector scientific plot from audited counts; dots are seeds, not confidence bounds."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from common import ROOT,read,write,sha,now


def main():
    result=read(ROOT/'artifacts/results_v1/results.json')
    out=ROOT/'artifacts/figures/coverage_v3';out.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.titlesize':10,'axes.labelsize':9,
                         'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(10.6,4.2),facecolor='white')
    gs=fig.add_gridspec(1,3,width_ratios=[1.65,1,1],left=.065,right=.985,top=.74,bottom=.29,wspace=.38)
    ax=fig.add_subplot(gs[0]);colors=['#697786','#187ca0'];conditions=['paired_24','coverage_24']
    labels=['Same-number\n4 pairs','Additional covered\n8 pairs','Held-out\n4 pairs'];keys=['same_number','trained_cross','heldout']
    for ci,c in enumerate(conditions):
        totals=result['totals']['test'][c]
        xs=np.arange(3)+(ci-.5)*.34
        vals=[100*totals[k]['successes']/totals[k]['executions'] for k in keys]
        ax.bar(xs,vals,.30,color=colors[ci],label=['4-pair training','12-pair training'][ci],zorder=2)
        for x,k,v in zip(xs,keys,vals):
            a=totals[k]
            ax.text(x,107+ci*7,f"{a['successes']}/{a['executions']}",ha='center',va='bottom',fontsize=8,color=colors[ci])
            for j,seed in enumerate(('7','17','27')):
                m=totals['per_seed'][seed][k]
                ax.scatter(x+(j-1)*.05,100*m['successes']/m['executions'],s=14,facecolor='white',edgecolor='#222222',linewidth=.65,zorder=3)
    ax.set_xticks(np.arange(3),labels);ax.set_ylim(-3,126);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('Strict success (%)')
    ax.grid(axis='y',color='#e1e6ea',zorder=0);ax.set_axisbelow(True);ax.tick_params(axis='x',length=0,pad=8)
    ax.set_title('a   Frozen test outcomes',loc='left',pad=16,fontweight='bold')
    ax.legend(frameon=False,loc='upper left',bbox_to_anchor=(-.03,-.30),ncol=2,fontsize=8)
    images=[]
    for ci,c in enumerate(conditions):
        hm=fig.add_subplot(gs[ci+1]);matrix=np.zeros((4,4),int)
        for row in result['all_cases']:
            if row['suite']=='test' and row['condition']==c:
                matrix[int(row['start_id'][1])-1,int(row['goal_id'][1])-1]+=int(row['success'])
        image=hm.pcolormesh(np.arange(5)-.5,np.arange(5)-.5,matrix/6,cmap='Blues',vmin=0,vmax=1)
        hm.set_xlim(-.5,3.5);hm.set_ylim(3.5,-.5);hm.set_aspect('equal');images.append(image)
        for i in range(4):
            for j in range(4):
                hm.text(j,i,f'{matrix[i,j]}/6',ha='center',va='center',fontsize=9,color='white' if matrix[i,j]>=4 else '#152c3b')
                trained=i==j if ci==0 else [f'P{i+1}',f'G{j+1}'] in result['study']['training_pair_rule']
                if trained:
                    hm.add_patch(Rectangle((j-.46,i-.46),.92,.92,fill=False,edgecolor='#212121',lw=1.15))
        hm.set_xticks(range(4),['G1','G2','G3','G4']);hm.set_yticks(range(4),['P1','P2','P3','P4'])
        hm.tick_params(length=0);hm.set_xlabel('Prescribed goal');hm.set_title(['b   4-pair training','c   12-pair training'][ci],loc='left',pad=16,fontweight='bold')
    cbax=fig.add_axes([.63,.13,.30,.025]);cbar=fig.colorbar(images[0],cax=cbax,orientation='horizontal',ticks=[0,.5,1])
    cbar.solids.set_rasterized(False)
    cbar.ax.set_xticklabels(['0%','50%','100%']);cbar.outline.set_visible(False);cbar.ax.tick_params(length=0,labelsize=8)
    fig.text(.065,.95,'Goal-conditioned pick-and-place: matched training coverage',fontsize=13,fontweight='bold')
    fig.text(.065,.875,'24 teacher episodes per condition · same CNN–GRU, 2,500 updates and 3 seeds · current RGB + robot state + goal',fontsize=9,color='#4d5963')
    fig.text(.065,.035,'Dots: three model seeds. Each matrix cell: two starts × three seeds. Outlines: training-covered pairs. Known locations; new combinations.',fontsize=8,color='#4d5963')
    fig.savefig(out/'coverage_results.svg',facecolor='white')
    fig.savefig(out/'coverage_results.png',dpi=300,facecolor='white')
    plt.close(fig)
    caption=('Matched-budget training coverage study. Both conditions use 24 executed teacher episodes, identical source/goal marginal frequencies, '
        'paired physical initial scenes, learner, initialization, update budget and augmentation streams. Panel a gives pooled strict-success counts with '
        'the three seed-level rates as individual points; no confidence interval is implied. Panels b/c show all 16 combinations, each evaluated at 2 object '
        'starts for 3 model seeds. Outlined cells occurred in training. Test tasks share 8 physical initial poses; the 96 executions per condition are not 96 '
        'independent environments. Goal positions are known; four entire combinations are held out from both training sets. Teacher success differs '
        '24/24 versus23/24 and all trajectories are retained. Both groups share existing personal-video goal geometry; no new video timing or real-robot result is claimed.')
    (out/'CAPTION_EN.md').write_text(caption+'\n',encoding='utf-8')
    write(out/'manifest.json',dict(created_utc=now(),results_sha256=sha(ROOT/'artifacts/results_v1/results.json'),
        code_sha256=sha(ROOT/'plot_results.py'),outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()},
        format='editable vector SVG and300dpi PNG',error_bars='none; points are observed seed rates'))
    print(out)


if __name__=='__main__':
    main()
