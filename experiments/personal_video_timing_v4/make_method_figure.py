"""Vector method figure; dimensions and information flow, without performance claims."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyBboxPatch,FancyArrowPatch
from common import ROOT,read,write,sha,now


def main():
    study=read(ROOT/'configs/study_v1.json')
    teacher=read(ROOT/'artifacts/robot_demos_v1/manifest.json')
    assert teacher['status']=='completed' and teacher['training_gate_passed']
    out=ROOT/'artifacts/figures/method_v1';out.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'Arial','font.size':10,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(12,7.2),facecolor='white')
    ax=fig.add_axes([.055,.58,.40,.34]);ax.set_aspect('equal');ax.axis('off')
    ax.set_xlim(-2,35);ax.set_ylim(-2,16)
    for x,color,prefix in [(0,'#c8524c','P'),(23,'#187cbe','G')]:
        ax.add_patch(Rectangle((x,0),10,13,ec='#9aa6ad',fc='#f7f9fa',lw=1.1))
        for n,(dx,dy) in enumerate([(2.5,9.75),(7.5,9.75),(2.5,3.25),(7.5,3.25)],1):
            ax.add_patch(Rectangle((x+dx-.3,dy-.3),.6,.6,fc=color,ec='none'))
            ax.text(x+dx,dy+.8,f'{prefix}{n}',ha='center',color=color,fontsize=10,fontweight='bold')
    ax.annotate('',(23,6.5),(10,6.5),arrowprops={'arrowstyle':'<->','color':'#64727a'})
    ax.text(16.5,7.4,'13 cm gap',ha='center',fontsize=9,color='#64727a')
    ax.text(5,-1.5,'10 × 13 cm',ha='center',fontsize=9);ax.text(28,-1.5,'10 × 13 cm',ha='center',fontsize=9)
    ax.text(-1,15.3,'a   Prescribed start and goal',fontweight='bold',fontsize=11)
    ax.text(16.5,1.8,'Schematic',ha='center',fontsize=8,color='#64727a')
    ax.add_patch(Rectangle((23+7.5-2,3.25-2),4,4,fill=False,ec='#187cbe',ls='--',lw=1))
    top=fig.add_axes([.50,.60,.45,.31]);top.axis('off')
    top.text(0,.94,'b   Causal deployment and strict scoring',fontsize=11,fontweight='bold')
    lines=[('Given before execution','Goal XY, corresponding to the selected G1–G4'),
           ('Policy input at each step','Current RGB + robot state + goal + previous command'),
           ('Learned output','Goal-relative XY, absolute Z and gripper width'),
           ('Success criterion','Lift, release, withdrawal and stable full-footprint\nplacement in a 40 mm target square')]
    for y,(head,body) in zip([.73,.51,.29,.07],lines):
        top.text(0,y,head,fontsize=9,fontweight='bold',color='#324b59')
        top.text(0,y-.085,body,fontsize=9,va='top',linespacing=1.25)
    flow=fig.add_axes([.045,.10,.91,.40]);flow.axis('off');flow.set_xlim(0,1);flow.set_ylim(0,1)
    flow.text(0,1.04,'c   Paired training intervention',fontsize=11,fontweight='bold')
    def box(x,y,w,h,title,body,color='#f0f4f6',edge='#a6b4bc'):
        flow.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.015',ec=edge,fc=color,lw=1))
        flow.text(x+.012,y+h-.075,title,fontsize=9.2,fontweight='bold',va='top')
        flow.text(x+.012,y+h-.19,body,fontsize=8.7,va='top',linespacing=1.38)
    def arrow(a,b,color='#596b75'):
        flow.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=11,lw=1.1,color=color,connectionstyle='arc3,rad=0'))
    box(.01,.24,.23,.59,'24 personal training clips','S01: 8 same-number clips\nX01: 16 cross-goal clips\n\nFirst-frame goal geometry\nCoarse time + 2D progress\nExplicit missing-label masks')
    box(.32,.57,.29,.30,'Shared geometry','24 simulated teacher rollouts\nStandard transport timing')
    box(.32,.14,.29,.30,'+ Personal timing / progress','Same 24 initial scenes and goals\nVideo guides transport commands',color='#e7f4f4',edge='#408d9b')
    box(.71,.24,.28,.59,'Identical CNN–GRU learners','12 P/G pairs × 2 demonstrations\n2,500 updates × 3 fixed seeds\nCurrent RGB + 12D input\nFinal EMA; all seeds retained\n\nX02: label development only\nX03: no training contribution')
    arrow((.24,.68),(.31,.72));arrow((.24,.39),(.31,.29),'#007d96')
    arrow((.62,.72),(.70,.65));arrow((.62,.29),(.70,.40),'#007d96')
    flow.text(.47,.49,'Only intervention: transport timing + progress',ha='center',fontsize=8,color='#4c606b')
    fig.text(.055,.035,'Robot action labels come from executed simulation commands. Human metric actions are not recovered. Runtime object truth is used only for scoring.',fontsize=8.5,color='#52636d')
    for suffix in ('svg','png'):fig.savefig(out/f'method.{suffix}',dpi=300,facecolor='white')
    plt.close(fig)
    write(out/'manifest.json',dict(created_utc=now(),study_sha256=sha(ROOT/'configs/study_v1.json'),
        code_sha256=sha(ROOT/'make_method_figure.py'),scope='Method diagram; schematic point placement, no measured performance',
        outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(str(out/'method.png'))


if __name__=='__main__':main()
