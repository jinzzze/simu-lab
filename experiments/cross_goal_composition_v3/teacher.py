"""Initial-state-only teacher with smooth engineered transport; no object feedback."""
import math
import numpy as np


def execute(env,initial_xy,max_speed_m_s=.10,record=False):
    xy=np.array(initial_xy,dtype=float)
    c=env.config;goal=env.goal_xy.copy()
    low,high=c['grasp_ee_z_m'],c['transport_ee_z_m']
    opened,closed=c['open_width_m'],c['closed_width_m']
    env._reset_score();env._step_count=0;env.trace=[];env._record=record
    env.frames=[env.render_oblique()] if record else []
    stages={}
    for name,target,width,settle in [('approach',np.r_[xy,high],opened,8),('descend',np.r_[xy,low],opened,8),
                                    ('close',np.r_[xy,low],closed,25),('lift',np.r_[xy,high],closed,12)]:
        stages[name]=env._move(target,width,name,settle_steps=settle)
    # Quintic interpolation has maximum derivative 1.875; this duration bounds
    # requested path speed without observing true object state during execution.
    duration=max(.5,1.875*np.linalg.norm(goal-xy)/max_speed_m_s)
    ticks=int(math.ceil(duration/env.sim.dt))
    for i in range(1,ticks+1):
        t=i/ticks;progress=10*t**3-15*t**4+6*t**5
        env._tick(np.r_[xy+(goal-xy)*progress,high],closed,'smooth_transport')
    stages['transport']=env._move(np.r_[goal,high],closed,'transport_endpoint_settle',settle_steps=12)
    for name,target,width,settle in [('place',np.r_[goal,low+.001],closed,8),
                                    ('release',np.r_[goal,low+.001],opened,20),('withdraw',np.r_[goal,high],opened,25)]:
        stages[name]=env._move(target,width,name,settle_steps=settle)
    row=env.outcome()
    row.update(stages_reached=stages,teacher='initial-XY privileged teacher; quintic engineered transport',
               teacher_max_transport_speed_m_s=max_speed_m_s,online_object_truth_correction=False,
               human_transport_profile_used=False)
    return row
