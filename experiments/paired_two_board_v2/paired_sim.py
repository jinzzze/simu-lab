"""Independent four-goal environment; no changes to historical grasp/DP backend."""
from pathlib import Path
import hashlib,json,math,sys
import numpy as np
import pybullet as p

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
sys.path.insert(0,str(PROJECT))
from src.sim.grasp_env import GraspEnv,box_corners,score_instantaneous
from goal_scoring import footprint_inside_goal,strict_goal_success


def goal_conditioned_vector(robot,goal_xy,previous_command):
    """Current robot state + preprovided goal + past command; no clock/object."""
    ee=np.asarray(robot.get_ee_position(),float)
    result=np.r_[ee,robot.get_fingers_width(),goal_xy,np.asarray(goal_xy)-ee[:2],previous_command].astype(np.float32)
    if result.shape!=(12,) or not np.isfinite(result).all():raise ValueError('Invalid goal-conditioned vector')
    return result


def goal_flags(position,quaternion,linear_velocity,angular_velocity,ee_position,finger_width,table_contact,robot_contact,config,goal_xy,side_m=.04):
    flags,clearance=score_instantaneous(position,quaternion,linear_velocity,angular_velocity,ee_position,
                                      finger_width,table_contact,robot_contact,config)
    flags.pop('fully_right')
    flags['inside_goal']=footprint_inside_goal(box_corners(position,quaternion,config['object_size_m'])[:,:2],goal_xy,side_m)
    return flags,clearance


def video_transport_profile(root,episode_id):
    """Offline teacher guide from TRAIN RGB candidates and reviewed time brackets.

    Projection onto the pre-action image start-to-goal direction is merely a
    dimensionless progress proxy. It does not measure lifted physical XY/XYZ.
    Missing human frames remain missing; knots omit them. Desired robot commands
    are engineered by interpolating these constraints, not fabricated labels.
    """
    root=Path(root)
    review_path=root/f'data/processed/S01_trajectories_v1/{episode_id}.reviewed.json'
    rows_path=root/f'data/processed/S01_trajectories_v1/{episode_id}.frames.jsonl'
    cal_path=root/f'data/processed/S01_calibration_v2/{episode_id}.json'
    review=json.loads(review_path.read_text(encoding='utf-8'))
    cal=json.loads(cal_path.read_text(encoding='utf-8'))
    rows=[json.loads(line) for line in rows_path.read_text(encoding='utf-8').splitlines()]
    intake_path=root/f'data/processed/S01_intake_v1/{episode_id}.json'
    intake=json.loads(intake_path.read_text(encoding='utf-8'))
    raw_path=root/intake['source']['relative_video_path']
    raw_digest=hashlib.sha256(raw_path.read_bytes()).hexdigest()
    if raw_digest!=review['source_sha256'] or raw_digest!=intake['source']['sha256']:
        raise ValueError('Human raw source hash mismatch')
    source_files={str(path.relative_to(root)).replace('\\','/'):hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in (review_path,rows_path,cal_path,intake_path,raw_path)}
    if review['split']!='train' or any(r['split']!='train' for r in rows):raise ValueError('Human guide must be train only')
    first=rows[0]['object']['center_px']
    if first is None:raise ValueError('Require valid pre-action visible object center')
    goal=np.array(cal['goals'][cal['prescribed_goal_id']]['center_image_pixels'])
    direction=goal-np.array(first);norm=float(direction@direction)
    if norm<100:raise ValueError('Degenerate pre-action image direction')
    ev={e['name']:e for e in review['events']}
    start=ev['transport']['time_interval_s'][0];end=ev['placement']['time_interval_s'][1]
    if end<=start:raise ValueError('Invalid coarse timing guide')
    knots=[[0.,0.]];source_frames=[]
    for row in rows:
        t=row['timestamp_s'];o=row['object']
        if start<t<end and o['auxiliary_mask_candidate']:
            progress=float(np.clip((np.array(o['center_px'])-first)@direction/norm,0,1))
            knots.append([(t-start)/(end-start),max(knots[-1][1],progress)])
            source_frames.append(row['frame_index'])
    if len(source_frames)<3:raise ValueError('Not enough valid RGB transport constraints')
    knots.append([1.,1.])
    return {'human_episode_id':episode_id,'human_session':'S01','human_split':'train','human_source_sha256':review['source_sha256'],
            'human_derivation_file_sha256':source_files,
            'goal_id':cal['prescribed_goal_id'],'duration_s':end-start,'progress_knots':knots,
            'valid_source_frame_indices':source_frames,'coarse_time_interval_s':[start,end],
            'semantics':'offline dimensionless image-progress/time guide, not recovered physical motion or robot action',
            'runtime_observation':False,'human_missing_labels_interpolated':False,
            'desired_robot_setpoints_interpolated':True,'physical_robot_height_source':'shared engineered grasp/transport heights'}


class PairedGoalEnv(GraspEnv):
    def __init__(self,image_size=224):
        self.task=json.loads((ROOT/'configs/sim_task_v1.json').read_text(encoding='utf-8'))
        self.goal_id='G1';self.goal_xy=np.array(self.task['goals']['G1']['xy_m'])
        self.capture_actions=False
        super().__init__(ROOT/self.task['physics_config'],image_size=image_size)

    def _make_markings(self):
        # Ghost markings have no collision shape: no raised blue tape assistance.
        for board,top_left,color in [('source',self.task['source_grid_top_left_xy_m'],[.9,.86,.86,1]),
                                    ('target',self.task['target_grid_top_left_xy_m'],[.86,.88,.94,1])]:
            x,y=top_left
            self.sim.create_box(f'{board}_paper',np.array([.05,.065,.00001]),0,np.array([x+.05,y-.065,.00002]),rgba_color=np.array(color),ghost=True)
        self.sim.create_box('separator',np.array([.0015,.075,.00001]),0,np.array([.045,0,.00006]),rgba_color=np.array([.02,.02,.02,1]),ghost=True)
        self.goal_visual_bodies={}
        for name,g in self.task['goals'].items():
            x,y=g['xy_m'];bodies=[]
            for side,(dx,dy,hx,hy) in enumerate([(-.02,0,.0004,.02),(.02,0,.0004,.02),(0,-.02,.02,.0004),(0,.02,.02,.0004)]):
                key=f'{name}_edge{side}'
                self.sim.create_box(key,np.array([hx,hy,.00001]),0,np.array([x+dx,y+dy,.00008]),rgba_color=np.array([.1,.25,.85,1]),ghost=True)
                bodies.append(self.sim._bodies_idx[key])
            self.goal_visual_bodies[name]=bodies
        for name,item in self.task['starts'].items():
            self.sim.create_box(name,np.array([.002,.002,.00001]),0,np.r_[item['xy_m'],.00008],rgba_color=np.array([.85,.05,.05,1]),ghost=True)

    def reset_task(self,scene_seed,start_id,goal_id):
        if start_id not in self.task['starts'] or goal_id not in self.task['goals']:raise ValueError('Unknown task IDs')
        self.start_id=start_id;self.goal_id=goal_id
        self.goal_xy=np.array(self.task['goals'][goal_id]['xy_m'],float)
        self.config['place_xy_m']=self.goal_xy.tolist()
        center=np.array(self.task['starts'][start_id]['xy_m']);j=self.task['start_jitter_halfwidth_m']
        self.config['start_low_xy_m']=(center-j).tolist();self.config['start_high_xy_m']=(center+j).tolist()
        for name,bodies in self.goal_visual_bodies.items():
            color=[.95,.7,.05,1] if name==goal_id else [.1,.25,.85,1]
            for body in bodies:self.client.changeVisualShape(body,-1,rgbaColor=color)
        self.capture_actions=False
        super().reset(scene_seed)
        self.previous_command=np.r_[self.robot.get_ee_position(),self.config['open_width_m']]
        return self.policy_observation()

    def policy_observation(self):
        return {'rgb':self.render(),'vector':goal_conditioned_vector(self.robot,self.goal_xy,self.previous_command),
                'goal_xy_m':self.goal_xy.astype(np.float32).copy()}

    def begin_demonstration(self):
        self.observations=[];self.images=[];self.actions=[];self.stages=[];self.commands_requested=[]
        self.capture_actions=True

    def effective_command(self,requested):
        requested=np.asarray(requested,float)
        if requested.shape!=(4,) or not np.isfinite(requested).all():raise ValueError('Finite XYZ+width required')
        limits=self.task['safety_limits']
        clipped=np.clip(requested,limits['xyz_low']+[limits['width_low']],limits['xyz_high']+[limits['width_high']])
        ee=self.robot.get_ee_position();delta=clipped[:3]-ee
        distance=np.linalg.norm(delta)
        if distance>limits['max_ee_step_m']:delta*=limits['max_ee_step_m']/distance
        return np.r_[ee+delta,clipped[3]]

    def _tick(self,target,width,stage):
        requested=np.r_[target,width];command=self.effective_command(requested)
        if self.capture_actions:
            obs=self.policy_observation()
            self.images.append(obs['rgb']);self.observations.append(obs['vector']);self.actions.append(command.astype(np.float32))
            self.commands_requested.append(requested.tolist());self.stages.append(stage)
        ee=self.robot.get_ee_position()
        self.robot.set_action(np.r_[(command[:3]-ee)/.05,(command[3]-self.robot.get_fingers_width())/.2])
        self.sim.step();self._step_count+=1
        self._observe_for_scoring(stage)
        self.previous_command=command.copy()
        if self._record and self._step_count%2==0:self.frames.append(self.render_oblique())

    def step_action(self,action):
        """Deployment action interface: no expert stages, object correction or clock."""
        action=np.asarray(action,float)
        if action.shape!=(4,) or not np.isfinite(action).all():raise ValueError('Invalid action')
        self._tick(action[:3],action[3],'learned_action')
        return self.previous_command.copy()

    def _observe_for_scoring(self,stage):
        # Privileged physics state below is used ONLY for scoring/diagnostic labels.
        pos=self.sim.get_base_position('object');quat=self.sim.get_base_orientation('object')
        obj=self.sim._bodies_idx['object']
        contacts=self.client.getContactPoints(obj,self.sim._bodies_idx['panda'])
        finger_links={c[4] for c in contacts if c[9]>1e-5}
        flags,clearance=goal_flags(pos,quat,self.sim.get_base_velocity('object'),self.sim.get_base_angular_velocity('object'),
            self.robot.get_ee_position(),self.robot.get_fingers_width(),
            bool(self.client.getContactPoints(obj,self.sim._bodies_idx['table'])),bool(contacts),self.config,self.goal_xy,self.task['goal_region_side_m'])
        bilateral={9,10}.issubset(finger_links)
        self._max_clearance=max(self._max_clearance,clearance)
        self._lift_run=self._lift_run+self.sim.dt if clearance>=self.config['min_lift_clearance_m'] and bilateral else 0.
        self._max_lift_run=max(self._max_lift_run,self._lift_run)
        self._stable_run=self._stable_run+self.sim.dt if all(flags.values()) else 0.
        self._flags=flags
        self.trace.append({'step_after_action':self._step_count,'t_s':self._step_count*self.sim.dt,'stage':stage,
                           'object_xyz_m':pos.tolist(),'ee_xyz_m':self.robot.get_ee_position().tolist(),
                           'finger_width_m':float(self.robot.get_fingers_width()),'clearance_m':clearance,
                           'bilateral_grasp_contact':bilateral,'flags':flags.copy(),'goal_stable_duration_s':self._stable_run,
                           'diagnostic_ground_truth_only':True})

    def outcome(self):
        physical={'stability_scope':'designated_goal_containment','final_flags':self._flags.copy(),
                  'lifted':self._max_lift_run+1e-9>=self.config['min_lift_duration_s'],
                  'max_lift_duration_s':self._max_lift_run,'final_stable_duration_s':self._stable_run}
        corners=box_corners(self.sim.get_base_position('object'),self.sim.get_base_orientation('object'),self.config['object_size_m'])
        success=strict_goal_success(corners[:,:2],self.goal_xy,physical,self.task['goal_region_side_m'])
        return dict(physical,success=success,scene_seed=self.seed,start_id=self.start_id,goal_id=self.goal_id,
                    goal_xy_m=self.goal_xy.tolist(),max_clearance_m=self._max_clearance,
                    final_object_xyz_m=self.sim.get_base_position('object').tolist(),
                    diagnostic_ground_truth_only=True,object_attached_or_teleported=False,
                    steps=self._step_count,simulated_duration_s=self._step_count*self.sim.dt)

    def execute_expert(self,initial_estimated_xy,profile=None,record=False,diagnostic_destination=None):
        xy=np.asarray(initial_estimated_xy,float)
        if xy.shape!=(2,) or not np.isfinite(xy).all():raise ValueError('Invalid initial XY')
        c=self.config;goal=self.goal_xy if diagnostic_destination is None else np.asarray(diagnostic_destination,float)
        self._reset_score();self._step_count=0;self.trace=[];self._record=record
        self.frames=[self.render_oblique()] if record else []
        low,high=c['grasp_ee_z_m'],c['transport_ee_z_m'];opened,closed=c['open_width_m'],c['closed_width_m']
        stages={}
        for name,target,width,settle in [
            ('approach',np.r_[xy,high],opened,8),('descend',np.r_[xy,low],opened,8),
            ('close',np.r_[xy,low],closed,25),('lift',np.r_[xy,high],closed,12)]:
            stages[name]=self._move(target,width,name,settle_steps=settle)
        if profile is None:
            stages['transport']=self._move(np.r_[goal,high],closed,'transport')
        else:
            if profile['goal_id']!=self.goal_id:raise ValueError('Video intended goal mismatch')
            minimum=np.linalg.norm(goal-xy)/c['max_ee_step_m']*self.sim.dt
            duration=max(profile['duration_s'],minimum)
            count=int(math.ceil(duration/self.sim.dt));knots=np.array(profile['progress_knots'])
            for index in range(1,count+1):
                progress=np.interp(index/count,knots[:,0],knots[:,1])
                self._tick(np.r_[xy+progress*(goal-xy),high],closed,'video_guided_transport')
            # Endpoint settling uses ONLY the preprovided target and robot state;
            # it does not read the actual object to repair position or grasp.
            stages['transport']=self._move(np.r_[goal,high],closed,'transport_endpoint_settle')
        for name,target,width,settle in [
            ('place',np.r_[goal,low+.001],closed,8),('release',np.r_[goal,low+.001],opened,20),
            ('withdraw',np.r_[goal,high],opened,25)]:
            stages[name]=self._move(target,width,name,settle_steps=settle)
        result=self.outcome();result['stages_reached']=stages
        result['expert_initial_state_source']='caller-supplied XY; collector explicitly uses privileged INITIAL truth for teacher only'
        result['video_guide_applied']=profile is not None
        return result
