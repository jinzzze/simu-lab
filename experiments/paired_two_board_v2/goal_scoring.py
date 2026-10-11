"""Goal containment geometry; simulator truth is allowed ONLY for scoring."""
import math

def footprint_inside_goal(corners_xy, goal_xy, side_m=0.04):
    if len(goal_xy)!=2 or not all(math.isfinite(float(x)) for x in goal_xy):
        raise ValueError('Goal must have two finite coordinates')
    if not math.isfinite(side_m) or side_m<=0:
        raise ValueError('Goal side must be finite and positive')
    corners=list(corners_xy)
    if not corners:
        raise ValueError('Need object footprint corners')
    half=side_m/2
    for point in corners:
        if len(point)!=2 or not all(math.isfinite(float(x)) for x in point):
            raise ValueError('Invalid object corner')
        if any(abs(float(point[i])-float(goal_xy[i]))>half for i in (0,1)):
            return False
    return True

def strict_goal_success(corners_xy, goal_xy, physical_outcome, side_m=0.04):
    # Replace the original line predicate; retain independent physical checks.
    if physical_outcome.get('stability_scope')!='designated_goal_containment':
        raise ValueError('Require stability timer accrued under designated-goal containment, not historical line scoring')
    flags=physical_outcome['final_flags']
    needed=('in_workspace','on_table','upright','slow','released','withdrawn')
    return bool(footprint_inside_goal(corners_xy,goal_xy,side_m)
                and physical_outcome['lifted']
                and physical_outcome['max_lift_duration_s']>=0.12
                and physical_outcome['final_stable_duration_s']>=0.5
                and all(flags[key] for key in needed))

# Integration must accrue the 0.5s stability timer with goal containment at EVERY
# step. A historical line-based final_stable_duration_s is not interchangeable.
