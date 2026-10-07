"""Conservative RGB-only pilot labels. No temporal filling and no hand dependency."""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class Reference:
    board_to_image: np.ndarray
    image_to_board: np.ndarray
    workspace: np.ndarray
    marker_exclusions: np.ndarray
    boundary_exclusions: np.ndarray
    marker_centers: dict
    marker_corners: dict
    marker_polygons: list
    boundary_polygons: list
    workspace_polygon: np.ndarray
    frame_shape: tuple


def project(points, matrix):
    return cv2.perspectiveTransform(np.asarray(points,np.float32)[None],matrix)[0]


def dark_mask(frame, threshold, kernel=3):
    mask=np.uint8(frame.max(axis=2)<=threshold)*255
    if kernel>1:
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((kernel,kernel),np.uint8))
    return mask


def make_reference(frame, marker_corners, geometry, cfg):
    expected=[m['id'] for m in geometry['markers']]
    found={int(k):np.asarray(v,np.float32) for k,v in marker_corners.items()}
    if not all(i in found for i in expected):
        raise ValueError('First frame must show both markers; no later-frame fallback is allowed.')
    src=np.concatenate([np.float32(m['corners_mm']) for m in geometry['markers']])
    dst=np.concatenate([found[i] for i in expected])
    h,_=cv2.findHomography(src,dst,0)
    if h is None or not np.isfinite(h).all(): raise ValueError('Invalid first-frame board reference')
    height,width=frame.shape[:2]
    workspace_poly=project(cfg['work_polygon_nominal_board_units'],h)
    workspace=np.zeros((height,width),np.uint8)
    cv2.fillPoly(workspace,[np.rint(workspace_poly).astype(np.int32)],255)
    markers=np.zeros_like(workspace)
    polys=[found[i] for i in expected]
    for p in polys:cv2.fillPoly(markers,[np.rint(p).astype(np.int32)],255)
    margin=cfg['marker_exclusion_margin_px']
    markers=cv2.dilate(markers,np.ones((2*margin+1,2*margin+1),np.uint8))
    boundaries=np.zeros_like(workspace); boundary_polys=[]
    contours,_=cv2.findContours(dark_mask(frame,cfg['dark_max_channel'],cfg['morph_open_kernel']),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x,y,w,bh=cv2.boundingRect(contour)
        rw,rh=cv2.minAreaRect(contour)[1]
        if min(rw,rh)<1:continue
        center=(min(width-1,x+w//2),min(height-1,y+bh//2))
        if workspace[center[1],center[0]] and max(rw,rh)/min(rw,rh)>4 and cv2.contourArea(contour)>500:
            cv2.drawContours(boundaries,[contour],-1,255,-1)
            boundary_polys.append(cv2.convexHull(contour).reshape(-1,2).astype(float).tolist())
    margin=cfg['boundary_exclusion_margin_px']
    boundaries=cv2.dilate(boundaries,np.ones((2*margin+1,2*margin+1),np.uint8))
    if not boundary_polys:raise ValueError('No elongated black boundary found in first frame; manual review needed.')
    return Reference(h,np.linalg.inv(h),workspace,markers,boundaries,
                     {i:found[i].mean(axis=0) for i in expected},{i:found[i] for i in expected},polys,boundary_polys,workspace_poly,(height,width))


def bbox_iou(a,b):
    ax,ay,aw,ah=a; bx,by,bw,bh=b
    area=max(0,min(ax+aw,bx+bw)-max(ax,bx))*max(0,min(ay+ah,by+bh)-max(ay,by))
    return area/max(1,aw*ah+bw*bh-area)


def expected_area(center, ref, cfg):
    board_center=project([center],ref.image_to_board)[0]
    half=np.asarray(cfg['object_size_mm'],np.float32)/(2*cfg['assumed_print_scale'])
    points=board_center+np.array([[-half[0],-half[1]],[half[0],-half[1]],[half[0],half[1]],[-half[0],half[1]]])
    return float(abs(cv2.contourArea(project(points,ref.board_to_image))))


def detect_object(frame, ref, cfg, reference_ok=True):
    """Current-frame dark component only. Output is a candidate, not amodal truth."""
    result={'status':'missing','bbox_xywh_px':None,'center_px':None,'polygon_px':None,
            'auxiliary_mask_candidate':False,'reason':'no_compact_dark_candidate','accepted_candidates':[],
            'rejected_counts':{},'depends_on_hand_labels':False,'temporally_filled':False,
            'target_semantics':'visible_dark_component_bbox_and_center_not_amodal_object_truth'}
    if frame.shape[:2]!=ref.frame_shape:raise ValueError('Frame resolution changed')
    if not reference_ok:
        result.update(status='uncertain',reason='reference_marker_shift_or_unavailable')
        return result
    mask=dark_mask(frame,cfg['dark_max_channel'],cfg['morph_open_kernel'])
    other=dark_mask(frame,cfg['comparison_dark_max_channel'],cfg['morph_open_kernel'])
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    secondary,_=cv2.findContours(other,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    secondary_boxes=[cv2.boundingRect(c) for c in secondary if cv2.contourArea(c)>50]
    height,width=ref.frame_shape
    def reject(why):result['rejected_counts'][why]=result['rejected_counts'].get(why,0)+1
    for c in contours:
        area=float(cv2.contourArea(c))
        if area<100:continue
        x,y,w,h=cv2.boundingRect(c); center=np.array([x+w/2,y+h/2])
        points=c.reshape(-1,2)
        if not ref.workspace[min(height-1,int(center[1])),min(width-1,int(center[0]))]:continue
        if x<=0 or y<=0 or x+w>=width or y+h>=height or not np.all(ref.workspace[points[:,1],points[:,0]]):
            reject('truncated_or_outside_workspace');continue
        if ref.marker_exclusions[y:y+h,x:x+w].any():reject('marker_region');continue
        if ref.boundary_exclusions[y:y+h,x:x+w].any():reject('boundary_region_or_merge');continue
        rect=cv2.minAreaRect(c); a,b=rect[1]
        if min(a,b)<1:continue
        ratio=area/max(1,expected_area(center,ref,cfg))
        solidity=area/max(1,cv2.contourArea(cv2.convexHull(c)))
        rectangularity=area/(a*b)
        aspect=max(a,b)/min(a,b)
        if not cfg['min_area_ratio']<=ratio<=cfg['max_area_ratio']:reject('size');continue
        if solidity<cfg['min_solidity'] or rectangularity<cfg['min_rectangularity'] or aspect>cfg['max_rotated_aspect']:
            reject('shape_or_partial_visibility');continue
        stability=max([bbox_iou((x,y,w,h),bb) for bb in secondary_boxes],default=0)
        if stability<cfg['min_threshold_bbox_iou']:reject('threshold_sensitive');continue
        result['accepted_candidates'].append({'bbox_xywh_px':[x,y,w,h],
            'center_px':[float(v) for v in rect[0]],'polygon_px':cv2.boxPoints(rect).astype(float).tolist(),
            'area_px':area,'approx_table_size_ratio':ratio,'solidity':solidity,'rectangularity':rectangularity,
            'threshold_bbox_iou':stability})
    candidates=result['accepted_candidates']
    if len(candidates)==1:
        c=candidates[0]
        result.update(status='visible_candidate',bbox_xywh_px=c['bbox_xywh_px'],center_px=c['center_px'],
                      polygon_px=c['polygon_px'],auxiliary_mask_candidate=True,reason='rgb_shape_size_threshold_checks')
    elif len(candidates)>1:
        result.update(status='ambiguous',reason='multiple_compact_dark_candidates')
    return result


def check_reference(marker_corners, ref, max_shift):
    present={int(k):np.float32(v) for k,v in marker_corners.items() if int(k) in ref.marker_centers}
    centers={str(k):float(np.linalg.norm(v.mean(axis=0)-ref.marker_centers[k])) for k,v in present.items()}
    corners={str(k):float(np.max(np.linalg.norm(v-ref.marker_corners[k],axis=1))) for k,v in present.items()}
    diagnostics={'center_shift_px':centers,'max_corner_shift_px':corners,'workspace_similarity_max_shift_px':None}
    if not present:return False,diagnostics
    src=np.concatenate([ref.marker_corners[k] for k in present])
    dst=np.concatenate([present[k] for k in present])
    # Similarity fit catches rotation/scale around a single visible marker while avoiding
    # the severe perspective extrapolation noise of a four-corner projective fit.
    transform,_=cv2.estimateAffinePartial2D(src,dst,method=cv2.LMEDS)
    if transform is None:return False,diagnostics
    moved=cv2.transform(ref.workspace_polygon[None],transform)[0]
    drift=float(np.max(np.linalg.norm(moved-ref.workspace_polygon,axis=1)))
    diagnostics['workspace_similarity_max_shift_px']=drift
    return max(corners.values())<=max_shift and drift<=max_shift,diagnostics


def hand_object_overlap_qc(hands, obj, shape):
    """Diagnostic projection overlap; never modifies the object auxiliary mask."""
    if obj['bbox_xywh_px'] is None or not hands:return None
    height,width=shape[:2];x,y,w,h=obj['bbox_xywh_px']
    rect=np.float32([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])
    overlap=0.
    for hand in hands:
        hull=cv2.convexHull(np.float32([[p[0]*width,p[1]*height] for p in hand]))
        if cv2.contourArea(hull)>0:
            area,_=cv2.intersectConvexConvex(rect,hull)
            overlap=max(overlap,float(area/max(1,w*h)))
    return {'projected_hand_hull_bbox_overlap_fraction':overlap,
            'possible_occlusion_or_contact':overlap>0.05,'used_for_object_mask':False,
            'note':'2D overlap only; not proof of occlusion, touch or grasp.'}
