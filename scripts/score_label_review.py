"""Score independently completed development boxes; reject incomplete annotations."""
import argparse,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/diagnostics/label_review_release_v1'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('annotations',type=Path);args=ap.parse_args()
    annotated=json.loads(args.annotations.read_text(encoding='utf-8-sig'))
    items=json.loads((BASE/'sample_manifest.json').read_text())['items']
    predictions={r['id']:r for r in json.loads((BASE/'predictions_for_scoring.json').read_text())}
    answers=annotated.get('answers',{})
    if not annotated.get('reviewer') or set(answers)!=set(r['id'] for r in items):raise ValueError('Named reviewer and all 48 annotations required')
    ious=[];centres=[];visible=0;detected=0;false_positive=0;unjudgeable=0
    for item in items:
        a=answers[item['id']];p=predictions[item['id']]['object'];status=a['visibility']
        if status not in ('visible','partial','absent','unjudgeable'):raise ValueError('Unknown visibility')
        valid=p.get('auxiliary_mask_candidate',False)
        if status=='unjudgeable':unjudgeable+=1;continue
        if status=='absent':false_positive+=int(valid);continue
        visible+=1
        box=np.asarray(a['bbox_xywh_px'],float)
        if box.shape!=(4,) or not np.isfinite(box).all() or (box[2:]<=0).any() or (box[:2]<0).any() or (box[:2]+box[2:]>[item['width'],item['height']]).any():raise ValueError('Invalid annotation box')
        if not valid:continue
        detected+=1;q=np.asarray(p['bbox_xywh_px'],float)
        inter=np.maximum(0,np.minimum(box[:2]+box[2:],q[:2]+q[2:])-np.maximum(box[:2],q[:2])).prod()
        ious.append(float(inter/(box[2:].prod()+q[2:].prod()-inter)))
        centres.append(float(np.linalg.norm(box[:2]+box[2:]/2-q[:2]-q[2:]/2)))
    result={'scope':'48 development frames; not independent generalization or hand accuracy','reviewer':annotated['reviewer'],'visible_frames':visible,'accepted_on_visible':detected,'accepted_on_absent':false_positive,'unjudgeable':unjudgeable,'conditional_mean_box_iou':float(np.mean(ious)) if ious else None,'conditional_mean_box_center_error_px':float(np.mean(centres)) if centres else None,'note':'Conditional metrics exclude missing predictions. Visible-component box centers, not hidden geometric centers.'}
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
