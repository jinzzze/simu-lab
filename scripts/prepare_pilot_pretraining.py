"""Export shared raw RGB and independent pseudo-targets for exploratory pretraining."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    report_path=ROOT/'artifacts/diagnostics/pilot_labels_v2/label_report.json'
    report=json.loads(report_path.read_text(encoding='utf-8'))
    decision_path=ROOT/'artifacts/diagnostics/pilot_labels_v2/development_review.json'
    decision=json.loads(decision_path.read_text(encoding='utf-8'))
    assert decision['approved_use']=='exploratory_development_pretraining'
    partition=json.loads((ROOT/'data/manifests/pilot_development_partition.json').read_text(encoding='utf-8'))
    expected={x['sha256'] for x in partition['episodes']}
    assert {v['source_sha256'] for v in report['videos']}==expected and len(report['videos'])==len(expected)
    images=[];objects=[];object_masks=[];hands=[];hand_masks=[];episode_ids=[];frame_ids=[];sample_sources=[]
    videos_metadata=[]
    for v in report['videos']:
        source=ROOT/v['source_video'];assert digest(source)==v['source_sha256']
        label_file=ROOT/v['labels'];rows=[json.loads(t) for t in label_file.read_text(encoding='utf-8').splitlines()]
        assert len(rows)==v['frame_count']
        cap=cv2.VideoCapture(str(source));fps=cap.get(cv2.CAP_PROP_FPS);stride=max(1,round(fps/5));index=0;selected=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            if index%stride==0:
                row=rows[index];assert row['frame_index']==index and row['source_sha256']==v['source_sha256']
                assert row['partition']=='development_only' and row['rgb_retained']
                assert row['object']['depends_on_hand_labels'] is False
                height,width=frame.shape[:2]
                rgb=cv2.cvtColor(cv2.resize(frame,(224,224),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
                images.append(rgb)
                target=np.zeros(4,np.float32);valid=bool(row['object']['auxiliary_mask_candidate'])
                if valid:
                    x,y,w,h=row['object']['bbox_xywh_px'];target[:]=[(x+w/2)/width,(y+h/2)/height,w/width,h/height]
                    assert np.isfinite(target).all() and ((0<=target)&(target<=1)).all()
                else:assert row['object']['bbox_xywh_px'] is None
                objects.append(target);object_masks.append(valid)
                hand=np.zeros((21,2),np.float32);mask=np.zeros(21,bool)
                candidates=row['hand']['landmarks_normalized']
                if len(candidates)==1:
                    xy=np.float32(candidates[0])[:,:2]
                    mask=np.asarray(row['hand']['inside_image_mask'][0],bool)&np.isfinite(xy).all(axis=1)
                    hand[mask]=xy[mask]
                hands.append(hand.reshape(-1));hand_masks.append(mask)
                episode_ids.append(v['episode_id']);frame_ids.append(index)
                sample_sources.append(v['source_video']);selected+=1
            index+=1
        cap.release();assert index==len(rows)
        videos_metadata.append({'episode_id':v['episode_id'],'source_video':v['source_video'],'source_sha256':v['source_sha256'],
                                'labels_sha256':digest(label_file),'sample_stride':stride,'exported_frames':selected})
    out=ROOT/'data/processed/pilot_pretraining_v1.npz'
    np.savez_compressed(out,images=np.stack(images),object_targets=np.stack(objects),object_mask=np.asarray(object_masks,bool),
        hand_targets=np.stack(hands),hand_mask=np.stack(hand_masks),episode_ids=np.asarray(episode_ids),frame_indices=np.asarray(frame_ids),
        source_video=np.asarray(sample_sources))
    metadata={'scope':'exploratory_development_pretraining_only','samples':len(images),'videos':videos_metadata,
        'partition':'all 12 whole episodes in development family; no real held-out generalization claim',
        'object_target_semantics':'normalized center and extent of visible dark RGB component; not amodal or 3D ground truth',
        'object_valid_samples':int(np.sum(object_masks)),'hand_valid_samples':int(np.any(np.stack(hand_masks),axis=1).sum()),
        'hand_target_semantics':'single detected hand 21 normalized xy landmarks; mask is inside-image validity, not confidence',
        'common_rgb_preprocessing':'same full raw frame resized 224x224 for every A/B/C/D group, no object crop or annotated overlays',
        'missing_labels':'image always retained; auxiliary loss masked independently',
        'npz_sha256':digest(out),'label_report_sha256':digest(report_path),'review_decision_sha256':digest(decision_path),
        'duplicate_directory_excluded':'data/raw/pilot_originals','formal_training_quality_validated':False}
    (ROOT/'data/manifests/pilot_pretraining_v1.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:metadata[k] for k in ['samples','object_valid_samples','hand_valid_samples','npz_sha256']}))

if __name__=='__main__':main()
