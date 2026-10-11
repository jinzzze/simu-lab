"""Record the applicant's naming confirmation and dated submission roadmap."""
from common import ROOT,read,write,sha,now


def main():
    plan=ROOT/'data/processed/test_intake/x03_naming_v1/rename_plan.json'
    mapping=read(plan)['mapping']
    raw=ROOT/'data/raw/X03_test'
    assert len(mapping)==16
    for row in mapping:
        assert sha(raw/row['new_name'])==row['sha256']
    target=ROOT/'receipts/x03_applicant_confirmation_v1.json'
    assert not target.exists()
    write(target,{
        'recorded_utc':now(),'client_date':'2026-10-10','session_id':'X03','split':'test',
        'source':'Direct user message in current chat',
        'applicant_message':'检查完了 都对的， 现在距离提交还差哪些步骤',
        'scope':'Applicant confirmation of the reviewed test-file names and corresponding P/G mappings.',
        'rename_plan_sha256':sha(plan),'confirmed_files':[
            {'filename':r['new_name'],'sha256':r['sha256'],
             'start_id':r['observed_start_id'],'goal_id':r['observed_destination_id']}
            for r in mapping],
        'names_and_pg_mapping_confirmed':True,
        'recording_dates_confirmed':False,'independent_sessions_confirmed':False,
        'strict_physical_success_confirmed':False,'training_allowed':False
    })
    roadmap=ROOT/'SUBMISSION_ROADMAP_20261010_ZH.md'
    assert roadmap.exists()
    write(ROOT/'receipts/submission_readiness_latest.json',{
        'updated_utc':now(),'roadmap':str(roadmap.relative_to(ROOT)),'roadmap_sha256':sha(roadmap),
        'applicant_confirmation':str(target.relative_to(ROOT)),'applicant_confirmation_sha256':sha(target),
        'new_videos_received':44,'intake_completed':True,
        'new_video_calibration_completed':False,'new_video_comparison_completed':False,
        'new_release_published':False,'application_submitted':False,
        'submission_deadline_local':'2026-10-16T23:59:00+01:00',
        'deadline_source':'https://uk.linkedin.com/company/humanoidai',
        'internal_submission_target':'2026-10-15'
    })
    print({'confirmed_test_files':16,'new_videos_received':44,'roadmap':str(roadmap)})


if __name__=='__main__':
    main()
