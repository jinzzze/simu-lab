"""Preserve applicant-supplied approximate times without inventing precision."""
from common import ROOT,read,write,sha,now

def main():
    path=ROOT/'receipts/capture_metadata_v1.json';assert not path.exists()
    sessions=[
        dict(session_id='X01',split='train',approximate_local_time='2026-10-09T20:00:00+01:00',
             time_precision='approximately an hour stated by applicant',view_change='base recording view'),
        dict(session_id='X02',split='val',approximate_local_time='2026-10-09T20:30:00+01:00',
             time_precision='approximately 30 minutes after X01',view_change='applicant reports slightly changed viewing angle'),
        dict(session_id='X03',split='test',approximate_local_time=None,
             plausible_local_interval=['2026-10-09T21:00:00+01:00','2026-10-09T21:30:00+01:00'],
             time_precision='One hour later; reference could be X01 or X02 and remains unresolved',
             view_change='applicant reports another slight viewing-angle change')]
    write(path,dict(created_utc=now(),source='direct applicant message in this chat',timezone='Europe/London',
        utc_offset_on_recording_date='+01:00',
        applicant_statement='X01 10.9, 伦敦时间晚上八点 X02半小时后拍摄，略微改变拍摄视角 X03一小时后拍摄， 再次略微改变视角',
        sessions=sessions,scope='Separate same-day recording blocks with small view changes; same table/object scene family, not evidence of broad environment diversity.',
        lighting_change_confirmed=False,actual_order_within_each_batch_known=False,
        timestamps_inferred_from_files=False))
    print({'metadata':str(path),'sessions':3,'x03_time_ambiguous':True})

if __name__=='__main__':main()
