"""Run the fixed experiment through reports and actual portable replay checks."""
import subprocess
import sys
from common import ROOT,read,write,sha,now


def main():
    state=ROOT/'artifacts/completion_state.json'
    def run(script,*args):
        write(state,dict(status='running',stage=script,updated_utc=now()))
        subprocess.run([sys.executable,'-B','-u',str(ROOT/script),*args],cwd=ROOT,check=True)
    try:
        run('run_pipeline.py')
        if not (ROOT/'artifacts/figures/personal_timing_v1/manifest.json').exists():
            run('report_results.py')
        for condition,tag in [('geometry_only_24','geometry_dev_replay_v1'),('video_timing_24','timing_dev_replay_v1')]:
            path=ROOT/f'artifacts/replays/{tag}/manifest.json'
            if not path.exists():
                run('replay.py','--condition',condition,'--seed','7','--suite','validation','--tag',tag)
            receipt=read(path)
            assert receipt['status']=='completed'
            assert receipt['all_arrays_identical'] and receipt['full_scoring_trace_identical']
        portable=ROOT/'receipts/portable_replay_v1/receipt.json'
        if not portable.exists():run('verify_portable.py','--suite','validation','--revision','v1')
        assert read(portable)['status']=='passed'
        result=read(ROOT/'artifacts/results_v1/results.json')
        assert result['status']=='completed' and result['benchmark_executions']==264
        write(state,dict(status='automated_checks_passed',updated_utc=now(),
            result_sha256=sha(ROOT/'artifacts/results_v1/results.json'),
            portable_receipt_sha256=sha(portable),
            remaining=['Visually inspect scientific figure and replay frames',
                'Refresh submission narrative and project navigation',
                'Public release and applicant review/submission remain separate actions']))
        print('Fixed experiment, reports and portable replay checks completed.',flush=True)
    except Exception as exc:
        write(state,dict(status='needs_attention',updated_utc=now(),error=str(exc)))
        raise


if __name__=='__main__':main()
