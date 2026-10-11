"""Deploy edited guidance and preserve earlier descriptions/source receipts."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, shutil

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--staging', required=True)
    args = parser.parse_args(); staging = Path(args.staging)
    receipt = ROOT / ('receipts/policy_document_update_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    receipt.mkdir(); shutil.copy2(__file__, receipt / Path(__file__).name)
    before = {}; changed = []
    def preserve(path):
        if path.exists():
            name = str(path.relative_to(PROJECT)).replace('\\', '__').replace('/', '__')
            shutil.copy2(path, receipt / name); before[str(path.relative_to(PROJECT))] = sha(path)
        changed.append(path)
    for name in ('POLICY_REPRODUCTION_ZH.md', 'TRAINING_RUNTIME_DIAGRAM.md'):
        path = ROOT / name; preserve(path); shutil.copy2(staging / name, path)
    path = ROOT / 'SIM_DEMONSTRATIONS.md'; preserve(path)
    text = path.read_text(encoding='utf-8')
    start, end = '<!-- learned-policy-status:start -->', '<!-- learned-policy-status:end -->'
    progress = read(ROOT / 'policy_progress.json')
    completed = progress['trained_models_completed']
    revisions = sum(r['training_completed'] > 0 for r in progress['revisions'])
    notice = (start + f'\n> **2026-10-08：** 下文描述采集教师示范时的冻结方案与来源。后续 {revisions} 种策略修订、{completed} 个模型已经训练，并实际执行闭环物理评测；当前所有种子和新场景结果见 [策略进度](POLICY_PROGRESS_ZH.md)。本页“没有训练”等表述仅属于采集阶段，不代表当前状态。两组共同使用 S01 动作前蓝点目标几何，`simulation_only` 省略的是搬运时长/进度。\n' + end + '\n')
    if start in text: text = text[:text.index(start)] + notice + text[text.index(end) + len(end):].lstrip('\n')
    else:
        at = text.find('\n') + 1; text = text[:at] + '\n' + notice + '\n' + text[at:]
    path.write_text(text, encoding='utf-8')
    for name in ('CHALLENGE_ALIGNMENT.md', 'SUBMISSION_EN.md', 'RELEASE_CHECKLIST_ZH.md'):
        path = PROJECT / 'docs' / name; preserve(path); text = path.read_text(encoding='utf-8')
        start, end = '<!-- paired-extension-update:start -->', '<!-- paired-extension-update:end -->'
        notice = (start + '\n> **2026-10-08 local extension:** The applicant reports the deadline was extended to 16 October; the exact closing time is unconfirmed. The 7 October official-page check remains historical evidence. The separate four-goal learned-control extension uses 16 new personal clips; its [current evidence and limits](../experiments/paired_two_board_v2/LEARNED_POLICY_REPORT_EN.md) and [review guide](../experiments/paired_two_board_v2/REVIEW_GUIDE_EN.md) describe new local results, which have not been published or submitted. Earlier release/experiment claims below keep their original scope.\n' + end + '\n')
        if start in text: text = text[:text.index(start)] + notice + text[text.index(end) + len(end):].lstrip('\n')
        else:
            at = text.find('\n') + 1; text = text[:at] + '\n' + notice + '\n' + text[at:]
        path.write_text(text, encoding='utf-8')
    replay = ROOT / 'artifacts/reports/policy_replays/v5_fixed_first_scene_replay_v1/visual_review/receipt.json'
    if replay.exists():
        data = read(replay)
        if data['visual_review'] == 'pending visual inspection':
            preserve(replay)
            data.update(visual_review='Codex inspected six evenly spaced decoded frames: approach, block grasp/lift, transport, table placement, release and withdrawal are visible; strict footprint/contact decisions remain from the full physical scorer',
                        reviewer='Codex; not applicant human annotation', reviewed_utc=datetime.now(timezone.utc).isoformat())
            write(replay, data)
    write(receipt / 'manifest.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
          'before_sha256': before, 'after_sha256': {str(p.relative_to(PROJECT)): sha(p) for p in changed},
          'source_sha256': sha(__file__), 'model_data_scoring_configs_unchanged': True,
          'cooperation_items_remain_unknown': True})
    print(json.dumps({'receipt': str(receipt), 'documents_updated': len(changed)}, ensure_ascii=False))

if __name__ == '__main__':
    main()
