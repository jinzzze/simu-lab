"""Data-derived, vector scientific figure for the frozen v6 study (no experiments)."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.text import Text

ROOT = Path(__file__).resolve().parent
REV = 'paired_recurrent_goal_rgbaug_v6'
REPORTS = ROOT / 'artifacts/reports' / REV
W, H = 180 / 25.4 * 72, 230 / 25.4 * 72
INK = '#20313D'; MUTED = '#596770'; RULE = '#CCD4D8'
BLUE = '#0072B2'; ORANGE = '#D55E00'; RED = '#C7554A'
PALE_BLUE = '#ECF4F9'; PALE_ORANGE = '#FCF1E8'; PALE = '#F5F7F8'
CONDITIONS = ('simulation_only', 'video_guided')

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, value): Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--draft', default='draft01')
    args = parser.parse_args()
    out = ROOT / 'artifacts/figures/paired_control_overview_v1'
    draft = out / 'tmp/pdfs' / args.draft
    assert not draft.exists(), 'Use a fresh draft name to preserve previous renderings'
    draft.mkdir(parents=True)
    cfg_path = ROOT / 'configs/policy_recurrent_goal_rgbaug_v6.json'
    task_path = ROOT / 'configs/sim_task_v1.json'
    cfg, task = read(cfg_path), read(task_path)
    sources = {str(p.relative_to(ROOT)): sha(p) for p in (cfg_path, task_path)}
    suites = ('nominal', 'appearance', 'wider_start', 'novel_appearance')
    results = {}
    for suite in suites:
        path = REPORTS / f'{suite}_analysis_v1/results.json'
        audit = REPORTS / f'{suite}_validation.json'
        assert audit.exists()
        results[suite] = read(path)
        assert results[suite]['training_seeds'] == [7, 17, 27]
        for p in (path, audit): sources[str(p.relative_to(ROOT))] = sha(p)
    recomb_path = REPORTS / 'consistent_goal_recombination_v1/analysis_v1/results.json'
    recomb = read(recomb_path); sources[str(recomb_path.relative_to(ROOT))] = sha(recomb_path)
    index_path = ROOT / 'artifacts/reports/paired_robot_demos_v1/training_index.json'
    index = read(index_path); sources[str(index_path.relative_to(ROOT))] = sha(index_path)
    assert all(len(index['train_groups'][c]) == 16 for c in CONDITIONS)
    assert len(index['shared_validation']) == 8
    assert cfg['training']['steps'] == 2500 and cfg['training']['batch_size'] == 2
    assert task['horizontal_grid_edge_gap_m'] == .13 and task['goal_region_side_m'] == .04
    for c in CONDITIONS:
        matrix = np.asarray(recomb['success_matrices'][c])
        assert int(matrix.trace()) == recomb['groups'][c]['seen_successes']
        assert int(matrix.sum() - matrix.trace()) == recomb['groups'][c]['unseen_successes']

    plt.rcParams.update({'font.family': 'Arial', 'font.size': 7.3, 'text.color': INK,
                         'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.linewidth': .6, 'svg.hashsalt': 'paired_control_overview_v1',
                         'mathtext.fontset': 'dejavusans'})
    fig = plt.figure(figsize=(W / 72, H / 72), facecolor='white')
    canvas = fig.add_axes([0, 0, 1, 1]); canvas.set(xlim=(0, W), ylim=(H, 0)); canvas.axis('off')
    texts = []
    def txt(x, y, s, size=7.3, color=INK, weight='normal', ha='left', va='top', **kw):
        obj = canvas.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                          ha=ha, va=va, linespacing=1.25, **kw); texts.append(obj); return obj
    def line(x1, y1, x2, y2, color=RULE, lw=.6, **kw):
        canvas.plot([x1, x2], [y1, y2], color=color, lw=lw, **kw)
    def arrow(a, b, color=INK, style='-|>', lw=.75, scale=7, connectionstyle='arc3,rad=0', **kw):
        canvas.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=scale,
                         linewidth=lw, color=color, connectionstyle=connectionstyle, shrinkA=0, shrinkB=0, **kw))
    def box(x, y, w, h, fill=PALE, edge=RULE):
        canvas.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=2',
                                      facecolor=fill, edgecolor=edge, linewidth=.65))
    def title(letter, label, x, y):
        txt(x, y, letter, size=10.2, weight='bold'); txt(x + 15, y + .4, label, size=8.7, weight='bold')
    def axes_at(x, y, w, h):
        return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])

    txt(12, 12, 'Goal-conditioned pick-and-place', size=13, weight='bold')
    txt(12, 30, 'Task, recurrent imitation strategy and measured simulation results (v6)', size=8.7, color=MUTED)
    line(12, 46, W - 12, 46, color=RULE)

    # a: vector layout uses the actual 100 x 130 mm grids and 130 mm edge gap.
    title('a', 'Task and strict success', 12, 57)
    scale = .60; left = 32.; top = 91.; grid_h = 130 * scale
    txt(62, 75, 'PICK SITES', size=7.0, color=RED, weight='bold', ha='center')
    txt(200, 75, 'GOAL CENTERS', size=7.0, color=BLUE, weight='bold', ha='center')
    for offset in (0, 230):
        x0 = left + offset * scale
        canvas.add_patch(Rectangle((x0, top), 100 * scale, grid_h, facecolor='#FAFBFC', edgecolor='#8997A1', lw=.65))
        for mm in range(10, 100, 10): line(x0 + mm * scale, top, x0 + mm * scale, top + grid_h, color='#E7ECEF', lw=.32)
        for mm in range(10, 130, 10): line(x0, top + mm * scale, x0 + 60, top + mm * scale, color='#E7ECEF', lw=.32)
    for i, p in enumerate(task['starts'].values(), 1):
        x = left + (p['xy_m'][0] - task['source_grid_top_left_xy_m'][0]) * 1000 * scale
        y = top + (task['source_grid_top_left_xy_m'][1] - p['xy_m'][1]) * 1000 * scale
        canvas.add_patch(Circle((x, y), 1.8, color=RED, zorder=4))
        txt(x, y + 6, f'P{i}', size=6.8, color=RED, ha='center')
        if i == 1:
            canvas.add_patch(Rectangle((x - 6, y - 6), 12, 12, facecolor='#2C3134', edgecolor='white', lw=.5, zorder=4))
    for i, g in enumerate(task['goals'].values(), 1):
        x = left + 230 * scale + g['human_grid_center_mm'][0] * scale
        y = top + g['human_grid_center_mm'][1] * scale
        canvas.add_patch(Rectangle((x - 12, y - 12), 24, 24, fill=False,
                                  edgecolor=ORANGE if i == 1 else BLUE, lw=1.1 if i == 1 else .65, zorder=3))
        canvas.add_patch(Circle((x, y), 1.8, color=BLUE, zorder=4))
        txt(x, y + 4, f'G{i}', size=6.5, color=BLUE, ha='center')
    line(131, 103, 131, 161, color='#3B4145', lw=1.35)
    arrow((101, 133), (161, 133), color=INK, scale=7)
    txt(131, 115, 'pick / lift / place', size=6.6, ha='center', bbox={'facecolor':'white','edgecolor':'none','pad':1})
    txt(131, 163, 'divider', size=6.0, color=MUTED, ha='center')
    for x1, x2, label in ((32, 92, '100 mm'), (92, 170, '130 mm gap'), (170, 230, '100 mm')):
        line(x1, 174, x1, 182, lw=.5); line(x2, 174, x2, 182, lw=.5)
        arrow((x1 + 1, 178), (x2 - 1, 178), style='<->', color=MUTED, lw=.55, scale=5)
        txt((x1 + x2) / 2, 183, label, size=6.6, color=MUTED, ha='center')
    arrow((24, top), (24, top + grid_h), style='<->', color=MUTED, lw=.55, scale=5)
    txt(15, top + grid_h / 2, '130 mm', size=6.6, color=MUTED, rotation=90, ha='center', va='center')
    txt(12, 199, 'Aligned SIM layout; G1 shown as the selected goal.', size=6.6, color=MUTED)
    txt(12, 211, 'Strict: lift, full footprint in 40 mm square, release,', size=7.0)
    txt(12, 222, 'withdrawal and stable placement for at least 0.5 s.', size=7.0)

    # b: personal data reaches both conditions through geometry, only B through timing/progress.
    title('b', 'Training data and paired contrast', 266, 57)
    box(267, 76, 231, 31)
    txt(382.5, 82, '16 personally recorded S01 clips', size=7.8, weight='bold', ha='center')
    txt(382.5, 94, 'Goal geometry + transport timing / 2D progress', size=6.8, ha='center', color=MUTED)
    arrow((324, 107), (324, 123), color=BLUE)
    arrow((441, 107), (441, 123), color=ORANGE)
    box(267, 123, 110, 37, fill=PALE_BLUE, edge=BLUE)
    box(388, 123, 110, 37, fill=PALE_ORANGE, edge=ORANGE)
    txt(322, 129, 'A  No added timing', size=7.3, color=BLUE, weight='bold', ha='center')
    txt(322, 143, '16 simulated robot demos', size=6.8, ha='center')
    txt(443, 129, 'B  + Timing/progress', size=7.0, color=ORANGE, weight='bold', ha='center')
    txt(443, 143, '16 simulated robot demos', size=6.8, ha='center')
    txt(267, 171, 'Both use S01 goal geometry; matched starts / physics.', size=6.8)
    txt(267, 184, 'Scripted physical teacher supplies RGB / state / actions.', size=6.8)
    txt(267, 197, '32 train + 8 shared development episodes (simulation).', size=6.8)
    txt(267, 210, 'Robot paths and action labels are engineered.', size=6.8, color=MUTED)
    txt(267, 222, 'Phone frames are not policy-training RGB.', size=6.8, color=MUTED)

    # c: actual online architecture and frozen optimization scheme.
    line(12, 236, W - 12, 236)
    title('c', 'Goal-conditioned recurrent behavior cloning (v6)', 12, 245)
    box(18, 269, 88, 35)
    txt(62, 279, 'Current RGB\n128 x 128', size=7.4, ha='center')
    arrow((106, 286.5), (123, 286.5))
    box(123, 269, 93, 35)
    txt(169.5, 277, '3-layer CNN\n64D visual features', size=7.3, ha='center')
    arrow((216, 286.5), (231, 286.5))
    box(231, 269, 18, 35, fill='white')
    txt(240, 286.5, 'concat', size=6.7, va='center', ha='center', rotation=90)
    arrow((249, 286.5), (264, 286.5))
    box(264, 269, 77, 35, fill=PALE_BLUE)
    txt(302.5, 278, 'GRU\n128D memory', size=7.4, ha='center')
    arrow((329, 269), (276, 269), color=BLUE, connectionstyle='arc3,rad=0.40', lw=.7, scale=6)
    arrow((341, 286.5), (359, 286.5))
    box(359, 269, 139, 35, fill=PALE_BLUE)
    txt(428.5, 277, 'XY offset from goal\nabsolute Z + gripper width', size=7.2, ha='center')
    box(18, 318, 198, 30)
    txt(117, 324, '12D vector: EE xyz, width, goal xy,\ngoal - EE xy, previous effective command', size=6.9, ha='center')
    line(216, 333, 240, 333, color=INK, lw=.75); arrow((240, 333), (240, 304))
    arrow((428.5, 304), (428.5, 318))
    box(359, 318, 139, 30)
    txt(428.5, 324, 'Coordinate decode + limits + IK\n25 Hz closed-loop execution', size=6.9, ha='center')
    txt(262, 323, 'Goal fixed before reset;\nXY + visual highlight.', size=6.6, color=MUTED)
    txt(18, 361, 'Weighted BC: physical XYZ precision + gripper-width supervision; mean loss per episode.', size=7.0)
    txt(18, 374, '2,500 updates  |  AdamW 3e-4  |  batch 2 episodes  |  3 seeds / condition  |  fixed final EMA', size=7.0)
    txt(18, 387, 'One brightness / contrast / channel-gain transform per episode, paired across conditions.', size=6.9, color=MUTED)

    # d: actual pooled rates and all three per-seed values, not fabricated confidence intervals.
    line(12, 402, W - 12, 402)
    title('d', 'Closed-loop strict success', 12, 413)
    line(27, 438, 39, 438, color=BLUE, lw=3)
    txt(44, 434, 'No added timing', size=6.8, color=BLUE)
    line(151, 438, 163, 438, color=ORANGE, lw=3)
    txt(168, 434, '+ Timing/progress', size=6.8, color=ORANGE)
    ax = axes_at(38, 458, 232, 124)
    ax.set_ylim(0, 114); ax.set_xlim(-.55, 3.55)
    ax.set_yticks([0, 25, 50, 75, 100]); ax.tick_params(axis='y', labelsize=6.4, length=2, pad=2)
    ax.set_ylabel('Strict success (%)', fontsize=7.0, labelpad=4)
    ax.set_xticks(range(4), ['Nominal\n16 scenes', 'Known color\n8 scenes', 'Start jitter\n8 scenes', 'New color\n8 scenes'])
    ax.tick_params(axis='x', labelsize=6.3, length=0, pad=5)
    ax.spines[['top', 'right']].set_visible(False); ax.spines[['bottom', 'left']].set_color('#77848E')
    ax.grid(axis='y', color='#E4E9EC', lw=.45, zorder=0)
    table = []
    for j, condition in enumerate(CONDITIONS):
        color = (BLUE, ORANGE)[j]
        for i, suite in enumerate(suites):
            group = results[suite]['groups'][condition]
            rate = 100 * group['successes'] / group['episodes']; x = i + (-.18 if j == 0 else .18)
            ax.bar(x, rate, width=.30, color=color, alpha=.20, edgecolor=color, linewidth=.65, zorder=2)
            per_seed = []
            for k, seed in enumerate((7, 17, 27)):
                m = results[suite]['per_model'][f'{condition}_seed{seed}']
                value = 100 * m['successes'] / m['episodes']; per_seed.append(value)
                ax.scatter(x + (k - 1) * .071, value, s=9, color=color, edgecolors='white', linewidths=.35, zorder=4)
            ax.text(x, 106, f'{group["successes"]}/{group["episodes"]}',
                    ha='center', va='bottom', fontsize=6.5, color=color, fontweight='bold', clip_on=False)
            table.append({'suite': suite, 'condition': condition, 'successes': group['successes'],
                          'executions': group['episodes'], 'percent': rate, 'per_seed_percent': per_seed})
    txt(12, 609, 'Bars: pooled rate. Dots: all 3 fixed training seeds.', size=6.5, color=MUTED)
    txt(12, 619, 'Labels: successes/executions; scenes reused by seeds.', size=6.5, color=MUTED)

    # e: every cell in the consistent-goal test, including all untrained pairs.
    title('e', 'Consistent goal recombination', 292, 413)
    txt(292, 433, '16 P/G tasks from 4 object-start poses', size=7.0, color=MUTED)
    cmap = LinearSegmentedColormap.from_list('success_rate', ['#F7F9FA', '#B5CEDC', '#1C638B'])
    for j, condition in enumerate(CONDITIONS):
        x = 310 if j == 0 else 420
        color = (BLUE, ORANGE)[j]
        txt(x + 38, 451, ('No added timing', '+ Timing/progress')[j], size=6.7, color=color, weight='bold', ha='center')
        hm = axes_at(x, 467, 76, 76)
        matrix = np.asarray(recomb['success_matrices'][condition])
        hm.pcolormesh(np.arange(5) - .5, np.arange(5) - .5, matrix,
                      vmin=0, vmax=3, cmap=cmap, shading='flat', edgecolors='none', rasterized=False)
        hm.set(xlim=(-.5, 3.5), ylim=(3.5, -.5)); hm.set_aspect('equal')
        hm.set_xticks(range(4), [f'G{k}' for k in range(1, 5)])
        hm.set_yticks(range(4), [f'P{k}' for k in range(1, 5)])
        hm.tick_params(length=0, labelsize=6.0, pad=2)
        for spine in hm.spines.values(): spine.set_linewidth(.5); spine.set_color('#8997A1')
        for row in range(4):
            for col in range(4):
                hm.text(col, row, f'{matrix[row, col]}/3', ha='center', va='center', fontsize=6.7,
                        color='white' if matrix[row, col] == 3 else INK)
            hm.add_patch(Rectangle((row - .5, row - .5), 1, 1, fill=False, edgecolor=INK, lw=1.0))
    txt(292, 565, 'Outlined cells: trained same-number pairs.', size=6.5, color=MUTED)
    ga, gb = [recomb['groups'][c] for c in CONDITIONS]
    txt(292, 578, f'Matched pairs: {ga["seen_successes"]}/12 vs {gb["seen_successes"]}/12', size=7.2)
    txt(292, 591, f'Cross-pair success: {ga["unseen_successes"]}/36 vs {gb["unseen_successes"]}/36', size=7.4, weight='bold')
    gt = recomb['groups']['privileged_teacher']
    txt(292, 607, f'Privileged teacher: cross-pair {gt["unseen_successes"]}/{gt["unseen_executions"]}', size=6.5, color=MUTED)
    txt(292, 617, '(initial object XY; separate control budget).', size=6.5, color=MUTED)
    line(12, 634, W - 12, 634, color=RULE)
    txt(12, 641, 'Simulation evidence only. All fixed seeds / failures retained; independent real-session validation pending.',
        size=6.6, color=MUTED)

    fig.canvas.draw(); renderer = fig.canvas.get_renderer(); outside = []
    for t in fig.findobj(match=Text):
        if not t.get_visible() or not t.get_text(): continue
        bb = t.get_window_extent(renderer)
        if bb.x0 < 0 or bb.y0 < 0 or bb.x1 > fig.bbox.width + .5 or bb.y1 > fig.bbox.height + .5:
            outside.append({'text': t.get_text(), 'bbox': list(bb.bounds)})
    assert not outside, f'Text outside page: {outside}'
    meta = {'Title': 'Goal-conditioned pick-and-place: task, recurrent imitation and v6 evaluation',
            'Author': 'simu-lab', 'Subject': 'Measured simulation results; all fixed seeds retained',
            'Keywords': 'robot learning, recurrent behavior cloning, goal conditioning, evaluation'}
    fig.savefig(draft / 'figure_native.pdf', metadata=meta)
    fig.savefig(draft / 'paired_control_overview.svg', metadata={'Title': meta['Title'], 'Description': meta['Subject']})
    fig.savefig(draft / 'paired_control_overview.png', dpi=600)
    fig.savefig(draft / 'preview.png', dpi=180)
    plt.close(fig)
    shutil.copy2(__file__, draft / Path(__file__).name)
    payload = {'created_utc': datetime.now(timezone.utc).isoformat(), 'revision': REV,
               'figure_size_mm': [180, 230], 'source_sha256': sources, 'builder_sha256': sha(__file__),
               'pooled_results': table, 'recombination': recomb['success_matrices'],
               'scene_reuse': '3 training seeds repeat the same suite scenes; no independent-binomial confidence intervals',
               'text_outside_page': outside, 'new_experiments_performed': False}
    write(draft / 'figure_data_and_sources.json', payload)
    print(json.dumps({'draft': str(draft), 'source_files': len(sources), 'size_mm': [180, 230]}, ensure_ascii=False))

if __name__ == '__main__': main()
