"""Record the completed human-visible figure check and local report-link audit."""
import re
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET
from PIL import Image
from common import ROOT, read, write, sha, now


def main():
    result = read(ROOT / 'artifacts/results_v1/results.json')
    assert result['status'] == 'completed' and result['benchmark_executions'] == 264
    report_paths = [ROOT / name for name in ('README.md', 'PROGRESS_ZH.md', 'APPLICATION_NARRATIVE_EN.md', 'REPRODUCTION_ZH.md')]
    links = []
    for path in report_paths:
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            if target.startswith('#') or urlsplit(target).scheme:
                continue
            resolved = (path.parent / unquote(target.split('#')[0])).resolve()
            assert resolved.exists(), (str(path), target)
            links.append({'document': str(path.relative_to(ROOT)), 'target': target, 'exists': True})
    figures = ROOT / 'artifacts/figures/coverage_v3'
    manifest = read(figures / 'manifest.json')
    assert manifest['results_sha256'] == sha(ROOT / 'artifacts/results_v1/results.json')
    for name, digest in manifest['outputs'].items():
        assert sha(figures / name) == digest, name
    svg = ET.parse(figures / 'coverage_results.svg').getroot()
    ns = {'s': 'http://www.w3.org/2000/svg'}
    assert not svg.findall('.//s:image', ns)
    text_count = len(svg.findall('.//s:text', ns))
    assert text_count > 50
    with Image.open(figures / 'coverage_results.png') as im:
        dimensions = list(im.size)
        im.verify()
    receipt = {
        'status': 'passed', 'completed_utc': now(), 'primary_executions': 264,
        'report_links': links,
        'figure': {'directory': str(figures.relative_to(ROOT)), 'png_dimensions': dimensions,
                   'svg_text_nodes': text_count, 'embedded_svg_rasters': 0,
                   'visual_review': 'PNG inspected in Codex after increasing top spacing; title, subtitle, panel headings, labels, counts and legend are legible and do not overlap.',
                   'superseded': 'coverage_v1 and v2 retained. In v1 the subtitle overlapped panel headings. In v2 the colorbar was auto-rasterized; v3 explicitly disables that SVG export default.'},
        'remaining': ['44 new applicant recordings and recording metadata',
                      'new-video contribution experiment', 'public release and application preparation'],
        'code_sha256': sha(ROOT / 'finalize_review.py')
    }
    receipt_path = ROOT / 'receipts/final_review_v1.json'
    assert not receipt_path.exists()
    write(receipt_path, receipt)
    path = ROOT / 'artifacts/primary_completion.json'
    primary = read(path)
    assert primary['status'] == 'reports_completed_visual_review_pending'
    primary.update(status='completed', review_completed_utc=receipt['completed_utc'],
                   review_receipt=str(receipt_path.relative_to(ROOT)),
                   review_receipt_sha256=sha(receipt_path),
                   current_figure=str(figures.relative_to(ROOT)))
    write(path, primary)
    print({'status': 'completed', 'local_links_checked': len(links), 'figure_dimensions': dimensions})


if __name__ == '__main__':
    main()
