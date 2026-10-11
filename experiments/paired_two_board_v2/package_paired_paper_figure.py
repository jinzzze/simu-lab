"""Publish the visually inspected draft to local final figure paths only."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, shutil
from PIL import Image
from pypdf import PdfReader

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p, value): Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

parser = argparse.ArgumentParser(); parser.add_argument('--staging', required=True); args = parser.parse_args()
ROOT = Path(__file__).resolve().parent
out = ROOT / 'artifacts/figures/paired_control_overview_v1'
draft = out / 'tmp/pdfs/draft03'
data = read(draft / 'figure_data_and_sources.json')
for relative, digest in data['source_sha256'].items():
    assert sha(ROOT / relative) == digest, f'Experiment source changed: {relative}'
final = out / 'output'; assert not final.exists(); (final / 'pdf').mkdir(parents=True)
receipts = out / 'receipts'; receipts.mkdir(exist_ok=True)
source = out / 'source'; source.mkdir(exist_ok=True)
for name in ('paired_control_overview.svg', 'paired_control_overview.png'):
    shutil.copy2(draft / name, final / name)
shutil.copy2(draft / 'paired_control_overview.pdf', final / 'pdf/paired_control_overview.pdf')
shutil.copy2(draft / 'figure_data_and_sources.json', out / 'figure_data_and_sources.json')
shutil.copy2(Path(args.staging) / 'FIGURE_CAPTION_EN_ZH.md', out / 'FIGURE_CAPTION_EN_ZH.md')
for name in ('build_paired_paper_figure.py', 'finalize_paired_figure_pdf.py', 'package_paired_paper_figure.py'):
    shutil.copy2(ROOT / name, source / name)
for label in ('draft01', 'draft02', 'draft03'):
    saved = receipts / label; saved.mkdir(exist_ok=True)
    for name in ('pdf_render.png', 'pdf_extracted_text.txt', 'figure_data_and_sources.json', 'build_paired_paper_figure.py'):
        original = out / 'tmp/pdfs' / label / name
        if (saved / name).exists(): assert sha(saved / name) == sha(original)
        else: shutil.copy2(original, saved / name)
pdf = PdfReader(final / 'pdf/paired_control_overview.pdf')
assert len(pdf.pages) == 1 and len(pdf.pages[0].images) == 0, 'Final PDF must remain vector'
page_text = pdf.pages[0].extract_text()
assert 'concat' in page_text and '44/48' in page_text and '26/48' in page_text and '0/36' in page_text
with Image.open(final / 'paired_control_overview.png') as image:
    image.load(); size = image.size; dpi = image.info.get('dpi')
    assert size[0] >= 4200 and size[1] >= 5400 and dpi and min(dpi) >= 599
files = [final / 'pdf/paired_control_overview.pdf', final / 'paired_control_overview.svg',
         final / 'paired_control_overview.png', out / 'FIGURE_CAPTION_EN_ZH.md', out / 'figure_data_and_sources.json']
manifest = {'status': 'completed', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'figure_size_mm': [180, 230], 'png_size_px': size, 'png_dpi': dpi,
            'pdf_pages': 1, 'pdf_raster_images': 0, 'sources_verified': len(data['source_sha256']),
            'visual_review': 'Codex inspected all three actual Poppler PDF renderings. Accepted draft03: all five panels legible; no clipped text; separator annotation and concatenation are explicit; raw counts, all seed dots and every matrix cell match source reports. Heatmaps are vector QuadMesh cells.',
            'reviewer': 'Codex; not applicant human task-outcome adjudication',
            'revision_scope': 'figure authoring only; no new training, rollouts, releases or application submission',
            'outputs_sha256': {str(p.relative_to(out)): sha(p) for p in files},
            'builder_sha256': data['builder_sha256'], 'packager_sha256': sha(__file__)}
write(out / 'manifest.json', manifest)
print(json.dumps({'output': str(final), 'png_size_px': size, 'pdf_vector_images': 0, 'sources_verified': 12}, ensure_ascii=False))
