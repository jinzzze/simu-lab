"""Finalize embedded fonts and verify the inspected figure without changing content."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, shutil
from PIL import Image, ImageChops
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
out = ROOT / 'artifacts/figures/paired_control_overview_v1'
clean = out / 'tmp/pdfs/font_clean'
receipt = out / 'receipts/final_font_check'
assert not receipt.exists()
pdf = PdfReader(clean / 'paired_control_overview.pdf')
assert len(pdf.pages) == 1 and len(pdf.pages[0].images) == 0
fonts = []
for key, ref in pdf.pages[0]['/Resources']['/Font'].items():
    font = ref.get_object()
    for face in font.get('/DescendantFonts', [font]):
        face = face.get_object(); descriptor = face['/FontDescriptor'].get_object()
        embedded = any(k in descriptor for k in ('/FontFile', '/FontFile2', '/FontFile3'))
        assert embedded
        fonts.append({'resource': str(key), 'name': str(face['/BaseFont']), 'embedded': embedded})
with Image.open(clean / 'pdf_render.png') as a, Image.open(out / 'tmp/pdfs/draft03/pdf_render.png') as b:
    assert a.size == b.size and ImageChops.difference(a.convert('RGB'), b.convert('RGB')).getbbox() is None
receipt.mkdir()
shutil.copy2(out / 'manifest.json', receipt / 'before_manifest.json')
shutil.copy2(out / 'output/pdf/paired_control_overview.pdf', receipt / 'before_font_cleanup.pdf')
shutil.copy2(out / 'source/finalize_paired_figure_pdf.py', receipt / 'before_finalizer.py')
for name in ('pdf_render.png', 'pdf_extracted_text.txt'):
    shutil.copy2(clean / name, receipt / name)
shutil.copy2(clean / 'paired_control_overview.pdf', out / 'output/pdf/paired_control_overview.pdf')
for name in ('finalize_paired_figure_pdf.py', 'verify_paired_paper_figure.py'):
    shutil.copy2(ROOT / name, out / 'source' / name)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
manifest['outputs_sha256']['output\\pdf\\paired_control_overview.pdf'] = sha(out / 'output/pdf/paired_control_overview.pdf')
manifest.update(pdf_fonts=fonts, all_pdf_fonts_embedded=True, final_pdf_render_identical_to_inspected_draft=True,
                finalized_utc=datetime.now(timezone.utc).isoformat(), font_verifier_sha256=sha(__file__))
(out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
(receipt / 'verification.json').write_text(json.dumps({'fonts': fonts, 'all_embedded': True,
    'pdf_images': 0, 'render_pixel_identical': True, 'final_pdf_sha256': sha(out / 'output/pdf/paired_control_overview.pdf')}, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'all_fonts_embedded': True, 'pdf_images': 0, 'render_pixel_identical': True, 'fonts': fonts}))
