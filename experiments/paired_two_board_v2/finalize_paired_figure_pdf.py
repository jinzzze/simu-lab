"""Create a one-page ReportLab PDF and preserve the Matplotlib vector artwork."""
from pathlib import Path
from io import BytesIO
import argparse, json
from reportlab.pdfgen import canvas
from pypdf import PdfReader, PdfWriter
from pypdf.generic import NameObject, DictionaryObject, DecodedStreamObject

parser = argparse.ArgumentParser(); parser.add_argument('draft'); args = parser.parse_args()
root = Path(args.draft)
native = PdfReader(root / 'figure_native.pdf')
assert len(native.pages) == 1
page = native.pages[0]; w, h = float(page.mediabox.width), float(page.mediabox.height)
buffer = BytesIO(); pdf = canvas.Canvas(buffer, pagesize=(w, h), pageCompression=1)
pdf.setTitle('Goal-conditioned pick-and-place: task, strategy and evaluation')
pdf.setAuthor('simu-lab'); pdf.showPage(); pdf.save(); buffer.seek(0)
template = PdfReader(buffer).pages[0]
# The ReportLab template is blank; discard its unused default Helvetica resource.
template[NameObject('/Resources')] = DictionaryObject()
template[NameObject('/Contents')] = DecodedStreamObject()
writer = PdfWriter(); target = writer.add_page(template); target.merge_page(page)
writer.add_metadata({'/Title': 'Goal-conditioned pick-and-place: task, strategy and evaluation',
                     '/Author': 'simu-lab', '/Subject': 'Frozen v6 simulation study; data-derived vector figure'})
path = root / 'paired_control_overview.pdf'
with path.open('wb') as f: writer.write(f)
check = PdfReader(path)
assert len(check.pages) == 1
text = check.pages[0].extract_text()
for expected in ('44/48', '26/48', '0/36', '2,500', '130 mm gap', '16 personally recorded'):
    assert expected in text, f'Missing expected figure text: {expected}'
assert abs(float(check.pages[0].mediabox.width) - 180 / 25.4 * 72) < .01
assert abs(float(check.pages[0].mediabox.height) - 230 / 25.4 * 72) < .01
(root / 'pdf_extracted_text.txt').write_text(text, encoding='utf-8')
print(json.dumps({'pdf': str(path), 'pages': 1, 'text_checks_passed': True, 'vector_artwork_preserved': True}))
