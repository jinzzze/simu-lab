"""Create an A4 calibration board and verify its marker raster locally."""
from pathlib import Path
import json
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'assets'/'calibration'
PPMM=8
MARKER_MM=30
BOARD_ID="pilot-a4-diagonal-2-v1"
POSITIONS={0:(15,25),2:(165,220)}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    canvas=np.full((297*PPMM,210*PPMM),255,np.uint8)
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="210mm" height="297mm" shape-rendering="crispEdges" viewBox="0 0 210 297">',
           '<rect width="210" height="297" fill="white"/>',
           '<g font-family="Arial,sans-serif" fill="black">',
           '<text x="15" y="12" font-size="5">PILOT BOARD - 2 MARKERS</text>',
           '<text x="15" y="19" font-size="3">A4 portrait | print 100% | no fit-to-page | check 50 mm ruler</text></g>',
           '<rect x="55" y="70" width="100" height="130" fill="none" stroke="#aaa" stroke-width="0.2"/>']
    for x in range(65,155,10):
        parts.append(f'<path d="M{x} 70V200" stroke="#ddd" stroke-width="0.15"/>')
    for y in range(80,200,10):
        parts.append(f'<path d="M55 {y}H155" stroke="#ddd" stroke-width="0.15"/>')
    markers=[]
    for marker_id,(x,y) in POSITIONS.items():
        marker=cv2.aruco.generateImageMarker(dictionary,marker_id,MARKER_MM*PPMM)
        canvas[y*PPMM:(y+MARKER_MM)*PPMM,x*PPMM:(x+MARKER_MM)*PPMM]=marker
        cells=cv2.aruco.generateImageMarker(dictionary,marker_id,6)
        for row in range(6):
            for col in range(6):
                if cells[row,col] == 0:
                    parts.append(f'<rect x="{x+col*5}" y="{y+row*5}" width="5" height="5" fill="black"/>')
        parts.append(f'<text x="{x}" y="{y+35}" font-family="Arial" font-size="3">ID {marker_id} | 30 mm</text>')
        markers.append({'id':marker_id,'corners_mm':[[x,y],[x+30,y],[x+30,y+30],[x,y+30]]})
    parts += ['<path d="M15 270H65 M15 267V273 M65 267V273" stroke="black" stroke-width="0.3"/>',
              '<text x="15" y="279" font-family="Arial" font-size="4">This line must measure 50 mm.</text>',
              '<text x="15" y="287" font-family="Arial" font-size="3">Keep flat. Markers calibrate the table plane, not elevated objects.</text>', '</svg>']
    svg='\n'.join(parts)
    (OUT/'pilot_board_a4.svg').write_text(svg,encoding='utf-8')
    html='<!doctype html><meta charset="utf-8"><title>Pilot A4 calibration board</title><style>@page{size:A4 portrait;margin:0}html,body{margin:0;padding:0;background:white}svg{display:block;width:210mm;height:297mm}</style>'+svg
    (OUT/'print_board.html').write_text(html,encoding='utf-8')
    cv2.imwrite(str(OUT/'marker_layout_check.png'),canvas)
    corners,ids,_=cv2.aruco.ArucoDetector(dictionary).detectMarkers(canvas)
    found=[] if ids is None else sorted(int(i) for i in ids.ravel())
    if found != sorted(POSITIONS):
        raise RuntimeError(f'Marker verification failed: {found}')
    errors=[]
    for c,marker_id in zip(corners,ids.ravel()):
        x,y=POSITIONS[int(marker_id)]
        expected=np.array([[x,y],[x+30,y],[x+30,y+30],[x,y+30]],dtype=float)
        errors.append(float(np.linalg.norm(c.reshape(4,2)/PPMM-expected,axis=1).max()))
    if max(errors)>0.3:
        raise RuntimeError(f'Marker coordinate check failed: {max(errors)} mm')
    meta={'board_id':BOARD_ID,'expected_marker_ids':sorted(POSITIONS),'dictionary':'DICT_4X4_50','page_mm':[210,297],'marker_side_mm':30,
          'origin':'page top-left','axes':'x right, y down; z=0 is the printed board plane',
          'corner_order':'top-left, top-right, bottom-right, bottom-left',
          'preview_pixels_per_mm':PPMM,'ruler_mm':50,'markers':markers,
          'verification':{'detected_ids':found,'max_corner_error_mm':max(errors),
                          'scope':'synthetic raster only; printed scale and phone footage not yet validated'}}
    (OUT/'board_geometry.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta['verification']))

if __name__=='__main__':
    main()
