#!/usr/bin/env python3
"""Check saved composite-to-source conversions without reference to GT."""
from collections import Counter
import json
import math
from pathlib import Path
import sys

from MultiAgent.gaze_rgb_routing import HEADER
from MultiAgent.projected2d_attention_study import save_json
from MultiAgent.projected2d_routed_rgb import GROUPS


def run(output):
    manifest = json.loads((output / 'input_manifest.json').read_text())
    counts, failures = Counter(), []
    for event in manifest['events']:
        panels = {p['frame_id']: p for p in event['panels']}
        for group in GROUPS:
            path = output / 'responses' / f"{event['id']}_{group}.json"
            if not path.is_file():
                failures.append({'event': event['id'], 'group': group, 'reason': 'missing_response'})
                continue
            result = json.loads(path.read_text())
            counts['responses'] += 1
            counts['vision_verified'] += bool(result.get('vision_verified'))
            counts['identical_input_reuse'] += bool(result.get('reused_identical_from'))
            counts[result['status']] += 1
            for conversion in result.get('coordinate_conversions', []):
                panel = panels[conversion['frame_id']]
                w, h = panel['display_size']
                u, v = conversion['source_uv']
                area = conversion['source_area_index']
                source_x, source_y = u*w, v*h
                row, col = divmod(area, 2)
                if area:
                    x0, y0, x1, y1 = panel['boxes'][group][area-1]
                    bw, bh = x1-x0, y1-y0
                    scale = min(w/bw, h/bh)
                    dw, dh = max(1, round(bw*scale)), max(1, round(bh*scale))
                    x = (source_x-x0)*dw/bw + (w-dw)//2
                    y = (source_y-y0)*dh/bh + (h-dh)//2
                else:
                    x, y = source_x, source_y
                composite = [(col*w+x)/(2*w)*1000,
                             (row*(h+HEADER)+HEADER+y)/(2*(h+HEADER))*1000]
                error = math.hypot(*(a-b for a,b in zip(composite, conversion['composite_xy'])))
                counts['conversions'] += 1
                counts['area_'+str(area)] += 1
                if error > 1e-7 or not (0 <= u <= 1 and 0 <= v <= 1):
                    failures.append({'event': event['id'], 'group': group,
                        'mention_id': conversion['mention_id'], 'round_trip_error': error})
    result = {'counts': dict(counts), 'failures': failures, 'all_roundtrips_valid': not failures,
              'scope': 'Arithmetic reversibility and valid source coordinates only; no GT or semantic validation.'}
    save_json(output / 'coordinate_audit.json', result)
    print(json.dumps(result, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    run(Path(sys.argv[1]))
