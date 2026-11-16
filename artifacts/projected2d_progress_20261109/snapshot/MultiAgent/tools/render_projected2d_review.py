#!/usr/bin/env python3
"""Render a fixed, GT-after-freeze review of four original 2D source panels."""
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont

from MultiAgent.projected2d_attention_study import read_csv


def run(output):
    if not (output / 'predictions_frozen.json').is_file():
        raise ValueError('Predictions must be frozen before GT review')
    event_ids = ['scene2_369', 'scene4_room1_34', 'scene5_712', 'scene1_34']
    manifest = json.loads((output / 'input_manifest.json').read_text())
    events = {e['id']: e for e in manifest['events']}
    gt = read_csv(output / 'rebuilt_manifest_all.csv')
    width, height, band = 512, 512, 58
    canvas = Image.new('RGB', (width*2, (height+band)*len(event_ids)), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    descriptions = []
    for index, event_id in enumerate(event_ids):
        event = events[event_id]
        panel = next(p for p in event['panels'] if p['frame_id'] == 'P1')
        with Image.open(panel['source_path']) as img:
            rgb = img.convert('RGB')
            rgb.thumbnail((width, height), Image.Resampling.LANCZOS)
        y0 = index*(height+band)
        canvas.paste(rgb, (0, y0+band))
        canvas.paste(rgb, (width, y0+band))
        draw.text((5,y0+5), f'{event_id} P1 | FULL_RGB model points', font=font, fill='black')
        draw.text((width+5,y0+5), f'{event_id} P1 | projected GT proxy', font=font, fill='black')
        response = json.loads((output / 'responses' / f'{event_id}_FULL_RGB.json').read_text())
        point_records, gt_records = [], []
        for mention in response.get('parsed', {}).get('mentions', []):
            for point in mention['predictions']:
                if point['frame_id'] != 'P1':
                    continue
                u,v = point['point_uv']; x,y = u*width, y0+band+v*height
                draw.ellipse((x-6,y-6,x+6,y+6), outline='red', width=3)
                draw.text((x+7,y), mention['mention_id'], fill='red', font=font, stroke_width=1, stroke_fill='white')
                point_records.append({'mention_id': mention['mention_id'], 'mention': mention['mention'], 'source_uv': [u,v]})
        for row in gt:
            if row['scene'] != event['scene_key'] or int(row['row_index']) != event['row_index'] or row['panel_id'] != 'P1':
                continue
            if row['projection_valid'] != 'True':
                continue
            u,v=float(row['gt_u_norm']),float(row['gt_v_norm'])
            x,y=width+u*width,y0+band+v*height
            draw.line((x-7,y,x+7,y),fill='magenta',width=3)
            draw.line((x,y-7,x,y+7),fill='magenta',width=3)
            draw.text((x+7,y),row['referent_name'],fill='magenta',font=font,stroke_width=1,stroke_fill='white')
            gt_records.append({'referent_name': row['referent_name'], 'uv': [u,v], 'evidence_acceptable': row['evidence_acceptable']})
        descriptions.append({'event': event_id, 'instruction': event['instruction'], 'panel': 'P1',
                             'condition': 'FULL_RGB', 'prediction_points': point_records, 'projected_gt': gt_records})
    target=output/'fixed_2d_points_review.jpg'
    canvas.save(target,quality=94)
    (output/'fixed_2d_points_review.json').write_text(json.dumps(descriptions,ensure_ascii=False,indent=2)+'\n')
    print(target)


if __name__ == '__main__':
    run(Path(sys.argv[1]))
