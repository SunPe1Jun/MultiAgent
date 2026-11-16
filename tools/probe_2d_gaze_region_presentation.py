#!/usr/bin/env python3
"""Test gaze-derived region corner marks on single full-frame 2D RGB inputs."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

from MultiAgent.projected2d_attention_study import CSV_COLUMNS, evaluator_adapter, read_csv, save_json, sha_file, write_csv
from MultiAgent.projected2d_exact_set import exact_counts
from MultiAgent.projected2d_main_image import normalize, verify_lock
from MultiAgent.projected2d_routed_rgb import MODEL, ROOT
from tools.probe_referent_temporal_prompt_v3 import prompt_for as temporal_prompt

MAIN = ROOT / '.local-sync/projected2d_main_rgb32_20261109'
ROUTED = ROOT / '.local-sync/projected2d_routed_rgb_20261109'
PROMPT = ROOT / '.local-sync/referent_temporal_prompt_v3_20261109'
SUBSET = ROOT / '.local-sync/referent_frame_choice_audit_20261109/rebuilt_manifest_all.csv'
OUTPUT = ROOT / '.local-sync/2d_gaze_region_presentation_20261109'
CONDITIONS = ('UNMARKED', 'GAZE_REGION', 'MIRRORED_REGION')


def draw_corners(image, box):
    """Draw corners inside a validated half-open box; preserve interior pixels."""
    image = image.convert('RGB').copy()
    left, top, right, bottom = box
    if not (0 <= left < right <= image.width and 0 <= top < bottom <= image.height):
        raise ValueError('Invalid image region')
    right -= 1
    bottom -= 1
    length = max(5, min(right - left, bottom - top) // 9)
    segments = [((left, top), (left + length, top)), ((left, top), (left, top + length)),
                ((right, top), (right - length, top)), ((right, top), (right, top + length)),
                ((left, bottom), (left + length, bottom)), ((left, bottom), (left, bottom - length)),
                ((right, bottom), (right - length, bottom)), ((right, bottom), (right, bottom - length))]
    draw = ImageDraw.Draw(image)
    for a, b in segments:
        draw.line((a, b), fill=(0, 0, 0), width=7)
        draw.line((a, b), fill=(45, 230, 110), width=3)
    return image


def mapped_box(box, size, source_size):
    width, height = size
    sw, sh = source_size
    x0, y0, x1, y1 = box
    return [max(0, min(width, round(x0 * width / sw))),
            max(0, min(height, round(y0 * height / sh))),
            max(0, min(width, round(x1 * width / sw))),
            max(0, min(height, round(y1 * height / sh)))]


def make_prompt(event):
    original = temporal_prompt(event)
    needle = 'For every mention_id return exactly one row.'
    if original.count(needle) != 1:
        raise ValueError('Temporal prompt contract changed')
    region = '''Some images may contain four green corner marks surrounding an area. The marks indicate
a rough attention region for that individual frame, derived from gaze shortly before it. The area is
not an object boundary, a target label, or a guarantee that gaze identifies the referent. Images may
have no marks when the signal is unavailable. For each mention compare which frame's indicated area,
if any, is consistent with the visible referent and the instruction, along with visual identity,
relations and the soft phase. Different mentions may correspond to different regions and frames.
The object may be partly outside the marks. Inspect the entire RGB image and do not simply point at
the marks or a region center. If gaze is missing, noisy or inconsistent, use the other evidence.

'''
    return original.replace(needle, region + needle)


def prepare(out):
    if (out / 'input_manifest.json').exists():
        raise ValueError('Refuse to overwrite frozen run')
    verify_lock(MAIN)
    verify_lock(PROMPT)
    source = json.loads((PROMPT / 'input_manifest.json').read_text())
    routed_path = ROUTED / 'input_manifest.json'
    route_lock = json.loads((ROUTED / 'protocol_lock.json').read_text())
    if sha_file(routed_path) != route_lock['input_manifest_sha256']:
        raise ValueError('Frozen routed input manifest changed')
    routed = {e['id']: e for e in json.loads(routed_path.read_text())['events']}
    events, audits = [], []
    for original in source['events']:
        event = json.loads(json.dumps(original))
        prev = {p['frame_id']: p for p in routed[event['id']]['panels']}
        for panel in event['panels']:
            prior = prev[panel['frame_id']]
            if panel['source_sha256'] != prior['source_sha256']:
                raise ValueError('Region/source frame mismatch')
            rgb = panel['inputs']['MAIN_RGB_768']
            if sha_file(Path(rgb['path'])) != rgb['sha256']:
                raise ValueError('Frozen RGB input changed')
            with Image.open(rgb['path']) as opened:
                image = opened.convert('RGB')
            supported = bool(prior['clock_supported']) and prior['attention_duration_by_phase'][-1] >= .10
            controls = {'UNMARKED': rgb}
            for condition, old_condition in [('GAZE_REGION', 'GAZE_RGB'), ('MIRRORED_REGION', 'MIRRORED_GAZE_RGB')]:
                box = mapped_box(prior['boxes'][old_condition][-1], image.size, prior['display_size']) if supported else None
                view = draw_corners(image, box) if box else image.copy()
                dest = out / 'inputs' / event['id'] / panel['frame_id'] / (condition + '.png')
                dest.parent.mkdir(parents=True, exist_ok=True)
                view.save(dest)
                original_pixels = np.asarray(image)
                pixels = np.asarray(view)
                changed = np.any(original_pixels != pixels, axis=2)
                if box:
                    x0, y0, x1, y1 = box
                    inset = max(8, min(x1 - x0, y1 - y0) // 8)
                    if np.any(changed[y0 + inset:y1 - inset, x0 + inset:x1 - inset]):
                        raise ValueError('Region interior changed')
                controls[condition] = {'path': str(dest), 'sha256': sha_file(dest), 'size': list(image.size)}
                audits.append({'event': event['id'], 'frame_id': panel['frame_id'], 'condition': condition,
                               'region_xyxy': box, 'clock_supported': bool(prior['clock_supported']),
                               'newest_support_seconds': prior['attention_duration_by_phase'][-1],
                               'changed_pixels': int(changed.sum()), 'image_pixels': int(changed.size),
                               'region_interior_unchanged': True})
            panel['region_inputs'] = controls
        events.append(event)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / 'input_manifest.json', {'version': '2d_gaze_region_presentation_v1', 'events': events,
        'conditions': list(CONDITIONS), 'sample_status': '10 previously seen development interactions; not held out',
        'source_main_manifest_sha256': sha_file(MAIN / 'input_manifest.json'),
        'source_route_manifest_sha256': sha_file(routed_path),
        'policy': 'Newest existing past-gaze routing window when clock and >=0.1s support exist; corner marks on whole RGB. Fixed 40% window is display allocation, not calibrated fixation uncertainty or target extent.',
        'controls': 'Identical instruction and format for unmarked, real gaze and mirrored gaze; mirrored is a placebo presented to model with the same neutral cue explanation.',
        'ground_truth_in_prompt': False, 'selection_after_prior_results': True})
    save_json(out / 'input_image_audit.json', {'images': audits,
        'no_target_labels_used': True, 'no_heatmap_pixels_sent': True,
        'note': 'Corners can occlude boundary pixels. Interior preserved. Box support is not referent identity.'})
    deps = [Path(__file__).resolve(), out / 'input_manifest.json', out / 'input_image_audit.json',
            MAIN / 'input_manifest.json', ROUTED / 'input_manifest.json', PROMPT / 'input_manifest.json',
            ROOT / 'tools/probe_referent_temporal_prompt_v3.py', ROOT / 'MultiAgent/qwen_backend.py',
            ROOT / 'MultiAgent/projected2d_main_image.py', ROOT / 'tools/run_full_frame_grounding_v4.py',
            SUBSET, ROOT.parent / 'TriModal-Referring/exam2/evaluate_2d_point_grounding.py', MODEL / 'config.json']
    save_json(out / 'protocol_lock.json', {'files': {str(p): sha_file(p) for p in deps}})
    print('PREPARED', len(events), 'events', len(audits), 'marked images', flush=True)


def infer(out):
    from MultiAgent.qwen_backend import QwenBackend
    from tools.run_full_frame_grounding_v4 import clean_visual_trace
    verify_lock(out)
    manifest = json.loads((out / 'input_manifest.json').read_text())
    backend = QwenBackend(out / 'model_cache', str(MODEL), local_only=True,
                          namespace='2d_gaze_region_presentation_v1', trace_visual_inputs=True)
    for index, event in enumerate(manifest['events'], 1):
        prompt = make_prompt(event)
        captions = [f"Frame {p['frame_id']} | video_time={p['requested_video_time']:.3f} seconds" for p in event['panels']]
        for condition in CONDITIONS:
            path = out / 'responses' / f"{event['id']}_{condition}.json"
            images = [p['region_inputs'][condition] for p in event['panels']]
            fingerprint = hashlib.sha256(json.dumps([prompt, captions, [x['sha256'] for x in images]]).encode()).hexdigest()
            if path.exists():
                if json.loads(path.read_text())['fingerprint'] != fingerprint:
                    raise ValueError('Response fingerprint mismatch')
                continue
            for image in images:
                if sha_file(Path(image['path'])) != image['sha256']:
                    raise ValueError('Frozen image changed')
            req = None
            try:
                raw = backend.ask(event['id'], condition, prompt, [i['path'] for i in images],
                                  max_new_tokens=2304, image_captions=captions)
                req = Path(backend.requests[-1])
                record = json.loads(req.read_text())
                parsed, errors, conversions = normalize(raw, event)
                if raw.get('_parse_warning'):
                    errors.append(raw['_parse_warning'])
                response = {'status': 'COMPLETE' if not errors else 'PARSE_WITH_WARNINGS', 'parsed': parsed,
                            'warnings': errors, 'conversions': conversions, 'raw_response': record['response'],
                            'request_path': str(req), 'vision_verified': bool((record.get('visual_tensor_trace') or {}).get('input_consumption_verified'))}
                if not response['vision_verified']:
                    raise ValueError('Vision input verification failed')
            except Exception as exc:
                response = {'status': 'FAILED', 'parsed': {'mentions': []}, 'error': f'{type(exc).__name__}: {exc}'}
            finally:
                if req:
                    clean_visual_trace(str(req))
            response.update(event=event['id'], condition=condition, fingerprint=fingerprint, prompt=prompt,
                            input_images=images, image_captions=captions)
            save_json(path, response)
            print('REGION', index, '/', len(manifest['events']), event['id'], condition, response['status'], flush=True)


def report(out):
    verify_lock(out)
    manifest = json.loads((out / 'input_manifest.json').read_text())
    paths = [out / 'responses' / f"{event['id']}_{condition}.json" for event in manifest['events'] for condition in CONDITIONS]
    if any(not p.exists() for p in paths):
        raise ValueError('Missing condition results')
    save_json(out / 'predictions_frozen.json', {'responses': {str(p): sha_file(p) for p in paths}})
    result = {}
    for condition in CONDITIONS:
        rows, counts, frames = [], Counter(), Counter()
        for event in manifest['events']:
            r = json.loads((out / 'responses' / f"{event['id']}_{condition}.json").read_text())
            counts[r['status']] += 1
            counts['vision_verified'] += r.get('vision_verified', False)
            pred = evaluator_adapter(r['parsed'])
            for row in r['parsed']['mentions']:
                for point in row['predictions']:
                    frames[point['frame_id']] += 1
            rows.append({'scene': event['scene_key'], 'row_index': event['row_index'], 'event_id': event['event_id'],
                         'instruction': event['instruction'], 'model_input_image': ';'.join(x['path'] for x in r['input_images']),
                         'model_raw_output': r.get('raw_response', ''), 'parsed_json': json.dumps(pred, ensure_ascii=False),
                         'parse_ok': str(r['status'] in ('COMPLETE', 'PARSE_WITH_WARNINGS')),
                         'prediction_count': len(pred['referents']), 'error_message': r.get('error', '')})
        pred_path = out / 'predictions' / (condition + '.csv')
        write_csv(pred_path, rows, CSV_COLUMNS)
        eval_dir = out / 'evaluation' / condition
        subprocess.run([sys.executable, str(ROOT.parent / 'TriModal-Referring/exam2/evaluate_2d_point_grounding.py'),
                        '--manifest', str(SUBSET), '--pred_csv', str(pred_path), '--output_dir', str(eval_dir),
                        '--coordinate_mode', 'panel'], check=True, capture_output=True, text=True)
        result[condition] = {'counts': dict(counts), 'frames': dict(frames),
                             'metrics': json.loads((eval_dir / '2d_eval_summary.json').read_text()),
                             'exact': exact_counts(read_csv(eval_dir / '2d_eval_detail.csv'))}
    save_json(out / 'summary.json', {'conditions': result,
        'caveats': ['Sample is previously scored and adaptively selected; not held out.',
                    'Marker dimensions are a routing display choice, not calibrated gaze uncertainty.',
                    'Control can change apparent attention; a lower P1 rate is not proof of better grounding.']})
    print('REPORTED', {k: {'frames': v['frames'], 'joint100': v['metrics']['joint_100_f1']} for k, v in result.items()}, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'infer', 'report'])
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    {'prepare': prepare, 'infer': infer, 'report': report}[args.stage](args.output.resolve())


if __name__ == '__main__':
    main()
