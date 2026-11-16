"""Full RGB images with direct coordinates; preserve frozen mention and time tasks."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

from .projected2d_attention_study import CSV_COLUMNS, evaluator_adapter, read_csv, save_json, sha_file, write_csv
from .projected2d_exact_set import collapse_explicit_coreference, exact_counts
from .projected2d_routed_rgb import MODEL, ROOT

CONDITIONS = ('MAIN_RGB_512', 'MAIN_RGB_768')


def prompt_for(event):
    frames = [{'frame_id': p['frame_id'], 'video_time_seconds': p['requested_video_time']}
              for p in event['panels']]
    tasks = [{**m, 'soft_time_hint': event['temporal_hints'].get(m['mention_id'])}
             for m in event['mentions']]
    return '''Solve 2D temporal localization for each mention in the full instruction.
Each supplied image is one COMPLETE RGB video frame. Images are chronological; the text immediately
before each image gives its actual frame_id and video time. Use visible appearance, action and
the instruction to resolve each mention. Time hints are soft instruction phases, not speech
timestamps; all frames remain eligible. Compare all supplied frames before selecting the best
one for each target; do not choose the first frame by default.

For every mention_id return exactly one entry, with status localized, ambiguous or not_visible.
For every localized physical target choose a supplied frame_id and point ON the visible object.
Return point_xy=[x,y] on a 0..1000 scale relative to that ENTIRE RGB image, origin top-left,
x right and y down. For multiple targets return one point per target; if uncertain give no point.
For a component locate the indicated whole parent. Use all frames to check action and identity.
Return JSON only with this structure, with actual numeric coordinates replacing x and y:
{"mentions":[{"mention_id":"...","status":"localized|ambiguous|not_visible",
"predictions":[{"frame_id":"one_of_the_supplied_frame_ids","point_xy":[x,y]}],"evidence":"..."}]}.
''' + json.dumps({'instruction': event['instruction'], 'source_frames': frames,
                   'localization_tasks': tasks}, ensure_ascii=False)


def normalize(raw, event):
    allowed = {m['mention_id']: m for m in event['mentions']}
    panels = {p['frame_id'] for p in event['panels']}
    rows = raw.get('mentions') if isinstance(raw, dict) else None
    if not isinstance(rows, list):
        rows = []
        errors = ['response_missing_mentions_array']
    else:
        errors = []
    counts = Counter(r.get('mention_id') for r in rows if isinstance(r, dict))
    output, conversions = [], []
    for row in rows:
        if not isinstance(row, dict) or row.get('mention_id') not in allowed:
            errors.append('unknown_or_malformed_mention')
            continue
        mid = row['mention_id']
        if counts[mid] != 1:
            errors.append(f'duplicate_mention:{mid}')
            continue
        status = row.get('status')
        if status not in ('localized', 'ambiguous', 'not_visible'):
            errors.append(f'invalid_status:{mid}')
            status = 'ambiguous'
        observations = row.get('predictions', [])
        if not isinstance(observations, list):
            errors.append(f'invalid_predictions:{mid}')
            observations = []
        predictions = []
        for point in observations:
            try:
                if status != 'localized':
                    raise ValueError('non-localized response included a point')
                frame_id, xy = point['frame_id'], point['point_xy']
                if frame_id not in panels:
                    raise ValueError('frame was not supplied')
                if (not isinstance(xy, (list, tuple)) or len(xy) != 2 or
                    any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                        not math.isfinite(v) or not 0 <= v <= 1000 for v in xy)):
                    raise ValueError('point_xy must contain two finite numbers in [0,1000]')
                uv = [v / 1000 for v in xy]
                predictions.append({'frame_id': frame_id, 'point_uv': uv})
                conversions.append({'mention_id': mid, 'frame_id': frame_id,
                                    'point_xy': xy, 'source_uv': uv})
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f'invalid_point:{mid}:{exc}')
        output.append({'mention_id': mid, 'mention': allowed[mid]['text'], 'status': status,
                       'predictions': predictions, 'evidence': str(row.get('evidence') or '')})
    for mid in sorted(set(allowed) - {r['mention_id'] for r in output}):
        errors.append(f'missing_mention:{mid}')
        output.append({'mention_id': mid, 'mention': allowed[mid]['text'], 'status': 'ambiguous',
                       'predictions': [], 'evidence': 'Missing or duplicate model row.'})
    return {'mentions': output}, errors, conversions


def verify_lock(output):
    lock = json.loads((output / 'protocol_lock.json').read_text())
    for path, digest in lock['files'].items():
        if sha_file(Path(path)) != digest:
            raise ValueError('Frozen study dependency changed: ' + path)
    return lock


def prepare(source, output):
    from PIL import Image
    if (output / 'input_manifest.json').exists():
        raise ValueError('Refuse to overwrite frozen main-image study')
    old = json.loads((source / 'input_manifest.json').read_text())
    old_lock = json.loads((source / 'protocol_lock.json').read_text())
    if sha_file(source / 'input_manifest.json') != old_lock['input_manifest_sha256']:
        raise ValueError('Frozen source manifest changed')
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / 'rebuilt_manifest_all.csv', output / 'rebuilt_manifest_all.csv')
    events, verified = [], []
    for original in old['events']:
        event = {k: original[k] for k in ('id', 'scene_key', 'row_index', 'event_id', 'instruction',
                                         'mentions', 'parser_valid', 'temporal_hints')}
        event['panels'] = []
        for panel in original['panels']:
            source_path = Path(panel['source_path'])
            if sha_file(source_path) != panel['source_sha256']:
                raise ValueError('Source RGB changed')
            inputs = {}
            for condition in CONDITIONS:
                edge = int(condition.rsplit('_', 1)[1])
                with Image.open(source_path) as opened:
                    rgb = opened.convert('RGB')
                    rgb.thumbnail((edge, edge), Image.Resampling.LANCZOS)
                target = output / 'inputs' / original['id'] / panel['frame_id'] / f'{condition}.png'
                target.parent.mkdir(parents=True, exist_ok=True)
                rgb.save(target)
                inputs[condition] = {'path': str(target), 'sha256': sha_file(target), 'size': list(rgb.size)}
                verified.append({'event': event['id'], 'frame': panel['frame_id'], 'condition': condition,
                                 'full_field': True, 'size': list(rgb.size)})
            event['panels'].append({k: panel[k] for k in ('frame_id', 'source_path', 'source_sha256',
                'source_size', 'requested_video_time', 'clock_supported')} | {'inputs': inputs})
        events.append(event)
    manifest = {'version': 'projected2d_main_rgb_v1', 'conditions': list(CONDITIONS), 'events': events,
        'model': str(MODEL), 'max_new_tokens': 1536,
        'source_input_manifest': str(source / 'input_manifest.json'),
        'source_input_manifest_sha256': sha_file(source / 'input_manifest.json'),
        'sample_status': old['sample_status'],
        'coordinate_contract': 'point_xy 0..1000 directly on whole RGB frame; source_uv=point_xy/1000',
        'task_policy': 'Preserve source mention definitions, parser failures, soft temporal hints and whole-parent component target.',
        'image_policy': 'One whole RGB image per chronological source frame. No collage, crop, added heatmap, labels or emphasis pixels. Frame IDs appear as adjacent text.',
        'condition_interpretation': '512 versus 768 full-image resolution; neither condition uses scripted gaze. Separate development controls, not original expected_count baseline.',
        'attention_followup': 'After the main-image control, test attention within one RGB coordinate plane with matched control; not implemented as efficacy evidence here.'}
    save_json(output / 'input_manifest.json', manifest)
    dependencies = [Path(__file__), ROOT / 'MultiAgent/qwen_backend.py',
        ROOT / 'MultiAgent/projected2d_attention_study.py', ROOT / 'MultiAgent/projected2d_exact_set.py',
        ROOT / 'tools/run_full_frame_grounding_v4.py',
        ROOT.parent / 'TriModal-Referring/exam2/evaluate_2d_point_grounding.py',
        output / 'input_manifest.json', output / 'rebuilt_manifest_all.csv', MODEL / 'config.json']
    save_json(output / 'protocol_lock.json', {'files': {str(p): sha_file(p) for p in dependencies}})
    save_json(output / 'input_audit.json', {'images': verified, 'gt_used_to_create_images': False,
        'annotation_count_sent_to_model': False, 'mentions_and_temporal_hints_unchanged': True})
    print('PREPARED_MAIN_RGB', len(events), 'events', len(verified), 'images', flush=True)


def infer(output):
    from .qwen_backend import QwenBackend
    from tools.run_full_frame_grounding_v4 import clean_visual_trace
    verify_lock(output)
    manifest = json.loads((output / 'input_manifest.json').read_text())
    backend = QwenBackend(output / 'model_cache', str(MODEL), local_only=True,
                          namespace='projected2d_main_rgb_v1', trace_visual_inputs=True)
    for index, event in enumerate(manifest['events'], 1):
        prompt = prompt_for(event)
        captions = [f"Frame {p['frame_id']} | video_time={p['requested_video_time']:.3f} seconds"
                    for p in event['panels']]
        for condition in CONDITIONS:
            target = output / 'responses' / f"{event['id']}_{condition}.json"
            images = [p['inputs'][condition] for p in event['panels']]
            fingerprint = hashlib.sha256(json.dumps([prompt, captions, [p['sha256'] for p in images]]).encode()).hexdigest()
            if target.exists():
                if json.loads(target.read_text())['fingerprint'] != fingerprint:
                    raise ValueError('Saved response belongs to different inputs')
                continue
            for image in images:
                if sha_file(Path(image['path'])) != image['sha256']:
                    raise ValueError('Input RGB changed')
            if not event['parser_valid'] or not event['mentions'] or not images:
                result = {'status': 'NOT_RUN', 'parsed': {'mentions': []},
                          'error': 'Source parser failure or missing original image'}
            else:
                request_path = None
                try:
                    raw = backend.ask(event['id'], condition, prompt, [p['path'] for p in images],
                                      max_new_tokens=manifest['max_new_tokens'], image_captions=captions)
                    request_path = Path(backend.requests[-1])
                    request = json.loads(request_path.read_text())
                    parsed, errors, conversions = normalize(raw, event)
                    if raw.get('_parse_warning'):
                        errors.append(str(raw['_parse_warning']))
                    result = {'status': 'COMPLETE' if not errors else 'PARSE_WITH_WARNINGS',
                        'parsed': parsed, 'parse_warning': errors, 'coordinate_conversions': conversions,
                        'raw_response': request['response'], 'request_path': str(request_path),
                        'processor_audit': request.get('processor_audit'),
                        'vision_verified': bool((request.get('visual_tensor_trace') or {}).get('input_consumption_verified'))}
                    if not result['vision_verified']:
                        raise ValueError('Image consumption was not verified')
                except Exception as exc:
                    result = {'status': 'FAILED', 'parsed': {'mentions': []},
                              'error': f'{type(exc).__name__}: {exc}'}
                    if backend.requests:
                        request_path = Path(backend.requests[-1])
                finally:
                    if request_path is not None:
                        clean_visual_trace(str(request_path))
            result.update(event_id=event['event_id'], scene=event['scene_key'], row_index=event['row_index'],
                          condition=condition, fingerprint=fingerprint, prompt=prompt,
                          input_images=images, image_captions=captions)
            save_json(target, result)
            print('MAIN_RGB', index, '/', len(manifest['events']), event['id'], condition, result['status'], flush=True)


def evaluate(output):
    verify_lock(output)
    manifest = json.loads((output / 'input_manifest.json').read_text())
    paths = [output / 'responses' / f"{e['id']}_{c}.json" for e in manifest['events'] for c in CONDITIONS]
    if any(not p.is_file() for p in paths):
        raise ValueError('Finish both full-image controls before scoring')
    freeze = output / 'predictions_frozen.json'
    hashes = {str(p): sha_file(p) for p in paths}
    if freeze.exists() and json.loads(freeze.read_text())['responses'] != hashes:
        raise ValueError('Frozen prediction changed')
    save_json(freeze, {'responses': hashes})
    summaries, exact, statuses, frames, merges = {}, {}, {}, {}, {}
    evaluator = ROOT.parent / 'TriModal-Referring/exam2/evaluate_2d_point_grounding.py'
    for condition in CONDITIONS:
        predictions, collapsed = [], []
        status, frame_counts, merged = Counter(), Counter(), []
        for event in manifest['events']:
            response = json.loads((output / 'responses' / f"{event['id']}_{condition}.json").read_text())
            status[response['status']] += 1
            parsed = response.get('parsed', {})
            fixed, event_merges = collapse_explicit_coreference(parsed, event['mentions'], event['panels'])
            merged.extend({'event': event['id'], **m} for m in event_merges)
            for mention in parsed.get('mentions', []):
                frame_counts.update(p['frame_id'] for p in mention['predictions'])
            for source, rows in ((parsed, predictions), (fixed, collapsed)):
                adapted = evaluator_adapter(source)
                rows.append({'scene': event['scene_key'], 'row_index': event['row_index'],
                    'event_id': event['event_id'], 'instruction': event['instruction'],
                    'model_input_image': ';'.join(x['path'] for x in response['input_images']),
                    'model_raw_output': response.get('raw_response') or '', 'parsed_json': json.dumps(adapted),
                    'parse_ok': str(response['status'] in ('COMPLETE', 'PARSE_WITH_WARNINGS')),
                    'prediction_count': len(adapted['referents']), 'error_message': response.get('error', '')})
        for suffix, rows in (('', predictions), ('_coref_collapsed', collapsed)):
            pred = output / 'predictions' / f'{condition}{suffix}.csv'
            write_csv(pred, rows, CSV_COLUMNS)
            destination = output / ('evaluation' + suffix) / condition
            subprocess.run([sys.executable, str(evaluator), '--manifest', str(output / 'rebuilt_manifest_all.csv'),
                            '--pred_csv', str(pred), '--output_dir', str(destination), '--coordinate_mode', 'panel'],
                            capture_output=True, text=True, check=True)
            metric = json.loads((destination / '2d_eval_summary.json').read_text())
            summaries[condition + suffix] = metric
            exact[condition + suffix] = exact_counts(read_csv(destination / '2d_eval_detail.csv'))
        statuses[condition], frames[condition], merges[condition] = dict(status), dict(frame_counts), merged
    save_json(output / 'summary.json', {'metrics': summaries, 'exact': exact, 'statuses': statuses,
        'panel_prediction_counts': frames, 'explicit_coreference_merges': merges,
        'sample_status': manifest['sample_status'], 'scope': manifest['condition_interpretation']})
    print('SCORED_MAIN_RGB', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'infer', 'evaluate'))
    parser.add_argument('--source', type=Path, default=ROOT / '.local-sync/projected2d_routed_composite_v2_20261109')
    parser.add_argument('--output', type=Path, default=ROOT / '.local-sync/projected2d_main_rgb32_20261109')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.source.resolve(), args.output.resolve())
    elif args.stage == 'infer':
        infer(args.output.resolve())
    else:
        evaluate(args.output.resolve())


if __name__ == '__main__':
    main()
