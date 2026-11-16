#!/usr/bin/env python3
"""Audit and probe per-referent frame selection on frozen 2D RGB inputs."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from MultiAgent.projected2d_attention_study import CSV_COLUMNS, evaluator_adapter, read_csv, save_json, sha_file, write_csv
from MultiAgent.projected2d_main_image import normalize, verify_lock
from MultiAgent.projected2d_routed_rgb import MODEL, ROOT

SOURCE = ROOT / '.local-sync/projected2d_main_rgb32_20261109'
FOCUS = ROOT / '.local-sync/projected2d_mention_focus8_labelled_20261109'
PRIOR = ROOT / '.local-sync/referent_rgb_expansion32_20261109_repaired/manifest.json'
OUTPUT = ROOT / '.local-sync/referent_frame_choice_audit_20261109'
CONDITIONS = ('NEUTRAL_CHRONO', 'NEUTRAL_REVERSED', 'PER_REFERENT_COMPARISON')


def prompt_for(event, condition):
    tasks = []
    for mention in event['mentions']:
        task = {**mention, 'soft_time_hint': event['temporal_hints'].get(mention['mention_id'])}
        if condition == 'PER_REFERENT_COMPARISON':
            task['action_context'] = (event['phase_context'].get(mention['mention_id']) or {}).get('action_context')
        tasks.append(task)
    frames = [{'frame_id': p['frame_id'], 'video_time_seconds': p['requested_video_time']} for p in event['panels']]
    shared = '''Solve 2D temporal localization for each mention in the complete instruction.
Every image is an independent full RGB video frame. The caption immediately before each image
states its frame_id and video time. Image order may differ from chronological order. Use frame_id
and video time to reason about the interaction, rather than the image's position in the request.
The reference list source_frames is chronological. Soft time hints are rough clause-order priors,
not actual speech timestamps. They can be too broad or contain no supplied frame. All frames remain
eligible; do not invent an action or a target when the frames do not show enough evidence.
Choose a frame independently for each mention. Different referents can have different best frames,
and the same frame is valid for more than one referent when evidence supports it.
For each mention return exactly one row. A localized physical target needs a frame_id and point_xy
[x,y] in the entire corresponding image, scale 0..1000, origin top-left. Return no point when
ambiguous or not_visible. A plural can have multiple physical targets. For a component locate the
indicated whole parent. Use RGB appearance and relationships to distinguish similar instances.
'''
    if condition == 'PER_REFERENT_COMPARISON':
        shared += '''For EACH mention, inspect EVERY supplied frame separately for visibility, identity and action
context BEFORE selecting its best evidence frame. Do not use earliest visibility as the selection
rule. For every mention row, include frame_checks: one object per supplied frame, each containing
frame_id and a concise observation (at most 8 words) about the specific referent in that frame.
If the rough action phase has no supplied frames, explicitly note the missing temporal evidence;
do not treat a phase guess as proof. Still use any frame whose RGB independently supports the target.
'''
    return shared + '''Return JSON only in this structure:
{"mentions":[{"mention_id":"...","status":"localized|ambiguous|not_visible",
"predictions":[{"frame_id":"one_of_the_supplied_frames","point_xy":[x,y]}],"evidence":"..."}]}
''' + json.dumps({'instruction': event['instruction'], 'source_frames': frames, 'localization_tasks': tasks}, ensure_ascii=False)


def code_lock(out):
    lock = json.loads((out / 'protocol_lock.json').read_text())
    for name, digest in lock['files'].items():
        if sha_file(Path(name)) != digest:
            raise ValueError('Locked file changed: ' + name)


def prepare(out):
    if (out / 'input_manifest.json').exists():
        raise ValueError('Refusing to overwrite frozen study')
    verify_lock(SOURCE)
    source = json.loads((SOURCE / 'input_manifest.json').read_text())
    focus = json.loads((FOCUS / 'input_manifest.json').read_text())
    prior = json.loads(PRIOR.read_text()); prior_map = {e['id']: e for e in prior['events']}
    chosen = {e['id'] for e in focus['events']} | {'scene2_412', 'scene2_482'}
    events = []
    for event in source['events']:
        if event['id'] not in chosen:
            continue
        temporal = prior_map[event['id']]['temporal_search']['mentions']
        event['phase_context'] = {k: {'action_context': v['action_context'],
            'source': 'instruction clause, not measured speech timing'} for k, v in temporal.items()}
        events.append(event)
    if len(events) != 10:
        raise ValueError('Expected 10 fixed events')
    out.mkdir(parents=True, exist_ok=True)
    manifest = {'version': 'referent_frame_choice_probe_v1', 'events': events, 'conditions': CONDITIONS,
        'sample_selection': 'Previously used eight multi-mention events, plus scene2_412 and scene2_482 to inspect multi-clause temporal coverage; selected after earlier scores; development only.',
        'condition_explanation': 'NEUTRAL_CHRONO and NEUTRAL_REVERSED have identical prompts and images, with captions attached to image identity; only presentation order differs. PER_REFERENT_COMPARISON adds action clauses and per-frame checks, chronological images.',
        'task': '2D full RGB frame points for each referent; no 3D candidate ID or target selection',
        'source_manifest_sha256': sha_file(SOURCE / 'input_manifest.json'),
        'source_frozen_predictions_sha256': sha_file(SOURCE / 'predictions_frozen.json'),
        'old_frozen_metrics_used_to_select': True, 'ground_truth_in_prompt': False}
    save_json(out / 'input_manifest.json', manifest)
    rows = read_csv(SOURCE / 'rebuilt_manifest_all.csv'); keys = {(e['scene_key'], e['row_index']) for e in events}
    subset = [r for r in rows if (r['scene'], int(r['row_index'])) in keys]
    write_csv(out / 'rebuilt_manifest_all.csv', subset, list(subset[0]))
    files = [Path(__file__).resolve(), out / 'input_manifest.json', out / 'rebuilt_manifest_all.csv',
        ROOT / 'MultiAgent/projected2d_main_image.py', ROOT / 'MultiAgent/qwen_backend.py',
        ROOT / 'MultiAgent/projected2d_attention_study.py', ROOT / 'tools/run_full_frame_grounding_v4.py',
        ROOT.parent / 'TriModal-Referring/exam2/evaluate_2d_point_grounding.py', MODEL / 'config.json']
    save_json(out / 'protocol_lock.json', {'files': {str(p): sha_file(p) for p in files}})
    print('FROZEN', len(events), 'events', sum(len(e['mentions']) for e in events), 'mentions', flush=True)


def infer(out):
    from MultiAgent.qwen_backend import QwenBackend
    from tools.run_full_frame_grounding_v4 import clean_visual_trace
    code_lock(out); m = json.loads((out / 'input_manifest.json').read_text())
    backend = QwenBackend(out / 'model_cache', str(MODEL), local_only=True, namespace='referent_frame_choice_v1', trace_visual_inputs=True)
    for i, event in enumerate(m['events'], 1):
        for condition in CONDITIONS:
            path = out / 'responses' / f"{event['id']}_{condition}.json"
            panels = list(event['panels'])
            if condition == 'NEUTRAL_REVERSED':
                panels.reverse()
            images = [p['inputs']['MAIN_RGB_768'] for p in panels]
            captions = [f"Frame {p['frame_id']} | video_time={p['requested_video_time']:.3f} seconds" for p in panels]
            prompt = prompt_for(event, condition)
            fingerprint = hashlib.sha256(json.dumps([prompt, captions, [x['sha256'] for x in images]]).encode()).hexdigest()
            if path.exists():
                if json.loads(path.read_text())['fingerprint'] != fingerprint:
                    raise ValueError('Saved result does not match frozen input')
                continue
            for x in images:
                if sha_file(Path(x['path'])) != x['sha256']:
                    raise ValueError('Image changed')
            req = None
            try:
                raw = backend.ask(event['id'], condition, prompt, [x['path'] for x in images], max_new_tokens=2304, image_captions=captions)
                req = Path(backend.requests[-1]); request = json.loads(req.read_text())
                parsed, errors, conversions = normalize(raw, event)
                checks = {}
                for row in raw.get('mentions', []):
                    if not isinstance(row, dict):
                        continue
                    mid = row.get('mention_id'); entries = row.get('frame_checks', [])
                    if not isinstance(entries, list):
                        entries = []
                    checks[mid] = {'ids': [x.get('frame_id') for x in entries if isinstance(x, dict)], 'entries': entries}
                comparisons = {mid: sorted(x['ids']) == sorted(p['frame_id'] for p in event['panels']) for mid, x in checks.items()}
                if raw.get('_parse_warning'):
                    errors.append(raw['_parse_warning'])
                result = {'status': 'COMPLETE' if not errors else 'PARSE_WITH_WARNINGS', 'parsed': parsed,
                    'warnings': errors, 'comparisons': comparisons, 'frame_checks': checks, 'conversions': conversions,
                    'raw_response': request['response'], 'request_path': str(req),
                    'processor_audit': request.get('processor_audit'),
                    'vision_verified': bool((request.get('visual_tensor_trace') or {}).get('input_consumption_verified'))}
                if not result['vision_verified']:
                    raise ValueError('Vision input not verified')
            except Exception as exc:
                result = {'status': 'FAILED', 'parsed': {'mentions': []}, 'error': f'{type(exc).__name__}: {exc}'}
                if backend.requests:
                    req = Path(backend.requests[-1])
            finally:
                if req:
                    clean_visual_trace(str(req))
            result.update(event_id=event['event_id'], scene=event['scene_key'], row_index=event['row_index'],
                condition=condition, fingerprint=fingerprint, prompt=prompt, input_images=images,
                input_frame_order=[p['frame_id'] for p in panels], image_captions=captions)
            save_json(path, result)
            print('PROBE', i, '/', len(m['events']), event['id'], condition, result['status'], flush=True)


def counts_metrics(out, condition, events):
    rows = []; stats = Counter(); per_event = {}
    for event in events:
        r = json.loads((out / 'responses' / f"{event['id']}_{condition}.json").read_text()); stats[r['status']] += 1
        pred = evaluator_adapter(r['parsed']); per_event[event['id']] = {p['mention_id']: p for p in r['parsed']['mentions']}
        for mention in r['parsed']['mentions']:
            for point in mention['predictions']:
                stats['point_' + point['frame_id']] += 1
                stats['point_first_input'] += point['frame_id'] == r['input_frame_order'][0]
                stats['point_total'] += 1
        rows.append({'scene': event['scene_key'], 'row_index': event['row_index'], 'event_id': event['event_id'],
          'instruction': event['instruction'], 'model_input_image': ';'.join(x['path'] for x in r['input_images']),
          'model_raw_output': r.get('raw_response', ''), 'parsed_json': json.dumps(pred, ensure_ascii=False),
          'parse_ok': str(r['status'] in ('COMPLETE', 'PARSE_WITH_WARNINGS')), 'prediction_count': len(pred['referents']),
          'error_message': r.get('error', '')})
    predpath = out / 'predictions' / f'{condition}.csv'; write_csv(predpath, rows, CSV_COLUMNS)
    evaluation = out / 'evaluation' / condition
    subprocess.run([sys.executable, str(ROOT.parent / 'TriModal-Referring/exam2/evaluate_2d_point_grounding.py'),
      '--manifest', str(out / 'rebuilt_manifest_all.csv'), '--pred_csv', str(predpath), '--output_dir', str(evaluation),
      '--coordinate_mode', 'panel'], check=True, capture_output=True, text=True)
    return {'stats': dict(stats), 'metrics': json.loads((evaluation / '2d_eval_summary.json').read_text()), 'per_event': per_event}


def evaluate(out):
    code_lock(out); m = json.loads((out / 'input_manifest.json').read_text())
    paths = [out / 'responses' / f"{e['id']}_{c}.json" for e in m['events'] for c in CONDITIONS]
    if any(not p.is_file() for p in paths):
        raise ValueError('Finish all conditions before scoring')
    save_json(out / 'predictions_frozen.json', {'responses': {str(p): sha_file(p) for p in paths}})
    res = {c: counts_metrics(out, c, m['events']) for c in CONDITIONS}
    compared = []
    for e in m['events']:
        a = res['NEUTRAL_CHRONO']['per_event'][e['id']]; b = res['NEUTRAL_REVERSED']['per_event'][e['id']]
        for mid in a.keys() & b.keys():
            pa = a[mid]['predictions']; pb = b[mid]['predictions']
            if len(pa) != 1 or len(pb) != 1:
                continue
            compared.append({'event': e['id'], 'mention_id': mid, 'chrono_frame': pa[0]['frame_id'],
               'reversed_frame': pb[0]['frame_id'], 'changed_frame': pa[0]['frame_id'] != pb[0]['frame_id'],
               'followed_first_in_both': pa[0]['frame_id'] == e['panels'][0]['frame_id'] and pb[0]['frame_id'] == e['panels'][-1]['frame_id']})
    comparison_checks = Counter()
    for e in m['events']:
        r = json.loads((out / 'responses' / f"{e['id']}_PER_REFERENT_COMPARISON.json").read_text())
        for mention in e['mentions']:
            mid = mention['mention_id']; comparison_checks['mentions'] += 1
            comparison_checks['all_frames_named'] += bool(r.get('comparisons', {}).get(mid))
    compact = {c: {k: v for k, v in value.items() if k != 'per_event'} for c, value in res.items()}
    save_json(out / 'summary.json', {'conditions': compact, 'counterfactual_same_mention_both_conditions': compared,
      'pair_count': len(compared), 'changed_frame_count': sum(x['changed_frame'] for x in compared),
      'first_position_follow_both_count': sum(x['followed_first_in_both'] for x in compared),
      'per_referent_comparison_coverage': dict(comparison_checks),
      'scope': 'Only 10 previously examined development events; comparison checks do not establish true visual understanding. Original 2D temporal labels are gaze-projection proxies.'})
    print('EVALUATED', len(m['events']), 'events', flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['prepare', 'infer', 'evaluate']); p.add_argument('--output', type=Path, default=OUTPUT); a = p.parse_args(); out = a.output.resolve()
    {'prepare': prepare, 'infer': infer, 'evaluate': evaluate}[a.stage](out)


if __name__ == '__main__':
    main()
