#!/usr/bin/env python3
"""Frozen V4 mention-accounting pilot on the same full-field inputs as V3."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time

from MultiAgent.full_frame_grounding import decode_grounding
from MultiAgent.full_frame_grounding_v4 import VERSION, grounding_prompt, validate_mention_contract
from MultiAgent.heatmap_temporal_pilot import save, sha
from MultiAgent.qwen_backend import QwenBackend
from tools.run_full_interaction_referring import ROOT, read
from tools.run_full_interaction_referring import evaluate as evaluate_final_sets

PARENT = ROOT / '.local-sync/full_interaction_referring_20261106/grounding_v3'
OUTPUT = ROOT / '.local-sync/full_interaction_referring_20261108/grounding_v4'


def clean_visual_trace(request_path):
    trace_dir = Path(request_path).with_suffix('.tensor_trace')
    removed = []
    for pattern in ('processor_inputs.pt', 'processor_rgb_*.png'):
        for path in trace_dir.glob(pattern):
            path.unlink()
            removed.append(str(path))
    return removed


def prepare(output):
    if (output / 'manifest.json').exists():
        raise ValueError('Never overwrite frozen V4 manifest')
    parent = read(PARENT / 'manifest.json')
    events = json.loads(json.dumps(parent['events']))
    requests = []
    for event in events:
        event['prompt'] = grounding_prompt(event)
        fingerprints = {}
        for frame in event['frames']:
            for image in frame['images'].values():
                original = PARENT / image['path']
                image['path'] = os.path.relpath(original.resolve(), output.resolve())
        for condition in parent['conditions']:
            hashes = [frame['images'][condition]['sha256'] for frame in event['frames']]
            fingerprint = hashlib.sha256(json.dumps([event['prompt'], hashes], sort_keys=True).encode()).hexdigest()
            request_id = event['id'] + '_' + condition
            canonical = fingerprints.setdefault(fingerprint, request_id)
            requests.append({'id': request_id, 'event_id': event['id'], 'condition': condition,
                             'canonical_request_id': canonical, 'model_input_fingerprint': fingerprint})
    names = ['MultiAgent/full_frame_grounding.py', 'MultiAgent/full_frame_grounding_v4.py',
             'tools/run_full_frame_grounding_v4.py', 'MultiAgent/qwen_backend.py',
             'MultiAgent/qwen_video_compat.py']
    manifest = {**parent, 'version': VERSION, 'events': events, 'requests': requests,
        'parent_manifest_sha256': sha(PARENT / 'manifest.json'),
        'planned_unique_model_calls': len({r['canonical_request_id'] for r in requests}),
        'model_task': 'One-row-per-parser-mention full-field grounding with GT-free scene-ID decoding.',
        'coordinate_contract': 'Model whole displayed image 0..1000; script converts grounded points to local RGB.',
        'mention_contract': 'Stable mention_id; exactly one row per occurrence; grounded or unresolved status.',
        'trace_retention_policy': 'Keep trace.json and response audit; delete processor_inputs.pt and processor_rgb_*.png after each response.',
        'default_pipeline_changed': False, 'unseen_validation': False, 'adaptive_regression_only': True,
        'source_sha256': {name: sha(ROOT / name) for name in names}}
    save(output / 'manifest.json', manifest)
    save(output / 'protocol_lock.json', {'manifest_sha256': sha(output / 'manifest.json'),
        'before_any_inference': True, 'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'gt_loaded': False, 'primary_metrics': ['Exact-set', 'micro-F1', 'mention coverage']})
    for name in names:
        destination = output / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    print('FROZEN', len(events), 'interactions', len(requests),
          'unique model calls', manifest['planned_unique_model_calls'], flush=True)


def infer(output, limit=None):
    manifest = read(output / 'manifest.json')
    assert sha(output / 'manifest.json') == read(output / 'protocol_lock.json')['manifest_sha256']
    backend = QwenBackend(output / 'model_cache', model=manifest['model'], local_only=True,
                          namespace=VERSION, trace_visual_inputs=True,
                          video_max_pixels=manifest['video_max_pixels'])
    events = {event['id']: event for event in manifest['events']}
    canonical = {request['id']: request for request in manifest['requests']
                 if request['id'] == request['canonical_request_id']}
    aliases = {rid: [request for request in manifest['requests']
                      if request['canonical_request_id'] == rid] for rid in canonical}
    schedule = list(canonical.values())
    random.Random(20261108).shuffle(schedule)
    completed = 0
    for request in schedule:
        if all((output / 'responses' / f"{alias['id']}.json").exists() for alias in aliases[request['id']]):
            continue
        event = events[request['event_id']]
        condition = request['condition']
        paths = [output / frame['images'][condition]['path'] for frame in event['frames']]
        for path, frame in zip(paths, event['frames']):
            assert sha(path) == frame['images'][condition]['sha256']
        metadata = [{key: frame[key] for key in ('frame_id', 'source_frame_id', 'source_frame_index', 'time_seconds')}
                    for frame in event['frames']]
        print('INFER', request['id'], flush=True)
        started = time.monotonic()
        parsed = backend.ask_video(request['id'], 'global_visual_grounding', event['prompt'], paths, metadata,
                                   max_new_tokens=manifest['max_new_tokens'], video_path=event['video_path'],
                                   video_time_range=[event['frames'][0]['time_seconds'], event['frames'][-1]['time_seconds']])
        raw = read(backend.requests[-1])
        audit = raw['processor_audit']
        checked = validate_mention_contract(parsed, event)
        decoded = decode_grounding(checked['grounded_referents'], event)
        validation = {'schema_valid': checked['schema_valid'], 'coverage_complete': checked['coverage_complete'],
                      'errors': checked['errors'], 'missing_mention_ids': checked['missing_mention_ids'],
                      'duplicate_mention_ids': checked['duplicate_mention_ids'],
                      'selected_object_names': decoded['selected_object_names'] if checked['schema_valid'] else []}
        for alias in aliases[request['id']]:
            save(output / 'responses' / f"{alias['id']}.json", {**alias, 'event': event['event'],
                'scene_key': event['scene_key'], 'row_index': event['row_index'], 'parsed': parsed,
                'mention_validation': checked, 'grounding_validation': checked, 'validation': validation,
                'decoder': decoded, 'request_path': backend.requests[-1], 'processor_audit': audit,
                'vision_verified': bool(audit.get('visual_tensor_trace', {}).get('input_consumption_verified')),
                'seconds': time.monotonic() - started, 'exact_input_reused': alias['id'] != request['id'],
                'trace_cleanup_policy': 'processor_inputs.pt and processor_rgb_*.png removed after response freeze'})
        removed = clean_visual_trace(backend.requests[-1])
        print('RESULT', request['id'], 'schema', checked['schema_valid'],
              'coverage', checked['coverage_complete'], 'grounded', len(checked['grounded_referents']),
              'decoded', validation['selected_object_names'], 'trace_removed', len(removed),
              'seconds', round(time.monotonic() - started, 2), flush=True)
        save(output / 'progress.json', {'completed_condition_results': len(list((output / 'responses').glob('*.json'))),
            'planned_condition_results': len(manifest['requests']), 'planned_unique_model_calls': len(canonical),
            'last': request['id'], 'gt_loaded': False})
        completed += 1
        if limit and completed >= limit:
            break


def revalidate(output):
    manifest = read(output / 'manifest.json')
    events = {event['id']: event for event in manifest['events']}
    response_paths = sorted((output / 'responses').glob('*.json'))
    updated = 0
    for path in response_paths:
        data = read(path)
        event = events[data['event_id']]
        checked = validate_mention_contract(data['parsed'], event)
        decoded = decode_grounding(checked['grounded_referents'], event)
        validation = {'schema_valid': checked['schema_valid'], 'coverage_complete': checked['coverage_complete'],
                      'errors': checked['errors'], 'missing_mention_ids': checked['missing_mention_ids'],
                      'duplicate_mention_ids': checked['duplicate_mention_ids'],
                      'selected_object_names': decoded['selected_object_names'] if checked['schema_valid'] else []}
        data['mention_validation'] = checked
        data['grounding_validation'] = checked
        data['validation'] = validation
        data['decoder'] = decoded
        data['validation_reprocessed_without_model_call'] = True
        save(path, data)
        updated += 1
    save(output / 'validation_reprocess.json', {'updated_responses': updated, 'model_calls_added': 0,
        'gt_loaded': False, 'reason': 'V4 validator amendment: grounded unresolved_reason may be omitted and is normalized to null.'})
    print('REVALIDATED', updated, 'responses; model calls added 0', flush=True)


def evaluate(output):
    evaluate_final_sets(output)
    manifest = read(output / 'manifest.json')
    analysis = read(output / 'analysis.json')
    results = [read(output / 'responses' / f"{request['id']}.json") for request in manifest['requests']]
    analysis['primary_scope'] = 'Final referent sets on 8 old fixed interactions. No GT-conditioned candidate filtering or crops.'
    analysis['unseen_validation'] = False
    analysis['default_pipeline_changed'] = False
    analysis['model_task'] = 'V4 one-row-per-mention full-field localization plus fixed GT-free scene-ID decoder.'
    analysis['mention_contract'] = {'responses': len(results),
        'schema_valid': sum(bool(r['mention_validation']['schema_valid']) for r in results),
        'coverage_complete': sum(bool(r['mention_validation']['coverage_complete']) for r in results),
        'grounded_rows': sum(len(r['mention_validation']['grounded_referents']) for r in results),
        'unresolved_rows': sum(len(r['mention_validation']['unresolved_mentions']) for r in results),
        'missing_mentions': sum(len(r['mention_validation']['missing_mention_ids']) for r in results),
        'duplicate_mentions': sum(len(r['mention_validation']['duplicate_mention_ids']) for r in results)}
    analysis['trace_retention'] = {'processor_tensor_files_retained': False, 'processor_rgb_trace_files_retained': False,
                                   'trace_json_retained': True}
    save(output / 'analysis.json', analysis)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'infer', 'revalidate', 'evaluate'])
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    {'prepare': prepare, 'infer': lambda path: infer(path, args.limit), 'revalidate': revalidate, 'evaluate': evaluate}[args.action](args.output.resolve())
