#!/usr/bin/env python3
"""Targeted V4.1 location-mention policy pilot; preserves V4 baseline."""
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
from MultiAgent.full_frame_grounding_v4 import VERSION as V4_VERSION, grounding_prompt, validate_mention_contract
from MultiAgent.heatmap_temporal_pilot import save, sha
from MultiAgent.qwen_backend import QwenBackend
from tools.run_full_interaction_referring import ROOT, read
from tools.run_full_frame_grounding_v4 import clean_visual_trace

PARENT = ROOT / '.local-sync/full_interaction_referring_20261108/grounding_v4'
OUTPUT = ROOT / '.local-sync/full_interaction_referring_20261108/grounding_v41_location'
EVENT_IDS = {'scene2_411', 'scene4_room2_149'}
LOCATION_POLICY = '''\n\nLOCATION MENTION POLICY: Every mention_id still needs one row. If a phrase is only a
relational or directional description with no independent scene object identity,
such as "over there" or "near the small table", use status unresolved with
unresolved_reason non_concrete_reference, and set category/view_id/x/y to null.
Do not invent category "location". If a location phrase clearly names a concrete
visible scene object, such as a warehouse or bedroom, it may be grounded using that
concrete scene category and a point on the object.\n'''


def prepare(output):
    if (output / 'manifest.json').exists():
        raise ValueError('Never overwrite V4.1 pilot')
    parent = read(PARENT / 'manifest.json')
    events = []
    for source_event in parent['events']:
        if source_event['id'] not in EVENT_IDS:
            continue
        event = json.loads(json.dumps(source_event))
        event['prompt'] = grounding_prompt(event) + LOCATION_POLICY
        for frame in event['frames']:
            for image in frame['images'].values():
                image['path'] = os.path.relpath((PARENT / image['path']).resolve(), output.resolve())
        events.append(event)
    events.sort(key=lambda event: event['id'])
    requests = []
    for event in events:
        fingerprints = {}
        for condition in parent['conditions']:
            hashes = [frame['images'][condition]['sha256'] for frame in event['frames']]
            fingerprint = hashlib.sha256(json.dumps([event['prompt'], hashes], sort_keys=True).encode()).hexdigest()
            rid = event['id'] + '_' + condition
            requests.append({'id': rid, 'event_id': event['id'], 'condition': condition,
                             'canonical_request_id': fingerprints.setdefault(fingerprint, rid),
                             'model_input_fingerprint': fingerprint})
    names = ['MultiAgent/full_frame_grounding.py', 'MultiAgent/full_frame_grounding_v4.py',
             'tools/run_full_frame_grounding_v4.py', 'tools/run_full_frame_grounding_v41_location.py',
             'MultiAgent/qwen_backend.py', 'MultiAgent/qwen_video_compat.py']
    manifest = {**parent, 'version': 'full_frame_grounding_v4_1_location_policy', 'events': events,
        'requests': requests, 'parent_v4_manifest_sha256': sha(PARENT / 'manifest.json'),
        'planned_unique_model_calls': len({r['canonical_request_id'] for r in requests}),
        'location_policy': LOCATION_POLICY, 'default_pipeline_changed': False,
        'unseen_validation': False, 'adaptive_regression_only': True,
        'trace_retention_policy': 'Keep response/audit/trace.json; delete processor_inputs.pt and processor_rgb_*.png after each response.',
        'source_sha256': {name: sha(ROOT / name) for name in names}}
    save(output / 'manifest.json', manifest)
    save(output / 'protocol_lock.json', {'manifest_sha256': sha(output / 'manifest.json'),
        'before_any_inference': True, 'gt_loaded': False,
        'frozen_utc': datetime.now(timezone.utc).isoformat()})
    for name in names:
        dest = output / 'source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    print('FROZEN', len(events), 'events', len(requests), 'conditions',
          manifest['planned_unique_model_calls'], 'unique calls', flush=True)


def infer(output):
    manifest = read(output / 'manifest.json')
    backend = QwenBackend(output / 'model_cache', model=manifest['model'], local_only=True,
                          namespace=manifest['version'], trace_visual_inputs=True,
                          video_max_pixels=manifest['video_max_pixels'])
    events = {event['id']: event for event in manifest['events']}
    canonical = {r['id']: r for r in manifest['requests'] if r['id'] == r['canonical_request_id']}
    aliases = {rid: [r for r in manifest['requests'] if r['canonical_request_id'] == rid] for rid in canonical}
    schedule = list(canonical.values()); random.Random(20261108).shuffle(schedule)
    for request in schedule:
        event = events[request['event_id']]; condition = request['condition']
        paths = [output / frame['images'][condition]['path'] for frame in event['frames']]
        metadata = [{key: frame[key] for key in ('frame_id', 'source_frame_id', 'source_frame_index', 'time_seconds')}
                    for frame in event['frames']]
        print('INFER', request['id'], flush=True)
        started = time.monotonic()
        parsed = backend.ask_video(request['id'], 'global_visual_grounding', event['prompt'], paths, metadata,
                                   max_new_tokens=manifest['max_new_tokens'], video_path=event['video_path'],
                                   video_time_range=[event['frames'][0]['time_seconds'], event['frames'][-1]['time_seconds']])
        raw = read(backend.requests[-1]); audit = raw['processor_audit']
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
                'seconds': time.monotonic() - started})
        removed = clean_visual_trace(backend.requests[-1])
        print('RESULT', request['id'], 'schema', checked['schema_valid'], 'coverage', checked['coverage_complete'],
              'grounded', len(checked['grounded_referents']), 'decoded', validation['selected_object_names'],
              'trace_removed', len(removed), 'seconds', round(time.monotonic() - started, 2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'infer'])
    args = parser.parse_args()
    {'prepare': prepare, 'infer': infer}[args.action](OUTPUT.resolve())
