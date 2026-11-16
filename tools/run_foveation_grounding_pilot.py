#!/usr/bin/env python3
"""Paired 8-interaction full-field visual grounding × foveation pilot."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np

from MultiAgent.attention_reliability import cluster_paired_interval
from MultiAgent.full_frame_grounding import grounding_prompt, validate_grounding, decode_grounding
from MultiAgent.full_interaction_referring import paired_set_metrics
from MultiAgent.heatmap_temporal_pilot import save, sha
from MultiAgent.qwen_backend import QwenBackend
from tools.run_full_interaction_referring import ROOT, read
from tools.run_foveation_pilot import CONDITIONS, OUTPUT as FOVEATION_ROOT

OUTPUT = ROOT / '.local-sync/full_interaction_foveation_grounding_20261106'
PARENT = ROOT / '.local-sync/full_interaction_referring_20261106'
SELECTED_IDS = ['scene1_261', 'scene2_411', 'scene3_577', 'scene4_room1_72',
                'scene4_room2_149', 'scene4_room3_94', 'scene4_room4_25', 'scene5_112']


def foveation_grounding_prompt(event):
    prompt = grounding_prompt(event)
    old = '''Each video frame has FOUR full-field panels: TOP LEFT original RGB, TOP RIGHT
EARLIER attention, BOTTOM LEFT MIDDLE attention, BOTTOM RIGHT RECENT attention.
The three maps cover the exact same full camera field as RGB, and summarize the
two seconds before that RGB frame. Orange is weak measured gaze density. It is
not a target label; it may be on background, hands, reference or other objects.
Black means missing added gaze, not negative evidence. Some graphics are control
conditions. Original RGB recorder marks remain. Judge RGB and language first.'''
    new = '''Each video frame has FOUR full-field panels. TOP LEFT is a complete RGB-derived
camera view; in some conditions its local sharpness varies with past gaze history,
while the entire field and all objects remain present. Sharpness is a rendering
cue, never an object label or a target box. TOP RIGHT / BOTTOM LEFT / BOTTOM RIGHT
are EARLIER / MIDDLE / RECENT attention maps from the preceding two seconds. They
cover the same full camera field. Orange is measured gaze density, not target
identity; it can fall on hands, background, references or unrelated objects.
Black means missing added gaze, not negative evidence. Some panels are controls.
Original RGB recorder marks remain. Ground objects from the full interaction,
visible appearance and action; use gaze evidence only when it supports the scene.'''
    if old not in prompt:
        raise ValueError('Visual grounding prompt changed; review before freezing')
    return prompt.replace(old, new, 1)


def prepare(output):
    if (output / 'manifest.json').exists():
        raise ValueError('Frozen followup exists; never overwrite its manifest')
    pilot = read(FOVEATION_ROOT / 'manifest.json')
    assert sha(FOVEATION_ROOT / 'manifest.json') == read(FOVEATION_ROOT / 'protocol_lock.json')['manifest_sha256']
    pilot_events = {event['id']: event for event in pilot['events']}
    events = [json.loads(json.dumps(pilot_events[event_id])) for event_id in SELECTED_IDS]
    requests = []
    for event in events:
        event['prompt'] = foveation_grounding_prompt(event)
        fingerprints = {}
        for frame in event['frames']:
            for condition in CONDITIONS:
                image = frame['images_foveation_pilot'][condition]
                source_path = FOVEATION_ROOT / image['path']
                assert sha(source_path) == image['sha256']
                image['path'] = os.path.relpath(source_path, output)
        for condition in CONDITIONS:
            hashes = [frame['images_foveation_pilot'][condition]['sha256'] for frame in event['frames']]
            fingerprint = hashlib.sha256(json.dumps([event['prompt'], hashes], sort_keys=True).encode()).hexdigest()
            request_id = event['id'] + '_' + condition
            canonical = fingerprints.setdefault(fingerprint, request_id)
            requests.append({'id': request_id, 'event_id': event['id'], 'condition': condition,
                'canonical_request_id': canonical, 'model_input_fingerprint': fingerprint})
    source_files = ['MultiAgent/foveation_attention.py', 'MultiAgent/full_frame_grounding.py',
        'MultiAgent/full_interaction_referring.py', 'tools/run_foveation_grounding_pilot.py',
        'MultiAgent/qwen_backend.py', 'MultiAgent/qwen_video_compat.py']
    manifest = {**pilot, 'version': 'full_field_foveation_grounding_pilot_v1', 'events': events,
        'requests': requests, 'conditions': list(CONDITIONS),
        'parent_manifest_sha256': sha(FOVEATION_ROOT / 'manifest.json'),
        'selected_event_ids': SELECTED_IDS,
        'selection_policy': 'Same eight historical median-per-subscene interactions as grounding_v3; fixed before this pilot, no GT, foveation outputs or model predictions used.',
        'planned_unique_model_calls': len({row['canonical_request_id'] for row in requests}),
        'model_task': 'Select every intended object from full chronological video as category + point in the complete top-left RGB field; same fixed all-scene geometric ID decoder for all four conditions.',
        'coordinate_contract': 'Model coordinates normalized within top-left RGB panel only; model sees the complete 2x2 image and all video frames. No crop, zoom, anchor projection or answer-conditioned candidate.',
        'evaluation_policy': 'Freeze all visual outputs before reading GT. Score decoded final scene-ID referent sets with original evaluator and paired event metrics.',
        'gt_loaded': False, 'anchors_projected': False, 'object_crops': False,
        'answer_conditioned_candidates': False,
        'source_sha256': {name: sha(ROOT / name) for name in source_files},
        'adaptive_regression_only': True, 'unseen_validation': False}
    save(output / 'manifest.json', manifest)
    save(output / 'protocol_lock.json', {'manifest_sha256': sha(output / 'manifest.json'),
        'before_any_inference': True, 'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'shared_prompt_across_conditions': True, 'gt_loaded': False,
        'primary_metrics': ['Exact-set', 'micro-F1']})
    for name in source_files:
        destination = output / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    print('FROZEN', len(events), 'interactions', len(requests), 'conditions',
        manifest['planned_unique_model_calls'], 'unique calls', flush=True)


def infer(output, limit=None):
    manifest = read(output / 'manifest.json')
    assert sha(output / 'manifest.json') == read(output / 'protocol_lock.json')['manifest_sha256']
    backend = QwenBackend(output / 'model_cache', model=manifest['model'], local_only=True,
        namespace=manifest['version'], trace_visual_inputs=True,
        video_max_pixels=manifest['video_max_pixels'])
    events = {event['id']: event for event in manifest['events']}
    canonical = {row['id']: row for row in manifest['requests'] if row['id'] == row['canonical_request_id']}
    aliases = {rid: [row for row in manifest['requests'] if row['canonical_request_id'] == rid]
        for rid in canonical}
    schedule = list(canonical.values())
    random.Random(20261106).shuffle(schedule)
    completed = 0
    for request in schedule:
        if all((output / 'responses' / f"{alias['id']}.json").exists() for alias in aliases[request['id']]):
            continue
        event = events[request['event_id']]
        condition = request['condition']
        paths = [output / frame['images_foveation_pilot'][condition]['path'] for frame in event['frames']]
        for path, frame in zip(paths, event['frames']):
            assert sha(path) == frame['images_foveation_pilot'][condition]['sha256']
        metadata = [{key: frame[key] for key in ('frame_id', 'source_frame_id', 'source_frame_index', 'time_seconds')}
            for frame in event['frames']]
        start = time.monotonic()
        print('INFER', request['id'], flush=True)
        parsed = backend.ask_video(request['id'], 'full_frame_referent_localization', event['prompt'], paths,
            metadata, max_new_tokens=1000, video_path=event['video_path'],
            video_time_range=[event['frames'][0]['time_seconds'], event['frames'][-1]['time_seconds']])
        raw = read(backend.requests[-1])
        audit = raw['processor_audit']
        grounded = validate_grounding(parsed, event)
        decoder = decode_grounding(grounded['referents'], event)
        validation = {'schema_valid': grounded['schema_valid'], 'errors': grounded['errors'],
            'selected_object_names': decoder['selected_object_names'] if grounded['schema_valid'] else []}
        for alias in aliases[request['id']]:
            save(output / 'responses' / f"{alias['id']}.json", {**alias, 'event': event['event'],
                'scene_key': event['scene_key'], 'row_index': event['row_index'], 'parsed': parsed,
                'grounding_validation': grounded, 'decoder': decoder, 'validation': validation,
                'request_path': backend.requests[-1], 'processor_audit': audit,
                'vision_verified': audit['visual_tensor_trace']['input_consumption_verified'],
                'seconds': time.monotonic() - start, 'exact_input_reused': alias['id'] != request['id']})
        print('RESULT', request['id'], 'schema', grounded['schema_valid'],
            'visual_rows', len(grounded['referents']), 'selected', validation['selected_object_names'],
            'tokens', audit['expanded_video_token_count'], 'seconds', round(time.monotonic()-start, 2), flush=True)
        save(output / 'progress.json', {'completed_condition_results': len(list((output / 'responses').glob('*.json'))),
            'planned_condition_results': len(manifest['requests']), 'planned_unique_model_calls': len(canonical),
            'last': request['id'], 'gt_loaded': False})
        completed += 1
        if limit and completed >= limit:
            break



def paired_micro_interval(rows, baseline, treatment, seed=20261106, replicates=5000):
    arrays = {}
    for condition in (baseline, treatment):
        arrays[condition] = np.array([[row['metrics'][condition][key] for key in ('tp', 'fp', 'fn')]
            for row in rows], dtype=float)
    def score(counts):
        tp, fp, fn = counts.sum(axis=0)
        denominator = 2*tp + fp + fn
        return 2*tp/denominator if denominator else 0.
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(rows), size=(replicates, len(rows)))
    draws = {}
    for condition, values in arrays.items():
        totals = values[indices].sum(axis=1)
        denominator = 2*totals[:, 0] + totals[:, 1] + totals[:, 2]
        draws[condition] = np.divide(2*totals[:, 0], denominator,
            out=np.zeros(replicates), where=denominator > 0)
    return {'events': len(rows), 'gain': score(arrays[treatment])-score(arrays[baseline]),
        'interval_95': np.quantile(draws[treatment]-draws[baseline], [0.025, 0.975]).tolist(),
        'replicates': replicates, 'resampling_unit': 'paired interaction'}

def evaluate(output):
    manifest = read(output / 'manifest.json')
    assert sha(output / 'manifest.json') == read(output / 'protocol_lock.json')['manifest_sha256']
    responses = {row['id']: read(output / 'responses' / f"{row['id']}.json") for row in manifest['requests']}
    if len(responses) != len(manifest['requests']):
        raise ValueError('All condition outputs must be frozen before evaluation')
    save(output / 'predictions_frozen.json', {'manifest_sha256': sha(output / 'manifest.json'),
        'before_gt_evaluation': True, 'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'response_sha256': {key: sha(output / 'responses' / f'{key}.json') for key in responses}})
    from MultiAgent.baseline import module, scene_paths
    evaluator = module('eval')
    gt_by_scene = {}
    for scene in sorted({event['scene_key'] for event in manifest['events']}):
        aliases, _ = evaluator.load_anchor_alias_map(scene_paths(scene)['anchor'])
        gt_by_scene[scene] = (aliases, evaluator.load_gt_rows(scene_paths(scene)['gt'], aliases))
    result = {'conditions': {}, 'events': [], 'all_schema_valid': all(row['grounding_validation']['schema_valid'] for row in responses.values()),
        'all_vision_verified': all(row['vision_verified'] for row in responses.values()),
        'gt_loaded_only_after_frozen_predictions': True, 'historical_eight_only': True,
        'expanded_video_tokens': sorted({row['processor_audit']['expanded_video_token_count'] for row in responses.values()}),
        'source_manifest_sha256': sha(output / 'manifest.json')}
    gt_by_event = {}
    for condition in CONDITIONS:
        rows = []
        original_rows = []
        original_gt = []
        for event in manifest['events']:
            response = responses[event['id'] + '_' + condition]
            prediction = response['validation']['selected_object_names']
            destination = output / 'predictions' / condition / event['scene_key'] / f"row_{event['row_index']}.json"
            save(destination, {'row_index': event['row_index'], 'event_id': event['event_id'],
                'parsed_response': {'selected_object_names': prediction},
                'response_status': 'ok' if response['validation']['schema_valid'] else 'schema_failure',
                'used_storyboard_fallback': False})
            aliases, gt_rows = gt_by_scene[event['scene_key']]
            gt = gt_rows[event['row_index']]
            gt_by_event[event['id']] = gt['mapped_referents']
            rows.append({'id': event['id'], 'gt': gt['mapped_referents'], 'prediction': prediction})
            original_rows.extend(evaluator.build_eval_rows([destination], gt_rows, aliases))
            original_gt.append(gt)
        summary = evaluator.summarize(original_rows, original_gt)
        save(output / 'evaluation' / condition / 'original_summary.json', summary)
        evaluator.write_eval_csv(output / 'evaluation' / condition / 'evaluation.csv', original_rows)
        result['conditions'][condition] = {'set_metrics': paired_set_metrics([row for row in rows if row['gt']]),
            'original_evaluator': summary, 'schema_failures': sum(not responses[event['id']+'_'+condition]['validation']['schema_valid']
                for event in manifest['events']), 'visual_rows': sum(len(responses[event['id']+'_'+condition]['grounding_validation']['referents'])
                for event in manifest['events']), 'decoded_rows': sum(sum(d['selected_candidate'] is not None
                for d in responses[event['id']+'_'+condition]['decoder']['decisions']) for event in manifest['events'])}
    for event in manifest['events']:
        gt = set(gt_by_event[event['id']])
        predictions = {condition: responses[event['id']+'_'+condition]['validation']['selected_object_names'] for condition in CONDITIONS}
        metrics = {condition: {'exact': set(prediction) == gt, 'tp': len(set(prediction)&gt),
            'fp': len(set(prediction)-gt), 'fn': len(gt-set(prediction)),
            'f1': 2*len(set(prediction)&gt)/(len(prediction)+len(gt)) if len(prediction)+len(gt) else 0.}
            for condition, prediction in predictions.items()}
        result['events'].append({'id': event['id'], 'event': event['event'], 'instruction': event['instruction'],
            'gt': sorted(gt), 'predictions': predictions, 'metrics': metrics})
    contrasts = {}
    for baseline, treatment, label in [('RGB', 'FOVEATION', 'foveation_without_heat'),
                                       ('HEATMAP', 'FOVEATION_HEATMAP', 'foveation_with_heat')]:
        rows = result['events']
        contrasts[label] = {
            'micro_f1': paired_micro_interval(rows, baseline, treatment),
            'exact': cluster_paired_interval([{'cluster': row['id'], 'baseline': float(row['metrics'][baseline]['exact']),
                'treatment': float(row['metrics'][treatment]['exact'])} for row in rows]),
            'mean_event_f1': cluster_paired_interval([{'cluster': row['id'], 'baseline': row['metrics'][baseline]['f1'],
                'treatment': row['metrics'][treatment]['f1']} for row in rows]),
            'helpful_interactions': sum(row['metrics'][treatment]['f1']>row['metrics'][baseline]['f1'] for row in rows),
            'harmful_interactions': sum(row['metrics'][treatment]['f1']<row['metrics'][baseline]['f1'] for row in rows)}
    result['paired_contrasts'] = contrasts
    save(output / 'analysis.json', result)
    print(json.dumps({key: value['set_metrics'] for key, value in result['conditions'].items()}, ensure_ascii=False, indent=2))


def build_viewer(output):
    manifest = read(output / 'manifest.json')
    analysis = read(output / 'analysis.json') if (output / 'analysis.json').exists() else None
    responses = {path.stem: read(path) for path in (output / 'responses').glob('*.json')}
    events = [{'id': event['id'], 'event': event['event'], 'instruction': event['instruction'],
        'frames': [{'frame_id': frame['frame_id'], 'time_seconds': frame['time_seconds'],
            'images': frame['images_foveation_pilot'], 'foveation': frame['foveation']} for frame in event['frames']]}
        for event in manifest['events']]
    slim = {key: {'parsed': val['parsed'], 'validation': val['validation'],
        'grounding': val['grounding_validation'], 'decoder': val['decoder']} for key, val in responses.items()}
    payload = json.dumps({'events': events, 'conditions': CONDITIONS, 'responses': slim,
        'analysis': analysis}, ensure_ascii=False).replace('</', '<\\/')
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Foveation grounding pilot</title><style>body{font:16px system-ui;line-height:1.5;margin:18px;background:#f1f4f7}header{position:sticky;top:0;background:#f1f4f7}select{font:inherit;margin:6px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(510px,1fr));gap:12px}figure{margin:0;background:white;padding:8px}img{width:100%}pre{white-space:pre-wrap;background:white;padding:10px}</style><header><h2>完整画面 referent 定位 × foveation</h2><select id="e"></select><select id="c"></select><div id="i"></div></header><pre id="r"></pre><main></main><script>const D=__D__,E=document.querySelector('#e'),C=document.querySelector('#c');D.events.forEach(x=>{let o=document.createElement('option');o.value=x.id;o.textContent=x.event;E.append(o)});D.conditions.forEach(x=>{let o=document.createElement('option');o.value=x;o.textContent=x;C.append(o)});function go(){let e=D.events.find(x=>x.id===E.value),r=D.responses[e.id+'_'+C.value],a=D.analysis?.events.find(x=>x.id===e.id);document.querySelector('#i').textContent=e.instruction;document.querySelector('#r').textContent='视觉点 '+JSON.stringify(r?.grounding)+'\n场景ID '+JSON.stringify(r?.decoder)+'\n评分 '+JSON.stringify(a?.metrics);let m=document.querySelector('main');m.replaceChildren();e.frames.forEach(f=>{let g=document.createElement('figure'),im=document.createElement('img'),t=document.createElement('figcaption');im.src=f.images[C.value].path;im.loading='lazy';t.textContent=f.frame_id+' '+f.time_seconds.toFixed(3)+'s '+JSON.stringify(f.foveation);g.append(im,t);m.append(g)})}E.onchange=C.onchange=go;go()</script></html>'''
    (output / 'index.html').write_text(page.replace('__D__', payload))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'infer', 'evaluate', 'viewer'])
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    output = args.output.resolve()
    {'prepare': prepare, 'infer': lambda path: infer(path, args.limit),
     'evaluate': evaluate, 'viewer': build_viewer}[args.action](output)
