#!/usr/bin/env python3
"""Paired full-interaction foveation pilot; evaluation is post-inference only."""
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
from PIL import Image, ImageDraw, ImageFont

from MultiAgent.attention_reliability import cluster_paired_interval
from MultiAgent.foveation_attention import VERSION as FOVEATION_VERSION, foveate_from_history
from MultiAgent.full_interaction_referring import paired_set_metrics, render_full_frame, validate_prediction
from MultiAgent.heatmap_temporal_pilot import save, sha
from MultiAgent.qwen_backend import QwenBackend
from tools.run_full_interaction_referring import OUTPUT as PARENT, ROOT, read

OUTPUT = ROOT / '.local-sync/full_interaction_foveation_20261106'
CONDITIONS = ('RGB', 'FOVEATION', 'HEATMAP', 'FOVEATION_HEATMAP')
HEAT_CONDITIONS = {'HEATMAP', 'FOVEATION_HEATMAP'}
FOVEATION_CONDITIONS = {'FOVEATION', 'FOVEATION_HEATMAP'}


def render_foveated_composite(rgb, maps, foveated_rgb, show_heat):
    return render_full_frame(foveated_rgb, maps,
        'TRUE_HEAT' if show_heat else 'NO_ADDED_HEAT')


def prompt_for(event):
    prompt = event['prompt']
    old = 'Every supplied frame has FOUR full-field panels. TOP LEFT is the original RGB;\nTOP RIGHT / BOTTOM LEFT / BOTTOM RIGHT are EARLIER / MIDDLE / RECENT graphical\nattention during the two seconds preceding that source frame. Each panel covers\nthe exact same full field and uses the same native orientation as RGB. Orange\nmeans measured attention density, not target identity. Black means missing added\nevidence, never evidence against an object. Some graphics are experimental\ncontrols. RGB already contains recorder marks. Attention may be on a hand,\nbackground, a reference or an unrelated object. Use it only when consistent with\nlanguage and RGB. Never report heat colors or coordinates as your answer.'
    new = 'Every supplied frame has FOUR full-field panels. TOP LEFT is full-field RGB;\nits local sharpness may be varied by a gaze-history rendering cue while retaining\nthe complete image and every object. TOP RIGHT / BOTTOM LEFT / BOTTOM RIGHT are\nEARLIER / MIDDLE / RECENT graphical attention during the two seconds preceding\nthat source frame. Each panel covers the exact same full field and uses the same\nnative orientation as RGB. Orange means measured attention density, not target\nidentity. Black means missing added evidence, never evidence against an object.\nSome graphics are experimental controls. RGB already contains recorder marks.\nAttention and sharpness may be on a hand, background, a reference or an\nunrelated object. Use them only when consistent with language and visible RGB.\nNever report heat colors or rendering sharpness as your answer.'
    if old not in prompt:
        raise ValueError('Parent prompt visual description changed; review before freezing')
    return prompt.replace(old, new, 1)


def prepare(output):
    if (output / 'manifest.json').exists():
        raise ValueError('Frozen experiment exists; never overwrite its manifest')
    parent_manifest = read(PARENT / 'manifest.json')
    events = json.loads(json.dumps(parent_manifest['events']))
    requests = []
    foveation_rows = []
    for event in events:
        event['prompt'] = prompt_for(event)
        fingerprints = {}
        for frame in event['frames']:
            source_path = Path(frame['source_path'])
            assert sha(source_path) == frame['source_sha256']
            map_path = PARENT / frame['maps_path']
            assert sha(map_path) == frame['maps_sha256']
            with Image.open(source_path) as opened:
                rgb = opened.convert('RGB')
            with np.load(map_path) as archive:
                maps = np.asarray(archive['maps'], dtype=np.float32)
            geometry = frame['geometry']
            camera = geometry['reference_pose']
            fov = float(camera['fov_vertical_degrees']) if camera else 90.
            frame_dir = output / 'events' / event['scene_key'] / str(event['row_index']) / frame['frame_id']
            frame_dir.mkdir(parents=True, exist_ok=True)
            image_records = {}
            foveated_rgb, frame_foveation_meta = foveate_from_history(rgb, maps,
                clock_supported=bool(geometry['clock_available']), vertical_fov_degrees=fov)
            for condition in CONDITIONS:
                if condition == 'RGB':
                    parent_image = frame['images']['NO_ADDED_HEAT']
                    image_records[condition] = {'path': os.path.relpath(PARENT / parent_image['path'], output),
                        'sha256': parent_image['sha256'], 'size': parent_image['size']}
                    continue
                if condition == 'HEATMAP':
                    parent_image = frame['images']['TRUE_HEAT']
                    image_records[condition] = {'path': os.path.relpath(PARENT / parent_image['path'], output),
                        'sha256': parent_image['sha256'], 'size': parent_image['size']}
                    continue
                if not frame_foveation_meta['foveation_enabled']:
                    parent_key = 'TRUE_HEAT' if condition == 'FOVEATION_HEATMAP' else 'NO_ADDED_HEAT'
                    parent_image = frame['images'][parent_key]
                    image_records[condition] = {'path': os.path.relpath(PARENT / parent_image['path'], output),
                        'sha256': parent_image['sha256'], 'size': parent_image['size']}
                    continue
                canvas = render_foveated_composite(rgb, maps, foveated_rgb,
                    show_heat=condition == 'FOVEATION_HEATMAP')
                path = frame_dir / f'{condition}.png'
                canvas.save(path, compress_level=3)
                image_records[condition] = {'path': str(path.relative_to(output)),
                    'sha256': sha(path), 'size': list(canvas.size)}
            frame['images_foveation_pilot'] = image_records
            frame['foveation'] = frame_foveation_meta
            foveation_rows.append({'event': event['id'], 'frame_id': frame['frame_id'], **frame['foveation']})
        for condition in CONDITIONS:
            image_hashes = [f['images_foveation_pilot'][condition]['sha256'] for f in event['frames']]
            fingerprint = hashlib.sha256(json.dumps([event['prompt'], image_hashes],
                sort_keys=True).encode()).hexdigest()
            request_id = event['id'] + '_' + condition
            canonical = fingerprints.setdefault(fingerprint, request_id)
            requests.append({'id': request_id, 'event_id': event['id'], 'condition': condition,
                'canonical_request_id': canonical, 'model_input_fingerprint': fingerprint})
    source_files = ['MultiAgent/foveation_attention.py', 'MultiAgent/full_interaction_referring.py',
        'tools/run_foveation_pilot.py', 'MultiAgent/qwen_backend.py', 'MultiAgent/qwen_video_compat.py']
    manifest = {**parent_manifest, 'version': FOVEATION_VERSION, 'events': events,
        'requests': requests, 'conditions': list(CONDITIONS),
        'parent_manifest_sha256': sha(PARENT / 'manifest.json'),
        'planned_unique_model_calls': len({r['canonical_request_id'] for r in requests}),
        'model_task': 'Same full-interaction referent set selection and full-scene candidate mapping as parent; only full-field rendering varies.',
        'layout': 'Fixed 2x2 complete-field canvas. Foveation changes local RGB sharpness only; no crop, zoom, boxes, projected anchors, or gaze-conditioned candidate changes.',
        'conditions_definition': {
            'RGB': 'Original RGB, no added attention panels.',
            'FOVEATION': 'Full-field RGB with multi-focus sharpness field from two-second measured gaze-duration history; attention panels black.',
            'HEATMAP': 'Original RGB with the same three actual full-field temporal attention maps.',
            'FOVEATION_HEATMAP': 'Foveated full-field RGB with the same three actual temporal maps.'},
        'foveation_algorithm': {'version': FOVEATION_VERSION, 'source_maps': 'Frozen causal three-bin 2s attention volume in each frame camera.',
            'multiple_focus_policy': 'Combine density modes into a spatially varying detail field; no single-point or referent-selected center.',
            'sparse_support_policy': 'Scale aperture by measured duration, fallback exactly to RGB when clock unavailable or maps empty.',
            'field_policy': 'Retain original full frame and fixed composite pixel dimensions; vary sharpness with multi-scale local detail.'},
        'evaluation_policy': 'Run all visual inferences and freeze all responses before accessing original referent annotations. Original evaluator and candidate mapping reused; paired event-level set metrics.',
        'source_sha256': {name: sha(ROOT / name) for name in source_files},
        'gt_loaded': False, 'anchors_projected': False, 'object_crops': False,
        'answer_conditioned_candidates': False,
        'known_limits': list(parent_manifest.get('known_limits', [])) + [
            'Fixed 24-event historical cohort; not an unseen validation sample.',
            'Foveation is a rendering pilot: its total spatial history is pooled across bins, so the RGB sharpness cue does not encode temporal order.',
            'Visual quality changes may remove useful peripheral detail; all full-field pixels remain but not equally legible.',
            'The control battery estimates foveation and heat-map effects with two main effects and their interaction.']}
    save(output / 'manifest.json', manifest)
    save(output / 'protocol_lock.json', {'manifest_sha256': sha(output / 'manifest.json'),
        'before_any_inference': True, 'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'primary_metrics': ['Exact-set', 'micro-F1'],
        'no_gt_annotation_reads_before_frozen_predictions': True})
    save(output / 'foveation_render_audit.json', {'frames': foveation_rows,
        'foveated_frames': sum(row.get('foveation_enabled', False) for row in foveation_rows),
        'rgb_fallback_frames': sum(row.get('status') == 'RGB_FALLBACK' for row in foveation_rows),
        'not_applied_frames': sum(row.get('status') == 'NOT_APPLIED' for row in foveation_rows)})
    for name in source_files:
        dest = output / 'source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    print('FROZEN', len(events), 'interactions', len(requests), 'conditions',
        manifest['planned_unique_model_calls'], 'unique model calls', flush=True)


def infer(output, limit=None):
    manifest = read(output / 'manifest.json')
    assert sha(output / 'manifest.json') == read(output / 'protocol_lock.json')['manifest_sha256']
    backend = QwenBackend(output / 'model_cache', model=manifest['model'], local_only=True,
        namespace=FOVEATION_VERSION, trace_visual_inputs=True,
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
        parsed = backend.ask_video(request['id'], 'full_referent_set_foveation', event['prompt'], paths,
            metadata, max_new_tokens=manifest['max_new_tokens'], video_path=event['video_path'],
            video_time_range=[event['frames'][0]['time_seconds'], event['frames'][-1]['time_seconds']])
        raw = read(backend.requests[-1])
        audit = raw['processor_audit']
        checked = validate_prediction(parsed, [row['candidate_id'] for row in event['scene_candidate_table']],
            {group['view_id'] for group in event['qwen_temporal_groups']}, event['instruction'])
        for alias in aliases[request['id']]:
            save(output / 'responses' / f"{alias['id']}.json", {**alias,
                'event': event['event'], 'scene_key': event['scene_key'], 'row_index': event['row_index'],
                'parsed': parsed, 'validation': checked, 'request_path': backend.requests[-1],
                'processor_audit': audit,
                'vision_verified': audit['visual_tensor_trace']['input_consumption_verified'],
                'seconds': time.monotonic() - start, 'exact_input_reused': alias['id'] != request['id']})
        print('RESULT', request['id'], checked['schema_valid'], checked['selected_object_names'],
            'tokens', audit['expanded_video_token_count'], 'seconds', round(time.monotonic()-start, 2), flush=True)
        save(output / 'progress.json', {'completed_condition_results': len(list((output / 'responses').glob('*.json'))),
            'planned_condition_results': len(manifest['requests']), 'planned_unique_model_calls': len(canonical),
            'last': request['id'], 'gt_loaded': False})
        completed += 1
        if limit and completed >= limit:
            break


def micro_f1(counts):
    tp, fp, fn = np.asarray(counts, dtype=float).sum(axis=0)
    return float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.


def paired_micro_interval(rows, baseline, treatment, seed=20261106, replicates=5000):
    arrays = {}
    for condition in (baseline, treatment):
        arrays[condition] = np.array([[row['metrics'][condition][key] for key in ('tp', 'fp', 'fn')]
            for row in rows], dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(rows), size=(replicates, len(rows)))
    scores = {}
    for condition, array in arrays.items():
        totals = array[indices].sum(axis=1)
        denominator = 2 * totals[:, 0] + totals[:, 1] + totals[:, 2]
        scores[condition] = np.divide(2 * totals[:, 0], denominator,
            out=np.zeros(replicates), where=denominator > 0)
    return {'events': len(rows), 'gain': micro_f1(arrays[treatment]) - micro_f1(arrays[baseline]),
        'interval_95': np.quantile(scores[treatment] - scores[baseline], [0.025, 0.975]).tolist(),
        'replicates': replicates, 'resampling_unit': 'paired interaction'}


def evaluate(output):
    manifest = read(output / 'manifest.json')
    assert sha(output / 'manifest.json') == read(output / 'protocol_lock.json')['manifest_sha256']
    responses = {row['id']: read(output / 'responses' / f"{row['id']}.json") for row in manifest['requests']}
    if len(responses) != len(manifest['requests']):
        raise ValueError('Every condition response must be frozen before evaluation')
    save(output / 'predictions_frozen.json', {'manifest_sha256': sha(output / 'manifest.json'),
        'before_gt_evaluation': True, 'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'response_sha256': {key: sha(output / 'responses' / f'{key}.json') for key in responses}})
    from MultiAgent.baseline import module, scene_paths
    evaluator = module('eval')
    gt_by_scene = {}
    for scene in sorted({event['scene_key'] for event in manifest['events']}):
        aliases, _ = evaluator.load_anchor_alias_map(scene_paths(scene)['anchor'])
        rows = evaluator.load_gt_rows(scene_paths(scene)['gt'], aliases)
        gt_by_scene[scene] = (aliases, rows)
    result = {'conditions': {}, 'events': [], 'source_manifest_sha256': sha(output / 'manifest.json'),
        'gt_loaded_only_after_frozen_predictions': True, 'historical_cohort_only': True,
        'all_schema_valid': all(row['validation']['schema_valid'] for row in responses.values()),
        'all_vision_verified': all(row['vision_verified'] for row in responses.values()),
        'expanded_video_tokens': sorted({row['processor_audit']['expanded_video_token_count'] for row in responses.values()}),
        'processor_effective_frame_sizes': sorted({tuple(row['processor_audit']['processor_effective_frame_size_px']) for row in responses.values()})}
    gt_by_event = {}
    for condition in CONDITIONS:
        records = []
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
            records.append({'id': event['id'], 'gt': gt['mapped_referents'], 'prediction': prediction})
            original_rows.extend(evaluator.build_eval_rows([destination], gt_rows, aliases))
            original_gt.append(gt)
        summary = evaluator.summarize(original_rows, original_gt)
        save(output / 'evaluation' / condition / 'original_summary.json', summary)
        evaluator.write_eval_csv(output / 'evaluation' / condition / 'evaluation.csv', original_rows)
        result['conditions'][condition] = {'set_metrics': paired_set_metrics([row for row in records if row['gt']]),
            'original_evaluator': summary,
            'schema_failures': sum(not responses[event['id']+'_'+condition]['validation']['schema_valid']
                for event in manifest['events'])}
    for event in manifest['events']:
        gt = set(gt_by_event[event['id']])
        predictions = {condition: responses[event['id']+'_'+condition]['validation']['selected_object_names']
            for condition in CONDITIONS}
        metrics = {condition: {'exact': set(prediction) == gt,
            'tp': len(set(prediction) & gt), 'fp': len(set(prediction) - gt), 'fn': len(gt - set(prediction)),
            'f1': 2 * len(set(prediction) & gt) / (len(prediction) + len(gt))
                if len(prediction) + len(gt) else 0.} for condition, prediction in predictions.items()}
        result['events'].append({'id': event['id'], 'event': event['event'], 'instruction': event['instruction'],
            'gt': sorted(gt), 'predictions': predictions, 'metrics': metrics,
            'heat_supported_frames': event['clock_available_frames']})
    contrasts = {}
    for baseline, treatment, label in [
        ('RGB', 'FOVEATION', 'foveation_without_heat'),
        ('HEATMAP', 'FOVEATION_HEATMAP', 'foveation_with_heat')]:
        rows = result['events']
        contrasts[label] = {
            'micro_f1': paired_micro_interval(rows, baseline, treatment),
            'exact_set_rate': cluster_paired_interval([{'cluster': row['id'],
                'baseline': float(row['metrics'][baseline]['exact']),
                'treatment': float(row['metrics'][treatment]['exact'])} for row in rows]),
            'mean_event_f1': cluster_paired_interval([{'cluster': row['id'],
                'baseline': float(row['metrics'][baseline]['f1']),
                'treatment': float(row['metrics'][treatment]['f1'])} for row in rows]),
            'helpful_interactions': sum(row['metrics'][treatment]['f1'] > row['metrics'][baseline]['f1'] for row in rows),
            'harmful_interactions': sum(row['metrics'][treatment]['f1'] < row['metrics'][baseline]['f1'] for row in rows)}
    result['paired_contrasts'] = contrasts
    result['factorial_foveation_average_micro_f1'] = {
        'foveation_main_effect': (contrasts['foveation_without_heat']['micro_f1']['gain']
            + contrasts['foveation_with_heat']['micro_f1']['gain']) / 2,
        'heat_foveation_interaction': (contrasts['foveation_with_heat']['micro_f1']['gain']
            - contrasts['foveation_without_heat']['micro_f1']['gain'])}
    result['support_strata'] = {}
    for label, subset in [('some_supported_heat', [row for row in result['events'] if row['heat_supported_frames'] > 0]),
                          ('rgb_fallback_only', [row for row in result['events'] if row['heat_supported_frames'] == 0])]:
        result['support_strata'][label] = {condition: paired_set_metrics([{'gt': row['gt'],
            'prediction': row['predictions'][condition]} for row in subset]) for condition in CONDITIONS}
    save(output / 'analysis.json', result)
    print(json.dumps({condition: summary['set_metrics'] for condition, summary in result['conditions'].items()},
        ensure_ascii=False, indent=2))


def build_viewer(output):
    manifest = read(output / 'manifest.json')
    analysis = read(output / 'analysis.json') if (output / 'analysis.json').exists() else None
    responses = {path.stem: read(path) for path in (output / 'responses').glob('*.json')}
    compact = [{'id': event['id'], 'event': event['event'], 'instruction': event['instruction'],
        'candidate_count': len(event['scene_candidate_table']), 'heat_supported_frames': event['clock_available_frames'],
        'frames': [{'frame_id': frame['frame_id'], 'time_seconds': frame['time_seconds'],
            'images': frame['images_foveation_pilot'], 'source_size': frame['source_size'],
            'foveation': frame['foveation']} for frame in event['frames']],
        'temporal_groups': event['qwen_temporal_groups']} for event in manifest['events']]
    response_data = {key: {'parsed': val['parsed'], 'validation': val['validation'],
        'effective_size': val['processor_audit']['processor_effective_frame_size_px'],
        'visual_tokens': val['processor_audit']['expanded_video_token_count']}
        for key, val in responses.items()}
    data = json.dumps({'events': compact, 'conditions': CONDITIONS, 'responses': response_data,
        'analysis': analysis}, ensure_ascii=False).replace('</', '<\\/')
    html = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Foveation pilot</title>
<style>body{font:16px system-ui;line-height:1.55;background:#f1f4f7;margin:18px}header{position:sticky;top:0;background:#f1f4f7;padding:10px;z-index:1}select{font:inherit;padding:4px;margin:5px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(530px,1fr));gap:14px}figure{background:white;padding:10px;margin:0}img{width:100%}#result{padding:12px;background:white;white-space:pre-wrap}</style>
<header><h2>全交互 referent set：foveation × heatmap</h2><select id="event"></select><select id="condition"></select><div id="instruction"></div></header>
<p>四组共享原始指令、全场景候选、相机、时间帧与提示词。RGB 面板始终覆盖完整画面；foveation 仅改变多处 gaze-history 区域的清晰度。支持不足时逐帧回退原 RGB。评分在所有模型输出冻结后使用原始 referent 集合评价。</p><div id="result"></div><main></main>
<script>const data=__DATA__,q=document.querySelector('#event'),c=document.querySelector('#condition');data.events.forEach(e=>{let o=document.createElement('option');o.value=e.id;o.textContent=e.event;q.append(o)});data.conditions.forEach(x=>{let o=document.createElement('option');o.value=x;o.textContent=x;c.append(o)});function render(){let e=data.events.find(e=>e.id===q.value);document.querySelector('#instruction').textContent=e.instruction+' | 全场景候选 '+e.candidate_count+' | 热图支持 '+e.heat_supported_frames+'/12帧';let r=data.responses[e.id+'_'+c.value],sc=data.analysis?.events.find(x=>x.id===e.id);document.querySelector('#result').textContent=(r?'输出 '+JSON.stringify(r.parsed)+' | schema '+r.validation.schema_valid+' | '+r.visual_tokens+'视觉token | 实际组合帧 '+r.effective_size.join('×'):'尚未运行')+(sc?'\nGT '+JSON.stringify(sc.gt)+'\n各条件评分 '+JSON.stringify(sc.metrics):'');let main=document.querySelector('main');main.replaceChildren();e.frames.forEach(f=>{let fig=document.createElement('figure'),im=document.createElement('img'),t=document.createElement('figcaption');im.src=f.images[c.value].path;im.loading='lazy';t.textContent=f.frame_id+' | PTS '+f.time_seconds.toFixed(6)+'s | '+JSON.stringify(f.foveation);fig.append(im,t);main.append(fig)})}q.onchange=render;c.onchange=render;render();</script></html>'''
    (output / 'index.html').write_text(html.replace('__DATA__', data))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'infer', 'evaluate', 'viewer'])
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    output = args.output.resolve()
    {'prepare': prepare, 'infer': lambda path: infer(path, args.limit),
     'evaluate': evaluate, 'viewer': build_viewer}[args.action](output)
