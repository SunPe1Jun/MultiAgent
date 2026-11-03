"""Check whether Qwen can read the gaze cue from rendered video pixels."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

from .local_attention_probe import ROOT, WORKSPACE, read_json, save, sha_file
from .qwen_backend import QwenBackend


MODEL_DEFAULT = WORKSPACE / 'Qwen3-VL-8B-Instruct'
CONDITIONS = ['R', 'H_local', 'F_local']
PROMPT = '''You are auditing only which visual attention encoding is perceptible in a short ordered video. Do not identify any objects, do not resolve a referring expression, and do not infer gaze from scene semantics.

Encoding legend: a red/orange translucent region is the heatmap and marks accumulated gaze history; relatively sharper pixels surrounded by blur are foveation and mark greater gaze emphasis. These are uncertain rendering cues, not target labels. Raw RGB has no added cue. If a cue is not visually detectable, report none rather than guessing.

The video is divided into the listed chronological Qwen temporal groups. Each group combines two actual source frames. For each group, inspect its pixels and report the visually detectable cue type and the normalized [u,v] center of the colored/clear region, with origin at the top-left. For foveation, estimate the center of the clearest gaze-emphasized patch, not the blurred scene center. If no cue is detectable or its center cannot be localized, use none/unclear and null. Do not copy coordinates from the metadata; it contains only view IDs and times.

Return only JSON: {"attention_trace":[{"view_id":"G001","cue_type":"none|heatmap|foveation|unclear","center_uv":null,"confidence":0.0,"visual_basis":"brief pixel-based description"}]}. Return exactly one row per supplied view ID, in order. Never fabricate a cue or coordinate.'''


def trace_by_view(parsed, expected):
    trace = parsed.get('attention_trace', []) if isinstance(parsed, dict) else []
    by_id = {row.get('view_id'): row for row in trace if isinstance(row, dict)}
    checked = []
    valid_types = {'none', 'heatmap', 'foveation', 'unclear'}
    for view_id, expected_center in expected.items():
        row = dict(by_id.get(view_id) or {})
        cue_type = row.get('cue_type')
        center = row.get('center_uv')
        if cue_type not in valid_types:
            cue_type = 'invalid_or_missing'
        valid_center = None
        if isinstance(center, (list, tuple)) and len(center) == 2:
            try:
                point = [float(center[0]), float(center[1])]
                if all(math.isfinite(v) and 0 <= v <= 1 for v in point):
                    valid_center = point
            except (TypeError, ValueError):
                pass
        error = (float(math.dist(expected_center, valid_center))
                 if expected_center is not None and valid_center is not None else None)
        row.update({'view_id': view_id, 'cue_type': cue_type,
                    'center_uv': valid_center,
                    'expected_gaze_history_center_uv_audit_only': expected_center,
                    'center_error_uv_distance': error})
        checked.append(row)
    detected = sum(row['cue_type'] in ('heatmap', 'foveation') for row in checked)
    localizable = sum(row['center_error_uv_distance'] is not None for row in checked)
    return {'attention_trace': checked, 'cue_detected_count': detected,
            'expected_cue_count': sum(center is not None for center in expected.values()),
            'localized_center_count': localizable,
            'mean_center_error_uv_distance': (
                sum(row['center_error_uv_distance'] for row in checked
                    if row['center_error_uv_distance'] is not None) / localizable
                if localizable else None)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-run', default='runs/local_attention_guidance_probe_20261022_pilot4')
    parser.add_argument('--output-dir', default='runs/local_attention_cue_validation_20261022_pilot4')
    parser.add_argument('--cache-dir', default='runs/local_attention_cue_validation_20261022_cache')
    parser.add_argument('--model', default=str(MODEL_DEFAULT))
    parser.add_argument('--namespace', default='local_attention_cue_validation_qwen8b_20261022')
    parser.add_argument('--video-max-pixels', type=int, default=8_388_608)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()

    input_root = Path(args.input_run)
    if not input_root.is_absolute():
        input_root = (ROOT / input_root).resolve()
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = (ROOT / output).resolve()
    config = read_json(input_root / 'config.json')
    manifest = read_json(input_root / 'manifest.json')
    selected = manifest['samples'][:args.limit] if args.limit else manifest['samples']
    save(output / 'config.json', {
        'version': 'local_attention_cue_validation_20261022_v1',
        'input_run': str(input_root), 'source_sample_list': selected,
        'model': str(Path(args.model).resolve()), 'namespace': args.namespace,
        'cache_dir': str(Path(args.cache_dir).resolve()),
        'source_frames_per_request': config['source_frame_count'],
        'conditions': CONDITIONS,
        'prompt_scope': 'cue perception only; no mention candidates or object identity',
        'scope': 'original heatmap/foveation perception; no circle inputs',
        'offline_full_interaction': True,
        'ground_truth_loaded': False,
    })
    backend = QwenBackend(args.cache_dir, args.model, namespace=args.namespace,
                          local_only=True, video_max_pixels=args.video_max_pixels,
                          trace_visual_inputs=True)
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(20261022)
    events, failures = [], []

    for index, sample in enumerate(selected, 1):
        scene, event = sample['scene_key'], sample['event_id']
        print(f'[{index}/{len(selected)}] {scene}/{event} cue-only validation', flush=True)
        sample_root = input_root / 'samples' / scene / event
        input_audit = read_json(sample_root / 'frozen_input_audit.json')
        out_sample = output / 'samples' / scene / event
        try:
            variants = {condition: {} for condition in CONDITIONS}
            for frame in input_audit['frames']:
                frame_id = frame['frame_id']
                expected = input_audit['attention_maps'][frame_id]['gaze']['expected_attention_center_uv']
                variants['R'][frame_id] = str(sample_root / 'inputs' / frame_id / 'R.png')
                variants['H_local'][frame_id] = str(sample_root / 'inputs' / frame_id / 'H_local.png')
                variants['F_local'][frame_id] = str(sample_root / 'inputs' / frame_id / 'F_local.png')
            save(out_sample / 'frozen_input_audit.json', input_audit)

            metadata = input_audit['frames']
            groups = input_audit['temporal_groups']
            expected = {
                group['view_id']: (
                    [sum(input_audit['attention_maps'][frame_id]['gaze']['expected_attention_center_uv'][axis]
                         for frame_id in group['frame_ids']
                         if input_audit['attention_maps'][frame_id]['gaze']['expected_attention_center_uv'] is not None)
                     / sum(input_audit['attention_maps'][frame_id]['gaze']['expected_attention_center_uv'] is not None
                           for frame_id in group['frame_ids']) for axis in (0, 1)]
                    if any(input_audit['attention_maps'][frame_id]['gaze']['expected_attention_center_uv'] is not None
                           for frame_id in group['frame_ids']) else None)
                for group in groups}
            calls = {}
            for condition in CONDITIONS:
                paths = [variants[condition][frame['frame_id']] for frame in metadata]
                request_key = f'{scene}/{event}/attention_only'
                stage = f'cue_only_{condition}'
                started = time.monotonic()
                parsed = backend.ask_video(
                    request_key, stage, PROMPT + '\n\nTemporal groups: ' +
                    json.dumps(groups, ensure_ascii=False), paths, metadata,
                    max_new_tokens=512, video_path=input_audit['source_video'],
                    video_time_range=[metadata[0]['time_seconds'], metadata[-1]['time_seconds']],
                    video_max_pixels=args.video_max_pixels)
                request_path = Path(backend.requests[-1])
                record = read_json(request_path)
                result = trace_by_view(parsed, expected)
                result.update({
                    'condition': condition, 'raw_response': record.get('response'),
                    'request_path': str(request_path),
                    'request_sha256': hashlib.sha256(request_path.read_bytes()).hexdigest(),
                    'input_frame_records': record.get('images'),
                    'processor_audit': record.get('processor_audit'),
                    'visual_tensor_trace': record.get('processor_audit', {}).get('visual_tensor_trace'),
                    'elapsed_seconds': record.get('elapsed_seconds'),
                    'wall_seconds_this_invocation': time.monotonic()-started,
                })
                calls[condition] = result
                save(out_sample / 'predictions' / f'cue_only_{condition}.json', result)
                trace = [(row['view_id'], row['cue_type'], row['center_uv'])
                         for row in result['attention_trace']]
                print(f'  {condition}: detected={result["cue_detected_count"]}/'
                      f'{result["expected_cue_count"]} centers={result["localized_center_count"]} '
                      f'trace={trace}', flush=True)
            events.append({'scene_key': scene, 'event_id': event,
                           'source_frame_count': len(metadata),
                           'qwen_temporal_groups': len(groups), 'conditions': calls})
        except Exception as exc:
            failure = {'scene_key': scene, 'event_id': event,
                       'error': f'{type(exc).__name__}: {exc}'}
            failures.append(failure)
            save(out_sample / 'failure.json', failure)
            print(f'  FAILED: {failure["error"]}', flush=True)

    save(output / 'summary.json', {
        'status': 'COMPLETE' if not failures else 'COMPLETE_WITH_FAILURES',
        'planned_events': len(selected), 'completed_events': len(events),
        'failed_events': len(failures), 'conditions': CONDITIONS,
        'expected_model_calls': len(selected) * len(CONDITIONS),
        'actual_calls': len(backend.call_records), 'events': events,
        'failures': failures, 'call_audit': backend.call_records,
        'ground_truth_loaded': False, 'candidate_identity_task': False,
        'fusion': 'NOT_RUN',
    })
    print(json.dumps({'completed': len(events), 'failed': len(failures),
                      'calls': len(backend.call_records)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
