#!/usr/bin/env python3
"""Paired original-2D full RGB vs heatmap-driven foveation with mirrored control."""
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
OUTPUT = ROOT / '.local-sync/2d_heatmap_foveation_recovery_20261109'
CONDITIONS = ('RGB', 'HEATMAP_FOVEATION', 'MIRRORED_FOVEATION')


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
    # Identical original-RGB 2D task prompt across all three visual conditions.
    return temporal_prompt(event)


def prepare(out):
    from MultiAgent.attention_volume import build_volume
    from MultiAgent.camera_geometry import CameraPoseTimeline, aligned_frame_geometry
    from MultiAgent.context import vrtriref_sample_to_agent_context
    from MultiAgent.foveation_attention import foveate_from_history
    from MultiAgent.frame_pose_alignment import source_frame_pose
    from MultiAgent.gaze_evidence import gaze_records
    from MultiAgent.head_motion_diagnostics import camera_records, probe_pts, verify_saved_frame_pts
    if (out / 'input_manifest.json').exists():
        raise ValueError('Refuse to overwrite frozen experiment')
    verify_lock(MAIN); verify_lock(PROMPT)
    source = json.loads((PROMPT / 'input_manifest.json').read_text())
    prior = ROOT / '.local-sync/referent_rgb_expansion32_20261109_repaired'
    original = ROOT / '.local-sync/referent_rgb_expansion32_20261109'
    rows = {(r['scene'], int(r['row_index']), r['panel_id']): r for r in read_csv(MAIN / 'rebuilt_manifest_all.csv') if r['panel_id']}
    events, audits, files = [], [], []
    for old in source['events']:
        event = json.loads(json.dumps(old)); scene = event['scene_key']; idx = event['row_index']
        cached = prior / 'inputs' / scene / f'{idx}_row.json'
        contract_path = original / 'pose_alignment' / scene / str(idx) / 'frame_pose_contract.json'
        contract = json.loads(contract_path.read_text())
        ctx = vrtriref_sample_to_agent_context(json.loads(cached.read_text())['row'], load_ground_truth=False)
        if contract['source_hashes'] != {'video_sha256': sha_file(Path(ctx.video_path)), 'telemetry_sha256': sha_file(Path(ctx.json_path))}:
            raise ValueError('Video or gaze telemetry changed')
        timeline = CameraPoseTimeline(camera_records(ctx)); records = gaze_records(ctx); pts, _ = probe_pts(ctx.video_path, 'ffprobe')
        files.extend([cached, contract_path])
        for panel in event['panels']:
            row = rows[(scene, idx, panel['frame_id'])]; requested = float(row['video_frame_time'])
            source_path = Path(panel['source_path']); source_hash = sha_file(source_path)
            if source_hash != panel['source_sha256']:
                raise ValueError('Source panel changed')
            verified = verify_saved_frame_pts(ctx.video_path, {'video_time_seconds': requested, 'source_sha256': source_hash}, pts, 'ffmpeg')
            camera = source_frame_pose(timeline, verified, contract['clock'], allow_candidate=True)
            geometry = aligned_frame_geometry({'frame_time': verified, 'source_sha256': source_hash, 'attention_camera': camera},
                                              source_hash, requested, float(row['video_time_offset_seconds'] or 0))
            rgb_meta = panel['inputs']['MAIN_RGB_768']; rgb_path = Path(rgb_meta['path'])
            if sha_file(rgb_path) != rgb_meta['sha256']:
                raise ValueError('Frozen RGB changed')
            with Image.open(rgb_path) as opened:
                rgb = opened.convert('RGB')
            maps, stats = build_volume(records, query=geometry['telemetry_time_seconds'], pose=geometry['reference_pose'],
                image_size=rgb.size, clock_supported=geometry['clock_available'], history=2., bins=3, sigma_degrees=2.)
            reference_pose = geometry['reference_pose']; fov = reference_pose['fov_vertical_degrees'] if reference_pose else 60.
            folder = out / 'inputs' / event['id'] / panel['frame_id']; folder.mkdir(parents=True, exist_ok=True)
            map_path = folder / 'gaze_history_heatmap.npz'; np.savez_compressed(map_path, maps=maps)
            inputs = {'RGB': rgb_meta}; conditions_audit = {}
            for group, applied in [('HEATMAP_FOVEATION', maps), ('MIRRORED_FOVEATION', maps[:, :, ::-1].copy())]:
                rendered, meta = foveate_from_history(rgb, applied, clock_supported=geometry['clock_available'], vertical_fov_degrees=fov)
                target = folder / (group + '.png'); rendered.save(target)
                changed = np.any(np.asarray(rendered) != np.asarray(rgb), axis=2)
                inputs[group] = {'path': str(target), 'sha256': sha_file(target), 'size': list(rgb.size)}
                conditions_audit[group] = {'foveation': meta, 'changed_pixels': int(changed.sum()), 'image_pixels': int(changed.size)}
                if meta['status'] == 'RGB_FALLBACK' and np.any(changed):
                    raise ValueError('Unsupported gaze must fall back exactly to RGB')
            panel['region_inputs'] = inputs
            audits.append({'event': event['id'], 'frame_id': panel['frame_id'], 'source_sha256': source_hash,
                'video_time': geometry['video_time_seconds'], 'clock_supported': geometry['clock_available'],
                'heatmap_path': str(map_path), 'heatmap_sha256': sha_file(map_path),
                'history_seconds': 2., 'phase_support_seconds': [b['duration'] for b in stats['bins']],
                'fov_degrees': fov, 'conditions': conditions_audit})
            print('FOVEATION_INPUT',event['id'],panel['frame_id'],conditions_audit['HEATMAP_FOVEATION']['foveation']['status'],flush=True)
        events.append(event)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / 'input_manifest.json', {'version': '2d_heatmap_foveation_recovery_v1', 'events': events,
        'conditions': list(CONDITIONS), 'sample_status': '10 previously seen development interactions; not held out',
        'method': 'Existing causal 2-second gaze duration heatmap drives full-field multi-focus foveation, not delivered as additional heatmap image. Three conditions have identical 2D referent task and RGB frame inputs except pixel rendering.',
        'heatmap_input_policy': 'Gaze telemetry and camera/clock alignment only; never GT referent or target proposals.',
        'control': 'RGB vs actual heatmap foveation vs horizontal mirror of the same heatmap; invalid clock/gaze exactly falls back to RGB.',
        'task': 'Original Experiment 2 P1/P2/P3 and 2D panel evaluator; frozen per-mention tasks and original-P-panel temporal priors.',
        'prior_expected_count': 'Current per-mention protocol does not use original GT-derived expected_count; published original baseline is historical reference.'})
    save_json(out / 'input_image_audit.json', {'panels': audits, 'no_gt_referent_input': True,
        'same_prompt_and_frame_ids_for_all_conditions': True, 'heatmap_not_separately_shown_to_model': True})
    deps = [Path(__file__).resolve(), out / 'input_manifest.json', out / 'input_image_audit.json',
            MAIN / 'input_manifest.json', PROMPT / 'input_manifest.json', MAIN / 'rebuilt_manifest_all.csv',
            ROOT / 'tools/probe_referent_temporal_prompt_v3.py', ROOT / 'MultiAgent/qwen_backend.py',
            ROOT / 'MultiAgent/projected2d_main_image.py', ROOT / 'MultiAgent/attention_volume.py',
            ROOT / 'MultiAgent/foveation_attention.py', ROOT / 'MultiAgent/camera_geometry.py',
            ROOT / 'MultiAgent/frame_pose_alignment.py', ROOT / 'MultiAgent/gaze_evidence.py',
            ROOT / 'tools/run_full_frame_grounding_v4.py', SUBSET,
            ROOT.parent / 'TriModal-Referring/exam2/evaluate_2d_point_grounding.py', MODEL / 'config.json'] + files
    save_json(out / 'protocol_lock.json', {'files': {str(p): sha_file(p) for p in deps}})
    print('PREPARED_FOVEATION', len(events), 'events', len(audits), 'panels', flush=True)


def infer(out):
    from MultiAgent.qwen_backend import QwenBackend
    from tools.run_full_frame_grounding_v4 import clean_visual_trace
    verify_lock(out)
    manifest = json.loads((out / 'input_manifest.json').read_text())
    backend = QwenBackend(out / 'model_cache', str(MODEL), local_only=True,
                          namespace='2d_heatmap_foveation_recovery_v1', trace_visual_inputs=True)
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
            print('FOVEATION', index, '/', len(manifest['events']), event['id'], condition, response['status'], flush=True)


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
                    'Gaze calibration is not independently validated and may be noisy.',
                    'Heatmap is a past gaze-density estimate, not a target label; foveation cannot recover original-source details lost by downsampling.']})
    print('REPORTED', {k: {'frames': v['frames'], 'joint100': v['metrics']['joint_100_f1']} for k, v in result.items()}, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'infer', 'report'])
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    {'prepare': prepare, 'infer': infer, 'report': report}[args.stage](args.output.resolve())


if __name__ == '__main__':
    main()
