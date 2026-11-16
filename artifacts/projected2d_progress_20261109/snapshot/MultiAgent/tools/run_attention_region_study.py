#!/usr/bin/env python3
"""Seam-aware visual attention proposals and an instruction/RGB screening pilot.

All samples are previously inspected regression cases. Model conditions share
exactly the same gaze-derived candidate crops: this measures the incremental
heat display, NOT the value of gaze versus a gaze-free video baseline.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation
from MultiAgent.attention_regions import (VERSION, extract_regions, mass_envelope,
    spherical_components, region_observations, representative_observation, angular_coverage)
from MultiAgent.camera_geometry import interpolate_pose
from MultiAgent.context import load_contexts
from MultiAgent.gaze_evidence import gaze_records, project_gaze
from MultiAgent.heatmap_temporal_pilot import save, sha
from MultiAgent.local_video_visual import probe_frames
from MultiAgent.perspective_attention import face_grid, sphere_indices, cube_owner
from MultiAgent.perspective_attention_study import ROOT, SOURCE, OUTPUT as PREVIOUS, read
from MultiAgent.qwen_backend import QwenBackend
from MultiAgent.spherical_attention import solid_angle_weights, spherical_grid, angles_to_directions
from MultiAgent.spherical_attention_study import AUDIT

OUTPUT = ROOT / '.local-sync/attention_regions_20261106'
CONDITIONS = ['TRUE_HEAT', 'NO_HEAT', 'REVERSE_TIME']
PHASES = ['EARLY', 'MIDDLE', 'LATE']
PROMPT = '''Inspect these historical RGB attention-region proposals for the original instruction below.
Each printed region ID has an actual source RGB CONTEXT (left), a DETAIL crop (right), then THREE registered graphical HEAT panels EARLY / MIDDLE / LATE. All three heat panels share that candidate's native camera orientation and crop, so their spatial positions correspond. Source phase/time is printed per card. Missing heat means no added attention information, not evidence against an object.
The script already extracted spatial proposals and bound source times. Do not rank heat brightness or report gaze coordinates. Heat shows where the recorder looked over time; it is soft attention evidence, NOT an object label, visibility proof, or a required target. The panels may be counterfactual controls. A crop can contain multiple objects and repeated cards may show the same object.
Use RGB and the instruction to screen EVERY displayed ID: keep = a visibly plausible mentioned object or spatial reference; review = identity/relevance remains unclear; reject = visibly unrelated. Retain plausible alternatives. Never reject solely for absent or weak heat. Do not invent hidden objects from language. Describe only what is visible in a few words, using unclear when necessary. This is evidence screening, NOT final object identity or final referents. Other pages may contain further candidates.
Return JSON only: {"decisions":[{"id":"R01","decision":"review","visual":"unclear"}]}. Exactly one entry for each ID printed in this image; no other IDs. decision must be keep, review or reject.
ORIGINAL INSTRUCTION: __INSTRUCTION__'''


def native_angular_lookup(pose, canonical_pose, size):
    w, h = size
    yy, xx = np.meshgrid(np.arange(h) + .5, np.arange(w) + .5, indexing='ij')
    focal = h / (2 * math.tan(math.radians(pose['fov_vertical_degrees']) / 2))
    rays = np.stack([(xx - w / 2) / focal, -(yy - h / 2) / focal, np.ones_like(xx)], axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    world = Rotation.from_quat(pose['rotation_xyzw']).apply(rays.reshape(-1, 3))
    return Rotation.from_quat(canonical_pose['rotation_xyzw']).inv().apply(world).reshape(rays.shape)


def component_crop(mask, projected_xy, size):
    """Include all visible envelope pixels; keep at least 18% of each dimension."""
    w, h = size
    y, x = np.nonzero(mask)
    if not len(x):
        return None
    lo = np.array([x.min(), y.min()], float)
    hi = np.array([x.max() + 1, y.max() + 1], float)
    if projected_xy is not None:
        lo = np.minimum(lo, projected_xy)
        hi = np.maximum(hi, projected_xy)
    extent = np.maximum(hi - lo + np.array([w, h]) * .08, np.array([w, h]) * .18)
    center = (lo + hi) / 2
    result = [max(0, math.floor(center[0] - extent[0] / 2)),
              max(0, math.floor(center[1] - extent[1] / 2)),
              min(w, math.ceil(center[0] + extent[0] / 2)),
              min(h, math.ceil(center[1] + extent[1] / 2))]
    return result


def heat_tile(values, global_peak):
    base = np.full((*values.shape, 3), 20., np.float32)
    if global_peak > 0:
        alpha = (.95 * np.sqrt(np.clip(values / global_peak, 0, 1)))[..., None]
        base = base * (1 - alpha) + np.array([255, 85, 25]) * alpha
    return Image.fromarray(base.clip(0, 255).astype(np.uint8))


def font(size):
    return ImageFont.load_default(size=size)


def render_page(output, candidates, condition):
    image = Image.new('RGB', (1008, 1008), '#15212b')
    draw = ImageDraw.Draw(image)
    draw.text((12, 8), 'RGB EVIDENCE + REGISTERED ATTENTION HISTORY', fill='white', font=font(18))
    for i, r in enumerate(candidates):
        x = 12 + i % 3 * 328
        y = 42 + i // 3 * 470
        draw.text((x + 4, y), f"{r['id']} | {PHASES[r['phase']]} | {r['source_frame']['pts_seconds']:.2f}s",
                  fill='white', font=font(16))
        for j, key in enumerate(['context_image', 'crop_image']):
            with Image.open(output / r[key]['path']) as source:
                thumb = source.convert('RGB').resize((148, 148), Image.Resampling.LANCZOS)
            image.paste(thumb, (x + 4 + j * 160, y + 48))
            draw.text((x + 4 + j * 160, y + 28), ['CONTEXT', 'DETAIL'][j], fill='white', font=font(13))
        for k, name in enumerate(PHASES):
            source_phase = 2 - k if condition == 'REVERSE_TIME' else k
            tile_path = output / r['heat_images'][source_phase]['path']
            if condition == 'NO_HEAT':
                tile = Image.new('RGB', (96, 96), (20, 20, 20))
            else:
                with Image.open(tile_path) as source:
                    tile = source.convert('RGB').resize((96, 96), Image.Resampling.BILINEAR)
            draw.text((x + 4 + k * 104, y + 216), name, fill='white', font=font(13))
            image.paste(tile, (x + 4 + k * 104, y + 238))
            draw.rectangle((x + 4 + k * 104, y + 238, x + 99 + k * 104, y + 333), outline='#52616e')
        draw.text((x + 4, y + 354), 'Attention is soft evidence.', fill='#c7d4df', font=font(13))
        draw.text((x + 4, y + 376), 'Keep / review / reject using RGB.', fill='#c7d4df', font=font(13))
    draw.text((12, 982), 'All candidates retained across pages. Source frames are real; angular heat ignores parallax.',
              fill='white', font=font(12))
    return image


def seam_audit():
    previous = read(PREVIOUS / 'manifest.json')
    multi = read(PREVIOUS / 'multi_region/manifest.json')
    cases, same_face = [], []
    for event in previous['events']:
        mass = np.load(SOURCE / 'events' / event['event'] / 'volume.npz')['duration_maps']
        density = mass / solid_angle_weights(mass.shape[2], mass.shape[1])[None]
        tiles = np.load(PREVIOUS / 'events' / event['event'] / 'perspective_density.npz')['density']
        yg, pg = spherical_grid(mass.shape[2], mass.shape[1])
        owners = cube_owner(angles_to_directions(yg, pg))
        for phase in (0, 2):
            labels, n = spherical_components(density[phase] >= .5 * density[phase].max())
            components = []
            for c in range(1, n + 1):
                values = np.where(labels == c, density[phase], -1)
                y, x = np.unravel_index(values.argmax(), values.shape)
                components.append({'component': c, 'peak_face': int(owners[y, x]), 'peak_yx': [int(y), int(x)]})
            counts = Counter(c['peak_face'] for c in components)
            if any(count > 1 for count in counts.values()):
                same_face.append({'event': event['event'], 'phase': phase, 'half_peak_components': components})
            request = next(r for r in multi['requests'] if r['event_id'] == event['id'] and r['phase'] == phase and r['condition'] == 'TRUE_HEAT')
            if len(request['expected_top2']) != 2:
                continue
            peaks = []
            for name in request['expected_top2']:
                region = event['labels_by_region'].index(name)
                y, x = np.unravel_index(tiles[phase, region].argmax(), tiles.shape[-2:])
                ray = face_grid(event['selected_faces'][region], tiles.shape[-1])[y, x]
                sx, sy = sphere_indices(ray, mass.shape[2], mass.shape[1])
                peaks.append({'old_label': name, 'face': event['selected_faces'][region], 'ray': ray.tolist(),
                              'spherical_component': int(labels[sy, sx])})
            angle = math.degrees(math.acos(float(np.clip(np.dot(peaks[0]['ray'], peaks[1]['ray']), -1, 1))))
            cases.append({'event': event['event'], 'phase': phase, 'peaks': peaks, 'peak_separation_degrees': angle,
                          'same_connected_half_peak_region': peaks[0]['spherical_component'] == peaks[1]['spherical_component'] != 0,
                          'half_peak_component_count': n})
    return {'old_two_face_cases': cases, 'multiple_half_peak_components_in_one_face': same_face,
            'caveat': 'Connected regions and half-peak thresholds are engineering constructs, not object or referent labels.',
            'old_threshold_mismatch': 'Prompt compared brightness; expected sets compared raw density; sqrt rendering changes the ratio.'}


def prepare(output):
    if (output / 'manifest.json').exists():
        raise ValueError('Prepared inputs are immutable; use another output directory')
    output.mkdir(parents=True, exist_ok=True)
    source = read(SOURCE / 'manifest.json')
    selection = read(AUDIT / 'selection.json')
    keys = {e['event'] for e in source['events']}
    contexts = load_contexts({'samples': [s for s in selection['samples'] if f"{s['scene_key']}/{s['row_index']}" in keys]}, load_ground_truth=False)
    contexts = {f'{c.scene_key}/{c.row_index}': c for c in contexts}
    events, requests = [], []
    for event in source['events']:
        ctx = contexts[event['event']]
        contract = read(AUDIT / 'pose_alignment' / event['event'] / 'frame_pose_contract.json')
        if sha(ctx.video_path) != contract['source_hashes']['video_sha256'] or sha(ctx.json_path) != contract['source_hashes']['telemetry_sha256']:
            raise ValueError('Clock-fit source has changed')
        data_path = SOURCE / 'events' / event['event'] / 'volume.npz'
        maps = np.load(data_path)['duration_maps']
        labels, regions, audits = extract_regions(maps, .9)
        area = solid_angle_weights(maps.shape[2], maps.shape[1])
        density = maps / area[None]
        peak = float(density.max())
        directory = output / 'events' / event['event']
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / 'regions.npz', labels=labels, duration_maps=maps,
                            residual_maps=np.where(labels == 0, maps, 0))
        records = gaze_records(ctx)
        by_index = {r['source_index']: r for r in records}
        video_frames, stream = probe_frames(ctx.video_path)
        offset = event['frames'][-1]['geometry']['selected_clock_offset_seconds']
        entry = {'id': event['id'], 'event': event['event'], 'instruction': ctx.text,
                 'source_video_sha256': contract['source_hashes']['video_sha256'],
                 'source_telemetry_sha256': contract['source_hashes']['telemetry_sha256'],
                 'source_volume_path': str(data_path), 'source_volume_sha256': sha(data_path),
                 'source_clock_contract_path': str(AUDIT / 'pose_alignment' / event['event'] / 'frame_pose_contract.json'),
                 'source_clock_contract_sha256': sha(AUDIT / 'pose_alignment' / event['event'] / 'frame_pose_contract.json'),
                 'canonical_pose': event['canonical_pose'], 'clock_offset_seconds': offset,
                 'regions_npz': str((directory / 'regions.npz').relative_to(output)),
                 'regions_npz_sha256': sha(directory / 'regions.npz'), 'mass_audits': audits, 'regions': []}
        rng = random.Random('attention-region-v1:' + event['id'])
        names = [f'R{i + 1:02d}' for i in range(len(regions))]
        rng.shuffle(names)
        for region, name in zip(regions, names):
            phase = region['phase']
            b = event['volume_stats']['bins'][phase]
            observations = region_observations(b['observations'], event['canonical_pose'], labels[phase], region['component'])
            representative = representative_observation(observations, event['canonical_pose'], region)
            row = {**region, 'id': name, 'window': b['window'], 'observations': observations,
                   'observation_duration_seconds': sum(o['duration_seconds'] for o in observations),
                   'representative_observation': representative, 'status': 'NO_OBSERVATION_IN_COMPONENT'}
            entry['regions'].append(row)
            if representative is None:
                continue
            target = representative['gaze_time_seconds'] + offset
            available = [f for f in video_frames if f['pts_seconds'] <= b['window'][1] + offset + 1e-6]
            if not available:
                row['status'] = 'NO_FRAME_BEFORE_PHASE_END'; continue
            frame = min(available, key=lambda f: abs(f['pts_seconds'] - target))
            row.update(source_frame=frame, gaze_frame_error_seconds=frame['pts_seconds'] - target,
                       source_in_phase=b['window'][0] <= frame['pts_seconds'] - offset <= b['window'][1] + 1e-6)
            if abs(row['gaze_frame_error_seconds']) > .08:
                row['status'] = 'NO_FRAME_WITHIN_80MS'; continue
            row['status'] = 'PENDING_EXTRACTION'
        pending = [r for r in entry['regions'] if r['status'] == 'PENDING_EXTRACTION']
        indices = sorted({r['source_frame']['source_frame_index'] for r in pending})
        native = directory / 'native'; native.mkdir(exist_ok=True)
        if indices:
            with tempfile.TemporaryDirectory(dir=native) as temporary:
                expr = '+'.join(f'eq(n\\,{i})' for i in indices)
                subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-threads', '4', '-i', ctx.video_path,
                                '-vf', f'select={expr},scale=640:640', '-vsync', '0', '-q:v', '2', str(Path(temporary) / 'frame_%04d.jpg')],
                               capture_output=True, check=True)
                extracted = sorted(Path(temporary).glob('frame_*.jpg'))
                if len(extracted) != len(indices):
                    raise ValueError('Exact decoded source-frame binding failed')
                for i, path in zip(indices, extracted):
                    path.rename(native / f'source_{i:06d}.jpg')
        for row in pending:
            frame = row['source_frame']; path = native / f"source_{frame['source_frame_index']:06d}.jpg"
            pose = interpolate_pose(records, frame['pts_seconds'] - offset)
            with Image.open(path) as source_image:
                rgb = source_image.convert('RGB')
            if stream['width'] != stream['height']:
                raise ValueError('Explicitly support only the audited square video pilot')
            projection = project_gaze(by_index[row['representative_observation']['source_index']], pose, *rgb.size)
            row.update(native_source_path=str(path.relative_to(output)), native_source_sha256=sha(path),
                       source_pose=pose, projection=projection)
            if projection['status'] != 'projected_frustum_only':
                row['status'] = 'NATIVE_GAZE_PROJECTION_UNSUPPORTED'; continue
            rays = native_angular_lookup(pose, event['canonical_pose'], rgb.size)
            sx, sy = sphere_indices(rays, maps.shape[2], maps.shape[1])
            visible = labels[row['phase'], sy, sx] == row['component']
            box = component_crop(visible, projection['pixel_xy'], rgb.size)
            if box is None:
                row['status'] = 'COMPONENT_OUTSIDE_NATIVE_FOV'; continue
            context = rgb.copy(); ImageDraw.Draw(context).rectangle(box, outline='white', width=3)
            paths = {'context_image': context, 'crop_image': rgb.crop(box)}
            row['crop_box_xyxy'] = box; row['heat_images'] = []
            for key, image in paths.items():
                dest = directory / f"{row['id']}_{key}.png"; image.save(dest)
                row[key] = {'path': str(dest.relative_to(output)), 'sha256': sha(dest)}
            for phase in range(3):
                image = heat_tile(density[phase, sy, sx], peak).crop(box)
                dest = directory / f"{row['id']}_heat{phase}.png"; image.save(dest)
                row['heat_images'].append({'path': str(dest.relative_to(output)), 'sha256': sha(dest)})
            row['status'] = 'RGB_EVIDENCE_READY'
        ready = [r for r in entry['regions'] if r['status'] == 'RGB_EVIDENCE_READY']
        # Independent of proposal mass and time; the model must read printed IDs.
        rng.shuffle(ready)
        for page_index in range(math.ceil(len(ready) / 6)):
            page = ready[page_index * 6:(page_index + 1) * 6]
            for condition in CONDITIONS:
                rid = f"{event['id']}_{condition}_page{page_index}"
                image = render_page(output, page, condition)
                path = directory / f'{condition}_page{page_index}.png'; image.save(path)
                requests.append({'id': rid, 'event_id': event['id'], 'event': event['event'], 'condition': condition,
                    'page': page_index, 'region_ids': [r['id'] for r in page], 'image': str(path.relative_to(output)),
                    'sha256': sha(path), 'prompt': PROMPT.replace('__INSTRUCTION__', ctx.text)})
        events.append(entry)
        print('PREPARED', event['event'], dict(Counter(r['status'] for r in entry['regions'])), flush=True)
    snapshots = {}
    for relative in ['MultiAgent/attention_regions.py', 'tools/run_attention_region_study.py']:
        dest = output / 'source' / relative; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, dest); snapshots[relative] = sha(dest)
    save(output / 'seam_audit.json', seam_audit())
    save(output / 'manifest.json', {'version': VERSION, 'retained_mass_fraction': .9, 'events': events, 'requests': requests,
        'conditions': CONDITIONS, 'regression_only': True, 'ground_truth_loaded': False, 'anchors_projected': False,
        'source_manifest_path': str(SOURCE / 'manifest.json'), 'source_manifest_sha256': sha(SOURCE / 'manifest.json'),
        'code_snapshots': snapshots,
        'protocol': 'All connected components of each 90%-mass envelope retained without top-k; residual stays in full volume. No object/mention GT.',
        'model_task': 'Original-instruction/RGB candidate screening; graphical attention only, no text attention scores.',
        'control_scope': 'Same gaze-derived candidates/crops/phase labels across conditions. NOT a gaze-free baseline or referent evaluation.',
        'temporal_scope': 'Offline whole recorded supported history, three equal bins; no exact speech-stage alignment.',
        'geometry_scope': 'Rotation-only angular alignment. No parallax, visibility or object tracking claims.'})
    save(output / 'protocol_lock.json', {'manifest_sha256': sha(output / 'manifest.json'), 'frozen_before_model': True,
        'timestamp_utc': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()})
    report(output)


def run(output):
    manifest = read(output / 'manifest.json')
    if sha(output / 'manifest.json') != read(output / 'protocol_lock.json')['manifest_sha256']:
        raise ValueError('Frozen protocol changed')
    backend = QwenBackend(output / 'model_cache', model='/ai/data/Qwen3-VL-8B-Instruct', local_only=True,
                          namespace='attention_regions_instruction_screen_v1', trace_visual_inputs=True)
    schedule = manifest['requests'][:]; random.Random(20261106).shuffle(schedule)
    for request in schedule:
        destination = output / 'responses' / f"{request['id']}.json"
        if destination.exists():
            continue
        path = output / request['image']
        if sha(path) != request['sha256']:
            raise ValueError('Frozen input changed')
        print('INFER', request['id'], flush=True); start = time.monotonic()
        parsed = backend.ask(request['id'], 'screen', request['prompt'], [path], max_new_tokens=512)
        audit = read(backend.requests[-1]); decisions = parsed.get('decisions', [])
        valid = (set(parsed) == {'decisions'} and isinstance(decisions, list) and
                 len(decisions) == len(request['region_ids']) and all(isinstance(d, dict) for d in decisions) and
                 sorted(d.get('id', '') for d in decisions) == sorted(request['region_ids']) and
                 all(set(d) == {'id', 'decision', 'visual'} and d['decision'] in {'keep', 'review', 'reject'}
                     and isinstance(d['visual'], str) for d in decisions))
        save(destination, {'id': request['id'], 'event_id': request['event_id'], 'condition': request['condition'],
            'page': request['page'], 'parsed': parsed, 'schema_valid': valid,
            'request_path': backend.requests[-1], 'seconds': time.monotonic() - start,
            'processor_audit': audit['processor_audit'],
            'vision_verified': audit['visual_tensor_trace']['input_consumption_verified']})
        print('RESULT', request['id'], valid, parsed, flush=True)
    report(output)


def report(output):
    manifest = read(output / 'manifest.json')
    responses = {p.stem: read(p) for p in (output / 'responses').glob('*.json')}
    previous_routes = read(PREVIOUS / 'routed_visual_evidence.json')['rows']
    sphere = {e['id']: e for e in read(SOURCE / 'manifest.json')['events']}
    coverage_rows, status = [], Counter()
    for event in manifest['events']:
        status.update(r['status'] for r in event['regions'])
        for phase in (0, 2):
            observations = sphere[event['id']]['volume_stats']['bins'][phase]['observations']
            candidates = sorted([r for r in event['regions'] if r['phase'] == phase and r['status'] == 'RGB_EVIDENCE_READY'],
                                key=lambda r: (-r['mass_seconds'], r['component']))
            old = [r['representative_observation'] for r in previous_routes if r['event_id'] == event['id'] and
                   r['phase'] == phase and r['method'] == 'visual_shortlist' and r['status'] == 'RGB_EVIDENCE_READY']
            pools = {'old_model_max2_face': old}
            for k in (1, 2, 'all'):
                pool = candidates if k == 'all' else candidates[:k]
                pools[f'component_{k}'] = [r['representative_observation'] for r in pool]
            for name, representatives in pools.items():
                for radius in (4, 8, 12):
                    coverage_rows.append({'event': event['event'], 'phase': phase, 'method': name, 'radius_degrees': radius,
                        'proposals': len(representatives), **angular_coverage(observations, event['canonical_pose'], representatives, radius)})
    coverage_summary = []
    for name in ['old_model_max2_face', 'component_1', 'component_2', 'component_all']:
        for radius in (4, 8, 12):
            rows = [r for r in coverage_rows if r['method'] == name and r['radius_degrees'] == radius]
            total = sum(r['total_seconds'] for r in rows); kept = sum(r['covered_seconds'] for r in rows)
            coverage_summary.append({'method': name, 'radius_degrees': radius, 'n': len(rows),
                'proposals': sum(r['proposals'] for r in rows), 'duration_weighted_angular_coverage': kept / total if total else None})
    calls, decision_counts, predictions = [], {}, {}
    for request in manifest['requests']:
        result = responses.get(request['id'])
        if result is None:
            continue
        calls.append(result)
        if not result['schema_valid']:
            continue
        for row in result['parsed']['decisions']:
            predictions[(request['event_id'], request['condition'], row['id'])] = row
    for condition in CONDITIONS:
        decision_counts[condition] = dict(Counter(d['decision'] for (event_id, c, rid), d in predictions.items() if c == condition))
    comparisons = {}
    for condition in ['NO_HEAT', 'REVERSE_TIME']:
        paired = [(d, predictions[(event_id, condition, rid)]) for (event_id, c, rid), d in predictions.items()
                  if c == 'TRUE_HEAT' and (event_id, condition, rid) in predictions]
        comparisons[condition] = {'paired_region_cards': len(paired),
            'decision_changes': sum(a['decision'] != b['decision'] for a, b in paired),
            'retained_set_changes': sum((a['decision'] != 'reject') != (b['decision'] != 'reject') for a, b in paired)}
    audits = [a for e in manifest['events'] for a in e['mass_audits']]
    summary = {'events': len(manifest['events']), 'time_bins': len(audits),
        'region_count': sum(len(e['regions']) for e in manifest['events']), 'status_counts': dict(status),
        'phase_mass_fraction_min': min(a['retained_fraction'] for a in audits if a['retained_fraction'] is not None),
        'total_supported_seconds': sum(a['total_seconds'] for a in audits),
        'envelope_seconds': sum(a['retained_seconds'] for a in audits), 'residual_seconds': sum(a['residual_seconds'] for a in audits),
        'planned_model_calls': len(manifest['requests']), 'complete_model_calls': len(calls),
        'all_schema_valid': all(c['schema_valid'] for c in calls), 'all_vision_verified': all(c['vision_verified'] for c in calls),
        'image_tokens': sorted({c['processor_audit']['expanded_image_token_count'] for c in calls}),
        'model_decision_counts': decision_counts, 'paired_condition_comparison': comparisons,
        'angular_coverage': coverage_summary, 'object_accuracy_available': False,
        'caveat': 'Angular coverage grows with representative budget; neither region count nor changed model outputs proves target accuracy or temporal understanding.'}
    save(output / 'analysis.json', summary); save(output / 'angular_coverage.json', coverage_rows)
    payload = {'events': manifest['events'], 'requests': manifest['requests'], 'responses': responses, 'summary': summary}
    data = json.dumps(payload, ensure_ascii=False).replace('</', '<\\/')
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>连续关注区域与指令筛选</title><style>body{font:15px system-ui;background:#f3f5f7;margin:20px;color:#182d3d}header{position:sticky;top:0;background:#f3f5f7;padding:10px;z-index:1}select{font:inherit;margin:6px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:15px}figure{background:white;margin:0;padding:12px}img{width:100%}pre{white-space:pre-wrap}p{max-width:1300px}</style><header><h2>脚本提取连续关注区域 → 原生 RGB → 结合原始指令筛选</h2><select id="event"></select><select id="condition"><option>TRUE_HEAT</option><option>NO_HEAT</option><option>REVERSE_TIME</option></select><p id="instruction"></p></header><p>区域在完整球面上按每段90%支持时长的最高密度包络提取，跨显示边界连通；每个分量均保留，没有最多两个的截断。剩余约10%仍存于完整体积。区域不是对象，源帧旁的细节图可以包含多个物体。同一物体也可能重复出现。热图仍是图像：早/中/晚三张均映射到该真实源帧的相机方向及同一个裁剪范围。原始时序与源帧可追溯；旋转对齐没有解决平移视差。</p><p>三个条件共享同一批 gaze 派生裁剪，所以无热图条件仅检查新增热图显示的作用，不能作为完全没有 gaze 的基线。模型输出 keep/review/reject 只是未标注的候选审阅，不能当目标准确率。所有13交互都是历史样例，尚无独立验证。</p><p><a href="seam_audit.json">旧双热点判定复核</a> · <a href="analysis.json">统计</a> · <a href="angular_coverage.json">方向覆盖审计</a></p><main></main><script>const d=__DATA__,q=document.querySelector('#event'),c=document.querySelector('#condition');d.events.forEach(e=>{let o=document.createElement('option');o.value=e.id;o.textContent=e.event;q.append(o)});function render(){let e=d.events.find(e=>e.id===q.value);document.querySelector('#instruction').textContent=e.instruction;let main=document.querySelector('main');main.replaceChildren();d.requests.filter(r=>r.event_id===q.value&&r.condition===c.value).forEach(r=>{let f=document.createElement('figure'),im=document.createElement('img'),text=document.createElement('pre');im.src=r.image;let z=d.responses[r.id];text.textContent='第'+(r.page+1)+'页 | '+(z?JSON.stringify(z.parsed,null,2):'尚未运行');f.append(im,text);main.append(f)})}q.onchange=render;c.onchange=render;render();</script></html>'''
    (output / 'index.html').write_text(page.replace('__DATA__', data))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'report'])
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    {'prepare': prepare, 'run': run, 'report': report}[args.action](args.output.resolve())
