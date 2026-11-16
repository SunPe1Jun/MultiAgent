#!/usr/bin/env python3
"""Freeze, run and evaluate script-routed RGB evidence on project-new interactions."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import csv
import hashlib
import html
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import time
import traceback

import numpy as np
from PIL import Image

from MultiAgent.agents import ReferentialAgent
from MultiAgent.attention_volume import build_volume
from MultiAgent.baseline import ROOT as TRI, WORKSPACE_ROOT, module, scene_paths
from MultiAgent.camera_geometry import aligned_frame_geometry
from MultiAgent.context import vrtriref_sample_to_agent_context
from MultiAgent.full_frame_grounding import decode_grounding
from MultiAgent.full_frame_grounding_v4 import validate_mention_contract
from MultiAgent.full_interaction_referring import paired_set_metrics
from MultiAgent.gaze_evidence import gaze_records
from MultiAgent.gaze_rgb_routing import CONDITIONS, VERSION, RoutingConfig, grounding_prompt, render_rgb_evidence, route_regions, literal_mention_hints
from MultiAgent.heatmap_temporal_pilot import save, sha
from MultiAgent.local_video_visual import bind_pose_contract, extract_dense_frames, probe_duration, qwen_temporal_groups
from MultiAgent.qwen_backend import QwenBackend
from tools.run_full_frame_grounding_v4 import clean_visual_trace

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / '.local-sync/gaze_rgb_routing_20261109_mention_first'
SEED = 20261109
MODEL = str(WORKSPACE_ROOT / 'Qwen3-VL-8B-Instruct')
VIDEO_MAX_PIXELS = 23_330_816
SOURCES = ['MultiAgent/gaze_rgb_routing.py', 'MultiAgent/agents.py', 'MultiAgent/mention_mapping.py', 'MultiAgent/full_frame_grounding.py',
    'MultiAgent/full_frame_grounding_v4.py', 'MultiAgent/attention_volume.py',
    'MultiAgent/camera_geometry.py', 'MultiAgent/frame_pose_alignment.py',
    'MultiAgent/local_video_visual.py', 'MultiAgent/gaze_evidence.py',
    'MultiAgent/qwen_backend.py', 'MultiAgent/qwen_video_compat.py',
    'tools/run_gaze_rgb_routing.py']


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def source_lock():
    return {name: sha(ROOT / name) for name in SOURCES}


def check_lock(output):
    manifest = read(output / 'manifest.json')
    if sha(output / 'manifest.json') != read(output / 'protocol_lock.json')['manifest_sha256']:
        raise ValueError('Frozen manifest changed')
    if source_lock() != manifest['source_sha256']:
        raise ValueError('Runtime sources differ from frozen protocol; use a new run')
    return manifest


def historical_exposure():
    """Conservative union of addressed events in existing manifests/selections.

    This excludes preparation as well as inference exposure. It cannot establish
    absence from model pretraining or an unavailable historical server.
    """
    seen, records = set(), []
    paths = set()
    for base in [ROOT / 'runs', ROOT / 'artifacts', ROOT / '.local-sync']:
        for pattern in ['*manifest*.json', '*selection*.json', '*fixed_samples*.json']:
            paths.update(base.rglob(pattern))
    def walk(value):
        if isinstance(value, dict):
            scene = value.get('scene_key', value.get('scene'))
            index = value.get('row_index')
            if isinstance(scene, str) and scene.startswith('scene') and isinstance(index, (int, str)):
                try:
                    seen.add((scene, int(index)))
                except ValueError:
                    pass
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    for path in sorted(paths):
        if not path.is_file():
            continue
        try:
            walk(read(path))
            records.append({'path': str(path), 'sha256': sha(path)})
        except (ValueError, OSError):
            continue
    return seen, records


def instruction_lookup():
    """Read original instruction columns only; never load referent columns."""
    import openpyxl
    merged = WORKSPACE_ROOT / 'TriModal-Referring.local-materials-20261104/data/instruction_set_merged.csv'
    lookup, provenance = {}, []
    with merged.open(encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            scene_id, order = row['scene_id'], int(row['instruction_order'])
            if scene_id not in {'4', '5'}:
                continue
            scene = f'scene4_room{(order - 1) // 200 + 1}' if scene_id == '4' else 'scene5'
            index = (order - 1) % 200 if scene_id == '4' else order - 1
            lookup[(scene, index)] = row['instruction_text']
    provenance.append({'path': str(merged), 'sha256': sha(merged), 'columns_read': ['scene_id', 'instruction_order', 'instruction_text']})
    for number in (1, 2, 3):
        path = WORKSPACE_ROOT / f'Scene{number}InstructionCollection.xlsx'
        book = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            for name, offset in [('Basic', 0), ('Medium', 400)]:
                sheet = book[name]
                # Column positions are established by source formats; read
                # only the instruction column, never the adjacent referents.
                col = 3 if number == 1 and name == 'Basic' else 5 if number == 2 else 4
                start = 1 if number == 3 and name == 'Medium' else 2
                for index, row in enumerate(sheet.iter_rows(min_row=start, max_row=start + 399,
                                                            min_col=col, max_col=col, values_only=True)):
                    lookup[(f'scene{number}', offset + index)] = str(row[0] or '').strip()
        finally:
            book.close()
        provenance.append({'path': str(path), 'sha256': sha(path), 'columns_read': 'instruction column only'})
    # Validate row addresses against all 100 archived instruction strings.
    archived = read(ROOT / '.local-sync/pilot100.manifest.json')['samples']
    mismatches = [(s['scene_key'], s['row_index']) for s in archived
                  if lookup.get((s['scene_key'], s['row_index'])) != s['instruction']]
    if mismatches:
        raise ValueError(f'Instruction/address validation failed: {mismatches}')
    return lookup, provenance


def select(output):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose a new empty output; preserve all previous runs')
    lookup, instructions = instruction_lookup()
    seen, exposure = historical_exposure()
    rng, selected = random.Random(SEED), []
    for scene in sorted({s for s, _ in lookup}):
        pool = sorted(index for (s, index), text in lookup.items() if s == scene and text and (s, index) not in seen)
        if not pool:
            raise ValueError(f'No project-new instruction addresses for {scene}')
        index = rng.choice(pool)
        sid = scene[5:].split('_')[0]
        selected.append({'scene_key': scene, 'row_index': index, 'event_id': f'scene{sid}_{index + 1}',
                         'instruction': lookup[(scene, index)]})
    # Lock selection and settings before inspecting videos, support or responses.
    save(output / 'selection.json', {'seed': SEED, 'samples': selected,
        'project_previously_addressed_events': len(seen), 'prior_artifacts_scanned': exposure,
        'instruction_sources': instructions, 'archived_instruction_address_checks': 100,
        'selection_uses_gt_heat_or_responses': False, 'selected_utc': now(),
        'scope': 'One per subscene, excluded from available local manifests/selections. Not a pretraining holdout; missing historical-server exposure cannot be ruled out.',
        'conditions': list(CONDITIONS), 'routing_config': RoutingConfig().__dict__,
        'frames_per_interaction': 12, 'model': MODEL, 'video_max_pixels': VIDEO_MAX_PIXELS,
        'max_new_tokens': 1536, 'source_sha256': source_lock()})
    print('SELECTED', [(s['scene_key'], s['row_index']) for s in selected], flush=True)


def make_context(item, output):
    helper = TRI / 'scripts/data_prep/build_scene_api_input.py'
    spec = importlib.util.spec_from_file_location('gaze_rgb_input_builder', helper)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    prep = builder.load_prep_module()
    scene, index = item['scene_key'], item['row_index']
    folder = WORKSPACE_ROOT / 'V3dMD' / (('scene4/' + scene.split('_')[1]) if scene.startswith('scene4_') else scene) / str(index + 1)
    video = builder.resolve_single_match(folder, ['ScreenRecord_*.mp4', '*.mp4'], 'video')
    raw = builder.resolve_single_match(folder, ['multimodal_data.json'], 'telemetry')
    timed = prep.collect_timed_samples(prep.load_multimodal_samples(raw))
    if not timed:
        raise ValueError('No timed telemetry')
    end, peak = timed[-1][0], timed[-1][0] / 2
    peak_sample = prep.select_peak_sample(timed, peak)
    peak_samples = prep.select_peak_window_samples(timed, t_peak=peak, peak_window_seconds=.25, fallback_sample=peak_sample)
    payload = prep.build_event_json_payload(event_id=item['event_id'], json_path=raw,
        t_start=0., t_peak=peak, t_end=end, peak_window_seconds=.25,
        window_samples=[s for _, s in timed], peak_window_samples=peak_samples, peak_sample=peak_sample,
        image_width=0, image_height=0,
        spatial_prior={'source': 'none', 'u_norm': None, 'v_norm': None, 'world_point': None, 'sample_count': 0},
        include_prior_metadata=False)
    event_path = prep.write_event_json(output / 'inputs' / scene, item['event_id'], payload)
    row = {'event_id': item['event_id'], 'sample_id': str(index + 1), 'scene_id': scene[5:].split('_')[0],
           'instruction_text': item['instruction'], 'utterance_text': '', 'video_path': str(video),
           'json_path': str(raw), 'event_json_path': str(event_path), 't_start': '0', 't_peak': str(peak), 't_end': str(end),
           '_scene_key': scene, '_row_index': index}
    context = vrtriref_sample_to_agent_context(row, load_ground_truth=False)
    context.provenance['input_csv'] = None
    context.provenance['input_source'] = 'Original instruction text + local raw telemetry; reconstructed event payload'
    save(output / 'inputs' / scene / f'{index}_row.json', {'row': row, 'provenance': context.provenance,
        'raw_telemetry_sha256': sha(raw), 'event_json_sha256': sha(event_path)})
    return context


def prepare_event(item, output_string):
    output = Path(output_string)
    directory = output / 'events' / item['scene_key'] / str(item['row_index'])
    cache = directory / 'event_prepared.json'
    if cache.exists():
        return read(cache)
    ctx = make_context(item, output)
    print('CPU_START', ctx.scene_key, ctx.row_index, flush=True)
    frames = extract_dense_frames(Path(ctx.video_path), directory / 'frames', 0., probe_duration(Path(ctx.video_path)),
                                  min_frames=12, max_frames=12)
    alignment = bind_pose_contract(ctx, frames, directory, output / 'pose_alignment')
    contract = read(output / 'pose_alignment' / ctx.scene_key / str(ctx.row_index) / 'frame_pose_contract.json')
    records, rendered = gaze_records(ctx), []
    for source in contract['frames']:
        path = Path(source['source_path'])
        geometry = aligned_frame_geometry(source, sha(path), source['frame_time']['actual_container_pts_seconds'], 0.)
        with Image.open(path) as image:
            rgb = image.convert('RGB')
        maps, stats = build_volume(records, query=geometry['telemetry_time_seconds'], pose=geometry['reference_pose'],
            image_size=rgb.size, clock_supported=geometry['clock_available'], history=2., bins=3, sigma_degrees=2.)
        frame_dir = directory / source['frame_id']
        frame_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(frame_dir / 'attention_internal.npz', maps=maps)
        images, routing = {}, {}
        for condition in CONDITIONS:
            route = route_regions(maps, rgb.size, clock_supported=geometry['clock_available'], condition=condition)
            target = frame_dir / f'{condition}.png'
            composite = render_rgb_evidence(rgb, route)
            composite.save(target)
            images[condition] = {'path': str(target.relative_to(output)), 'sha256': sha(target), 'size': list(composite.size)}
            routing[condition] = route
        rendered.append({'frame_id': source['frame_id'], 'source_path': str(path), 'source_sha256': sha(path),
            'source_size': list(rgb.size), 'source_frame_index': source['frame_time']['source_frame_index'],
            'source_frame_id': f"frame_{source['frame_time']['source_frame_index']:06d}",
            'time_seconds': geometry['video_time_seconds'], 'geometry': geometry,
            'attention_stats': stats, 'routing': routing, 'images': images,
            'maps_path': str((frame_dir / 'attention_internal.npz').relative_to(output)),
            'maps_sha256': sha(frame_dir / 'attention_internal.npz')})
    candidates = [{'candidate_id': c.candidate_id, 'aliases': list(c.aliases),
                   'position_world': [c.x_world, c.y_world, c.z_world]} for c in ctx.candidates]
    random.Random('full-scene-candidates:' + ctx.scene_key).shuffle(candidates)
    event = {'id': f'{ctx.scene_key}_{ctx.row_index}', 'event': f'{ctx.scene_key}/{ctx.row_index}',
        **item, 'frames': rendered, 'video_path': ctx.video_path, 'video_sha256': sha(Path(ctx.video_path)),
        'telemetry_path': ctx.json_path, 'telemetry_sha256': sha(Path(ctx.json_path)),
        'scene_candidate_table': candidates, 'qwen_temporal_groups': qwen_temporal_groups(rendered),
        'clock_available_frames': sum(f['geometry']['clock_available'] for f in rendered),
        'clock_status': alignment, 'gt_loaded': False, 'answer_conditioned_candidates': False}
    save(cache, event)
    print('CPU_DONE', event['event'], 'supported', event['clock_available_frames'], flush=True)
    return event


def prepare(output, workers=3):
    if (output / 'manifest.json').exists():
        raise ValueError('Manifest already frozen')
    selection = read(output / 'selection.json')
    if selection['source_sha256'] != source_lock():
        raise ValueError('Sources changed since selection; choose a new selection')
    events = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(prepare_event, item, str(output)) for item in selection['samples']]
        for future in as_completed(futures):
            events.append(future.result())
    events.sort(key=lambda e: (e['scene_key'], e['row_index']))
    backend = QwenBackend(output / 'language_cache', model=MODEL, local_only=True, namespace=VERSION + '_mention_first_v2')
    parser = ReferentialAgent(backend)
    for event in events:
        print('MENTIONS', event['event'], flush=True)
        persisted_row = read(output / 'inputs' / event['scene_key'] / f"{event['row_index']}_row.json")['row']
        context = vrtriref_sample_to_agent_context(persisted_row, load_ground_truth=False)
        try:
            parsed = parser.run(context)
            hints = [{key: mention.get(key) for key in
                      ('mention_id', 'text', 'span', 'object_type', 'role', 'kind', 'count', 'coreference_id')}
                     for mention in parsed['mentions']]
        except ValueError as exc:
            # Keep the raw parser failure. This experiment needs literal
            # occurrences, not the old parser's candidate-parent mapping.
            if 'must be a literal span inside mention' not in str(exc):
                raise
            raw = read(backend.requests[-1])['parsed']
            hints = literal_mention_hints(raw, context.text)
            parsed = {'mentions': hints, 'candidate_mapping_schema_error': str(exc),
                      'fallback': 'Literal spans + occurrence counts + explicit actor coverage; no parent identity repair.'}
        event['mention_hints'] = hints
        save(output / 'language' / f"{event['id']}.json", parsed)
        event['language_request_path'] = backend.requests[-1]
        event['language_request_sha256'] = sha(Path(backend.requests[-1]))
        event['prompts'] = {condition: grounding_prompt(event, condition) for condition in CONDITIONS}
    requests = []
    for event in events:
        fingerprints = {}
        for condition in CONDITIONS:
            image_hashes = [f['images'][condition]['sha256'] for f in event['frames']]
            fingerprint = hashlib.sha256(json.dumps([event['prompts'][condition], image_hashes], sort_keys=True).encode()).hexdigest()
            rid = f"{event['id']}_{condition}"
            requests.append({'id': rid, 'event_id': event['id'], 'condition': condition,
                             'canonical_request_id': fingerprints.setdefault(fingerprint, rid), 'input_fingerprint': fingerprint})
    manifest = {'version': VERSION, 'events': events, 'conditions': list(CONDITIONS), 'requests': requests,
        'selection_sha256': sha(output / 'selection.json'), 'source_sha256': source_lock(),
        'model': MODEL, 'model_config_sha256': sha(Path(MODEL) / 'config.json'),
        'max_new_tokens': 1536, 'video_max_pixels': VIDEO_MAX_PIXELS,
        'planned_unique_visual_calls': len({r['canonical_request_id'] for r in requests}),
        'language_model_calls': len(backend.call_records), 'gt_loaded': False,
        'routing_config': RoutingConfig().__dict__, 'heatmap_sent_to_model': False,
        'full_field_retained': True, 'full_scene_candidates_retained': True,
        'default_pipeline_changed': False, 'unseen_validation': 'project-new relative to available artifact addresses; missing server exposure unknown',
        'decoder': {'module': 'MultiAgent.full_frame_grounding.decode_grounding', 'unchanged': True, 'uses_gt_or_gaze_to_select_id': False},
        'controls': 'FULL_RGB repeats full field. UNIFORM_RGB uses fixed windows with matched support gating. GAZE_RGB uses internal phase mass. MIRRORED_GAZE_RGB reflects mass only; RGB stays unchanged.',
        'known_limits': ['RGB retains original recorder marks; no condition is gaze-free.',
            'Detail windows depend on approximate history projection, not object identity.',
            'Three 40%-of-short-side detail windows do not guarantee all attention modes are covered.',
            'UNIFORM_RGB shares gaze availability/support gating; spatial selection control only.',
            'Model still locates points in full RGB; scene-ID decoder remains approximate.',
            'No paired direct-heatmap condition in this pilot; archived scores are not a matched comparison.',
            'One interaction per subscene is an engineering pilot, not a powered efficacy test.']}
    save(output / 'manifest.json', manifest)
    save(output / 'protocol_lock.json', {'manifest_sha256': sha(output / 'manifest.json'), 'frozen_utc': now(),
        'before_any_visual_grounding': True, 'language_only_parses_already_frozen': True, 'gt_loaded': False})
    for name in SOURCES:
        destination = output / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    print('FROZEN', len(events), 'events', len(requests), 'conditions', manifest['planned_unique_visual_calls'], 'unique calls', flush=True)


def infer(output, limit=None):
    m = check_lock(output)
    backend = QwenBackend(output / 'model_cache', model=m['model'], local_only=True, namespace=VERSION,
                          trace_visual_inputs=True, video_max_pixels=m['video_max_pixels'])
    events = {e['id']: e for e in m['events']}
    canonical = [r for r in m['requests'] if r['canonical_request_id'] == r['id']]
    random.Random(SEED).shuffle(canonical)
    completed = 0
    for request in canonical:
        aliases = [r for r in m['requests'] if r['canonical_request_id'] == request['id']]
        if all((output / 'responses' / f"{r['id']}.json").exists() for r in aliases):
            continue
        event, condition = events[request['event_id']], request['condition']
        paths = [output / f['images'][condition]['path'] for f in event['frames']]
        for path, frame in zip(paths, event['frames']):
            if sha(path) != frame['images'][condition]['sha256']:
                raise ValueError('Frozen visual input changed')
        metadata = [{k: f[k] for k in ('frame_id', 'source_frame_id', 'source_frame_index', 'time_seconds')} for f in event['frames']]
        started, audit, parsed = time.monotonic(), {}, None
        print('INFER', request['id'], flush=True)
        try:
            parsed = backend.ask_video(request['id'], 'grounding', event['prompts'][condition], paths, metadata,
                max_new_tokens=m['max_new_tokens'], video_path=event['video_path'],
                video_time_range=[event['frames'][0]['time_seconds'], event['frames'][-1]['time_seconds']])
            raw = read(backend.requests[-1])
            audit = raw['processor_audit']
            checked = validate_mention_contract(parsed, event)
            decoded = decode_grounding(checked['grounded_referents'], event)
            valid = checked['schema_valid'] and checked['coverage_complete']
            result = {'parsed': parsed, 'mention_validation': checked, 'decoder': decoded,
                'validation': {'schema_valid': checked['schema_valid'], 'coverage_complete': checked['coverage_complete'],
                               'selected_object_names': decoded['selected_object_names'] if valid else []},
                'vision_verified': bool(audit.get('visual_tensor_trace', {}).get('input_consumption_verified'))}
        except Exception as exc:
            result = {'parsed': parsed, 'error': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc(),
                'mention_validation': {}, 'decoder': {'decisions': []},
                'validation': {'schema_valid': False, 'coverage_complete': False, 'selected_object_names': []}, 'vision_verified': False}
        request_path = Path(backend.requests[-1]) if backend.requests else None
        for alias in aliases:
            save(output / 'responses' / f"{alias['id']}.json", {**alias, **result,
                'scene_key': event['scene_key'], 'row_index': event['row_index'], 'event': event['event'],
                'processor_audit': audit, 'request_path': str(request_path) if request_path else None,
                'request_sha256': sha(request_path) if request_path and request_path.exists() else None,
                'seconds': time.monotonic() - started, 'exact_input_reused': alias['id'] != request['id']})
        if request_path:
            clean_visual_trace(request_path)
        completed += 1
        save(output / 'progress.json', {'completed_conditions': len(list((output / 'responses').glob('*.json'))),
            'planned_conditions': len(m['requests']), 'last': request['id'], 'gt_loaded': False})
        print('RESULT', request['id'], result['validation'], 'vision', result['vision_verified'],
              'seconds', round(time.monotonic() - started, 2), flush=True)
        if limit and completed >= limit:
            break


def verify(output):
    m = check_lock(output)
    totals = Counter()
    for event in m['events']:
        if event['gt_loaded'] or event['answer_conditioned_candidates']:
            raise ValueError('Inference contamination flag')
        for frame in event['frames']:
            if sha(Path(frame['source_path'])) != frame['source_sha256']:
                raise ValueError('Source image changed')
            with Image.open(frame['source_path']) as image:
                source = image.convert('RGB')
            maps_file = output / frame['maps_path']
            if sha(maps_file) != frame['maps_sha256']:
                raise ValueError('Internal attention changed')
            with np.load(maps_file) as archive:
                maps = archive['maps']
            width, height = source.size
            for condition in CONDITIONS:
                route = route_regions(maps, source.size, clock_supported=frame['geometry']['clock_available'], condition=condition)
                if route != frame['routing'][condition]:
                    raise ValueError('Region routing not reproducible')
                path = output / frame['images'][condition]['path']
                if sha(path) != frame['images'][condition]['sha256']:
                    raise ValueError('Composite input changed')
                with Image.open(path) as image:
                    pixels = np.asarray(image)
                np.testing.assert_array_equal(pixels[28:28 + height, :width], source)
                np.testing.assert_array_equal(pixels, render_rgb_evidence(source, route))
                totals['full_rgb_exact_matches'] += 1
                totals['detail_slots'] += len(route['slots'])
                totals['routed_detail_slots'] += sum(s['fallback_reason'] is None for s in route['slots'])
        for condition in CONDITIONS:
            if grounding_prompt(event, condition) != event['prompts'][condition]:
                raise ValueError('Model prompt not reproducible')
    save(output / 'verification.json', {'counts': dict(totals), 'source_and_input_hashes_verified': True,
        'routing_rebuilt_from_internal_maps': True, 'every_composite_pixel_rebuilt_from_rgb': True,
        'heatmap_model_pixels': 0, 'gt_read': False, 'verified_utc': now()})
    print('VERIFIED', dict(totals), flush=True)


def freeze_predictions(output):
    m = check_lock(output)
    paths = {r['id']: output / 'responses' / f"{r['id']}.json" for r in m['requests']}
    missing = [key for key, path in paths.items() if not path.exists()]
    if missing:
        raise ValueError(f'Cannot score before every condition has a frozen result: {missing}')
    target = output / 'predictions_frozen.json'
    hashes = {key: sha(path) for key, path in paths.items()}
    if target.exists():
        if read(target)['response_sha256'] != hashes:
            raise ValueError('Predictions changed after freeze')
    else:
        save(target, {'manifest_sha256': sha(output / 'manifest.json'), 'before_gt_evaluation': True,
                      'frozen_utc': now(), 'response_sha256': hashes})
    return m, {key: read(path) for key, path in paths.items()}


def evaluate(output):
    m, predictions = freeze_predictions(output)
    evaluator = module('eval')
    gt_by_scene = {}
    for scene in sorted({e['scene_key'] for e in m['events']}):
        aliases, _ = evaluator.load_anchor_alias_map(scene_paths(scene)['anchor'])
        gt_by_scene[scene] = (aliases, evaluator.load_gt_rows(scene_paths(scene)['gt'], aliases))
    result = {'conditions': {}, 'events': [], 'gt_loaded_after_prediction_freeze': True,
              'unique_visual_requests': len({r['canonical_request_id'] for r in m['requests']}),
              'condition_results': len(predictions), 'schema_valid': sum(r['validation']['schema_valid'] for r in predictions.values()),
              'coverage_complete': sum(r['validation']['coverage_complete'] for r in predictions.values()),
              'vision_verified': sum(r['vision_verified'] for r in predictions.values()),
              'expanded_video_tokens': sorted({r['processor_audit'].get('expanded_video_token_count') for r in predictions.values() if r['processor_audit']}),
              'decoder_reasons': dict(Counter(d['reason'] for r in predictions.values() for d in r['decoder']['decisions']))}
    for condition in CONDITIONS:
        records, eval_rows, original_gt = [], [], []
        for event in m['events']:
            response = predictions[f"{event['id']}_{condition}"]
            names = response['validation']['selected_object_names']
            dest = output / 'predictions' / condition / event['scene_key'] / f"row_{event['row_index']}.json"
            save(dest, {'row_index': event['row_index'], 'event_id': event['event_id'],
                'parsed_response': {'selected_object_names': names},
                'response_status': 'ok' if response['validation']['schema_valid'] and response['validation']['coverage_complete'] else 'schema_failure',
                'used_storyboard_fallback': False})
            aliases, gt_rows = gt_by_scene[event['scene_key']]
            gt = gt_rows[event['row_index']]
            records.append({'gt': gt['mapped_referents'], 'prediction': names})
            eval_rows.extend(evaluator.build_eval_rows([dest], gt_rows, aliases))
            original_gt.append(gt)
        summary = evaluator.summarize(eval_rows, original_gt)
        save(output / 'evaluation' / condition / 'original_summary.json', summary)
        evaluator.write_eval_csv(output / 'evaluation' / condition / 'evaluation.csv', eval_rows)
        result['conditions'][condition] = {'set_metrics': paired_set_metrics(records), 'original_evaluator': summary}
    for event in m['events']:
        aliases, gt_rows = gt_by_scene[event['scene_key']]
        gt = set(gt_rows[event['row_index']]['mapped_referents'])
        choices = {c: predictions[f"{event['id']}_{c}"]['validation']['selected_object_names'] for c in CONDITIONS}
        result['events'].append({'id': event['id'], 'instruction': event['instruction'], 'gt': sorted(gt), 'predictions': choices,
            'supported_frames': event['clock_available_frames'],
            'metrics': {c: {'exact': set(p) == gt, 'f1': 2 * len(set(p) & gt) / (len(set(p)) + len(gt)) if p or gt else 0.}
                        for c, p in choices.items()}})
    from MultiAgent.attention_reliability import cluster_paired_interval
    result['paired_event_f1_contrasts'] = {}
    for control in ('FULL_RGB', 'UNIFORM_RGB', 'MIRRORED_GAZE_RGB'):
        rows = result['events']
        contrast = cluster_paired_interval([{'cluster': e['id'], 'baseline': e['metrics'][control]['f1'],
                                             'treatment': e['metrics']['GAZE_RGB']['f1']} for e in rows])
        contrast['helpful'] = sum(e['metrics']['GAZE_RGB']['f1'] > e['metrics'][control]['f1'] for e in rows)
        contrast['harmful'] = sum(e['metrics']['GAZE_RGB']['f1'] < e['metrics'][control]['f1'] for e in rows)
        result['paired_event_f1_contrasts'][control] = contrast
    save(output / 'analysis.json', result)
    print(json.dumps({c: r['set_metrics'] for c, r in result['conditions'].items()}, indent=2), flush=True)


def report(output):
    m = check_lock(output)
    analysis = read(output / 'analysis.json') if (output / 'analysis.json').exists() else None
    body = ['<!doctype html><meta charset="utf-8"><title>RGB evidence routing</title>',
        '<style>body{font:16px sans-serif;margin:24px;background:#14202b;color:#eee}img{width:100%;max-width:650px}details{margin:16px 0}pre{white-space:pre-wrap}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}a{color:#8de}</style>',
        '<h1>Gaze-guided RGB evidence</h1><p>Top left: full RGB. Other panels: RGB detail views. Attention stays inside the script. All conditions retain the entire scene candidate space.</p>']
    if analysis:
        body.append('<pre>' + html.escape(json.dumps({c: r['set_metrics'] for c, r in analysis['conditions'].items()}, ensure_ascii=False, indent=2)) + '</pre>')
    for event in m['events']:
        body.append('<h2>' + html.escape(event['event']) + '</h2><p>' + html.escape(event['instruction']) + '</p>')
        for frame in event['frames']:
            body.append('<details><summary>' + html.escape(frame['frame_id']) + f" · PTS {frame['time_seconds']:.3f}s</summary><div class=grid>")
            for condition in CONDITIONS:
                path = frame['images'][condition]['path']
                body.append('<div><p>' + condition + '</p><a href="' + html.escape(path) + '"><img loading=lazy src="' + html.escape(path) + '"></a><pre>' +
                    html.escape(json.dumps([{'box': s['bbox_source_xyxy'], 'fallback': s['fallback_reason']} for s in frame['routing'][condition]['slots']], ensure_ascii=False)) + '</pre></div>')
            body.append('</div></details>')
        if analysis:
            row = next(r for r in analysis['events'] if r['id'] == event['id'])
            body.append('<pre>' + html.escape(json.dumps(row, ensure_ascii=False, indent=2)) + '</pre>')
    (output / 'index.html').write_text('\n'.join(body))
    print('VIEWER', output / 'index.html', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['select', 'prepare', 'infer', 'verify', 'evaluate', 'report'])
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    output = args.output.resolve()
    actions = {'select': select, 'prepare': lambda p: prepare(p, args.workers),
               'infer': lambda p: infer(p, args.limit), 'verify': verify, 'evaluate': evaluate, 'report': report}
    actions[args.action](output)
