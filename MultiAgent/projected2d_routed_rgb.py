"""Frozen projected-2D RGB routing experiment. No scene-ID decoder."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from .baseline import ROOT as TRI
from .projected2d_attention_study import CSV_COLUMNS, evaluator_adapter, normalize_model_mentions, read_csv, save_json, sha_file, write_csv

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / '.local-sync/referent_rgb_expansion32_20261109_repaired'
ORIGINAL = ROOT / '.local-sync/referent_rgb_expansion32_20261109'
MODEL = ROOT.parent / 'Qwen3-VL-8B-Instruct'
GROUPS = ('FULL_RGB', 'UNIFORM_RGB', 'GAZE_RGB', 'MIRRORED_GAZE_RGB')


def prepare_inputs(output: Path):
    from PIL import Image
    from .attention_volume import build_volume
    from .camera_geometry import CameraPoseTimeline, aligned_frame_geometry
    from .context import vrtriref_sample_to_agent_context
    from .frame_pose_alignment import source_frame_pose
    from .gaze_evidence import gaze_records
    from .gaze_rgb_routing import render_rgb_evidence, route_regions
    from .head_motion_diagnostics import camera_records, probe_pts, verify_saved_frame_pts

    audit = json.loads((output / 'panel_parity_audit.json').read_text())
    if audit['mismatch_count']:
        raise ValueError('Rebuilt projections differ from the released manifest')
    old = json.loads((PRIOR / 'manifest.json').read_text())
    rows = read_csv(output / 'rebuilt_manifest_all.csv')
    by_event = defaultdict(dict)
    for row in rows:
        if row['panel_id']:
            by_event[(row['scene'], int(row['row_index']))].setdefault(row['panel_id'], row)
    events = []
    for event in old['events']:
        scene, index = event['scene_key'], event['row_index']
        cached = json.loads((PRIOR / 'inputs' / scene / f'{index}_row.json').read_text())['row']
        ctx = vrtriref_sample_to_agent_context(cached, load_ground_truth=False)
        contract = json.loads((ORIGINAL / 'pose_alignment' / scene / str(index) / 'frame_pose_contract.json').read_text())
        if contract['source_hashes'] != {'video_sha256': sha_file(Path(ctx.video_path)),
                                         'telemetry_sha256': sha_file(Path(ctx.json_path))}:
            raise ValueError(f'Clock source hashes differ: {scene}/{index}')
        timeline = CameraPoseTimeline(camera_records(ctx))
        records = gaze_records(ctx)
        pts, _ = probe_pts(ctx.video_path, 'ffprobe')
        panels = []
        for panel_id, row in sorted(by_event[(scene, index)].items()):
            source = Path(row['frame_path'])
            if not source.is_file():
                raise FileNotFoundError(source)
            requested = float(row['video_frame_time'])
            source_sha = sha_file(source)
            verified = verify_saved_frame_pts(ctx.video_path,
                {'video_time_seconds': requested, 'source_sha256': source_sha}, pts, 'ffmpeg')
            attention_camera = source_frame_pose(timeline, verified, contract['clock'], allow_candidate=True)
            frame = {'frame_time': verified, 'source_sha256': source_sha, 'attention_camera': attention_camera}
            geometry = aligned_frame_geometry(frame, source_sha, requested,
                                              float(row['video_time_offset_seconds'] or 0))
            with Image.open(source) as img:
                rgb = img.convert('RGB')
                rgb.thumbnail((512, 512), Image.Resampling.LANCZOS)
            maps, stats = build_volume(records, query=geometry['telemetry_time_seconds'],
                pose=geometry['reference_pose'], image_size=rgb.size,
                clock_supported=geometry['clock_available'], history=2., bins=3, sigma_degrees=2.)
            inputs, boxes = {}, {}
            for group in GROUPS:
                route = route_regions(maps, rgb.size, clock_supported=geometry['clock_available'], condition=group)
                target = output / 'inputs' / scene / str(index) / panel_id / f'{group}.png'
                target.parent.mkdir(parents=True, exist_ok=True)
                render_rgb_evidence(rgb, route).save(target)
                inputs[group] = {'path': str(target), 'sha256': sha_file(target)}
                boxes[group] = [slot['bbox_source_xyxy'] for slot in route['slots']]
            panels.append({'frame_id': panel_id, 'source_path': str(source), 'source_sha256': source_sha,
                'source_size': [int(row['image_width']), int(row['image_height'])],
                'display_size': list(rgb.size), 'requested_video_time': requested,
                'pts_verification': verified['status'], 'actual_pts': verified.get('actual_container_pts_seconds'),
                'clock_supported': geometry['clock_available'], 'geometry_status': geometry['status'],
                'attention_duration_by_phase': [x['duration'] for x in stats['bins']],
                'boxes': boxes, 'inputs': inputs})
        fields = ('mention_id', 'text', 'span', 'object_type', 'kind', 'role', 'count',
                  'component', 'parent_text', 'anchor_text', 'coreference_id',
                  'parent_mention_id', 'coreference_note', 'relation')
        mentions = [{k: m.get(k) for k in fields} for m in event['mention_hints']]
        hints = {}
        for mid, focus in event['temporal_search']['mentions'].items():
            hints[mid] = {'soft_phase_range': focus['primary_range'],
                          'soft_phase_panels': [p['frame_id'] for p in panels
                              if focus['primary_range'][0] <= p['requested_video_time'] <= focus['primary_range'][1]]}
        events.append({'id': event['id'], 'scene_key': scene, 'row_index': index,
            'event_id': event['event_id'], 'instruction': event['instruction'],
            'mentions': mentions, 'parser_valid': event['parser_valid'], 'temporal_hints': hints, 'panels': panels})
        print('INPUTS', scene, index, 'panels', len(panels),
              'supported', sum(p['clock_supported'] for p in panels), flush=True)
    manifest = {'version': 'projected2d_routed_rgb_v1', 'conditions': list(GROUPS), 'events': events,
        'model': str(MODEL), 'model_config_sha256': sha_file(MODEL / 'config.json'),
        'selection_sha256': sha_file(PRIOR / 'selection.json'),
        'parity_audit_sha256': sha_file(output / 'panel_parity_audit.json'),
        'scope': 'paired projected-2D source-panel points; no scene-ID output or decoder',
        'attention': 'three past-only phases route RGB detail windows; no rendered heatmap',
        'sample_status': '32 previously scored on a different ID-set diagnostic; exploratory, not held-out',
        'annotation_reconstruction': 'projected positions match; source API rows and temporal acceptability reconstructed',
        'source_sha256': sha_file(Path(__file__))}
    save_json(output / 'input_manifest.json', manifest)
    save_json(output / 'protocol_lock.json', {'input_manifest_sha256': sha_file(output / 'input_manifest.json'),
        'source_sha256': sha_file(Path(__file__))})


def prompt_for(event, group):
    panels = [{'frame_id': p['frame_id'], 'video_time_seconds': p['requested_video_time'],
               'source_image_size': p['display_size'], 'detail_rectangles_xyxy': p['boxes'][group]}
              for p in event['panels']]
    tasks = [{**m, 'soft_time_hint': event['temporal_hints'].get(m['mention_id'])} for m in event['mentions']]
    payload = {'instruction': event['instruction'], 'source_frames': panels, 'localization_tasks': tasks}
    return '''Solve 2D temporal localization for each mention in the instruction.
Each supplied image is one chronological panel. Its TOP LEFT area is the full RGB source frame;
the other three are optional RGB details from that SAME frame. Their source pixel rectangles
are listed below. They are not target labels. Look throughout the full frame and use the full
instruction, visible appearance, action and relationships to distinguish instances.
The time hints are approximate instruction-phase hypotheses, not speech timestamps; all panels
remain eligible. Repeated expressions may denote the same physical object when supported by context.
For each mention_id return exactly one entry. status is localized, ambiguous, or not_visible.
For localized entries give predictions with frame_id and point_uv=[u,v] in the FULL SOURCE FRAME,
normalized 0..1 with origin at top-left. Never use composite image coordinates. Give one
point per physical target; for a plural expression give multiple points only when justified.
Ground a part on its indicated parent object. Do not invent points when evidence is insufficient.
Return JSON only with this schema, using no default coordinates:
{"mentions":[{"mention_id":"...","status":"localized|ambiguous|not_visible",
"predictions":[{"frame_id":"one_of_the_supplied_panel_ids","point_uv":[u,v]}],"evidence":"..."}]}.
''' + json.dumps(payload, ensure_ascii=False)


def infer(output: Path):
    from .qwen_backend import QwenBackend
    from tools.run_full_frame_grounding_v4 import clean_visual_trace

    manifest = json.loads((output / 'input_manifest.json').read_text())
    lock = json.loads((output / 'protocol_lock.json').read_text())
    if sha_file(output / 'input_manifest.json') != lock['input_manifest_sha256'] or sha_file(Path(__file__)) != lock['source_sha256']:
        raise ValueError('Frozen input or code changed')
    backend = QwenBackend(output / 'model_cache', str(MODEL), local_only=True,
                          namespace='projected2d_routed_rgb_v1', trace_visual_inputs=True)
    for i, event in enumerate(manifest['events'], 1):
        aliases = {}
        for group in GROUPS:
            result_path = output / 'responses' / f"{event['id']}_{group}.json"
            if result_path.exists():
                saved = json.loads(result_path.read_text())
                aliases[saved['fingerprint']] = saved
                continue
            images = [p['inputs'][group] for p in event['panels']]
            for image in images:
                if sha_file(Path(image['path'])) != image['sha256']:
                    raise ValueError('Frozen image changed')
            prompt = prompt_for(event, group)
            fingerprint = hashlib.sha256(json.dumps([prompt, [p['sha256'] for p in images]],
                                                   ensure_ascii=False).encode()).hexdigest()
            if fingerprint in aliases:
                result = {**aliases[fingerprint], 'condition': group,
                          'reused_identical_from': aliases[fingerprint]['condition']}
            elif not event['parser_valid'] or not event['mentions'] or not images:
                result = {'status': 'NOT_RUN', 'parsed': {'mentions': []}, 'error': 'parser failure or missing panel'}
            else:
                try:
                    raw = backend.ask(event['id'], group, prompt, [p['path'] for p in images], max_new_tokens=1536)
                    request_path = Path(backend.requests[-1])
                    request = json.loads(request_path.read_text())
                    parsed, warning = normalize_model_mentions(raw, {'mentions': event['mentions']})
                    allowed = {p['frame_id'] for p in event['panels']}
                    rejected = 0
                    for mention in (parsed or {}).get('mentions', []):
                        predictions = mention['predictions']
                        mention['predictions'] = [p for p in predictions if p['frame_id'] in allowed]
                        rejected += len(predictions) - len(mention['predictions'])
                    result = {'status': 'COMPLETE' if not warning and not rejected else 'PARSE_WITH_WARNINGS',
                        'parsed': parsed or {'mentions': []}, 'parse_warning': warning,
                        'invalid_panel_predictions': rejected, 'raw_response': request.get('response'),
                        'vision_verified': bool((request.get('visual_tensor_trace') or {}).get('input_consumption_verified')),
                        'processor_audit': request.get('processor_audit'), 'request_path': str(request_path)}
                    clean_visual_trace(str(request_path))
                except Exception as exc:
                    result = {'status': 'FAILED', 'parsed': {'mentions': []},
                              'error': f'{type(exc).__name__}: {exc}'}
            result.update(event_id=event['event_id'], scene=event['scene_key'], row_index=event['row_index'],
                          condition=group, fingerprint=fingerprint, prompt=prompt, input_images=images)
            save_json(result_path, result)
            aliases[fingerprint] = result
            print('INFER', i, '/', len(manifest['events']), event['id'], group, result['status'], flush=True)


def evaluate(output: Path):
    manifest = json.loads((output / 'input_manifest.json').read_text())
    lock = json.loads((output / 'protocol_lock.json').read_text())
    if sha_file(output / 'input_manifest.json') != lock['input_manifest_sha256']:
        raise ValueError('Frozen input changed')
    paths = [output / 'responses' / f"{e['id']}_{g}.json" for e in manifest['events'] for g in GROUPS]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise ValueError(f'Missing {len(missing)} condition responses')
    save_json(output / 'predictions_frozen.json', {'responses': {str(p): sha_file(p) for p in paths}})
    summary = {}
    for group in GROUPS:
        predictions = []
        for event in manifest['events']:
            response = json.loads((output / 'responses' / f"{event['id']}_{group}.json").read_text())
            adapted = evaluator_adapter(response.get('parsed', {}))
            predictions.append({'scene': event['scene_key'], 'row_index': event['row_index'],
                'event_id': event['event_id'], 'instruction': event['instruction'],
                'model_input_image': ';'.join(x['path'] for x in response.get('input_images', [])),
                'model_raw_output': response.get('raw_response') or '',
                'parsed_json': json.dumps(adapted, ensure_ascii=False),
                'parse_ok': str(response['status'] in ('COMPLETE', 'PARSE_WITH_WARNINGS')),
                'prediction_count': len(adapted['referents']), 'error_message': response.get('error', '')})
        pred = output / 'predictions' / f'{group}.csv'
        write_csv(pred, predictions, CSV_COLUMNS)
        target = output / 'evaluation' / group
        subprocess.run([sys.executable, str(TRI / 'exam2/evaluate_2d_point_grounding.py'),
               '--manifest', str(output / 'rebuilt_manifest_all.csv'), '--pred_csv', str(pred),
               '--output_dir', str(target), '--coordinate_mode', 'panel'],
               cwd=TRI, check=True, capture_output=True, text=True)
        summary[group] = json.loads((target / '2d_eval_summary.json').read_text())
    save_json(output / 'summary.json', {'status': 'COMPLETE', 'metrics': summary,
        'scoring': 'original Experiment 2 projected 2D evaluator; labels are a provisional proxy',
        'sample_status': manifest['sample_status'], 'prediction_freeze': str(output / 'predictions_frozen.json')})
    print('EVALUATED', len(manifest['events']), 'events', flush=True)
