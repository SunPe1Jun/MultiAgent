"""Mention-focused, soft-phase ordering on the same projected-2D RGB panels."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

from .projected2d_attention_study import read_csv, save_json, sha_file, write_csv
from .projected2d_routed_composite import normalize
from .projected2d_routed_rgb import GROUPS, MODEL


def focus_event(event, mention):
    """Promote nearby phase panels without suppressing any original panel."""
    hint = event['temporal_hints'].get(mention['mention_id']) or {}
    bounds = hint.get('soft_phase_range')
    if bounds is None:
        panels = list(event['panels'])
    else:
        center = (bounds[0]+bounds[1])/2
        panels = sorted(event['panels'], key=lambda p: (abs(p['requested_video_time']-center), p['frame_id']))
    return {**event, 'mentions': [mention], 'panels': panels}, hint


def prompt_for(focused, mention, hint, group):
    frames = [{'frame_id': p['frame_id'], 'video_time_seconds': p['requested_video_time'],
               'display_source_size': p['display_size'], 'detail_rectangles_xyxy': p['boxes'][group]}
              for p in focused['panels']]
    return '''Solve the given-expression 2D temporal localization for the ONE current mention.
Images follow source_frames order, and may be out of chronological order. The first image is
closest to the soft instruction-phase hint; inspect ALL supplied images for context and action.
The hint is approximate, not a speech timestamp. You may select any supplied frame_id.
In each composite, the upper left is the complete RGB source frame and the other three areas
are optional RGB details from that same frame. Details are not target labels. Find the intended
object using the full instruction, its relationships, appearance and the visible scene.
For a component, ground its indicated whole parent. If one mention denotes several physical
targets, provide one point per target. Avoid duplicate observations of the same target.

Return one JSON mention entry with the exact mention_id and status localized, ambiguous or
not_visible. For each localized target, select an actual frame_id and point ON the object in
ANY of that composite's four image areas. Return composite_xy=[x,y] on a 0..1000 scale relative
to the ENTIRE composite, origin top-left; x increases right and y increases down. Do not convert
to source-frame coordinates. Do not point in title bars or blank padding. If unsure return no point.
Use this JSON structure, replacing x and y with numeric values:
{"mentions":[{"mention_id":"...","status":"localized|ambiguous|not_visible",
"predictions":[{"frame_id":"one_of_the_supplied_panel_ids","composite_xy":[x,y]}],"evidence":"..."}]}.
''' + json.dumps({'instruction': focused['instruction'], 'current_mention': mention,
                   'soft_time_hint': hint, 'source_frames': frames}, ensure_ascii=False)


def prepare(source: Path, output: Path):
    if (output/'input_manifest.json').exists():
        raise ValueError('Do not overwrite a frozen focused study')
    if not (source/'predictions_frozen.json').is_file():
        raise ValueError('Source 2D predictions must be frozen first')
    previous = json.loads((source/'input_manifest.json').read_text())
    lock = json.loads((source/'protocol_lock.json').read_text())
    if sha_file(source/'input_manifest.json') != lock['input_manifest_sha256']:
        raise ValueError('Source input manifest changed')
    samples = []
    for scene in sorted({e['scene_key'] for e in previous['events']}):
        eligible = sorted((e for e in previous['events'] if e['scene_key'] == scene
                           and e['parser_valid'] and len(e['mentions']) >= 2), key=lambda e:e['row_index'])
        if not eligible:
            raise ValueError(f'No valid multi-mention event in {scene}')
        samples.append(eligible[0])
    output.mkdir(parents=True, exist_ok=True)
    annotations = read_csv(source/'rebuilt_manifest_all.csv')
    keys = {(e['scene_key'], e['row_index']) for e in samples}
    subset = [r for r in annotations if (r['scene'], int(r['row_index'])) in keys]
    write_csv(output/'rebuilt_manifest_all.csv', subset, list(subset[0]))
    shutil.copy2(source/'exact_set_policy.json', output/'exact_set_policy.json')
    manifest = {**previous, 'version': 'projected2d_focused_composite_v1', 'events': samples,
        'sample_status': 'Eight multi-mention events chosen by source address from already scored 32-event 2D development set; exploratory, not held-out.',
        'selection_rule': 'First parser-valid event with at least two instruction mentions in each subscene; no metric or GT used by selection.',
        'sampling_frozen_after_source_2d_scoring': True,
        'focus_policy': 'Each mention has its own visual call; reorder all same source panels by distance to soft phase center, never remove context; shared policy across all conditions.',
        'source_sha256': sha_file(Path(__file__)), 'source_input_manifest_sha256': sha_file(source/'input_manifest.json')}
    save_json(output/'input_manifest.json', manifest)
    save_json(output/'protocol_lock.json', {'input_manifest_sha256': sha_file(output/'input_manifest.json'),
        'source_sha256': sha_file(Path(__file__)), 'evaluator_manifest_sha256': sha_file(output/'rebuilt_manifest_all.csv')})
    print('FOCUS_SAMPLES', [(e['id'], len(e['mentions'])) for e in samples], flush=True)


def infer(output: Path):
    from .qwen_backend import QwenBackend
    from tools.run_full_frame_grounding_v4 import clean_visual_trace
    manifest = json.loads((output/'input_manifest.json').read_text())
    lock = json.loads((output/'protocol_lock.json').read_text())
    if (sha_file(output/'input_manifest.json') != lock['input_manifest_sha256'] or
            sha_file(Path(__file__)) != lock['source_sha256'] or
            sha_file(output/'rebuilt_manifest_all.csv') != lock['evaluator_manifest_sha256']):
        raise ValueError('Frozen focus inputs or code changed')
    backend = QwenBackend(output/'model_cache', str(MODEL), local_only=True,
                          namespace='projected2d_focused_composite_v1', trace_visual_inputs=True)
    for n, event in enumerate(manifest['events'], 1):
        grouped = {g: [] for g in GROUPS}
        for mention in event['mentions']:
            focused, hint = focus_event(event, mention)
            aliases = {}
            for group in GROUPS:
                target = output/'calls'/f"{event['id']}_{mention['mention_id']}_{group}.json"
                if target.exists():
                    record = json.loads(target.read_text())
                    grouped[group].append(record)
                    aliases[record['fingerprint']] = record
                    continue
                images = [p['inputs'][group] for p in focused['panels']]
                for image in images:
                    if sha_file(Path(image['path'])) != image['sha256']:
                        raise ValueError('Frozen image changed')
                prompt = prompt_for(focused, mention, hint, group)
                fingerprint = hashlib.sha256(json.dumps([prompt, [x['sha256'] for x in images]],
                                                       ensure_ascii=False).encode()).hexdigest()
                if fingerprint in aliases:
                    record = {**aliases[fingerprint], 'condition': group,
                              'reused_identical_from': aliases[fingerprint]['condition']}
                else:
                    try:
                        raw = backend.ask(event['id']+'/'+mention['mention_id'], group, prompt,
                                          [x['path'] for x in images], max_new_tokens=1024)
                        request_path = Path(backend.requests[-1])
                        request = json.loads(request_path.read_text())
                        parsed, errors, conversions = normalize(raw, focused, group)
                        record = {'status': 'COMPLETE' if not errors else 'PARSE_WITH_WARNINGS',
                            'parsed': parsed, 'parse_warning': errors, 'coordinate_conversions': conversions,
                            'raw_response': request.get('response'),
                            'vision_verified': bool((request.get('visual_tensor_trace') or {}).get('input_consumption_verified')),
                            'processor_audit': request.get('processor_audit'), 'request_path': str(request_path)}
                        clean_visual_trace(str(request_path))
                    except Exception as exc:
                        record = {'status': 'FAILED', 'parsed': {'mentions': []},
                                  'error': f'{type(exc).__name__}: {exc}'}
                record.update(event_id=event['event_id'], scene=event['scene_key'], row_index=event['row_index'],
                    mention_id=mention['mention_id'], condition=group, fingerprint=fingerprint,
                    prompt=prompt, input_images=images, supplied_frame_order=[p['frame_id'] for p in focused['panels']])
                save_json(target, record)
                aliases[fingerprint] = record
                grouped[group].append(record)
                print('FOCUS', n, '/', len(manifest['events']), event['id'], mention['mention_id'], group,
                      record['status'], flush=True)
        for group, records in grouped.items():
            errors = [r for r in records if r['status'] != 'COMPLETE']
            result = {'event_id': event['event_id'], 'scene': event['scene_key'], 'row_index': event['row_index'],
                'condition': group, 'status': 'COMPLETE' if not errors else 'PARSE_WITH_WARNINGS',
                'parsed': {'mentions': [m for r in records for m in r.get('parsed', {}).get('mentions', [])]},
                'coordinate_conversions': [c for r in records for c in r.get('coordinate_conversions', [])],
                'vision_verified': all(r.get('vision_verified') for r in records),
                'call_paths': [str(output/'calls'/f"{event['id']}_{m['mention_id']}_{group}.json") for m in event['mentions']],
                'input_images': records[0]['input_images'],
                'processor_audit': records[0].get('processor_audit'), 'raw_response': None,
                'error': ';'.join(r.get('error', '') for r in errors)}
            save_json(output/'responses'/f"{event['id']}_{group}.json", result)
