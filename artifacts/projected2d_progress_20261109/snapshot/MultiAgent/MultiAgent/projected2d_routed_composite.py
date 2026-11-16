"""Explicit composite coordinates mapped to the original 2D panel, without 3D identity."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil

from .gaze_rgb_routing import HEADER
from .projected2d_attention_study import save_json, sha_file
from .projected2d_routed_rgb import GROUPS, MODEL


def composite_to_source(panel, group, point):
    """Invert the frozen full-frame/detail composite using the actual render geometry."""
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        raise ValueError('composite_xy must have two coordinates')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           or v < 0 or v >= 1000 for v in point):
        raise ValueError('composite_xy must be finite and in [0,1000)')
    width, height = panel['display_size']
    px = float(point[0]) * (2 * width) / 1000
    py = float(point[1]) * (2 * (height + HEADER)) / 1000
    col, row = int(px // width), int(py // (height + HEADER))
    panel_index = row * 2 + col
    local_x = px - col * width
    local_y = py - row * (height + HEADER) - HEADER
    if not 0 <= local_y < height:
        raise ValueError('point falls in a title/header')
    if panel_index == 0:
        return [local_x / width, local_y / height], panel_index
    x0, y0, x1, y1 = panel['boxes'][group][panel_index-1]
    bw, bh = x1-x0, y1-y0
    if bw <= 0 or bh <= 0:
        raise ValueError('invalid source crop')
    scale = min(width / bw, height / bh)
    dw, dh = max(1, round(bw*scale)), max(1, round(bh*scale))
    pad_x, pad_y = (width-dw)//2, (height-dh)//2
    if not (pad_x <= local_x < pad_x+dw and pad_y <= local_y < pad_y+dh):
        raise ValueError('point falls in detail letterboxing')
    return [(x0+(local_x-pad_x)*bw/dw)/width,
            (y0+(local_y-pad_y)*bh/dh)/height], panel_index


def prompt_for(event, group):
    frames = [{'frame_id': p['frame_id'], 'video_time_seconds': p['requested_video_time'],
               'display_source_size': p['display_size'], 'detail_rectangles_xyxy': p['boxes'][group]}
              for p in event['panels']]
    tasks = [{**m, 'soft_time_hint': event['temporal_hints'].get(m['mention_id'])} for m in event['mentions']]
    return '''Solve 2D temporal localization for each mention in the full instruction.
Each supplied image is a separate chronological composite. The upper left is the complete RGB
source frame; the other three areas are optional RGB details from that same source frame.
The detail rectangles are listed below. Details are not target labels; objects anywhere in the
full frame remain eligible. Use visible RGB appearance, action and the instruction to resolve
each mention. Time hints are soft instruction phases, not speech timestamps; all panels remain eligible.

For every mention_id return exactly one entry, with status localized, ambiguous or not_visible.
For every localized physical target choose a supplied frame_id and point ON the visible object
in ANY of the four image areas of that frame. Return composite_xy=[x,y] on a 0..1000 scale
relative to the ENTIRE composite image, origin top-left, x right and y down. Do NOT convert to
full-source coordinates: the script handles all detail and title offsets. Do not point in titles
or blank padding. For multiple targets return one point per target; if uncertain give no point.
For a component locate the indicated whole parent. Use all frames to check action and identity.
Return JSON only with this structure, with actual numeric coordinates replacing x and y:
{"mentions":[{"mention_id":"...","status":"localized|ambiguous|not_visible",
"predictions":[{"frame_id":"one_of_the_supplied_panel_ids","composite_xy":[x,y]}],"evidence":"..."}]}.
''' + json.dumps({'instruction': event['instruction'], 'source_frames': frames,
                   'localization_tasks': tasks}, ensure_ascii=False)


def normalize(raw, event, group):
    allowed = {m['mention_id']: m for m in event['mentions']}
    panels = {p['frame_id']: p for p in event['panels']}
    rows = raw.get('mentions') if isinstance(raw, dict) else None
    if not isinstance(rows, list):
        return {'mentions': []}, ['response_missing_mentions_array'], []
    counts = Counter(r.get('mention_id') for r in rows if isinstance(r, dict))
    result, errors, conversions = [], [], []
    for row in rows:
        if not isinstance(row, dict) or row.get('mention_id') not in allowed:
            errors.append('unknown_or_malformed_mention')
            continue
        mid = row['mention_id']
        if counts[mid] != 1:
            errors.append(f'duplicate_mention:{mid}')
            continue
        status = row.get('status')
        if status not in ('localized', 'ambiguous', 'not_visible'):
            errors.append(f'invalid_status:{mid}')
            status = 'ambiguous'
        observations = row.get('predictions', [])
        if not isinstance(observations, list):
            errors.append(f'invalid_predictions:{mid}')
            observations = []
        predictions = []
        for item in observations:
            try:
                if status != 'localized':
                    raise ValueError('non-localized response included a point')
                frame_id = item['frame_id']
                if frame_id not in panels:
                    raise ValueError('panel was not supplied')
                uv, quadrant = composite_to_source(panels[frame_id], group, item['composite_xy'])
                predictions.append({'frame_id': frame_id, 'point_uv': uv})
                conversions.append({'mention_id': mid, 'frame_id': frame_id,
                                    'composite_xy': item['composite_xy'], 'source_uv': uv,
                                    'source_area_index': quadrant})
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f'invalid_point:{mid}:{exc}')
        result.append({'mention_id': mid, 'mention': allowed[mid]['text'], 'status': status,
                       'predictions': predictions, 'evidence': str(row.get('evidence') or '')})
    for mid in sorted(set(allowed)-{r['mention_id'] for r in result}):
        errors.append(f'missing_mention:{mid}')
        result.append({'mention_id': mid, 'mention': allowed[mid]['text'], 'status': 'ambiguous',
                       'predictions': [], 'evidence': 'Missing or duplicate model row.'})
    return {'mentions': result}, errors, conversions


def prepare(source: Path, output: Path):
    if (output / 'input_manifest.json').exists():
        raise ValueError('Do not overwrite frozen study')
    source_lock = json.loads((source / 'protocol_lock.json').read_text())
    if sha_file(source / 'input_manifest.json') != source_lock['input_manifest_sha256']:
        raise ValueError('Prior 2D input changed')
    if not json.loads((source / 'input_reconstruction_verification.json').read_text())['all_verified']:
        raise ValueError('Prior 2D inputs were not verified')
    output.mkdir(parents=True, exist_ok=True)
    for name in ('rebuilt_manifest_all.csv', 'panel_parity_audit.json', 'exact_set_policy.json',
                 'input_reconstruction_verification.json'):
        shutil.copy2(source / name, output / name)
    manifest = json.loads((source / 'input_manifest.json').read_text())
    manifest.update(version='projected2d_routed_composite_v2',
        coordinate_contract='model gives whole composite xy 0..1000; script maps RGB area to source-frame point_uv',
        source_input_manifest_sha256=sha_file(source / 'input_manifest.json'),
        source_sha256=sha_file(Path(__file__)),
        prior_contract_diagnosis='v1 requested source UV but responses frequently appeared to use composite coordinates')
    save_json(output / 'input_manifest.json', manifest)
    save_json(output / 'protocol_lock.json', {'input_manifest_sha256': sha_file(output / 'input_manifest.json'),
        'source_sha256': sha_file(Path(__file__)), 'evaluator_manifest_sha256': sha_file(output / 'rebuilt_manifest_all.csv')})


def infer(output: Path):
    from .qwen_backend import QwenBackend
    from tools.run_full_frame_grounding_v4 import clean_visual_trace

    manifest = json.loads((output / 'input_manifest.json').read_text())
    lock = json.loads((output / 'protocol_lock.json').read_text())
    if (sha_file(output / 'input_manifest.json') != lock['input_manifest_sha256'] or
            sha_file(Path(__file__)) != lock['source_sha256'] or
            sha_file(output / 'rebuilt_manifest_all.csv') != lock['evaluator_manifest_sha256']):
        raise ValueError('Frozen inputs or code changed')
    backend = QwenBackend(output / 'model_cache', str(MODEL), local_only=True,
                          namespace='projected2d_composite_v2', trace_visual_inputs=True)
    for i, event in enumerate(manifest['events'], 1):
        aliases = {}
        for group in GROUPS:
            target = output / 'responses' / f"{event['id']}_{group}.json"
            if target.exists():
                existing = json.loads(target.read_text())
                aliases[existing['fingerprint']] = existing
                continue
            images = [p['inputs'][group] for p in event['panels']]
            for image in images:
                if sha_file(Path(image['path'])) != image['sha256']:
                    raise ValueError('Frozen source image changed')
            prompt = prompt_for(event, group)
            fingerprint = hashlib.sha256(json.dumps([prompt, [x['sha256'] for x in images]],
                                                   ensure_ascii=False).encode()).hexdigest()
            if fingerprint in aliases:
                result = {**aliases[fingerprint], 'condition': group,
                          'reused_identical_from': aliases[fingerprint]['condition']}
            elif not event['parser_valid'] or not event['mentions'] or not images:
                result = {'status': 'NOT_RUN', 'parsed': {'mentions': []}, 'error': 'parser failure or missing panel'}
            else:
                try:
                    raw = backend.ask(event['id'], group, prompt, [x['path'] for x in images], max_new_tokens=1536)
                    request_path = Path(backend.requests[-1])
                    request = json.loads(request_path.read_text())
                    parsed, errors, conversions = normalize(raw, event, group)
                    result = {'status': 'COMPLETE' if not errors else 'PARSE_WITH_WARNINGS',
                        'parsed': parsed, 'parse_warning': errors, 'coordinate_conversions': conversions,
                        'invalid_panel_predictions': sum('panel was not supplied' in err for err in errors),
                        'raw_response': request.get('response'),
                        'vision_verified': bool((request.get('visual_tensor_trace') or {}).get('input_consumption_verified')),
                        'processor_audit': request.get('processor_audit'), 'request_path': str(request_path)}
                    clean_visual_trace(str(request_path))
                except Exception as exc:
                    result = {'status': 'FAILED', 'parsed': {'mentions': []}, 'error': f'{type(exc).__name__}: {exc}'}
            result.update(event_id=event['event_id'], scene=event['scene_key'], row_index=event['row_index'],
                          condition=group, fingerprint=fingerprint, prompt=prompt, input_images=images)
            save_json(target, result)
            aliases[fingerprint] = result
            print('INFER_V2', i, '/', len(manifest['events']), event['id'], group, result['status'], flush=True)
