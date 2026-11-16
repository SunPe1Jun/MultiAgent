"""Controlled, GT-free presentation/readout pilot on saved PTS/pose contracts.

Run with the reconstructed-pilot100 environment sourced. Preparation never fits
clocks or projects anchors. Cue readout is not target-localization accuracy.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image, ImageDraw

from .camera_geometry import aligned_frame_geometry
from .context import load_contexts
from .gaze_evidence import DEFAULTS, build_heatmap, gaze_records, render_presentation
from .qwen_backend import QwenBackend

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'temporal_gaze_readout_pilot_v1'
SAMPLES = [('scene1', 261), ('scene2', 411), ('scene3', 577),
           ('scene4_room1', 72), ('scene5', 28)]
CONDITIONS = ('RGB', 'INSTANT_HEATMAP', 'HISTORY_HEATMAP', 'ORDERED_HEATMAP',
              'REVERSED_TIME_HEATMAP', 'INSTANT_FOVEATION', 'ORDERED_PLUS_FOVEATION')
LABELS = ('原始 RGB', '即时热图', '2秒累计热图', '分时热图：蓝旧→红新',
          '反向时间颜色对照', '即时 foveation', '分时热图 + 即时 foveation')
TEMPORAL = {'ORDERED_HEATMAP', 'REVERSED_TIME_HEATMAP', 'ORDERED_PLUS_FOVEATION'}
CONFIG = {**DEFAULTS, 'overlay_max_alpha': 0.50}
COLORS = np.array([[40, 120, 255], [255, 215, 40], [255, 70, 0]], np.float32)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def peak(heat):
    if not np.any(heat > 0):
        return None
    y, x = np.unravel_index(np.argmax(heat), heat.shape)
    return [float(x), float(y)]


def temporal_maps(projected, query, sigma, size, history=2.0):
    """Partition *the same* weighted gaze kernels into three equal time bins."""
    width, height = size
    maps = np.zeros((3, height, width), np.float32)
    radius = max(2, math.ceil(sigma * 3))
    for row in projected:
        if row.get('status') != 'projected_frustum_only':
            continue
        gaze_time = float(row['gaze_time_seconds'])
        if not query - history <= gaze_time <= query:
            raise ValueError('History contains a future/out-of-window observation')
        index = min(2, int((gaze_time - (query - history)) / history * 3))
        cx, cy = row['pixel_xy']
        x0, x1 = max(0, int(cx) - radius), min(width, int(cx) + radius + 1)
        y0, y1 = max(0, int(cy) - radius), min(height, int(cy) + radius + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        kernel = np.exp(-((xx-cx)**2 + (yy-cy)**2)/(2*sigma*sigma)).astype(np.float32)
        maps[index, y0:y1, x0:x1] += kernel * float(row['weight_seconds'])
    return maps


def render_temporal(rgb, maps, reverse=False):
    """Color age bins without changing their spatial support or total alpha."""
    heat = maps.sum(axis=0)
    alpha = (1 - np.exp(-heat * CONFIG['display_scale'])) * CONFIG['overlay_max_alpha']
    colors = COLORS[::-1] if reverse else COLORS
    color = np.einsum('khw,kc->hwc', maps, colors) / np.maximum(heat[..., None], 1e-12)
    base = np.asarray(rgb, dtype=np.float32)
    output = np.clip(base*(1-alpha[..., None]) + color*alpha[..., None], 0, 255).astype(np.uint8)
    return Image.fromarray(output), alpha


def label_image(image, frame_id, pts, size=448):
    # A sequence of separately encoded images keeps source frames distinct;
    # Qwen video temporal patches would merge two frames per visual observation.
    image = image.resize((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', (size, size + 32), '#151b22')
    canvas.paste(image, (0, 32))
    ImageDraw.Draw(canvas).text((8, 8), f'{frame_id} | PTS {pts:.3f}s', fill='white')
    return canvas


def view_point(xy, source_size, view_size=(448, 480)):
    if xy is None:
        return None
    width, height = source_size
    return [xy[0] / width * view_size[0], xy[1] / height * 448 + 32]


def prompt(event):
    frames = ', '.join(row['frame_id'] for row in event['frames'])
    return f'''Read the attention cues in these 12 chronological first-person RGB frames.
The frame ID and source PTS are in the header above each image. IDs: {frames}.
Instruction context: {event['instruction']}

Red/orange translucent heat indicates gaze evidence, not a target label. A multicolor
heatmap represents the past two seconds: BLUE=oldest third, YELLOW=middle third,
ORANGE/RED=newest third. A plain red/orange heatmap may instead be an instantaneous
cue or an accumulation without time colors. Soft foveation preserves local detail
around gaze and mildly reduces peripheral detail. Some frames have no cue.

For each frame, give focus_xy as the strongest visible heat region, or the locally
clearest region for foveation alone. Give null if no added attention cue is readable.
For multicolor maps only, locate the strongest BLUE (oldest) and ORANGE/RED (newest)
regions separately as older_xy and newer_xy; otherwise return null for both.
Describe the visible RGB object/region around focus_xy, or null if unclear. Natural
object colors and the header are not gaze marks. Do not infer an object from the
instruction alone. Every coordinate uses the WHOLE supplied image, including its
header: [x,y] normalized to [0,1000], origin upper left. Use integers. Cue absence is
an allowed answer. Include all 12 IDs once. Do not return reasoning or extra fields.
Return strict JSON: {{"frames":[{{"frame_id":"V001","focus_xy":null,
"older_xy":null,"newer_xy":null,"object":null}}]}}'''


def prepare(audit_root, output, manifest_path):
    if output.exists():
        raise ValueError('Choose a new output directory; preparation never overwrites a pilot')
    selection = json.loads((audit_root/'selection.json').read_text())
    indexed = {(row['scene_key'], row['row_index']): row for row in selection['samples']}
    selected = [indexed[key] for key in SAMPLES]
    # The explicit source manifest protects against silently switching reconstructed data.
    source = json.loads(manifest_path.read_text())
    source_keys = {(row['scene_key'], row['row_index'], row['event_id']) for row in source['samples']}
    if any((r['scene_key'], r['row_index'], r['event_id']) not in source_keys for r in selected):
        raise ValueError('Pilot event differs from the source manifest')
    contexts = load_contexts({'samples': selected}, load_ground_truth=False)
    with (audit_root/'per_frame.csv').open(encoding='utf-8-sig', newline='') as stream:
        motion = {(r['scene'], int(r['row_index']), r['frame_id']):
                  {k: r[k] for k in ('motion_bin', 'angular_speed_deg_s')} for r in csv.DictReader(stream)}
    output.mkdir(parents=True)
    events, validation = [], []
    for ctx in contexts:
        key = f'{ctx.scene_key}/{ctx.row_index}'
        contract_path = audit_root/'pose_alignment'/key/'frame_pose_contract.json'
        contract = json.loads(contract_path.read_text())
        if sha(ctx.json_path) != contract['source_hashes']['telemetry_sha256']:
            raise ValueError(f'Telemetry changed: {key}')
        records = gaze_records(ctx)
        event = {'key': key, 'event_id': ctx.event_id, 'instruction': ctx.text,
                 'contract_path': str(contract_path), 'contract_sha256': sha(contract_path),
                 'telemetry_sha256': sha(ctx.json_path), 'frames': []}
        for frame in contract['frames']:
            source_path = Path(frame['source_path'])
            geometry = aligned_frame_geometry(frame, sha(source_path), frame['requested_seek_seconds'],
                                               contract['clock']['original_offset_seconds'])
            with Image.open(source_path) as image:
                rgb = image.convert('RGB')
            common = dict(frame_time_video=geometry['video_time_seconds'],
                          frame_time_telemetry=geometry['telemetry_time_seconds'],
                          event_window=[float(ctx.api_row['t_start']), float(ctx.api_row['t_end'])],
                          clock_available=geometry['clock_available'], image_size=rgb.size,
                          config=CONFIG, reference_pose=geometry['reference_pose'])
            current, current_stats, current_projections = build_heatmap(ctx, records, strategy='instant', **common)
            history, history_stats, projections = build_heatmap(ctx, records, strategy='local_history', **common)
            maps = temporal_maps(projections, geometry['telemetry_time_seconds'],
                                 history_stats['kernel_sigma_pixels_at_model_input'], rgb.size)
            np.testing.assert_allclose(maps.sum(axis=0), history, atol=1e-6, rtol=1e-6)
            ordered, alpha = render_temporal(rgb, maps)
            reversed_time, reversed_alpha = render_temporal(rgb, maps, reverse=True)
            np.testing.assert_array_equal(alpha, reversed_alpha)
            foveation, _ = render_presentation(rgb, current, 'F_inst', CONFIG)
            combined, _ = render_temporal(foveation, maps)
            instant_image, _ = render_presentation(rgb, current, 'H_inst', CONFIG)
            history_image, _ = render_presentation(rgb, history, 'H_local', CONFIG)
            outputs = dict(zip(CONDITIONS, [rgb, instant_image, history_image, ordered,
                                            reversed_time, foveation, combined]))
            foveation_points = [r.get('pixel_xy') for r in current_projections
                                if r['status'] == 'projected_frustum_only']
            focus = {'RGB': None, 'INSTANT_HEATMAP': peak(current), 'HISTORY_HEATMAP': peak(history),
                     'ORDERED_HEATMAP': peak(history), 'REVERSED_TIME_HEATMAP': peak(history),
                     'INSTANT_FOVEATION': peak(current), 'ORDERED_PLUS_FOVEATION': peak(history)}
            oldest, newest = peak(maps[0]), peak(maps[2])
            row = {'frame_id': frame['frame_id'], 'source_path': str(source_path),
                   'source_sha256': frame['source_sha256'],
                   'source_frame_index': frame['frame_time']['source_frame_index'],
                   'pts_seconds': geometry['video_time_seconds'],
                   'telemetry_query_seconds': geometry['telemetry_time_seconds'],
                   'offset_seconds': geometry['selected_clock_offset_seconds'],
                   'clock_supported': geometry['clock_available'],
                   **motion[(ctx.scene_key, ctx.row_index, frame['frame_id'])],
                   'current_stats': current_stats, 'history_stats': history_stats,
                   'instant_gaze_point': foveation_points[0] if foveation_points else None,
                   'conditions': {}}
            for condition, rendered in outputs.items():
                path = output/key/'inputs'/condition/f'{frame["frame_id"]}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                labelled = label_image(rendered, frame['frame_id'], row['pts_seconds'])
                labelled.save(path)
                old, new = (newest, oldest) if condition == 'REVERSED_TIME_HEATMAP' else (oldest, newest)
                row['conditions'][condition] = {'path': str(path.relative_to(output)), 'sha256': sha(path),
                    'image_size': list(labelled.size),
                    'expected_focus_xy': view_point(focus[condition], rgb.size),
                    'expected_older_xy': view_point(old, rgb.size) if condition in TEMPORAL else None,
                    'expected_newer_xy': view_point(new, rgb.size) if condition in TEMPORAL else None}
            if not geometry['clock_available']:
                if len({r['sha256'] for r in row['conditions'].values()}) != 1:
                    raise ValueError('Unsupported clock must give identical RGB in every condition')
            event['frames'].append(row)
            validation.append({'event': key, 'frame_id': row['frame_id'],
                'history_partition_equal': True, 'reversal_alpha_equal': True,
                'clock_equation_equal': abs(row['pts_seconds'] - row['telemetry_query_seconds']
                                            - float(row['offset_seconds'])) < 1e-6,
                'fallback_identical': not geometry['clock_available']})
        if len(event['frames']) != 12:
            raise ValueError('Expected exactly 12 frames per event')
        events.append(event)
        print(f'PREPARED {key}', flush=True)
    manifest = {'version': VERSION, 'conditions': list(CONDITIONS), 'labels': list(LABELS),
                'source_manifest': str(manifest_path), 'source_manifest_sha256': sha(manifest_path),
                'sample_selection': 'Fixed developmental pilot: five named subscenes, supported clocks; not random population evaluation',
                'audit_root': str(audit_root), 'ground_truth_loaded': False, 'anchors_projected': False,
                'config': CONFIG, 'image_size': [448, 480], 'transport': 'chronological_multi_image',
                'events': events, 'code_sha256': sha(__file__)}
    save(output/'stimulus_manifest.json', manifest)
    save(output/'validation.json', {'frames': len(validation), 'images': len(validation)*len(CONDITIONS),
         'checks': validation, 'source_hashes_verified': True, 'all_conditions_identical_frame_order': True,
         'model_calls': 0})
    write_gallery(output, manifest)
    return manifest


def write_gallery(output, manifest):
    dataset = json.dumps(manifest, ensure_ascii=False).replace('</', '<\\/')
    template = '''<!doctype html><html lang="zh"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>热图时序初步试验</title>
<style>body{font:15px system-ui;margin:20px;background:#f4f5f7;color:#162332}header{position:sticky;top:0;background:#f4f5f7;padding:8px;z-index:2}select,input,button{font:inherit;padding:5px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}figure{margin:0;background:white;padding:10px;border-radius:8px}img{width:100%;height:auto}figcaption{font-weight:600}#meta{white-space:pre-wrap}h1{font-size:21px}.note{color:#596578;max-width:1100px}</style>
<header><h1>热图时序与逐帧 foveation：固定输入对照</h1><select id="event"></select>
<input id="frame" type="range" min="0" max="11" value="6"><button id="play">播放 / 暂停</button><span id="tick"></span></header>
<p class="note">每个条件使用同一张真实RGB。蓝=最旧1/3、黄=中间1/3、橙红=最新1/3；时间反转对照只交换颜色。没有时钟支持的帧全部回退RGB。此处不含GT十字，不能据此验证目标定位精度。</p>
<p id="instruction"></p><p id="meta"></p><main class="grid" id="grid"></main>
<script>const data=__DATA__;const es=document.querySelector('#event'),sl=document.querySelector('#frame'),grid=document.querySelector('#grid');
data.events.forEach((e,i)=>{let o=document.createElement('option');o.value=i;o.textContent=e.key;es.append(o)});
function render(){const ev=data.events[+es.value],fr=ev.frames[+sl.value];document.querySelector('#instruction').textContent=ev.instruction;document.querySelector('#tick').textContent=fr.frame_id+' | '+fr.pts_seconds.toFixed(3)+'s';document.querySelector('#meta').textContent='clock supported: '+fr.clock_supported+' | head speed: '+(fr.angular_speed_deg_s||'N/A')+' deg/s | '+fr.motion_bin;grid.replaceChildren();data.conditions.forEach((c,i)=>{let fig=document.createElement('figure'),cap=document.createElement('figcaption'),img=document.createElement('img');cap.textContent=data.labels[i];img.src=fr.conditions[c].path;img.alt=c;fig.append(cap,img);grid.append(fig)})}
es.onchange=render;sl.oninput=render;let timer=null;document.querySelector('#play').onclick=()=>{if(timer){clearInterval(timer);timer=null}else timer=setInterval(()=>{sl.value=(+sl.value+1)%12;render()},800)};render();</script></html>'''
    (output/'index.html').write_text(template.replace('__DATA__', dataset))
    # A compact contact sheet remains useful without opening the HTML gallery.
    width, height = 250, 290
    sheet = Image.new('RGB', (width*7, height*len(manifest['events'])), 'white')
    draw = ImageDraw.Draw(sheet)
    for row_index, event in enumerate(manifest['events']):
        frame = event['frames'][6]
        for column, condition in enumerate(CONDITIONS):
            with Image.open(output/frame['conditions'][condition]['path']) as image:
                sheet.paste(image.resize((224, 240)), (column*width+10, row_index*height+40))
            draw.text((column*width+8, row_index*height+4), f'{event["key"]} V007', fill='black')
            draw.text((column*width+8, row_index*height+20), condition, fill='black')
    sheet.save(output/'comparison.jpg', quality=94)


def coordinate(value):
    if not isinstance(value, list) or len(value) != 2 or any(isinstance(x, bool) for x in value):
        return None
    try:
        values = list(map(float, value))
    except (ValueError, TypeError):
        return None
    return values if all(math.isfinite(x) and 0 <= x <= 1000 for x in values) else None


def score_response(parsed, event, condition):
    rows = parsed.get('frames', []) if isinstance(parsed, dict) else []
    if not isinstance(rows, list):
        rows = []
    ids = [r.get('frame_id') for r in rows if isinstance(r, dict)]
    expected_ids = [r['frame_id'] for r in event['frames']]
    valid_schema = (Counter(ids) == Counter(expected_ids)
                    and not parsed.get('_parse_warning'))
    by_id = {r['frame_id']: r for r in rows if isinstance(r, dict) and 'frame_id' in r}
    scored = []
    for frame in event['frames']:
        answer = by_id.get(frame['frame_id'], {})
        inputs = frame['conditions'][condition]
        width, height = inputs['image_size']
        result = {'frame_id': frame['frame_id'], 'motion_bin': frame['motion_bin'],
                  'clock_supported': frame['clock_supported'], 'schema_valid': valid_schema,
                  'object': answer.get('object')}
        for field, target_field in [('focus_xy', 'expected_focus_xy'),
                                    ('older_xy', 'expected_older_xy'), ('newer_xy', 'expected_newer_xy')]:
            target = inputs[target_field]
            predicted = coordinate(answer.get(field))
            error = (math.hypot(predicted[0]*width/1000-target[0], predicted[1]*height/1000-target[1])
                     / math.hypot(width, height) if target is not None and predicted is not None else None)
            result[field] = {'expected': target, 'predicted_1000': predicted,
                             'error_diagonal_fraction': error, 'eligible': target is not None,
                             'success_at_10pct': valid_schema and error is not None and error <= .10}
        a, b = inputs['expected_older_xy'], inputs['expected_newer_xy']
        result['order_eligible'] = (a is not None and b is not None
                                   and math.dist(a, b)/math.hypot(width, height) >= .10)
        result['both_endpoints_success'] = (result['order_eligible']
            and result['older_xy']['success_at_10pct'] and result['newer_xy']['success_at_10pct'])
        scored.append(result)
    return scored, valid_schema


def summarize(output, manifest):
    calls, scores = [], []
    for event in manifest['events']:
        for condition in CONDITIONS:
            result_path = output/'responses'/event['key']/f'{condition}.json'
            if not result_path.is_file():
                continue
            result = json.loads(result_path.read_text())
            calls.append({k: result.get(k) for k in ['event', 'condition', 'status', 'request_path', 'error']})
            scored, schema = score_response(result.get('parsed', {}), event, condition)
            scores.extend({'event': event['key'], 'condition': condition,
                           'call_status': result['status'], **row} for row in scored)
    result = {'version': VERSION, 'calls': calls, 'call_counts': dict(Counter(r['status'] for r in calls)),
              'model_calls_completed': sum(r['status'] == 'COMPLETE' for r in calls), 'conditions': {}}
    for condition in CONDITIONS:
        selected = [r for r in scores if r['condition'] == condition]
        metrics = {}
        for scope in ['all', 'low_lt10', 'medium_10to45', 'high_ge45', 'pose_unavailable']:
            subset = [r for r in selected if scope == 'all' or r['motion_bin'] == scope]
            eligible = [r for r in subset if r['focus_xy']['eligible']]
            read = [r['focus_xy']['error_diagonal_fraction'] for r in eligible
                    if r['focus_xy']['error_diagonal_fraction'] is not None and r['schema_valid']]
            no_cue = [r for r in subset if not r['focus_xy']['eligible']]
            order = [r for r in subset if r['order_eligible']]
            metrics[scope] = {'frames_total': len(subset), 'focus_eligible': len(eligible),
                'focus_valid_predictions': len(read),
                'focus_success_at_10pct': sum(r['focus_xy']['success_at_10pct'] for r in eligible),
                'median_error_fraction_among_valid': float(np.median(read)) if read else None,
                'no_cue_frames': len(no_cue),
                'no_cue_false_coordinates': sum(r['focus_xy']['predicted_1000'] is not None for r in no_cue),
                'order_eligible': len(order), 'both_endpoints_success': sum(r['both_endpoints_success'] for r in order)}
        result['conditions'][condition] = metrics
    save(output/'summary.json', result)
    save(output/'per_frame_scores.json', scores)
    return result


def infer(output, manifest, model, max_calls=None):
    backend = QwenBackend(output/'model_cache', model=str(model), local_only=True,
                          namespace=VERSION, trace_visual_inputs=True)
    schedule = [(event, condition) for event in manifest['events'] for condition in CONDITIONS]
    random.Random(20261105).shuffle(schedule)
    save(output/'schedule.json', [{'event': e['key'], 'condition': c} for e, c in schedule])
    calls = 0
    for event, condition in schedule:
        paths = [output/frame['conditions'][condition]['path'] for frame in event['frames']]
        for path, frame in zip(paths, event['frames']):
            if sha(path) != frame['conditions'][condition]['sha256']:
                raise ValueError('Stimulus changed since preparation')
        result_path = output/'responses'/event['key']/f'{condition}.json'
        if result_path.is_file():
            previous = json.loads(result_path.read_text())
            if previous['status'] == 'COMPLETE':
                continue
        if max_calls is not None and calls >= max_calls:
            break
        print(f'INFER {event["key"]} {condition}', flush=True)
        started = time.monotonic()
        try:
            parsed = backend.ask(event['key'], condition, prompt(event), paths, max_new_tokens=2600)
            scored, schema = score_response(parsed, event, condition)
            cache_path = Path(backend.requests[-1])
            request = json.loads(cache_path.read_text())
            result = {'event': event['key'], 'condition': condition, 'status': 'COMPLETE',
                'parsed': parsed, 'schema_valid': schema, 'request_path': str(cache_path),
                'processor_audit': request.get('processor_audit'),
                'visual_tensor_trace': request.get('visual_tensor_trace'),
                'seconds': time.monotonic()-started, 'frame_scores': scored}
        except Exception as exc:
            result = {'event': event['key'], 'condition': condition, 'status': 'FAILED',
                'error': f'{type(exc).__name__}: {exc}', 'seconds': time.monotonic()-started,
                'request_path': backend.requests[-1] if backend.requests else None}
            save(result_path, result)
            summarize(output, manifest)
            raise
        save(result_path, result)
        calls += 1
        summarize(output, manifest)
        print(f'RESULT {event["key"]} {condition} schema={schema} {result["seconds"]:.1f}s', flush=True)
    return summarize(output, manifest)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'infer', 'summarize'])
    parser.add_argument('--audit-dir', type=Path, default=ROOT/'.local-sync/projection_expansion24_20261105')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'.local-sync/heatmap_temporal_pilot_20261105')
    parser.add_argument('--source-manifest', type=Path,
        default=ROOT.parent/'TriModal-Referring/.local-sync/reconstructed_pilot100/pilot100.manifest.json')
    parser.add_argument('--model', type=Path, default=ROOT.parent/'Qwen3-VL-8B-Instruct')
    parser.add_argument('--max-calls', type=int)
    args = parser.parse_args(argv)
    if args.action == 'prepare':
        manifest = prepare(args.audit_dir.resolve(), args.output_dir.resolve(), args.source_manifest.resolve())
        print(f'Prepared {len(manifest["events"])} events x 12 frames x {len(CONDITIONS)} conditions', flush=True)
    else:
        manifest = json.loads((args.output_dir/'stimulus_manifest.json').read_text())
        if manifest['version'] != VERSION or sha(__file__) != manifest['code_sha256']:
            raise ValueError('Pilot code/version changed; prepare a new run directory')
        if args.action == 'infer':
            infer(args.output_dir.resolve(), manifest, args.model.resolve(), args.max_calls)
        else:
            summarize(args.output_dir.resolve(), manifest)


if __name__ == '__main__':
    main()
