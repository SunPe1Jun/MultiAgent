"""Small, frozen-input audit of keyframe heatmaps and pose-based panoramas."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .context import load_contexts
from .gaze_evidence import (DEFAULTS, project_gaze, select_gaze_observations)
from .local_attention_probe import MODEL_DEFAULT, prepare_sample, read_json, save, sha_file, weighted_center
from .local_video_visual import _candidate_scope
from .qwen_backend import QwenBackend
from .spatial_evidence import (build_directional_panorama, camera_rotation, heat_guide,
                               pixel_rays, view_atlas)


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
CONDITION_ORDER = ('C0', 'A-H', 'B-H', 'A-F', 'B-F')


def _attention_center_for_group(sample_dir, group):
    arrays = [np.load(sample_dir / 'inputs' / fid / 'attention_map.npz')
              ['H_unnormalized'].astype(np.float64) for fid in group['frame_ids']]
    return weighted_center(np.sum(arrays, axis=0)) if arrays else None


def _gaze_directions(records, frames, offset, config, image_size):
    if offset is None:
        return [], {'status': 'clock_unavailable', 'selected_count': 0,
                    'projected_count': 0, 'rejected': []}
    frame_times = [float(row['time_seconds']) for row in frames]
    telemetry_window = [min(frame_times)-float(offset)-float(config.get('local_history_seconds', 2.0)),
                        max(frame_times)-float(offset)]
    gaze_config = {**DEFAULTS, **config,
                   'utterance_pre_seconds': 0.0, 'utterance_post_seconds': 0.0}
    selected = select_gaze_observations(
        records, strategy='utterance_window', frame_time=telemetry_window[1],
        event_window=telemetry_window, clock_available=True, config=gaze_config)
    width, height = image_size
    projected, rejected = [], []
    for item in selected['selected']:
        row, weight = item['record'], float(item['weight_seconds'])
        if row.get('camera_position') is None or row.get('camera_rotation_xyzw') is None \
                or row.get('camera_fov_degrees') is None:
            rejected.append({'source_index': row['source_index'], 'reason': 'camera_pose_or_fov_missing'})
            continue
        pose = {'position_world': row['camera_position'],
                'rotation_xyzw': row['camera_rotation_xyzw'],
                'fov_vertical_degrees': row['camera_fov_degrees']}
        result = project_gaze(row, pose, width, height)
        video_time = float(row['telemetry_time_seconds']) + float(offset)
        nearest = min(frames, key=lambda frame: abs(float(frame['time_seconds'])-video_time))
        if result.get('status') != 'projected_frustum_only':
            rejected.append({'source_index': row['source_index'],
                             'gaze_time_telemetry_seconds': row['telemetry_time_seconds'],
                             'gaze_time_video_seconds': video_time,
                             'reason': result.get('status')})
            continue
        camera_ray = pixel_rays(result['normalized_uv'], image_size, pose['fov_vertical_degrees'])
        world_ray = camera_rotation(pose).apply(camera_ray)
        projected.append({
            'source_index': int(row['source_index']), 'timestamp': row['timestamp'],
            'gaze_time_seconds': video_time,
            'gaze_time_telemetry_seconds': float(row['telemetry_time_seconds']),
            'weight_seconds': weight, 'direction_world': world_ray.tolist(),
            'projection_source': result.get('projection_source'),
            'normalized_uv_at_observation_camera': result['normalized_uv'],
            'status': 'projected_frustum_only', 'visibility': 'visibility_unknown',
            'associated_view_id': nearest['view_id'],
            'source_frame_ids': nearest.get('source_frame_ids', [nearest['source_frame_id']]),
            'rgb_gaze_delta_seconds': abs(float(nearest['time_seconds'])-video_time),
        })
    return projected, {
        'status': 'ok' if projected else 'no_projectable_gaze_in_shared_observation_window',
        'telemetry_window_seconds': telemetry_window,
        'selected_count': len(selected['selected']), 'projected_count': len(projected),
        'rejected_count': len(rejected), 'rejected': rejected,
        'projection': 'same vertical-FOV pinhole model as RGB; ray uses gaze-sample camera pose',
        'translation_and_visibility': 'direction-only; no depth, occlusion, dynamic identity, or anchor association',
    }


def _prepare_spatial_inputs(sample_dir, frames, raw_records, offset, config, image_size):
    spatial_dir = sample_dir / 'inputs' / 'B1_directional_panorama'
    gaze, gaze_audit = _gaze_directions(raw_records, frames, offset, config, image_size)
    save(spatial_dir / 'gaze_projection_audit.json', gaze_audit)
    geometry = build_directional_panorama(
        frames, raw_records, spatial_dir, offset=offset, config={**DEFAULTS, **config},
        shape=tuple(config.get('panorama_shape', [384, 768])),
        compute_residuals=True, gaze_observations=gaze)
    if geometry is None:
        geometry = read_json(spatial_dir / 'geometry.json')

    atlas_h = spatial_dir / 'B3_H_spatial_atlas.jpg'
    atlas_guide = spatial_dir / 'B3_heatmap_guide_atlas.jpg'
    atlas_rgb = spatial_dir / 'B3_RGB_atlas.jpg'
    atlas_f = spatial_dir / 'B3_F_spatial_atlas.jpg'
    view_atlas([{**frame, 'path': frame['rendered_h_path']} for frame in frames]).save(atlas_h, quality=94)
    view_atlas([{**frame, 'path': frame['heatmap_guide_path']} for frame in frames]).save(atlas_guide, quality=94)
    view_atlas(frames).save(atlas_rgb, quality=94)
    view_atlas([{**frame, 'path': frame['rendered_f_path']} for frame in frames]).save(atlas_f, quality=94)
    use_panorama = geometry.get('status') == 'VALIDITY_UNVERIFIED'
    if use_panorama:
        paths = {
            'B-H': [str(spatial_dir / 'temporal_heatmap.png'), str(spatial_dir / 'heatmap_overlay.png')],
            'B-F': [str(spatial_dir / 'temporal_heatmap.png'), str(spatial_dir / 'foveation.png')],
        }
        heat = np.load(spatial_dir / 'attention_and_provenance.npz')['H_unnormalized']
        source_ids = sorted({fid for row in geometry['source_views'] for fid in row['source_frame_ids']})
        shared = {'expected_center': weighted_center(heat), 'source_frame_ids': source_ids,
                  'time_range': geometry.get('temporal_layer_range_seconds')}
        views = {
            'B-H': [{'view_id': 'B_HEATMAP', **shared}, {'view_id': 'B_PANORAMA', **shared}],
            'B-F': [{'view_id': 'B_HEATMAP', **shared}, {'view_id': 'B_FOVEATION', **shared}],
        }
        representation = 'B1_directional_panorama'
    else:
        paths = {'B-H': [str(atlas_guide), str(atlas_rgb)], 'B-F': [str(atlas_f)]}
        views = {group: [
            {'view_id': frame['view_id'], 'expected_center': frame['expected_attention_center_uv'],
             'source_frame_ids': frame.get('source_frame_ids', [frame['source_frame_id']]),
             'time_range': [frame['time_seconds'], frame['time_seconds']]}
            for frame in frames] for group in ('B-H', 'B-F')}
        representation = 'B3_spatial_atlas'
    save(spatial_dir / 'model_input_index.json', {
        'projection_type': representation, 'B1_attempt': geometry,
        'B2': {'status': 'NOT_AVAILABLE', 'reason': 'no reliable depth or correspondable scene mesh'},
        'B3_atlas_images': {'B-H': str(atlas_h), 'B-F': str(atlas_f)},
        'model_input_images': paths, 'model_input_views': views,
        'source_gaze_projection_audit': gaze_audit,
    })
    return spatial_dir, geometry, paths, views, representation


def _prompt(ctx, mention, condition, views, media_kind, *, projection_type=None):
    scope, candidates, candidate_ids = _candidate_scope(ctx, mention)
    if media_kind == 'video':
        media_note = 'The supplied video contains the listed views in chronological PTS order.'
    elif condition == 'A-H':
        media_note = (
            'Images are ordered pairs for each key time: first a standalone grayscale heatmap, '
            'then its corresponding untouched RGB frame. Heatmap brightness encodes accumulated '
            'gaze history available up to that frame. Pair IDs and timestamps below bind each map '
            'to its RGB frame; inspect the map, then the paired RGB for appearance and action.')
    elif projection_type == 'B1_directional_panorama':
        media_note = (
            'Image 1 is a cumulative spherical gaze heatmap: brightness encodes accumulated valid evidence; '
            'blue-to-yellow hue encodes weighted mean observation time, not full chronology. Image 2 is an '
            'equirectangular panorama from those source frames. Dark gray areas are unknown/unobserved. '
            'B-H uses a mild red heat overlay; B-F uses attention-guided foveation. Image 1 guides image 2.')
    elif condition == 'B-H':
        media_note = (
            'Image 1 is an atlas of standalone grayscale gaze heatmaps. Image 2 is the matching '
            'RGB spatial_atlas: both have exactly the same tile layout and time/view labels. '
            'This is the documented fallback after a directional-panorama validity failure, '
            'not a panorama. Estimate each heatmap center inside its own tile [0,1] coordinates, '
            'then inspect the matching RGB tile.')
    else:
        media_note = (
            'The image is a time-labeled spatial_atlas of the same real RGB views. Each tile keeps a stable '
            'view_id and source timestamp. It is not a panorama. B-H tiles show a mild heat overlay; '
            'B-F tiles show attention-guided foveation.')
    if condition == 'C0':
        cue_note = ('C0 contains raw RGB only. No gaze heatmap or foveation was rendered. Do not infer '
                    'attention from the scene; report none if no explicit cue is visible.')
    elif condition == 'A-H' or (condition == 'B-H' and projection_type != 'B1_directional_panorama'):
        if condition == 'A-H':
            cue_note = ('The first image of each pair is a grayscale gaze heatmap. Brighter pixels encode '
                        'stronger accumulated gaze evidence, not an object label or target probability. '
                        'Inspect its spatial center before examining the paired RGB image.')
        else:
            cue_note = ('The first atlas contains grayscale per-keyframe gaze heatmaps. Brighter pixels encode '
                        'stronger accumulated evidence, not an object label or target probability. Estimate '
                        'each map center in its labeled tile, then inspect the same tile in the RGB atlas.')
    elif condition.endswith('-H'):
        cue_note = ('A mild red heat overlay is constructed from projected gaze history. More saturated red '
                    'means stronger accumulated numerical attention evidence, not an object label or target '
                    'probability. Inspect the pixels, estimate its center, then compare the region with the '
                    'mention using RGB, action, and relations.')
    else:
        cue_note = ('Foveation preserves relatively sharper regions around projected gaze history; blur is '
                    'only a rendering choice and does not mean absence or occlusion. Inspect the pixels, '
                    'estimate the sharper region center, then compare it with the mention.')
    view_ids = [row['view_id'] for row in views]
    start, end = mention.get('span', [0, 0])
    view_contract = {
        'required_attention_trace_view_order': view_ids,
        'attention_trace_row_schema': {
            'view_id': 'exact listed ID', 'cue_type': 'none | heatmap | foveation | unclear',
            'cue_center_uv': '[u,v] in that view, top-left origin, or null',
            'cue_confidence': '0..1', 'cue_observation': 'what is visible'},
        'attention_trace_rule': 'Return exactly one row per listed view. Do not guess centers.',
        'evidence_row_schema': {
            'mention_id': mention['mention_id'], 'candidate_id': 'exact allowed candidate ID',
            'support_score': '0..1', 'confidence': '0..1',
            'frame_id': 'exact listed view_id', 'explanation': 'brief visual justification'},
        'visual_observation_row_schema': {
            'mention_id': mention['mention_id'], 'frame_id': 'exact listed view_id',
            'description': 'brief visible RGB observation', 'confidence': '0..1'},
        'candidate_scope': scope, 'allowed_candidate_ids': candidate_ids,
        'scene_local_candidate_table': candidates,
        'views': [{key: value for key, value in row.items()
                   if key not in ('expected_center', 'heatmap_sha256')} for row in views],
    }
    if condition == 'A-H':
        view_contract['image_order_contract'] = (
            'A-H uploads two consecutive images per view: standalone heatmap then paired RGB; indices are 1-based')
    return '''You are the Visual agent in an offline, mention-first closed-set referring system.
Solve exactly one mention. Candidate IDs come only from this isolated scene and this mention.
Do not use numeric gaze values, world coordinates, camera pose, ground truth, candidate distance,
or anchor IDs from another scene. The attention display is weak corroboration, never target identity
or probability. Prefer actual RGB appearance, action, and relations; report ambiguity when needed.

''' + media_note + '\n' + cue_note + '''
First audit whether the cue is perceptible in each listed view. Then solve the mention using the
same RGB evidence. Cite only listed views and exact allowed candidate IDs. If the candidate list
is empty, evidence MUST be empty and return observations only.
Return compact JSON with keys attention_trace, evidence, visual_observations, identity_status.
There must be exactly one attention_trace row per required view in the listed order; use none or
unclear rather than inventing a cue or center. No mention-level ground truth is provided.

Input: ''' + json.dumps({
        'scene_key': ctx.scene_key, 'instruction': ctx.text,
        'highlighted_instruction': ctx.text[:start] + '[MENTION]' + ctx.text[start:end] + '[/MENTION]' + ctx.text[end:],
        'mention': mention, 'mention_candidate_ids': candidate_ids,
        'candidate_scope': scope, 'mention_scene_local_candidate_table': candidates,
        'visual_views': view_contract,
    }, ensure_ascii=False)


def _prediction(parsed, allowed_ids, expected_views):
    parsed = parsed if isinstance(parsed, dict) else {}
    trace = parsed.get('attention_trace', [])
    if not isinstance(trace, list):
        trace = []
    by_id = {row.get('view_id'): row for row in trace if isinstance(row, dict)}
    errors = []
    for view in expected_views:
        row = by_id.get(view['view_id'], {})
        point, expected = row.get('cue_center_uv'), view.get('expected_center')
        observed = None
        if isinstance(point, (list, tuple)) and len(point) == 2:
            try:
                x, y = map(float, point)
                if 0 <= x <= 1 and 0 <= y <= 1 and math.isfinite(x+y):
                    observed = [x, y]
            except (TypeError, ValueError):
                pass
        error = math.dist(expected, observed) if expected is not None and observed is not None else None
        if error is not None:
            errors.append(error)
        row['expected_center_for_audit_only'] = expected
        row['center_error_uv_distance'] = error
    positive = []
    evidence = parsed.get('evidence', [])
    for row in evidence if isinstance(evidence, list) else []:
        if not isinstance(row, dict) or row.get('candidate_id') not in allowed_ids:
            continue
        try:
            if float(row.get('support_score', 0)) > 0 and float(row.get('confidence', 0)) > 0:
                positive.append(row['candidate_id'])
        except (TypeError, ValueError):
            continue
    return {
        'attention_trace': trace, 'trace_rows_returned': len(trace),
        'trace_views_with_reported_cue': sum(row.get('cue_type') in ('heatmap', 'foveation')
                                             for row in trace if isinstance(row, dict)),
        'trace_views_with_center': sum(isinstance(row.get('cue_center_uv'), (list, tuple))
                                       for row in trace if isinstance(row, dict)),
        'attention_center_error_mean': float(np.mean(errors)) if errors else None,
        'positive_candidate_ids': sorted(set(positive)),
        'identity_status': parsed.get('identity_status'),
        'visual_observations': parsed.get('visual_observations', []),
        'candidate_evidence': evidence,
    }


def _run_call(backend, ctx, condition, prompt, paths, media_kind, frames, views,
              allowed_ids, video_max_pixels):
    started = time.monotonic()
    key = f'{ctx.scene_key}/{ctx.event_id}/m0'
    if media_kind == 'video':
        parsed = backend.ask_video(key, f'organization_{condition}', prompt, paths, frames,
            max_new_tokens=2048, video_path=ctx.video_path,
            video_time_range=[frames[0]['time_seconds'], frames[-1]['time_seconds']],
            video_max_pixels=video_max_pixels)
    else:
        parsed = backend.ask(key, f'organization_{condition}', prompt, paths, max_new_tokens=2048)
    request_path = Path(backend.requests[-1])
    response_path = request_path if request_path.exists() else request_path.with_suffix('.failed.json')
    response = read_json(response_path) if response_path.exists() else {}
    prediction = _prediction(parsed, allowed_ids, views)
    input_rows = []
    for filename in paths:
        with Image.open(filename) as image:
            dimensions = image.size
        input_rows.append({'path': str(filename), 'sha256': sha_file(filename),
                           'size_px': list(dimensions), 'bytes': Path(filename).stat().st_size})
    audit = response.get('processor_audit') or {}
    trace = audit.get('visual_tensor_trace') or response.get('visual_tensor_trace')
    request_manifest = request_path.with_suffix('.request.json')
    return {
        'condition': condition, 'backend': 'local_transformers_' + media_kind,
        'request_path': str(request_path),
        'request_manifest_path': str(request_manifest if request_manifest.exists() else request_path),
        'raw_response': response.get('response'), 'parsed_response': parsed,
        'input_images': input_rows, 'processor_audit': audit,
        'visual_tensor_trace': trace, 'elapsed_seconds': response.get('elapsed_seconds'),
        'inference_seconds': response.get('inference_seconds'),
        'wall_seconds_this_invocation': time.monotonic()-started,
        'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
        'expected_views': views,
        'citation_source_map': {view['view_id']: {
            key: view.get(key) for key in ('source_frame_ids', 'time_range', 'source_times_seconds')}
            for view in views},
        **prediction,
    }


def _write_report(out, summary):
    lines = [
        '# Visual attention organization probe', '',
        '本次只审计 Visual 视觉引导的可见性与输入传递，不加载 GT、不运行 Coordinator，不能解读为 Any-hit/F1/Exact-set 结果。', '',
        f"固定样本：{summary['completed_samples']} 条；每条使用相同的 {summary['source_frame_count']} 个真实 PTS 源帧。",
        'A-H 每个关键时刻将独立灰度热图与其原始 RGB 帧成对发送；A-F 是同一 8 帧序列的 foveation。',
        'B 先尝试相机姿态方向全景，检查不通过时显式以带位姿索引的 spatial_atlas 回退。',
        'A-H 输入 16 张图，视觉 token 预算高于 8 帧视频条件；它只用于线索可读性诊断，不作预算公平的效果比较。',
        '请求保留图像哈希、processor 视觉 token、视觉编码器入口张量哈希、原始响应与耗时。', '',
        '## 每样本几何状态', '',
    ]
    for sample in summary['samples']:
        reasons = ', '.join(sample.get('b1_failure_reasons', [])) or '无明显失败项，但仍不宣称世界几何准确'
        lines.append(f"- `{sample['scene_key']}/{sample['event_id']}`：实际 B=`{sample['projection_type']}`；"
                     f"B1=`{sample['b1_status']}`；{reasons}。")
    lines.extend(['', '## 输入与可见性检查', ''])
    for sample in summary['samples']:
        lines.append(f"- `{sample['scene_key']}/{sample['event_id']}`")
        for condition in CONDITION_ORDER:
            row = sample['conditions'].get(condition)
            if not row or 'error' in row:
                lines.append(f"  - {condition}: NOT_RUN {row.get('error') if row else ''}")
                continue
            audit, trace = row.get('processor_audit') or {}, row.get('visual_tensor_trace') or {}
            tokens = audit.get('processor_video_tokens_estimate', audit.get('processor_image_tokens_estimate'))
            centers = [item.get('cue_center_uv') for item in row.get('attention_trace', [])
                       if isinstance(item, dict) and item.get('cue_center_uv') is not None]
            normalized_centers = [point for point in centers
                if isinstance(point, (list, tuple)) and len(point) == 2
                and all(isinstance(value, (int, float)) and 0 <= value <= 1 for value in point)]
            invalid_centers = len(centers)-len(normalized_centers)
            unique_centers = len({tuple(point) for point in normalized_centers})
            lines.append(f"  - {condition}: cue={row.get('trace_views_with_reported_cue')}/"
                         f"{row.get('trace_rows_returned')} views; center_error={row.get('attention_center_error_mean')}; "
                         f"unique_normalized_centers={unique_centers}; nonnormalized_centers={invalid_centers}; "
                         f"images={len(row.get('input_images', []))}; vision_tokens={tokens}; "
                         f"encoder_input_verified={trace.get('input_consumption_verified')}; "
                         f"seconds={row.get('elapsed_seconds')}")
    lines.extend([
        '',
        '模型自述的 cue 与几何中心误差用于检查它是否看见并定位了显示线索，不足以证明线索改善目标推理；prompt 明确说明了线索含义。',
        'B1 忽略平移，仅作方向投影；缺姿态、配准残差过大或证据不足时保留失败记录并切到 spatial_atlas。无深度时未构造 B2。',
        '所有输入均为离线完整交互。该小样本验证输入链路与线索可读性，不代表完整数据集结论。', '',
        '## 复现命令', '',
        f"```bash\npython -m MultiAgent.run_v3 --manifest {summary['source_manifest']} \\\n  --output-dir {summary['output_dir']} --cache-dir {summary['cache_dir']} \\\n  --visual-study organization --study-config {summary['config_path']} \\\n  --qwen-model {summary['model']}\n```", '',
    ])
    (out / 'AUDIT.md').write_text('\n'.join(lines), encoding='utf-8')


def run(args, config):
    config_path = Path(args.study_config)
    if not config_path.is_absolute():
        config_path = (ROOT / config_path).resolve()
    config = dict(config)
    old_root = (ROOT / config['gaze_source_run']).resolve()
    local_root = (ROOT / config['local_visual_source_run']).resolve()
    source_manifest_path = Path(args.manifest).resolve()
    source_manifest = read_json(source_manifest_path)
    requested = {tuple(row) for row in config['sample_keys']}
    items = [row for row in source_manifest['samples']
             if (row['scene_key'], row['event_id']) in requested]
    found = {(row['scene_key'], row['event_id']) for row in items}
    if found != requested:
        raise ValueError(f'Fixed sample keys absent from source manifest: {sorted(requested-found)}')
    items.sort(key=lambda row: config['sample_keys'].index([row['scene_key'], row['event_id']]))
    if args.limit:
        items = items[:args.limit]
    manifest = {'seed': config.get('seed', 20261022),
                'selection': 'fixed sample keys in config; no GT or prediction filtering',
                'samples': items}
    contexts = load_contexts(manifest, load_ground_truth=False)
    context_by_key = {(ctx.scene_key, ctx.event_id): ctx for ctx in contexts}
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = (ROOT / out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    model_path = Path(config.get('model_path', MODEL_DEFAULT))
    if not model_path.is_absolute():
        model_path = (WORKSPACE / model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f'Local Qwen model not found: {model_path}')
    if args.base_url:
        raise ValueError('Organization audit is pinned to local Qwen3-VL-8B; remove --base-url')
    config_record = {**config, 'config_path': str(config_path),
        'source_manifest': str(source_manifest_path), 'model_path': str(model_path),
        'model_config_sha256': sha_file(model_path / 'config.json'),
        'local_backend': True, 'namespace': config.get('namespace', 'visual_organization_qwen8b_20261022_v1'),
        'conditions': list(CONDITION_ORDER), 'offline_full_interaction': True,
        'ground_truth_loaded': False, 'coordinator_run': False,
        'frame_selection': 'same fixed PTS frame IDs in every condition; uniform strata from frozen local event frames',
        'prompting': 'stepwise heatmap semantics in prompt; standalone keyframe heatmaps in A-H; no drawn marks',
        'keyframe_heatmap_contract': 'one standalone grayscale map immediately before its paired untouched RGB image; each pair shares source frame ID and PTS',
        'b2_depth_reprojection': 'NOT_AVAILABLE: no reliable depth or correspondable scene mesh',
    }
    save(out / 'manifest.json', manifest)
    save(out / 'config.json', config_record)
    backend = None if args.cpu_only else QwenBackend(
        args.cache_dir, str(model_path), namespace=config_record['namespace'], local_only=True,
        video_max_pixels=int(config.get('video_max_pixels', 8_388_608)), trace_visual_inputs=True)
    if backend:
        import torch
        torch.set_num_threads(4)
        torch.manual_seed(int(config.get('seed', 20261022)))

    results, failures = [], []
    for index, item in enumerate(items, 1):
        ctx = context_by_key[(item['scene_key'], item['event_id'])]
        sample_dir = out / 'samples'
        print(f"[{index}/{len(items)}] {ctx.scene_key}/{ctx.event_id}: freeze 8 PTS views", flush=True)
        try:
            referential, mention, frames, groups, rendered, _, freeze = prepare_sample(
                ctx, sample_dir, args, config, old_root, local_root)
            sample_root = sample_dir / ctx.scene_key / ctx.event_id
            source_rows = []
            for frame in frames:
                fid = frame['frame_id']
                source_rows.append({**frame, 'path': rendered['R'][fid],
                    'rendered_h_path': rendered['H_local'][fid],
                    'rendered_f_path': rendered['F_local'][fid],
                    'expected_attention_center_uv': freeze['attention_maps'][fid]['gaze']['expected_attention_center_uv']})
            for frame in source_rows:
                heat = np.load(sample_root / 'inputs' / frame['frame_id'] / 'attention_map.npz')
                guide_path = sample_root / 'inputs' / frame['frame_id'] / 'heatmap_guide.png'
                heat_guide(heat['H_unnormalized']).save(guide_path)
                frame['heatmap_guide_path'] = str(guide_path)
            old_sample = old_root / 'samples' / ctx.scene_key / ctx.event_id
            raw_records = read_json(old_sample / 'gaze_records.json')
            offset = freeze.get('telemetry_to_video_offset_seconds') if freeze['clock_available'] else None
            spatial_dir, geometry, b_paths, b_views, projection_type = _prepare_spatial_inputs(
                sample_root, source_rows, raw_records, offset, config.get('gaze', {}),
                (frames[0]['width'], frames[0]['height']))
            frame_by_id = {row['frame_id']: row for row in frames}
            video_views = [{
                'view_id': group['view_id'], 'source_frame_ids': group['source_frame_ids'],
                'source_times_seconds': [frame_by_id[fid]['time_seconds'] for fid in group['frame_ids']],
                'time_range': group['source_time_range'],
                'expected_center': _attention_center_for_group(sample_root, group),
            } for group in groups]
            keyframe_views, keyframe_heatmaps = [], []
            for index, frame in enumerate(source_rows):
                keyframe_heatmaps.extend([frame['heatmap_guide_path'], frame['path']])
                keyframe_views.append({
                    'view_id': frame['view_id'],
                    'heatmap_image_index': 2*index+1, 'rgb_image_index': 2*index+2,
                    'source_frame_ids': frame.get('source_frame_ids', [frame['source_frame_id']]),
                    'source_times_seconds': [frame['time_seconds']],
                    'time_range': [frame['time_seconds'], frame['time_seconds']],
                    'expected_center': frame['expected_attention_center_uv'],
                    'heatmap_sha256': sha_file(frame['heatmap_guide_path']),
                })
            inputs = {
                'C0': ([rendered['R'][row['frame_id']] for row in frames], 'video', video_views),
                'A-H': (keyframe_heatmaps, 'image', keyframe_views),
                'B-H': (b_paths['B-H'], 'image', b_views['B-H']),
                'A-F': ([rendered['F_local'][row['frame_id']] for row in frames], 'video', video_views),
                'B-F': (b_paths['B-F'], 'image', b_views['B-F']),
            }
            sample_result = {
                'scene_key': ctx.scene_key, 'event_id': ctx.event_id,
                'mention_id': mention['mention_id'], 'mention': mention,
                'candidate_ids': list(mention.get('candidate_ids') or []),
                'source_frame_count': len(frames), 'source_frames': frames,
                'temporal_groups': video_views,
                'frozen_referential_source': freeze.get('referential_source'),
                'frozen_referential_sha256': freeze.get('referential_sha256'),
                'frozen_input_audit': str(sample_root / 'frozen_input_audit.json'),
                'projection_type': projection_type, 'b1_status': geometry.get('status'),
                'b1_failure_reasons': geometry.get('failure_reasons', []),
                'b1_geometry_path': str(spatial_dir / 'geometry.json'),
                'b1_gaze_audit_path': str(spatial_dir / 'gaze_projection_audit.json'),
                'conditions': {},
            }
            if backend:
                for condition in CONDITION_ORDER:
                    paths, kind, views = inputs[condition]
                    prompt = _prompt(ctx, mention, condition, views, kind,
                        projection_type=projection_type if condition.startswith('B-') else None)
                    print(f"  {condition}: {len(paths)} {kind} input(s)", flush=True)
                    try:
                        row = _run_call(backend, ctx, condition, prompt, paths, kind,
                            frames, views, set(mention.get('candidate_ids') or []),
                            int(config.get('video_max_pixels', 8_388_608)))
                        sample_result['conditions'][condition] = row
                        save(sample_root / 'predictions' / f'{condition}.json', row)
                        verified = (row.get('visual_tensor_trace') or {}).get('input_consumption_verified')
                        print(f"    cue={row['trace_views_with_reported_cue']}/{row['trace_rows_returned']} "
                              f"vision_verified={verified}", flush=True)
                    except Exception as exc:
                        failure = {'condition': condition, 'error': f'{type(exc).__name__}: {exc}'}
                        sample_result['conditions'][condition] = failure
                        failures.append({'sample': f'{ctx.scene_key}/{ctx.event_id}', **failure})
                        save(sample_root / 'predictions' / f'{condition}_failure.json', failure)
            save(sample_root / 'organization_sample.json', sample_result)
            results.append(sample_result)
        except Exception as exc:
            failure = {'scene_key': ctx.scene_key, 'event_id': ctx.event_id,
                       'error': f'{type(exc).__name__}: {exc}'}
            failures.append(failure)
            save(sample_dir / ctx.scene_key / ctx.event_id / 'organization_failure.json', failure)
            print(f"  FAILED: {failure['error']}", flush=True)
    summary = {
        'status': 'COMPLETE' if not failures else 'COMPLETE_WITH_FAILURES',
        'version': 'visual_organization_guidance_probe_20261022_v1',
        'planned_samples': len(items), 'completed_samples': len(results),
        'failed_calls_or_samples': len(failures), 'failures': failures,
        'source_frame_count': int(config.get('source_frame_count', 8)),
        'conditions': list(CONDITION_ORDER), 'samples': results,
        'call_audit': backend.call_records if backend else [],
        'ground_truth_loaded': False, 'coordinator_run': False,
        'evaluation_scope': 'cue-perception and visual-input delivery only; no mention or final-set metrics',
        'source_manifest': str(source_manifest_path), 'output_dir': str(out),
        'cache_dir': str(Path(args.cache_dir).resolve()), 'config_path': str(config_path),
        'model': str(model_path), 'offline_full_interaction': True,
    }
    save(out / 'summary.json', summary)
    _write_report(out, summary)
    print(json.dumps({'completed_samples': len(results), 'failures': len(failures),
                      'model_calls': len(backend.call_records) if backend else 0}, ensure_ascii=False),
          flush=True)
    return summary
