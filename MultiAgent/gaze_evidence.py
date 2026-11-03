"""Time-windowed world gaze projection and fixed-budget RGB presentations."""
from __future__ import annotations

from datetime import datetime
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from .baseline import module


DEFAULTS = {
    'instant_max_delta_seconds': 0.18,
    'local_history_seconds': 2.0,
    'utterance_pre_seconds': 0.5,
    'utterance_post_seconds': 0.5,
    'max_gap_seconds': 0.25,
    'decay': 'uniform',
    'decay_tau_seconds': 1.0,
    'kernel_sigma_degrees': 2.0,
    'display_scale': 2.0,
    'overlay_max_alpha': 0.28,
    'foveation_min_clear_fraction': 0.55,
    'foveation_max_blur_sigma_px': 18.0,
    'foveation_downsample_factors': [2, 4, 8],
}


def _vec(value, keys=('x', 'y', 'z')):
    if not isinstance(value, dict):
        return None
    try:
        result = tuple(float(value[k]) for k in keys)
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(x) for x in result):
        return None
    return result


def _norm(v):
    return math.sqrt(sum(x*x for x in v))


def _unit_quaternion(value):
    q = _vec(value, ('x', 'y', 'z', 'w'))
    if q is None or _norm(q) < 1e-8:
        return None
    norm = _norm(q)
    return tuple(x/norm for x in q)


def _slerp(a, b, t):
    dot = sum(x*y for x, y in zip(a, b))
    if dot < 0:
        b = tuple(-x for x in b)
        dot = -dot
    if dot > 0.9995:
        return _unit_quaternion(dict(zip(('x', 'y', 'z', 'w'),
                                        (x+t*(y-x) for x, y in zip(a, b)))))
    angle = math.acos(max(-1., min(1., dot)))
    sine = math.sin(angle)
    return tuple((math.sin((1-t)*angle)*x + math.sin(t*angle)*y)/sine
                 for x, y in zip(a, b))


def timestamp_seconds(value):
    try:
        text = str(value).replace('Z', '+00:00')
        text = __import__('re').sub(r'\.(\d{6})\d+(?=[+-]\d{2}:\d{2}$)', r'.\1', text)
        return datetime.fromisoformat(text).timestamp()
    except (TypeError, ValueError, OverflowError):
        return None


def gaze_records(ctx):
    api = module('api')
    from .telemetry import load_telemetry
    raw = load_telemetry(Path(ctx.json_path))
    original_indices = {}
    for index, sample in enumerate(raw):
        original_indices.setdefault(str(sample.get('timestamp', '')), []).append(index)
    samples = api.collect_timed_samples(raw)
    rows = []
    for sorted_index, (seconds, sample) in enumerate(samples):
        stamp = str(sample.get('timestamp', ''))
        indices = original_indices.get(stamp, [])
        source_index = indices.pop(0) if indices else sorted_index
        eye = sample.get('eyeGaze') if isinstance(sample.get('eyeGaze'), dict) else {}
        point = _vec(eye.get('gazePoint'))
        if point is not None and _norm(point) < 1e-8:
            point = None  # released exporter uses all-zero mesh intersection for no hit
        vector = _vec(eye.get('gazeVector'))
        if vector is not None and _norm(vector) < 1e-8:
            vector = None
        open_eye = eye.get('isEyeOpen') is not False
        validity = ('valid' if open_eye and (point is not None or vector is not None)
                    else 'invalid_sample')
        rows.append({'source_index': source_index, 'telemetry_time_seconds': float(seconds),
                     'timestamp': stamp, 'timestamp_epoch_seconds': timestamp_seconds(stamp),
                     'gaze_point_world': point, 'gaze_vector_world': vector,
                     'gaze_origin_world': _vec(eye.get('gazeOrigin')),
                     'camera_gaze_direction': _vec(eye.get('cameraGazeDirection')),
                     'eye_open': eye.get('isEyeOpen'), 'validity': validity,
                     'camera_position': _vec(sample.get('cameraPosition')),
                     'camera_rotation_xyzw': _unit_quaternion(sample.get('cameraRotation')),
                     'camera_fov_degrees': _finite(sample.get('cameraFOV')),
                     'quality_available': any(k in eye for k in ('confidence', 'quality', 'validityScore')),
                     'hand_tracked': bool((sample.get('handData') or {}).get('isLeftHandTracked')
                                          or (sample.get('handData') or {}).get('isRightHandTracked'))})
    return rows


def _finite(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def interpolate_camera_pose(records, target_time, max_gap_seconds=0.25):
    if not records or target_time is None:
        return None
    before = next((row for row in reversed(records)
                   if row['telemetry_time_seconds'] <= target_time
                   and row['camera_position'] is not None and row['camera_rotation_xyzw'] is not None
                   and row['camera_fov_degrees'] is not None), None)
    after = next((row for row in records
                  if row['telemetry_time_seconds'] >= target_time
                  and row['camera_position'] is not None and row['camera_rotation_xyzw'] is not None
                  and row['camera_fov_degrees'] is not None), None)
    if before is None or after is None:
        return None
    left, right = before['telemetry_time_seconds'], after['telemetry_time_seconds']
    gap = right-left
    if gap > max_gap_seconds:
        return None
    ratio = 0. if gap <= 1e-9 else (target_time-left)/gap
    position = tuple(a+(b-a)*ratio for a, b in zip(before['camera_position'], after['camera_position']))
    rotation = _slerp(before['camera_rotation_xyzw'], after['camera_rotation_xyzw'], ratio)
    fov = before['camera_fov_degrees'] + (after['camera_fov_degrees']-before['camera_fov_degrees'])*ratio
    return {'position_world': position, 'rotation_xyzw': rotation, 'fov_vertical_degrees': fov,
            'pose_interpolation': 'linear_position_slerp_rotation',
            'pose_source_indices': [before['source_index'], after['source_index']],
            'pose_source_times': [left, right], 'pose_bracket_gap_seconds': gap}


def project_gaze(record, pose, width, height, visibility_test=None):
    if record['validity'] != 'valid':
        return {'status': 'invalid_sample', 'visibility': 'visibility_unknown'}
    if pose is None:
        return {'status': 'camera_pose_unavailable', 'visibility': 'visibility_unknown'}
    api = module('api')
    q = pose['rotation_xyzw']
    inverse = api.quaternion_conjugate(q)
    point = record['gaze_point_world']
    if point is not None:
        delta = tuple(v-o for v, o in zip(point, pose['position_world']))
        local = api.rotate_vector_by_quaternion(delta, inverse)
        source = 'gazePoint_finite_world_intersection'
    else:
        local = api.rotate_vector_by_quaternion(record['gaze_vector_world'], inverse)
        source = 'gazeVector_world_direction_infinite_distance'
    x, y, z = local
    if z <= 1e-6:
        return {'status': 'behind_camera', 'visibility': 'visibility_unknown',
                'projection_source': source, 'camera_xyz': local}
    tan = math.tan(math.radians(pose['fov_vertical_degrees'])/2.)
    aspect = width/height
    if tan <= 1e-8 or aspect <= 0:
        return {'status': 'invalid_camera_intrinsics', 'visibility': 'visibility_unknown'}
    # Unity Camera.fieldOfView is vertical; pixels have square aspect and origin is top-left.
    u = 0.5 + x/(2*z*tan*aspect)
    v = 0.5 - y/(2*z*tan)
    px, py = u*width, v*height
    result = {'projection_source': source, 'camera_xyz': local,
              'pixel_xy': [px, py], 'normalized_uv': [u, v],
              'visibility': 'visibility_unknown', 'visibility_method': 'frustum_only_no_depth_or_mesh'}
    if not (math.isfinite(px) and math.isfinite(py)):
        result.update(status='invalid_projection', visibility='visibility_unknown')
    elif u < 0 or u >= 1 or v < 0 or v >= 1:
        result.update(status='outside_frame', visibility='visibility_unknown')
    else:
        result['status'] = 'projected_frustum_only'
        if visibility_test is not None:
            verdict = visibility_test(record, pose, result)
            if verdict not in ('verified_visible', 'verified_occluded', 'visibility_unknown'):
                raise ValueError('invalid visibility-test verdict')
            result['visibility'] = verdict
            result['visibility_method'] = 'external_depth_or_replayable_geometry'
            if verdict == 'verified_occluded':
                result['status'] = 'verified_occluded'
    return result


def _support_weights(records, selected, start, end, max_gap, decay, tau, target):
    times = [r['telemetry_time_seconds'] for r in records]
    weights = {}
    for row in selected:
        t = row['telemetry_time_seconds']
        index = row['source_index']
        before = [v for v in times if v < t]
        after = [v for v in times if v > t]
        left_gap = min(t-before[-1], max_gap) if before else min(max_gap, max_gap/2)
        right_gap = min(after[0]-t, max_gap) if after else min(max_gap, max_gap/2)
        left = max(start, t-left_gap/2)
        right = min(end, t+right_gap/2)
        duration = max(0., right-left)
        if duration <= 0:
            duration = min(max_gap/2, max(0., end-start))
        temporal = math.exp(-max(0., target-t)/tau) if decay == 'exponential' and tau > 0 else 1.
        weights[index] = duration*temporal
    return weights


def select_gaze_observations(records, *, strategy, frame_time, event_window,
                             clock_available, config=None):
    cfg = {**DEFAULTS, **(config or {})}
    if not clock_available or frame_time is None:
        return {'selected': [], 'window': None, 'observation_cutoff': None,
                'status': 'clock_unavailable_full_rgb_fallback', 'protocol': 'offline'}
    valid = [r for r in records if r['validity'] == 'valid']
    if strategy == 'instant':
        nearest = min(valid, key=lambda r: abs(r['telemetry_time_seconds']-frame_time), default=None)
        if nearest is None or abs(nearest['telemetry_time_seconds']-frame_time) > cfg['instant_max_delta_seconds']:
            return {'selected': [], 'window': [frame_time-cfg['instant_max_delta_seconds'],
                                                frame_time+cfg['instant_max_delta_seconds']],
                    'candidate_sample_count': sum(abs(r['telemetry_time_seconds']-frame_time) <= cfg['instant_max_delta_seconds'] for r in records),
                    'candidate_invalid_count': sum(r['validity'] != 'valid' and abs(r['telemetry_time_seconds']-frame_time) <= cfg['instant_max_delta_seconds'] for r in records),
                    'observation_cutoff': frame_time, 'status': 'no_valid_gaze_within_instant_tolerance',
                    'protocol': 'offline'}
        selected = [nearest]
        candidates=[r for r in records if abs(r['telemetry_time_seconds']-frame_time) <= cfg['instant_max_delta_seconds']]
        interval = min(cfg['max_gap_seconds'], _median_gap(records))
        weights = {nearest['source_index']: interval}
        window = [frame_time-cfg['instant_max_delta_seconds'], frame_time+cfg['instant_max_delta_seconds']]
        cutoff = frame_time
    elif strategy == 'local_history':
        start = frame_time-cfg['local_history_seconds']
        window = [start, frame_time]
        selected = [r for r in valid if start <= r['telemetry_time_seconds'] <= frame_time]
        candidates=[r for r in records if start <= r['telemetry_time_seconds'] <= frame_time]
        weights = _support_weights(records, selected, start, frame_time,
                                    cfg['max_gap_seconds'], cfg['decay'],
                                    cfg['decay_tau_seconds'], frame_time)
        cutoff = frame_time
    elif strategy == 'utterance_window':
        start = event_window[0]-cfg['utterance_pre_seconds']
        end = event_window[1]+cfg['utterance_post_seconds']
        window = [start, end]
        selected = [r for r in valid if start <= r['telemetry_time_seconds'] <= end]
        candidates=[r for r in records if start <= r['telemetry_time_seconds'] <= end]
        weights = _support_weights(records, selected, start, end,
                                    cfg['max_gap_seconds'], cfg['decay'],
                                    cfg['decay_tau_seconds'], end)
        cutoff = end
    else:
        raise ValueError(f'unknown gaze evidence strategy: {strategy}')
    return {'selected': [{'record': r, 'weight_seconds': weights.get(r['source_index'], 0.)}
                         for r in selected if weights.get(r['source_index'], 0.) > 0],
            'candidate_sample_count':len(candidates),
            'candidate_invalid_count':sum(r['validity']!='valid' for r in candidates),
            'window': window, 'observation_cutoff': cutoff,
            'status': 'ok' if selected else 'no_valid_gaze_in_window',
            'protocol': 'offline', 'decay': cfg['decay']}


def _median_gap(records):
    diffs = [b['telemetry_time_seconds']-a['telemetry_time_seconds']
             for a, b in zip(records, records[1:])
             if 0 < b['telemetry_time_seconds']-a['telemetry_time_seconds'] <= 2.]
    return float(np.median(diffs)) if diffs else 0.1


def build_heatmap(ctx, records, *, frame_time_video, frame_time_telemetry, event_window,
                  clock_available, image_size, strategy, config=None, reference_pose=None):
    cfg = {**DEFAULTS, **(config or {})}
    width, height = image_size
    heat = np.zeros((height, width), dtype=np.float32)
    selected = select_gaze_observations(records, strategy=strategy,
        frame_time=frame_time_telemetry, event_window=event_window,
        clock_available=clock_available, config=cfg)
    projected = []
    valid_duration = 0.
    pose_cache = {frame_time_telemetry: reference_pose} if reference_pose is not None else {}
    sigma = max(1., height*math.radians(cfg['kernel_sigma_degrees']) /
                (2*math.tan(math.radians(_reference_fov(records))/2)))
    radius = max(2, int(math.ceil(sigma*3)))
    for item in selected['selected']:
        record, weight = item['record'], item['weight_seconds']
        if frame_time_telemetry not in pose_cache:
            pose_cache[frame_time_telemetry] = interpolate_camera_pose(records, frame_time_telemetry,
                                                                        cfg['max_gap_seconds'])
        pose = pose_cache[frame_time_telemetry]
        result = project_gaze(record, pose, width, height)
        projected.append({'source_index': record['source_index'],
                          'gaze_timestamp': record['timestamp'],
                          'gaze_time_seconds': record['telemetry_time_seconds'],
                          'frame_time_seconds': frame_time_video,
                          'weight_seconds': weight,
                          'gaze_point_world': record['gaze_point_world'],
                          'gaze_vector_world': record['gaze_vector_world'],
                          **result})
        if result.get('status') != 'projected_frustum_only':
            continue
        cx, cy = result['pixel_xy']
        x0, x1 = max(0, int(cx)-radius), min(width, int(cx)+radius+1)
        y0, y1 = max(0, int(cy)-radius), min(height, int(cy)+radius+1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        patch = np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*sigma*sigma)).astype(np.float32)
        heat[y0:y1, x0:x1] += patch*float(weight)
        valid_duration += float(weight)
    stats = {'strategy': strategy, 'protocol': 'offline',
             'source_frame_time_video_seconds': frame_time_video,
             'source_frame_time_telemetry_seconds': frame_time_telemetry,
             'observation_window_telemetry_seconds': selected['window'],
             'observation_cutoff_telemetry_seconds': selected['observation_cutoff'],
             'selected_sample_count': len(selected['selected']),
             'candidate_sample_count_in_window':selected.get('candidate_sample_count',0),
             'valid_projected_count': sum(p.get('status') == 'projected_frustum_only' for p in projected),
             'invalid_sample_count': selected.get('candidate_invalid_count',0),
             'behind_camera_count': sum(p.get('status') == 'behind_camera' for p in projected),
             'outside_frame_count': sum(p.get('status') == 'outside_frame' for p in projected),
             'visibility_unknown_count': sum(p.get('visibility') == 'visibility_unknown' for p in projected),
             'verified_visible_count': 0, 'verified_occluded_count': 0,
             'visibility_method': 'frustum_only_no_depth_or_replayable_mesh',
             'supported_world_surface': 'unsupported_dynamic_correspondence_unknown',
             'quality_available': any(r['quality_available'] for r in records),
             'valid_evidence_duration_seconds': valid_duration,
             'max_unnormalized_heat': float(heat.max()),
             'unnormalized_heat_sum': float(heat.sum()),
             'kernel_sigma_degrees': cfg['kernel_sigma_degrees'],
             'kernel_sigma_pixels_at_model_input': sigma,
             'kernel_semantics': 'engineering spatial tolerance; not probability or calibrated device noise',
             'projection_model': 'vertical_fov_pinhole_square_pixels_top_left_origin',
             'pose_interpolation': 'linear_position_slerp_rotation_max_gap_0.25s',
             'status': selected['status']}
    return heat, stats, projected


def _display_map(heat, scale):
    return np.clip(1.-np.exp(-np.maximum(heat, 0.)*scale), 0., 1.).astype(np.float32)


def render_presentation(image, heat, method, config=None):
    cfg = {**DEFAULTS, **(config or {})}
    source = image.convert('RGB')
    if not np.any(heat > 0):
        return source.copy(), {'status': 'full_rgb_fallback_no_effective_gaze',
                               'clear_fraction_floor': cfg['foveation_min_clear_fraction']}
    focus = _display_map(heat, cfg['display_scale'])
    rgb = np.asarray(source, dtype=np.float32)
    if method in ('H_inst', 'H_local', 'H_utterance'):
        alpha = (focus*cfg['overlay_max_alpha'])[..., None]
        color = np.zeros_like(rgb); color[..., 0] = 255.; color[..., 1] = 70.
        output = np.clip(rgb*(1-alpha)+color*alpha, 0, 255).astype(np.uint8)
        return Image.fromarray(output), {'status': 'gaze_history_heatmap_overlay',
            'max_alpha': cfg['overlay_max_alpha'], 'display_scale': cfg['display_scale'],
            'valid_heat_fraction': float(np.mean(heat > 0))}
    if method in ('F_inst', 'F_local', 'F_utterance'):
        factors=cfg['foveation_downsample_factors']
        levels=[]
        for factor in factors:
            reduced_size=(max(1,source.width//factor),max(1,source.height//factor))
            reduced=source.resize(reduced_size,Image.Resampling.BOX)
            levels.append(np.asarray(reduced.resize(source.size,Image.Resampling.BICUBIC),dtype=np.float32))
        if not levels:
            raise ValueError('foveation_downsample_factors must contain at least one scale')
        # Low attention selects coarser downsample/upsample levels; high attention
        # retains the sharp source. The shared floor preserves global RGB detail.
        level=np.clip((1.-focus)*len(levels),0.,len(levels)-1)
        lower=np.floor(level).astype(np.intp)
        upper=np.minimum(lower+1,len(levels)-1)
        mix=(level-lower)[...,None]
        level_stack=np.stack(levels,axis=0)
        yy,xx=np.indices((source.height,source.width))
        blurred=level_stack[lower,yy,xx]*(1.-mix)+level_stack[upper,yy,xx]*mix
        sharp = cfg['foveation_min_clear_fraction'] + (1-cfg['foveation_min_clear_fraction'])*focus
        output = np.clip(rgb*sharp[..., None]+blurred*(1-sharp[..., None]), 0, 255).astype(np.uint8)
        return Image.fromarray(output), {'status': 'heuristic_gaze_guided_soft_foveation',
            'clear_fraction_floor': cfg['foveation_min_clear_fraction'],
            'downsample_factors': factors, 'display_scale': cfg['display_scale'],
            'valid_heat_fraction': float(np.mean(heat > 0)),
            'paper_method': False}
    return source.copy(), {'status': 'raw_rgb_control'}


def _reference_fov(records):
    values = [r['camera_fov_degrees'] for r in records if r['camera_fov_degrees']]
    return float(np.median(values)) if values else 90.
