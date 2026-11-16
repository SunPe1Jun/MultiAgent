"""PTS-based RGB/camera clock fitting and explicit source-frame pose contracts.

No GT, gaze, anchors or language enter fitting. Original images/evaluator stay intact.
Pure rotation is an approximation; accepted fits are candidates, not exposure calibration.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import time
import warnings

import numpy as np
from .camera_geometry import CameraPoseTimeline
from .video_timing import bound_frame_pts

from .context import load_contexts
from .head_motion_diagnostics import camera_records, probe_pts, verify_saved_frame_pts
from .projected2d_attention_study import save_json, sha_file, write_csv

VERSION = 'pts_camera_clock_v2'
DEFAULT_CONFIG = {'max_size': 640, 'anchor_step_seconds': .20, 'pair_lags_seconds': [.16, .36],
    'start_seconds': .4, 'block_seconds': 1.5, 'block_guard_seconds': .10,
    'offset_bounds_seconds': [-3., 6.], 'offset_grid_seconds': .02,
    'max_pose_gap_seconds': .25, 'minimum_pairs': 6, 'minimum_coverage': .5,
    'minimum_motion_px': 2., 'profile_delta_px': 1., 'maximum_profile_width_seconds': .15,
    'maximum_reverse_fit_difference_seconds': .15, 'maximum_validation_error_px': 15.,
    'maximum_validation_regression_fraction': .10, 'minimum_original_validation_retention': .8}


def source_frame_pose(timeline, frame, clock, *, diagnostic=False, allow_candidate=False):
    """Bind frame PTS to a reference pose, keeping uncalibrated fits diagnostic."""
    if frame.get('status') not in {'VERIFIED_EXACT_SOURCE_JPEG', 'VERIFIED_CONTAINER_PTS_SEQUENCE'}:
        return {'status': 'FRAME_TIME_UNVERIFIED', 'extrapolated': False}
    candidate_valid = clock.get('status') == 'VALIDATED_CANDIDATE'
    if diagnostic or (allow_candidate and candidate_valid):
        offset = clock.get('offset_seconds')
        pose_clock_status = ('REJECTED_CANDIDATE_DIAGNOSTIC_ONLY' if diagnostic and not candidate_valid
                             else 'VALIDATED_CANDIDATE')
    elif clock.get('original_offset_seconds') is not None:
        offset = clock['original_offset_seconds']
        if candidate_valid:
            pose_clock_status = 'ORIGINAL_CLOCK_REFERENCE_PTS_POSE'
        elif clock.get('original_clock_status') == 'ORIGINAL_CLOCK_SUPPORTED':
            pose_clock_status = 'ORIGINAL_CLOCK_SUPPORTED'
        else:
            pose_clock_status = 'ORIGINAL_CLOCK_FALLBACK_PTS_POSE'
    elif candidate_valid:
        offset = clock.get('offset_seconds')
        pose_clock_status = 'VALIDATED_CANDIDATE_NO_ORIGINAL_CLOCK'
    else:
        return {'status': 'NO_CLOCK_OFFSET', 'reasons': clock.get('rejection_reasons', []),
                'candidate_rejected': True, 'extrapolated': False}
    if offset is None or not np.isfinite(offset):
        return {'status': 'CLOCK_UNAVAILABLE', 'extrapolated': False}
    video_time = frame.get('actual_container_pts_seconds')
    if video_time is None or not np.isfinite(video_time):
        return {'status': 'FRAME_TIME_UNVERIFIED', 'extrapolated': False}
    pose = timeline.pose_at(video_time - offset)
    return {**pose, 'video_frame_pts_seconds': video_time, 'clock_offset_seconds': offset,
            'clock_status': pose_clock_status, 'candidate_clock_status': clock['status'],
            'original_clock_status': clock.get('original_clock_status'),
            'original_clock_offset_source': clock.get('original_offset_source'),
            'candidate_offset_seconds': clock.get('offset_seconds'),
            'diagnostic_only': pose_clock_status == 'REJECTED_CANDIDATE_DIAGNOSTIC_ONLY',
            'clock_validated': pose_clock_status == 'VALIDATED_CANDIDATE',
            'attention_clock_supported': (pose_clock_status in {'VALIDATED_CANDIDATE',
                'VALIDATED_CANDIDATE_NO_ORIGINAL_CLOCK', 'ORIGINAL_CLOCK_SUPPORTED'}),
            'position_projection_clock_source': ('RGB_rotation_candidate' if pose_clock_status.startswith('VALIDATED_CANDIDATE')
                else 'original_manifest_clock'),
            'clock_convention': 'video_pts = telemetry_seconds + offset',
            'exposure_time_calibration': 'unverified_container_pts_vs_render_exposure'}


def block_partition(pair, config):
    """A pair must lie inside one guarded time block. Opposite folds share no frames."""
    block = int(pair['time_a'] // config['block_seconds'])
    start = block * config['block_seconds']
    guard = config['block_guard_seconds']
    if pair['time_a'] < start + guard or pair['time_b'] > start + config['block_seconds'] - guard:
        return 'guard'
    return 'train' if block % 2 == 0 else 'validation'


def track_pts_features(video, ffmpeg, ffprobe, config):
    import cv2
    cv2.setNumThreads(4)
    cv2.setRNGSeed(20261023)
    pts, _ = probe_pts(video, ffprobe)
    if len(pts) < 2 or not np.all(np.diff(pts) > 0):
        raise ValueError('Non-monotonic or insufficient decoded frame PTS')
    metadata = json.loads(subprocess.check_output([ffprobe, '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height', '-of', 'json', str(video)], timeout=60))['streams'][0]
    scale = min(1., config['max_size'] / max(metadata['width'], metadata['height']))
    width, height = round(metadata['width'] * scale), round(metadata['height'] * scale)
    desired_pairs, requested = set(), set()
    pts_array = np.array(pts)
    nearest = lambda value: int(np.argmin(np.abs(pts_array - value)))
    for anchor in np.arange(config['start_seconds'], pts[-1] - .15, config['anchor_step_seconds']):
        a = nearest(anchor)
        for lag in config['pair_lags_seconds']:
            if anchor + lag > pts[-1]:
                continue
            b = nearest(anchor + lag)
            if b > a:
                desired_pairs.add((a, b))
                requested.update((a, b))
    command = [ffmpeg, '-v', 'error', '-threads', '2', '-i', str(video), '-map', '0:v:0',
               '-an', '-sn', '-vf', f'scale={width}:{height},format=gray', '-fps_mode', 'passthrough',
               '-threads', '1', '-f', 'rawvideo', 'pipe:1']
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frames, index = {}, 0
    sift = cv2.SIFT_create(nfeatures=900)
    try:
        while True:
            payload = proc.stdout.read(width * height)
            if not payload:
                break
            if len(payload) != width * height:
                raise ValueError('Incomplete raw decoded frame')
            if index in requested:
                gray = np.frombuffer(payload, np.uint8).reshape(height, width)
                kp, desc = sift.detectAndCompute(gray, None)
                frames[index] = {'points': kp, 'descriptors': desc, 'gray_sha256': hashlib.sha256(payload).hexdigest()}
            index += 1
        stderr = proc.stderr.read().decode(errors='replace')
        if proc.wait(timeout=60) != 0:
            raise RuntimeError(stderr)
    finally:
        proc.stdout.close()
        proc.stderr.close()
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    if index != len(pts):
        raise ValueError(f'PTS/frame decode mismatch: {len(pts)} timestamps vs {index} images')
    matcher, pairs, rejected = cv2.BFMatcher(), [], Counter()
    for a, b in sorted(desired_pairs):
        fa, fb = frames[a], frames[b]
        if fa['descriptors'] is None or fb['descriptors'] is None:
            rejected['no_descriptors'] += 1
            continue
        matches = [m[0] for m in matcher.knnMatch(fa['descriptors'], fb['descriptors'], k=2)
                   if len(m) == 2 and m[0].distance < .70 * m[1].distance]
        if len(matches) < 20:
            rejected['insufficient_matches'] += 1
            continue
        x = np.array([fa['points'][m.queryIdx].pt for m in matches])
        y = np.array([fb['points'][m.trainIdx].pt for m in matches])
        homography, mask = cv2.findHomography(x, y, cv2.RANSAC, 2.5)
        if mask is None:
            rejected['homography_failed'] += 1
            continue
        valid = mask.ravel().astype(bool)
        x, y = x[valid], y[valid]
        if len(x) < 18 or valid.mean() < .4:
            rejected['weak_inliers'] += 1
            continue
        cells = np.floor(x / [width / 4, height / 4]).astype(int)
        occupied = len(set(map(tuple, np.clip(cells, 0, 3))))
        if occupied < 4:
            rejected['insufficient_spatial_coverage'] += 1
            continue
        chosen = np.linspace(0, len(x) - 1, min(100, len(x))).astype(int)
        pair = {'time_a': pts[a], 'time_b': pts[b], 'source_frame_indices': [a, b],
                'source_gray_sha256': [fa['gray_sha256'], fb['gray_sha256']],
                'size_hw': [height, width], 'points_a': x[chosen].tolist(), 'points_b': y[chosen].tolist(),
                'inlier_count': len(x), 'inlier_ratio': float(valid.mean()), 'occupied_grid_cells': occupied,
                'median_observed_displacement_px': float(np.median(np.linalg.norm(x - y, axis=1)))}
        pair['fold'] = block_partition(pair, config)
        pairs.append(pair)
    return {'video': str(video), 'video_sha256': sha_file(Path(video)), 'decoder_command': command,
            'time_source': 'ffprobe best_effort_timestamp_time; sequential FFmpeg images; frame counts agree',
            'frame_count': index, 'frame_pts': pts, 'size_hw': [height, width],
            'requested_pairs': len(desired_pairs), 'rejected_pairs': dict(rejected), 'pairs': pairs,
            'limitations': 'RANSAC dominant background assumption; no depth, occlusion or dynamic-object guarantee'}


class MotionScorer:
    def __init__(self, pairs, timeline):
        self.pairs, self.timeline = pairs, timeline
        self._error_cache = {}
        self.points_a = np.array([p['points_a'] + [p['points_a'][-1]] * (100 - len(p['points_a'])) for p in pairs])
        self.points_b = np.array([p['points_b'] + [p['points_b'][-1]] * (100 - len(p['points_b'])) for p in pairs])
        self.lengths = np.array([len(p['points_a']) for p in pairs])
        self.widths = np.array([p['size_hw'][1] for p in pairs])
        self.heights = np.array([p['size_hw'][0] for p in pairs])
        self.ta = np.array([p['time_a'] for p in pairs])
        self.tb = np.array([p['time_b'] for p in pairs])
        self.folds = np.array([p['fold'] for p in pairs])
        self.motion = np.array([p['median_observed_displacement_px'] for p in pairs])

    def errors(self, offset):
        if offset in self._error_cache:
            return self._error_cache[offset]
        valid_a, qa, fa = self.timeline.sample_many(self.ta - offset)
        valid_b, qb, fb = self.timeline.sample_many(self.tb - offset)
        focal_a = self.heights / (2 * np.tan(np.deg2rad(fa) / 2))
        focal_b = self.heights / (2 * np.tan(np.deg2rad(fb) / 2))
        rays = np.stack(((self.points_a[:, :, 0] - self.widths[:, None] / 2) / focal_a[:, None],
                         -(self.points_a[:, :, 1] - self.heights[:, None] / 2) / focal_a[:, None],
                         np.ones(self.points_a.shape[:2])), axis=-1)
        relative = np.swapaxes(qb, 1, 2) @ qa
        local = np.einsum('nij,nkj->nki', relative, rays)
        forward = local[:, :, 2] > 1e-6
        denominator = np.where(forward, local[:, :, 2], 1.)
        projected = np.stack((self.widths[:, None] / 2 + focal_b[:, None] * local[:, :, 0] / denominator,
                              self.heights[:, None] / 2 - focal_b[:, None] * local[:, :, 1] / denominator), axis=-1)
        error = np.linalg.norm(projected - self.points_b, axis=-1)
        invalid_points = np.arange(100)[None, :] >= self.lengths[:, None]
        error[invalid_points | ~forward] = np.nan
        valid = valid_a & valid_b & ((forward & ~invalid_points).sum(axis=1) >= .8 * self.lengths)
        # Every real match contributes once; padding never affects medians.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            result = np.nanmedian(error, axis=1)
        result[~valid] = np.nan
        result.setflags(write=False)
        self._error_cache[offset] = result
        return result


def loss(errors, mask, config):
    count = int(mask.sum())
    values = errors[mask & np.isfinite(errors)]
    coverage = len(values) / count if count else 0.
    if len(values) < config['minimum_pairs'] or coverage < config['minimum_coverage']:
        return float('inf')
    return float(np.minimum(values, 100.).mean() + 10. * (1 - coverage))


def assess_original_clock(scorer, offset, config):
    eligible = (scorer.folds == 'validation') & (scorer.motion >= config['minimum_motion_px'])
    errors = scorer.errors(offset)
    supported = eligible & np.isfinite(errors)
    count, total = int(supported.sum()), int(eligible.sum())
    coverage = count / total if total else 0.
    median = float(np.median(errors[supported])) if count else None
    reasons = []
    if count < config['minimum_pairs'] or coverage < config['minimum_coverage']:
        reasons.append('insufficient_original_clock_validation_pairs')
    if median is None or median > config['maximum_validation_error_px']:
        reasons.append('high_original_clock_validation_rotation_residual')
    return {'status': 'ORIGINAL_CLOCK_SUPPORTED' if not reasons else 'ORIGINAL_CLOCK_UNCERTAIN',
            'validation_informative_pairs': count, 'eligible_informative_pairs': total,
            'coverage': coverage, 'validation_median_px': median, 'rejection_reasons': reasons,
            'meaning': 'RGB background rotation consistency only; not absolute projection or target visibility'}


def fit_offset(scorer, fold, config):
    mask = (scorer.folds == fold) & (scorer.motion >= config['minimum_motion_px'])
    lo, hi = config['offset_bounds_seconds']
    grid = np.arange(lo, hi + config['offset_grid_seconds'] / 2, config['offset_grid_seconds'])
    profiles = [(float(x), loss(scorer.errors(x), mask, config)) for x in grid]
    finite = [(x, score) for x, score in profiles if np.isfinite(score)]
    if not finite:
        return {'failure': f'insufficient {fold} informative overlapping pairs', 'profile': [], 'fold': fold}
    best, score = min(finite, key=lambda row: row[1])
    fine = [(float(x), loss(scorer.errors(x), mask, config))
            for x in np.arange(max(lo, best - .025), min(hi, best + .025) + .0005, .001)]
    best, score = min(fine, key=lambda row: row[1])
    near = [x for x, value in profiles + fine if value <= score + config['profile_delta_px']]
    return {'offset_seconds': best, 'training_loss_px': score, 'fold': fold,
            'profile_near_optimal_range_seconds': [min(near), max(near)] if near else [best, best],
            'profile_width_seconds': max(near) - min(near) if near else 0.,
            'profile_semantics': 'engineering loss support; not a statistical confidence interval',
            'profile': [{'offset_seconds': x, 'loss_px': s if np.isfinite(s) else None} for x, s in profiles],
            'fine_profile': [{'offset_seconds': x, 'loss_px': s if np.isfinite(s) else None} for x, s in fine],
            'informative_pairs': int(mask.sum())}


def evaluate_fits(scorer, candidate, original, previous, reverse, config):
    chosen = scorer.errors(candidate['offset_seconds'])
    baselines = {'original': scorer.errors(original)}
    if previous is not None:
        baselines['previous_candidate'] = scorer.errors(previous)
    summary = {}
    for label, old in baselines.items():
        for fold in ['train', 'validation']:
            common = (scorer.folds == fold) & np.isfinite(old) & np.isfinite(chosen)
            informative = common & (scorer.motion >= config['minimum_motion_px'])
            summary[f'{label}_{fold}'] = {'common_pairs': int(common.sum()),
                'original_median_px': float(np.median(old[common])) if common.any() else None,
                'candidate_median_px': float(np.median(chosen[common])) if common.any() else None,
                'informative_common_pairs': int(informative.sum()),
                'original_informative_median_px': float(np.median(old[informative])) if informative.any() else None,
                'candidate_informative_median_px': float(np.median(chosen[informative])) if informative.any() else None}
    eligible_validation = (scorer.folds == 'validation') & (scorer.motion >= config['minimum_motion_px'])
    validation = eligible_validation & np.isfinite(chosen)
    vmed = float(np.median(chosen[validation])) if validation.any() else None
    reasons = []
    if validation.sum() < config['minimum_pairs'] or validation.sum() < config['minimum_coverage'] * eligible_validation.sum():
        reasons.append('insufficient_informative_validation_pairs')
    if vmed is None or vmed > config['maximum_validation_error_px']:
        reasons.append('high_validation_rotation_residual')
    if min(abs(candidate['offset_seconds'] - b) for b in config['offset_bounds_seconds']) < config['offset_grid_seconds']:
        reasons.append('offset_at_search_boundary')
    if candidate['profile_width_seconds'] > config['maximum_profile_width_seconds']:
        reasons.append('offset_profile_ambiguous')
    difference = abs(reverse['offset_seconds'] - candidate['offset_seconds']) if reverse.get('offset_seconds') is not None else None
    if difference is None or difference > config['maximum_reverse_fit_difference_seconds']:
        reasons.append('time_blocks_disagree')
    common = summary['original_validation']
    if (common['informative_common_pairs'] < config['minimum_pairs'] or
            common['candidate_informative_median_px'] > common['original_informative_median_px'] * (1 + config['maximum_validation_regression_fraction'])):
        reasons.append('original_common_validation_not_supported')
    original_supported = eligible_validation & np.isfinite(baselines['original'])
    retained = original_supported & np.isfinite(chosen)
    retention = float(retained.sum() / original_supported.sum()) if original_supported.any() else 0.
    if original_supported.any() and retention < config.get('minimum_original_validation_retention', .8):
        reasons.append('candidate_loses_original_validation_coverage')
    original_evidence = assess_original_clock(scorer, original, config)
    return {'status': 'VALIDATED_CANDIDATE' if not reasons else 'CLOCK_UNCERTAIN',
            'rejection_reasons': reasons, 'offset_seconds': candidate['offset_seconds'],
            'original_offset_seconds': original,
            'original_clock_status': original_evidence['status'],
            'original_clock_evidence': original_evidence,
            'reverse_fold_offset_seconds': reverse.get('offset_seconds'), 'fold_offset_difference_seconds': difference,
            'validation_informative_pairs': int(validation.sum()), 'validation_median_px': vmed,
            'original_validation_supported_pairs': int(original_supported.sum()),
            'retained_original_validation_pairs': int(retained.sum()),
            'original_validation_retention_fraction': retention,
            'comparisons': summary, 'absolute_pixel_calibration': 'UNVERIFIED',
            'pair_predictions': [{'pair_index': i, 'candidate_error_px': float(v) if np.isfinite(v) else None,
                                  'fold': scorer.pairs[i]['fold']} for i, v in enumerate(chosen)]}


def audit_event(ctx, root, output, config, ffmpeg, ffprobe, reuse):
    out = output / ctx.scene_key / str(ctx.row_index)
    out.mkdir(parents=True, exist_ok=True)
    source_hashes = {'video_sha256': sha_file(Path(ctx.video_path)), 'telemetry_sha256': sha_file(Path(ctx.json_path))}
    key = hashlib.sha256(json.dumps({'version': VERSION, 'config': config, **source_hashes}, sort_keys=True).encode()).hexdigest()
    cache = out / 'feature_tracks.json'
    if reuse and cache.exists() and json.loads(cache.read_text()).get('cache_key') == key:
        tracks = json.loads(cache.read_text())
    else:
        tracks = track_pts_features(ctx.video_path, ffmpeg, ffprobe, config)
        tracks['cache_key'] = key
        save_json(cache, tracks)
    timeline = CameraPoseTimeline(camera_records(ctx), config['max_pose_gap_seconds'])
    pairs = tracks['pairs']
    if not pairs:
        raise ValueError('No supported image feature pairs')
    scorer = MotionScorer(pairs, timeline)
    fit = fit_offset(scorer, 'train', config)
    reverse = fit_offset(scorer, 'validation', config)
    input_audit_path = root / 'inputs' / ctx.scene_key / str(ctx.row_index) / 'input_audit.json'
    panels = json.loads(input_audit_path.read_text())['panels']
    oldpath = root / 'diagnostics/geometry' / f'{ctx.scene_key}_{ctx.row_index}_motion_alignment.json'
    old = json.loads(oldpath.read_text()) if oldpath.is_file() else {}
    manifest_offsets = [float(p['video_time_offset_seconds']) for p in panels
                        if p.get('video_time_offset_seconds') not in (None, '')]
    manifest_offset = manifest_offsets[0] if manifest_offsets else None
    if manifest_offsets and any(abs(value - manifest_offset) > 1e-6 for value in manifest_offsets):
        raise ValueError('Source panels disagree on their original video/telemetry offset')
    original_offset = old.get('original_offset_seconds', manifest_offset)
    if original_offset is None:
        raise ValueError('No original clock offset in legacy diagnostic or source frame manifest')
    original_offset_source = (panels[0].get('video_time_offset_source') if panels else None)
    previous = old.get('fixed_fov_offset_training', {}).get('offset_seconds')
    clock = ({'status': 'CLOCK_UNCERTAIN', 'rejection_reasons': [fit['failure']], 'offset_seconds': None}
             if fit.get('failure') else evaluate_fits(scorer, fit, original_offset, previous, reverse, config))
    clock.setdefault('original_offset_seconds', original_offset)
    clock['original_offset_source'] = original_offset_source
    clock.setdefault('original_clock_status', 'ORIGINAL_CLOCK_UNCERTAIN')
    if clock['status'] == 'VALIDATED_CANDIDATE':
        clock['pose_selection_status'] = 'VALIDATED_CANDIDATE'
    elif clock.get('original_offset_seconds') is not None:
        clock['pose_selection_status'] = 'ORIGINAL_CLOCK_FALLBACK_PTS_POSE'
    else:
        clock['pose_selection_status'] = 'NO_CLOCK_OFFSET'
    train_ids = {i for p in pairs if p['fold'] == 'train' for i in p['source_frame_indices']}
    validation_ids = {i for p in pairs if p['fold'] == 'validation' for i in p['source_frame_indices']}
    if train_ids & validation_ids:
        raise AssertionError('Train/validation share a source frame')
    contract = {'version': VERSION, 'scene': ctx.scene_key, 'row_index': ctx.row_index,
                'clock': clock, 'source_hashes': source_hashes, 'fit_uses_gt': False, 'fit_uses_gaze': False,
                'fixed_intrinsics': True, 'clock_scale': 1., 'observation_protocol': 'offline_full_interaction',
                'feature_pairs': len(pairs), 'fold_pair_counts': dict(Counter(p['fold'] for p in pairs)),
                'train_validation_shared_frames': 0, 'fit': fit, 'reverse_fit': reverse,
                'original_offset_seconds': original_offset, 'previous_candidate_offset_seconds': previous,
                'limitations': ['rotation-only ignores translation/parallax', 'recorded FOV not render projection matrix',
                               'camera sampling does not record subinterval acceleration', 'container PTS not exposure timestamp']}
    save_json(out / 'clock_fit.json', contract)
    # Recover the precise saved image PTS; no GT/anchor data are opened here.
    prior_path = root / 'diagnostics/head_motion/sensitivity_audit.json'
    prior_event = None
    if prior_path.is_file():
        prior = json.loads(prior_path.read_text())
        prior_event = next((e for e in prior['events'] if e['scene'] == ctx.scene_key
                            and e['row_index'] == ctx.row_index), None)
    frames = []
    for panel in panels:
        if sha_file(Path(panel['source_path'])) != panel['source_sha256']:
            raise ValueError('Frozen source JPEG hash mismatch')
        if panel.get('container_pts_binding_verified'):
            source_index = panel['source_frame_index']
            expected_pts = bound_frame_pts(source_index, panel.get('actual_container_pts_seconds'), tracks['frame_pts'])
            exact = {'status': 'VERIFIED_CONTAINER_PTS_SEQUENCE',
                     'actual_container_pts_seconds': expected_pts,
                     'source_frame_index': source_index,
                     'binding_method': panel.get('container_pts_binding_method'),
                     'source_sha256': panel['source_sha256']}
        else:
            prior_panel = next((p for p in (prior_event or {}).get('panels', [])
                            if p['frame_id'] == panel['frame_id']), None)
            exact = (prior_panel or {}).get('saved_frame_pts_verification', {})
            if (not exact.get('source_jpeg_sha256_matches')
                    or exact.get('replayed_jpeg_sha256') != panel['source_sha256']):
                exact = verify_saved_frame_pts(ctx.video_path, panel, tracks['frame_pts'], ffmpeg)
        reference = source_frame_pose(timeline, exact, clock)
        candidate = source_frame_pose(timeline, exact, clock, allow_candidate=True)
        diagnostic = source_frame_pose(timeline, exact, clock, diagnostic=True)
        frames.append({'frame_id': panel['frame_id'], 'source_path': panel['source_path'], 'source_sha256': panel['source_sha256'],
                       'width': panel['width'], 'height': panel['height'], 'requested_seek_seconds': panel['video_time_seconds'],
                       'frame_time': exact, 'reference_camera': reference,
                       'candidate_camera': candidate, 'attention_camera': candidate,
                       'diagnostic_camera': diagnostic})
    save_camera_timeline(out, tracks['frame_pts'], timeline, clock, scorer)
    save_json(out / 'frame_pose_contract.json', {'version': VERSION, 'clock': clock, 'frames': frames,
              'source_hashes': source_hashes, 'clock_fit_path': str(out / 'clock_fit.json'),
              'original_clock_offset_source': original_offset_source,
              'geometry_status': 'candidate_alignment_absolute_projection_unverified'})
    return contract, frames


def save_camera_timeline(output, frame_pts, timeline, clock, scorer):
    """All observed video frames; diagnostics retain uncertain fits explicitly."""
    rows = []
    for index, pts in enumerate(frame_pts):
        frame = {'status': 'VERIFIED_CONTAINER_PTS_SEQUENCE', 'actual_container_pts_seconds': pts}
        # These are sequentially decoded frames, not all frozen JPEGs. The time
        # binding is checked by matching decoded frame count to the PTS table.
        selected = source_frame_pose(timeline, frame, clock)
        diagnostic = source_frame_pose(timeline, frame, clock, diagnostic=True)
        entry = {'source_frame_index': index, 'container_pts_seconds': pts,
                 'reference_status': selected['status'],
                 'diagnostic_pose_status': diagnostic['status'], 'clock_status': clock['status'],
                 'pose_selection_status': clock.get('pose_selection_status'),
                 'telemetry_query_seconds': selected.get('telemetry_time_seconds'),
                 'reference_clock_offset_seconds': selected.get('clock_offset_seconds'),
                 'reference_clock_source': selected.get('clock_status')}
        if selected['status'] == 'POSE_AVAILABLE':
            entry.update(reference_position_world=selected['position_world'],
                         reference_rotation_xyzw=selected['rotation_xyzw'],
                         reference_fov_vertical_degrees=selected['fov_vertical_degrees'],
                         reference_pose_source_times=selected['pose_source_times'],
                         reference_pose_source_indices=selected['pose_source_indices'],
                         reference_pose_bracket_gap_seconds=selected['pose_bracket_gap_seconds'])
        if diagnostic['status'] == 'POSE_AVAILABLE':
            entry.update(diagnostic_position_world=diagnostic['position_world'],
                         diagnostic_rotation_xyzw=diagnostic['rotation_xyzw'],
                         diagnostic_fov_vertical_degrees=diagnostic['fov_vertical_degrees'],
                         diagnostic_pose_source_times=diagnostic['pose_source_times'],
                         diagnostic_pose_source_indices=diagnostic['pose_source_indices'],
                         diagnostic_pose_bracket_gap_seconds=diagnostic['pose_bracket_gap_seconds'])
        rows.append(entry)
    write_csv(output / 'video_frame_pose_timeline.csv', rows, list(dict.fromkeys(k for r in rows for k in r)))
    save_json(output / 'video_frame_pose_timeline_meta.json', {
        'frames': len(rows), 'clock_status': clock['status'],
        'pose_available_diagnostic': sum(r['diagnostic_pose_status'] == 'POSE_AVAILABLE' for r in rows),
        'binding': 'sequential decoded frame index and verified ffprobe PTS count; no FPS-derived times',
        'no_pose_extrapolation': True, 'absolute_pixel_projection_calibrated': False})


def run(root, output, config, ffmpeg, ffprobe, reuse, events=None):
    output.mkdir(parents=True, exist_ok=True)
    config = {**DEFAULT_CONFIG, **config}
    manifest = json.loads((root / 'fixed_samples.json').read_text())
    if events:
        manifest['samples'] = [s for s in manifest['samples'] if f"{s['scene_key']}:{s['row_index']}" in events]
    save_json(output / 'config.json', {'version': VERSION, 'config': config, 'source_root': str(root),
              'input_changes': 'none', 'gt_fit': False, 'heatmap_foveation_changed': False})
    rows = []
    for ctx in load_contexts(manifest, load_ground_truth=False):
        started = time.perf_counter()
        row = {'scene': ctx.scene_key, 'row_index': ctx.row_index}
        try:
            result, frames = audit_event(ctx, root, output, config, ffmpeg, ffprobe, reuse)
            clock = result['clock']
            row.update(status=clock['status'], pose_selection_status=clock['pose_selection_status'],
                       original_clock_status=clock['original_clock_status'],
                       offset_seconds=clock.get('offset_seconds'),
                       selected_pose_offset_seconds=(clock.get('offset_seconds') if clock['status'] == 'VALIDATED_CANDIDATE'
                           else clock.get('original_offset_seconds') if clock['pose_selection_status'] == 'ORIGINAL_CLOCK_FALLBACK_PTS_POSE' else None),
                       original_offset_seconds=result['original_offset_seconds'],
                       previous_offset_seconds=result['previous_candidate_offset_seconds'],
                       rejection_reasons=clock.get('rejection_reasons'),
                       profile_width_seconds=result['fit'].get('profile_width_seconds'),
                       fold_difference_seconds=clock.get('fold_offset_difference_seconds'),
                       validation_median_px=clock.get('validation_median_px'),
                       common_validation=clock.get('comparisons', {}).get('original_validation'),
                       reference_poses_available=sum(f['reference_camera']['status'] == 'POSE_AVAILABLE' for f in frames),
                        candidate_attention_poses_supported=sum(f['attention_camera'].get('attention_clock_supported', False) for f in frames),
                       total_frames=len(frames))
        except Exception as exc:
            row.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
            save_json(output / ctx.scene_key / str(ctx.row_index) / 'failure.json', row)
        row['elapsed_seconds'] = time.perf_counter() - started
        rows.append(row)
        save_json(output / 'summary.json', {'status': 'RUNNING', 'events': rows})
        print(json.dumps(row, ensure_ascii=False), flush=True)
    summary = {'status': 'COMPLETE', 'events': rows, 'counts': dict(Counter(r['status'] for r in rows)),
               'fit_uses_gt': False, 'heatmap_foveation_changed': False,
               'original_evaluator_or_manifest_changed': False, 'model_inference': 'NOT_RUN_not_needed_for_clock_fit'}
    save_json(output / 'summary.json', summary)
    write_csv(output / 'per_event.csv', rows, list(dict.fromkeys(k for r in rows for k in r)))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, default=Path('runs/pts_camera_clock_v2'))
    parser.add_argument('--ffmpeg', default='ffmpeg')
    parser.add_argument('--ffprobe', default='ffprobe')
    parser.add_argument('--reuse-tracks', action='store_true')
    parser.add_argument('--events', nargs='+')
    args = parser.parse_args(argv)
    run(args.root, args.output, {}, args.ffmpeg, args.ffprobe, args.reuse_tracks, args.events)


if __name__ == '__main__':
    main()
