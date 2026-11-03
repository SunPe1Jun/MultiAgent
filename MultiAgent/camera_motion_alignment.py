"""CPU RGB/telemetry rotation diagnostic. Fits no GT or gaze-to-target matches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from PIL import Image, ImageDraw, ImageFont

from .context import load_contexts
from .projected2d_attention_study import read_csv, save_json, write_csv
from .projection_diagnostics import project_world_anchor


def track_static_features(video, max_size=640):
    import cv2
    cv2.setNumThreads(4)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sift = cv2.SIFT_create(nfeatures=1000)
    frames = []
    for index in range(max(0, int(.6 * fps)), count - 1, max(1, int(.5 * fps))):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = cap.read()
        if not ok:
            continue
        h, w = image.shape[:2]
        scale = max_size / max(w, h)
        image = cv2.resize(image, (round(w * scale), round(h * scale)))
        keypoints, descriptors = sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)
        frames.append({'time': index / fps, 'frame_index': index,
                       'reported_msec': cap.get(cv2.CAP_PROP_POS_MSEC),
                       'shape': image.shape[:2], 'points': keypoints, 'descriptors': descriptors})
    cap.release()
    matcher = cv2.BFMatcher()
    pairs = []
    for a, b in zip(frames, frames[1:]):
        if a['descriptors'] is None or b['descriptors'] is None:
            continue
        matches = [m for m, n in matcher.knnMatch(a['descriptors'], b['descriptors'], k=2)
                   if m.distance < .7 * n.distance]
        if len(matches) < 18:
            continue
        x = np.array([a['points'][m.queryIdx].pt for m in matches])
        y = np.array([b['points'][m.trainIdx].pt for m in matches])
        homography, mask = cv2.findHomography(x, y, cv2.RANSAC, 2.5)
        if mask is None:
            continue
        valid = mask.ravel().astype(bool)
        if valid.sum() < 15:
            continue
        # Cap uniform deterministic indices; robust homography excludes moving actors.
        x, y = x[valid], y[valid]
        sample = np.linspace(0, len(x) - 1, min(100, len(x))).astype(int)
        pairs.append({'time_a': a['time'], 'time_b': b['time'], 'source_frame_indices': [a['frame_index'], b['frame_index']],
                      'size_hw': a['shape'], 'points_a': x[sample].tolist(), 'points_b': y[sample].tolist(),
                      'inliers': int(valid.sum()), 'tentative_matches': len(matches),
                      'median_observed_displacement_px': float(np.median(np.linalg.norm(x - y, axis=1)))})
    return pairs, {'fps': fps, 'frame_count': count, 'sampled_frames': len(frames),
                    'decoder': 'OpenCV frame-index seek', 'time_semantics': 'frame_index/fps and reported POS_MSEC; no camera exposure timestamp'}


def evaluate_candidate(pairs, times, rotations, offset, fov_scale, parity='standard', split=None):
    interpolator = Slerp(times, Rotation.from_quat(rotations))
    errors = []
    for i, pair in enumerate(pairs):
        if split is not None and i % 2 != split:
            continue
        a, b = pair['time_a'] - offset, pair['time_b'] - offset
        if a < times[0] or b > times[-1]:
            continue
        h, w = pair['size_hw']
        # fov_scale = multiplier on tan(recorded vertical FOV/2), not a GT fit.
        tan = pair['recorded_tan_fov'] * fov_scale
        focal = h / (2 * tan)
        q = interpolator([a, b]).as_matrix()
        relative = q[1].T @ q[0]
        x, y = np.asarray(pair['points_a']), np.asarray(pair['points_b'])
        rays = np.column_stack(((x[:, 0] - w / 2) / focal, -(x[:, 1] - h / 2) / focal, np.ones(len(x))))
        if parity == 'mirror_x':
            rays[:, 0] *= -1
        rays = rays @ relative.T
        valid = rays[:, 2] > 1e-5
        if valid.sum() < len(rays) * .8:
            continue
        pred = np.column_stack((w / 2 + focal * rays[:, 0] / rays[:, 2] * (-1 if parity == 'mirror_x' else 1),
                                h / 2 - focal * rays[:, 1] / rays[:, 2]))
        error = np.linalg.norm(pred[valid] - y[valid], axis=1)
        errors.append({'pair_index': i, 'median_error_px': float(np.median(error))})
    # Each pair has equal weight, preventing texture-rich frames from dominating.
    total_pairs = sum(split is None or i % 2 == split for i in range(len(pairs)))
    # Penalize missing overlap and large residuals; a median alone can favor a few quiet pairs.
    score = (float(np.mean(np.minimum([e['median_error_px'] for e in errors], 150)))
             + 20 * (total_pairs - len(errors))) if errors else None
    return {'offset_seconds': float(offset), 'fov_tan_scale': float(fov_scale), 'axis_hypothesis': parity,
            'pair_count': len(errors), 'median_pair_error_px': float(np.median([e['median_error_px'] for e in errors])) if errors else None,
            'training_selection_score': score, 'total_split_pairs': total_pairs,
            'pair_errors': errors}


def audit_event(ctx, manifest, out):
    event = [r for r in manifest if r['scene'] == ctx.scene_key and int(r['row_index']) == ctx.row_index]
    original_offset = float(event[0]['video_time_offset_seconds'])
    pairs, video = track_static_features(ctx.video_path)
    times = np.array([t for t, s in ctx.timed_samples])
    rotations = np.array([[s['cameraRotation'][k] for k in ('x', 'y', 'z', 'w')] for t, s in ctx.timed_samples])
    unique = np.r_[True, np.diff(times) > 1e-8]
    times, rotations = times[unique], rotations[unique]
    for pair in pairs:
        midpoint = (pair['time_a'] + pair['time_b']) / 2 - original_offset
        _, sample = min(ctx.timed_samples, key=lambda x: abs(x[0] - midpoint))
        pair['recorded_tan_fov'] = float(np.tan(np.deg2rad(float(sample['cameraFOV'])) / 2))
    minimum_train = max(3, int(np.ceil(sum(i % 2 == 0 for i in range(len(pairs))) * .65)))
    trials = []
    for offset in np.arange(-3, 6.001, .1):
        for scale in (.55, .7, .85, 1., 1.15, 1.3, 1.5):
            r = evaluate_candidate(pairs, times, rotations, offset, scale, split=0)
            if r['pair_count'] >= minimum_train and r['median_pair_error_px'] is not None:
                trials.append(r)
    result = {'scene': ctx.scene_key, 'row_index': ctx.row_index, 'video_path': ctx.video_path,
              'ground_truth_used': False, 'gaze_used': False, 'video': video,
              'feature_pairs': pairs, 'original_offset_seconds': original_offset,
              'model': 'camera rotation only; translation/parallax/static-feature assumptions may fail',
              'fit_split': 'alternating feature pairs; odd-index pairs held out',
              'calibration_status': 'DIAGNOSTIC_ONLY_NOT_APPLIED'}
    if not trials:
        result['failure'] = 'insufficient overlapping feature/pose pairs'
    else:
        best = min(trials, key=lambda r: r['training_selection_score'])
        fine = []
        for offset in np.arange(best['offset_seconds'] - .12, best['offset_seconds'] + .121, .02):
            for scale in np.arange(max(.2, best['fov_tan_scale'] - .1), best['fov_tan_scale'] + .101, .025):
                r = evaluate_candidate(pairs, times, rotations, offset, scale, split=0)
                if r['pair_count'] >= minimum_train:
                    fine.append(r)
        best = min(fine, key=lambda r: r['training_selection_score'])
        fixed_fov = min([r for r in trials if r['fov_tan_scale'] == 1.], key=lambda r: r['training_selection_score'])
        fixed_fine = [evaluate_candidate(pairs, times, rotations, offset, 1., split=0)
                      for offset in np.arange(fixed_fov['offset_seconds'] - .08, fixed_fov['offset_seconds'] + .081, .01)]
        fixed_fov = min([r for r in fixed_fine if r['pair_count'] >= minimum_train], key=lambda r: r['training_selection_score'])
        # This one-factor diagnosis holds intrinsics fixed and varies only the clock.
        result['fixed_fov_offset_training'] = fixed_fov
        result['fixed_fov_offset_heldout'] = evaluate_candidate(pairs, times, rotations, fixed_fov['offset_seconds'], 1., split=1)
        common = {e['pair_index'] for e in result['fixed_fov_offset_heldout']['pair_errors']} & {
            e['pair_index'] for e in evaluate_candidate(pairs, times, rotations, original_offset, 1., split=1)['pair_errors']}
        result['common_heldout_pair_indices'] = sorted(common)
        for label, candidate in [('fixed_fov', result['fixed_fov_offset_heldout']),
                                 ('original', evaluate_candidate(pairs, times, rotations, original_offset, 1., split=1))]:
            result[label + '_common_heldout_error_px'] = float(np.median([e['median_error_px'] for e in candidate['pair_errors'] if e['pair_index'] in common])) if common else None
        result.update(best_training=best,
                      best_heldout=evaluate_candidate(pairs, times, rotations, best['offset_seconds'], best['fov_tan_scale'], split=1),
                      original_training=evaluate_candidate(pairs, times, rotations, original_offset, 1., split=0),
                      original_heldout=evaluate_candidate(pairs, times, rotations, original_offset, 1., split=1),
                      unchanged_fov_best=fixed_fov,
                      mirror_x_at_best=evaluate_candidate(pairs, times, rotations, best['offset_seconds'], best['fov_tan_scale'], parity='mirror_x', split=1))
        positions = np.array([[s['cameraPosition'][k] for k in ('x', 'y', 'z')] for t, s in ctx.timed_samples])
        result['camera_position_extent_world_units'] = np.ptp(positions, axis=0).tolist()
    save_json(out / f'{ctx.scene_key}_{ctx.row_index}_motion_alignment.json', result)
    print(json.dumps({k: result.get(k) for k in ('scene', 'row_index', 'original_offset_seconds', 'best_training', 'best_heldout', 'original_heldout', 'failure')}, ensure_ascii=False), flush=True)
    return result


def render_clock_hypothesis(ctx, manifest, diagnostic, out):
    """Same JPEGs; compare only offset, using a fit made without anchors/gaze."""
    candidate = diagnostic.get('fixed_fov_offset_training')
    if not candidate:
        return
    event = [r for r in manifest if r['scene'] == ctx.scene_key and int(r['row_index']) == ctx.row_index]
    panels = sorted({r['panel_id'] for r in event})
    width, height, footer = 520, 520, 130
    sheet = Image.new('RGB', (width * len(panels), (height + footer) * 2), '#f4f4f4')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    times = np.array([t for t, _ in ctx.timed_samples])
    rotations = Rotation.from_quat([[s['cameraRotation'][k] for k in 'xyzw'] for _, s in ctx.timed_samples])
    unique = np.r_[True, np.diff(times) > 1e-8]
    interpolation = Slerp(times[unique], rotations[unique])
    poses = []
    for row_index, (label, offset) in enumerate([('ORIGINAL clock', diagnostic['original_offset_seconds']),
                                                 ('RGB-motion offset HYPOTHESIS / NOT applied', candidate['offset_seconds'])]):
        for col, fid in enumerate(panels):
            targets = [r for r in event if r['panel_id'] == fid]
            meta = targets[0]
            # Use identical interpolation in both rows; changing offset is the only comparison factor.
            video_time = float(meta['video_frame_time'])
            time = video_time - offset
            available = times[0] <= time <= times[-1]
            x0, y0 = col * width, row_index * (height + footer)
            with Image.open(meta['frame_path']) as image:
                sheet.paste(image.convert('RGB').resize((width, height)), (x0, y0))
            draw.text((x0 + 5, y0 + height + 3), label, fill='black', font=font)
            draw.text((x0 + 5, y0 + height + 23), f'{fid} video={video_time:.3f}s telemetry={time:.3f}s offset={offset:.3f}s', fill='black', font=font)
            if not available:
                draw.text((x0 + 5, y0 + height + 47), 'NO TELEMETRY: no GT or gaze projection fabricated', fill='#b03010', font=font)
                continue
            position = [np.interp(time, times, [s['cameraPosition'][k] for _, s in ctx.timed_samples]) for k in 'xyz']
            fov = np.interp(time, times, [s['cameraFOV'] for _, s in ctx.timed_samples])
            pose = {'cameraPosition': dict(zip('xyz', position)),
                    'cameraRotation': dict(zip('xyzw', interpolation([time]).as_quat()[0])), 'cameraFOV': float(fov)}
            records = []
            for index, target in enumerate(targets):
                projection = project_world_anchor([float(target['anchor_' + k]) for k in 'xyz'], pose, int(meta['image_width']), int(meta['image_height']))
                name = target['referent_name']
                records.append({'referent_name': name, **projection})
                draw.text((x0 + 5, y0 + height + 47 + index * 22), f"{name}: {projection['status']}", fill='#8f008f' if projection['in_frustum'] else '#b03010', font=font)
                if projection['in_frustum']:
                    u, v = projection['normalized_uv_unclipped']
                    x, y = x0 + u * width, y0 + v * height
                    draw.line((x - 7, y, x + 7, y), fill='magenta', width=2)
                    draw.line((x, y - 7, x, y + 7), fill='magenta', width=2)
                    draw.text((min(x0 + width - 115, max(x0, x + 6)), min(y0 + height - 20, max(y0, y))), name, fill='magenta', font=font)
            poses.append({'label': label, 'frame_id': fid, 'video_time': video_time, 'telemetry_time': time,
                          'camera_pose': pose, 'target_projections': records,
                          'calibration_status': 'DIAGNOSTIC_ONLY; GT not used to choose offset'})
    destination = out / f'{ctx.scene_key}_{ctx.row_index}_clock_hypothesis.jpg'
    sheet.save(destination, quality=94)
    save_json(destination.with_suffix('.json'), {'image': str(destination), 'frames': poses,
              'model_inputs_changed': False, 'evaluator_manifest_changed': False,
              'interpretation': 'Targets used for visualization only after GT-free RGB-motion fit; residual errors remain.'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--events', nargs='+', default=['scene4_room2:145', 'scene2:139', 'scene4_room1:170'])
    parser.add_argument('--all', action='store_true', help='Audit every fixed sample, including failures')
    parser.add_argument('--reuse-existing', action='store_true', help='Reuse saved RGB-motion diagnostics')
    args = parser.parse_args()
    manifest = json.loads((args.root / 'fixed_samples.json').read_text())
    if not args.all:
        wanted = set(args.events)
        manifest['samples'] = [x for x in manifest['samples'] if f"{x['scene_key']}:{x['row_index']}" in wanted]
    out = args.root / 'diagnostics' / 'geometry'
    out.mkdir(parents=True, exist_ok=True)
    records = read_csv(args.root / 'manifest_all.csv')
    summary = {'status': 'RUNNING', 'ground_truth_used_for_fit': False,
               'clock_convention': 'video_seconds = telemetry_seconds + offset_seconds',
               'applied_to_model_or_evaluator': False, 'events': []}
    for ctx in load_contexts(manifest, load_ground_truth=False):
        existing = out / f'{ctx.scene_key}_{ctx.row_index}_motion_alignment.json'
        try:
            if args.reuse_existing and existing.exists():
                result = json.loads(existing.read_text())
            else:
                result = audit_event(ctx, records, out)
            render_clock_hypothesis(ctx, records, result, out)
            candidate = result.get('fixed_fov_offset_training', {})
            entry = {'scene': ctx.scene_key, 'row_index': ctx.row_index,
                     'status': 'FAILED' if result.get('failure') else 'DIAGNOSTIC_ONLY_NOT_APPLIED',
                     'failure': result.get('failure'), 'detail_path': str(existing),
                     'original_offset_seconds': result['original_offset_seconds'],
                     'candidate_offset_seconds': candidate.get('offset_seconds'),
                     'common_heldout_pairs': len(result.get('common_heldout_pair_indices', [])),
                     'original_error_px': result.get('original_common_heldout_error_px'),
                     'candidate_error_px': result.get('fixed_fov_common_heldout_error_px')}
        except Exception as exc:
            entry = {'scene': ctx.scene_key, 'row_index': ctx.row_index, 'status': 'FAILED',
                     'failure': f'{type(exc).__name__}: {exc}'}
        summary['events'].append(entry)
        save_json(out / 'motion_alignment_summary.json', summary)
        print(json.dumps(entry, ensure_ascii=False), flush=True)
    comparable = [e for e in summary['events']
                  if e.get('original_error_px') is not None and e.get('candidate_error_px') is not None]
    summary['counts'] = {'fixed_samples': len(summary['events']),
        'diagnostic_failures': sum(e['status'] == 'FAILED' for e in summary['events']),
        'common_heldout_comparable': len(comparable),
        'lower_candidate_residual': sum(e['candidate_error_px'] < e['original_error_px'] for e in comparable),
        'higher_candidate_residual': sum(e['candidate_error_px'] > e['original_error_px'] for e in comparable)}
    summary['aggregate'] = {
        'original_median_of_event_medians_px': float(np.median([e['original_error_px'] for e in comparable])) if comparable else None,
        'candidate_median_of_event_medians_px': float(np.median([e['candidate_error_px'] for e in comparable])) if comparable else None,
        'units': 'pixels at max_size=640; RGB feature motion, not target localization',
        'interpretation': 'Diagnostic only. A lower feature residual does not prove absolute RGB/pose calibration or GT projection accuracy.'}
    write_csv(out / 'motion_alignment_per_event.csv', summary['events'],
              ['scene', 'row_index', 'status', 'original_offset_seconds', 'candidate_offset_seconds',
               'common_heldout_pairs', 'original_error_px', 'candidate_error_px', 'failure', 'detail_path'])
    summary['status'] = 'COMPLETE'
    save_json(out / 'motion_alignment_summary.json', summary)


if __name__ == '__main__':
    main()
