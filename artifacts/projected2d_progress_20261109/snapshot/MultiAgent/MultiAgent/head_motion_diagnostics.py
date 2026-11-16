"""GT-free sensitivity to pose sampling and RGB/telemetry timing during head motion.

Offsets are diagnostic hypotheses, never calibration truth. This module writes only
new diagnostics, and never changes model images, evaluator labels or selected frames.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

import numpy as np
from scipy.spatial.transform import Rotation

from .context import load_contexts
from .gaze_evidence import interpolate_camera_pose
from .video_timing import frame_pts_table
from .camera_geometry import camera_values
from .projected2d_attention_study import save_json, write_csv


def camera_records(ctx):
    records = []
    for index, (seconds, sample) in enumerate(ctx.timed_samples):
        try:
            position, q, fov = camera_values([sample['cameraPosition'][k] for k in 'xyz'],
                                              [sample['cameraRotation'][k] for k in 'xyzw'], sample['cameraFOV'])
            seconds = float(seconds)
            if not np.isfinite(seconds) or (records and seconds <= records[-1]['telemetry_time_seconds']):
                continue
            records.append({'source_index': index, 'telemetry_time_seconds': seconds,
                            'camera_position': position.tolist(), 'camera_rotation_xyzw': q.tolist(),
                            'camera_fov_degrees': fov})
        except (KeyError, TypeError, ValueError):
            continue
    return records


def rotation_sensitivity(reference_q, alternative_q, width, height, vertical_fov):
    """Nine fixed image rays; no GT, anchor, gaze or depth. Pixel units = source size."""
    focal = height / (2 * math.tan(math.radians(vertical_fov) / 2))
    xy = np.array([(u * width, v * height) for u in (.25, .5, .75) for v in (.25, .5, .75)])
    rays = np.column_stack(((xy[:, 0] - width / 2) / focal,
                            -(xy[:, 1] - height / 2) / focal, np.ones(len(xy))))
    world = Rotation.from_quat(reference_q).apply(rays)
    local = Rotation.from_quat(alternative_q).inv().apply(world)
    forward = local[:, 2] > 1e-6
    projected = np.column_stack((width / 2 + focal * local[:, 0] / local[:, 2],
                                 height / 2 - focal * local[:, 1] / local[:, 2]))
    shifts = np.linalg.norm(projected[forward] - xy[forward], axis=1)
    return {'median_ray_displacement_px': float(np.median(shifts)) if len(shifts) else None,
            'max_ray_displacement_px': float(np.max(shifts)) if len(shifts) else None,
            'forward_rays': int(forward.sum()), 'total_rays': len(xy),
            'meaning': 'rotation-only timing sensitivity; not measured GT localization error'}


def inspect_time(records, target_time, width, height):
    result = {'telemetry_time_seconds': float(target_time)}
    if len(records) < 2:
        return {**result, 'status': 'insufficient_camera_samples'}
    pose = interpolate_camera_pose(records, target_time)
    if pose is None:
        return {**result, 'status': 'pose_unavailable_no_extrapolation'}
    times = np.array([r['telemetry_time_seconds'] for r in records])
    right = int(np.clip(np.searchsorted(times, target_time), 1, len(times) - 1))
    a, b = records[right - 1], records[right]
    gap = b['telemetry_time_seconds'] - a['telemetry_time_seconds']
    qa, qb = Rotation.from_quat(a['camera_rotation_xyzw']), Rotation.from_quat(b['camera_rotation_xyzw'])
    omega = float(np.rad2deg((qa.inv() * qb).magnitude()) / gap)
    speed = float(np.linalg.norm(np.subtract(b['camera_position'], a['camera_position'])) / gap)
    nearest = min(records, key=lambda r: abs(r['telemetry_time_seconds'] - target_time))
    result.update(status='COMPLETE', interpolated_pose=pose,
                  bracket_angular_speed_deg_s=omega, bracket_translation_speed_world_units_s=speed,
                  speed_bracket_times=[a['telemetry_time_seconds'], b['telemetry_time_seconds']],
                  nearest_pose_time_delta_seconds=nearest['telemetry_time_seconds'] - target_time,
                  nearest_vs_interpolated=rotation_sensitivity(pose['rotation_xyzw'], nearest['camera_rotation_xyzw'],
                                                              width, height, pose['fov_vertical_degrees']))
    result['time_perturbations'] = []
    for delta in (-.05, -.02, -.01, .01, .02, .05):
        shifted = interpolate_camera_pose(records, target_time + delta)
        row = {'delta_seconds': delta, 'status': 'pose_unavailable_no_extrapolation'}
        if shifted is not None:
            row.update(status='COMPLETE', **rotation_sensitivity(pose['rotation_xyzw'], shifted['rotation_xyzw'],
                                                                 width, height, pose['fov_vertical_degrees']))
        result['time_perturbations'].append(row)
    return result


def probe_pts(video, executable):
    command = [executable, '-v', 'error', '-select_streams', 'v:0', '-show_frames',
               '-show_entries', 'frame=best_effort_timestamp_time,pkt_duration_time', '-of', 'json', str(video)]
    payload = json.loads(subprocess.check_output(command, timeout=60))
    pts = frame_pts_table(payload.get('frames', []))
    return pts, {'command': command, 'frame_pts': pts,
                 'scope': 'container presentation times; not camera exposure times; saved JPEG mapping unverified'}


def verify_saved_frame_pts(video, panel, frame_pts, executable):
    """Reproduce the source JPEG byte-for-byte and retain its decoded input PTS."""
    command = [executable, '-hide_banner', '-loglevel', 'info', '-copyts',
               '-ss', f"{panel['video_time_seconds']:.3f}", '-i', str(video),
               '-frames:v', '1', '-q:v', '2', '-vf', 'showinfo',
               '-vcodec', 'mjpeg', '-f', 'image2pipe', 'pipe:1']
    completed = subprocess.run(command, capture_output=True, timeout=60)
    stderr = completed.stderr.decode(errors='replace')
    sha = hashlib.sha256(completed.stdout).hexdigest()
    matching = sha == panel['source_sha256']
    lines = [line for line in stderr.splitlines() if 'showinfo' in line and ' n:' in line]
    first = next((re.search(r'\bn:\s*0\s+pts:\s*[-\d]+\s+pts_time:([-\d.eE+]+)', line)
                  for line in lines if re.search(r'\bn:\s*0\s+pts:', line)), None)
    result = {'command': command, 'returncode': completed.returncode,
              'source_jpeg_sha256_matches': matching, 'replayed_jpeg_sha256': sha,
              'showinfo_lines': lines, 'status': 'UNVERIFIED'}
    if matching and completed.returncode == 0 and first and frame_pts:
        value = float(first.group(1))
        index = min(range(len(frame_pts)), key=lambda i: abs(frame_pts[i] - value))
        if abs(frame_pts[index] - value) < .00002:
            result.update(status='VERIFIED_EXACT_SOURCE_JPEG', source_frame_index=index,
                          actual_container_pts_seconds=frame_pts[index],
                          pts_minus_requested_seconds=frame_pts[index] - panel['video_time_seconds'])
    return result


def aggregate_sensitivity(summary, rows):
    panels = [p for event in summary['events'] for p in event.get('panels', [])]
    exact = [p['saved_frame_pts_verification'] for p in panels
             if p.get('saved_frame_pts_verification', {}).get('status') == 'VERIFIED_EXACT_SOURCE_JPEG']
    delays = [p['pts_minus_requested_seconds'] for p in exact]
    summary['frame_pts_verification'] = {'source_frames': len(panels), 'verified_source_jpegs': len(exact),
        'median_pts_minus_requested_seconds': float(np.median(delays)) if delays else None,
        'max_pts_minus_requested_seconds': max(delays, default=None),
        'scope': 'exact frozen JPEG replay matches; container PTS is not camera exposure timestamp'}
    summary['median_event_camera_interval_seconds'] = float(np.median([
        e['median_camera_sample_interval_seconds'] for e in summary['events'] if e.get('status') == 'COMPLETE']))
    for label, stats in summary['summaries'].items():
        selected = [r for r in rows if r['clock_hypothesis'] == label and r['status'] == 'COMPLETE']
        for name in ['nearest_vs_interpolated_px', 'timing_10ms_max_direction_median_px',
                     'timing_20ms_max_direction_median_px', 'timing_50ms_max_direction_median_px',
                     'decoded_pts_correction_sensitivity_px']:
            values = [float(r[name]) for r in selected if r.get(name) not in ('', None)]
            stats[name] = {'median': float(np.median(values)) if values else None,
                           'max': max(values, default=None), 'available_panels': len(values)}
    return summary


def run(root, ffprobe, *, verify_frame_pts=False, ffmpeg='ffmpeg'):

    output = root / 'diagnostics' / 'head_motion'
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / 'fixed_samples.json').read_text())
    contexts = load_contexts(manifest, load_ground_truth=False)
    events, flat = [], []
    for ctx in contexts:
        event = {'scene': ctx.scene_key, 'row_index': ctx.row_index, 'ground_truth_used': False}
        try:
            records = camera_records(ctx)
            audit = json.loads((root / 'inputs' / ctx.scene_key / str(ctx.row_index) / 'input_audit.json').read_text())
            alignment = json.loads((root / 'diagnostics/geometry' / f'{ctx.scene_key}_{ctx.row_index}_motion_alignment.json').read_text())
            candidate = alignment.get('fixed_fov_offset_training', {}).get('offset_seconds')
            offsets = [('original', alignment['original_offset_seconds'])]
            if candidate is not None:
                offsets.append(('diagnostic_candidate_not_applied', candidate))
            pts, pts_meta = probe_pts(ctx.video_path, ffprobe)
            save_json(output / f'{ctx.scene_key}_{ctx.row_index}_video_pts.json', pts_meta)
            intervals = np.diff([r['telemetry_time_seconds'] for r in records])
            event.update(status='COMPLETE', camera_samples=len(records),
                         median_camera_sample_interval_seconds=float(np.median(intervals)),
                         max_camera_sample_interval_seconds=float(np.max(intervals)),
                         video_pts_count=len(pts), panels=[])
            for panel in audit['panels']:
                t = panel['video_time_seconds']
                nearest_pts = min(pts, key=lambda p: abs(p - t)) if pts else None
                record = {'frame_id': panel['frame_id'], 'requested_video_time_seconds': t,
                          'source_image_sha256': panel['source_sha256'],
                          'nearest_container_pts_seconds': nearest_pts,
                          'nearest_pts_minus_requested_seconds': nearest_pts - t if nearest_pts is not None else None,
                          'pts_is_exact_saved_jpeg_match': False, 'clock_hypotheses': []}
                exact = verify_saved_frame_pts(ctx.video_path, panel, pts, ffmpeg) if verify_frame_pts else {'status': 'NOT_RUN'}
                record['saved_frame_pts_verification'] = exact
                record['pts_is_exact_saved_jpeg_match'] = exact['status'] == 'VERIFIED_EXACT_SOURCE_JPEG'
                for label, offset in offsets:
                    estimate = inspect_time(records, t - offset, panel['width'], panel['height'])
                    estimate.update(clock_hypothesis=label, offset_seconds=offset)
                    if exact.get('actual_container_pts_seconds') is not None:
                        actual = inspect_time(records, exact['actual_container_pts_seconds'] - offset, panel['width'], panel['height'])
                        estimate['at_verified_decoded_pts'] = actual
                        if estimate['status'] == actual['status'] == 'COMPLETE':
                            estimate['decoded_pts_correction_sensitivity'] = rotation_sensitivity(
                                estimate['interpolated_pose']['rotation_xyzw'], actual['interpolated_pose']['rotation_xyzw'],
                                panel['width'], panel['height'], estimate['interpolated_pose']['fov_vertical_degrees'])
                    record['clock_hypotheses'].append(estimate)
                    flat.append({'scene': ctx.scene_key, 'row_index': ctx.row_index, 'frame_id': panel['frame_id'],
                                 'clock_hypothesis': label, 'status': estimate['status'],
                                 'exact_source_pts_status': exact['status'],
                                 'actual_pts_minus_requested_s': exact.get('pts_minus_requested_seconds'),
                                 'decoded_pts_correction_sensitivity_px': estimate.get('decoded_pts_correction_sensitivity', {}).get('median_ray_displacement_px'),
                                 'angular_speed_deg_s': estimate.get('bracket_angular_speed_deg_s'),
                                 'pose_gap_s': estimate.get('interpolated_pose', {}).get('pose_bracket_gap_seconds'),
                                 'nearest_pose_time_delta_s': estimate.get('nearest_pose_time_delta_seconds'),
                                 'nearest_vs_interpolated_px': estimate.get('nearest_vs_interpolated', {}).get('median_ray_displacement_px'),
                                 **{f'timing_{int(ms*1000)}ms_max_direction_median_px': max((
                                     e['median_ray_displacement_px'] for e in estimate.get('time_perturbations', [])
                                     if e.get('status') == 'COMPLETE' and abs(abs(e['delta_seconds']) - ms) < 1e-9), default=None)
                                     for ms in (.01, .02, .05)}})
                event['panels'].append(record)
        except Exception as exc:
            event.update(status='FAILED', failure=f'{type(exc).__name__}: {exc}')
        events.append(event)
        print(f"{ctx.scene_key}/{ctx.row_index}: {event['status']}", flush=True)
    summary = {'status': 'COMPLETE', 'events': events, 'source_frames_changed': False,
               'model_or_evaluator_changed': False, 'ground_truth_used': False,
               'caution': 'Offsets remain uncalibrated. Pure rotation at nine test rays omits finite-depth translation/parallax and is not absolute target error.',
               'summaries': {}}
    for label in ('original', 'diagnostic_candidate_not_applied'):
        subset = [r for r in flat if r['clock_hypothesis'] == label]
        complete = [r for r in subset if r['status'] == 'COMPLETE']
        summary['summaries'][label] = {'panels': len(subset), 'pose_available': len(complete),
            'median_angular_speed_deg_s': float(np.median([r['angular_speed_deg_s'] for r in complete])) if complete else None,
            'over_45deg_s_panels': sum(r['angular_speed_deg_s'] > 45 for r in complete),
            'max_angular_speed_deg_s': max((r['angular_speed_deg_s'] for r in complete), default=None)}
    aggregate_sensitivity(summary, flat)
    write_csv(output / 'per_panel.csv', flat, list(dict.fromkeys(k for r in flat for k in r)))
    save_json(output / 'sensitivity_audit.json', summary)
    try:
        render_plot(flat, output)
    except ModuleNotFoundError as exc:
        if exc.name != 'matplotlib':
            raise
        save_json(output / 'motion_sensitivity_plot_status.json', {
            'status': 'NOT_RENDERED_MATPLOTLIB_UNAVAILABLE',
            'install_not_required_for_numeric_audit': True})
    print(json.dumps(summary['summaries'], ensure_ascii=False), flush=True)
    return summary


def render_plot(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for label, color in [('original', '#3973ac'), ('diagnostic_candidate_not_applied', '#d47523')]:
        subset = [r for r in rows if r['clock_hypothesis'] == label and r['status'] == 'COMPLETE']
        for ax, field, title in zip(axes,
                ['nearest_vs_interpolated_px', 'timing_20ms_max_direction_median_px'],
                ['Nearest sampled pose vs interpolated pose', 'Sensitivity to +/-20 ms timing perturbation']):
            valid = [r for r in subset if r.get(field) is not None]
            ax.scatter([r['angular_speed_deg_s'] for r in valid], [r[field] for r in valid],
                       s=22, alpha=.7, label=label, c=color)
            ax.set(xlabel='Bracket mean angular speed (deg/s)', ylabel='Median 9-ray displacement (source px)', title=title)
            ax.grid(alpha=.25)
    axes[1].legend(fontsize=7)
    fig.suptitle('GT-free rotation sensitivity; candidate clocks NOT calibrated/applied', fontsize=11)
    fig.tight_layout()
    fig.savefig(output / 'motion_sensitivity.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--ffprobe', default='ffprobe')
    parser.add_argument('--ffmpeg', default='ffmpeg')
    parser.add_argument('--verify-saved-frame-pts', action='store_true')
    args = parser.parse_args()
    run(args.root, args.ffprobe, verify_frame_pts=args.verify_saved_frame_pts, ffmpeg=args.ffmpeg)


if __name__ == '__main__':
    main()
