"""Report world-anchor frustum states without changing the released evaluator."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from .context import load_contexts
from .projected2d_attention_study import read_csv, save_json, sha_file, write_csv


def project_world_anchor(point, sample, width, height):
    """Retain unbounded UV; in-frustum is not a depth/occlusion test."""
    result = {"point_world": list(point), "occlusion": "unknown_no_depth_or_mesh",
              "pixel_calibration": "unverified_recorded_camera_vs_video",
              "projection_model": "vertical_fov_square_pixels_ideal_pinhole",
              "in_frustum": False}
    try:
        position = np.array([sample['cameraPosition'][k] for k in ('x', 'y', 'z')], float)
        q = np.array([sample['cameraRotation'][k] for k in ('x', 'y', 'z', 'w')], float)
        fov = float(sample['cameraFOV'])
        if (width <= 0 or height <= 0 or not 0 < fov < 180
                or not np.all(np.isfinite(position)) or not np.all(np.isfinite(q))
                or not np.all(np.isfinite(point)) or np.linalg.norm(q) < 1e-8):
            raise ValueError('invalid camera or world point')
        local = Rotation.from_quat(q).inv().apply(np.asarray(point, float) - position)
        result.update(camera_xyz=local.tolist(), camera_position_world=position.tolist(),
                      camera_rotation_xyzw=(q / np.linalg.norm(q)).tolist(), vertical_fov_degrees=fov)
        if local[2] <= 1e-6:
            result.update(status='behind_camera', normalized_uv_unclipped=None, pixel_xy_unclipped=None)
            return result
        focal = height / (2 * math.tan(math.radians(fov) / 2))
        u = .5 + focal * local[0] / (local[2] * width)
        v = .5 - focal * local[1] / (local[2] * height)
        inside = 0 <= u < 1 and 0 <= v < 1
        result.update(status='in_frustum_visibility_unknown' if inside else 'outside_frame',
                      in_frustum=bool(inside), normalized_uv_unclipped=[float(u), float(v)],
                      pixel_xy_unclipped=[float(u * width), float(v * height)],
                      intrinsics={'fx': focal, 'fy': focal, 'cx': width / 2, 'cy': height / 2},
                      outside_edges=([k for k, test in [('left', u < 0), ('right', u >= 1),
                                                      ('top', v < 0), ('bottom', v >= 1)] if test] if not inside else []))
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        result.update(status='invalid_geometry', error=str(exc))
    return result


def audit(root: Path):
    manifest = read_csv(root / 'manifest_all.csv')
    contexts = load_contexts(json.loads((root / 'fixed_samples.json').read_text()), load_ground_truth=False)
    out = root / 'diagnostics' / 'geometry'
    out.mkdir(parents=True, exist_ok=True)
    rows, json_events = [], []
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    for ctx in contexts:
        event_rows = [r for r in manifest if r['scene'] == ctx.scene_key and int(r['row_index']) == ctx.row_index]
        frame_results = []
        for fid in sorted({r['panel_id'] for r in event_rows}):
            targets = [r for r in event_rows if r['panel_id'] == fid]
            meta = targets[0]
            time = float(meta['json_sample_time'])
            selected_time, sample = min(ctx.timed_samples, key=lambda x: abs(x[0] - time))
            frame = {'frame_id': fid, 'requested_video_seconds': float(meta['video_frame_time']),
                     'selected_telemetry_seconds': selected_time, 'timestamp': sample['timestamp'],
                     'clock_offset_seconds': float(meta['video_time_offset_seconds']),
                     'clock_source': meta['video_time_offset_source'], 'source_path': meta['frame_path'],
                     'source_sha256': sha_file(Path(meta['frame_path'])), 'targets': []}
            for target in targets:
                point = [float(target['anchor_' + k]) for k in ('x', 'y', 'z')]
                projection = project_world_anchor(point, sample, int(meta['image_width']), int(meta['image_height']))
                original_valid = target['projection_valid'] == 'True'
                record = {'scene': ctx.scene_key, 'row_index': ctx.row_index, 'frame_id': fid,
                          'referent_name': target['referent_name'], **projection,
                          'original_projection_valid': original_valid,
                          'frustum_agrees_with_original': projection['in_frustum'] == original_valid,
                          'selected_telemetry_seconds': selected_time,
                          'requested_video_seconds': float(meta['video_frame_time']),
                          'clock_offset_seconds': float(meta['video_time_offset_seconds']),
                          'actual_rgb_time_calibrated': False}
                rows.append(record)
                frame['targets'].append(record)
            frame_results.append(frame)
        # Outside and behind-camera targets appear as text below the frame, never as clipped points.
        width, photo_height = 520, 520
        footer = 80 + max(len(f['targets']) for f in frame_results) * 24
        sheet = Image.new('RGB', (width * len(frame_results), photo_height + footer), '#f4f4f4')
        draw = ImageDraw.Draw(sheet)
        for index, frame in enumerate(frame_results):
            x0 = index * width
            with Image.open(frame['source_path']) as image:
                sheet.paste(image.convert('RGB').resize((width, photo_height)), (x0, 0))
            draw.text((x0 + 6, photo_height + 5), f"{frame['frame_id']} video={frame['requested_video_seconds']:.3f}s telemetry={frame['selected_telemetry_seconds']:.3f}s", fill='black', font=font)
            draw.text((x0 + 6, photo_height + 27), 'Magenta = projected anchor; occlusion / RGB sync UNKNOWN', fill='black', font=font)
            for k, target in enumerate(frame['targets']):
                name, status = target['referent_name'], target['status']
                uv = target.get('normalized_uv_unclipped')
                description = f"{name}: {status}"
                if uv:
                    description += f" uv=({uv[0]:.2f},{uv[1]:.2f})"
                draw.text((x0 + 6, photo_height + 57 + k * 24), description, fill='#8f008f' if target['in_frustum'] else '#b03010', font=font)
                if target['in_frustum']:
                    x, y = x0 + uv[0] * width, uv[1] * photo_height
                    draw.line((x - 7, y, x + 7, y), fill='magenta', width=2)
                    draw.line((x, y - 7, x, y + 7), fill='magenta', width=2)
                    draw.text((min(x0 + width - 110, max(x0 + 3, x + 8)), min(photo_height - 20, max(0, y))), name, fill='magenta', font=font)
        path = out / f'{ctx.scene_key}_{ctx.row_index}_frustum_audit.jpg'
        sheet.save(path, quality=94)
        json_events.append({'scene': ctx.scene_key, 'row_index': ctx.row_index, 'instruction': ctx.text,
                            'comparison_image': str(path), 'frames': frame_results})
    counts = dict(Counter(r['status'] for r in rows))
    result = {'scope': 'posthoc world-anchor diagnostics; GT never used for model inputs or camera calibration',
              'status_counts': counts, 'all_frustum_checks_agree_with_original': all(r['frustum_agrees_with_original'] for r in rows),
              'gt_is_gazePoint': False, 'coordinate_formula': 'p_camera = inverse(R_camera_to_world) * (GT_world - C_world)',
              'gaze_role': 'independent world/camera consistency and attention cues; not a substitute for GT_world',
              'limitations': ['frustum inclusion is not verified visibility', 'no occlusion or full-object extent test',
                             'ideal intrinsics and RGB/telemetry offset not calibrated'], 'events': json_events}
    save_json(out / 'frustum_audit.json', result)
    write_csv(out / 'frustum_detail.csv', rows, list(dict.fromkeys(k for r in rows for k in r)))
    print(json.dumps({'targets': len(rows), 'status_counts': counts,
                      'matches_original': result['all_frustum_checks_agree_with_original']}, ensure_ascii=False))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    audit(args.root)


if __name__ == '__main__':
    main()
