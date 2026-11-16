"""Report world-anchor frustum states without changing the released evaluator."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from .camera_geometry import project_world

from .context import load_contexts
from .projected2d_attention_study import read_csv, save_json, sha_file, write_csv


def project_world_anchor(point, sample, width, height):
    """Use the same unclipped camera transform as gaze; frustum inclusion is not visibility."""
    pose = {'position_world': [(sample.get('cameraPosition') or {}).get(k) for k in 'xyz'],
            'rotation_xyzw': [(sample.get('cameraRotation') or {}).get(k) for k in 'xyzw'],
            'fov_vertical_degrees': sample.get('cameraFOV')}
    return {'point_world': list(point), **project_world(point, pose, width, height)}


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
