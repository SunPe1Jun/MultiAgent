"""Post-fit world-anchor projection audit using the PTS pose contract.

This is a diagnostic adapter. It never changes the released manifest/evaluator and
never participates in clock fitting. Uncertain clocks are retained as explicit rows.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .projection_diagnostics import project_world_anchor
from .projected2d_attention_study import read_csv, save_json, sha_file, write_csv


def contract_sample(frame):
    pose = frame.get('reference_camera') or {}
    if pose.get('status') != 'POSE_AVAILABLE':
        return None
    return {'cameraPosition': dict(zip('xyz', pose['position_world'])),
            'cameraRotation': dict(zip('xyzw', pose['rotation_xyzw'])),
            'cameraFOV': pose['fov_vertical_degrees']}


def project_event(root, alignment_root, scene, row_index, manifest_rows, render=True):
    contract_path = alignment_root / scene / str(row_index) / 'frame_pose_contract.json'
    contract = json.loads(contract_path.read_text())
    frames = {row['frame_id']: row for row in contract['frames']}
    rows = [row for row in manifest_rows if row['scene'] == scene and int(row['row_index']) == row_index]
    by_panel = {}
    output_rows = []
    for row in rows:
        panel = row['panel_id']
        by_panel.setdefault(panel, []).append(row)
        frame = frames.get(panel)
        pose = contract_sample(frame) if frame else None
        base = {'scene': scene, 'row_index': row_index, 'frame_id': panel,
                'referent_name': row.get('referent_name', ''),
                'clock_status': (frame or {}).get('reference_camera', {}).get('clock_status', 'NO_RELIABLE_CLOCK'),
                'pose_clock_source': (frame or {}).get('reference_camera', {}).get('clock_status'),
                'candidate_clock_status': contract['clock']['status'],
                'source_frame_sha256': row.get('source_sha256') or sha_file(Path(row['frame_path'])),
                'requested_video_seconds': float(row['video_frame_time']),
                'actual_container_pts_seconds': (frame or {}).get('frame_time', {}).get('actual_container_pts_seconds'),
                'telemetry_query_seconds': (frame or {}).get('reference_camera', {}).get('telemetry_time_seconds'),
                'old_gt_x': float(row['gt_x']) if row.get('gt_x') else None,
                'old_gt_y': float(row['gt_y']) if row.get('gt_y') else None,
                'old_projection_valid': row.get('projection_valid') == 'True'}
        if pose is None:
            camera = (frame or {}).get('reference_camera', {})
            output_rows.append({**base, 'status': camera.get('status', 'FRAME_CONTRACT_MISSING'),
                                'pose_status': camera.get('status', 'FRAME_CONTRACT_MISSING'),
                                'projection_status': 'NOT_PROJECTED',
                                'new_projection_valid': False, 'new_uv_unclipped': None,
                                'new_pixel_xy_unclipped': None, 'old_new_pixel_displacement': None})
            continue
        point = [float(row['anchor_' + key]) for key in 'xyz']
        projected = project_world_anchor(point, pose, int(row['image_width']), int(row['image_height']))
        displacement = None
        if base['old_gt_x'] is not None and projected.get('pixel_xy_unclipped') is not None:
            dx = projected['pixel_xy_unclipped'][0] - base['old_gt_x']
            dy = projected['pixel_xy_unclipped'][1] - base['old_gt_y']
            displacement = (dx * dx + dy * dy) ** .5
        output_rows.append({**base, 'status': projected['status'],
                            'pose_status': frame['reference_camera']['status'],
                            'projection_status': projected['status'],
                            'new_projection_valid': projected['in_frustum'],
                            'new_uv_unclipped': projected.get('normalized_uv_unclipped'),
                            'new_pixel_xy_unclipped': projected.get('pixel_xy_unclipped'),
                            'new_camera_xyz': projected.get('camera_xyz'),
                            'new_outside_edges': projected.get('outside_edges', []),
                            'old_new_pixel_displacement': displacement,
                            'occlusion': projected.get('occlusion')})
    return output_rows, by_panel


def render_event(output, scene, row_index, by_panel, rows):
    panels = sorted(by_panel, key=lambda value: int(value[1:]))
    width = height = 520
    footer = 100 + 20 * max((len(by_panel[p]) for p in panels), default=1)
    sheet = Image.new('RGB', (width * len(panels), height + footer), '#f4f4f4')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    for col, panel in enumerate(panels):
        source = Path(by_panel[panel][0]['frame_path'])
        with Image.open(source) as image:
            sheet.paste(image.convert('RGB').resize((width, height)), (col * width, 0))
            draw.text((col * width + 5, height + 4),
                      f'{panel} old=magenta; fitted PTS pose=cyan; original-clock PTS pose=blue', fill='black', font=font)
        for index, row in enumerate([r for r in rows if r['frame_id'] == panel]):
            text = f"{row['referent_name']}: {row['status']}"
            if row['status'] == 'outside_frame' and row.get('new_uv_unclipped') is not None:
                text += f" uv=({row['new_uv_unclipped'][0]:.2f},{row['new_uv_unclipped'][1]:.2f})"
            draw.text((col * width + 5, height + 26 + index * 20), text,
                      fill='#007f7f' if row['new_projection_valid'] else '#b03010', font=font)
            if row.get('old_gt_x') is not None:
                x = col * width + row['old_gt_x'] / float(by_panel[panel][0]['image_width']) * width
                y = row['old_gt_y'] / float(by_panel[panel][0]['image_height']) * height
                draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline='magenta', width=2)
                draw.line((x - 6, y, x + 6, y), fill='magenta', width=2)
                draw.line((x, y - 6, x, y + 6), fill='magenta', width=2)
            uv = row.get('new_uv_unclipped')
            if uv is not None and row['new_projection_valid']:
                x, y = col * width + uv[0] * width, uv[1] * height
                color = '#1688ff' if row['pose_clock_source'] == 'ORIGINAL_CLOCK_FALLBACK_PTS_POSE' else 'cyan'
                draw.line((x - 6, y, x + 6, y), fill=color, width=2)
                draw.line((x, y - 6, x, y + 6), fill=color, width=2)
    path = output / f'{scene}_{row_index}_pts_projection_audit.jpg'
    sheet.save(path, quality=94)
    return str(path)


def run(root: Path, alignment_root: Path, render=True):
    manifest = read_csv(root / 'manifest_all.csv')
    samples = json.loads((root / 'fixed_samples.json').read_text())['samples']
    output = alignment_root / 'projection_audit'
    output.mkdir(parents=True, exist_ok=True)
    all_rows, events = [], []
    for sample in samples:
        scene, row_index = sample['scene_key'], int(sample['row_index'])
        try:
            rows, by_panel = project_event(root, alignment_root, scene, row_index, manifest, render)
            image = render_event(output, scene, row_index, by_panel, rows) if render else None
            events.append({'scene': scene, 'row_index': row_index, 'clock_status': rows[0]['clock_status'],
                           'rows': len(rows), 'image': image,
                           'new_projection_generated': sum(r['projection_status'] != 'NOT_PROJECTED' for r in rows),
                           'new_in_frustum': sum(r['new_projection_valid'] for r in rows),
                           'not_projected_rows': sum(r['projection_status'] == 'NOT_PROJECTED' for r in rows),
                           'pose_sources': dict(Counter(r['pose_clock_source'] or r['status'] for r in rows)),
                           'median_old_new_pixel_displacement': (sorted(r['old_new_pixel_displacement'] for r in rows
                               if r['old_new_pixel_displacement'] is not None)[len([r for r in rows if r['old_new_pixel_displacement'] is not None]) // 2]
                               if any(r['old_new_pixel_displacement'] is not None for r in rows) else None)})
            all_rows.extend(rows)
        except Exception as exc:
            events.append({'scene': scene, 'row_index': row_index, 'clock_status': 'FAILED',
                           'error': f'{type(exc).__name__}: {exc}'})
    result = {'status': 'COMPLETE', 'events': events, 'rows': len(all_rows),
              'status_counts': dict(Counter(row['status'] for row in all_rows)),
              'new_projection_generated': sum(row['projection_status'] != 'NOT_PROJECTED' for row in all_rows),
              'new_projection_not_generated': sum(row['projection_status'] == 'NOT_PROJECTED' for row in all_rows),
              'fit_uses_gt': False, 'gt_used_only_for_postfit_projection': True,
              'model_inputs_changed': False, 'evaluator_changed': False,
              'interpretation': 'Cyan = validated candidate clock with exact frame PTS/interpolated pose; blue = original clock fallback with exact frame PTS/interpolated pose; magenta = old projection. A rejected candidate no longer suppresses the new projection when the original offset and exact frame time are available.'}
    save_json(output / 'summary.json', result)
    write_csv(output / 'per_target.csv', all_rows, list(dict.fromkeys(k for row in all_rows for k in row)))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--alignment-root', type=Path, default=Path('runs/pts_pose_alignment_20261023'))
    parser.add_argument('--no-render', action='store_true')
    args = parser.parse_args(argv)
    print(json.dumps(run(args.root, args.alignment_root, not args.no_render), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
