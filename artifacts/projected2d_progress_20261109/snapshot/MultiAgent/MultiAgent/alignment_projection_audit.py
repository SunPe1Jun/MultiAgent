"""CPU-only frame/clock audit with post-fit projections of this event's referents."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .baseline import module
from .camera_geometry import ATTENTION_CLOCK_POLICY, VERSION, aligned_frame_geometry, project_world
from .context import load_contexts
from .gaze_evidence import build_heatmap, gaze_records, render_presentation
from .local_video_visual import bind_pose_contract, extract_dense_frames, file_sha, probe_duration, save


def load_event_referents(ctx):
    """Read labels after fitting, solely to choose the referents drawn in this audit."""
    evaluator = module('eval')
    gt_path = Path(ctx.provenance['gt_file'])
    anchor_path = Path(ctx.provenance['anchor_table'])
    aliases, _ = evaluator.load_anchor_alias_map(anchor_path)
    rows = evaluator.load_gt_rows(gt_path, aliases)
    if not 0 <= ctx.row_index < len(rows):
        raise ValueError(f'GT row unavailable for {ctx.scene_key}/{ctx.row_index}')
    gt = rows[ctx.row_index]
    instruction = ' '.join(str(gt.get('instruction') or '').split())
    if instruction and instruction != ' '.join(ctx.text.split()):
        raise ValueError(f'GT instruction does not match event {ctx.event_id}')
    targets = []
    for name in dict.fromkeys(gt['mapped_referents']):
        anchor = ctx.candidate(name)
        targets.append({'referent_name': name, 'anchor_id': name,
                        'point_world': ([anchor.x_world, anchor.y_world, anchor.z_world]
                                        if anchor is not None else None),
                        'mapping_status': 'mapped' if anchor is not None else 'referent_anchor_unavailable'})
    for name in dict.fromkeys(gt.get('unmapped_referents', [])):
        targets.append({'referent_name': name, 'anchor_id': None, 'point_world': None,
                        'mapping_status': 'unmapped_referent'})
    return targets, {'gt_file': str(gt_path), 'gt_sha256': file_sha(gt_path),
                     'anchor_table': str(anchor_path), 'anchor_table_sha256': file_sha(anchor_path),
                     'row_index': ctx.row_index, 'excel_row_index': gt.get('excel_row_index'),
                     'role': 'post_fit_referent_projection_only'}


def project_referents(targets, camera, original_camera, width, height):
    """Keep every event referent, including missing, off-screen and unavailable poses."""
    def project(target, pose):
        if target['point_world'] is None:
            status = target['mapping_status']
        elif pose.get('status') != 'POSE_AVAILABLE':
            status = 'pose_unavailable'
        else:
            return project_world(target['point_world'], pose, width, height)
        return {'status': status, 'in_frustum': False,
                'normalized_uv_unclipped': None, 'pixel_xy_unclipped': None}

    projections = []
    for target in targets:
        selected = project(target, camera)
        original = project(target, original_camera)
        shift = None
        if selected.get('pixel_xy_unclipped') is not None and original.get('pixel_xy_unclipped') is not None:
            shift = float(np.linalg.norm(np.subtract(selected['pixel_xy_unclipped'],
                                                     original['pixel_xy_unclipped'])))
        projections.append({**target, **selected, 'original_projection': original,
                            'selected_minus_original_clock_displacement_px': shift})
    return projections


def render_audit_frame(rgb, heat, projections, *, scene, row_index, frame_id, geometry):
    rendered, _ = render_presentation(rgb, heat, 'H_local')
    canvas = Image.new('RGB', (rgb.width, rgb.height + 110 + 18 * len(projections)), '#eeeeee')
    canvas.paste(rendered, (0, 0))
    draw = ImageDraw.Draw(canvas)
    for target in projections:
        # Draw each clock independently: a target may enter the frame only after selection.
        original = target['original_projection']
        if original['in_frustum']:
            x, y = original['pixel_xy_unclipped']
            draw.ellipse((x-5, y-5, x+5, y+5), outline='cyan', width=2)
        if target['in_frustum']:
            x, y = target['pixel_xy_unclipped']
            draw.line((x-4, y, x+4, y), fill='#ff9800', width=2)
            draw.line((x, y-4, x, y+4), fill='#ff9800', width=2)
            label_x = max(2, min(rgb.width - 8 * len(target['referent_name']), x + 7))
            draw.text((label_x, max(0, min(rgb.height - 12, y))), target['referent_name'], fill='#ff9800')
    footer = [f"{scene}/{row_index} {frame_id} PTS={geometry['video_time_seconds']:.6f}",
              f"Selected clock: {geometry['selected_pose_clock_source']}  offset={geometry['selected_clock_offset_seconds']}",
              f"Attention clock supported: {geometry['clock_available']}",
              'Cyan circle: original clock. Orange cross: selected clock.',
              'Red: supported gaze. Event referents only; visibility unknown.',
              'GT used after fitting for audit only; no model/evaluator run.']
    for index, text in enumerate(footer):
        draw.text((5, rgb.height + 4 + index * 16), text, fill='black')
    for index, target in enumerate(projections):
        uv = target.get('normalized_uv_unclipped')
        text = f"{target['referent_name']}: {target['original_projection']['status']} -> {target['status']}"
        if uv is not None and not target['in_frustum']:
            text += f" uv=({uv[0]:.2f},{uv[1]:.2f})"
        draw.text((5, rgb.height + 106 + index * 18), text, fill='black')
    return canvas


def audit_context(ctx, output, frames=4):
    event_dir = output / 'events' / ctx.scene_key / str(ctx.row_index)
    event_dir.mkdir(parents=True, exist_ok=True)
    duration = probe_duration(Path(ctx.video_path))
    selected = extract_dense_frames(Path(ctx.video_path), event_dir / 'frames', 0., duration,
                                    min_frames=frames, max_frames=frames)
    alignment = bind_pose_contract(ctx, selected, event_dir, output / 'pose_alignment')
    # Label loading deliberately follows the GT-free clock fit and frame selection.
    targets, referent_source = load_event_referents(ctx)
    records = gaze_records(ctx)
    panels = []
    for frame in selected['frames']:
        original_camera = frame['camera_pose']
        contract_frame = {'source_sha256': frame['sha256'], 'frame_time': frame['frame_pts_binding'],
                          'reference_camera': original_camera,
                          'attention_camera': frame['attention_camera_pose']}
        geometry = aligned_frame_geometry(contract_frame, file_sha(Path(frame['path'])),
                                          frame['time_seconds'], alignment['original_offset_seconds'])
        camera = geometry['reference_pose'] or {}
        projections = project_referents(targets, camera, original_camera, frame['width'], frame['height'])
        with Image.open(frame['path']) as image:
            rgb = image.convert('RGB')
        heat, stats, gaze = build_heatmap(ctx, records,
            frame_time_video=geometry['video_time_seconds'],
            frame_time_telemetry=geometry['telemetry_time_seconds'],
            event_window=[float(ctx.api_row['t_start']), float(ctx.api_row['t_end'])],
            clock_available=geometry['clock_available'], image_size=rgb.size, strategy='local_history',
            reference_pose=geometry['reference_pose'])
        canvas = render_audit_frame(rgb, heat, projections, scene=ctx.scene_key, row_index=ctx.row_index,
                                   frame_id=frame['frame_id'], geometry=geometry)
        path = event_dir / f"{frame['frame_id']}_projection.jpg"
        canvas.save(path, quality=92)
        panels.append({'frame_id': frame['frame_id'], 'source_frame_index': frame['source_frame_index'],
            'source_path': frame['path'], 'source_sha256': frame['sha256'],
            'frame_pts_seconds': geometry['video_time_seconds'], 'pose': frame['attention_camera_pose'],
            'original_pose': original_camera, 'frame_geometry': geometry,
            'referent_projections': projections,
            'attention_clock_supported': geometry['clock_available'],
            'gaze_stats': stats, 'gaze_projections': gaze, 'projection_image': str(path)})
    result = {'scene': ctx.scene_key, 'row_index': ctx.row_index, 'event_id': ctx.event_id,
              'instruction': ctx.text, 'clock': alignment, 'clock_policy': ATTENTION_CLOCK_POLICY,
              'frames': panels, 'referent_count': len(targets),
              'referents': [target['referent_name'] for target in targets],
              'projection_scope': 'event_referents_only', 'referent_source': referent_source,
              'context_ground_truth_loaded': ctx.provenance.get('ground_truth_loaded', False),
              'ground_truth_loaded': True, 'ground_truth_role': 'post_fit_referent_projection_only',
              'fit_uses_gt': False, 'fit_uses_gaze': False, 'model_calls': 0, 'exposure_calibrated': False}
    save(event_dir / 'audit.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--limit', type=int, default=2)
    parser.add_argument('--frames', type=int, default=4)
    args = parser.parse_args(argv)
    args.manifest = args.manifest.resolve()
    args.output = args.output.resolve()
    if args.limit <= 0 or args.frames < 2 or args.frames % 2:
        parser.error('limit must be positive; frames must be positive and even (at least two)')
    if args.output.exists():
        parser.error('Choose a new output directory to preserve previous diagnostics')
    manifest = json.loads(args.manifest.read_text())
    manifest = {**manifest, 'samples': manifest['samples'][:args.limit]}
    if not manifest['samples']:
        parser.error('Manifest has no events to audit')
    contexts = load_contexts(manifest, load_ground_truth=False)
    args.output.mkdir(parents=True)
    events = []
    for ctx in contexts:
        print(f'AUDIT {ctx.scene_key}/{ctx.event_id}', flush=True)
        try:
            result = audit_context(ctx, args.output, args.frames)
            events.append({'scene': ctx.scene_key, 'row_index': ctx.row_index, 'status': 'COMPLETE',
                           'clock_status': result['clock']['candidate_clock_status'],
                           'frames': len(result['frames']), 'referents': result['referents'],
                           'attention_frames': sum(f['attention_clock_supported'] for f in result['frames']),
                           'audit_path': str(args.output/'events'/ctx.scene_key/str(ctx.row_index)/'audit.json')})
        except Exception as exc:
            events.append({'scene': ctx.scene_key, 'row_index': ctx.row_index, 'status': 'FAILED',
                           'error': f'{type(exc).__name__}: {exc}'})
        save(args.output/'summary.json', {'events': events, 'model_calls': 0,
                                        'fit_uses_gt': False, 'fit_uses_gaze': False})
    summary = {'events': events, 'counts': dict(Counter(e['status'] for e in events)),
               'model_calls': 0, 'fit_uses_gt': False, 'fit_uses_gaze': False,
               'ground_truth_role': 'post_fit_referent_projection_only',
               'projection_scope': 'event_referents_only', 'clock_policy': ATTENTION_CLOCK_POLICY,
               'original_evaluator_changed': False, 'source_manifest': str(args.manifest),
               'geometry_version': VERSION,
               'meaning': 'Frame binding and geometric consistency only; not absolute target localization accuracy'}
    save(args.output/'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 1 if any(e['status'] == 'FAILED' for e in events) else 0


if __name__ == '__main__':
    raise SystemExit(main())
