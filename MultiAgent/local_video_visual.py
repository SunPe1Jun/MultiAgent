"""Dense local video Visual-agent inference with scene-isolated candidates.

This module intentionally stops before Hand fusion and the global Coordinator.
It is used to validate the Visual agent's actual temporal input and mention
candidate boundary before any downstream policy is allowed to change the set.
"""
from __future__ import annotations

import argparse
import bisect
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

from .agents import ReferentialAgent, sample_key
from .baseline import DATA_ROOT, ROOT as TRIMODAL_ROOT, WORKSPACE_ROOT, module
from .context import load_contexts
from .mention_mapping import category
from .qwen_backend import QwenBackend


DEFAULT_FPS = 4.0
DEFAULT_MIN_FRAMES = 16
DEFAULT_MAX_FRAMES = 100
VERSION = 'local_dense_video_visual_v7_pts_pose_fallback_literal_referential_repair'
DEFAULT_MAX_SIDE = 672
DEFAULT_VIDEO_MAX_PIXELS = 8_388_608


def save(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')


def original_video_offset(ctx):
    """Recreate the released Projected-2D offset rule; never read GT points."""
    spec = importlib.util.spec_from_file_location(
        'multiagent_original_2d_manifest_builder', TRIMODAL_ROOT / 'exam2/build_2d_eval_manifest.py')
    if spec is None or spec.loader is None:
        raise RuntimeError('Cannot load the released Projected-2D clock helper')
    builder = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = builder
    spec.loader.exec_module(builder)
    raw_samples = [sample for _, sample in ctx.timed_samples]
    offset, source = builder.effective_video_time_offset(
        Path(ctx.json_path), Path(ctx.video_path), raw_samples,
        manual_adjustment=0.0, auto_enabled=True, source_strategy='hybrid',
        threshold_seconds=1.0, video_bias_seconds=0.5)
    return float(offset), source


def bind_pose_contract(ctx, frame_manifest, out_dir, pose_root):
    """Fit RGB/telemetry offset, fallback to original offset, bind selected PTS to poses."""
    from .frame_pose_alignment import DEFAULT_CONFIG, audit_event

    source_root = out_dir / 'pose_source'
    event_input = source_root / 'inputs' / ctx.scene_key / str(ctx.row_index)
    event_input.mkdir(parents=True, exist_ok=True)
    original_offset, original_offset_source = original_video_offset(ctx)
    panels = []
    method = ('ffprobe best_effort_timestamp frame_index; FFmpeg select=eq(n,index) from same '
              'decoded presentation sequence; no FPS-derived timestamp')
    for frame in frame_manifest['frames']:
        image_path = Path(frame['path']).resolve()
        panels.append({'frame_id': frame['frame_id'], 'source_path': str(image_path),
            'source_sha256': frame['sha256'], 'width': int(frame['width']), 'height': int(frame['height']),
            'video_time_seconds': float(frame['time_seconds']),
            'actual_container_pts_seconds': float(frame['time_seconds']),
            'source_frame_index': int(frame['source_frame_index']),
            'container_pts_binding_verified': True, 'container_pts_binding_method': method,
            'video_time_offset_seconds': original_offset,
            'video_time_offset_source': original_offset_source})
    save(event_input / 'input_audit.json', {'scene': ctx.scene_key, 'row_index': ctx.row_index,
        'event_id': ctx.event_id, 'panels': panels, 'ground_truth_loaded': False,
        'binding_scope': method, 'original_clock_source': original_offset_source,
        'original_offset_seconds': original_offset})
    config = dict(DEFAULT_CONFIG)
    alignment, source_frames = audit_event(ctx, source_root, pose_root, config,
                                          'ffmpeg', 'ffprobe', reuse=True)
    by_id = {row['frame_id']: row for row in source_frames}
    for frame in frame_manifest['frames']:
        pose = by_id[frame['frame_id']]
        frame.update(actual_container_pts_seconds=pose['frame_time'].get('actual_container_pts_seconds'),
                     frame_pts_binding=pose['frame_time'],
                     camera_pose=pose['reference_camera'],
                     diagnostic_candidate_pose=pose['diagnostic_camera'],
                     pose_status=pose['reference_camera'].get('status'),
                     pose_clock_status=pose['reference_camera'].get('clock_status'),
                     clock_offset_seconds=pose['reference_camera'].get('clock_offset_seconds'),
                     telemetry_pose_time_seconds=pose['reference_camera'].get('telemetry_time_seconds'),
                     original_clock_offset_source=pose['reference_camera'].get('original_clock_offset_source'))
    # Refresh Qwen temporal groups after pose provenance has been attached.
    frame_manifest['qwen_temporal_groups'] = qwen_temporal_groups(frame_manifest['frames'])
    summary = {'candidate_clock_status': alignment['clock']['status'],
        'candidate_offset_seconds': alignment['clock'].get('offset_seconds'),
        'original_offset_seconds': alignment['clock'].get('original_offset_seconds'),
        'original_offset_source': original_offset_source,
        'pose_selection_status': alignment['clock'].get('pose_selection_status'),
        'rejection_reasons': alignment['clock'].get('rejection_reasons', []),
        'frame_count': len(source_frames),
        'verified_pts_frames': sum(f['frame_time'].get('status') == 'VERIFIED_CONTAINER_PTS_SEQUENCE'
                                   for f in source_frames),
        'pose_available_frames': sum(f['reference_camera'].get('status') == 'POSE_AVAILABLE'
                                     for f in source_frames),
        'pose_clock_sources': dict(__import__('collections').Counter(
            f['reference_camera'].get('clock_status', f['reference_camera'].get('status'))
            for f in source_frames)),
        'fit_uses_gt': False, 'fit_uses_gaze': False,
        'absolute_pixel_projection_calibrated': False,
        'container_pts_is_exposure_time': False}
    return summary


def file_sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probe_duration(video_path: Path) -> float:
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)],
        capture_output=True, text=True, check=True)
    duration = float(result.stdout.strip())
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f'Invalid video duration for {video_path}: {duration}')
    return duration


def probe_frames(video_path):
    result = subprocess.run([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames',
        '-show_streams', '-show_entries',
        'frame=best_effort_timestamp_time:stream=width,height,avg_frame_rate,time_base',
        '-of', 'json', str(video_path)], capture_output=True, text=True, check=True)
    payload = json.loads(result.stdout)
    frames = [{'source_frame_index': i, 'pts_seconds': float(row['best_effort_timestamp_time'])}
              for i, row in enumerate(payload['frames']) if 'best_effort_timestamp_time' in row]
    if not frames or any(b['pts_seconds'] <= a['pts_seconds'] for a, b in zip(frames, frames[1:])):
        raise ValueError('Missing or non-increasing source video PTS')
    return frames, payload['streams'][0]


def qwen_temporal_groups(frame_metadata, temporal_patch_size=2):
    if temporal_patch_size <= 0 or len(frame_metadata) % temporal_patch_size:
        raise ValueError('Qwen temporal patching requires a complete even-sized frame grouping')
    groups = []
    for start in range(0, len(frame_metadata), temporal_patch_size):
        members = frame_metadata[start:start + temporal_patch_size]
        times = [float(row['time_seconds']) for row in members]
        telemetry_times = [row.get('telemetry_pose_time_seconds') for row in members
                           if row.get('telemetry_pose_time_seconds') is not None]
        groups.append({
            'group_index': len(groups),
            'view_id': f'G{len(groups)+1:03d}',
            'frame_id': f'G{len(groups)+1:03d}',
            'source_frame_ids': [row.get('source_frame_id') for row in members],
            'source_frame_indices': [row.get('source_frame_index') for row in members],
            'frame_ids': [row.get('frame_id') for row in members],
            'source_time_range': [times[0], times[-1]],
            'actual_container_pts_seconds': [row.get('actual_container_pts_seconds') for row in members],
            'pose_statuses': [row.get('pose_status') for row in members],
            'pose_clock_sources': [row.get('pose_clock_status') for row in members],
            'telemetry_pose_time_range': ([min(telemetry_times), max(telemetry_times)]
                                          if telemetry_times else None),
            'time_seconds': sum(times) / len(times),
            'time_label_seconds': round(sum(times) / len(times), 1),
        })
    return groups


def prompt_temporal_groups(groups):
    """Give Visual source-frame time provenance without numeric pose/gaze data."""
    fields = ('view_id', 'source_frame_ids', 'source_frame_indices', 'frame_ids',
              'source_time_range', 'time_seconds', 'time_label_seconds')
    return [{key: row.get(key) for key in fields if key in row} for row in groups]


def extract_dense_frames(video_path: Path, output_dir: Path, t_start: float, t_end: float,
                         *, fps: float = DEFAULT_FPS,
                         min_frames: int = DEFAULT_MIN_FRAMES,
                         max_frames: int = DEFAULT_MAX_FRAMES, max_side=DEFAULT_MAX_SIDE):
    """Select actual source frame indices; never fabricate timestamps or duplicate frames."""
    if fps <= 0 or min_frames <= 0 or max_frames < min_frames:
        raise ValueError('Invalid dense-frame configuration')
    duration = probe_duration(video_path)
    source_frames, stream = probe_frames(video_path)
    start = min(max(0.0, float(t_start)), max(0.0, duration - 0.05))
    end = min(max(start + 0.05, float(t_end)), duration)
    clip_duration = max(end - start, 0.05)
    # `fps` is the desired model temporal-group rate. Qwen merges two source
    # frames per temporal patch, so select about 2*fps actual frames per second.
    min_even = min_frames + min_frames % 2
    max_even = max_frames - max_frames % 2
    if max_even < min_even:
        raise ValueError('Frame bounds must allow at least one even-sized Qwen temporal group')
    requested_count = min(max_even, max(min_even, int(math.ceil(clip_duration * fps * 2))))
    if requested_count % 2:
        requested_count = requested_count + 1 if requested_count < max_even else requested_count - 1
    eligible = [row for row in source_frames if start <= row['pts_seconds'] <= end]
    if not eligible:
        raise ValueError('No source video frames inside observation window')
    requested_count = min(requested_count, len(eligible) - len(eligible) % 2)
    if requested_count < min_frames:
        raise ValueError(f'Only {requested_count} distinct source frames; minimum is {min_frames}')
    times = [row['pts_seconds'] for row in eligible]
    interval = max(1e-6, (times[-1] - times[0]) / max(1, requested_count - 1))
    selected_indices = set()
    for i in range(requested_count):
        target = times[0] + (times[-1] - times[0]) * i / max(1, requested_count - 1)
        pos = bisect.bisect_left(times, target)
        nearest = min({max(0, pos - 1), min(len(times) - 1, pos)},
                      key=lambda j: abs(times[j] - target))
        selected_indices.add(nearest)
    while len(selected_indices) < requested_count:
        remaining = set(range(len(eligible))) - selected_indices
        selected_indices.add(max(remaining, key=lambda j: min(
            abs(times[j] - times[k]) for k in selected_indices)))
    selected = [eligible[i] for i in sorted(selected_indices)]
    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob('frame_*.jpg'):
        old.unlink()
    pattern = str(output_dir / 'frame_%04d.jpg')
    select = '+'.join(f'eq(n\\,{row["source_frame_index"]})' for row in selected)
    resize = f"scale=w='min(iw,{max_side})':h='min(ih,{max_side})':force_original_aspect_ratio=decrease"
    result = subprocess.run(
        ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-threads', '4',
         '-i', str(video_path), '-vf', f'select={select},{resize}', '-vsync', '0',
         '-frames:v', str(len(selected)), '-q:v', '2', pattern], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'ffmpeg dense extraction failed: {result.stderr.strip()}')
    paths = sorted(output_dir.glob('frame_*.jpg'))
    if len(paths) != len(selected):
        raise RuntimeError(f'Expected {len(selected)} source frames, extracted {len(paths)}')
    frames = []
    from PIL import Image
    for index, (path, source) in enumerate(zip(paths, selected)):
        with Image.open(path) as image:
            width, height = image.size
        source_id = f'frame_{source["source_frame_index"]:06d}'
        frames.append({'frame_id': f'V{index+1:03d}', 'view_id': f'V{index+1:03d}',
                       'source_frame_id': source_id, 'source_frame_ids': [source_id],
                       'source_frame_index': source['source_frame_index'],
                       'time_seconds': source['pts_seconds'], 'timestamp_source': 'ffprobe_best_effort_pts',
                       'source_width': stream['width'], 'source_height': stream['height'],
                       'width': width, 'height': height, 'pixel_origin': 'top_left',
                       'resize_scale_xy': [width/stream['width'], height/stream['height']],
                       'crop': None, 'path': str(path), 'sha256': file_sha(path),
                       'selection_reason': 'uniform_time_coverage_nearest_real_pts_no_gt'})
    group_times = [sum(f['time_seconds'] for f in frames[i:i+2]) / 2
                   for i in range(0, len(frames), 2)]
    temporal_groups = qwen_temporal_groups(frames)
    measured_hz = len(group_times) / clip_duration
    return {'video_path': str(video_path), 'video_sha256': file_sha(video_path),
            'duration_seconds': duration, 'requested_time_range': [float(t_start), float(t_end)],
            'clipped_time_range': [start, end],
            'target_qwen_temporal_groups_per_second': fps,
            'target_source_frames_per_second': 2 * fps,
            'actual_source_frame_rate': (len(frames)-1) / max(1e-9, frames[-1]['time_seconds']-frames[0]['time_seconds']),
            'actual_qwen_temporal_group_rate': measured_hz,
            'target_group_rate_met': measured_hz >= fps * 0.9,
            'cadence_limit_reason': None if measured_hz >= fps * 0.9 else 'frame_cap_or_source_density',
            'qwen_temporal_patch_size': 2, 'qwen_temporal_groups': temporal_groups,
            'source_stream': stream,
            'frame_count_requested': requested_count, 'frame_count': len(frames),
            'selection_interval_seconds': interval,
            'selection_method': 'uniform_time_coverage_nearest_real_pts',
            'time_metadata_estimated': False, 'frames': frames}


def scene_candidate_rows(ctx):
    """Return only the anchor table loaded for this event's scene."""
    return [{'candidate_id': c.candidate_id, 'category': category(c.candidate_id),
             'aliases': list(c.aliases)} for c in ctx.candidates]


def _candidate_scope(ctx, mention):
    ids = list(mention.get('candidate_ids') or [])
    by_id = {c['candidate_id']: c for c in scene_candidate_rows(ctx)}
    if ids:
        if not set(ids) <= by_id.keys():
            raise ValueError(f'Mention candidates outside {ctx.scene_key} anchor table: {ids}')
        return 'mention_closed_set', [by_id[name] for name in ids], ids
    tokens = set(''.join(ch.lower() if ch.isalnum() else ' ' for ch in
                         str(mention.get('text', '')).replace("'", ' ')).split())
    is_spatial = (mention.get('kind') in ('region', 'location')
                  or bool(tokens) and tokens <= {'here', 'there', 'inside', 'outside'})
    if is_spatial:
        # Vague locations are not lexical candidate sets. Do not pass the full
        # anchor table as if it were a mention candidate set. The Visual agent
        # records a spatial observation only; a later spatial resolver may map
        # that observation to an anchor using the current scene table.
        return 'unresolved_spatial_observation', [], []
    return 'unresolved_object_observation', [], []


def visual_prompt(ctx, parse, mention, frame_manifest):
    scope, rows, mention_ids = _candidate_scope(ctx, mention)
    frame_meta = prompt_temporal_groups(frame_manifest['qwen_temporal_groups'])
    mid = mention['mention_id']
    output_contract = f'''Return compact JSON with these keys: evidence, visual_observations,
and identity_status. Each evidence row must contain mention_id, candidate_id,
support_score, confidence, frame_id, and explanation. Each observation row must
contain mention_id, frame_id, description, and confidence. Use exact candidate IDs
from this mention's allowed set and exact Gxxx view IDs from video_temporal_views.
There is no example frame or example score to copy. G001 is not a default answer:
cite it only when the evidence is actually visible in that first temporal group.
For a positive evidence row, both support_score and confidence must be strictly
greater than zero. If an ID cannot be associated from RGB, return no evidence and
use ambiguous_instances or not_visible. If the category is visible but the instance
identity is not recoverable, report it only in visual_observations. IDs are arbitrary
labels and are not left-to-right order. Use empty arrays for absent evidence.
'''
    if not mention_ids:
        output_contract += ('This mention has NO candidate IDs. evidence MUST be empty. '
                            'Return observations only, without any object/anchor ID.\n')
    start, end = mention.get('span', [0, 0])
    return '''You are the Visual agent in a mention-first closed-set referring system.
You are solving exactly ONE language mention from a chronological event video.
The supplied video contains the original RGB event frames in temporal order. Use the
whole sequence and compare changes across time; do not decide from the first frame.

The current event belongs to exactly one isolated scene. Candidate IDs from another
scene are forbidden. The candidate table below is the only table available when
this mention has a lexical closed set.
For an ordinary object mention, output only IDs in its mention closed set. For a
deictic/spatial mention with an empty lexical candidate set, do not choose an anchor
ID in this Visual stage; describe the visible spatial destination or region instead.

Do not use gaze numbers, gazePoint, gazeVector, camera pose, spatial_context_text,
ground-truth labels, or candidate distance as direct evidence in this Visual call.
Do not assume a named object is visible. Do not invent an ID. Do not output an ID
outside the allowed scope. Distinguish object identity from category visibility and
compare same-class instances using visible appearance, position, and temporal action.
Language order is a soft action-phase cue, not a precise mention timestamp. This is
offline full-interaction observation. A repeated phrase is the bracketed occurrence.
Each Gxxx view is one Qwen temporal patch made from its two listed source frames.
Evidence citations must use Gxxx; provenance expands that group to both real frames.

''' + output_contract + '''

Instruction: ''' + json.dumps({
        'scene_key': ctx.scene_key,
        'instruction': ctx.text,
        'highlighted_instruction': ctx.text[:start] + '[MENTION]' + ctx.text[start:end] + '[/MENTION]' + ctx.text[end:],
        'mention': mention,
        'mention_candidate_ids': mention_ids,
        'candidate_scope': scope,
        'mention_scene_local_candidate_table': rows,
        'video_temporal_views': frame_meta,
    }, ensure_ascii=False)


def visual_event_prompt(ctx, parse, frame_manifest):
    mention_specs = []
    for mention in parse['mentions']:
        scope, rows, ids = _candidate_scope(ctx, mention)
        start, end = mention.get('span', [0, 0])
        mention_specs.append({
            'mention_id': mention['mention_id'],
            'mention': mention,
            'highlighted_instruction': (
                ctx.text[:start] + '[MENTION]' + ctx.text[start:end] + '[/MENTION]' + ctx.text[end:]),
            'candidate_scope': scope,
            'allowed_candidate_ids': ids,
            'mention_scene_local_candidate_table': rows,
            'required_output': ('observations_only_no_anchor_id' if not ids else 'closed_set_evidence'),
        })
    return '''You are the Visual agent in a mention-first referring system. Inspect this
offline, full-interaction RGB video once, then answer each listed mention independently.
Use the entire chronological sequence. Each mention has its OWN closed candidate set;
never transfer candidates or visual identity assumptions between mentions. Candidate
IDs are scene-local labels, not image coordinates or left-to-right order. If identical
instances cannot be linked to these IDs from RGB evidence, say ambiguous and return no
candidate-specific evidence. A mention with an empty allowed_candidate_ids list MUST
have empty evidence and may return only a spatial/visual observation, never an anchor ID.
Language order is only a soft action-phase clue; there is no precise mention timestamp.
This is offline full-interaction observation, not an online result. Do not use gaze,
camera pose, candidate world coordinates, candidate distance, GT or anchor IDs from
another scene.

Each Gxxx view is one Qwen temporal patch made from two actual source frames. Cite Gxxx;
the saved provenance maps each citation to both original frame IDs and PTS values.
Return exactly one object with a mentions array. For each mention object, include
mention_id, evidence, visual_observations, and identity_status. Each evidence row
must contain mention_id, candidate_id, support_score, confidence, frame_id, and a
brief explanation. Each observation row must contain mention_id, frame_id,
description, and confidence. Return at most 4 visual_observations total, selecting
only frames that show a meaningful change or distinguish the candidate instances.
Do not describe every temporal group. Return at most 2 evidence rows per candidate;
do not emit zero-score rows merely to describe absence. Copy an exact candidate ID
from that mention's allowed set and an exact Gxxx ID from the supplied temporal views. There is no example frame
or score to copy. G001 is not a default answer: cite it only when the evidence is
actually visible in that first temporal group. Positive evidence requires both
support_score and confidence to be strictly greater than zero; otherwise return no
evidence and use ambiguous_instances or not_visible. Keep evidence brief, at most
two rows per candidate, and cite only views that support the claim.

Input: ''' + json.dumps({
        'scene_key': ctx.scene_key,
        'instruction': ctx.text,
        'mentions': mention_specs,
        'video_temporal_views': prompt_temporal_groups(frame_manifest['qwen_temporal_groups']),
    }, ensure_ascii=False)


def validate_batched_visual_response(payload, ctx, parse, frame_manifest):
    if not isinstance(payload, dict) or not isinstance(payload.get('mentions'), list):
        raise ValueError('Batched Visual response must contain a mentions array')
    by_id = {m['mention_id']: m for m in parse['mentions']}
    outputs, rejected = {}, []
    for row in payload['mentions']:
        if not isinstance(row, dict):
            rejected.append({'row': row, 'reason': 'mention_output_not_object'})
            continue
        mid = row.get('mention_id')
        if mid not in by_id:
            rejected.append({'row': row, 'reason': 'unknown_mention_id'})
            continue
        if mid in outputs:
            rejected.append({'row': row, 'reason': 'duplicate_mention_output'})
            continue
        mention = by_id[mid]
        item = {k: v for k, v in row.items() if k in
                ('evidence', 'visual_observations', 'spatial_observations', 'identity_status')}
        for evidence in item.get('evidence', []):
            if isinstance(evidence, dict):
                evidence.setdefault('mention_id', mid)
        for observation_key in ('visual_observations', 'spatial_observations'):
            for observation in item.get(observation_key, []):
                if isinstance(observation, dict):
                    observation.setdefault('mention_id', mid)
        outputs[mid] = validate_visual_response(item, ctx, mention, frame_manifest)
    return outputs, rejected


def validate_visual_response(payload, ctx, mention, frame_manifest):
    if not isinstance(payload, dict):
        raise ValueError('Visual response must be an object')
    allowed_mention = set(mention.get('candidate_ids') or [])
    location_scope = not allowed_mention
    views = {f['view_id']: f for f in frame_manifest['qwen_temporal_groups']}
    frame_ids = set(views)
    # Bare anchor names can also be ordinary category words (e.g. crate,
    # truck). Only unambiguous instance labels require candidate evidence.
    scene_ids = [candidate.candidate_id for candidate in ctx.candidates
                 if re.search(r'\d|[A-Z_]', candidate.candidate_id)]

    def observation_mentions_id(value):
        text = str(value or '')
        return any(re.search(r'(?<![A-Za-z0-9_])' + re.escape(name) + r'(?![A-Za-z0-9_])',
                             text, re.IGNORECASE) for name in scene_ids)

    rows, rejected = [], []
    spatial_observations = []
    for list_name in ('spatial_observations', 'visual_observations'):
        for observation in payload.get(list_name, []):
            if not isinstance(observation, dict):
                rejected.append({'row': observation, 'reason': 'spatial_observation_not_object'})
                continue
            if observation.get('mention_id') != mention['mention_id']:
                rejected.append({'row': observation, 'reason': 'wrong_mention_id'})
                continue
            frame_id = observation.get('frame_id')
            if frame_id not in frame_ids:
                rejected.append({'row': observation, 'reason': 'spatial_observation_missing_frame'})
                continue
            try:
                confidence = float(observation.get('confidence', 0))
            except (TypeError, ValueError):
                rejected.append({'row': observation, 'reason': 'invalid_confidence'})
                continue
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                rejected.append({'row': observation, 'reason': 'invalid_confidence'})
                continue
            frame = views[frame_id]
            if any(key in observation for key in ('candidate_id', 'candidate_ids')):
                rejected.append({'row': observation, 'reason': 'anchor_id_for_unresolved_mention'})
                continue
            if observation_mentions_id(observation.get('description')):
                rejected.append({'row': observation,
                                 'reason': 'candidate_id_text_requires_candidate_specific_evidence'})
                continue
            clean = {k: observation[k] for k in
                     ('mention_id', 'frame_id', 'description', 'visible_category', 'confidence')
                     if k in observation}
            spatial_observations.append({**clean, 'view_id': frame_id,
                'observation_type': list_name,
                'source_frame_ids': frame['source_frame_ids'],
                'source_time_range': frame['source_time_range'],
                'qwen_group_timestamp_seconds': frame['time_seconds']})
    if location_scope:
        if payload.get('evidence'):
            rejected.extend({'row': row, 'reason': 'object_evidence_for_unresolved_spatial'}
                             for row in payload['evidence'])
        return {'evidence': rows, 'visual_observations': spatial_observations,
                'identity_status': _candidate_scope(ctx, mention)[0],
                'reported_identity_status': payload.get('identity_status', 'unspecified'),
                'rejected_evidence': rejected}
    allowed_scene = {c.candidate_id for c in ctx.candidates}
    for row in payload.get('evidence', []):
        if not isinstance(row, dict):
            rejected.append({'row': row, 'reason': 'not_object'})
            continue
        if row.get('mention_id') != mention['mention_id']:
            rejected.append({'row': row, 'reason': 'wrong_mention_id'})
            continue
        name = row.get('candidate_id')
        if name not in allowed_scene:
            rejected.append({'row': row, 'reason': 'outside_scene_anchor_table'})
            continue
        if not location_scope and name not in allowed_mention:
            rejected.append({'row': row, 'reason': 'outside_mention_closed_set'})
            continue
        try:
            score, confidence = float(row.get('support_score', 0)), float(row.get('confidence', 0))
        except (TypeError, ValueError):
            rejected.append({'row': row, 'reason': 'invalid_score'})
            continue
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in (score, confidence)):
            rejected.append({'row': row, 'reason': 'invalid_score'})
            continue
        frame_id = row.get('frame_id')
        if score * confidence > 0 and frame_id not in frame_ids:
            rejected.append({'row': row, 'reason': 'positive_evidence_missing_frame'})
            continue
        frame = views.get(frame_id)
        rows.append({**row, 'support_score': score, 'confidence': confidence,
                     'support_status': 'positive' if score * confidence > 0 else 'no_positive_support',
                     'view_id': frame_id,
                     'source_frame_ids': frame['source_frame_ids'] if frame else [],
                     'source_time_range': frame['source_time_range'] if frame else None,
                     'qwen_group_timestamp_seconds': frame['time_seconds'] if frame else None,
                     'evidence_scope': 'scene_local_spatial' if location_scope else 'mention_closed_set'})
    reported_identity_status = payload.get('identity_status', 'unspecified')
    positive_rows = [row for row in rows if row['support_status'] == 'positive']
    if reported_identity_status == 'supported' and not positive_rows:
        identity_status = 'no_positive_instance_support'
    else:
        identity_status = reported_identity_status
    return {'evidence': rows, 'visual_observations': spatial_observations,
            'identity_status': identity_status,
            'reported_identity_status': reported_identity_status,
            'rejected_evidence': rejected}


def temporal_windows(frame_manifest, max_groups=0):
    """Partition whole temporal patches without dropping or duplicating frames."""
    source_groups = frame_manifest['qwen_temporal_groups']
    if max_groups < 0:
        raise ValueError('temporal-window group count must be nonnegative')
    step = max_groups or len(source_groups)
    windows = []
    for start in range(0, len(source_groups), step):
        original = source_groups[start:start + step]
        frames = frame_manifest['frames'][start * 2:(start + len(original)) * 2]
        window = {**frame_manifest, 'frames': frames, 'frame_count': len(frames),
                  'clipped_time_range': [frames[0]['time_seconds'], frames[-1]['time_seconds']],
                  'qwen_temporal_groups': qwen_temporal_groups(frames),
                  'temporal_window_id': f'W{len(windows) + 1:03d}',
                  'citation_granularity': 'temporal_window' if max_groups else 'event_sequence',
                  'parent_group_indices': [start, start + len(original)],
                  'local_to_global_view_id': {
                      f'G{i + 1:03d}': group['view_id'] for i, group in enumerate(original)}}
        origin = float(frames[0]['time_seconds'])
        for group in window['qwen_temporal_groups']:
            group['request_relative_time_seconds'] = group['time_seconds'] - origin
            group['request_relative_time_label_seconds'] = round(group['time_seconds'] - origin, 1)
        windows.append(window)
    return windows


def remap_window_evidence(checked, window):
    """Keep model citations while using a verified sequence/window provenance."""
    result = copy.deepcopy(checked)
    for key in ('evidence', 'visual_observations'):
        for row in result.get(key, []):
            local_id = row.get('view_id') or row.get('frame_id')
            global_id = window['local_to_global_view_id'].get(local_id)
            source_frames = [frame['source_frame_id'] for frame in window['frames']]
            source_indices = [frame['source_frame_index'] for frame in window['frames']]
            row.update(request_local_view_id=local_id,
                       model_cited_view_id=global_id or local_id,
                       temporal_window_id=window['temporal_window_id'],
                       citation_granularity=window['citation_granularity'],
                       citation_exactness='unverified_model_view',
                       source_frame_ids=source_frames,
                       source_frame_indices=source_indices,
                       source_time_range=window['clipped_time_range'],
                       qwen_group_timestamp_seconds=None)
            if window['citation_granularity'] == 'event_sequence':
                row.update(view_id='EVENT', frame_id='EVENT')
            else:
                row.update(view_id=window['temporal_window_id'], frame_id=window['temporal_window_id'])
    return result


def merge_window_evidence(checked_windows, candidate_scope):
    """Concatenate audited observations; do not make a final referent decision."""
    output = {key: [row for part in checked_windows for row in part.get(key, [])]
              for key in ('evidence', 'visual_observations', 'rejected_evidence')}
    statuses = [part.get('identity_status') for part in checked_windows]
    positive = any(row.get('support_status') == 'positive' for row in output['evidence'])
    output['identity_status'] = (
        'supported' if positive else
        'unresolved_spatial_observation' if candidate_scope == 'unresolved_spatial_observation' else
        'unresolved_object_observation' if candidate_scope == 'unresolved_object_observation' else
        'ambiguous_instances' if 'ambiguous_instances' in statuses else
        'no_positive_instance_support' if 'no_positive_instance_support' in statuses else
        'not_visible')
    output['window_identity_statuses'] = statuses
    output['reported_identity_statuses'] = [part.get('reported_identity_status')
                                            for part in checked_windows]
    return output


def run_visual_calls(ctx, parse, frame_manifest, backend, event_dir, *,
                     granularity='mention', window_groups=0):
    if granularity not in ('mention', 'event'):
        raise ValueError('Visual granularity must be mention or event')
    windows = temporal_windows(frame_manifest, window_groups)
    accumulated = {mention['mention_id']: [] for mention in parse['mentions']}
    failures, responses = [], []
    call_count = 0
    for window in windows:
        # Partition a fixed per-mention video budget across windows. Record the
        # actual processor token count; rounding is not a guarantee of equality.
        pixels = max(4096, int(backend.video_max_pixels * window['frame_count']
                              / frame_manifest['frame_count']))
        groups = [[mention] for mention in parse['mentions']] if granularity == 'mention' else [parse['mentions']]
        for mentions in groups:
            suffix = mentions[0]['mention_id'] if granularity == 'mention' else 'all_mentions'
            stage = f'v6_visual_{suffix}_{window["temporal_window_id"]}'
            prompt = (visual_prompt(ctx, parse, mentions[0], window) if granularity == 'mention'
                      else visual_event_prompt(ctx, {'mentions': mentions}, window))
            call_count += 1
            record = {'stage': stage, 'window_id': window['temporal_window_id'],
                      'mention_ids': [m['mention_id'] for m in mentions],
                      'frame_count': window['frame_count'], 'video_max_pixels': pixels,
                      'source_frame_ids': [f['source_frame_id'] for f in window['frames']],
                      'local_to_global_view_id': window['local_to_global_view_id']}
            try:
                raw = backend.ask_video(
                    sample_key(ctx), stage, prompt, [f['path'] for f in window['frames']],
                    frame_metadata=[{key: f[key] for key in
                                     ('frame_id', 'source_frame_id', 'source_frame_index', 'time_seconds',
                                      'actual_container_pts_seconds', 'pose_status', 'pose_clock_status',
                                      'clock_offset_seconds', 'telemetry_pose_time_seconds')
                                     if key in f}
                                    for f in window['frames']],
                    video_path=ctx.video_path, video_time_range=window['clipped_time_range'],
                    video_max_pixels=pixels, max_new_tokens=768 if granularity == 'mention' else 1536)
                if granularity == 'mention':
                    checked = {mentions[0]['mention_id']: validate_visual_response(
                        raw, ctx, mentions[0], window)}
                    rejected = []
                else:
                    checked, rejected = validate_batched_visual_response(
                        raw, ctx, {'mentions': mentions}, window)
                for mention in mentions:
                    mid = mention['mention_id']
                    if mid not in checked:
                        failures.append({'mention_id': mid, 'window_id': window['temporal_window_id'],
                                         'error': 'Visual response omitted this mention_id'})
                    else:
                        accumulated[mid].append(remap_window_evidence(checked[mid], window))
                record.update(raw_response=raw, rejected_batched_rows=rejected)
            except Exception as exc:
                record['error'] = f'{type(exc).__name__}: {exc}'
                failures.extend({'mention_id': mention['mention_id'],
                                 'window_id': window['temporal_window_id'],
                                 'error': record['error']} for mention in mentions)
            responses.append(record)
            save(event_dir / 'visual_requests' / f'{stage}.json', record)
    results = []
    for mention in parse['mentions']:
        mid = mention['mention_id']
        scope, rows, ids = _candidate_scope(ctx, mention)
        parts = accumulated[mid]
        errors = [row for row in failures if row['mention_id'] == mid]
        result = {'mention': mention, 'candidate_scope': scope, 'candidate_ids': ids,
                  'scene_candidate_count': len(rows), 'scene_anchor_count': len(ctx.candidates),
                  'status': 'PARTIAL' if parts and errors else 'RUN' if parts else 'FAILED',
                  'visual': merge_window_evidence(parts, scope),
                  'windows_observed': len(parts), 'windows_planned': len(windows)}
        if errors:
            result['error'] = '; '.join(f"{e['window_id']}: {e['error']}" for e in errors)
        results.append(result)
        save(event_dir / f'visual_{mid}.json', result)
    save(event_dir / 'visual_response.json', {'requests': responses})
    return results, failures, call_count, windows


def load_frozen_inputs(ctx, source_run):
    folder = Path(source_run) / 'samples' / ctx.scene_key / ctx.event_id
    event_path = folder / 'visual_event.json'
    event = json.loads(event_path.read_text())
    if event.get('scene_key') != ctx.scene_key or event.get('event_id') != ctx.event_id or event.get('instruction') != ctx.text:
        raise ValueError('Frozen event address or instruction mismatch')
    scope_path = folder / 'scene_anchor_scope.json'
    scope = json.loads(scope_path.read_text())
    if scope.get('anchor_table_sha256') != file_sha(Path(ctx.provenance['anchor_table'])):
        raise ValueError('Frozen scene anchor table changed')
    parse = copy.deepcopy(event['referential'])
    frame_manifest = copy.deepcopy(event['frame_manifest'])
    if frame_manifest.get('video_sha256') != file_sha(Path(ctx.video_path)):
        raise ValueError('Frozen source video changed')
    for frame in frame_manifest['frames']:
        path = Path(frame['path']).resolve()
        if file_sha(path) != frame['sha256']:
            raise ValueError(f'Frozen source frame changed: {path}')
        frame['path'] = str(path)
    for mention in parse['mentions']:
        _candidate_scope(ctx, mention)
    return parse, frame_manifest, {
        'source_event': str(event_path.resolve()), 'source_event_sha256': file_sha(event_path),
        'source_anchor_scope': str(scope_path.resolve()),
        'referential_replayed': True, 'frame_plan_replayed': True,
        'image_bytes_verified': True, 'referential_model_calls': 0,
    }


def run_event(ctx, backend, out_dir, *, fps, min_frames, max_frames,
              referential_override=None, referential_recovery_source=None,
              frame_manifest_override=None, frozen_input_source=None,
              granularity='mention', window_groups=0, pose_root=None):
    key = sample_key(ctx)
    event_dir = out_dir / 'samples' / ctx.scene_key / ctx.event_id
    event_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    parse = (referential_override if referential_override is not None
             else ReferentialAgent(backend).run(ctx))
    anchor_path = Path(ctx.provenance['anchor_table'])
    if anchor_path.name != f'{ctx.scene_key}_anchor_table.tsv':
        raise ValueError(f'Anchor table/scene mismatch: {ctx.scene_key}: {anchor_path}')
    candidate_ids = [c.candidate_id for c in ctx.candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError('Duplicate candidate IDs inside scene anchor table')
    duration = probe_duration(Path(ctx.video_path))
    frame_manifest = (copy.deepcopy(frame_manifest_override) if frame_manifest_override is not None
                      else extract_dense_frames(
                          Path(ctx.video_path), event_dir / 'frames', 0.0, duration,
                          fps=fps, min_frames=min_frames, max_frames=max_frames))
    frame_manifest.update(observation_protocol='offline_full_interaction_recording',
                          event_telemetry_range_hint=[float(ctx.api_row['t_start']), float(ctx.api_row['t_end'])],
                          precise_mention_time_used=False, gaze_used_for_selection=False)
    if pose_root is not None:
        pose_summary = bind_pose_contract(ctx, frame_manifest, out_dir, pose_root)
        frame_manifest['pose_alignment'] = pose_summary
        save(event_dir / 'pose_alignment_event.json', pose_summary)
    save(event_dir / 'referential.json', parse)
    save(event_dir / 'frame_manifest.json', frame_manifest)
    save(event_dir / 'scene_anchor_scope.json', {
        'scene_key': ctx.scene_key, 'anchor_table': ctx.provenance['anchor_table'],
        'anchor_table_sha256': file_sha(Path(ctx.provenance['anchor_table'])),
        'candidate_count': len(ctx.candidates),
        'candidate_ids': [c.candidate_id for c in ctx.candidates],
        'cross_scene_anchor_tables_loaded': False,
        'anchor_table_basename_matches_scene': anchor_path.name == f'{ctx.scene_key}_anchor_table.tsv',
        'candidate_ids_are_scene_local': True,
    })
    visual_mentions, failures, call_count, windows = run_visual_calls(
        ctx, parse, frame_manifest, backend, event_dir,
        granularity=granularity, window_groups=window_groups)
    elapsed = time.monotonic() - started
    event = {'version': VERSION, 'scene_key': ctx.scene_key, 'event_id': ctx.event_id,
             'instruction': ctx.text, 'referential': parse,
             'frame_manifest': frame_manifest, 'visual_mentions': visual_mentions,
             'pose_alignment': frame_manifest.get('pose_alignment', {'status': 'NOT_RUN'}),
             'failures': failures, 'fusion_status': 'NOT_RUN',
             'gaze_numeric_input_to_visual': False,
             'hand_input_to_visual': False, 'elapsed_seconds': elapsed}
    if referential_recovery_source is not None:
        event['referential_recovery_source'] = referential_recovery_source
    event['frozen_input_source'] = frozen_input_source
    event['ground_truth_loaded'] = bool(ctx.provenance.get('ground_truth_loaded', True))
    event['visual_call_granularity'] = granularity
    event['visual_temporal_window_count'] = len(windows)
    event['visual_model_calls'] = call_count
    event['pose_alignment'] = frame_manifest.get('pose_alignment', {'status': 'NOT_RUN'})
    save(event_dir / 'visual_event.json', event)
    return event


def run(args):
    if getattr(args, 'base_url', None):
        raise ValueError('--local-dense-video-visual is local-only; omit --base-url')
    fps = getattr(args, 'visual_fps', getattr(args, 'fps', DEFAULT_FPS))
    min_frames = getattr(args, 'visual_min_frames', getattr(args, 'min_frames', DEFAULT_MIN_FRAMES))
    max_frames = getattr(args, 'visual_max_frames', getattr(args, 'max_frames', DEFAULT_MAX_FRAMES))
    max_pixels = getattr(args, 'visual_max_pixels', getattr(args, 'max_video_pixels', DEFAULT_VIDEO_MAX_PIXELS))
    granularity = getattr(args, 'visual_call_granularity', 'mention')
    window_groups = getattr(args, 'visual_window_groups', 0)
    replay_from = getattr(args, 'replay_from', None)
    manifest = json.loads(Path(args.manifest).read_text())
    if getattr(args, 'limit', None):
        manifest = {**manifest, 'samples': manifest['samples'][:args.limit]}
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = {'version': VERSION, 'model': args.qwen_model,
              'local_files_only': True, 'fps': fps,
              'min_frames': min_frames, 'max_frames': max_frames,
              'fusion': 'not_run', 'gaze_numeric_input_to_visual': False,
              'hand_input_to_visual': False, 'scene_anchor_isolation': True,
              'backend_namespace': getattr(args, 'backend_namespace', 'local_dense_video_visual_8b_20261022_v6'),
              'cache_dir': str(Path(args.cache_dir).resolve()),
              'model_video_max_pixels': max_pixels,
              'model_video_token_budget_estimate': max_pixels // 2048,
              'manifest': str(Path(args.manifest).resolve()),
              'frame_selection': 'real source PTS, uniform time coverage, no GT, no processor resampling',
              'frame_pose_binding': 'per event RGB motion fit on real container PTS; rejected candidate falls back to released Projected-2D hybrid offset; fit uses neither GT nor gaze',
              'visual_prompt_pose_policy': 'camera pose, gaze values, and offsets stay in provenance; Visual receives the chronologically ordered source frames and actual PTS labels',
              'processor_video_sampling': False,
              'timestamp_policy': 'relative PTS microsecond ticks in VideoMetadata; absolute PTS retained per frame',
              'visual_call_granularity': granularity,
              'visual_window_groups': window_groups,
              'replay_from': str(Path(replay_from).resolve()) if replay_from else None,
              'ground_truth_context_loading': False,
              'ground_truth_used_for_prediction': False,
              'model_call_policy': 'one video request per mention and temporal window' if granularity == 'mention'
                                   else 'one video request per event temporal window'}
    save(out / 'config.json', config)
    save(out / 'manifest.json', manifest)
    backend = QwenBackend(args.cache_dir, args.qwen_model,
                          namespace=args.backend_namespace, local_only=True,
                          video_max_pixels=max_pixels)
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(20261022)
    contexts = load_contexts(manifest, load_ground_truth=False)
    try:
        attn = importlib.metadata.version('flash-attn')
    except importlib.metadata.PackageNotFoundError:
        attn = None
    config.update({'torch_version': torch.__version__,
                   'transformers_version': importlib.metadata.version('transformers'),
                   'flash_attention_2_installed': attn is not None,
                   'flash_attention_2_version': attn,
                   'attention_backend': 'Transformers model config; recorded per Visual call'})
    save(out / 'config.json', config)
    results, failures = [], []
    run_started = time.monotonic()
    event_durations = []
    for index, ctx in enumerate(contexts, 1):
        print(f'[{index}/{len(contexts)}] {ctx.scene_key}/{ctx.event_id} local dense Visual', flush=True)
        event_started = time.monotonic()
        try:
            overrides = {}
            if replay_from:
                parse, frames, source = load_frozen_inputs(ctx, replay_from)
                overrides = {'referential_override': parse, 'frame_manifest_override': frames,
                             'frozen_input_source': source}
            event = run_event(ctx, backend, out, fps=fps,
                              min_frames=min_frames, max_frames=max_frames,
                              granularity=granularity, window_groups=window_groups,
                              pose_root=out / 'pose_alignment', **overrides)
            stale_failure = out / 'samples' / ctx.scene_key / ctx.event_id / 'event_failure.json'
            if stale_failure.exists():
                stale_failure.unlink()
            results.append({'scene_key': ctx.scene_key, 'event_id': ctx.event_id,
                            'mention_count': len(event['referential']['mentions']),
                            'frame_count': event['frame_manifest']['frame_count'],
                            'pose_available_frames': event.get('pose_alignment', {}).get('pose_available_frames', 0),
                            'candidate_clock_status': event.get('pose_alignment', {}).get('candidate_clock_status'),
                            'pose_selection_status': event.get('pose_alignment', {}).get('pose_selection_status'),
                            'pose_alignment_summary': event.get('pose_alignment', {}),
                            'visual_calls': event['visual_model_calls'],
                            'visual_mentions': len(event['visual_mentions']),
                            'visual_failures': len(event['failures']),
                            'elapsed_seconds': event['elapsed_seconds']})
        except Exception as exc:
            item = {'scene_key': ctx.scene_key, 'event_id': ctx.event_id,
                    'error': f'{type(exc).__name__}: {exc}'}
            failures.append(item)
            save(out / 'samples' / ctx.scene_key / ctx.event_id / 'event_failure.json', item)
            print(f'  FAILED: {item["error"]}', flush=True)
        event_durations.append(time.monotonic() - event_started)
        elapsed = time.monotonic() - run_started
        average = sum(event_durations) / len(event_durations)
        save(out / 'progress.json', {'completed': index, 'total': len(contexts),
                                     'last_event': f'{ctx.scene_key}/{ctx.event_id}',
                                     'failures': len(failures),
                                     'local_model_calls': len([c for c in backend.call_records
                                                               if not c.get('cache_hit')]),
                                     'cache_hits': len([c for c in backend.call_records
                                                        if c.get('cache_hit')]),
                                     'last_event_seconds': event_durations[-1],
                                     'mean_event_seconds': average,
                                     'elapsed_seconds': elapsed,
                                     'estimated_remaining_seconds': average * (len(contexts)-index)})
    calls = backend.call_records
    save(out / 'summary.json', {
        'version': VERSION, 'planned_events': len(contexts),
        'completed_events': len(results), 'failed_events': len(failures),
        'visual_model_calls': sum(r['visual_calls'] for r in results),
        'visual_mentions': sum(r['visual_mentions'] for r in results),
        'visual_failures': sum(r['visual_failures'] for r in results),
        'pose_available_frames': sum(r.get('pose_available_frames', 0) for r in results),
        'pose_selection_status_counts': dict(__import__('collections').Counter(
            r.get('pose_selection_status', 'NOT_RUN') for r in results)),
        'pose_alignment_summaries': [r.get('pose_alignment_summary', {}) for r in results],
        'frame_count_distribution': sorted({r['frame_count'] for r in results}),
        'real_or_local_calls': len([c for c in calls if not c.get('cache_hit')]),
        'cache_hits': len([c for c in calls if c.get('cache_hit')]),
        'call_audit': calls, 'events': results, 'failures': failures,
    })
    print(json.dumps({'completed_events': len(results), 'failed_events': len(failures),
                      'visual_calls': sum(r['visual_calls'] for r in results),
                      'visual_failures': sum(r['visual_failures'] for r in results)}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--model', dest='qwen_model', default='/media/kunyu/16TB/spj/Qwen3-VL-8B-Instruct')
    parser.add_argument('--backend-namespace', default='local_dense_video_visual_8b_20261022_v6')
    parser.add_argument('--replay-from', help='Freeze Referential and exact source-frame bytes from a prior run.')
    parser.add_argument('--visual-call-granularity', choices=['mention', 'event'], default='mention')
    parser.add_argument('--visual-window-groups', type=int, default=0,
                        help='Maximum two-frame temporal groups per request; 0 keeps the whole sequence.')
    parser.add_argument('--fps', type=float, default=DEFAULT_FPS,
                        help='Target Qwen temporal groups per second; source selection targets twice this rate.')
    parser.add_argument('--min-frames', type=int, default=DEFAULT_MIN_FRAMES)
    parser.add_argument('--max-frames', type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument('--max-video-pixels', type=int, default=DEFAULT_VIDEO_MAX_PIXELS)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
