"""Full-frame, full-interaction referent selection; no answer-conditioned objects.

Candidate identities come from the entire scene anchor table. No anchor is
projected, cropped, or chosen using referents. Ground truth belongs only to the
separate post-inference evaluator.
"""
from __future__ import annotations
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

VERSION = 'full_interaction_referring_v1'
CONDITIONS = ('NO_ADDED_HEAT', 'TRUE_HEAT', 'SPATIAL_MIRROR', 'TIME_REVERSE')
PHASE_NAMES = ('EARLIER', 'MIDDLE', 'RECENT')


def control_maps(maps, condition):
    maps = np.asarray(maps, dtype=np.float32)
    if maps.ndim != 3 or maps.shape[0] != 3 or not np.all(np.isfinite(maps)) or np.any(maps < 0):
        raise ValueError('Three nonnegative finite attention planes required')
    if condition == 'NO_ADDED_HEAT': return np.zeros_like(maps)
    if condition == 'TRUE_HEAT': return maps.copy()
    if condition == 'SPATIAL_MIRROR': return maps[:, :, ::-1].copy()
    if condition == 'TIME_REVERSE': return maps[::-1].copy()
    raise ValueError(condition)


def render_full_frame(rgb, maps, condition):
    """2x2 full-field panels. Original RGB pixels are unaltered at top left.

    All three dark heat panels have the complete source field, same size and
    aspect, rather than object crops. Printed labels stay outside panel pixels.
    Heat has seconds-of-support semantics, not calibrated target probability.
    """
    rgb = rgb.convert('RGB')
    width, height = rgb.size
    transformed = control_maps(maps, condition)
    if transformed.shape != (3, height, width):
        raise ValueError('Heat and full RGB field must match')
    header = 28
    canvas = Image.new('RGB', (width * 2, (height + header) * 2), '#15212b')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=17)
    positions = [(0, header), (width, header), (0, height + header * 2), (width, height + header * 2)]
    labels = ['ORIGINAL RGB | FULL FIELD', 'EARLIER ATTENTION | FULL FIELD',
              'MIDDLE ATTENTION | FULL FIELD', 'RECENT ATTENTION | FULL FIELD']
    for label, (x, y) in zip(labels, positions):
        draw.text((x + 8, y - header + 3), label, fill='white', font=font)
    canvas.paste(rgb, positions[0])
    for k in range(3):
        mass = float(transformed[k].sum())
        peak = float(transformed[k].max())
        alpha = (.95 * np.sqrt(transformed[k] / peak) * min(1., mass / .25)
                 if peak > 0 else np.zeros((height, width), dtype=np.float32))
        base = np.full((height, width, 3), 20., dtype=np.float32)
        heat = base * (1 - alpha[..., None]) + np.array([255, 85, 25]) * alpha[..., None]
        canvas.paste(Image.fromarray(heat.clip(0, 255).astype(np.uint8)), positions[k + 1])
    return canvas


PROMPT = '''Resolve the ORIGINAL INSTRUCTION for this entire first-person interaction.
Use the complete chronological video, its visible hand/action cues, and spatial
relations to select ALL intended concrete object identities, including actors,
acted-on objects, destinations and concrete reference objects. A component such
as a hood or door resolves to its parent anchor when no component anchor exists.
Resolve repeated mentions individually. Do not reduce the instruction to its
last action or to the object with the most attention.

Every supplied frame has FOUR full-field panels. TOP LEFT is the original RGB;
TOP RIGHT / BOTTOM LEFT / BOTTOM RIGHT are EARLIER / MIDDLE / RECENT graphical
attention during the two seconds preceding that source frame. Each panel covers
the exact same full field and uses the same native orientation as RGB. Orange
means measured attention density, not target identity. Black means missing added
evidence, never evidence against an object. Some graphics are experimental
controls. RGB already contains recorder marks. Attention may be on a hand,
background, a reference or an unrelated object. Use it only when consistent with
language and RGB. Never report heat colors or coordinates as your answer.

The candidate table contains the ENTIRE current scene, not an answer-filtered
pair. Candidate names are stable scene addresses; numeric suffixes do not imply
left-to-right order. Shared world coordinates and per-frame camera pose may help
connect visual objects to these IDs. They contain no gaze or answer coordinates.
Unity convention: x right, y up, z forward in camera space. Camera forward/up are
world directions. The pose is approximate and anchor centers may not be on the
visible surface. No anchor projection or target boxes are drawn. Do not claim a
candidate visible from its name alone; if an instance cannot be resolved, leave
that mention unselected and retain its text in unresolved_mentions. Do not add
spatial 'here/there' as an object unless it actually names a concrete anchor.

Return one compact JSON object with exactly these keys:
selected_object_names: unique array of intended candidate IDs from the table;
decisions: array of {mention_text, candidate_ids, view_ids}; one entry per
concrete mention occurrence, candidate_ids can be empty, and view_ids cite Gxxx
video temporal groups where supporting visual evidence occurs;
unresolved_mentions: array of literal instruction phrases whose object identity
cannot be resolved. Do not output explanations, confidence scores, or attention
readout. The selected_object_names must equal the union of decisions' IDs.
Two source frames form each Gxxx temporal view. This is offline observation of
the full interaction, not online prediction or exact speech-time alignment.

INPUT: '''


def referring_prompt(instruction, candidate_table, frames):
    from .local_video_visual import qwen_temporal_groups, prompt_temporal_groups
    groups = prompt_temporal_groups(qwen_temporal_groups(frames))
    return PROMPT + json.dumps({'instruction': instruction, 'scene_candidates': candidate_table,
        'video_views': groups,
        'source_frame_camera_poses': [{'frame_id': f['frame_id'], 'time_seconds': f['time_seconds'],
                                     'camera': f['prompt_camera']} for f in frames]}, ensure_ascii=False)


def validate_prediction(parsed, candidate_ids, view_ids, instruction):
    """Reject unsupported structure rather than silently repair semantic choices."""
    errors = []
    if not isinstance(parsed, dict): return {'schema_valid': False, 'errors': ['not_object'], 'selected_object_names': []}
    required = {'selected_object_names', 'decisions', 'unresolved_mentions'}
    if set(parsed) != required: errors.append('wrong_keys')
    selected, decisions, unresolved = (parsed.get(k) for k in ('selected_object_names', 'decisions', 'unresolved_mentions'))
    allowed = set(candidate_ids)
    if not isinstance(selected, list) or not all(isinstance(x, str) and x in allowed for x in selected):
        errors.append('invalid_selected_ids')
        selected = []
    elif len(set(selected)) != len(selected): errors.append('duplicate_selected_ids')
    union = set()
    if not isinstance(decisions, list): errors.append('decisions_not_array')
    else:
        for row in decisions:
            if not isinstance(row, dict) or set(row) != {'mention_text', 'candidate_ids', 'view_ids'}:
                errors.append('invalid_decision'); continue
            text, ids, views = (row[k] for k in ('mention_text', 'candidate_ids', 'view_ids'))
            if not isinstance(text, str) or not text or text not in instruction: errors.append('nonliteral_mention')
            if not isinstance(ids, list) or not all(isinstance(x, str) and x in allowed for x in ids):
                errors.append('invalid_decision_ids'); continue
            if len(ids) != len(set(ids)): errors.append('duplicate_decision_ids')
            union.update(ids)
            if not isinstance(views, list) or not all(isinstance(x, str) and x in view_ids for x in views): errors.append('invalid_view_ids')
    if union != set(selected): errors.append('selected_union_mismatch')
    if not isinstance(unresolved, list) or not all(isinstance(x, str) and x and x in instruction for x in unresolved):
        errors.append('invalid_unresolved_mentions')
    return {'schema_valid': not errors, 'errors': sorted(set(errors)),
            'selected_object_names': list(selected) if not errors else [],
            'failure_policy': 'Schema failures score as empty selection; raw output remains archived.'}


def paired_set_metrics(rows):
    """Point estimates from evaluable interaction rows; no resampling frames."""
    tp = fp = fn = exact = any_hit = 0
    for row in rows:
        gt, pred = set(row['gt']), set(row['prediction'])
        tp += len(gt & pred); fp += len(pred - gt); fn += len(gt - pred)
        exact += gt == pred; any_hit += bool(gt & pred)
    p = tp / (tp + fp) if tp + fp else 0.
    r = tp / (tp + fn) if tp + fn else 0.
    return {'events': len(rows), 'exact_set_count': exact, 'exact_set': exact / len(rows) if rows else None,
            'any_hit_count': any_hit, 'any_hit': any_hit / len(rows) if rows else None,
            'tp': tp, 'fp': fp, 'fn': fn, 'micro_precision': p, 'micro_recall': r,
            'micro_f1': 2*p*r/(p+r) if p+r else 0.}
