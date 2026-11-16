"""Use supported attention internally to route RGB evidence, never as model pixels.

The complete source field is always kept. Detail windows are evidence allocation,
not object proposals or referent predictions. No anchors, labels or language are
accepted by the routing API. Every detail has an explicit source-pixel transform.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import maximum_filter, uniform_filter

from .full_frame_grounding_v4 import mention_units
from .full_frame_grounding import categories_for_candidate
from .local_video_visual import prompt_temporal_groups
from .mention_mapping import actor_coverage_checks, literal_span

VERSION = 'gaze_rgb_routing_v1'
CONDITIONS = ('FULL_RGB', 'UNIFORM_RGB', 'GAZE_RGB', 'MIRRORED_GAZE_RGB')
HEADER = 28


@dataclass(frozen=True)
class RoutingConfig:
    crop_fraction: float = .40
    minimum_support_seconds: float = .10
    maximum_overlap_iou: float = .60

    def __post_init__(self):
        if not math.isfinite(self.crop_fraction) or not 0 < self.crop_fraction <= 1:
            raise ValueError('crop_fraction must be in (0,1]')
        if not math.isfinite(self.minimum_support_seconds) or self.minimum_support_seconds <= 0:
            raise ValueError('minimum_support_seconds must be positive')
        if not math.isfinite(self.maximum_overlap_iou) or not 0 <= self.maximum_overlap_iou <= 1:
            raise ValueError('maximum_overlap_iou must be in [0,1]')


def _box(x, y, side, width, height):
    left = int(np.clip(round(x - side / 2), 0, width - side))
    top = int(np.clip(round(y - side / 2), 0, height - side))
    return [left, top, left + side, top + side]


def _iou(a, b):
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter)


def route_regions(maps, image_size, *, clock_supported, condition='GAZE_RGB', config=None):
    """Three chronological slots, one per historical phase; full RGB fallback.

    Pick a maximum-mass fixed-size window in each phase. When possible, use a
    near-maximum distinct window rather than repeat an earlier slot. Low-support
    phases retain a full-field fallback; missing evidence never becomes a target.
    Scores remain in audit output and must not be sent to the model.
    """
    config = config or RoutingConfig()
    width, height = image_size
    if min(width, height) <= 0 or condition not in CONDITIONS:
        raise ValueError('Positive image size and known condition required')
    values = np.asarray(maps, dtype=np.float64)
    if values.shape != (3, height, width) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Three finite nonnegative registered attention planes required')
    if condition == 'MIRRORED_GAZE_RGB':
        values = values[:, :, ::-1].copy()
    full = [0, 0, width, height]
    side = max(1, min(min(width, height), int(round(min(width, height) * config.crop_fraction))))
    slots, selected = [], []
    for phase, plane in enumerate(values):
        seconds = float(plane.sum())
        fallback = None
        if condition == 'FULL_RGB':
            fallback = 'full_field_control'
        elif not clock_supported:
            fallback = 'unsupported_clock_or_pose'
        elif seconds < config.minimum_support_seconds:
            fallback = 'insufficient_phase_support'
        if fallback:
            box = full[:]
        elif condition == 'UNIFORM_RGB':
            box = _box(width * (.2 + .3 * phase), height * .5, side, width, height)
        else:
            # Filter mass at window centers. Clamp to legal source crops rather
            # than padding synthetic pixels around image edges.
            mass = uniform_filter(plane, size=side, mode='constant') * side * side
            lo = side // 2
            hi_x, hi_y = width - side + lo, height - side + lo
            legal = mass[lo:hi_y + 1, lo:hi_x + 1]
            local_max = legal == maximum_filter(legal, size=max(3, side // 3), mode='constant')
            yy, xx = np.nonzero(local_max & (legal > 0))
            candidates = sorted(zip(yy, xx), key=lambda p: (-legal[p], p[0], p[1]))
            if not candidates:
                fallback, box = 'no_positive_window', full[:]
            else:
                best_mass = float(legal[candidates[0]])
                # Never invent a remote ROI just to fill a slot. Diversity is
                # allowed only among windows with >=80% of the phase maximum.
                chosen = candidates[0]
                for y, x in candidates:
                    if legal[y, x] < .8 * best_mass:
                        break
                    candidate_box = _box(x + lo, y + lo, side, width, height)
                    if all(_iou(candidate_box, prior) <= config.maximum_overlap_iou for prior in selected):
                        chosen = (y, x)
                        break
                box = _box(chosen[1] + lo, chosen[0] + lo, side, width, height)
        x0, y0, x1, y1 = box
        captured = float(plane[y0:y1, x0:x1].sum())
        slots.append({'slot': phase + 1, 'phase_old_to_new': phase,
                      'bbox_source_xyxy': box, 'fallback_reason': fallback,
                      'phase_support_seconds': seconds, 'captured_support_seconds': captured,
                      'captured_fraction': captured / seconds if seconds else None})
        if not fallback:
            selected.append(box)
    return {'version': VERSION, 'condition': condition, 'config': asdict(config),
            'image_size': [width, height], 'clock_supported': bool(clock_supported),
            'slots': slots, 'full_field_retained': True, 'candidate_filtering': False,
            'object_identity_assigned': False, 'heatmap_sent_to_model': False}


def render_rgb_evidence(rgb, routing):
    """Full RGB plus three RGB detail windows; labels never encode heat or rank."""
    rgb = rgb.convert('RGB')
    width, height = rgb.size
    if routing['image_size'] != [width, height]:
        raise ValueError('Routing/source dimensions disagree')
    canvas = Image.new('RGB', (width * 2, (height + HEADER) * 2), '#15212b')
    positions = [(0, HEADER), (width, HEADER), (0, height + HEADER * 2), (width, height + HEADER * 2)]
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=17)
    for index, (x, y) in enumerate(positions):
        label = 'FULL RGB | ORIGINAL FIELD' if index == 0 else f'RGB DETAIL {index}'
        draw.text((x + 8, y - HEADER + 3), label, fill='white', font=font)
        if index == 0:
            panel = rgb
        else:
            box = routing['slots'][index - 1]['bbox_source_xyxy']
            panel = rgb.crop(box)
            # Preserve aspect ratio using letterboxing. Record dimensions in
            # the per-frame prompt; model points still refer only to full RGB.
            scale = min(width / panel.width, height / panel.height)
            size = (max(1, round(panel.width * scale)), max(1, round(panel.height * scale)))
            panel = panel.resize(size, Image.Resampling.LANCZOS)
            padded = Image.new('RGB', (width, height), '#141414')
            padded.paste(panel, ((width - size[0]) // 2, (height - size[1]) // 2))
            panel = padded
        canvas.paste(panel, (x, y))
    return canvas


def literal_mention_hints(payload, instruction):
    """Validate instruction-only spans independently of candidate mapping.

    A pronoun component can have a valid literal span even when a parser's
    optional parent_text violates the old candidate-mapping schema. Those
    identity fields are not needed here and are never repaired or forwarded.
    """
    if (not isinstance(payload, dict) or 'selected_object_names' in payload
            or not isinstance(payload.get('mentions'), list) or not payload['mentions']):
        raise ValueError('Nonempty instruction-only mention list required')
    counts, ids, hints = {}, set(), []
    for index, raw in enumerate(payload['mentions']):
        if not isinstance(raw, dict):
            raise ValueError('Mention must be an object')
        text = raw.get('text')
        if not isinstance(text, str) or not text or text not in instruction:
            raise ValueError('Mention must be an exact literal instruction substring')
        occurrence = counts.get(text.casefold(), 0)
        span = literal_span(text, instruction, occurrence)
        if span is None:
            raise ValueError('Mention occurrence does not exist in instruction')
        counts[text.casefold()] = occurrence + 1
        mention_id = raw.get('mention_id', f'm{index}')
        if not isinstance(mention_id, str) or not mention_id or mention_id in ids:
            raise ValueError('Mention IDs must be unique strings')
        ids.add(mention_id)
        hints.append({'mention_id': mention_id, 'text': text, 'span': span,
                      'role': raw.get('role'), 'kind': raw.get('kind'),
                      'coreference_id': raw.get('coreference_id')})
    missing = actor_coverage_checks(instruction, hints)
    if missing:
        raise ValueError('Missing explicit actor mention(s): ' + ', '.join(m['text'] for m in missing))
    return hints


PROMPT = '''Ground every listed mention occurrence in the complete chronological first-person interaction.
The TOP LEFT panel of every frame is the unchanged FULL RGB field. The other three
panels are supplementary RGB details from that same frame, sometimes a repeated
full field. Their source rectangles are given below in original source pixels.
Details are optional viewing aids, not target labels or object proposals. Objects
outside details remain valid. Use full frames, actions, pointing/contact, appearance,
spatial relations and the original instruction. Do not select an object merely
because it has a detail view. Resolve repeated mention occurrences separately.

For every mention_id return exactly one grounded or unresolved row. Grounded points
use the WHOLE displayed composite on a 0..1000 scale, and MUST fall ON the intended
object in the TOP LEFT FULL RGB panel, even if identified using a detail. Never
point in another panel. Scripts match against the entire scene candidate table.
Do not guess scene anchor IDs. Return compact JSON only:
{"mentions":[{"mention_id":"M001","mention":"literal phrase","status":"grounded",
"category":"scene category","view_id":"Gxxx","x":0,"y":0,"unresolved_reason":null}]}
For grounded rows category/view_id/x/y are required. For unresolved rows those four
fields must be null; unresolved_reason is not_visible, uncertain_identity,
no_supported_view, non_concrete_reference or other. Copy literal mention text.
For a purely relational/directional phrase with no independent object identity,
use unresolved/non_concrete_reference. A named concrete destination may be grounded.
For parts without an independent scene category, ground the parent object.
Two source frames form each Gxxx view. Full sequence observation is offline;
there is no exact speech-time alignment. INPUT: '''


def grounding_prompt(event, condition):
    vocabulary = sorted(set().union(*(categories_for_candidate(c) for c in event['scene_candidate_table'])))
    details = [{'frame_id': frame['frame_id'], 'time_seconds': frame['time_seconds'],
                'source_size': frame['source_size'],
                'detail_source_rectangles_xyxy': [slot['bbox_source_xyxy'] for slot in frame['routing'][condition]['slots']]}
               for frame in event['frames']]
    return PROMPT + json.dumps({'instruction': event['instruction'],
        'mention_units': mention_units(event), 'scene_category_vocabulary': vocabulary,
        'video_views': prompt_temporal_groups(event['qwen_temporal_groups']),
        'rgb_detail_provenance': details}, ensure_ascii=False)
