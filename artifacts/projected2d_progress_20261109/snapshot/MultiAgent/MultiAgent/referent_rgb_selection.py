"""Mention-aware RGB search and transparent exact-set assembly, without GT.

Reuses the existing language mapping and soft temporal planner. Attention remains
inside the RGB router. Model points are decoded in a named original source frame,
including exact inversion of RGB detail crops; no temporal-patch averaging.
"""
from __future__ import annotations

from collections import Counter
import copy
import json
import math
import re
from types import SimpleNamespace

from PIL import Image, ImageDraw, ImageFont

from .full_frame_grounding import categories_for_candidate, decode_grounding, normalized_category
from .gaze_rgb_routing import HEADER, literal_mention_hints
from .local_video_visual import prompt_temporal_groups, qwen_temporal_groups
from .mention_mapping import actor_coverage_checks, resolve_mentions
from .preprocess import plan_frames
from .temporal_focus import focus_spec

VERSION = 'referent_rgb_selection_v2'
UNRESOLVED = {'not_visible', 'uncertain_identity', 'no_supported_view', 'non_concrete_reference', 'other'}


def recover_component_schema(raw, context):
    """Move an explicit pronoun-parent field to the old supported form.

    Only literal prior antecedent + shared coreference + literal component allow
    this syntax normalization. No candidate identity is selected or guessed.
    """
    repaired = copy.deepcopy(raw)
    literal = literal_mention_hints(raw, context.text)
    audit = []
    for index, row in enumerate(repaired['mentions']):
        parent, component = row.get('parent_text'), row.get('component')
        if not parent or parent in row['text']:
            continue
        antecedents = [other for other in repaired['mentions'][:index]
                       if other.get('text') == parent == row.get('anchor_text')
                       and other.get('coreference_id') == row.get('coreference_id')]
        if (not isinstance(component, str) or not component or component not in row['text']
                or not re.match(r'^(its|their|his|her)\b', row['text'], re.I) or not antecedents):
            raise ValueError('Unverified pronoun-parent relation')
        row['parent_text'] = ''
        audit.append({'mention_id': literal[index]['mention_id'], 'literal_parent_text_moved_to_anchor': parent})
    resolved = resolve_mentions(context, repaired)
    if actor_coverage_checks(context.text, resolved['mentions']):
        raise ValueError('Explicit actor missing after component normalization')
    resolved['component_schema_normalization'] = audit
    return resolved


def temporal_search(event):
    """Existing clause-based prior in video coordinates; never a speech timestamp."""
    first, last = event['frames'][0]['time_seconds'], event['frames'][-1]['time_seconds']
    context = SimpleNamespace(text=event['instruction'], timed_samples=[],
        api_row={'t_start': first, 't_end': last, 't_peak': (first + last) / 2})
    alignment = {'clock': {'telemetry_to_video': 0., 'calibration_status': 'video_coordinate_soft_search_only'},
                 'mentions': {}, 'instruction_coverage': 0.}
    plan = plan_frames(context, event['mention_hints'], alignment, last + .1, timed=False)
    frames = [{'frame_id': f['frame_id'], 'time': f['time_seconds']} for f in event['frames']]
    specs = {m['mention_id']: focus_spec(event['instruction'], m,
               plan['mention_windows'][m['mention_id']], frames, False) for m in event['mention_hints']}
    return {'plan': plan, 'mentions': specs, 'asr_used': False,
            'soft_prior_not_restriction': True, 'all_event_frames_retained_in_initial_pass': True}


def labelled_rgb(image, frame):
    """Actual source ID in the existing header; original full RGB is untouched."""
    image = image.convert('RGB').copy()
    width = image.width // 2
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width - 1, HEADER - 1), fill='#15212b')
    draw.text((8, 3), f"{frame['frame_id']} | PTS {frame['time_seconds']:.3f}s | FULL RGB",
              fill='white', font=ImageFont.load_default(size=17))
    return image


def panel_to_source(frame, condition, panel, x, y):
    if type(panel) is not int or not 0 <= panel <= 3:
        raise ValueError('Unknown RGB panel')
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1000 for v in (x, y)):
        raise ValueError('Panel points must be finite coordinates in [0,1000]')
    width, height = frame['source_size']
    if panel == 0:
        return float(x) / 1000., float(y) / 1000.
    left, top, right, bottom = frame['routing'][condition]['slots'][panel - 1]['bbox_source_xyxy']
    bw, bh = right - left, bottom - top
    scale = min(width / bw, height / bh)
    dw, dh = max(1, round(bw * scale)), max(1, round(bh * scale))
    pad_x, pad_y = (width - dw) // 2, (height - dh) // 2
    px, py = float(x) * width / 1000., float(y) * height / 1000.
    if not pad_x <= px <= pad_x + dw or not pad_y <= py <= pad_y + dh:
        raise ValueError('Point is in detail letterboxing, not source RGB')
    return (left + (px - pad_x) * bw / dw) / width, (top + (py - pad_y) * bh / dh) / height


def composite_to_source(frame, condition, x, y):
    """Keep V4's whole-composite convention; derive the panel in the script."""
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1000 for v in (x, y)):
        raise ValueError('Whole-composite point must be finite in [0,1000]')
    width, height = frame['source_size']
    px, py = x / 1000. * width * 2, y / 1000. * (height + HEADER) * 2
    column = min(1, int(px // width))
    row = min(1, int(py // (height + HEADER)))
    top = row * (height + HEADER) + HEADER
    if not top <= py <= top + height:
        raise ValueError('Point falls on a header instead of RGB')
    panel = row * 2 + column
    local_x = (px - column * width) / width * 1000
    local_y = (py - top) / height * 1000
    sx, sy = panel_to_source(frame, condition, panel, local_x, local_y)
    return sx, sy, panel


def validate_choices(parsed, event, condition, supplied_ids=None):
    by_mid = {m['mention_id']: m for m in event['mention_hints']}
    frames = {f['frame_id']: f for f in event['frames']}
    supplied = set(frames) if supplied_ids is None else set(supplied_ids)
    result = {'schema_valid': False, 'coverage_complete': False, 'rows': [], 'rejected': [],
              'missing_ids': list(by_mid), 'duplicate_ids': []}
    if not isinstance(parsed, dict) or set(parsed) != {'mentions'} or not isinstance(parsed['mentions'], list):
        result['rejected'].append({'reason': 'wrong_top_level_schema'})
        return result
    counts = Counter(row.get('mention_id') for row in parsed['mentions']
                     if isinstance(row, dict) and isinstance(row.get('mention_id'), str))
    for row in parsed['mentions']:
        try:
            if not isinstance(row, dict) or set(row) - {'mention_id', 'status', 'instances', 'unresolved_reason'}:
                raise ValueError('Wrong mention row fields')
            mid = row.get('mention_id')
            if not isinstance(mid, str) or mid not in by_mid or counts[mid] != 1:
                raise ValueError('Unknown or duplicate mention occurrence')
            status = row.get('status')
            instances = row.get('instances')
            if status == 'unresolved':
                if instances != [] or not isinstance(row.get('unresolved_reason'), str) or row['unresolved_reason'] not in UNRESOLVED:
                    raise ValueError('Unresolved row must have empty instances and a reason')
                result['rows'].append({**row, 'instances': []})
                continue
            if status != 'grounded' or not isinstance(instances, list) or not 1 <= len(instances) <= 20:
                raise ValueError('Grounded row needs 1..20 observations')
            if row.get('unresolved_reason') is not None:
                raise ValueError('Grounded row has an unresolved reason')
            normalized, errors = [], []
            vocabulary = set().union(*(categories_for_candidate(c) for c in event['scene_candidate_table']))
            for instance in instances:
                try:
                    if not isinstance(instance, dict) or set(instance) != {'frame_id', 'x', 'y', 'category'}:
                        raise ValueError('Wrong instance schema')
                    fid = instance['frame_id']
                    if not isinstance(fid, str) or fid not in frames or fid not in supplied:
                        raise ValueError('Source frame was not supplied')
                    category = instance['category']
                    if not isinstance(category, str) or normalized_category(category) not in vocabulary:
                        raise ValueError('Unknown scene category')
                    x, y, panel = composite_to_source(frames[fid], condition, instance['x'], instance['y'])
                    normalized.append({**instance, 'source_panel': panel, 'x_source': x, 'y_source': y})
                except (ValueError, TypeError, KeyError) as exc:
                    errors.append({'instance': instance, 'reason': str(exc)})
            if not normalized:
                raise ValueError('No valid source observations')
            result['rows'].append({**row, 'instances': normalized, 'rejected_instances': errors})
            result['rejected'].extend({'mention_id': mid, **error} for error in errors)
        except (ValueError, TypeError, KeyError) as exc:
            result['rejected'].append({'mention_id': row.get('mention_id') if isinstance(row, dict) else None,
                                       'reason': str(exc)})
    result['missing_ids'] = sorted(set(by_mid) - {r['mention_id'] for r in result['rows']})
    result['duplicate_ids'] = sorted(mid for mid, count in counts.items() if count > 1)
    result['schema_valid'] = not result['rejected']
    result['coverage_complete'] = not result['missing_ids'] and not result['duplicate_ids']
    return result


def deictic_visual_search(mention):
    return (not mention.get('candidate_ids') and
            (mention.get('candidate_mapping_reason') == 'contextual_location_unresolved'
             or mention.get('kind') in ('location', 'region')))


def visual_decisions(checked, event):
    """Same geometric thresholds, but a named source frame and lexical pool."""
    candidates = {c['candidate_id']: c for c in event['scene_candidate_table']}
    by_mid = {m['mention_id']: m for m in event['mention_hints']}
    ledger = {}
    for row in checked['rows']:
        mid = row['mention_id']
        mention = by_mid[mid]
        compatible = ([candidates[c] for c in mention.get('candidate_ids', []) if c in candidates]
                      if not deictic_visual_search(mention) else list(candidates.values()))
        decisions, names = [], []
        for instance in row['instances']:
            family = normalized_category(instance['category'])
            eligible = [c for c in compatible if family in categories_for_candidate(c)]
            if not eligible:
                decisions.append({'reason': 'category_incompatible_with_language', 'visual_choice': instance,
                                  'selected_candidate': None})
                continue
            source = instance['frame_id']
            decoder_event = {**event, 'scene_candidate_table': eligible,
                'qwen_temporal_groups': [{'view_id': source, 'frame_ids': [source]}]}
            decoder_row = {'mention': mention['text'], 'mention_id': mid, 'category': instance['category'],
                           'view_id': source, 'x': instance['x_source'], 'y': instance['y_source']}
            decoded = decode_grounding([decoder_row], decoder_event)
            decisions.extend(decoded['decisions'])
            names.extend(decoded['selected_object_names'])
        unique = list(dict.fromkeys(names))
        count = mention.get('count', 1)
        conflict = isinstance(count, int) and len(unique) > count
        ledger[mid] = {'selected': [] if conflict else unique, 'decoded_before_cardinality': unique,
                       'cardinality_conflict': conflict, 'decisions': decisions, 'status': row['status'],
                       'unresolved_reason': row.get('unresolved_reason')}
    return ledger


def assemble(event, checked, *, lexical=True):
    """Per-occurrence union; no unrestricted global coordinator or GT repair."""
    visual = visual_decisions(checked, event)
    ledger, selected = {}, []
    for mention in event['mention_hints']:
        mid = mention['mention_id']
        decision = visual.get(mid, {'selected': [], 'decisions': [], 'cardinality_conflict': False})
        names = list(decision['selected'])
        source = 'visual_source_frame_geometry' if names else 'unresolved'
        compatible = list(dict.fromkeys(mention.get('candidate_ids', [])))
        concrete = bool(compatible) and mention.get('candidate_mapping_reason') != 'contextual_location_unresolved'
        if lexical and mention.get('explicit_ids'):
            names, source = list(mention['explicit_ids']), 'explicit_language_id'
        elif lexical and concrete and len(compatible) == 1:
            names, source = compatible, 'unique_language_compatible_identity'
        parent = mention.get('parent_mention_id')
        if lexical and parent in ledger and ledger[parent]['selected']:
            inherited = [name for name in ledger[parent]['selected'] if name in compatible]
            if inherited:
                names, source = inherited, 'verified_language_component_or_pronoun_coreference'
        if not concrete and names and deictic_visual_search(mention):
            source = 'visual_concrete_resolution_of_deictic_location'
        elif not concrete:
            names, source = [], 'no_lexical_concrete_scene_identity'
        count = mention.get('count', 1)
        conflict = isinstance(count, int) and len(names) > count
        if conflict:
            names, source = [], 'ambiguous_cardinality'
        ledger[mid] = {**decision, 'selected': names, 'selection_source': source,
                       'cardinality_conflict': decision.get('cardinality_conflict', False) or conflict,
                       'candidate_ids': compatible, 'count': count,
                       'visual_visibility_asserted': source in ('visual_source_frame_geometry', 'visual_concrete_resolution_of_deictic_location')}
        for name in names:
            if name not in selected:
                selected.append(name)
    return {'selected_object_names': selected, 'mention_decisions': ledger,
            'gt_used': False, 'global_cross_mention_reassignment': False}


def recovery_mentions(event, assembled):
    pending = []
    for mention in event['mention_hints']:
        row = assembled['mention_decisions'][mention['mention_id']]
        if not row['candidate_ids'] and not deictic_visual_search(mention):
            continue
        count = row.get('count')
        short = isinstance(count, int) and len(row['selected']) < count
        if (not row['selected'] or short or row['cardinality_conflict']
                or len(row['candidate_ids']) > 1 and row['selection_source'] == 'visual_source_frame_geometry'
                or deictic_visual_search(mention)):
            pending.append(mention['mention_id'])
    return pending


def recovery_frames(event, pending, maximum=8):
    """Global coverage floor + phase-prior primary/context; never gaze-ranked time."""
    frames = event['frames']
    if len(frames) <= maximum:
        return frames[:]
    chosen = {round(i * (len(frames) - 1) / 3) for i in range(4)}
    index = {f['frame_id']: i for i, f in enumerate(frames)}
    specs = event['temporal_search']['mentions']
    priorities = []
    for mid in pending:
        spec = specs[mid]
        primary = spec['primary_frame_ids']
        if primary:
            priorities.append(index[primary[len(primary) // 2]])
        priorities.extend(index[fid] for fid in spec['context_frame_ids'])
        priorities.extend(index[fid] for fid in primary)
    for i in priorities:
        if len(chosen) >= maximum:
            break
        chosen.add(i)
    while len(chosen) < min(maximum, len(frames)):
        remaining = set(range(len(frames))) - chosen
        chosen.add(max(remaining, key=lambda i: (min(abs(i - j) for j in chosen), -i)))
    if len(chosen) % 2:
        chosen.add(min(set(range(len(frames))) - chosen))
    return [frames[i] for i in sorted(chosen)]


def merge_recovery(initial, recovery, pending):
    """Only targeted occurrences can change; retain first pass on missing rows."""
    rows = {r['mention_id']: r for r in initial['rows']}
    replacements = []
    for row in recovery['rows']:
        if row['mention_id'] in pending:
            rows[row['mention_id']] = row
            replacements.append(row['mention_id'])
    expected = {r['mention_id'] for r in initial['rows']} | set(initial['missing_ids'])
    effective_missing = sorted(expected - set(rows))
    return {**initial, 'rows': list(rows.values()), 'effective_coverage_complete': not effective_missing,
            'effective_missing_ids': effective_missing, 'recovery_replaced_ids': replacements,
            'recovery_untargeted_rows_ignored': [r['mention_id'] for r in recovery['rows'] if r['mention_id'] not in pending]}


PROMPT = '''Resolve the ORIGINAL INSTRUCTION over the complete first-person interaction.
Use every concrete language mention, including actors, action objects, destinations
and reference objects. RGB is the visual evidence. Each labelled source frame has
panel 0 FULL RGB and panels 1/2/3 supplementary RGB details of that same frame.
Details are optional viewing aids, never target labels. All objects outside detail
windows remain valid. Match language, appearance, actions and spatial relations.

The script supplies each mention's role, component/parent, count, compatible scene
categories and action context. Soft phase windows prioritize search; they are NOT
speech timestamps. Search outside a phase when needed. Repeated literal phrases
are distinct occurrences and may name the same or different object. Only explicit
component/pronoun links assert shared identity. A part without its own scene class
resolves to its parent object. Vague here/there has no lexical identity. Resolve
a concrete scene object only when RGB actions and spatial relations support that
interpretation; otherwise return non_concrete_reference. Never invent a destination
merely because a scene category or a gaze-selected detail is available.

Return a JSON object with exactly one key "mentions", an array containing exactly
one row per requested mention_id. Each row has these keys: mention_id, status,
instances, unresolved_reason. For grounded rows, instances is a nonempty array of
objects with exactly these keys: frame_id (a printed Vxxx source ID), x (number),
y (number), category (compatible scene category string).
Grounded rows have unresolved_reason null. No default source frame or point is
valid: never copy a formatting example, reuse the image centre, or cite the first
frame for all mentions unless actual visible objects support each such choice.
Choose a point ON each intended visible instance. Coordinates use the WHOLE
displayed composite image on a 0..1000 scale, as in the previous protocol. You may
point inside full RGB or any RGB detail, never on header text. Scripts identify the
panel and convert that point into the exact source frame. Cite the printed Vxxx SOURCE FRAME,
not a Gxxx temporal patch, not the ordinal position. Category must describe the
parent scene object using the supplied compatible categories. Do not output IDs.
Frame metadata reports whether scene-ID geometry is supported. When several
clear views show the same object, prefer one with supported geometry for identity
mapping. This flag is not visibility or target evidence; unsupported frames remain
valid RGB observations and all-unsupported cases must not invent an identity.
For singular mentions return ONE best supporting frame and point. Do not repeat
a fixed coordinate across a list of frames. For plural mentions use one observation
per distinct visible instance. For an explicit
plural include all supported instances up to count; unspecified plural is not all
scene objects of that class. If invisible/uncertain, status unresolved, instances
[], and reason not_visible/uncertain_identity/no_supported_view/non_concrete_reference/other.
Never silently omit a requested mention. INPUT: '''


def grounding_prompt(event, condition, *, pending=None, frames=None):
    frames = event['frames'] if frames is None else frames
    candidates = {c['candidate_id']: c for c in event['scene_candidate_table']}
    units = []
    selected = set(pending) if pending is not None else None
    for mention in event['mention_hints']:
        mid = mention['mention_id']
        if selected is not None and mid not in selected:
            continue
        categories = sorted(set().union(*(categories_for_candidate(candidates[c])
            for c in mention.get('candidate_ids', []) if c in candidates)))
        spec = event['temporal_search']['mentions'][mid]
        units.append({key: mention.get(key) for key in
            ('mention_id', 'text', 'span', 'object_type', 'role', 'kind', 'count', 'component',
             'parent_text', 'parent_mention_id', 'coreference_id', 'coreference_note')})
        units[-1].update(compatible_categories=categories, highlighted_instruction=spec['highlighted_instruction'],
            action_context=spec['action_context'], soft_primary_range=spec['primary_range'],
            primary_frame_ids=[fid for fid in spec['primary_frame_ids'] if any(f['frame_id'] == fid for f in frames)],
            context_frame_ids=[fid for fid in spec['context_frame_ids'] if any(f['frame_id'] == fid for f in frames)])
    return PROMPT + json.dumps({'instruction': event['instruction'], 'mention_units': units,
        'pass': 'initial_complete_interaction' if pending is None else 'targeted_recovery_with_global_context',
        'frame_sources': [{'frame_id': f['frame_id'], 'time_seconds': f['time_seconds'],
            'source_size': f['source_size'], 'scene_id_geometry_supported': bool(f['geometry']['clock_available']),
            'detail_rectangles_source_xyxy':
                [s['bbox_source_xyxy'] for s in f['routing'][condition]['slots']]} for f in frames],
        'video_views': prompt_temporal_groups(qwen_temporal_groups(frames)),
        'scene_category_vocabulary': sorted(set().union(*(categories_for_candidate(c) for c in candidates.values())))},
        ensure_ascii=False)
