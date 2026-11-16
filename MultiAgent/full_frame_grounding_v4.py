"""Prototype mention-accounting contract for full-field referring.

This module is deliberately experimental and is not connected to the default
pipeline. It separates one-row-per-mention accounting from visual grounding
and from the approximate scene-ID decoder.
"""
from __future__ import annotations

import json
import math
from collections import Counter

from .full_frame_grounding import categories_for_candidate, normalized_category
from .local_video_visual import prompt_temporal_groups

VERSION = 'full_frame_grounding_v4_mention_accounting'
STATUSES = {'grounded', 'unresolved'}
UNRESOLVED_REASONS = {
    'not_visible',
    'uncertain_identity',
    'no_supported_view',
    'non_concrete_reference',
    'other',
}

PROMPT = '''Ground every parser-listed mention occurrence in this complete first-person interaction.
Use the full chronological video, original RGB, visible actions, contact/pointing,
appearance and spatial relations. Attention panels are optional weak visual evidence;
they are not target labels. Do not return only the last target and do not omit a
mention because it is difficult. Repeated literal text can refer to different
mention_id occurrences.

Each frame is a four-panel full-field display. TOP LEFT is the original RGB. The
other panels are earlier / middle / recent attention from the preceding two seconds.
Coordinates use the WHOLE displayed composite image on a 0..1000 scale. A grounded
point must be on the intended object in the TOP LEFT RGB panel. Do not point into a
heat panel. Scripts will decode a grounded point against the full scene candidate
table; do not guess an anchor ID.

Return exactly one row for every mention_id in mention_units, including unresolved
mentions. Return JSON only:
{"mentions":[{"mention_id":"M001","mention":"literal phrase","status":"grounded",
"category":"scene category","view_id":"Gxxx","x":0,"y":0,"unresolved_reason":null}]}

For status grounded, category/view_id/x/y are required and unresolved_reason is null.
For status unresolved, category/view_id/x/y must be null and unresolved_reason must be
one of: not_visible, uncertain_identity, no_supported_view, non_concrete_reference,
other. Copy mention exactly. Do not duplicate or invent mention_id values.

INPUT: '''


def mention_units(event):
    units = []
    for index, hint in enumerate(event.get('mention_hints', []), start=1):
        if not isinstance(hint, dict):
            continue
        text = hint.get('text')
        if not isinstance(text, str) or not text:
            continue
        mention_id = hint.get('mention_id')
        if not isinstance(mention_id, str) or not mention_id:
            mention_id = f'M{index:03d}'
        units.append({
            'mention_id': mention_id,
            'mention': text,
            'role': hint.get('role'),
            'kind': hint.get('kind'),
            'coreference_id': hint.get('coreference_id'),
        })
    return units


def grounding_prompt(event):
    vocab = sorted(set().union(*(categories_for_candidate(c) for c in event['scene_candidate_table'])))
    return PROMPT + json.dumps({
        'instruction': event['instruction'],
        'mention_units': mention_units(event),
        'scene_category_vocabulary': vocab,
        'video_views': prompt_temporal_groups(event['qwen_temporal_groups']),
    }, ensure_ascii=False)


def _local_rgb_point(row, event):
    width, height = event['frames'][0]['source_size']
    x = float(row['x']) / 1000.0 * 2.0
    y = (float(row['y']) / 1000.0 * ((height + 28) * 2.0) - 28.0) / height
    return x, y


def validate_mention_contract(parsed, event):
    """Validate rows without guessing IDs, categories, statuses, or points.

    schema_valid describes all rows; coverage_complete requires one valid row
    per expected occurrence. Independently valid rows are exposed for diagnosis.
    Duplicate IDs are rejected together instead of choosing the first copy.
    """
    expected = mention_units(event)
    expected_ids = [unit['mention_id'] for unit in expected]
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError('Input mention IDs must be unique')
    expected_by_id = {unit['mention_id']: unit for unit in expected}
    views = {group['view_id'] for group in event['qwen_temporal_groups']}
    vocab = set().union(*(categories_for_candidate(c) for c in event['scene_candidate_table']))
    result = {
        'schema_valid': False,
        'coverage_complete': False,
        'errors': [],
        'mentions': [],
        'grounded_referents': [],
        'unresolved_mentions': [],
        'missing_mention_ids': [],
        'duplicate_mention_ids': [],
        'rejected_rows': [],
        'coordinate_contract': 'whole_composite_0_to_1000',
    }
    if not isinstance(parsed, dict) or set(parsed) != {'mentions'} or not isinstance(parsed.get('mentions'), list):
        result['errors'] = ['wrong_keys_or_mentions_array']
        result['missing_mention_ids'] = expected_ids
        return result

    seen = Counter(raw['mention_id'] for raw in parsed['mentions']
                   if isinstance(raw, dict) and isinstance(raw.get('mention_id'), str))
    for index, raw in enumerate(parsed['mentions']):
        errors = []
        mention_id = raw.get('mention_id') if isinstance(raw, dict) else None
        if not isinstance(raw, dict):
            errors.append('row_not_object')
        else:
            required = {'mention_id', 'mention', 'status'}
            allowed = required | {'category', 'view_id', 'x', 'y', 'unresolved_reason'}
            if not required.issubset(raw) or set(raw) - allowed:
                errors.append('wrong_row_keys')
            if not isinstance(mention_id, str) or mention_id not in expected_by_id:
                errors.append('unknown_mention_id')
            elif raw.get('mention') != expected_by_id[mention_id]['mention']:
                errors.append('mention_text_mismatch')
            if isinstance(mention_id, str) and seen[mention_id] > 1:
                errors.append('duplicate_mention_id')
            status = raw.get('status')
            if not isinstance(status, str) or status not in STATUSES:
                errors.append('invalid_status')
            elif status == 'grounded':
                fields = ('category', 'view_id', 'x', 'y')
                if any(key not in raw for key in fields):
                    errors.append('missing_grounded_field')
                if not isinstance(raw.get('category'), str) or normalized_category(raw['category']) not in vocab:
                    errors.append('unknown_category')
                if not isinstance(raw.get('view_id'), str) or raw['view_id'] not in views:
                    errors.append('unknown_view')
                point_ok = all(isinstance(raw.get(key), (int, float))
                               and not isinstance(raw[key], bool)
                               and math.isfinite(raw[key]) and 0 <= raw[key] <= 1000
                               for key in ('x', 'y'))
                if not point_ok:
                    errors.append('invalid_composite_coordinate')
                else:
                    x, y = _local_rgb_point(raw, event)
                    if not 0 <= x <= 1 or not 0 <= y <= 1:
                        errors.append('point_outside_original_rgb_panel')
                if raw.get('unresolved_reason') is not None:
                    errors.append('grounded_has_unresolved_reason')
            elif status == 'unresolved':
                fields = ('category', 'view_id', 'x', 'y', 'unresolved_reason')
                if any(key not in raw for key in fields):
                    errors.append('missing_unresolved_field')
                if any(raw.get(key) is not None for key in ('category', 'view_id', 'x', 'y')):
                    errors.append('unresolved_contains_grounding_fields')
                reason = raw.get('unresolved_reason')
                if not isinstance(reason, str) or reason not in UNRESOLVED_REASONS:
                    errors.append('invalid_unresolved_reason')
        if errors:
            result['rejected_rows'].append({'index': index, 'mention_id': mention_id,
                                            'reason': sorted(set(errors))})
            result['errors'].extend(errors)
            continue
        normalized = dict(raw)
        if raw['status'] == 'grounded':
            normalized.setdefault('unresolved_reason', None)
            normalized['x'], normalized['y'] = _local_rgb_point(raw, event)
            result['grounded_referents'].append(normalized)
        else:
            result['unresolved_mentions'].append(normalized)
        result['mentions'].append(normalized)

    result['duplicate_mention_ids'] = sorted(key for key, count in seen.items() if count > 1)
    result['missing_mention_ids'] = sorted(set(expected_by_id) - set(seen))
    result['coverage_complete'] = (not result['duplicate_mention_ids']
                                   and len(result['mentions']) == len(expected)
                                   and {r['mention_id'] for r in result['mentions']} == set(expected_by_id))
    result['schema_valid'] = not result['rejected_rows']
    result['errors'] = sorted(set(result['errors']))
    return result
